# SPDX-License-Identifier: Apache-2.0
"""The chat page's "check for updates" (POST /webchat/check_update): the launcher asks for the latest release at once, as item 7 of
its menu does, and a newer one arms the page's update button.  No GPU and no real network: the fake release server of test_update
stands in for GitHub and counts the requests; run_web runs with the stand-in for `yasakani serve` of test_quickstart_web."""
import ast
import contextlib
import io
import json
import re
import shutil
import subprocess
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests import test_quickstart_web as W
from tests import test_update as T

QS = W.QS
H = "X-YASAKANI-Control"


def setUpModule():
    W.setUpModule()                                     # the stand-in package for `yasakani serve` (RunWebCheckTests)


def tearDownModule():
    W.tearDownModule()


class CheckNowBase(W.Base):
    def setUp(self):
        super().setUp()
        self.gh = T.FakeGitHub()
        self.addCleanup(self.gh.close)
        self.src = QS.UpdateSource(self.gh.base)
        self.st, self.saved, self.checks = {}, 0, 0
        self.out = io.StringIO()
        r = contextlib.redirect_stdout(self.out)
        r.__enter__()
        self.addCleanup(r.__exit__, None, None, None)

    def save(self):
        self.saved += 1

    def check(self, result):
        """What main() gives run_web: update_check with force=True (menu item 7)."""
        self.checks += 1
        return QS.update_check(self.ui, self.st, self.save, self.src, force=True, result=result)

    def serve_check(self, check=None, wait=QS.CHECK_NOW_WAIT, control=True, checknow=True):
        self.update = QS.UpdateRequest()
        self.control = QS.ControlRequest() if control else None
        self.checknow = QS.UpdateCheckNow(check or self.check, self.update, wait) if checknow else None
        port = W.free_port()
        conf = {"lang": "en", "model": "quality", "version": QS.QS_VERSION, "repo": QS.REPO_URL, "support": QS.SUPPORT_URL}
        handler = QS.make_web_handler(self.page, conf, 1, port, self.update, self.control,
                                      QS.WebSession("quality", None, 1, installed=("quality",)), None, self.checknow)
        httpd = QS.LocalServer(("127.0.0.1", port), handler)
        threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
        self.addCleanup(httpd.close_connections)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        if self.checknow is not None:
            self.addCleanup(self.checknow.close)
        return port

    def press(self, port, headers=None, token=True, body=None):
        h = self.own(port)
        if token:
            h[H] = self.control.token
        h.update(headers or {})
        code, data, _ = W.request(port, "POST", "/webchat/check_update", h, body)
        return code, json.loads(data) if data else None


class EndpointTests(CheckNowBase):
    def test_refused_without_the_token_and_asks_nobody(self):
        self.gh.publish("9.9.9")
        port = self.serve_check()
        own = "http://127.0.0.1:%d" % port
        tok = self.control.token
        for headers, want in (({H: None}, 403),                                     # no token
                              ({H: "0" * 64}, 403),                                 # a wrong one
                              ({QS.UpdateRequest.HEADER: tok, H: None}, 403),       # the token in the other header
                              ({"Origin": None}, 403),                              # no Origin
                              ({"Origin": "http://example.com"}, 403),
                              ({"Host": "example.com"}, 403),
                              ({"Sec-Fetch-Site": "cross-site"}, 403)):
            h = {"Origin": own, H: tok}
            h.update(headers)
            h = {k: v for k, v in h.items() if v is not None}
            code, data, _ = W.request(port, "POST", "/webchat/check_update", h)
            self.assertEqual(code, want, headers)
        h = dict(self.own(port), **{H: self.control.token})
        self.assertEqual(W.too_long(port, "/webchat/check_update", h, QS.CONTROL_MAX_BODY + 1), 413)   # see W.too_long
        self.assertEqual(W.request(port, "GET", "/webchat/check_update", {"Origin": own, H: tok})[0], 404)
        self.assertEqual((self.gh.requests, self.checks, self.st, self.update.rel), ([], 0, {}, None))
        self.assertEqual(self.control.token, tok)                                   # nothing was used up
        self.control.close()                                                        # the launcher left the page
        self.assertEqual(self.press(port, {H: tok}, token=False)[0], 403)
        self.assertEqual(self.gh.requests, [])

    def test_only_with_the_control_token_of_the_page(self):
        """A launcher without the page's control (--web-test, the server of a model switch's checks) has no such request."""
        port = self.serve_check(control=False)
        self.assertEqual(W.request(port, "POST", "/webchat/check_update", self.own(port, **{H: "0" * 64}))[0], 404)
        c = W.get_json(port, "/webchat/config")
        self.assertNotIn("update_check_now", c)
        port = self.serve_check(checknow=False)
        self.assertEqual(self.press(port)[0], 404)
        self.assertNotIn("update_check_now", W.get_json(port, "/webchat/config"))
        self.assertEqual(self.gh.requests, [])

    def test_newer_arms_the_update_button(self):
        self.gh.publish("9.9.9")
        port = self.serve_check()
        c = W.get_json(port, "/webchat/config")
        self.assertIs(c["update_check_now"], True)
        self.assertNotIn("update", c)
        self.assertNotIn("update_token", c)
        self.assertEqual(W.request(port, "POST", "/webchat/update", self.own(port, **{QS.UpdateRequest.HEADER: "0" * 64}))[0], 404)
        t0 = time.monotonic()
        code, answer = self.press(port)
        self.assertLess(time.monotonic() - t0, QS.CHECK_NOW_WAIT)
        self.assertEqual((code, answer), (200, {"state": "newer", "version": "9.9.9"}))
        self.assertEqual(len(self.gh.api_calls()), 1)
        path, headers = self.gh.requests[-1]                                         # the request of fetch_latest, nothing more
        self.assertEqual(path, "/repos/yasakani-dev/yasakani/releases/latest")
        self.assertFalse({k.lower() for k in headers} - {"host", "user-agent", "accept", "accept-encoding", "connection"})
        self.assertEqual(self.st["update_latest"], {"version": "9.9.9", "tag": "v9.9.9"})
        self.assertAlmostEqual(self.st["update_checked"], time.time(), delta=60)
        self.assertGreaterEqual(self.saved, 1)
        self.assertIn("github.com", self.out.getvalue())                             # the black window says so, as the menu
        c = W.get_json(port, "/webchat/config")
        self.assertEqual(c["update"], {"version": "9.9.9", "page": "https://github.com/yasakani-dev/yasakani/releases/tag/v9.9.9"})
        self.assertRegex(c["update_token"], r"^[0-9a-f]{64}$")
        self.assertEqual(c["control_token"], self.control.token)
        # a second press asks nobody: the button is there
        self.assertEqual(self.press(port), (200, {"state": "newer", "version": "9.9.9"}))
        self.assertEqual(len(self.gh.api_calls()), 1)
        # the update button works as today
        code, data, _ = W.request(port, "POST", "/webchat/update", self.own(port, **{QS.UpdateRequest.HEADER: c["update_token"]}))
        self.assertEqual((code, json.loads(data)), (202, {"status": "updating", "version": "9.9.9"}))
        self.assertTrue(self.update.event.wait(5))
        self.assertEqual(QS.wait_for_stop(SimpleNamespace(poll=lambda: None), self.update, poll=0.05, control=self.control), "update")

    def test_latest_arms_nothing(self):
        self.gh.publish(QS.QS_VERSION)
        port = self.serve_check()
        self.assertEqual(self.press(port), (200, {"state": "latest", "version": QS.QS_VERSION}))
        self.assertEqual(self.st["update_latest"]["version"], QS.QS_VERSION)
        self.assertIn("update_checked", self.st)
        self.assertIn("This is the latest version (%s)" % QS.QS_VERSION, self.out.getvalue())
        c = W.get_json(port, "/webchat/config")
        self.assertNotIn("update", c)
        self.assertNotIn("update_token", c)
        self.assertIsNone(self.update.rel)
        self.assertEqual(W.request(port, "POST", "/webchat/update", self.own(port, **{QS.UpdateRequest.HEADER: "0" * 64}))[0], 404)
        self.assertEqual(self.press(port)[1]["state"], "latest")                     # every press asks once
        self.assertEqual(len(self.gh.api_calls()), 2)

    def test_an_older_release_or_a_skipped_one_is_latest(self):
        cur = QS.parse_version(QS.QS_VERSION)
        self.gh.publish("%d.%d.%d" % (cur[0], max(cur[1] - 1, 0), 0) if cur[1] else "0.0.0")
        port = self.serve_check()
        self.assertEqual(self.press(port)[1], {"state": "latest", "version": QS.QS_VERSION})

    def test_failures_change_nothing(self):
        port = self.serve_check()
        for release, word in ((500, "HTTPError"), (["not", "a", "release"], "not a release"), (None, "not a release")):
            self.gh.release = release
            before = dict(self.st, update_checked=None)
            code, answer = self.press(port)
            self.assertEqual((code, answer["state"]), (200, "failed"), release)
            self.assertIn(word, answer["message"])
            self.assertLessEqual(len(answer["message"]), 160)
            self.assertNotIn("\n", answer["message"])
            self.assertIsNone(self.update.rel)
            self.assertNotIn("update_latest", self.st)                               # as the menu's check: only the time
            self.assertEqual(dict(self.st, update_checked=None), before)
        self.assertIn("Could not check", self.out.getvalue())
        c = W.get_json(port, "/webchat/config")
        self.assertNotIn("update", c)
        self.assertEqual(c["control_token"], self.control.token)

    def test_offline_answers_in_time_without_a_local_path(self):
        self.gh.close()
        port = self.serve_check()
        t0 = time.monotonic()
        code, answer = self.press(port)
        self.assertLess(time.monotonic() - t0, QS.CHECK_NOW_WAIT + 2)
        self.assertEqual((code, answer["state"]), (200, "failed"))
        self.assertIsNone(re.search(r"[A-Za-z]:\\|/Users/|\\Users\\", answer["message"]), answer["message"])
        self.assertIsNone(self.update.rel)

    def test_two_presses_make_one_request(self):
        self.gh.publish("9.9.9")
        gate, calls = threading.Event(), []

        def slow(result):
            calls.append(1)
            gate.wait(10)
            return self.check(result)

        port = self.serve_check(slow)
        answers = []
        threads = [threading.Thread(target=lambda: answers.append(self.press(port))) for _ in range(3)]
        for t in threads:
            t.start()
        self.assertTrue(W.wait_until(lambda: calls, 5))
        time.sleep(0.3)                                                              # the other presses are waiting by now
        gate.set()
        for t in threads:
            t.join(20)
        self.assertEqual(answers, [(200, {"state": "newer", "version": "9.9.9"})] * 3)
        self.assertEqual((len(calls), len(self.gh.api_calls())), (1, 1))

    def test_busy_when_the_check_does_not_end_in_time(self):
        gate, calls = threading.Event(), []
        self.gh.publish(QS.QS_VERSION)

        def slow(result):
            calls.append(1)
            gate.wait(10)
            return self.check(result)

        port = self.serve_check(slow, wait=0.3)
        self.assertEqual(self.press(port)[1]["state"], "busy")
        self.assertEqual(self.press(port)[1]["state"], "busy")                       # the same check, not a second one
        self.assertEqual(len(calls), 1)
        gate.set()
        self.assertTrue(W.wait_until(lambda: self.gh.api_calls(), 5))
        time.sleep(0.2)
        self.assertEqual(self.press(port)[1], {"state": "latest", "version": QS.QS_VERSION})
        self.assertEqual(len(self.gh.api_calls()), 2)                                # a new press after the end asks again

    def test_closed_check_asks_nobody(self):
        self.gh.publish("9.9.9")
        port = self.serve_check()
        self.checknow.close()
        self.assertEqual(self.press(port)[1]["state"], "failed")
        self.update.close()
        self.assertFalse(self.update.arm({"version": "9.9.9", "page": ""}))          # the launcher left the page: no button
        self.assertEqual(self.gh.requests, [])


class DailyRuleTests(CheckNowBase):
    def test_the_daily_check_is_unchanged(self):
        """After a press the automatic check within 24 h asks nobody (it knows the answer), and later it asks once again."""
        self.gh.publish(QS.QS_VERSION)
        port = self.serve_check()
        self.assertEqual(self.press(port)[1]["state"], "latest")
        now = self.st["update_checked"]
        self.assertIsNone(QS.update_check(self.ui, self.st, self.save, self.src, now=now + 3600))
        self.assertEqual(len(self.gh.api_calls()), 1)
        QS.update_check(self.ui, self.st, self.save, self.src, now=now + 25 * 3600)
        self.assertEqual(len(self.gh.api_calls()), 2)
        self.assertEqual(QS.CHECK_EVERY, 24 * 3600)

    def test_result_is_only_an_addition(self):
        self.gh.publish("9.9.9")
        r = {}
        self.assertEqual(QS.update_check(self.ui, self.st, self.save, self.src, force=True, result=r)["version"], "9.9.9")
        self.assertEqual(r, {"state": "newer"})
        r = {}
        self.assertIsNone(QS.update_check(self.ui, self.st, self.save, self.src, disabled=True, result=r))     # not forced: nothing
        self.assertEqual(r, {})

    def test_no_update_check_does_not_turn_off_the_manual_check(self):
        self.gh.publish("9.9.9")
        a = SimpleNamespace(no_update_check=True)
        self.assertTrue(QS.check_disabled(a, {"update_check": False}, {"YASAKANI_NO_UPDATE_CHECK": "1"}))
        self.assertIsNone(QS.update_check(self.ui, self.st, self.save, self.src, disabled=True))
        self.assertEqual(self.gh.requests, [])
        port = self.serve_check()
        self.assertEqual(self.press(port)[1], {"state": "newer", "version": "9.9.9"})
        self.assertEqual(len(self.gh.api_calls()), 1)

    def test_main_passes_the_forced_check_to_the_page(self):
        """main(): the function given to run_web is update_check with force=True and nothing that turns it off, and a release that
        the page found is the one the update button installs."""
        src = Path(QS.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        fn = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == "check_now")
        call = next(c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "update_check")
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        self.assertEqual(kw.get("force"), "True")
        self.assertNotIn("disabled", kw)
        runs = [c for c in ast.walk(main) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "run_web"]
        self.assertTrue(runs)
        for c in runs:
            self.assertEqual(ast.unparse(c.args[-1]), "check_now")
        self.assertIn("pending = getattr(a, 'web_update', None) or pending", ast.unparse(main))


class RunWebCheckTests(W.RunWebBase):
    save, check = CheckNowBase.save, CheckNowBase.check

    def setUp(self):
        super().setUp()
        self.gh = T.FakeGitHub()
        self.addCleanup(self.gh.close)
        self.src = QS.UpdateSource(self.gh.base)
        self.st, self.saved, self.checks = {}, 0, 0

    def start_check(self, a, update_rel=None):
        box = {}

        def run():
            try:
                box["rc"] = QS.run_web(self.ui, a, self.vpy, self.env, str(self.tmp / "models" / "quality"),
                                       str(self.tmp / "engine" / "yskn.dll"), "quality", update_rel, self.gpu, ["quality"], None,
                                       self.check)
            except BaseException as e:  # noqa: BLE001 - shown by the test
                box["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, box

    def test_found_release_is_the_update_of_the_page(self):
        self.gh.publish("9.9.9")
        a = self.args()
        t, box = self.start_check(a)
        port = self.page_port(a, box)
        c = W.get_json(port, "/webchat/config")
        self.assertIs(c["update_check_now"], True)
        self.assertNotIn("update_token", c)
        code, data, _ = W.request(port, "POST", "/webchat/check_update", self.own(port, **{H: c["control_token"]}))
        self.assertEqual((code, json.loads(data)), (200, {"state": "newer", "version": "9.9.9"}))
        c = W.get_json(port, "/webchat/config")
        code, _, _ = W.request(port, "POST", "/webchat/update", self.own(port, **{QS.UpdateRequest.HEADER: c["update_token"]}))
        self.assertEqual(code, 202)
        self.finish(t, box, QS.UPDATE_RC)
        self.assertEqual(a.web_update["version"], "9.9.9")                          # main() installs this one
        self.assertEqual(len(self.gh.api_calls()), 1)

    def test_nothing_found_leaves_no_update(self):
        self.gh.publish(QS.QS_VERSION)
        a = self.args()
        t, box = self.start_check(a)
        port = self.page_port(a, box)
        tok = W.get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(json.loads(W.request(port, "POST", "/webchat/check_update", self.own(port, **{H: tok}))[1])["state"],
                         "latest")
        self.assertEqual(W.request(port, "POST", "/webchat/shutdown", self.own(port, **{H: tok}))[0], 202)
        self.finish(t, box, QS.QUIT_RC)
        self.assertIsNone(a.web_update)


class PageTests(W.Base):
    def text(self):
        return self.page.decode("utf-8")

    def test_strings_in_both_languages(self):
        text = self.text()
        ja, en = text.index("  ja: {"), text.index("  en: {")
        for part in (text[ja:en], text[en:]):
            for key in ("chk", "chk_run", "chk_tip", "chk_latest", "chk_fail", "chk_busy"):
                self.assertIn(" %s: '" % key, part, key)
        self.assertIn("chk: '更新を確認'", text[ja:en])
        self.assertIn("chk_latest: '最新の版です（{v}）'", text[ja:en])
        self.assertIn("chk: 'Check for updates'", text[en:])
        self.assertIn("chk_latest: 'You have the latest version ({v})'", text[en:])

    def test_the_control_beside_the_version(self):
        text = self.text()
        row = re.search(r'<div class="kv"><div class="k" id="md_l_ver"></div>.*?</div></div>', text).group(0)
        self.assertIn('<span id="ver_v"></span>', row)
        self.assertIn('<button class="btn2" id="chk_upd" type="button" hidden></button>', row)
        self.assertIn('id="chk_note"', row)
        self.assertIn("fetch('/webchat/check_update', { method: 'POST', headers: { 'X-YASAKANI-Control': token }", text)
        self.assertIn("const show = controls && cfg.update_check_now === true && !cfg.update && !quiet();", text)
        self.assertIsNone(re.search(r"<[^>]+\son[a-z]+\s*=", text), "an inline handler")
        self.assertIn("$('chk_upd').addEventListener('click'", text)

    def node(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not on PATH")
        return node

    def test_script_parses(self):
        node = self.node()
        script = re.search(r"<script>(.*)</script>", self.text(), re.S).group(1)
        f = self.tmp / "page.js"
        f.write_text(script, encoding="utf-8")
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_shown_only_with_a_control_token_and_no_update(self):
        """paintCheck of the page, run by node with stand-ins for the page: hidden without the control token or the launcher's
        word, while an update is known or under way, and after the end."""
        node = self.node()
        text = self.text()
        fn = re.search(r"function paintCheck\(\) \{.*?\n\}", text, re.S).group(0)
        js = """
const els = {};
const $ = id => (els[id] = els[id] || { hidden: false, disabled: false, textContent: '', title: '' });
const put = (id, v) => { $(id).textContent = v; };
const L = { chk: 'C', chk_run: 'R', chk_tip: 'T' };
let controls, cfg, phase, checkingUpd = false;
const quiet = () => ['failed', 'stopped', 'ended', 'updating'].includes(phase);
%s
const out = [];
for (const [c, f, p] of [[false, { update_check_now: true }, 'ready'], [true, {}, 'ready'], [true, { update_check_now: true }, 'ready'],
                         [true, { update_check_now: true }, 'loading'], [true, { update_check_now: true, update: { version: '9.9.9' } }, 'ready'],
                         [true, { update_check_now: true }, 'updating'], [true, { update_check_now: true }, 'stopped']]) {
  controls = c; cfg = f; phase = p; paintCheck(); out.push($('chk_upd').hidden);
}
checkingUpd = true; controls = true; cfg = { update_check_now: true }; phase = 'ready'; paintCheck();
out.push([$('chk_upd').disabled, $('chk_upd').textContent]);
console.log(JSON.stringify(out));
""" % fn
        f = self.tmp / "paint.js"
        f.write_text(js, encoding="utf-8")
        r = subprocess.run([node, str(f)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), [True, True, False, False, True, True, True, [True, "R"]])

    def test_readme_tells_of_it(self):
        readme = (W.ROOT.parent / "quickstart" / "README.txt").read_bytes()
        text = readme.decode("utf-8")
        self.assertIn("「更新を確認」を押すと、その場で 1 回だけ問い合わせます", text)
        self.assertIn('"Check for updates" in the Model tab of the page asks at once, one request, only when you press it', text)
        self.assertIn("1 日に 1 回まで", text)
        self.assertIn("at most once a day", text)
        self.assertNotIn(b"\n", readme.replace(b"\r\n", b""))                       # CRLF kept


class StateFileTests(unittest.TestCase):
    def test_two_threads_save_the_state(self):
        """The check of the page saves the state from its own thread while the launcher's main thread may save too: every save
        ends, and the file is whole each time (one writer at a time through the same temporary file)."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.json")
            errors, stop = [], threading.Event()

            def writer(tag):
                st = {"who": tag, "n": 0}
                try:
                    for i in range(40):
                        st["n"] = i
                        st["k%d" % (i % 7)] = i                 # the dictionary changes between saves, as the real state does
                        QS.save_state(path, st)
                except Exception as e:  # noqa: BLE001 - the test reports whatever a save raised
                    errors.append(repr(e))
                finally:
                    stop.set()

            threads = [threading.Thread(target=writer, args=(t,)) for t in ("check", "main", "switch")]
            for t in threads:
                t.start()
            reads = 0
            while not stop.is_set() or any(t.is_alive() for t in threads):
                st = QS.load_state(path)
                if st:
                    self.assertIn(st.get("who"), ("check", "main", "switch"))
                    reads += 1
                time.sleep(0.002)                               # a reader that opens the file again and again
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertEqual(QS.load_state(path).get("n"), 39)
            self.assertFalse(Path(path + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
