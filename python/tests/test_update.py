# SPDX-License-Identifier: Apache-2.0
"""The update of the quickstart (check, download, checks, backup, rollback, START.cmd while it runs, the button of the chat
page) and `yasakani update`.  No GPU and no real network: a fake release server on 127.0.0.1 stands in for GitHub."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from tests import support  # noqa: F401  (puts the package on sys.path)
from yasakani import update as U

ROOT = Path(__file__).resolve().parent.parent
QS_PATH = ROOT.parent / "quickstart" / "quickstart.py"
START_CMD = ROOT.parent / "quickstart" / "START.cmd"


def load_quickstart():
    p = os.environ.get("YASAKANI_QUICKSTART") or str(QS_PATH)
    if not Path(p).is_file():
        return None
    spec = importlib.util.spec_from_file_location("yasakani_quickstart_update_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


QS = load_quickstart()


class FakeGitHub:
    """GET /repos/yasakani-dev/yasakani/releases/latest and /yasakani-dev/yasakani/releases/download/<tag>/<name>."""

    def __init__(self):
        self.release = None          # dict sent as JSON, or an int status
        self.files = {}              # "<tag>/<name>" -> bytes
        self.requests = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                fake.requests.append((self.path, dict(self.headers)))
                if self.path == "/repos/yasakani-dev/yasakani/releases/latest":
                    if isinstance(fake.release, int):
                        return self._send(fake.release, b"{}")
                    return self._send(200, json.dumps(fake.release).encode())
                pre = "/yasakani-dev/yasakani/releases/download/"
                if self.path.startswith(pre) and self.path[len(pre):] in fake.files:
                    return self._send(200, fake.files[self.path[len(pre):]])
                self._send(404, b"not found")

            def _send(self, code, body):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.base = "http://127.0.0.1:%d" % self.port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def api_calls(self):
        return [p for p, _ in self.requests if p.startswith("/repos/")]

    def publish(self, version, files=None, sums=None, prerelease=False, assets=None):
        """A release of the start files ``files`` (default: a quickstart.py of that version and a START.cmd)."""
        tag = "v" + version
        if files is None:
            files = {"quickstart.py": ('QS_VERSION = "%s"\nprint("new")\n' % version).encode(), "START.cmd": b"@echo off\r\necho new\r\n",
                     "LICENSE": b"license\n"}
        zname = "yasakani-quickstart-%s.zip" % version
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as z:
            for n, b in files.items():
                z.writestr(n, b)
        data = bio.getvalue()
        self.files[tag + "/" + zname] = data
        self.files[tag + "/SHA256SUMS.txt"] = sums if sums is not None else (
            "%s  %s\n%s  yasakani-%s-win-x64.zip\n" % (hashlib.sha256(data).hexdigest(), zname, "0" * 64, version)).encode()
        if assets is None:
            assets = [{"name": n, "browser_download_url": "%s/yasakani-dev/yasakani/releases/download/%s/%s" % (self.base, tag, n)}
                      for n in (zname, "SHA256SUMS.txt")]
        self.release = {"tag_name": tag, "draft": False, "prerelease": prerelease, "assets": assets,
                        "html_url": "https://github.com/yasakani-dev/yasakani/releases/tag/" + tag}
        return data


class Base(unittest.TestCase):
    def setUp(self):
        if QS is None:
            self.skipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")
        self.gh = FakeGitHub()
        self.addCleanup(self.gh.close)
        self.src = QS.UpdateSource(self.gh.base)
        self.tmp = Path(tempfile.mkdtemp(prefix="yskn_upd_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ui = QS.Ui("en")
        self.st = {}
        self.saved = 0

    def save(self):
        self.saved += 1

    def quiet(self):
        return contextlib.redirect_stdout(io.StringIO())

    def folder(self, name="qs"):
        d = self.tmp / name
        d.mkdir()
        for n in QS.START_FILES:
            (d / n).write_bytes(("old " + n).encode())
        (d / "quickstart.py").write_bytes(('QS_VERSION = "%s"\n' % QS.QS_VERSION).encode())
        return d


class VersionTests(unittest.TestCase):
    def test_compare(self):
        for mod in [U] + ([QS] if QS else []):
            self.assertEqual(mod.parse_version("v0.1.10"), (0, 1, 10))
            self.assertTrue(mod.is_newer("0.1.10", "0.1.9"))
            self.assertTrue(mod.is_newer("v0.2.0", "0.1.99"))
            self.assertTrue(mod.is_newer("v0.1.7", "0.1.6"))
            self.assertTrue(mod.is_newer("v0.2.0", "0.1.7"))
            self.assertTrue(mod.is_newer("v0.2.1", "0.2.0"))
            self.assertTrue(mod.is_newer("v0.3.0", "0.2.1"))
            self.assertTrue(mod.is_newer("v0.3.1", "0.2.1"))
            self.assertTrue(mod.is_newer("v0.3.1", "0.3.0"))
            self.assertTrue(mod.is_newer("v0.3.2", "0.3.1"))
            self.assertTrue(mod.is_newer("v0.3.3", "0.3.2"))
            self.assertTrue(mod.is_newer("v0.3.4", "0.3.3"))
            self.assertTrue(mod.is_newer("v0.3.5", "0.3.4"))
            self.assertFalse(mod.is_newer("0.1.9", "0.1.10"))
            self.assertFalse(mod.is_newer("0.1.6", "0.1.6"))
            self.assertIsNone(mod.parse_version("v0.1.7-test1"))
            self.assertFalse(mod.is_newer("v0.1.7-test1", "0.1.6"))
            self.assertIsNone(mod.parse_version("../0.1.7"))

    def test_same_rule_in_both(self):
        if QS is None:
            self.skipTest("no quickstart.py")
        self.assertEqual(QS.UPDATE_API, U.API)


class CheckTests(Base):
    def check(self, **kw):
        with self.quiet() as out:
            rel = QS.update_check(self.ui, self.st, self.save, self.src, **kw)
        return rel, out.getvalue()

    def test_newer_same_older_prerelease(self):
        cur = QS.parse_version(QS.QS_VERSION)
        newer = "%d.%d.%d" % (cur[0], cur[1], cur[2] + 1)
        older = "%d.%d.%d" % (cur[0], cur[1] - 1, 0) if cur[1] else "0.0.0"
        self.gh.publish(newer)
        rel, out = self.check(force=True)
        self.assertEqual(rel["version"], newer)
        self.assertIn("github.com", out.splitlines()[0])
        self.gh.publish(QS.QS_VERSION)
        rel, out = self.check(force=True)
        self.assertIsNone(rel)
        self.assertIn("latest version", out)
        self.gh.publish(older)
        self.assertIsNone(self.check(force=True)[0])
        self.gh.publish(newer, prerelease=True)
        rel, out = self.check(force=True)
        self.assertIsNone(rel)
        self.assertIn("pre-release", out)
        self.gh.publish(newer)
        self.gh.release["tag_name"] = "v%s-test1" % newer
        self.assertIsNone(self.check(force=True)[0])

    def test_request_carries_nothing_about_this_pc(self):
        self.gh.publish(QS.QS_VERSION)
        self.check(force=True)
        path, headers = self.gh.requests[-1]
        self.assertEqual(path, "/repos/yasakani-dev/yasakani/releases/latest")
        self.assertEqual(headers.get("User-Agent"), "yasakani-quickstart")
        self.assertFalse({k.lower() for k in headers} - {"host", "user-agent", "accept", "accept-encoding", "connection"})

    def test_once_a_day(self):
        self.gh.publish("9.9.9")
        now = 1_000_000_000
        self.assertEqual(self.check(now=now)[0]["version"], "9.9.9")
        self.assertEqual(len(self.gh.api_calls()), 1)
        rel, out = self.check(now=now + 3600)               # known from the state, no request, no line
        self.assertEqual(rel["version"], "9.9.9")
        self.assertEqual(out, "")
        self.assertEqual(len(self.gh.api_calls()), 1)
        self.check(now=now + 25 * 3600)
        self.assertEqual(len(self.gh.api_calls()), 2)

    def test_disabled_makes_no_request(self):
        self.gh.publish("9.9.9")
        a = SimpleNamespace(no_update_check=False)
        self.assertTrue(QS.check_disabled(a, {}, {"YASAKANI_NO_UPDATE_CHECK": "1"}))
        self.assertTrue(QS.check_disabled(a, {"update_check": False}, {}))
        self.assertTrue(QS.check_disabled(SimpleNamespace(no_update_check=True), {}, {}))
        self.assertFalse(QS.check_disabled(a, {}, {"YASAKANI_NO_UPDATE_CHECK": "0"}))
        rel, out = self.check(disabled=True)
        self.assertIsNone(rel)
        self.assertEqual(out, "")
        self.assertEqual(self.gh.requests, [])

    def test_network_failure_does_not_block(self):
        self.gh.close()
        t0 = time.time()
        rel, out = self.check(force=True)
        self.assertIsNone(rel)
        self.assertIn("Could not check", out)
        self.assertLess(time.time() - t0, 10)
        self.assertGreaterEqual(self.saved, 1)

    def test_server_error_and_bad_json(self):
        self.gh.release = 500
        self.assertIsNone(self.check(force=True)[0])
        self.gh.release = ["not", "a", "release"]
        rel, out = self.check(force=True)
        self.assertIsNone(rel)
        self.assertIn("not a release", out)

    def test_dev_base_only_this_computer(self):
        for bad in ("http://example.com:80", "https://127.0.0.1:1", "http://127.0.0.1:80/x", "http://10.0.0.1:8000"):
            with self.assertRaises(QS.UpdateError):
                QS.UpdateSource(bad)
        self.assertEqual(QS.UpdateSource().api, "https://api.github.com/repos/yasakani-dev/yasakani/releases/latest")
        self.assertEqual(QS.UpdateSource().download, "https://github.com/yasakani-dev/yasakani/releases/download/")


class DownloadTests(Base):
    def run_update(self, folder, rel):
        calls = []
        orig = QS.run_quickstart
        QS.run_quickstart = lambda f, args: calls.append((f, list(args))) or 0
        try:
            with self.quiet() as out:
                rc = QS.apply_update(self.ui, str(folder), rel, self.src, True, ["--action", "none"], self.st, self.save)
        finally:
            QS.run_quickstart = orig
        return rc, calls, out.getvalue()

    def fetch(self, version="9.9.9"):
        with self.quiet():
            return QS.fetch_latest(self.src)

    def assert_unchanged(self, d):
        self.assertEqual((d / "START.cmd").read_bytes(), b"old START.cmd")
        self.assertFalse((d / "backup").exists())
        self.assertNotIn("update_backup", self.st)

    def test_update_backup_and_rollback(self):
        d = self.folder()
        old = {n: (d / n).read_bytes() for n in QS.START_FILES}
        new_qs = b'QS_VERSION = "9.9.9"\nprint("new")\n'
        self.gh.publish("9.9.9", files={"quickstart.py": new_qs, "START.cmd": b"new start\r\n", "chat.html": b"<p>new</p>"})
        rc, calls, out = self.run_update(d, self.fetch())
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [(str(d), ["--action", "none"])])
        self.assertEqual((d / "quickstart.py").read_bytes(), new_qs)
        self.assertEqual((d / "START.cmd").read_bytes(), b"new start\r\n")
        self.assertEqual((d / "README.txt").read_bytes(), old["README.txt"])        # not in the new zip: left alone
        bdir = d / "backup" / QS.QS_VERSION
        for n in QS.START_FILES:
            self.assertEqual((bdir / n).read_bytes(), old[n])
        self.assertEqual(self.st["update_backup"], {"version": QS.QS_VERSION, "from": "9.9.9"})
        self.assertFalse(list(d.glob("*.new")))
        # rollback
        calls = []
        orig = QS.run_quickstart
        QS.run_quickstart = lambda f, args: calls.append(list(args)) or 0
        try:
            with self.quiet():
                rc = QS.apply_rollback(self.ui, str(d), self.st, self.save, True, ["--action", "none"])
        finally:
            QS.run_quickstart = orig
        self.assertEqual(rc, 0)
        for n in QS.START_FILES:
            self.assertEqual((d / n).read_bytes(), old[n], n)
        self.assertEqual(self.st.get("update_skip"), "9.9.9")
        self.assertNotIn("update_backup", self.st)
        with self.quiet() as out:
            self.assertEqual(QS.apply_rollback(self.ui, str(d), self.st, self.save, True, []), 1)
        self.assertIn("no backup", out.getvalue())

    def test_rollback_forgets_a_model_the_old_launcher_does_not_know(self):
        for known in (False, True):
            d = self.folder("qs_known" if known else "qs_unknown")
            old_qs = b'T = {"m_quality": 1, "m_speed": 2' + (b', "m_max": 3' if known else b'') + b'}\n'
            (d / "quickstart.py").write_bytes(old_qs)
            self.st.clear()
            self.gh.publish("9.9.9", files={"quickstart.py": b'QS_VERSION = "9.9.9"\n', "START.cmd": b"new start\r\n"})
            rc, calls, out = self.run_update(d, self.fetch())
            self.assertEqual(rc, 0, out)
            self.st["model"] = "max"
            calls = []
            orig = QS.run_quickstart
            QS.run_quickstart = lambda f, args: calls.append(list(args)) or 0
            try:
                with self.quiet() as out:
                    rc = QS.apply_rollback(self.ui, str(d), self.st, self.save, True, ["--action", "none"])
            finally:
                QS.run_quickstart = orig
            self.assertEqual(rc, 0)
            self.assertEqual((d / "quickstart.py").read_bytes(), old_qs)
            self.assertNotIn("update_backup", self.st)
            if known:                                   # that launcher has the model: as before
                self.assertEqual((self.st.get("model"), calls), ("max", [["--action", "none"]]))
            else:                                       # it would stop on the name: the choice is forgotten, nothing is started
                self.assertEqual((self.st.get("model"), calls), (None, []))
                self.assertIn("choose a model", out.getvalue())

    def test_rollback_to_0_3_1_forgets_inkling2(self):
        """0.3.2 brought Inkling-2 Mini into the model question; the launcher of 0.3.1 does not know the name."""
        d = self.folder("qs_031")
        old_qs = b'T = {"m_quality": 1, "m_speed": 2, "m_max": 3}\n'
        (d / "quickstart.py").write_bytes(old_qs)
        self.st.clear()
        self.gh.publish("9.9.9", files={"quickstart.py": b'QS_VERSION = "9.9.9"\n', "START.cmd": b"new start\r\n"})
        rc, calls, out = self.run_update(d, self.fetch())
        self.assertEqual(rc, 0, out)
        self.st["model"] = "inkling2"
        calls = []
        orig = QS.run_quickstart
        QS.run_quickstart = lambda f, args: calls.append(list(args)) or 0
        try:
            with self.quiet() as out:
                rc = QS.apply_rollback(self.ui, str(d), self.st, self.save, True, ["--action", "none"])
        finally:
            QS.run_quickstart = orig
        self.assertEqual((rc, self.st.get("model"), calls), (0, None, []))
        self.assertIn("cannot use the Inkling-2 Mini model (experimental)", out.getvalue())

    def test_answer_no_changes_nothing(self):
        d = self.folder()
        self.gh.publish("9.9.9")
        rel = self.fetch()
        orig_ask = QS.ask
        QS.ask = lambda prompt, default="": "n"
        try:
            with self.quiet():
                rc = QS.apply_update(self.ui, str(d), rel, self.src, False, [], self.st, self.save)
        finally:
            QS.ask = orig_ask
        self.assertIsNone(rc)
        self.assert_unchanged(d)
        self.assertEqual([p for p, _ in self.gh.requests if "/download/" in p], [])

    def test_sha_mismatch_refused(self):
        d = self.folder()
        self.gh.publish("9.9.9", sums=("%s  yasakani-quickstart-9.9.9.zip\n" % ("ab" * 32)).encode())
        rc, calls, out = self.run_update(d, self.fetch())
        self.assertEqual(rc, 9)
        self.assertEqual(calls, [])
        self.assertIn("SHA-256", out)
        self.assert_unchanged(d)

    def test_zip_not_listed_in_sums_refused(self):
        d = self.folder()
        self.gh.publish("9.9.9", sums=b"")
        rc, _, out = self.run_update(d, self.fetch())
        self.assertEqual(rc, 9)
        self.assert_unchanged(d)

    def test_bad_zips_refused(self):
        good = {"quickstart.py": b'QS_VERSION = "9.9.9"\n', "START.cmd": b"x"}
        cases = {
            "traversal": dict(good, **{"../evil.py": b"x"}),
            "path": dict(good, **{"sub/quickstart.py": b"x"}),
            "backslash": dict(good, **{"..\\START.cmd": b"x"}),
            "unknown": dict(good, **{"yskn.dll": b"MZ"}),
            "no START.cmd": {"quickstart.py": b'QS_VERSION = "9.9.9"\n'},
            "other version": {"quickstart.py": b'QS_VERSION = "9.9.8"\n', "START.cmd": b"x"},
        }
        for why, files in cases.items():
            with self.subTest(why):
                d = self.folder("f%d" % len(list(self.tmp.iterdir())))
                self.gh.publish("9.9.9", files=files)
                rc, calls, out = self.run_update(d, self.fetch())
                self.assertEqual(rc, 9, out)
                self.assertEqual(calls, [])
                self.assert_unchanged(d)
                self.assertFalse((self.tmp / "evil.py").exists())

    def test_member_limits(self):
        def zipped(files, **kw):
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, "w", **kw) as z:
                for n, b in files.items():
                    z.writestr(n, b)
            return bio.getvalue()
        ok = {"quickstart.py": b'QS_VERSION = "9.9.9"\n', "START.cmd": b"x"}
        self.assertEqual(set(QS.check_start_zip(zipped(ok), "9.9.9")), {"quickstart.py", "START.cmd"})
        big = dict(ok, **{"README.txt": b"a" * (QS.MAX_START_FILE + 1)})
        with self.assertRaises(QS.UpdateError):
            QS.check_start_zip(zipped(big, compression=zipfile.ZIP_DEFLATED), "9.9.9")   # small zip, large member
        with self.assertRaises(QS.UpdateError):
            QS.check_start_zip(b"not a zip", "9.9.9")
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as z:
            z.writestr("quickstart.py", ok["quickstart.py"])
            z.writestr("START.cmd", b"x")
            z.writestr(zipfile.ZipInfo("chat.html/"), b"")
        with self.assertRaises(QS.UpdateError):
            QS.check_start_zip(bio.getvalue(), "9.9.9")

    def test_url_outside_prefix_refused(self):
        d = self.folder()
        self.gh.publish("9.9.9", assets=[{"name": "yasakani-quickstart-9.9.9.zip", "browser_download_url": "https://example.com/x.zip"},
                                         {"name": "SHA256SUMS.txt", "browser_download_url": self.gh.base +
                                          "/yasakani-dev/yasakani/releases/download/v9.9.9/SHA256SUMS.txt"}])
        rc, calls, out = self.run_update(d, self.fetch())
        self.assertEqual(rc, 9)
        self.assertIn("outside", out)
        self.assert_unchanged(d)
        rel = {"version": "9.9.9", "tag": "v9.9.9", "assets": {"SHA256SUMS.txt": self.gh.base + "/other/owner/releases/download/v9.9.9/SHA256SUMS.txt"}}
        with self.assertRaises(QS.UpdateError):
            QS.release_url(self.src, rel, "SHA256SUMS.txt")
        real = QS.UpdateSource()
        self.assertEqual(QS.release_url(real, {"tag": "v9.9.9", "assets": {}}, "SHA256SUMS.txt"),
                         "https://github.com/yasakani-dev/yasakani/releases/download/v9.9.9/SHA256SUMS.txt")

    def test_cached_release_without_assets(self):
        d = self.folder()
        self.gh.publish("9.9.9")
        rel = {"version": "9.9.9", "tag": "v9.9.9", "page": "https://github.com/yasakani-dev/yasakani/releases/tag/v9.9.9", "assets": {}}
        rc, calls, out = self.run_update(d, rel)
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(calls), 1)

    def test_pointer_saved_before_the_first_file_is_replaced(self):
        """An update that stops while the start files are replaced (the window closed, the PC turned off) can be undone with
        --action rollback: the pointer to the backup is in the saved state before any start file changes."""
        d = self.folder()
        old = {n: (d / n).read_bytes() for n in QS.START_FILES}
        self.gh.publish("9.9.9", files={"quickstart.py": b'QS_VERSION = "9.9.9"\n', "START.cmd": b"new start\r\n",
                                        "chat.html": b"<p>new</p>"})
        rel = self.fetch()
        saved_states = []
        state_file = self.tmp / "quickstart_state.json"

        def save():
            state_file.write_text(json.dumps(self.st), encoding="utf-8")
            saved_states.append(json.loads(state_file.read_text(encoding="utf-8")))

        seen = []
        orig_write = QS._write_files

        def write_then_stop(folder, files):
            # what is on disk at the moment the first start file would be replaced
            seen.append(json.loads(state_file.read_text(encoding="utf-8")))
            first = next(iter(files))
            os.replace(self._new_file(folder, first, files[first]), os.path.join(folder, first))
            raise KeyboardInterrupt                       # stops halfway: one file is new, the others are old

        QS._write_files = write_then_stop
        try:
            with self.quiet(), self.assertRaises(KeyboardInterrupt):
                QS.apply_update(self.ui, str(d), rel, self.src, True, [], self.st, save)
        finally:
            QS._write_files = orig_write
        self.assertEqual(seen, [{"update_backup": {"version": QS.QS_VERSION, "from": "9.9.9"}}])
        # a new run of the launcher reads the saved state and goes back
        st = json.loads(state_file.read_text(encoding="utf-8"))
        calls = []
        orig = QS.run_quickstart
        QS.run_quickstart = lambda f, args: calls.append(list(args)) or 0
        try:
            with self.quiet():
                rc = QS.apply_rollback(self.ui, str(d), st, lambda: None, True, ["--action", "none"])
        finally:
            QS.run_quickstart = orig
        self.assertEqual(rc, 0)
        for n in QS.START_FILES:
            self.assertEqual((d / n).read_bytes(), old[n], n)

    def _new_file(self, folder, name, data):
        p = os.path.join(folder, name + ".new")
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_failed_replace_keeps_the_earlier_pointer(self):
        """When a start file cannot be replaced and the old files are put back, the pointer of an earlier update stays."""
        d = self.folder()
        earlier = {"version": "0.0.1", "from": QS.QS_VERSION}
        self.st["update_backup"] = dict(earlier)
        self.gh.publish("9.9.9", files={"quickstart.py": b'QS_VERSION = "9.9.9"\n', "START.cmd": b"new start\r\n"})
        rel = self.fetch()
        orig_write = QS._write_files
        calls = []

        def fail_once(folder, files):
            calls.append(sorted(files))
            if len(calls) == 1:
                raise PermissionError("in use")
            return orig_write(folder, files)

        QS._write_files = fail_once
        try:
            rc, _, out = self.run_update(d, rel)
        finally:
            QS._write_files = orig_write
        self.assertEqual(rc, 9, out)
        self.assertEqual(len(calls), 2)                   # the try and the restore
        self.assertEqual(self.st["update_backup"], earlier)
        self.assertEqual((d / "START.cmd").read_bytes(), b"old START.cmd")


class YesTests(Base):
    """--yes answers the update question only with --action update; the check at the menu always asks."""

    def test_rule(self):
        A = SimpleNamespace
        self.assertTrue(QS.update_answered(A(yes=True, action="update")))
        self.assertFalse(QS.update_answered(A(yes=True, action="menu")))
        self.assertFalse(QS.update_answered(A(yes=True, action="web")))
        self.assertFalse(QS.update_answered(A(yes=False, action="update")))

    def test_menu_asks_with_yes(self):
        d = self.folder()
        self.gh.publish("9.9.9")
        with self.quiet():
            rel = QS.fetch_latest(self.src)
        asked = []
        orig_ask = QS.ask
        QS.ask = lambda prompt, default="": asked.append(prompt) or "n"
        try:
            with self.quiet():
                rc = QS.apply_update(self.ui, str(d), rel, self.src, QS.update_answered(SimpleNamespace(yes=True, action="menu")),
                                     [], self.st, self.save)
        finally:
            QS.ask = orig_ask
        self.assertIsNone(rc)
        self.assertEqual(len(asked), 1)
        self.assertEqual((d / "START.cmd").read_bytes(), b"old START.cmd")

    def test_every_call_in_main_follows_the_rule(self):
        """In main(), each apply_update gets update_answered(a), except the chat page's button (its press is the yes)."""
        import ast
        tree = ast.parse(Path(QS.__file__).read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        yes_args = [ast.unparse(c.args[4]) for c in ast.walk(main)
                    if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "apply_update"]
        self.assertEqual(sorted(yes_args), ["True", "update_answered(a)", "update_answered(a)", "update_answered(a)"])


@unittest.skipUnless(sys.platform == "win32", "cmd.exe is Windows only")
class StartCmdTests(unittest.TestCase):
    """START.cmd runs quickstart.py; the update replaces START.cmd while that cmd.exe still runs it."""

    def setUp(self):
        if QS is None:
            self.skipTest("no quickstart.py")
        self.tmp = Path(tempfile.mkdtemp(prefix="yskn_cmd_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # a different START.cmd whose bytes at the old offsets would run something else if cmd.exe read on
        self.evil = ("@echo off\r\nrem " + "x" * 40 + "\r\necho GARBAGE-1\r\necho GARBAGE-2\r\necho GARBAGE-3\r\nexit /b 99\r\n").encode()

    def run_start(self, start_bytes):
        d = self.tmp / ("run%d" % len(list(self.tmp.iterdir())))
        d.mkdir()
        (d / "START.cmd").write_bytes(start_bytes)
        (d / "new_start.bin").write_bytes(self.evil)
        # the "quickstart.py" of this test replaces START.cmd the way the update does, then ends with exit code 7
        (d / "quickstart.py").write_text(
            "import importlib.util, os, sys\n"
            "spec = importlib.util.spec_from_file_location('q', %r)\n"
            "q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)\n"
            "here = os.path.dirname(os.path.abspath(__file__))\n"
            "q._write_files(here, {'START.cmd': open(os.path.join(here, 'new_start.bin'), 'rb').read()})\n"
            "print('replaced START.cmd'); sys.exit(7)\n" % str(QS_PATH), encoding="utf-8")
        r = subprocess.run(["cmd.exe", "/d", "/c", str(d / "START.cmd")], input=b"\r\n", capture_output=True, timeout=120)
        return r, (d / "START.cmd").read_bytes()

    def test_replaced_while_running(self):
        r, now = self.run_start(START_CMD.read_bytes())
        out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
        if "Python 3.10 or newer was not found" in out:
            self.skipTest("no Python on PATH for START.cmd")
        self.assertIn("replaced START.cmd", out)
        self.assertEqual(now, self.evil)
        self.assertNotIn("GARBAGE", out)
        self.assertEqual(r.returncode, 7, out)

    def test_old_layout_would_break(self):
        """The control: the START.cmd of 0.1.6 (no block) reads on in the new file after quickstart.py ends."""
        old = (b'@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\nset "PY=python"\r\n:run\r\n%PY% "%~dp0quickstart.py" %*\r\n'
               b'set "RC=%ERRORLEVEL%"\r\necho.\r\npause\r\nexit /b %RC%\r\n')
        r, _ = self.run_start(old)
        out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
        if "replaced START.cmd" not in out:
            self.skipTest("no python on PATH")
        self.assertTrue("GARBAGE" in out or r.returncode != 7, out)


class PageButtonTests(Base):
    def serve(self, update):
        port = QS.pick_port(18766)
        httpd = QS.LocalServer(("127.0.0.1", port), QS.make_web_handler(b"<html>YASAKANI</html>", {"version": QS.QS_VERSION}, 1, port, update))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return port

    def req(self, port, method, path, headers=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            h = {"Host": "127.0.0.1:%d" % port}
            h.update(headers or {})
            c.request(method, path, body=b"" if method == "POST" else None, headers=h)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def test_absent_without_the_launcher_check(self):
        port = self.serve(None)
        own = {"Origin": "http://127.0.0.1:%d" % port}
        self.assertEqual(self.req(port, "POST", "/webchat/update", dict(own, **{"X-Yasakani-Token": "x"}))[0], 404)
        conf = json.loads(self.req(port, "GET", "/webchat/config")[1])
        self.assertNotIn("update_token", conf)

    def test_token_header_and_origin(self):
        up = QS.UpdateRequest({"version": "9.9.9", "page": "https://github.com/yasakani-dev/yasakani/releases/tag/v9.9.9"})
        port = self.serve(up)
        conf = json.loads(self.req(port, "GET", "/webchat/config")[1])
        tok = conf["update_token"]
        self.assertEqual(conf["update"]["version"], "9.9.9")
        own = "http://127.0.0.1:%d" % port
        H = QS.UpdateRequest.HEADER
        self.assertEqual(self.req(port, "POST", "/webchat/update", {"Origin": own})[0], 403)                       # no token
        self.assertEqual(self.req(port, "POST", "/webchat/update", {"Origin": own, H: "0" * 64})[0], 403)           # wrong token
        self.assertEqual(self.req(port, "POST", "/webchat/update", {H: tok})[0], 403)                               # no Origin
        self.assertEqual(self.req(port, "POST", "/webchat/update", {"Origin": "http://example.com", H: tok})[0], 403)
        self.assertEqual(self.req(port, "POST", "/webchat/update", {"Origin": own, H: tok, "Host": "example.com"})[0], 403)
        self.assertEqual(self.req(port, "POST", "/webchat/update", {"Origin": own, H: tok, "Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.req(port, "GET", "/webchat/update", {"Origin": own, H: tok})[0], 404)
        self.assertFalse(up.event.is_set())
        code, body = self.req(port, "POST", "/webchat/update", {"Origin": own, H: tok})
        self.assertEqual(code, 202)
        self.assertEqual(json.loads(body)["version"], "9.9.9")
        self.assertTrue(up.event.wait(5))
        self.assertEqual(self.req(port, "POST", "/webchat/update", {"Origin": own, H: tok})[0], 403)             # once only

    def test_wait_for_stop_sees_the_button(self):
        up = QS.UpdateRequest({"version": "9.9.9", "page": ""})
        proc = SimpleNamespace(poll=lambda: None)
        threading.Timer(0.3, up.event.set).start()
        self.assertEqual(QS.wait_for_stop(proc, up, poll=0.05), "update")
        self.assertEqual(QS.wait_for_stop(SimpleNamespace(poll=lambda: 0), None, poll=0.05), "ended")


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.gh = FakeGitHub()
        self.addCleanup(self.gh.close)
        self.api = self.gh.base + "/repos/yasakani-dev/yasakani/releases/latest"
        self.tmp = Path(tempfile.mkdtemp(prefix="yskn_cli_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def call(self, check=False, prefix=None, yes=False):
        out, ran = io.StringIO(), []
        rc = U.cmd_update(SimpleNamespace(check=check, yes=yes), api=self.api, prefix=str(prefix or self.tmp / "env"),
                          run=lambda cmd: ran.append(list(cmd)) or 0, out=out)
        return rc, out.getvalue(), ran

    def test_manual_install_prints_steps_and_changes_nothing(self):
        self.gh.publish("9.9.9")
        rc, out, ran = self.call()
        self.assertEqual(rc, 0)
        self.assertIn("latest     9.9.9", out)
        self.assertIn("pip install --upgrade", out)
        self.assertIn("Nothing was changed.", out)
        self.assertEqual(ran, [])
        self.assertEqual(len(self.gh.api_calls()), 1)
        self.assertEqual(self.gh.requests[0][1].get("User-Agent"), "yasakani-update")

    def test_check_and_latest(self):
        from yasakani import __version__
        self.gh.publish("9.9.9")
        rc, out, _ = self.call(check=True)
        self.assertEqual(rc, 0)
        self.assertIn("installed  " + __version__, out)
        self.assertNotIn("pip install", out)
        self.gh.publish(__version__)
        rc, out, _ = self.call()
        self.assertIn("This is the latest version.", out)
        self.gh.publish("9.9.9", prerelease=True)
        rc, out, _ = self.call()
        self.assertEqual(rc, 1)
        self.gh.close()
        rc, out, _ = self.call()
        self.assertEqual(rc, 1)
        self.assertIn("could not check", out)

    def test_quickstart_folder_hands_over(self):
        home = self.tmp / "qs"
        env = home / "venv"
        env.mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("home = %s\n" % os.path.dirname(sys.executable), encoding="utf-8")
        (home / "quickstart_state.json").write_text(json.dumps({"quickstart_dir": str(home)}), encoding="utf-8")
        (home / "quickstart.py").write_text("", encoding="utf-8")
        (home / "START.cmd").write_text("", encoding="utf-8")
        rc, out, ran = self.call(prefix=env, yes=True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(ran), 1)
        self.assertEqual(ran[0][-4:], ["update", "--home", str(home), "--yes"])
        if sys.platform == "win32":
            self.assertEqual(ran[0][:5], ["cmd.exe", "/c", "start", "YASAKANI update", str(home / "START.cmd")])
        self.assertEqual(self.gh.requests, [])                    # the launcher asks GitHub, not this command
        rc, out, ran = self.call(prefix=env, check=True)             # --check only looks
        self.assertEqual(ran, [])
        self.assertEqual(len(self.gh.api_calls()), 1)

    def test_not_a_quickstart_folder(self):
        self.assertIsNone(U.quickstart_install(str(self.tmp / "venv")))
        (self.tmp / "quickstart_state.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(U.quickstart_install(str(self.tmp / "venv")))       # no quickstart.py

    def test_other_commands_have_no_network_code(self):
        pkg = ROOT / "yasakani"
        for p in sorted(pkg.glob("*.py")):
            if p.name == "update.py":
                continue
            s = p.read_text(encoding="utf-8")
            for bad in ("urllib.request", "http.client", "create_connection", "urlopen"):
                self.assertNotIn(bad, s, "%s uses %s" % (p.name, bad))
        cli = (pkg / "cli.py").read_text(encoding="utf-8")
        self.assertEqual(cli.count("from .update import"), 1)
        self.assertIn("def cmd_update(args) -> int:\n    from .update import", cli)


class RemoveTreeTests(unittest.TestCase):
    """The engine folder is replaced right after an update, while the server of the version before may still be ending."""

    def setUp(self):
        if QS is None:
            self.skipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")
        self.tmp = Path(tempfile.mkdtemp(prefix="yskn_rm_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def folder(self):
        d = self.tmp / "engine"
        d.mkdir()
        (d / "yskn.dll").write_bytes(b"x")
        return d

    def test_plain_removal_and_missing_folder(self):
        d = self.folder()
        self.assertTrue(QS.remove_tree(str(d)))
        self.assertFalse(d.exists())
        self.assertTrue(QS.remove_tree(str(d)))             # already gone: nothing to do

    def test_waits_while_a_file_is_in_use(self):
        d = self.folder()
        real, calls, said, slept = shutil.rmtree, [], [], []

        def busy_twice(path, *a, **kw):
            calls.append(path)
            if len(calls) <= 2:
                raise PermissionError(13, "in use", str(d / "yskn.dll"))
            return real(path, *a, **kw)

        QS.shutil.rmtree = busy_twice
        self.addCleanup(setattr, QS.shutil, "rmtree", real)
        self.assertTrue(QS.remove_tree(str(d), say=lambda: said.append(1), sleep=slept.append))
        self.assertFalse(d.exists())
        self.assertEqual((len(calls), len(said), slept), (3, 1, [0.5, 0.5]))      # said once, tried again twice

    def test_gives_up_after_the_time(self):
        d = self.folder()
        real, now, said = shutil.rmtree, [0.0], []

        def always_busy(path, *a, **kw):
            raise PermissionError(13, "in use", str(d / "yskn.dll"))

        QS.shutil.rmtree = always_busy
        self.addCleanup(setattr, QS.shutil, "rmtree", real)
        ok = QS.remove_tree(str(d), seconds=2.0, say=lambda: said.append(1),
                            sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0])
        self.assertFalse(ok)
        self.assertTrue((d / "yskn.dll").is_file())
        self.assertEqual(len(said), 1)

    def test_main_uses_it_for_the_engine_folder(self):
        src = QS_PATH.read_text(encoding="utf-8")
        self.assertIn('remove_tree(edir, say=lambda: ui.say("engine_wait"))', src)
        self.assertNotIn("shutil.rmtree(edir)", src)

    def test_wait_released(self):
        d = self.folder()
        dll = str(d / "yskn.dll")
        self.assertTrue(QS.wait_released(dll))                               # nobody holds it
        self.assertTrue(QS.wait_released(str(d / "missing.dll")))
        now, slept, tries = [0.0], [], []

        def sleep(s):
            slept.append(s)
            now[0] += s

        def busy_three_times(path, mode="r", *a, **kw):                      # stands in for a library that is still loaded
            if path == dll and "+" in mode:
                tries.append(1)
                if len(tries) <= 3:
                    raise PermissionError(13, "in use", dll)
            return open(path, mode, *a, **kw)

        QS.open = busy_three_times                                           # the module's own name wins over the built-in
        self.addCleanup(lambda: QS.__dict__.pop("open", None))
        self.assertTrue(QS.wait_released(dll, sleep=sleep, clock=lambda: now[0]))
        self.assertEqual((len(tries), slept), (4, [0.2, 0.2, 0.2]))

        def always_busy(path, mode="r", *a, **kw):
            raise PermissionError(13, "in use", dll)

        QS.open = always_busy
        self.assertFalse(QS.wait_released(dll, seconds=1.0, sleep=sleep, clock=lambda: now[0]))

    def test_stop_waits_for_the_library(self):
        seen = []
        real = QS.wait_released
        QS.wait_released = lambda path, *a, **kw: seen.append(path) or True
        self.addCleanup(setattr, QS, "wait_released", real)
        ended = SimpleNamespace(poll=lambda: 0)
        QS.stop_process(ended)                                               # no library named: nothing to wait for
        QS.stop_process(ended, lib="engine/yskn.dll")
        self.assertEqual(seen, ["engine/yskn.dll"])
        src = QS_PATH.read_text(encoding="utf-8")
        self.assertIn("stop_process(proc, lib=dll)", src)


if __name__ == "__main__":
    unittest.main()
