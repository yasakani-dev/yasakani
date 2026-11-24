# SPDX-License-Identifier: Apache-2.0
"""The page in the browser of the quickstart (quickstart/quickstart.py): what the launcher's local server answers, the quit button
and the model switch of the page (control requests), what the page is told while the model loads, the direct start from the second
run and the menu.

No GPU, no model and no network beyond 127.0.0.1.  A small stand-in for the yasakani package is written into a temporary folder: its
`python -m yasakani serve` answers /health ("loading", then "ok"), /v1/models and a short stream, and every other command writes one
line into a log.  The launcher starts it the way it starts the real one.  Test servers use ports 28100 to 28999 only.

The keys of the console window are a stand-in too (RunWebBase.console), and START.cmd is run by cmd.exe with a quickstart.py that
only ends with a given exit code.
"""
import contextlib
import http.client
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent

FAKE_YASAKANI = r'''"""A stand-in for the yasakani package (tests of the quickstart's page in the browser)."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID, CONTEXT = os.environ.get("YSKN_FAKE_MODEL_ID") or "qwen3.8-27b", 262144


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() < self.server.t_ready:
                return self._json(503, {"status": "loading", "version": "fake", "model": MODEL_ID})
            return self._json(200, {"status": "ok", "version": "fake", "model": MODEL_ID, "max_context_tokens": CONTEXT})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
        if self.path.startswith("/fake/exit/") and self.server.may_exit:
            os._exit(int(self.path.rsplit("/", 1)[1]))      # as a server that ends by itself, with this exit code
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for i in range(1, 41):
            ev = {"choices": [{"index": 0, "delta": {"content": "%d " % i}}]}
            self.wfile.write(b"data: " + json.dumps(ev).encode("utf-8") + b"\n\n")
            self.wfile.flush()
            time.sleep(0.005)
        usage = {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 40, "total_tokens": 52}}
        self.wfile.write(b"data: " + json.dumps(usage).encode("utf-8") + b"\n\ndata: [DONE]\n\n")
        self.wfile.flush()


class Engine(ThreadingHTTPServer):
    daemon_threads = True

    may_exit = False                                    # only the server that runs as a process of its own (main below)

    def __init__(self, port, load=0.0):
        self.t_ready = time.monotonic() + load
        super().__init__(("127.0.0.1", port), Handler)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""

    def opt(name):
        return argv[argv.index(name) + 1] if name in argv[:-1] else None

    log = os.environ.get("YSKN_FAKE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cmd": cmd, "model_dir": opt("--model-dir"), "pid": os.getpid()}) + "\n")
    if cmd != "serve":
        time.sleep(float(os.environ.get("YSKN_FAKE_CHECK_SECONDS", "0")))     # as a check that takes its time
        rc_file = os.environ.get("YSKN_FAKE_RC_FILE")       # a file, so that a test can change it while the launcher runs
        if rc_file and os.path.isfile(rc_file):
            with open(rc_file, encoding="utf-8") as f:
                return int(f.read().strip() or 0)
        return 0
    if os.environ.get("YSKN_FAKE_SERVE_FAILS"):
        return 3                                        # as a server that cannot load its model
    srv = Engine(int(opt("--port")), float(os.environ.get("YSKN_FAKE_LOAD", "0.3")))
    srv.may_exit = True
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def load_quickstart():
    """quickstart/quickstart.py of the repository (or the file YASAKANI_QUICKSTART names); None when it is not there (the engine
    zip)."""
    for p in (os.environ.get("YASAKANI_QUICKSTART"), ROOT.parent / "quickstart" / "quickstart.py"):
        if p and Path(p).is_file():
            spec = importlib.util.spec_from_file_location("yasakani_quickstart_web_test", str(p))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


QS = load_quickstart()
REAL_ASK = getattr(QS, "ask", None)                     # LauncherTests puts a stand-in in its place
PAGE_PATH = Path(QS.__file__).resolve().parent / "chat.html" if QS is not None else None
START_CMD = Path(QS.__file__).resolve().parent / "START.cmd" if QS is not None else None
FAKE_DIR = None                                         # the folder with the stand-in package (setUpModule)
FAKE = None                                             # the same file, loaded here for the servers inside this process
_next_port = [28100 + (os.getpid() * 7) % 600]


def setUpModule():
    global FAKE_DIR, FAKE
    FAKE_DIR = Path(tempfile.mkdtemp(prefix="yskn_qsweb_"))
    pkg = FAKE_DIR / "yasakani"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__main__.py").write_text(FAKE_YASAKANI, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("yskn_fake_engine", str(pkg / "__main__.py"))
    FAKE = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(FAKE)
    finally:
        sys.dont_write_bytecode = old


def tearDownModule():
    shutil.rmtree(str(FAKE_DIR), ignore_errors=True)


def free_port():
    """A free port between 28100 and 28940 (the launcher may go up to 59 above the port it is given)."""
    for _ in range(840):
        p = _next_port[0]
        _next_port[0] = 28100 + (p - 28100 + 61) % 840
        if QS.port_is_free(p):
            return p
    raise unittest.SkipTest("no free port between 28100 and 28940")


def closed(port):
    """True when nothing listens on the port.  On Windows this is asked by binding it (a refused connection takes about two seconds
    there); elsewhere by connecting (a port that was just closed cannot be bound again at once)."""
    if sys.platform == "win32":
        return QS.port_is_free(port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def wait_until(fn, seconds=30.0, step=0.05):
    end = time.monotonic() + seconds
    while True:
        v = fn()
        if v or time.monotonic() >= end:
            return v
        time.sleep(step)


def request(port, method, path, headers=None, body=None, timeout=20):
    """One request to 127.0.0.1:port with the Host a browser sends.  -> (status, body bytes, headers)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        h = {"Host": "127.0.0.1:%d" % port}
        h.update(headers or {})
        if method == "POST" and body is None:
            body = b""
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        return r.status, r.read(), r
    finally:
        c.close()


def too_long(port, path, headers, n, timeout=20):
    """A POST that says it has n bytes of body and sends none.  -> the status.  The server refuses a body over its limit from
    Content-Length alone and closes the connection without reading it; a body that was sent would still wait unread there, and on
    Windows closing a socket with unread data resets the connection, which can drop the answer before it is read
    (ConnectionAbortedError, WinError 10053, under load).  The answer to the header is what is checked here."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.putrequest("POST", path, skip_host=True, skip_accept_encoding=True)
        for k, v in dict({"Host": "127.0.0.1:%d" % port}, **(headers or {})).items():
            c.putheader(k, v)
        c.putheader("Content-Length", str(n))
        c.endheaders()
        r = c.getresponse()
        r.read()
        return r.status
    finally:
        c.close()


def get_json(port, path):
    """The JSON of a GET, or None while nothing answers at that port."""
    try:
        code, body, _ = request(port, "GET", path, timeout=5)
        return json.loads(body) if code == 200 else None
    except (OSError, http.client.HTTPException, ValueError):
        return None


class Children:
    """Stands in for subprocess.Popen while the launcher runs.  The Python of the private environment does not exist in a test:
    a command that names it runs with the Python of the test, in the folder of the stand-in package (so `-m yasakani` finds that
    one and never the real package).  Every child is remembered, to check that none is left."""

    def __init__(self):
        self.procs, self._real = [], subprocess.Popen

    def __call__(self, args, **kw):
        if args and str(args[0]).replace("\\", "/").endswith("venv/Scripts/python.exe"):
            args = [sys.executable] + list(args[1:])
            kw["cwd"] = str(FAKE_DIR)
        p = self._real(args, **kw)
        self.procs.append(p)
        return p

    def all_ended(self):
        return all(p.poll() is not None for p in self.procs)

    def stop_all(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


class Base(unittest.TestCase):
    def setUp(self):
        if QS is None:
            self.skipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")
        if not PAGE_PATH.is_file():
            self.skipTest("quickstart/chat.html not found")
        self.page = PAGE_PATH.read_bytes()
        self.tmp = Path(tempfile.mkdtemp(prefix="yskn_qsweb_t_"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        self.ui = QS.Ui("en")

    def quiet(self):
        return contextlib.redirect_stdout(io.StringIO())

    def engine(self, load=0.0):
        """The stand-in engine inside this process.  -> its port."""
        port = free_port()
        srv = FAKE.Engine(port, load)
        threading.Thread(target=srv.serve_forever, args=(0.05,), daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return port

    def serve(self, engine_port=1, update=None, control=None, session=None, gpu=None, conf=None, page=None):
        """The launcher's local server with its handler.  -> its port."""
        port = free_port()
        conf = conf or {"lang": "en", "model": "quality", "version": QS.QS_VERSION, "repo": QS.REPO_URL, "support": QS.SUPPORT_URL}
        handler = QS.make_web_handler(self.page if page is None else page, conf, engine_port, port, update, control, session, gpu)
        httpd = QS.LocalServer(("127.0.0.1", port), handler)
        threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
        self.addCleanup(httpd.close_connections)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return port

    def own(self, port, **more):
        h = {"Origin": "http://127.0.0.1:%d" % port}
        h.update(more)
        return h


# ------------------------------------------------------------------------------------------------------------------ the page
class PageTests(Base):
    def test_page_with_the_strict_csp(self):
        port = self.serve()
        code, body, r = request(port, "GET", "/")
        self.assertEqual(code, 200)
        self.assertTrue((r.getheader("Content-Type") or "").startswith("text/html"))
        csp = r.getheader("Content-Security-Policy") or ""
        self.assertEqual(csp, QS.PAGE_CSP)
        self.assertEqual(csp, "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
                              "base-uri 'none'; form-action 'none'")
        self.assertEqual(body, self.page)
        self.assertIn(b"YASAKANI", body)
        self.assertEqual(r.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(request(port, "GET", "/index.html")[0], 200)

    def test_page_loads_nothing_from_outside(self):
        port = self.serve()
        page = request(port, "GET", "/")[1]
        outside = re.compile(rb"^(?:[a-z][a-z0-9+.\-]*:)?//", re.I)        # http://host, https://host, //host
        self.assertIsNone(re.search(rb"<link\b", page, re.I), "<link>")
        self.assertIsNone(re.search(rb"<script[^>]*\ssrc\s*=", page, re.I), "<script src>")
        self.assertIsNone(re.search(rb"@import", page, re.I), "@import")
        for m in re.finditer(rb"\b(?:src|href)\s*=\s*[\"']?\s*([^\s\"'>;]*)", page, re.I):
            self.assertIsNone(outside.match(m.group(1)), m.group(0))
        for m in re.finditer(rb"url\(\s*[\"']?\s*([^)\"'\s]*)", page, re.I):
            self.assertIsNone(outside.match(m.group(1)), m.group(0))

    def test_page_knows_every_model_of_the_launcher(self):
        """0.3.2: the page names every model of MODELS in both languages (Inkling-2 Mini marked experimental) and can switch to it."""
        text = self.page.decode("utf-8")
        known = re.search(r"const KNOWN = \{([^}]*)\}", text).group(1)
        self.assertEqual(sorted(re.findall(r"(\w+): 1", known)), sorted(QS.MODELS))
        ja, en = text.index("  ja: {"), text.index("  en: {")
        for part in (text[ja:en], text[en:]):
            for name in QS.MODELS:
                self.assertIn("m_%s: '" % name, part)
            self.assertIn("e_title_ink: '", part)
        self.assertIn("m_inkling2: 'Inkling-2 Mini（試験的）'", text[ja:en])
        self.assertIn("m_inkling2: 'Inkling-2 Mini model (experimental)'", text[en:])
        self.assertIn("rows.length < Object.keys(KNOWN).length", text)               # the hint while not all are downloaded

    def test_page_builds_no_html_from_strings(self):
        for word in (rb"innerHTML", rb"outerHTML", rb"insertAdjacentHTML", rb"document\.write", rb"\beval\s*\(",
                     rb"new\s+Function"):
            self.assertIsNone(re.search(word, self.page), word)

    def test_other_files_are_not_served(self):
        port = self.serve()
        for path in ("/yasakani_env.cmd", "/quickstart_state.json", "/chat.html", "/webchat/", "/webchat/shutdown"):
            self.assertEqual(request(port, "GET", path)[0], 404, path)

    def test_no_answer_may_be_shown_inside_another_page(self):
        """X-Frame-Options: DENY on the page and on the JSON answers (the CSP of the page stays as it is)."""
        control = QS.ControlRequest()
        port = self.serve(self.engine(), None, control, QS.WebSession("quality", installed=("quality", "speed")))
        for path, want in (("/", 200), ("/index.html", 200), ("/webchat/config", 200), ("/webchat/status", 200), ("/health", 200),
                           ("/v1/models", 200), ("/favicon.ico", 404), ("/nothing", 404)):
            code, _, r = request(port, "GET", path)
            self.assertEqual((code, r.getheader("X-Frame-Options")), (want, "DENY"), path)
        code, _, r = request(port, "GET", "/", {"Host": "example.com"})
        self.assertEqual((code, r.getheader("X-Frame-Options")), (403, "DENY"))
        code, _, r = request(port, "POST", "/webchat/shutdown", self.own(port))                    # no token
        self.assertEqual((code, r.getheader("X-Frame-Options")), (403, "DENY"))
        code, _, r = request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: control.token}))
        self.assertEqual((code, r.getheader("X-Frame-Options")), (202, "DENY"))
        self.assertNotIn("frame-ancestors", QS.PAGE_CSP)                                           # the CSP was not touched for it


# ------------------------------------------------------------------------------------------- Host, Origin and Sec-Fetch-Site
class RefusalTests(Base):
    GETS = ("/", "/index.html", "/webchat/config", "/webchat/status", "/health", "/v1/models")
    POSTS = ("/v1/chat/completions", "/webchat/update", "/webchat/shutdown", "/webchat/switch_model")

    def test_wrong_host_origin_and_cross_site_are_403(self):
        update = QS.UpdateRequest({"version": "9.9.9", "page": QS.REPO_URL + "/releases/tag/v9.9.9"})
        control = QS.ControlRequest()
        session = QS.WebSession("quality", installed=("quality", "speed"))
        port = self.serve(self.engine(), update, control, session)
        tokens = {QS.UpdateRequest.HEADER: update.token, QS.ControlRequest.HEADER: control.token}
        bad = ({"Host": "example.com"}, {"Host": "127.0.0.1:%d" % (port + 1)}, {"Host": "127.0.0.1"},
               {"Origin": "http://example.com"}, {"Origin": "http://127.0.0.1:%d" % (port + 1)}, {"Origin": "null"},
               {"Origin": "https://127.0.0.1:%d" % port}, {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"})
        for wrong in bad:
            for path in self.GETS:
                with self.subTest(method="GET", path=path, wrong=wrong):
                    self.assertEqual(request(port, "GET", path, wrong)[0], 403)
            for path in self.POSTS:
                with self.subTest(method="POST", path=path, wrong=wrong):
                    h = dict(self.own(port), **tokens)      # everything else is right: only the one header is wrong
                    h.update(wrong)
                    self.assertEqual(request(port, "POST", path, h, b'{"model": "speed"}')[0], 403)
        self.assertFalse(update.event.is_set() or control.event.is_set())
        self.assertEqual((update.token, control.token), (tokens[QS.UpdateRequest.HEADER], tokens[QS.ControlRequest.HEADER]))
        self.assertEqual(session.state(), "loading")

    def test_the_same_requests_from_this_address_are_answered(self):
        """The control of the test above: with the right headers each path answers (so the 403s come from the one wrong header)."""
        update = QS.UpdateRequest({"version": "9.9.9", "page": ""})
        port = self.serve(self.engine(), update, QS.ControlRequest(), QS.WebSession("quality", installed=("quality", "speed")))
        good = dict(self.own(port), **{"Sec-Fetch-Site": "same-origin"})
        for path in self.GETS:
            self.assertEqual(request(port, "GET", path, good)[0], 200, path)
        self.assertEqual(request(port, "GET", "/", {"Host": "localhost:%d" % port})[0], 200)
        code, body, _ = request(port, "POST", "/v1/chat/completions", good, b"{}")
        self.assertEqual(code, 200)
        self.assertIn(b"[DONE]", body)


# ------------------------------------------------------------------------------------------------------ config and status
class ConfigStatusTests(Base):
    def test_config_keys(self):
        control = QS.ControlRequest()
        session = QS.WebSession("quality", installed=("quality", "speed"))
        gpu = QS.GpuWatch(dict(index=0, uuid="GPU-a", name="NVIDIA GeForce RTX 5090", total=32607, used=900, driver="616.92"),
                          query=lambda: [dict(index=0, uuid="GPU-a", name="NVIDIA GeForce RTX 5090", total=32607, used=14321,
                                              driver="616.92")])
        port = self.serve(28765, None, control, session, gpu)
        code, body, r = request(port, "GET", "/webchat/config")
        self.assertEqual(code, 200)
        self.assertEqual(r.getheader("Cache-Control"), "no-store")
        c = json.loads(body)
        self.assertEqual(c["api_base"], "http://127.0.0.1:28765/v1")
        self.assertEqual(c["control_token"], control.token)
        self.assertRegex(c["control_token"], r"^[0-9a-f]{64}$")
        self.assertEqual(c["gpu"], {"name": "NVIDIA GeForce RTX 5090", "driver": "616.92", "total_mib": 32607, "used_mib": 14321})
        self.assertEqual(c["models_installed"], [{"name": "quality", "current": True}, {"name": "speed", "current": False}])
        self.assertEqual((c["model_id"], c["max_context_tokens"]), ("qwen3.8-27b", None))        # the model is not ready yet
        self.assertEqual((c["lang"], c["model"], c["version"], c["repo"], c["support"]),
                         ("en", "quality", QS.QS_VERSION, QS.REPO_URL, QS.SUPPORT_URL))
        self.assertNotIn("update", c)
        self.assertNotIn("update_token", c)

    def test_two_servers_have_two_tokens(self):
        self.assertNotEqual(QS.ControlRequest().token, QS.ControlRequest().token)

    def test_handler_without_session_and_control(self):
        """As --web-test and the callers of before: the new keys are there, nothing can be asked of the launcher."""
        port = self.serve()
        c = json.loads(request(port, "GET", "/webchat/config")[1])
        self.assertIsNone(c["control_token"])
        self.assertEqual(c["models_installed"], [])
        self.assertEqual(c["gpu"], {"name": None, "driver": None, "total_mib": None, "used_mib": None})
        self.assertEqual(c["api_base"], "http://127.0.0.1:1/v1")
        self.assertEqual(request(port, "GET", "/webchat/status")[0], 404)
        for path in ("/webchat/shutdown", "/webchat/switch_model", "/webchat/update"):
            self.assertEqual(request(port, "POST", path, self.own(port, **{QS.ControlRequest.HEADER: "0" * 64}))[0], 404, path)

    def test_status_moves_from_loading_to_ready(self):
        eport = self.engine(load=0.6)
        told = []
        session = QS.WebSession("speed", None, eport, on_ready=told.append)
        port = self.serve(eport, None, QS.ControlRequest(), session)
        s = json.loads(request(port, "GET", "/webchat/status")[1])
        self.assertEqual((s["state"], s["model"]), ("loading", "speed"))
        self.assertIsInstance(s["elapsed"], (int, float))
        self.assertEqual(sorted(s), ["elapsed", "model", "state"])
        session.watch(poll=0.05)
        session.watch(poll=0.05)                            # a second call starts nothing
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 20))
        s2 = json.loads(request(port, "GET", "/webchat/status")[1])
        self.assertGreaterEqual(s2["elapsed"], s["elapsed"])
        c = json.loads(request(port, "GET", "/webchat/config")[1])
        self.assertEqual((c["model_id"], c["max_context_tokens"]), ("qwen3.8-27b", 262144))
        self.assertTrue(wait_until(lambda: len(told) == 1, 5))
        time.sleep(0.2)
        self.assertEqual(len(told), 1)

    def test_status_has_a_phase_only_when_there_is_one(self):
        s = QS.WebSession("speed", None, None, installed=("quality", "speed"))
        self.assertEqual(sorted(s.status()), ["elapsed", "model", "state"])
        s.leave("switching")
        s.phase = "checking"
        self.assertEqual({k: v for k, v in s.status().items() if k != "elapsed"},
                         {"state": "switching", "phase": "checking", "model": "speed"})

    def test_server_for_the_checks_of_a_model_switch(self):
        """open_check_server: the page's address answers while no `yasakani serve` runs.  It says what happens and does nothing."""
        port = free_port()
        self.assertIsNone(QS.open_check_server(self.ui, None, "speed"))                # no page waits anywhere
        httpd = QS.open_check_server(self.ui, port, "speed", ("quality", "speed"))
        self.assertIsNotNone(httpd)
        try:
            self.assertEqual(httpd.server_address[1], port)
            s = get_json(port, "/webchat/status")
            self.assertEqual(sorted(s), ["elapsed", "model", "phase", "state"])
            self.assertEqual((s["state"], s["phase"], s["model"]), ("switching", "checking", "speed"))
            self.assertIsInstance(s["elapsed"], (int, float))
            time.sleep(0.3)
            self.assertGreater(get_json(port, "/webchat/status")["elapsed"], s["elapsed"])     # the seconds since the checks began
            c = get_json(port, "/webchat/config")
            self.assertIsNone(c["control_token"])
            self.assertIsNone(c["api_base"])                                           # no engine runs now
            self.assertEqual((c["model"], c["version"], c["lang"]), ("speed", QS.QS_VERSION, "en"))
            self.assertEqual(c["models_installed"], [{"name": "quality", "current": False}, {"name": "speed", "current": True}])
            self.assertNotIn("update_token", c)
            code, body, r = request(port, "GET", "/")
            self.assertEqual((code, body, r.getheader("Content-Security-Policy")), (200, self.page, QS.PAGE_CSP))
            t0 = time.monotonic()
            for method, path in (("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions")):
                self.assertEqual(request(port, method, path, self.own(port), b"{}" if method == "POST" else None)[0], 503, path)
            self.assertLess(time.monotonic() - t0, 3)                                  # answered at once: nothing is asked
            h = self.own(port, **{QS.ControlRequest.HEADER: "0" * 64, QS.UpdateRequest.HEADER: "0" * 64})
            for path in ("/webchat/shutdown", "/webchat/switch_model", "/webchat/update"):
                self.assertEqual(request(port, "POST", path, h, b'{"model": "quality"}')[0], 404, path)
            self.assertEqual(request(port, "GET", "/webchat/status", {"Host": "example.com"})[0], 403)
            self.assertEqual(request(port, "GET", "/webchat/status", {"Origin": "http://example.com"})[0], 403)
        finally:
            QS.close_page_server(httpd)
        self.assertTrue(wait_until(lambda: closed(port), 10))

    def test_status_failed_stopping_switching(self):
        ended = []
        proc = SimpleNamespace(poll=lambda: 3 if ended else None)
        session = QS.WebSession("quality", proc, 1)
        port = self.serve(1, None, None, session)
        self.assertEqual(get_json(port, "/webchat/status")["state"], "loading")
        self.assertFalse(session.told_failed.is_set())
        ended.append(1)                                     # the server ended and the launcher did not ask for it
        self.assertEqual(get_json(port, "/webchat/status")["state"], "failed")
        self.assertTrue(session.told_failed.is_set())
        for state in ("stopping", "switching"):
            s = QS.WebSession("quality", SimpleNamespace(poll=lambda: None), 1)
            s.leave(state)
            s.leave("stopping" if state == "switching" else "switching")       # the first call counts
            s.proc = SimpleNamespace(poll=lambda: 0)        # the launcher has stopped it: that is not "failed"
            self.assertEqual(s.state(), state)

    def test_probe_takes_only_what_it_understands(self):
        answers = {}
        orig = QS.get_json
        QS.get_json = lambda port, path, headers=None, timeout=5: answers[path]
        try:
            s = QS.WebSession("quality", None, 1)
            answers["/health"] = (503, {"status": "loading"}, None)
            self.assertFalse(s.probe())
            answers["/health"] = (200, ["ok"], None)
            self.assertFalse(s.probe())
            answers["/health"] = (200, {"status": "ok", "max_context_tokens": True}, None)
            answers["/v1/models"] = (401, {"error": {}}, None)
            self.assertTrue(s.probe())
            self.assertEqual((s.max_context, s.model_id), (None, QS.DEFAULT_MODEL_ID))
            s = QS.WebSession("quality", None, 1)
            answers["/health"] = (200, {"status": "ok", "max_context_tokens": 131072}, None)
            answers["/v1/models"] = (200, {"data": [{"id": "x" * 500}]}, None)
            self.assertTrue(s.probe())
            self.assertEqual((s.max_context, s.model_id), (131072, QS.DEFAULT_MODEL_ID))
            # the id is passed on only when it is a plain name: the page puts it into the samples for other programs
            for odd in ('m"&calc&rem x y', "two words", "a&b", 'a"b', "a'b", "a|b", "a<b", "a>b", "a^b", "a%PATH%b", "a`b", "a$b",
                        "a;b", "a\\b", "a(b)", "a\nb", "a\tb", "name\n", " name", "name ", "-name", ".name", "/name", "", "x" * 201,
                        "\u30e2\u30c7\u30eb", 7, None, ["name"]):
                s = QS.WebSession("quality", None, 1)
                answers["/v1/models"] = (200, {"data": [{"id": odd}]}, None)
                self.assertTrue(s.probe())
                self.assertEqual(s.model_id, QS.DEFAULT_MODEL_ID, repr(odd))
            for plain in ("qwen3.8-27b", "Qwen3.8-27B-YASAKANI-quality", "org/name_1.2:latest@a+b", "x" * 200, "7"):
                s = QS.WebSession("quality", None, 1)
                answers["/v1/models"] = (200, {"data": [{"id": plain}]}, None)
                self.assertTrue(s.probe())
                self.assertEqual(s.model_id, plain)
        finally:
            QS.get_json = orig

    def test_odd_model_id_of_the_engine_is_not_passed_on(self):
        """An engine (a model folder of the user's own) that reports an id with quotes, & and spaces: the page gets the usual id."""
        odd = 'm"&calc&rem "two words"'
        eport = self.engine()
        session = QS.WebSession("custom", None, eport)
        port = self.serve(eport, None, None, session)
        with mock.patch.object(FAKE, "MODEL_ID", odd):
            self.assertEqual(json.loads(request(port, "GET", "/v1/models")[1])["data"][0]["id"], odd)      # the engine says so
            session.watch(poll=0.05)
            self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 20))
        code, body, _ = request(port, "GET", "/webchat/config")
        self.assertEqual(json.loads(body)["model_id"], QS.DEFAULT_MODEL_ID)
        self.assertEqual(json.loads(body)["max_context_tokens"], 262144)                # the rest of the answer is taken
        for piece in (b"calc", b"&", b"two words"):
            self.assertNotIn(piece, body)

    def test_gpu_line_is_cached_and_never_fails(self):
        calls, now, fail = [], [100.0], []
        first = dict(index=1, uuid="GPU-b", name="NVIDIA GeForce RTX 4090", total=24564, used=700, driver="580.88")

        def query():
            calls.append(now[0])
            if fail:
                raise OSError("nvidia-smi is gone")
            return [dict(index=0, uuid="GPU-a", name="other", total=1, used=1, driver="1"),
                    dict(first, used=20000 + len(calls))]

        g = QS.GpuWatch(first, query=query, clock=lambda: now[0])
        self.assertEqual(g.get(), {"name": "NVIDIA GeForce RTX 4090", "driver": "580.88", "total_mib": 24564, "used_mib": 20001})
        now[0] += 1.9
        self.assertEqual(g.get()["used_mib"], 20001)        # within 2 s: the same answer, nvidia-smi is not asked again
        self.assertEqual(len(calls), 1)
        now[0] += 0.2
        self.assertEqual(g.get()["used_mib"], 20002)
        self.assertEqual(g.get(fresh=True)["used_mib"], 20003)
        fail.append(1)
        now[0] += 5
        self.assertEqual(g.get(), {"name": "NVIDIA GeForce RTX 4090", "driver": "580.88", "total_mib": 24564, "used_mib": None})
        self.assertEqual(QS.GPU_CACHE_SECONDS, 2.0)
        # a GPU without a UUID is found by its index; a GPU that is not listed any more keeps the facts of step 1
        g = QS.GpuWatch(dict(first, uuid=None), query=lambda: [dict(first, uuid=None, used=5)])
        self.assertEqual(g.get()["used_mib"], 5)
        g = QS.GpuWatch(first, query=lambda: [])
        self.assertEqual(g.get(), {"name": "NVIDIA GeForce RTX 4090", "driver": "580.88", "total_mib": 24564, "used_mib": None})

    def test_gpu_query_uses_the_launchers_parser_with_a_short_time_limit(self):
        seen = {}

        def run(cmd, **kw):
            seen.update(cmd=cmd, kw=kw)
            return SimpleNamespace(returncode=0, stdout="0, GPU-a, NVIDIA GeForce RTX 5090, 12.0, 32607, 4321, 616.92\n")

        with mock.patch.object(QS.subprocess, "run", run):
            g = QS.GpuWatch(dict(index=0, uuid="GPU-a", name="x", total=1, used=1, driver="1"))
            self.assertEqual(g.get(), {"name": "NVIDIA GeForce RTX 5090", "driver": "616.92", "total_mib": 32607, "used_mib": 4321})
        self.assertEqual(seen["cmd"][0], "nvidia-smi")
        self.assertEqual(seen["kw"]["timeout"], QS.GPU_QUERY_TIMEOUT)
        self.assertLessEqual(QS.GPU_QUERY_TIMEOUT, 10)


# ------------------------------------------------------------------------------------------------------- control requests
class ControlTests(Base):
    def setUp(self):
        super().setUp()
        self.control = QS.ControlRequest()
        self.session = QS.WebSession("quality", SimpleNamespace(poll=lambda: None), 1, installed=("quality", "speed"))
        self.port = self.serve(1, None, self.control, self.session)
        self.H = QS.ControlRequest.HEADER
        self.tok = json.loads(request(self.port, "GET", "/webchat/config")[1])["control_token"]

    def post(self, path, headers, body=None):
        code, data, _ = request(self.port, "POST", path, headers, body)
        try:
            return code, json.loads(data)
        except ValueError:
            return code, None

    def test_header_name(self):
        self.assertEqual(self.H, "X-YASAKANI-Control")
        self.assertEqual(QS.UpdateRequest.HEADER, "X-Yasakani-Token")

    def test_shutdown(self):
        own = self.own(self.port)
        self.assertEqual(self.post("/webchat/shutdown", own)[0], 403)                                       # no token
        self.assertEqual(self.post("/webchat/shutdown", dict(own, **{self.H: "0" * 64}))[0], 403)           # a wrong token
        self.assertEqual(self.post("/webchat/shutdown", dict(own, **{self.H: self.tok[:-1]}))[0], 403)
        self.assertEqual(self.post("/webchat/shutdown", {self.H: self.tok})[0], 403)                        # no Origin
        self.assertEqual(too_long(self.port, "/webchat/shutdown", dict(own, **{self.H: self.tok}), 4097), 413)
        self.assertEqual(request(self.port, "GET", "/webchat/shutdown", dict(own, **{self.H: self.tok}))[0], 404)
        self.assertFalse(self.control.event.is_set())
        self.assertEqual(get_json(self.port, "/webchat/status")["state"], "loading")
        code, body = self.post("/webchat/shutdown", dict(own, **{self.H: self.tok}))
        self.assertEqual((code, body), (202, {"status": "stopping"}))
        self.assertTrue(self.control.event.wait(5))                                                         # the stop is signalled
        self.assertEqual((self.control.action, self.control.model), ("shutdown", None))
        self.assertEqual(QS.wait_for_stop(SimpleNamespace(poll=lambda: None), None, poll=0.05, control=self.control), "shutdown")
        self.assertEqual(get_json(self.port, "/webchat/status")["state"], "stopping")
        self.assertIsNone(get_json(self.port, "/webchat/config")["control_token"])
        self.assertEqual(self.post("/webchat/shutdown", dict(own, **{self.H: self.tok}))[0], 403)           # the same token again
        self.assertEqual(self.post("/webchat/switch_model", dict(own, **{self.H: self.tok}), b'{"model": "speed"}')[0], 403)

    def test_switch_model(self):
        h = self.own(self.port, **{self.H: self.tok})
        for body in (b'{"model": "turbo"}', b'{"model": "../speed"}', b'{"model": ["speed"]}', b'{"model": null}', b'{}',
                     b'["speed"]', b'"speed"', b'not json', b'', b'\xff\xfe', b'[' * 4000):
            with self.subTest(body=body[:24]):
                self.assertEqual(self.post("/webchat/switch_model", h, body)[0], 400)                       # not a model name
        self.assertEqual(self.post("/webchat/switch_model", h, b'{"model": "max"}')[0], 409)                # not downloaded here
        self.assertEqual(self.post("/webchat/switch_model", h, b'{"model": "quality"}')[0], 409)            # in use already
        self.assertEqual(too_long(self.port, "/webchat/switch_model", h, 4097), 413)
        self.assertEqual(self.control.token, self.tok)                                                      # still valid
        self.assertFalse(self.control.event.is_set())
        own = self.own(self.port)
        self.assertEqual(self.post("/webchat/switch_model", own, b'{"model": "speed"}')[0], 403)            # no token
        self.assertEqual(self.post("/webchat/switch_model", dict(own, **{self.H: "f" * 64}), b'{"model": "speed"}')[0], 403)
        self.assertEqual(self.post("/webchat/switch_model", {self.H: self.tok}, b'{"model": "speed"}')[0], 403)   # no Origin
        self.assertEqual(self.session.state(), "loading")
        code, body = self.post("/webchat/switch_model", h, b'{"model": "speed"}')
        self.assertEqual((code, body), (202, {"status": "switching", "model": "speed"}))
        self.assertTrue(self.control.event.wait(5))
        self.assertEqual((self.control.action, self.control.model), ("switch", "speed"))                    # handed to the launcher
        self.assertEqual(QS.wait_for_stop(SimpleNamespace(poll=lambda: None), None, poll=0.05, control=self.control), "switch")
        self.assertEqual(get_json(self.port, "/webchat/status")["state"], "switching")
        self.assertEqual(self.post("/webchat/switch_model", h, b'{"model": "speed"}')[0], 403)              # replay
        self.assertEqual(self.post("/webchat/shutdown", h)[0], 403)

    def test_every_model_name_of_the_launcher_and_no_other(self):
        self.assertEqual(sorted(QS.MODELS), ["inkling2", "max", "quality", "speed"])
        self.assertEqual(QS.MODEL_ORDER, ("quality", "speed", "max", "inkling2"))              # 0.3.2: Inkling-2 Mini is the 4th

    def test_switch_to_and_from_inkling2_is_accepted(self):
        for current, other in (("quality", "inkling2"), ("inkling2", "quality")):
            control = QS.ControlRequest()
            port = self.serve(1, None, control, QS.WebSession(current, None, 1, installed=("quality", "inkling2")))
            h = self.own(port, **{self.H: control.token})
            self.assertEqual(request(port, "POST", "/webchat/switch_model", h, json.dumps({"model": current}).encode())[0], 409)
            self.assertEqual(request(port, "POST", "/webchat/switch_model", h, b'{"model": "max"}')[0], 409)   # not downloaded
            code, body, _ = request(port, "POST", "/webchat/switch_model", h, json.dumps({"model": other}).encode())
            self.assertEqual((code, json.loads(body)), (202, {"status": "switching", "model": other}))
            self.assertEqual((control.action, control.model), ("switch", other))

    def test_no_switching_with_a_model_folder_of_ones_own(self):
        control = QS.ControlRequest()
        port = self.serve(1, None, control, QS.WebSession("custom", None, 1, installed=()))
        h = self.own(port, **{self.H: control.token})
        self.assertEqual(request(port, "POST", "/webchat/switch_model", h, b'{"model": "speed"}')[0], 409)
        self.assertEqual(request(port, "POST", "/webchat/switch_model", h, b'{"model": "custom"}')[0], 400)
        self.assertEqual(json.loads(request(port, "GET", "/webchat/config")[1])["models_installed"], [])
        self.assertEqual(request(port, "POST", "/webchat/shutdown", h)[0], 202)                            # quitting still works

    def test_closed_control_takes_nothing(self):
        self.control.close()
        h = self.own(self.port, **{self.H: self.tok})
        self.assertEqual(self.post("/webchat/shutdown", h)[0], 403)
        self.assertEqual(self.post("/webchat/switch_model", h, b'{"model": "speed"}')[0], 403)
        self.assertIsNone(get_json(self.port, "/webchat/config")["control_token"])
        self.assertFalse(self.control.event.is_set())

    def test_closed_update_takes_nothing(self):
        """The update button ends with the page's wait, like the other buttons (UpdateRequest.close)."""
        update = QS.UpdateRequest({"version": "9.9.9", "page": QS.REPO_URL + "/releases/tag/v9.9.9"})
        port = self.serve(1, update, None, None)
        tok = get_json(port, "/webchat/config")["update_token"]
        self.assertEqual(tok, update.token)
        self.assertTrue(update.close())                                                # it was open
        self.assertFalse(update.close())
        c = get_json(port, "/webchat/config")
        self.assertEqual((c["update"]["version"], c["update_token"]), ("9.9.9", None))      # the offer stays, the button is off
        self.assertEqual(request(port, "POST", "/webchat/update", self.own(port, **{QS.UpdateRequest.HEADER: tok}))[0], 403)
        self.assertFalse(update.event.is_set())
        used = QS.UpdateRequest({"version": "9.9.9", "page": ""})
        self.assertTrue(used.take(used.token))
        self.assertFalse(used.close())                                                 # used before: the caller can tell
        self.assertTrue(QS.ControlRequest().close())
        taken = QS.ControlRequest()
        self.assertTrue(taken.take(taken.token, "shutdown"))
        self.assertFalse(taken.close())

    def test_token_is_compared_in_constant_time(self):
        import ast
        tree = ast.parse(Path(QS.__file__).read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ControlRequest")
        take = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "take")
        self.assertIn("hmac.compare_digest", ast.unparse(take))
        self.assertNotIn("==", ast.unparse(take))

    def test_update_button_is_unchanged(self):
        update = QS.UpdateRequest({"version": "9.9.9", "page": QS.REPO_URL + "/releases/tag/v9.9.9"})
        control = QS.ControlRequest()
        port = self.serve(1, update, control, QS.WebSession("quality", None, 1, installed=("quality", "speed")))
        c = json.loads(request(port, "GET", "/webchat/config")[1])
        self.assertEqual((c["update"]["version"], c["update_token"]), ("9.9.9", update.token))
        own = self.own(port)
        U = QS.UpdateRequest.HEADER
        self.assertEqual(request(port, "POST", "/webchat/update", own)[0], 403)
        self.assertEqual(request(port, "POST", "/webchat/update", dict(own, **{self.H: c["update_token"]}))[0], 403)  # other header
        self.assertEqual(request(port, "POST", "/webchat/update", dict(own, **{U: c["control_token"]}))[0], 403)      # other token
        self.assertEqual(request(port, "POST", "/webchat/shutdown", dict(own, **{self.H: c["update_token"]}))[0], 403)
        code, body, _ = request(port, "POST", "/webchat/update", dict(own, **{U: c["update_token"]}))
        self.assertEqual((code, json.loads(body)), (202, {"status": "updating", "version": "9.9.9"}))
        self.assertTrue(update.event.wait(5))
        self.assertEqual(QS.wait_for_stop(SimpleNamespace(poll=lambda: None), update, poll=0.05, control=control), "update")
        self.assertEqual(request(port, "POST", "/webchat/update", dict(own, **{U: c["update_token"]}))[0], 403)
        self.assertEqual(control.token, c["control_token"])                                                # not used up by it


# -------------------------------------------------------------------------------------- options, direct start and the menu
class OptionTests(unittest.TestCase):
    def setUp(self):
        if QS is None:
            self.skipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")

    def test_default_action_and_menu_flag(self):
        self.assertEqual(QS.parse_args([]).action, "auto")
        self.assertEqual(QS.parse_args(["--menu"]).action, "menu")
        self.assertEqual(QS.parse_args(["--action", "menu"]).action, "menu")
        self.assertEqual(QS.parse_args(["--action", "web", "--menu"]).action, "web")
        self.assertEqual(QS.parse_args(["--web-test"]).action, "web")
        self.assertEqual(QS.parse_args(["--web-test", "--menu"]).action, "web")
        for old in ("menu", "bench", "compare", "web", "chat", "serve", "none", "update", "rollback"):
            self.assertEqual(QS.parse_args(["--action", old]).action, old)         # what an older launcher passes after an update
        a = QS.parse_args(["--action", "web", "--no-browser", "--web-port", "28766", "--lang", "ja", "--yes"])
        self.assertEqual((a.action, a.no_browser, a.web_port, a.lang, a.yes), ("web", True, 28766, "ja", True))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            QS.parse_args(["--action", "direct"])

    def test_direct_start_decision(self):
        d = QS.direct_start
        self.assertTrue(d("auto", "quality", {"quality": True}))                   # checked in an earlier run -> the page
        self.assertTrue(d("auto", "speed", {"quality": True, "speed": True}, None))
        self.assertFalse(d("auto", None, {}))                                      # the first run -> as the menu
        self.assertFalse(d("auto", "quality", {}))
        self.assertFalse(d("auto", "max", {"quality": True}))                      # another model, never checked
        self.assertFalse(d("auto", "quality", {"quality": False}))
        self.assertFalse(d("auto", "quality", {"quality": True}, "D:/my_model"))   # --model-dir -> as the menu
        self.assertFalse(d("auto", "turbo", {"turbo": True}))                      # a name this launcher does not know
        self.assertFalse(d("auto", "quality", None))
        for action in ("menu", "web", "bench", "compare", "chat", "serve", "none", "update", "rollback"):
            self.assertFalse(d(action, "quality", {"quality": True}), action)      # only auto decides this way

    def test_main_uses_the_decision_before_the_model_question(self):
        """main() decides from the state as it was saved before this run: before step 6 can mark the model as checked."""
        src = Path(QS.__file__).read_text(encoding="utf-8")
        body = src[src.index("def main("):]
        decide, question, checks = body.index("direct_start("), body.index('ask(ui.t("pick"'), body.index("rc = run_checks()")
        self.assertLess(decide, question)
        self.assertLess(question, checks)
        self.assertIn('a.action = "web" if direct else "menu"', body)

    def test_installed_models(self):
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qsweb_m_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        for m in ("quality", "speed", "max"):
            (tmp / "models" / m).mkdir(parents=True)
        (tmp / "models" / "speed" / "yasakani.json").write_text("{}")
        (tmp / "models" / "max" / "yasakani.json").write_text("{}")
        f = QS.installed_models
        self.assertEqual(f(str(tmp), {"quality": True, "speed": True, "max": True}, True), ["speed", "max"])    # the folder is gone
        self.assertEqual(f(str(tmp), {"speed": True}, True), ["speed"])                                         # never downloaded
        self.assertEqual(f(str(tmp), {"speed": True, "max": True}, False), ["speed"])                           # under 24 GB
        self.assertEqual(f(str(tmp), {}, True), [])
        self.assertEqual(f(str(tmp), None, True), [])
        (tmp / "models" / "quality" / "yasakani.json").write_text("{}")
        self.assertEqual(f(str(tmp), {"quality": True, "speed": True, "max": True}, True), ["quality", "speed", "max"])
        (tmp / "models" / "inkling2").mkdir(parents=True)                              # 0.3.2: Inkling-2 Mini, 24 GB or more
        (tmp / "models" / "inkling2" / "yasakani.json").write_text("{}")
        all4 = {"quality": True, "speed": True, "max": True, "inkling2": True}
        self.assertEqual(f(str(tmp), all4, True), ["quality", "speed", "max", "inkling2"])
        self.assertEqual(f(str(tmp), all4, False), ["speed"])
        self.assertEqual(f(str(tmp), {"speed": True}, True), ["speed"])

    def test_model_question_has_inkling2_as_4(self):
        self.assertEqual(QS.MODEL_KEYS, {"1": "quality", "2": "speed", "3": "max", "4": "inkling2"})
        # 0.3.5: text and images, the folder with the image file (0.3.2 to 0.3.4: text only, 15.3 GB; test_quickstart_ink2_vision)
        for lang, line in (("ja", "  4) Inkling-2 Mini〈試験的〉 文章と画像。依頼は 1 件ずつ。GPU 24 GB 以上。ダウンロード約 15.9 GB\n"),
                           ("en", "  4) Inkling-2 Mini (experimental) text and images, one request at a time, a GPU with 24 GB or more;\n"
                                  "                                  about 15.9 GB download\n")):
            text = QS.Ui(lang).t("pick", d="4")
            self.assertIn(line, text)
            self.assertTrue(text.endswith("[4]: "))
        self.assertEqual(QS.Ui("en").t("m_inkling2"), "Inkling-2 Mini model (experimental)")
        self.assertEqual(QS.Ui("ja").t("m_inkling2"), "Inkling-2 Mini（試験的）")
        self.assertEqual(QS.parse_args(["--model", "inkling2"]).model, "inkling2")

    def test_download_command_pins_the_revision_of_inkling2_only(self):
        e = QS.MODELS["inkling2"]
        self.assertEqual((e["repo"], e["revision"], e["need_gb"], e["get_gb"]),
                         ("yasakani-dev/Inkling-2-Mini-YASAKANI-quality", "e186e163e6d140f767a0fd1f1d31ed93e1a3e1a8", 17, "15.9"))
        self.assertGreaterEqual(e["need_gb"], 15.93 + 1)                               # the files and the margin of the others
        for name in ("quality", "speed", "max"):
            self.assertEqual(QS.download_command("py", name, "D:/m"), ["py", "-c", QS.DOWNLOAD_CODE, QS.MODELS[name]["repo"], "D:/m"])
            self.assertNotIn("revision", QS.MODELS[name])
        self.assertEqual(QS.download_command("py", "inkling2", "D:/m"),
                         ["py", "-c", QS.DOWNLOAD_CODE, e["repo"], "D:/m", e["revision"]])

    def test_download_code_passes_the_revision_to_snapshot_download(self):
        """The code that runs in the private environment, with a stand-in huggingface_hub."""
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qsweb_dl_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        (tmp / "huggingface_hub.py").write_text(
            "import json, sys\n"
            "def snapshot_download(**kw):\n"
            "    open(sys.argv[2] + '.json', 'w').write(json.dumps(kw))\n", encoding="utf-8")
        for model, want in (("quality", {"repo_id": QS.MODELS["quality"]["repo"], "local_dir": str(tmp / "q")}),
                            ("inkling2", {"repo_id": QS.MODELS["inkling2"]["repo"], "local_dir": str(tmp / "i"),
                                          "revision": QS.MODELS["inkling2"]["revision"]})):
            cmd = QS.download_command(sys.executable, model, want["local_dir"])
            r = subprocess.run(cmd, cwd=str(tmp), capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(Path(want["local_dir"] + ".json").read_text()), want)

    def test_fetch_model_tries_again_without_xet(self):
        calls = []

        def call(cmd, env=None):
            calls.append((cmd, dict(env)))
            return 1 if len(calls) == 1 else 0

        out = io.StringIO()
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qsweb_f_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        with contextlib.redirect_stdout(out):
            rc = QS.fetch_model(QS.Ui("en"), "py", {"A": "1"}, "inkling2", str(tmp / "m"), call=call, environ={})
        self.assertEqual(rc, 0)
        self.assertEqual([c[0] for c in calls], [QS.download_command("py", "inkling2", str(tmp / "m"))] * 2)
        self.assertEqual((calls[0][1], calls[1][1]), ({"A": "1"}, {"A": "1", "HF_HUB_DISABLE_XET": "1"}))
        self.assertIn("Downloading yasakani-dev/Inkling-2-Mini-YASAKANI-quality from Hugging Face (about 15.9 GB", out.getvalue())
        self.assertIn("Trying once more", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as o2:
            QS.fetch_model(QS.Ui("en"), "py", {}, "quality", str(tmp / "q"), call=lambda cmd, env=None: 0, environ={})
        self.assertIn("(about 15 GB", o2.getvalue())                                    # the Qwen models as before

    def test_test_override_links_the_files_and_asks_no_one(self):
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qsweb_o_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        src = tmp / "src"
        (src / ".cache").mkdir(parents=True)
        (src / "yasakani.json").write_text('{"architecture": "inkling2"}', encoding="utf-8")
        (src / "w.yskn").write_bytes(b"\0" * 100)
        (src / ".cache" / "x").write_text("x")
        dst = tmp / "home" / "models" / "inkling2"
        dst.mkdir(parents=True)
        (dst / "w.yskn").write_bytes(b"old")                                           # a file of a download before is replaced
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = QS.fetch_model(QS.Ui("en"), "py", {}, "inkling2", str(dst), call=lambda *a, **k: self.fail("downloaded"),
                                environ={"YASAKANI_QS_TEST_MODEL_SRC_INKLING2": str(src)})
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(p.name for p in dst.iterdir()), ["w.yskn", "yasakani.json"])
        self.assertEqual((dst / "w.yskn").read_bytes(), b"\0" * 100)
        self.assertTrue(QS.is_inkling2_dir(str(dst)))
        self.assertIn("Test only: the model files come from %s (YASAKANI_QS_TEST_MODEL_SRC_INKLING2" % src, out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):                                # another model's variable: a normal download
            self.assertEqual(QS.fetch_model(QS.Ui("en"), "py", {}, "quality", str(tmp / "q"), call=lambda *a, **k: 7,
                                            environ={"YASAKANI_QS_TEST_MODEL_SRC_INKLING2": str(src)}), 7)
            self.assertEqual(QS.fetch_model(QS.Ui("en"), "py", {}, "inkling2", str(tmp / "e"), call=lambda *a, **k: 0,
                                            environ={"YASAKANI_QS_TEST_MODEL_SRC_INKLING2": str(tmp / "nothing")}), 1)

    def test_is_inkling2_dir(self):
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qsweb_i_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        self.assertFalse(QS.is_inkling2_dir(str(tmp)))
        for text, want in (('{"architecture": "inkling2"}', True), ("{}", False), ("[1]", False), ("not json", False)):
            (tmp / "yasakani.json").write_text(text, encoding="utf-8")
            self.assertEqual(QS.is_inkling2_dir(str(tmp)), want, text)

    def test_page_names_of_inkling2(self):
        self.assertEqual(QS.WebSession("inkling2").model_id, "inkling-2-mini-quality")      # until the engine tells its own
        self.assertEqual(QS.WebSession("quality").model_id, "qwen3.8-27b")
        self.assertEqual(QS.WebSession("custom").model_id, "qwen3.8-27b")

    def test_menu_has_the_browser_first_and_enter_picks_it(self):
        for lang, word, mark in (("ja", "ブラウザーで開く", "〈おすすめ〉"), ("en", "open in the browser", "(recommended)")):
            text = QS.menu_prompt(QS.Ui(lang))
            items = [ln.strip() for ln in text.splitlines() if re.match(r"\s*\d\)", ln)]
            self.assertEqual([i[0] for i in items], ["1", "2", "3", "4", "5", "6", "7"], lang)
            self.assertIn(word, items[0])
            self.assertIn(mark, items[0])
            self.assertTrue(text.rstrip().endswith("[1]:"), lang)                  # the prompt shows what Enter does
            self.assertNotIn("{", text)
        self.assertEqual(QS.MENU_DEFAULT, "1")
        self.assertEqual(QS.MENU, {"1": "web", "2": "bench", "3": "compare", "4": "chat", "5": "serve", "7": "update"})
        asked = []
        orig = QS.ask

        def enter(prompt, default="", eof=None):
            asked.append((prompt, default, eof))
            return default                                                         # Enter alone

        try:
            QS.ask = enter
            self.assertEqual(QS.menu_choice(QS.Ui("en")), "web")
            for typed, want in (("1", "web"), ("2", "bench"), ("3", "compare"), ("4", "chat"), ("5", "serve"), ("6", None),
                                ("7", "update"), ("q", None), ("0", None)):
                QS.ask = lambda prompt, default="", eof=None, typed=typed: typed
                self.assertEqual(QS.menu_choice(QS.Ui("ja")), want, typed)
        finally:
            QS.ask = orig
        self.assertEqual(asked[0][1:], ("1", "6"))                                 # Enter: the page.  No keyboard: quit
        self.assertEqual(asked[0][0], QS.menu_prompt(QS.Ui("en")))

    def test_enter_at_the_menu_picks_the_browser(self):
        for typed in ("", "  ", "\t", "1", " 1 "):
            with mock.patch("builtins.input", return_value=typed) as inp:
                self.assertEqual(QS.menu_choice(QS.Ui("en")), "web", repr(typed))
            self.assertEqual(inp.call_args[0][0], QS.menu_prompt(QS.Ui("en")))

    def test_end_of_the_input_at_the_menu_is_quit(self):
        """No keyboard (the input is closed or a file that has ended): the menu quits.  With Enter = the page it would start the
        page and the server again and again."""
        with mock.patch("builtins.input", side_effect=EOFError) as inp:
            self.assertIsNone(QS.menu_choice(QS.Ui("en")))
            self.assertIsNone(QS.menu_choice(QS.Ui("ja")))
            self.assertEqual(inp.call_count, 2)                                    # asked once each: no second try
            self.assertEqual(QS.ask("model? ", "2"), "2")                          # the other questions: the default, as before
            self.assertEqual(QS.ask("update? ", "n"), "n")
            self.assertEqual(QS.ask("x? "), "")
            self.assertEqual(QS.ask("x? ", "1", eof="6"), "6")
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(QS.ask("x? ", "1", eof="6"), "1")                     # Enter is not the end of the input
        with mock.patch("builtins.input", return_value=" 3 "):
            self.assertEqual(QS.ask("x? ", "1", eof="6"), "3")

    def test_texts_that_name_menu_numbers_stay_true(self):
        for lang in ("ja", "en"):
            ui = QS.Ui(lang)
            self.assertIn("7", ui.t("upd_later"))
            self.assertTrue(ui.t("menu_update").lstrip().startswith("7)"))
            serve = next(ln for ln in QS.menu_prompt(ui).splitlines() if "127.0.0.1:8765/v1" in ln)
            self.assertTrue(serve.lstrip().startswith("5)"))
        self.assertEqual(QS.MENU["7"], "update")
        self.assertIn("新しい版を確認する", QS.T["menu_update"]["ja"])
        self.assertIn("check for a new version", QS.T["menu_update"]["en"])

    def test_every_text_has_both_languages(self):
        kw = dict(u="http://127.0.0.1:28766/", m="quality model", s="42", o="0.2.0", n="0.2.1", d="1", f="yasakani_env.cmd")
        for key in ("menu", "direct", "web_boot", "web_ready", "web_ready_manual", "web_again", "web_moved", "web_loaded",
                    "web_quit", "web_switch", "upd_page", "web_fail", "web_end", "web_died", "switch_back", "again",
                    "again_direct"):
            for lang in ("ja", "en"):
                with self.subTest(key=key, lang=lang):
                    text = QS.T[key][lang].format(**kw)
                    self.assertTrue(text.strip())
                    self.assertNotIn("{", text)
        for key, both in QS.T.items():
            self.assertEqual(sorted(both), ["en", "ja"], key)
        for key in ("direct", "again_direct"):                  # where the menu is: said in both, with the place to type it
            self.assertIn("START.cmd --menu", QS.T[key]["ja"])
            self.assertIn("START.cmd --menu", QS.T[key]["en"])
            self.assertIn("このフォルダーで開いたコマンドプロンプト", QS.T[key]["ja"])
            self.assertIn("a command prompt opened in this folder", QS.T[key]["en"])
        # the direct start can come with --model for another model that was checked before: nothing about "last time"
        self.assertNotIn("前回", QS.T["direct"]["ja"])
        self.assertNotIn("last time", QS.T["direct"]["en"])
        for key in ("switch_back",):
            for lang in ("ja", "en"):
                self.assertIn("{m}", QS.T[key][lang])
                self.assertIn("{o}", QS.T[key][lang])

    def test_exit_codes_of_the_page_do_not_collide(self):
        codes = [QS.UPDATE_RC, QS.SWITCH_RC, QS.QUIT_RC]
        self.assertEqual(len(set(codes)), 3)
        for c in codes:
            self.assertNotIn(c, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 130))


# --------------------------------------------------------------------------------------- small helpers of the page's server
class HelperTests(Base):
    def test_wait_gpu_free(self):
        now, slept = [0.0], []

        def sleep(s):
            slept.append(s)
            now[0] += s

        readings = [20000, 15000, 2300]
        self.assertTrue(QS.wait_gpu_free(lambda: readings.pop(0), 1500, seconds=15, margin=1024, sleep=sleep, clock=lambda: now[0]))
        self.assertEqual(len(slept), 2)
        now[0], slept[:] = 0.0, []
        self.assertFalse(QS.wait_gpu_free(lambda: 20000, 1500, seconds=3, margin=1024, sleep=sleep, clock=lambda: now[0]))
        self.assertLessEqual(sum(slept), 3.5)                                      # not longer than the limit
        self.assertFalse(QS.wait_gpu_free(lambda: None, 1500, sleep=sleep, clock=lambda: now[0]))      # nvidia-smi gave nothing
        self.assertFalse(QS.wait_gpu_free(lambda: self.fail("not asked"), None, sleep=sleep, clock=lambda: now[0]))

    def test_stay_for_the_page(self):
        """After Enter or the end of the server the page's server stays END_GRACE seconds; after a server that ended by itself
        it first waits (at most FAIL_GRACE seconds) until a page has asked."""
        self.assertEqual((QS.END_GRACE, QS.FAIL_GRACE), (2.0, 5.0))
        now, slept = [10.0], []

        def sleep(s):
            slept.append(round(s, 3))
            now[0] += s

        stopping = QS.WebSession("quality", SimpleNamespace(poll=lambda: 0), 1)
        stopping.leave("stopping")
        QS.stay_for_the_page(stopping, 10.0, clock=lambda: now[0], sleep=sleep)
        self.assertEqual(slept, [2.0])
        now[0], slept[:] = 10.5, []                         # stopping the server took half a second of it
        QS.stay_for_the_page(stopping, 10.0, clock=lambda: now[0], sleep=sleep)
        self.assertEqual(slept, [1.5])
        now[0], slept[:] = 40.0, []                         # it took longer than that: no more waiting
        QS.stay_for_the_page(stopping, 10.0, clock=lambda: now[0], sleep=sleep)
        self.assertEqual(slept, [])
        failed = QS.WebSession("quality", SimpleNamespace(poll=lambda: 3), 1)
        self.assertEqual(failed.state(), "failed")
        waits = []
        failed.told_failed = SimpleNamespace(wait=lambda seconds: waits.append(seconds) or now.__setitem__(0, now[0] + 0.4))
        now[0], slept[:] = 10.0, []
        QS.stay_for_the_page(failed, 10.0, clock=lambda: now[0], sleep=sleep)
        self.assertEqual((waits, slept), ([5.0], [1.6]))    # a page asked after 0.4 s: it stays until the 2 s are over

        def interrupted(s):
            raise KeyboardInterrupt

        QS.stay_for_the_page(stopping, now[0], clock=lambda: now[0], sleep=interrupted)        # Ctrl+C ends the wait, no error

    def test_stop_process(self):
        calls = []

        class Proc:
            def __init__(self, stubborn):
                self.stubborn, self.code = stubborn, None

            def poll(self):
                return self.code

            def terminate(self):
                calls.append("terminate")
                if not self.stubborn:
                    self.code = 1

            def kill(self):
                calls.append("kill")
                self.code = 1

            def wait(self, timeout=None):
                calls.append("wait %s" % timeout)
                if self.code is None:
                    raise subprocess.TimeoutExpired("x", timeout)
                return self.code

        QS.stop_process(Proc(False))
        self.assertEqual(calls, ["terminate", "wait 30"])
        calls[:] = []
        QS.stop_process(Proc(True), seconds=2)
        self.assertEqual(calls, ["terminate", "wait 2", "kill", "wait 5"])
        calls[:] = []
        done = Proc(False)
        done.code = 0
        QS.stop_process(done)
        self.assertEqual(calls, [])

    def test_open_page_server_waits_for_a_busy_port(self):
        port = free_port()
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        made = []

        def handler(p):
            made.append(p)
            return QS.make_web_handler(self.page, {"version": QS.QS_VERSION}, 1, p)

        try:
            self.assertIsNone(QS.open_page_server(handler, port))                  # busy, and no time to wait
            threading.Timer(0.5, holder.close).start()
            httpd = QS.open_page_server(handler, port, seconds=10)
        finally:
            holder.close()
        self.assertIsNotNone(httpd)
        self.addCleanup(httpd.server_close)
        self.assertEqual(httpd.server_address[1], port)
        self.assertEqual(set(made), {port})

    def test_pick_port_skips_a_port_that_answers(self):
        """On Windows a bind to 127.0.0.1 passes while another program listens on the port on every address (0.0.0.0); the
        server could then not listen there.  The listener of this test is on the loopback address (so no firewall question),
        and the bind test is made to pass as it does in that case."""
        port = free_port()
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", port))
        holder.listen(8)
        self.addCleanup(holder.close)
        self.assertTrue(QS.port_answers(port))
        asked = []
        with mock.patch.object(QS, "port_is_free", lambda p: asked.append(p) or True):
            got = QS.pick_port(port)
        self.assertIn(got, range(port + 10, port + 60))
        self.assertEqual(asked[0], port)                             # the bind test passed for it, and it was still not taken

    def test_a_port_without_a_listener_does_not_answer(self):
        port = free_port()
        t0 = time.monotonic()
        self.assertFalse(QS.port_answers(port))
        self.assertLess(time.monotonic() - t0, 1.5)                  # bounded: a refusal alone takes about two seconds on Windows
        self.assertEqual(QS.pick_port(port), port)
        self.assertIn(QS.pick_port(port, avoid=(port,)), range(port + 10, port + 60))

    def test_closed_server_ends_the_connections_it_accepted(self):
        """A browser keeps its connections open.  After the server was closed, the thread of such a connection must not go on
        answering for the server that is gone (after a model switch the page must reach the next one)."""
        session = QS.WebSession("quality", SimpleNamespace(poll=lambda: None), 1)
        port = free_port()
        httpd = QS.LocalServer(("127.0.0.1", port), QS.make_web_handler(self.page, {"version": QS.QS_VERSION}, 1, port,
                                                                        None, None, session))
        threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
        kept = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.addCleanup(kept.close)
        h = {"Host": "127.0.0.1:%d" % port}
        kept.request("GET", "/webchat/status", headers=h)
        self.assertEqual(json.loads(kept.getresponse().read())["state"], "loading")
        kept.request("GET", "/webchat/status", headers=h)                          # the same connection again: kept open
        self.assertEqual(kept.getresponse().status, 200)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            httpd.shutdown()
            httpd.server_close()
            httpd.close_connections()
            with self.assertRaises((OSError, http.client.HTTPException)):
                kept.request("GET", "/webchat/status", headers=h)
                kept.getresponse().read()
            time.sleep(0.3)
        self.assertEqual(err.getvalue(), "")                                       # and no traceback in the console for it
        self.assertTrue(wait_until(lambda: not httpd._conns, 5))


# ------------------------------------------------------------------- run_web with a stand-in for `yasakani serve` (a process)
class RunWebBase(Base):
    def setUp(self):
        super().setUp()
        self.children = Children()
        p = mock.patch.object(QS.subprocess, "Popen", self.children)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.children.stop_all)
        self.log = self.tmp / "fake.log"
        self.env = dict(os.environ, YSKN_FAKE_LOG=str(self.log), YSKN_FAKE_LOAD="1.0", PYTHONDONTWRITEBYTECODE="1")
        for name in ("ENGINE_PORT", "WEB_PORT"):
            q = mock.patch.object(QS, name, free_port())
            q.start()
            self.addCleanup(q.stop)
        self.opened = []
        w = mock.patch.object(QS.webbrowser, "open", lambda url: self.opened.append(url) or True)
        w.start()
        self.addCleanup(w.stop)
        self.out = io.StringIO()
        r = contextlib.redirect_stdout(self.out)
        r.__enter__()
        self.addCleanup(r.__exit__, None, None, None)
        self.vpy = str(self.tmp / "venv" / "Scripts" / "python.exe")              # does not exist: see Children
        self.gpu = QS.GpuWatch(dict(index=0, uuid="GPU-a", name="NVIDIA GeForce RTX 5090", total=32607, used=1200, driver="616.92"),
                               query=lambda: [dict(index=0, uuid="GPU-a", name="NVIDIA GeForce RTX 5090", total=32607, used=1200,
                                                   driver="616.92")])

    def args(self, **kw):
        a = SimpleNamespace(model_dir=None, web_test=False, no_browser=False, web_port=free_port())
        a.__dict__.update(kw)
        return a

    def console(self, typed=""):
        """Stands in for the keys of the console window (msvcrt) while the launcher runs: ``typed`` waits there already.
        -> the stand-in: put more keys into its ``keys``; ``taken`` are the keys the launcher has read."""
        con = SimpleNamespace(keys=list(typed), taken=[])
        con.kbhit = lambda: bool(con.keys)

        def getwch():
            c = con.keys.pop(0)
            con.taken.append(c)
            return c

        con.getwch = getwch
        m = mock.patch.dict(sys.modules, {"msvcrt": con})
        m.start()
        self.addCleanup(m.stop)
        s = mock.patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True))
        s.start()
        self.addCleanup(s.stop)
        return con

    def start(self, a, model="quality", installed=("quality", "speed"), update_rel=None, reopen=None, env=None):
        """run_web in a thread.  -> (thread, box); box["rc"] is its result."""
        box = {}

        def run():
            try:
                box["rc"] = QS.run_web(self.ui, a, self.vpy, env or self.env, str(self.tmp / "models" / model),
                                       str(self.tmp / "engine" / "yskn.dll"), model, update_rel, self.gpu, list(installed), reopen)
            except BaseException as e:  # noqa: BLE001 - shown by the test
                box["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, box

    def page_port(self, a, box):
        self.assertTrue(wait_until(lambda: getattr(a, "web_port_used", None) or box, 20), box)
        self.assertNotIn("error", box)
        port = a.web_port_used
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status"), 20))
        return port

    def served(self):
        if not self.log.is_file():
            return []
        return [json.loads(ln) for ln in self.log.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def finish(self, t, box, want):
        t.join(40)
        self.assertFalse(t.is_alive(), "run_web did not return")
        self.assertNotIn("error", box)
        self.assertEqual(box.get("rc"), want)
        self.assertTrue(wait_until(self.children.all_ended, 10), "a server process is left")


class RunWebTests(RunWebBase):
    def test_page_opens_at_once_then_chat_then_quit_button(self):
        a = self.args()
        t0 = time.monotonic()
        t, box = self.start(a)
        port = self.page_port(a, box)
        first = get_json(port, "/webchat/status")
        self.assertEqual((first["state"], first["model"]), ("loading", "quality"))    # the page is there before the model
        self.assertLess(time.monotonic() - t0, 20)
        self.assertEqual(self.opened, ["http://127.0.0.1:%d/" % port])                # and the browser was opened at once
        self.assertEqual(request(port, "GET", "/")[0], 200)
        c = get_json(port, "/webchat/config")
        eport = int(re.fullmatch(r"http://127\.0\.0\.1:(\d+)/v1", c["api_base"]).group(1))
        self.assertTrue(28100 <= eport <= 28999 and eport != port)
        self.assertEqual(c["models_installed"], [{"name": "quality", "current": True}, {"name": "speed", "current": False}])
        self.assertEqual(c["gpu"]["used_mib"], 1200)
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        c = get_json(port, "/webchat/config")
        self.assertEqual((c["model_id"], c["max_context_tokens"]), ("qwen3.8-27b", 262144))
        code, body, _ = request(port, "POST", "/v1/chat/completions", self.own(port),
                                json.dumps({"messages": [{"role": "user", "content": "count"}], "stream": True}).encode())
        self.assertEqual(code, 200)                                                    # the conversation works without a reload
        self.assertIn(b'"1 "', body)
        self.assertIn(b"[DONE]", body)
        self.assertTrue(t.is_alive())
        code, body, _ = request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]}))
        self.assertEqual((code, json.loads(body)), (202, {"status": "stopping"}))
        self.finish(t, box, QS.QUIT_RC)
        self.assertTrue(wait_until(lambda: closed(port), 10))                  # the page's server is closed
        self.assertTrue(wait_until(lambda: closed(eport), 10))                 # and nothing of the engine is left
        self.assertEqual([x["cmd"] for x in self.served()], ["serve"])
        out = self.out.getvalue()
        self.assertIn("The model is loaded", out)
        self.assertIn("The quit button of the page was pressed", out)
        self.assertLess(out.index("Opening the page in your browser"), out.index("The model is loaded"))

    def test_no_browser_prints_the_address(self):
        a = self.args(no_browser=True)
        t, box = self.start(a)
        port = self.page_port(a, box)
        self.assertEqual(self.opened, [])
        self.assertIn("Open this address in your browser: http://127.0.0.1:%d/" % port, self.out.getvalue())
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
        self.finish(t, box, QS.QUIT_RC)

    def test_web_test_with_inkling2(self):
        a = self.args(web_test=True)
        t, box = self.start(a, model="inkling2", installed=("quality", "inkling2"),
                            env=dict(self.env, YSKN_FAKE_MODEL_ID="inkling-2-mini-quality"))
        self.finish(t, box, 0)
        out = self.out.getvalue()
        self.assertIn("browser chat self-test: PASS", out)
        self.assertNotIn("FAIL", out)
        self.assertEqual([(x["cmd"], os.path.basename(x["model_dir"])) for x in self.served()], [("serve", "inkling2")])

    def test_inkling2_page_then_switch_to_quality(self):
        a = self.args(no_browser=True)
        t, box = self.start(a, model="inkling2", installed=("quality", "inkling2"),
                            env=dict(self.env, YSKN_FAKE_MODEL_ID="inkling-2-mini-quality"))
        port = self.page_port(a, box)
        c = get_json(port, "/webchat/config")
        self.assertEqual(c["model"], "inkling2")
        self.assertEqual(c["models_installed"], [{"name": "quality", "current": False}, {"name": "inkling2", "current": True}])
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        c = get_json(port, "/webchat/config")
        self.assertEqual(c["model_id"], "inkling-2-mini-quality")
        code, body, _ = request(port, "POST", "/v1/chat/completions", self.own(port),                # the speed tab's request
                                json.dumps({"messages": [{"role": "user", "content": "x"}], "stream": True,
                                            "stream_options": {"include_usage": True},
                                            "chat_template_kwargs": {"enable_thinking": False}}).encode())
        self.assertEqual(code, 200)
        self.assertIn(b"[DONE]", body)
        H = QS.ControlRequest.HEADER
        self.assertEqual(request(port, "POST", "/webchat/switch_model", self.own(port, **{H: c["control_token"]}),
                                 b'{"model": "quality"}')[0], 202)
        self.finish(t, box, QS.SWITCH_RC)
        self.assertEqual(a.web_switch, "quality")
        self.assertIn("The page picked the quality model.", self.out.getvalue())

    def test_web_test_waits_for_the_model_and_passes(self):
        a = self.args(web_test=True)
        t, box = self.start(a)
        self.finish(t, box, 0)
        out = self.out.getvalue()
        self.assertIn("browser chat self-test: PASS", out)
        self.assertNotIn("FAIL", out)
        for name in ("the page is served", "the settings of the page", "the address for other programs", "the model is ready",
                     "the page is told so", "a page of another site is refused", "another host name is refused",
                     "a cross-site request is refused", "quitting needs the token of the page",
                     "a model switch from another site is refused", "other files are not served", "an answer arrives as a stream"):
            self.assertRegex(out, re.escape(name) + r"\s+PASS")
        self.assertIn("HTTP 404", out)                                                 # no control request in --web-test
        self.assertEqual(self.opened, [])
        self.assertTrue(wait_until(lambda: closed(a.web_port_used), 10))

    def test_self_test_also_passes_with_the_buttons_on_and_stops_nothing(self):
        a = self.args(no_browser=True)
        t, box = self.start(a)
        port = self.page_port(a, box)
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        self.assertTrue(QS.web_selftest(port))
        self.assertIn("HTTP 403", self.out.getvalue())
        self.assertTrue(t.is_alive())
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
        self.finish(t, box, QS.QUIT_RC)

    def test_server_that_never_gets_ready(self):
        a = self.args(no_browser=True)
        t0 = time.monotonic()
        t, box = self.start(a, env=dict(self.env, YSKN_FAKE_SERVE_FAILS="1"))
        port = self.page_port(a, box)
        # the page hears it before its server is closed
        self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("state") == "failed", 20))
        self.finish(t, box, 7)
        self.assertLess(time.monotonic() - t0, 30)
        self.assertIn("The server did not start", self.out.getvalue())
        self.assertTrue(wait_until(lambda: closed(port), 10))

    def test_web_test_with_a_server_that_never_gets_ready(self):
        a = self.args(web_test=True)
        t, box = self.start(a, env=dict(self.env, YSKN_FAKE_SERVE_FAILS="1"))
        self.finish(t, box, 7)
        self.assertNotIn("self-test", self.out.getvalue())

    def test_server_that_ends_later(self):
        a = self.args(no_browser=True)
        t, box = self.start(a)
        port = self.page_port(a, box)
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        self.children.procs[0].kill()
        t0 = time.monotonic()
        self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("state") == "failed", 20))
        # a page was told; the address still answers for the rest of the two seconds (a second tab, a page that asks later)
        time.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))
        self.assertEqual((get_json(port, "/webchat/status") or {}).get("state"), "failed")
        self.assertIsNone(get_json(port, "/webchat/config")["control_token"])          # and its buttons are off
        self.finish(t, box, 7)                         # not 0: START.cmd keeps the window with the server's messages open
        self.assertGreaterEqual(time.monotonic() - t0, 1.8)
        self.assertLess(time.monotonic() - t0, 15)
        self.assertTrue(wait_until(lambda: closed(port), 10))
        out = self.out.getvalue()
        self.assertIn("The server stopped while it was running", out)
        self.assertNotIn("The server did not start", out)

    def test_server_that_ends_later_without_an_error(self):
        a = self.args(no_browser=True)
        t, box = self.start(a)
        port = self.page_port(a, box)
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        eport = int(get_json(port, "/webchat/config")["api_base"].split(":")[2].split("/")[0])
        try:
            request(eport, "GET", "/fake/exit/0", timeout=5)                           # the server ends itself, exit code 0
        except (OSError, http.client.HTTPException):
            pass
        self.finish(t, box, 0)
        self.assertNotIn("The server stopped while it was running", self.out.getvalue())

    def test_keys_typed_before_the_page_do_not_end_it_and_enter_ends_it(self):
        con = self.console("x\r\r")                                                    # typed while the steps before ran
        a = self.args(no_browser=True)
        t, box = self.start(a, update_rel={"version": "9.9.9", "page": QS.REPO_URL + "/releases/tag/v9.9.9"})
        port = self.page_port(a, box)
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        time.sleep(0.6)                                                                # three looks of the wait
        self.assertTrue(t.is_alive(), "the keys typed before ended the page")
        self.assertEqual((con.keys, con.taken), ([], ["x", "\r", "\r"]))                # they were dropped before the wait began
        con.keys.extend("abc ")                                                        # other keys end nothing
        time.sleep(0.6)
        self.assertTrue(t.is_alive())
        c = get_json(port, "/webchat/config")
        self.assertEqual(get_json(port, "/webchat/status")["state"], "ready")
        con.keys.append("\r")                                                          # Enter in the console
        t0 = time.monotonic()
        # the page is told first ...
        self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("state") == "stopping", 10))
        # ... its buttons are off, the update button too ...
        c2 = get_json(port, "/webchat/config")
        self.assertEqual((c2["control_token"], c2["update_token"], c2["update"]["version"]), (None, None, "9.9.9"))
        self.assertEqual(request(port, "POST", "/webchat/update",
                                 self.own(port, **{"X-Yasakani-Token": c["update_token"]}))[0], 403)
        self.assertEqual(request(port, "POST", "/webchat/shutdown",
                                 self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]}))[0], 403)
        # ... and the address goes on answering for about two seconds, after `yasakani serve` is gone (nothing is left behind)
        self.assertTrue(wait_until(self.children.all_ended, 10))
        time.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))
        self.assertEqual((get_json(port, "/webchat/status") or {}).get("state"), "stopping")
        self.assertTrue(t.is_alive())
        self.finish(t, box, 0)
        took = time.monotonic() - t0
        self.assertTrue(1.8 <= took < 12, took)
        self.assertTrue(wait_until(lambda: closed(port), 10))
        self.assertIn("The chat was stopped", self.out.getvalue())
        self.assertNotIn("asked for the update", self.out.getvalue())

    def test_button_taken_in_the_moment_the_wait_ends_counts(self):
        """The page has its 202 already when Enter or the end of the server is seen: the button is done, not dropped."""
        def pressed(what):
            def wait(proc, update, poll=0.2, control=None, session=None):
                if what == "update":
                    self.assertTrue(update.take(update.token))
                else:
                    self.assertTrue(control.take(control.token, what, "speed" if what == "switch" else None))
                return "enter"
            return wait

        rel = {"version": "9.9.9", "page": QS.REPO_URL + "/releases/tag/v9.9.9"}
        for what, want in (("update", QS.UPDATE_RC), ("shutdown", QS.QUIT_RC), ("switch", QS.SWITCH_RC)):
            with self.subTest(what=what), mock.patch.object(QS, "wait_for_stop", pressed(what)):
                a = self.args(no_browser=True)
                t0 = time.monotonic()
                t, box = self.start(a, update_rel=rel)
                self.finish(t, box, want)
                self.assertLess(time.monotonic() - t0, 30)
                if what == "switch":
                    self.assertEqual(a.web_switch, "speed")

    def test_update_token_ends_with_a_switch_and_with_the_quit_button(self):
        """The wait ended for another reason: the update button is not taken any more, although the page's server closes last."""
        seen = {}
        real_stop = QS.stop_process

        def stop(proc, seconds=30, lib=None):
            # the launcher is on its way out; the page's server still answers.  What an update button pressed now gets:
            port, tok = seen["port"], seen["update_token"]
            seen["codes"] = [request(port, "POST", "/webchat/update", self.own(port, **{"X-Yasakani-Token": tok}))[0],
                             get_json(port, "/webchat/config")["update_token"]]
            return real_stop(proc, seconds, lib)

        rel = {"version": "9.9.9", "page": QS.REPO_URL + "/releases/tag/v9.9.9"}
        for path, body, want in (("/webchat/switch_model", b'{"model": "speed"}', QS.SWITCH_RC),
                                 ("/webchat/shutdown", b"", QS.QUIT_RC)):
            with self.subTest(path=path), mock.patch.object(QS, "stop_process", stop):
                a = self.args(no_browser=True)
                t, box = self.start(a, update_rel=rel)
                port = self.page_port(a, box)
                c = get_json(port, "/webchat/config")
                seen.update(port=port, update_token=c["update_token"], codes=None)
                self.assertRegex(c["update_token"], r"^[0-9a-f]{64}$")
                h = self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]})
                self.assertEqual(request(port, "POST", path, h, body)[0], 202)
                self.finish(t, box, want)
                self.assertEqual(seen["codes"], [403, None])
                self.assertNotIn("asked for the update", self.out.getvalue())

    def test_update_button_still_ends_the_page(self):
        a = self.args(no_browser=True)
        t, box = self.start(a, update_rel={"version": "9.9.9", "page": QS.REPO_URL + "/releases/tag/v9.9.9"})
        port = self.page_port(a, box)
        c = get_json(port, "/webchat/config")
        self.assertEqual(c["update"]["version"], "9.9.9")
        h = self.own(port, **{"X-Yasakani-Token": c["update_token"]})
        self.assertEqual(request(port, "POST", "/webchat/update", h)[0], 202)
        self.finish(t, box, QS.UPDATE_RC)
        self.assertEqual(a.web_port_used, port)

    def test_model_switch_is_handed_to_the_launcher(self):
        a = self.args(no_browser=True)
        t, box = self.start(a)
        port = self.page_port(a, box)
        tok = get_json(port, "/webchat/config")["control_token"]
        h = self.own(port, **{QS.ControlRequest.HEADER: tok})
        self.assertEqual(request(port, "POST", "/webchat/switch_model", h, b'{"model": "max"}')[0], 409)
        self.assertEqual(request(port, "POST", "/webchat/switch_model", h, b'{"model": "turbo"}')[0], 400)
        code, body, _ = request(port, "POST", "/webchat/switch_model", h, b'{"model": "speed"}')
        self.assertEqual((code, json.loads(body)), (202, {"status": "switching", "model": "speed"}))
        self.finish(t, box, QS.SWITCH_RC)
        self.assertEqual((a.web_switch, a.web_port_used), ("speed", port))
        self.assertTrue(wait_until(lambda: closed(port), 10))
        self.assertIn("The page picked the fastest model", self.out.getvalue())
        self.assertNotIn("The chat was stopped", self.out.getvalue())                  # it goes on with the other model

    def test_model_folder_of_ones_own_cannot_switch(self):
        a = self.args(no_browser=True, model_dir=str(self.tmp / "mine"))
        t, box = self.start(a)
        port = self.page_port(a, box)
        c = get_json(port, "/webchat/config")
        self.assertEqual((c["model"], c["models_installed"]), ("custom", []))
        self.assertEqual(get_json(port, "/webchat/status")["model"], "custom")
        h = self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]})
        self.assertEqual(request(port, "POST", "/webchat/switch_model", h, b'{"model": "speed"}')[0], 409)
        self.assertEqual(request(port, "POST", "/webchat/shutdown", h)[0], 202)
        self.finish(t, box, QS.QUIT_RC)

    @unittest.skipUnless(sys.platform == "win32", "the launcher runs on Windows; elsewhere a closed port is not free again at once")
    def test_after_a_switch_the_page_comes_back_on_its_port(self):
        a = self.args()
        t, box = self.start(a)
        port = self.page_port(a, box)
        kept = http.client.HTTPConnection("127.0.0.1", port, timeout=10)              # a connection the browser keeps open
        self.addCleanup(kept.close)
        kept.request("GET", "/webchat/status", headers={"Host": "127.0.0.1:%d" % port})
        self.assertEqual(json.loads(kept.getresponse().read())["model"], "quality")
        old = get_json(port, "/webchat/config")["control_token"]
        h = self.own(port, **{QS.ControlRequest.HEADER: old})
        # the stand-in server writes its line when it has started; on a busy PC the switch could stop it before that
        self.assertTrue(wait_until(lambda: len(self.served()) >= 1, 30))
        self.assertEqual(request(port, "POST", "/webchat/switch_model", h, b'{"model": "speed"}')[0], 202)
        self.finish(t, box, QS.SWITCH_RC)
        first = self.children.procs[0]
        t2, box2 = self.start(a, model=a.web_switch, reopen=a.web_port_used)
        self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("model") == "speed", 30), box2)
        self.assertEqual(a.web_port_used, port)                                        # the same address
        self.assertEqual(self.opened, ["http://127.0.0.1:%d/" % port])                # and no second browser window
        try:                                                                           # the kept connection is not served by
            kept.request("GET", "/webchat/status", headers={"Host": "127.0.0.1:%d" % port})   # the server that is gone
            self.assertEqual(json.loads(kept.getresponse().read())["model"], "speed")
        except (OSError, http.client.HTTPException):
            pass
        c = get_json(port, "/webchat/config")
        self.assertEqual(c["models_installed"], [{"name": "quality", "current": False}, {"name": "speed", "current": True}])
        self.assertNotEqual(c["control_token"], old)
        self.assertEqual(request(port, "POST", "/webchat/shutdown", h)[0], 403)        # the token of the server before
        self.assertIsNotNone(first.poll())                                             # one server at a time
        self.assertEqual(len([p for p in self.children.procs if p.poll() is None]), 1)
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        self.assertIn("goes on at the same address", self.out.getvalue())
        h2 = self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]})
        self.assertEqual(request(port, "POST", "/webchat/shutdown", h2)[0], 202)
        self.finish(t2, box2, QS.QUIT_RC)
        self.assertEqual([os.path.basename(x["model_dir"]) for x in self.served()], ["quality", "speed"])


# ------------------------------------------------------------------------------------------ the launcher from start to end
@unittest.skipUnless(sys.platform == "win32", "the launcher itself runs on Windows only")
class LauncherTests(RunWebBase):
    """main() with a home that looks set up (engine, environment, models: empty files), a stand-in for nvidia-smi and for the
    yasakani package.  Nothing is downloaded and github.com is not asked (the automatic check is off in the state)."""

    def setUp(self):
        super().setUp()
        try:
            str(self.tmp).encode("ascii")
        except UnicodeEncodeError:
            self.skipTest("the temporary folder has a name outside ASCII (the launcher refuses such a home)")
        card = dict(index=0, uuid="GPU-a", name="NVIDIA GeForce RTX 5090", cc=(12, 0), total=32607, used=1200, driver="616.92")
        self.asked = []
        for name, value in (("query_gpus", lambda run=None: [dict(card)]), ("cuda_driver_version", lambda load=None: 13040),
                            ("ask", lambda prompt, default="", eof=None: self.asked.append(prompt) or default)):
            p = mock.patch.object(QS, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.rc_file = self.tmp / "check_rc.txt"
        e = mock.patch.dict(os.environ, {"YSKN_FAKE_LOG": str(self.log), "YSKN_FAKE_LOAD": "0.3", "YASAKANI_NO_UPDATE_CHECK": "1",
                                         "YSKN_FAKE_RC_FILE": str(self.rc_file), "PYTHONDONTWRITEBYTECODE": "1"})
        e.start()
        self.addCleanup(e.stop)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.network = []
        n = mock.patch.object(QS, "fetch_bytes", lambda *a, **k: self.network.append(a) or self.fail("the network was asked"))
        n.start()
        self.addCleanup(n.stop)

    def home(self, checked, downloaded=("quality", "speed"), model="quality"):
        h = self.tmp / "home"
        (h / "engine").mkdir(parents=True)
        (h / "engine" / "yskn.dll").write_bytes(b"")
        (h / "engine" / "pyproject.toml").write_bytes(b"")
        (h / "venv" / "Scripts").mkdir(parents=True)
        (h / "venv" / "Scripts" / "python.exe").write_bytes(b"")                      # never run: see Children
        for m in downloaded:
            (h / "models" / m).mkdir(parents=True)
            (h / "models" / m / "yasakani.json").write_text("{}", encoding="utf-8")
        st = {"engine_sha": QS.ENGINE_SHA256, "venv_ok": True, "model": model, "update_check": False,
              "models_downloaded": {m: True for m in downloaded}, "models_ok": {m: True for m in checked}}
        (h / "quickstart_state.json").write_text(json.dumps(st), encoding="utf-8")
        return h

    def launch(self, h, *more):
        box, port = {}, free_port()

        def run():
            try:
                box["rc"] = QS.main(["--home", str(h), "--lang", "en", "--web-port", str(port)] + list(more))
            except BaseException as e:  # noqa: BLE001 - shown by the test
                box["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status") or box, 40), self.out.getvalue())
        self.assertEqual(box, {}, self.out.getvalue())
        return t, box, port

    def state(self, h):
        return json.loads((h / "quickstart_state.json").read_text(encoding="utf-8"))

    def test_second_run_goes_straight_to_the_page_then_switch_then_quit(self):
        h = self.home(checked=("quality",))                                           # speed is downloaded, never checked
        t, box, port = self.launch(h)
        self.assertEqual(self.asked, [])                                               # no model question, no menu
        self.assertEqual(self.opened, ["http://127.0.0.1:%d/" % port])
        c = get_json(port, "/webchat/config")
        self.assertEqual(c["models_installed"], [{"name": "quality", "current": True}, {"name": "speed", "current": False}])
        self.assertEqual(c["gpu"]["name"], "NVIDIA GeForce RTX 5090")
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        H = QS.ControlRequest.HEADER
        self.assertEqual(request(port, "POST", "/webchat/switch_model", self.own(port, **{H: c["control_token"]}),
                                 b'{"model": "speed"}')[0], 202)
        # (while the checks of the new model run, the address answers "switching" for it: wait for its browser chat)
        self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("state") in ("loading", "ready"), 40),
                        self.out.getvalue())
        self.assertEqual(get_json(port, "/webchat/status")["model"], "speed")
        c2 = get_json(port, "/webchat/config")
        self.assertEqual((c2["model"], c2["version"]), ("speed", QS.QS_VERSION))
        self.assertNotEqual(c2["control_token"], c["control_token"])
        # the model that was never checked got its three checks before its server was started, and the state was saved
        self.assertTrue(wait_until(lambda: len(self.served()) >= 5, 20))               # the new server has written its line
        log = [(x["cmd"], os.path.basename(x["model_dir"])) for x in self.served()]
        self.assertEqual(log, [("serve", "quality"), ("info", "speed"), ("verify", "speed"), ("selfcheck", "speed"),
                               ("serve", "speed")])
        st = self.state(h)
        self.assertEqual((st["model"], st["models_ok"]), ("speed", {"quality": True, "speed": True}))
        self.assertIn(os.path.join("models", "speed"), (h / "yasakani_env.cmd").read_text(encoding="ascii"))
        self.assertEqual(len([p for p in self.children.procs if p.poll() is None]), 1)       # no second server
        self.assertEqual(self.opened, ["http://127.0.0.1:%d/" % port])                      # no second browser window
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{H: c2["control_token"]}))[0], 202)
        self.finish(t, box, 0)
        self.assertTrue(wait_until(lambda: closed(port), 10))
        self.assertEqual(self.asked, [])
        self.assertEqual(self.network, [])
        out = self.out.getvalue()
        self.assertIn("with the quality model (this model passed its checks before)", out)
        self.assertIn("type START.cmd --menu in a command prompt opened in this folder", out)
        self.assertNotIn("What next?", out)
        self.assertIn("Next time, double-clicking START.cmd opens the page in your browser directly", out)
        self.assertIn("Bye.", out)

    def test_checks_of_a_switch_are_told_to_the_page_and_the_state_follows_them(self):
        """A switch to a model that never passed the checks: while they run, the page's address says so; the state file and the
        settings file name the new model only after it passed."""
        h = self.home(checked=("quality",))
        with mock.patch.dict(os.environ, {"YSKN_FAKE_CHECK_SECONDS": "1.0"}):
            t, box, port = self.launch(h)
        c = get_json(port, "/webchat/config")
        self.assertNotIn("phase", get_json(port, "/webchat/status"))
        H = QS.ControlRequest.HEADER
        self.assertEqual(request(port, "POST", "/webchat/switch_model", self.own(port, **{H: c["control_token"]}),
                                 b'{"model": "speed"}')[0], 202)
        seen = []

        def checking():
            s = get_json(port, "/webchat/status")
            if s:
                seen.append((s["state"], s.get("phase"), s["model"]))
            return s if s and s.get("phase") == "checking" else None

        s = wait_until(checking, 30)
        self.assertTrue(s, (seen, self.out.getvalue()))
        self.assertEqual((s["state"], s["phase"], s["model"]), ("switching", "checking", "speed"))
        self.assertIsInstance(s["elapsed"], (int, float))
        # nothing can be asked of the launcher now, and the state still names the model that works
        c2 = get_json(port, "/webchat/config")
        self.assertEqual((c2["control_token"], c2["api_base"], c2["model"]), (None, None, "speed"))
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{H: c["control_token"]}))[0], 404)
        self.assertEqual(request(port, "GET", "/health")[0], 503)
        self.assertEqual(request(port, "GET", "/")[0], 200)
        st = self.state(h)
        self.assertEqual((st["model"], st["models_ok"]), ("quality", {"quality": True}))
        self.assertIn(os.path.join("models", "quality"), (h / "yasakani_env.cmd").read_text(encoding="ascii"))
        self.assertEqual(len([p for p in self.children.procs if p.poll() is None and "serve" in p.args]), 0)   # no server now
        # then the browser chat is back at the same address with the new model, and the state follows
        self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("state") in ("loading", "ready"), 40),
                        self.out.getvalue())
        s2 = get_json(port, "/webchat/status")
        self.assertEqual((s2["model"], sorted(s2)), ("speed", ["elapsed", "model", "state"]))
        self.assertEqual(seen[-1], ("switching", "checking", "speed"))
        self.assertNotIn(("failed", None, "quality"), seen)
        st = self.state(h)
        self.assertEqual((st["model"], st["models_ok"]), ("speed", {"quality": True, "speed": True}))
        self.assertIn(os.path.join("models", "speed"), (h / "yasakani_env.cmd").read_text(encoding="ascii"))
        self.assertEqual(self.opened, ["http://127.0.0.1:%d/" % port])
        c3 = get_json(port, "/webchat/config")
        self.assertRegex(c3["api_base"], r"^http://127\.0\.0\.1:\d+/v1$")
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{H: c3["control_token"]}))[0], 202)
        self.finish(t, box, 0)
        self.assertTrue(wait_until(lambda: closed(port), 10))

    def test_menu_without_a_keyboard_quits_once_and_starts_nothing(self):
        """The input is at its end (a closed input, a file that has ended).  With Enter = the page, the menu once started the
        page and the server again and again; now the end of the input is "quit"."""
        h = self.home(checked=())
        box = {}

        def run():
            try:
                box["rc"] = QS.main(["--home", str(h), "--lang", "en", "--web-port", str(free_port())])
            except BaseException as e:  # noqa: BLE001 - shown by the test
                box["error"] = e

        with mock.patch.object(QS, "ask", REAL_ASK), mock.patch("builtins.input", side_effect=EOFError) as inp:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(60)
            self.assertFalse(t.is_alive(), "the launcher did not end")
            self.assertEqual(box, {"rc": 0})
            prompts = [c[0][0] for c in inp.call_args_list]
        self.assertEqual(len(prompts), 2)                                              # the model question, then the menu: once
        self.assertIn("Which model?", prompts[0])
        self.assertEqual(prompts[1], QS.menu_prompt(QS.Ui("en")))
        self.assertEqual([x["cmd"] for x in self.served()], ["info", "verify", "selfcheck"])       # no server was started
        self.assertEqual((self.opened, self.children.all_ended()), ([], True))
        self.assertIn("Bye.", self.out.getvalue())

    def test_enter_at_the_menu_opens_the_page(self):
        h = self.home(checked=("quality",))
        answers = iter(["", ""])                                                       # Enter at the model question and at the menu
        box, port = {}, free_port()

        def run():
            try:
                box["rc"] = QS.main(["--home", str(h), "--lang", "en", "--web-port", str(port), "--menu"])
            except BaseException as e:  # noqa: BLE001 - shown by the test
                box["error"] = e

        with mock.patch.object(QS, "ask", REAL_ASK), mock.patch("builtins.input", side_effect=lambda prompt="": next(answers)):
            t = threading.Thread(target=run, daemon=True)
            t.start()
            self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status") or box, 40), self.out.getvalue())
            self.assertEqual(box, {})
            self.assertEqual(self.opened, ["http://127.0.0.1:%d/" % port])
            tok = get_json(port, "/webchat/config")["control_token"]
            self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
            self.finish(t, box, 0)                                                     # the quit button: no menu again
        self.assertEqual([x["cmd"] for x in self.served()], ["serve"])

    def test_last_lines_say_what_the_next_double_click_does(self):
        """A model that has not passed the checks (--skip-checks): the next double-click runs them and shows the menu."""
        h = self.home(checked=())
        t, box, port = self.launch(h, "--action", "web", "--no-browser", "--skip-checks")
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
        self.finish(t, box, 0)
        out = self.out.getvalue()
        self.assertIn("Next time just double-click START.cmd again (finished steps are skipped)", out)
        self.assertNotIn("opens the page in your browser directly", out)
        self.assertEqual(self.state(h)["models_ok"], {})

    def test_first_run_asks_and_shows_the_menu_and_the_quit_button_ends_it(self):
        h = self.home(checked=(), model="speed")
        t, box, port = self.launch(h)
        self.assertEqual(len(self.asked), 2)                                           # the model question, then the menu
        self.assertIn("Which model?", self.asked[0])
        self.assertTrue(self.asked[0].rstrip().endswith("[2]:"))                       # the model of the state is the default
        self.assertEqual(self.asked[1], QS.menu_prompt(QS.Ui("en")))                   # Enter there opened the page
        self.assertTrue(wait_until(lambda: len(self.served()) >= 4, 20))               # the server writes its line when started
        log = [(x["cmd"], os.path.basename(x["model_dir"])) for x in self.served()]
        self.assertEqual(log, [("info", "speed"), ("verify", "speed"), ("selfcheck", "speed"), ("serve", "speed")])
        self.assertEqual(self.state(h)["models_ok"], {"speed": True})
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
        self.finish(t, box, 0)                                                         # ends, although it came from the menu
        self.assertEqual(len(self.asked), 2)                                           # the menu is not shown again
        self.assertNotIn("passed its checks before", self.out.getvalue())              # that line belongs to the direct start

    def test_menu_flag_shows_the_menu_on_a_later_run(self):
        h = self.home(checked=("quality", "speed"))
        t, box, port = self.launch(h, "--menu")
        self.assertEqual(len(self.asked), 2)
        self.assertIn("What next?", self.asked[1])
        self.assertTrue(wait_until(lambda: len(self.served()) >= 1, 20))
        self.assertEqual([x["cmd"] for x in self.served()], ["serve"])                 # checked before: no checks again
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
        self.finish(t, box, 0)

    def test_action_web_as_after_an_update(self):
        """An older launcher that updated itself starts this one with --action web --no-browser --web-port N."""
        h = self.home(checked=())                                                      # also when the model was never checked
        t, box, port = self.launch(h, "--action", "web", "--no-browser")
        self.assertEqual((self.asked, self.opened), ([], []))
        self.assertTrue(wait_until(lambda: len(self.served()) >= 4, 20))
        self.assertEqual([x["cmd"] for x in self.served()], ["info", "verify", "selfcheck", "serve"])
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
        self.finish(t, box, 0)

    def test_model_dir_never_starts_directly(self):
        mine = self.tmp / "mine"
        mine.mkdir()
        (mine / "yasakani.json").write_text("{}", encoding="utf-8")
        h = self.home(checked=("quality", "speed"))
        t, box, port = self.launch(h, "--model-dir", str(mine), "--skip-checks")
        self.assertEqual(len(self.asked), 1)                                           # the menu (no model question with a folder)
        self.assertIn("What next?", self.asked[0])
        c = get_json(port, "/webchat/config")
        self.assertEqual((c["model"], c["models_installed"]), ("custom", []))
        self.assertEqual(request(port, "POST", "/webchat/shutdown",
                                 self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]}))[0], 202)
        self.finish(t, box, 0)

    def test_failed_checks_of_the_switched_model_go_back_to_the_model_before(self):
        """The state file never names a model whose checks failed; the page gets the model that worked back at its address."""
        h = self.home(checked=("quality",))
        t, box, port = self.launch(h)
        H = QS.ControlRequest.HEADER
        tok = get_json(port, "/webchat/config")["control_token"]
        self.rc_file.write_text("1", encoding="utf-8")                                 # from now on every check fails
        self.assertEqual(request(port, "POST", "/webchat/switch_model", self.own(port, **{H: tok}), b'{"model": "speed"}')[0], 202)

        def back():
            c = get_json(port, "/webchat/config")
            return c if c and c["control_token"] and c["control_token"] != tok else None

        c = wait_until(back, 40)                                                       # the browser chat answers again
        self.assertTrue(c, self.out.getvalue())
        self.assertEqual((c["model"], get_json(port, "/webchat/status")["model"]), ("quality", "quality"))
        self.assertEqual(c["models_installed"], [{"name": "quality", "current": True}, {"name": "speed", "current": False}])
        self.assertTrue(wait_until(lambda: len(self.served()) >= 3, 20))               # the new server has written its line
        log = [(x["cmd"], os.path.basename(x["model_dir"])) for x in self.served()]
        self.assertEqual(log, [("serve", "quality"), ("info", "speed"), ("serve", "quality")])     # no server for the failed one
        st = self.state(h)
        self.assertEqual((st["model"], st["models_ok"]), ("quality", {"quality": True}))
        env_file = (h / "yasakani_env.cmd").read_text(encoding="ascii")
        self.assertIn(os.path.join("models", "quality"), env_file)
        self.assertNotIn(os.path.join("models", "speed"), env_file)
        out = self.out.getvalue()
        self.assertIn("`yasakani info` failed (exit code 1)", out)
        self.assertIn("The fastest model did not pass the checks, so going back to the quality model.", out)
        self.assertEqual(self.opened, ["http://127.0.0.1:%d/" % port])                # the same address, no second browser window
        self.assertEqual(len([p for p in self.children.procs if p.poll() is None]), 1)
        self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status")["state"] == "ready", 30))
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{H: c["control_token"]}))[0], 202)
        self.finish(t, box, 0)
        self.assertTrue(wait_until(lambda: closed(port), 10))
        self.assertEqual(self.state(h)["model"], "quality")

    # ---- 0.3.2: Inkling-2 Mini as the 4th model
    def ink_folder(self):
        src = self.tmp / "hf_inkling2"
        src.mkdir()
        (src / "yasakani.json").write_text('{"architecture": "inkling2"}', encoding="utf-8")
        (src / "inkling_2_mini_quality.yskn").write_bytes(b"\0" * 64)
        return src

    def test_answer_4_downloads_inkling2_checks_it_and_opens_the_page(self):
        """The first run of the model: 4 at the model question, the files of the test folder instead of Hugging Face, the three
        checks, the menu, the page with Inkling-2 Mini, the quit button.  The next run opens the page with it; with --menu the
        question offers 4."""
        h = self.home(checked=("quality",), downloaded=("quality",))
        src = self.ink_folder()
        port = free_port()
        box = {}

        def answer(prompt, default="", eof=None):
            self.asked.append(prompt)
            return "4" if "Which model?" in prompt else default

        def run():
            try:
                box["rc"] = QS.main(["--home", str(h), "--lang", "en", "--web-port", str(port), "--menu"])
            except BaseException as e:  # noqa: BLE001 - shown by the test
                box["error"] = e

        with mock.patch.object(QS, "ask", answer), \
                mock.patch.dict(os.environ, {"YASAKANI_QS_TEST_MODEL_SRC_INKLING2": str(src),
                                             "YSKN_FAKE_MODEL_ID": "inkling-2-mini-quality"}):
            t = threading.Thread(target=run, daemon=True)
            t.start()
            self.assertTrue(wait_until(lambda: get_json(port, "/webchat/status") or box, 40), self.out.getvalue())
            self.assertEqual(box, {}, self.out.getvalue())
            c = get_json(port, "/webchat/config")
            self.assertEqual(c["model"], "inkling2")
            self.assertEqual(c["models_installed"], [{"name": "quality", "current": False}, {"name": "inkling2", "current": True}])
            self.assertEqual(request(port, "POST", "/webchat/shutdown",
                                     self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]}))[0], 202)
            self.finish(t, box, 0)
        self.assertTrue(self.asked[0].rstrip().endswith("[1]:"))                       # the quality model was the one before
        log = [(x["cmd"], os.path.basename(x["model_dir"])) for x in self.served()]
        self.assertEqual(log, [("info", "inkling2"), ("verify", "inkling2"), ("selfcheck", "inkling2"), ("serve", "inkling2")])
        self.assertTrue((h / "models" / "inkling2" / "inkling_2_mini_quality.yskn").is_file())
        st = self.state(h)
        self.assertEqual((st["model"], st["models_downloaded"].get("inkling2"), st["models_ok"].get("inkling2")),
                         ("inkling2", True, True))
        out = self.out.getvalue()
        self.assertIn("Model: Inkling-2 Mini model (experimental)", out)
        self.assertIn("(needed: about 17 GB)", out)
        self.assertIn("Test only: the model files come from", out)
        self.assertEqual(self.network, [])
        self.assertIn(os.path.join("models", "inkling2"), (h / "yasakani_env.cmd").read_text(encoding="ascii"))
        # the next run: the page at once with Inkling-2 Mini
        self.asked.clear()
        t, box, port = self.launch(h)
        self.assertEqual(self.asked, [])
        c = get_json(port, "/webchat/config")
        self.assertEqual(c["model"], "inkling2")
        self.assertEqual(request(port, "POST", "/webchat/shutdown",
                                 self.own(port, **{QS.ControlRequest.HEADER: c["control_token"]}))[0], 202)
        self.finish(t, box, 0)
        self.assertIn("Opening the page in your browser with the Inkling-2 Mini model (experimental)", self.out.getvalue())
        # with --menu the question offers 4, the model of the state
        t, box, port = self.launch(h, "--menu")
        self.assertTrue(self.asked[0].rstrip().endswith("[4]:"), self.asked[0])
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{QS.ControlRequest.HEADER: tok}))[0], 202)
        self.finish(t, box, 0)

    def test_switch_from_quality_to_inkling2_and_back(self):
        h = self.home(checked=("quality", "inkling2"), downloaded=("quality", "inkling2"))
        t, box, port = self.launch(h)
        H = QS.ControlRequest.HEADER
        for to in ("inkling2", "quality"):
            self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("state") == "ready", 30))
            tok = get_json(port, "/webchat/config")["control_token"]
            self.assertEqual(request(port, "POST", "/webchat/switch_model", self.own(port, **{H: tok}),
                                     json.dumps({"model": to}).encode())[0], 202)

            def there(to=to):
                s = get_json(port, "/webchat/status") or {}
                return s.get("model") == to and s.get("state") in ("loading", "ready")

            self.assertTrue(wait_until(there, 40), self.out.getvalue())
            self.assertEqual(self.state(h)["model"], to)
        self.assertTrue(wait_until(lambda: len(self.served()) >= 3, 20))
        self.assertEqual([os.path.basename(x["model_dir"]) for x in self.served()], ["quality", "inkling2", "quality"])
        self.assertTrue(wait_until(lambda: (get_json(port, "/webchat/status") or {}).get("state") == "ready", 30))
        tok = get_json(port, "/webchat/config")["control_token"]
        self.assertEqual(request(port, "POST", "/webchat/shutdown", self.own(port, **{H: tok}))[0], 202)
        self.finish(t, box, 0)
        self.assertIn("The page picked the Inkling-2 Mini model (experimental).", self.out.getvalue())

    def run_main(self, home, *more):
        return QS.main(["--home", str(home), "--lang", "en"] + list(more))

    def test_cards_under_24_gb_do_not_offer_inkling2(self):
        small = dict(index=0, uuid="GPU-s", name="NVIDIA GeForce RTX 4080", cc=(8, 9), total=16376, used=600, driver="616.92")
        h = self.home(checked=("speed",), downloaded=("speed", "inkling2"), model="speed")
        with mock.patch.object(QS, "query_gpus", lambda run=None: [dict(small)]):
            self.assertEqual(self.run_main(h, "--model", "inkling2", "--action", "none"), 2)
            self.assertIn("The quality and max-quality models and Inkling-2 Mini need a GPU with 24 GB or more", self.out.getvalue())
            self.assertEqual(self.run_main(h, "--action", "none"), 0)                   # no question: the fastest model
        self.assertEqual(self.asked, [])
        self.assertIn("only the fastest model can be used (the quality and max-quality models and Inkling-2 Mini do not fit)",
                      self.out.getvalue())
        self.assertEqual(QS.installed_models(str(h), self.state(h)["models_downloaded"], False), ["speed"])

    def test_context_line_for_inkling2_on_a_fuller_24_gb_card(self):
        h = self.home(checked=("inkling2",), downloaded=("inkling2",), model="inkling2")
        # 0.3.4: auto = the fast path on an RTX 4090 (8.9), so the launcher plans with it (12,288 tokens with 6,000 MiB in use)
        for used, want in ((800, None), (5000, None), (6000, QS.estimate_ink2_context(24564 - 6000, fast=True)), (9000, 0)):
            card = dict(index=0, uuid="GPU-b", name="NVIDIA GeForce RTX 4090", cc=(8, 9), total=24564, used=used, driver="616.92")
            self.out.truncate(0)
            self.out.seek(0)
            with mock.patch.object(QS, "query_gpus", lambda run=None: [dict(card)]), \
                    mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": ""}):
                self.assertEqual(self.run_main(h, "--action", "none"), 0)
            out = self.out.getvalue()
            if want is None:
                self.assertNotIn("context length", out)
            elif want:
                self.assertTrue(8192 <= want < 32768, want)
                self.assertIn("The context length of Inkling-2 Mini is chosen from the free GPU memory each time it starts "
                              "(%s tokens with what is free now; at most 32,768)" % "{:,}".format(want), out)
            else:
                self.assertIn("There is not enough free GPU memory now (9000 MiB in use)", out)

    def test_compare_item_is_for_qwen_folders_only(self):
        h = self.home(checked=("inkling2",), downloaded=("inkling2",), model="inkling2")
        (h / "models" / "inkling2" / "yasakani.json").write_text('{"architecture": "inkling2"}', encoding="utf-8")
        calls = []
        with mock.patch.object(QS.subprocess, "call", lambda *a, **k: calls.append(a) or 0):
            self.assertEqual(self.run_main(h, "--action", "compare"), 0)
        self.assertEqual(calls, [])
        self.assertIn("The look-ahead comparison works with the Qwen3.8 models only.", self.out.getvalue())


# ------------------------------------------------------------------------------------------------------------- START.cmd
@unittest.skipUnless(sys.platform == "win32", "cmd.exe is Windows only")
class StartCmdTests(unittest.TestCase):
    """START.cmd after quickstart.py ended: with exit code 0 the window closes by itself (no `pause`); with any other code it
    waits for a key, so that the messages can be read.  The exit code of quickstart.py is returned either way.  (That START.cmd
    can be replaced while it runs is checked in test_update.py.)"""

    def setUp(self):
        if START_CMD is None or not START_CMD.is_file():
            self.skipTest("quickstart/START.cmd not found")
        self.tmp = Path(tempfile.mkdtemp(prefix="yskn_qsweb_c_"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)

    def run_start(self, code, through_call):
        """-> (the exit code cmd.exe returns, what was printed after quickstart.py ended, everything printed)."""
        d = self.tmp / ("%s%d" % ("call" if through_call else "cmd", code))
        d.mkdir()
        shutil.copyfile(str(START_CMD), str(d / "START.cmd"))
        (d / "quickstart.py").write_text("import sys\nprint('stub', sys.argv[1:])\nprint('STUB-END')\nsys.exit(int(sys.argv[1]))\n",
                                         encoding="ascii")
        cmd = [str(d / "START.cmd"), str(code), "extra"]
        if through_call:
            (d / "outer.cmd").write_bytes(b'@echo off\r\ncall "%~dp0START.cmd" %*\r\necho AFTER-CALL %ERRORLEVEL%\r\n'
                                          b"exit /b %ERRORLEVEL%\r\n")
            cmd[0] = str(d / "outer.cmd")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run(["cmd.exe", "/d", "/c"] + cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, env=env)
        out = (r.stdout + r.stderr).decode("ascii", "replace")
        if "Python 3.10 or newer was not found" in out:
            self.skipTest("no Python on PATH for START.cmd")
        self.assertIn("stub ['%d', 'extra']" % code, out)                          # the arguments are passed on
        after = out.split("STUB-END", 1)[1]
        if through_call:
            self.assertIn("AFTER-CALL %d" % code, after, out)                          # the caller goes on, with the code
            after = after.split("AFTER-CALL", 1)[0]
        return r.returncode, after.strip(), out

    def test_exit_code_0_closes_the_window_and_errors_wait_for_a_key(self):
        for through_call in (False, True):
            for code in (0, 1, 7, 130):
                with self.subTest(code=code, through_call=through_call):
                    got, after, out = self.run_start(code, through_call)
                    self.assertEqual(got, code, out)
                    if code == 0:
                        self.assertEqual(after, "", out)                               # nothing more: no `pause`
                    else:
                        self.assertNotEqual(after, "", out)                            # the line of `pause` (in Windows' language)

    def test_everything_after_quickstart_is_one_block(self):
        """An update replaces START.cmd while it runs: cmd.exe must have read all that follows quickstart.py before it starts."""
        lines = START_CMD.read_bytes().split(b"\r\n")
        self.assertNotIn(b"\n", b"".join(lines))                                       # CRLF only
        self.assertTrue(all(c < 128 for c in START_CMD.read_bytes()))
        run = lines.index(b":run")
        body = [ln for ln in lines[run + 1:] if ln and not ln.lower().startswith(b"rem ")]
        self.assertEqual((body[0], body[-1]), (b"(", b")"))
        self.assertIn(b'%PY% "%~dp0quickstart.py" %*', body[1])
        self.assertEqual([ln for ln in body[1:-1] if ln.strip() in (b"(", b")")], [])  # one block, nothing nested or after it


if __name__ == "__main__":
    unittest.main()
