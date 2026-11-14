# SPDX-License-Identifier: Apache-2.0
"""Serve the fixed chat resource without opening the filesystem or weakening API checks."""
import http.client
import json
import threading
import unittest
from unittest import mock

from yasakani import cli
from yasakani._native import YasakaniError
from yasakani.server import ApiState, ServerConfig, bind


class WebUITests(unittest.TestCase):
    PAGE = "<!doctype html><title>YASAKANI チャット</title>".encode("utf-8")

    def start(self, *, web_ui=True, api_key=None):
        cfg = ServerConfig(port=0, api_key=api_key)
        cfg.web_ui = self.PAGE if web_ui else None
        state = ApiState(cfg)
        httpd = bind(state)
        self.port = httpd.server_address[1]
        worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()

        def stop():
            httpd.shutdown()
            httpd.server_close()
            worker.join(3)
            self.assertFalse(worker.is_alive())

        self.addCleanup(stop)

    def request(self, path, *, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            headers = {name: response.getheader(name) for name, _ in response.getheaders()}
            return response.status, headers, response.read()
        finally:
            connection.close()

    def test_serves_exact_page_with_safe_headers_while_model_loads(self):
        self.start()
        code, headers, body = self.request("/")
        self.assertEqual((code, body), (200, self.PAGE))
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(int(headers["Content-Length"]), len(self.PAGE))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin")
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        code, _, body = self.request("/health")
        self.assertEqual((code, json.loads(body)["status"]), (503, "loading"))

    def test_config_and_query_string_routes(self):
        self.start()
        code, _, body = self.request("/webchat/config?cache=1")
        self.assertEqual((code, json.loads(body)), (200, {"model": "custom"}))
        self.assertEqual(self.request("/?cache=1")[2], self.PAGE)

    def test_ui_is_opt_in_and_only_two_fixed_routes_are_exposed(self):
        self.start(web_ui=False)
        for path in ("/", "/webchat/config"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)

    def test_traversal_and_arbitrary_files_are_not_served(self):
        self.start()
        for path in ("/../LICENSE", "/%2e%2e/LICENSE", "/..%2fLICENSE", "/chat.html", "/webchat/../config",
                     "/webchat/config/", "/etc/passwd", "/tokenizer.json"):
            with self.subTest(path=path):
                code, _, body = self.request(path)
                self.assertEqual(code, 404)
                self.assertEqual(json.loads(body)["error"]["type"], "not_found")

    def test_bad_host_cannot_access_the_page_or_config(self):
        self.start()
        for path in ("/", "/webchat/config", "/health", "/v1/models"):
            with self.subTest(path=path):
                code, _, body = self.request(path, headers={"Host": "evil.example:%d" % self.port})
                self.assertEqual(code, 403)
                self.assertIn("host", json.loads(body)["error"]["message"])

    def test_web_ui_preserves_post_origin_and_fetch_site_rejections(self):
        self.start()
        for headers in ({"Origin": "https://evil.example"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/v1/chat/completions", method="POST", headers=headers)[0], 403)
        # A request from the UI's own origin passes the origin guard, reaching
        # the ordinary model-loading response instead of being rejected.
        headers = {"Origin": "http://127.0.0.1:%d" % self.port, "Sec-Fetch-Site": "same-origin"}
        self.assertEqual(self.request("/v1/chat/completions", method="POST", headers=headers)[0], 503)

    def test_api_key_check_is_not_bypassed_by_web_routes(self):
        self.start(api_key="fixture-key")
        for path in ("/", "/webchat/config", "/v1/models"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 401)
                self.assertEqual(self.request(path, headers={"Authorization": "Bearer incorrect"})[0], 401)
                self.assertEqual(self.request(path, headers={"Authorization": "Bearer fixture-key"})[0], 200)
        self.assertEqual(self.request("/health")[0], 503)

    def test_cli_rejects_ui_on_remote_or_authenticated_server_before_binding(self):
        for extra in (["--host", "0.0.0.0", "--api-key", "fixture-key"], ["--api-key", "fixture-key"]):
            args = cli.build_parser().parse_args(["serve", "--web-ui", "not-read.html", *extra])
            with self.subTest(extra=extra), mock.patch("yasakani.server.bind") as bind_mock:
                with self.assertRaisesRegex(YasakaniError, "loopback host without"):
                    cli.cmd_serve(args)
                bind_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
