#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise a prebuilt app's real model and HTTP API; requires downloaded weights."""
import argparse
import json
from pathlib import Path
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    resources = args.app.resolve() / "Contents" / "Resources"
    runtime = resources / "runtime" / "yasakani"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = "http://127.0.0.1:%d" % port
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    env = dict(os.environ)
    env.pop("YASAKANI_API_KEY", None)
    with tempfile.TemporaryFile(mode="w+") as log:
        child = subprocess.Popen([str(runtime), "serve", "--backend", "metal", "--model-dir", str(args.model_dir.resolve()),
                                  "--port", str(port), "--max-ctx", "512", "--no-warmup", "--web-ui", str(resources / "chat.html")],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if child.poll() is not None:
                    raise RuntimeError("the bundled server exited before becoming ready")
                log.seek(0)
                ready_log = "ready: OpenAI-compatible API at " in log.read()
                try:
                    with opener.open(base + "/health", timeout=2) as response:
                        health = json.load(response)
                    if ready_log and health["status"] == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("the model did not become ready within 180 seconds")
                time.sleep(0.25)
            with opener.open(base + "/", timeout=5) as response:
                assert response.read() == (resources / "chat.html").read_bytes()
            body = dict(model=health["model"], messages=[dict(role="user", content="What is 2 + 2? Answer briefly.")],
                        max_tokens=args.max_tokens, temperature=0, seed=42, enable_thinking=False)
            results = []
            for index in range(2):
                start = time.monotonic()
                request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"})
                with opener.open(request, timeout=600) as response:
                    result = json.load(response)
                message = result["choices"][0]["message"]
                assert message.get("content"), "real model returned no answer content"
                results.append(message)
                print(json.dumps(dict(run=index + 1, elapsed_seconds=round(time.monotonic() - start, 3),
                                      message=message, usage=result["usage"]), ensure_ascii=False), flush=True)
            assert results[0] == results[1], "repeated greedy requests differ"
            print("PASS: bundled runtime, real Metal generation, HTTP chat, repeatability")
        except BaseException:
            log.seek(0)
            print(log.read()[-12000:])
            raise
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    main()
