# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""OpenAI-compatible HTTP server (chat completions, streaming) around one engine.

* Listens on 127.0.0.1 by default.  The ``Host`` header must name the address the server is bound to, and a request that carries an
  ``Origin`` must come from that same address (so a web page open in your browser cannot talk to the server).  To serve other
  machines use ``--host 0.0.0.0 --api-key KEY``.
* One request is generated at a time; the others wait.  With ``--parallel N`` (experimental) up to N requests are generated at once
  (yasakani/sched.py); more wait, and with more than ``--max-queue`` waiting a request is answered 503 with ``Retry-After``.
* A stream carries one chunk per token (several chunks arrive together, about 64 times a second on Windows).
* Tool calling: ``tools`` (functions) with ``tool_choice`` "auto" or "none".  The model's own chat template describes the tools to
  the model; the calls it writes come back as ``tool_calls`` (in a stream: one chunk per finished call, with the whole arguments).
  ``tool_choice`` "required" or a named function is accepted but not enforced for Qwen3.8 models (no constrained decoding): it runs
  as "auto", with only the named function offered when one is named.  Inkling-2 model folders enforce it (yasakani/inkling2.py):
  after the thinking the call header is written for the model (a named function, or "required" with one function), or with
  "required" and several functions the model may write nothing but one of the offered names as the header (constrained decoding
  of the name).  The reply then ends after that call (one call per reply).  The arguments are the model's own (not checked
  against the schema), and a reply cut by max_tokens before the call has none.
* ``thinking_budget`` (a request field; the default comes from the environment variable YASAKANI_THINKING_BUDGET, 0 = no limit):
  when the thinking has taken that many tokens it is closed and the model goes on to the answer.
* Optional still-image input through ``image_url`` (Qwen3.8: configure --mmproj, --llama-dir and --vision-model; Inkling-2 Mini:
  on by itself when the model folder has its image tower, ``--vision off`` turns it off; the Linux
  engine of the published folder encodes on the GPU as well).  ``/v1/models`` says whether the model
  reads images (``"vision"``); image tokens count as prompt tokens in ``usage``.
  Not supported (answered with 400): the old ``functions`` / ``function_call`` fields,
  ``n`` > 1, ``logprobs``.
"""
from __future__ import annotations

import hmac
import json
import os
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ._native import YasakaniError
from .chat import (IMAGE_PART_TYPES, REASONING_EFFORT_ALIASES, ChatParams, ChatRunner, RequestError, Usage, clean_messages,
                   clean_tools, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P)

MAX_BODY = 32 * 1024 * 1024
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


class ServerConfig:
    def __init__(self, host="127.0.0.1", port=8765, api_key=None, model_id="qwen3.8-27b", default_max_tokens=32768,
                 max_tokens_cap=None, default_thinking_budget=None, hf_inkling2=False):
        self.host, self.port, self.api_key, self.model_id = host, int(port), api_key, model_id
        self.default_max_tokens, self.max_tokens_cap = int(default_max_tokens), max_tokens_cap
        self.hf_inkling2 = bool(hf_inkling2)               # the Inkling-2 engine of #28 (a folder without yasakani.json)
        if default_thinking_budget is None:               # no command-line option yet: the starter of the harness sets the variable
            try:
                default_thinking_budget = int(os.environ.get("YASAKANI_THINKING_BUDGET") or 0)
            except ValueError:
                default_thinking_budget = 0
        self.default_thinking_budget = max(0, int(default_thinking_budget))
        self.sampling_defaults = {}                       # the model's defaults (temperature, top_p, top_k); empty: those of Qwen3.8
        self.effort_aliases = None                        # accepted reasoning_effort values; None: REASONING_EFFORT_ALIASES
        self.keep_tool_ids = False                        # keep tool_call ids / tool_call_id / name of tool messages (Inkling-2)
        self.vision_parts = False                         # image parts are checked by the model's runner (Inkling-2 model folders)
        self.vision_problem = ""                          # why that model reads no images (the message of a refused image)


class ApiState:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.runner: Optional[ChatRunner] = None          # None while the model loads
        self.gen_lock = threading.Lock()
        self.version = ""
        self.sched = None                                   # serve --parallel N (N >= 2): the scheduler (yasakani.sched); None: gen_lock
        self.image_lock = threading.Lock()                  # with the scheduler: images are read and encoded one request at a time


class SingleServer(ThreadingHTTPServer):
    allow_reuse_address = False                              # Windows: SO_REUSEADDR would let two servers share one port
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return                                           # the client went away: nothing to report
        super().handle_error(request, client_address)


def _num(body, key, default, lo=None, hi=None, kind=float):
    v = body.get(key)
    if v is None:
        return default
    try:
        v = kind(v)
    except (TypeError, ValueError):
        raise RequestError("%s must be a number" % key)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise RequestError("%s must be between %s and %s" % (key, lo, hi))
    return v


def parse_chat_request(body: dict, cfg: ServerConfig, images: bool = False) -> ChatParams:
    """``images``: image input is configured.  Without it an image part is refused here, before the request waits for the GPU."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    for key in ("functions", "function_call"):
        if body.get(key):
            raise RequestError("%s is not supported; use tools" % key)
    tools = clean_tools(body.get("tools"))
    choice = body.get("tool_choice")
    forced = None
    if isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("function"), dict):
        forced = choice["function"].get("name")
        if not isinstance(forced, str) or not forced:
            raise RequestError("tool_choice of the type function needs function.name (a non-empty string)")
        tools = [t for t in tools or [] if t["function"]["name"] == forced]
        if not tools:
            raise RequestError("tool_choice names a function that is not in tools: %s" % json.dumps(forced)[:80])
    if choice == "none":
        tools = None
    elif choice == "required" or forced:
        # Qwen3.8: the engine has no constrained decoding, so a call cannot be enforced: the request runs as "auto" (with only
        # the named function offered, when one is named) and the model is left to make the call.  Inkling-2: enforced by its runner.
        if not tools:
            raise RequestError("tool_choice %s needs tools" % json.dumps(choice)[:80])
    elif choice not in (None, "auto"):
        raise RequestError("tool_choice %s is not supported; use \"auto\", \"none\", \"required\" or a function" % json.dumps(choice)[:80])
    if body.get("logprobs"):
        raise RequestError("logprobs is not supported")
    if body.get("n") not in (None, 1):
        raise RequestError("n must be 1 (one answer per request)")
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") not in (None, "text"):
        raise RequestError("response_format %r is not supported" % rf.get("type"))
    messages = clean_messages(body.get("messages"), images=images, tool_ids=bool(getattr(cfg, "keep_tool_ids", False)),
                              vision_parts=bool(getattr(cfg, "vision_parts", False)), vision_problem=getattr(cfg, "vision_problem", ""))
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    max_tokens = cfg.default_max_tokens if mt is None else _num({"v": mt}, "v", None, 1, None, int)
    if cfg.max_tokens_cap:
        max_tokens = min(max_tokens, int(cfg.max_tokens_cap))
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(s, str) for s in stop):
        raise RequestError("stop must be a string or a list of at most 4 strings")
    kw = dict(body.get("chat_template_kwargs") or {})
    if not isinstance(kw, dict):
        raise RequestError("chat_template_kwargs must be an object")
    thinking = body.get("thinking")                       # {"type": "enabled" | "disabled"}, as some clients send it
    if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled"):
        kw.setdefault("enable_thinking", thinking["type"] == "enabled")
    if isinstance(body.get("enable_thinking"), bool):
        kw.setdefault("enable_thinking", body["enable_thinking"])
    eff = body.get("reasoning_effort")
    aliases = getattr(cfg, "effort_aliases", None) or REASONING_EFFORT_ALIASES
    if eff is not None:
        if getattr(cfg, "hf_inkling2", False):           # the names of the chat template, or its number in [0, 0.99]
            if isinstance(eff, str):
                if eff.strip() not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                    raise RequestError("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, max or a number in [0.0, 0.99]")
                kw.setdefault("reasoning_effort", eff.strip())
            elif isinstance(eff, (int, float)) and not isinstance(eff, bool) and 0.0 <= float(eff) <= 0.99:
                kw.setdefault("reasoning_effort", float(eff))
            else:
                raise RequestError("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, max or a number in [0.0, 0.99]")
        else:
            if eff not in aliases:
                raise RequestError("reasoning_effort must be one of " + ("low, medium, high, xhigh" if aliases is REASONING_EFFORT_ALIASES
                                                                         else ", ".join(aliases)))
            kw.setdefault("reasoning_effort", aliases[eff])
    tb = body.get("thinking_budget", kw.pop("thinking_budget", None))
    thinking_budget = cfg.default_thinking_budget if tb is None else _num({"thinking_budget": tb}, "thinking_budget", 0, 0, None, int)
    seed = body.get("seed")
    d = getattr(cfg, "sampling_defaults", None) or {}
    return ChatParams(
        messages=messages, max_tokens=max_tokens,
        temperature=_num(body, "temperature", d.get("temperature", DEFAULT_TEMPERATURE), 0.0, 2.0),
        top_p=_num(body, "top_p", d.get("top_p", DEFAULT_TOP_P), 1e-6, 1.0),
        top_k=_num(body, "top_k", d.get("top_k", DEFAULT_TOP_K), None, None, int),
        min_p=_num(body, "min_p", 0.0, 0.0, 1.0),
        seed=None if seed is None else _num(body, "seed", 0, None, None, int),
        stop=list(stop), ignore_eos=bool(body.get("ignore_eos", False)), loop_guard=body.get("loop_guard", True) is not False,
        thinking_budget=thinking_budget, template_kwargs=kw, tools=tools,
        tool_choice="required" if tools and (choice == "required" or forced) else None)


def tool_call_json(call: dict, index: Optional[int] = None) -> dict:
    """{"name", "arguments"} from the chat runner -> an OpenAI tool call (the arguments as a JSON string)."""
    d = dict(id="call_" + uuid.uuid4().hex[:24], type="function",
             function=dict(name=call["name"], arguments=json.dumps(call["arguments"], ensure_ascii=False)))
    return d if index is None else dict(index=index, **d)


def usage_json(u: Usage) -> dict:
    d = dict(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, total_tokens=u.prompt_tokens + u.completion_tokens,
             prompt_tokens_details=dict(cached_tokens=u.cached_tokens),
             completion_tokens_details=dict(reasoning_tokens=u.reasoning_tokens))
    v = getattr(u, "vision", None)                        # a request with images (Inkling-2): they are prompt tokens already
    if isinstance(v, dict) and v.get("images"):
        d["prompt_tokens_details"].update(image_tokens=int(v.get("image_tokens", 0)), images=int(v["images"]))
    if u.loop_guard:
        d["loop_guard"] = u.loop_guard
    return d


def timings_json(u: Usage) -> dict:
    """The engine's own timing of the request, in the layout of the ``timings`` object of the llama.cpp server (read by benchmark
    tools).  As there, ``prompt_ms`` runs until the first generated token and ``predicted_ms`` from it to the last one; when a request
    runs in two or three stages (thinking budget, loop guard), the later readings of the prompt count as generation time.
    ``draft_n`` / ``draft_n_accepted`` (only when the look-ahead proposed tokens): the tokens it proposed and those the model's own pass
    confirmed.  These times are measured inside the engine, not by the client as for the published speed tables."""
    prompt_n = max(u.prompt_tokens - u.cached_tokens, 0)
    prompt_ms = u.prompt_ms
    predicted_ms = u.gen_ms + max(u.prefill_ms - u.prompt_ms, 0.0)
    per_s = lambda n, ms: n * 1000.0 / ms if ms > 0 else 0.0   # noqa: E731
    per_tok = lambda n, ms: ms / n if n > 0 else 0.0           # noqa: E731
    d = dict(cache_n=u.cached_tokens,
             prompt_n=prompt_n, prompt_ms=prompt_ms, prompt_per_token_ms=per_tok(prompt_n, prompt_ms),
             prompt_per_second=per_s(prompt_n, prompt_ms),
             predicted_n=u.completion_tokens, predicted_ms=predicted_ms, predicted_per_token_ms=per_tok(u.completion_tokens, predicted_ms),
             predicted_per_second=per_s(u.completion_tokens, predicted_ms))
    if u.draft_n > 0:
        d.update(draft_n=u.draft_n, draft_n_accepted=u.draft_n_accepted)
    if u.spec:                                            # Inkling-2 with its drafter: the verification passes and tokens per pass
        d.update(u.spec)
    if u.queue_ms is not None:                            # served by the scheduler (--parallel N): its wait, the streams that shared
        d.update(queue_ms=u.queue_ms, parallel_avg=u.parallel_avg, snap_mode=u.snap_mode)    # its passes, how its prompt was read
    return d


def client_gone(sock: socket.socket) -> bool:
    """The client closed the connection (or it broke): looks at the socket without reading from it.  POSIX uses poll() (select()
    cannot take a socket numbered 1024 or more).  When the socket cannot be looked at, the client is not taken for gone (a stream
    finds out by writing)."""
    try:
        if hasattr(select, "poll"):
            p = select.poll()
            p.register(sock, select.POLLIN | select.POLLPRI)
            ev = p.poll(0)
            if not ev:
                return False
            if ev[0][1] & select.POLLNVAL:
                return False                                   # not an open socket of this process: nothing to say
        else:
            ready, _, _ = select.select([sock], [], [], 0)
            if not ready:
                return False
        return sock.recv(1, socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)) == b""
    except (BlockingIOError, InterruptedError, ValueError):
        return False
    except OSError:
        return True


def has_images(params: ChatParams) -> bool:
    return any(isinstance(m.get("content"), list) and any(isinstance(q, dict) and q.get("type") in IMAGE_PART_TYPES for q in m["content"])
               for m in params.messages)


def model_json(state: "ApiState") -> dict:
    """The entry of /v1/models: the model id, and whether the loaded model reads images (the chat page shows its image button
    only then)."""
    runner = state.runner
    return dict(id=state.cfg.model_id, object="model", owned_by="local",
                vision=runner is not None and getattr(runner, "vision", None) is not None)


def make_handler(state: ApiState):
    cfg = state.cfg

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 300                                        # seconds a connection may sit idle between requests

        def log_message(self, *a):                           # no access log (the server prints its own status lines)
            pass

        def setup(self):
            super().setup()
            try:                                             # a stream is many small pieces (one per token): send each at once
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        # ---- plumbing ---------------------------------------------------------------------------------------------------------
        def end_headers(self):
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def _json(self, code, obj, close=False, headers=None):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(b)

        def _error(self, code, message, kind="invalid_request_error", close=False, err_code=None, headers=None):
            self._json(code, dict(error=dict(message=message, type=kind, code=err_code or code)), close, headers)

        def _client_gone(self) -> bool:
            return client_gone(self.connection)

        def _drain(self):
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                n = 0
            if 0 < n <= MAX_BODY:
                self.rfile.read(n)

        def _own_hosts(self):
            port = self.server.server_address[1]
            names = list(LOOPBACK) if cfg.host in ("127.0.0.1", "localhost", "::1") else [cfg.host]
            return tuple("%s:%d" % (h, port) for h in names)

        def _refused(self, post) -> Optional[str]:
            if cfg.host not in ("127.0.0.1", "localhost", "::1") and cfg.api_key:
                return None                                    # remote use is protected by the key
            own = self._own_hosts()
            if (self.headers.get("Host") or "").lower() not in own:
                return "host"
            if post:
                origin = self.headers.get("Origin")
                if origin is not None and origin.lower() not in tuple("http://" + h for h in own):
                    return "origin"
                if (self.headers.get("Sec-Fetch-Site") or "none").lower() not in ("same-origin", "none"):
                    return "cross-site"
            return None

        def _unauthorized(self) -> bool:
            if not cfg.api_key:
                return False
            auth = self.headers.get("Authorization") or ""
            ok = auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].encode(), str(cfg.api_key).encode())
            if not ok:
                self._error(401, "invalid or missing API key", "authentication_error")
            return not ok

        # ---- routes -----------------------------------------------------------------------------------------------------------
        def do_GET(self):
            why = self._refused(False)
            if why:
                return self._error(403, "forbidden: " + why, "forbidden")
            path = self.path.split("?", 1)[0]
            if path == "/health":
                code = 200 if state.runner else 503
                body = dict(status="ok" if state.runner else "loading", version=state.version, model=cfg.model_id)
                if state.runner:
                    body["max_context_tokens"] = state.runner.engine.max_ctx
                    body["thinking_budget"] = cfg.default_thinking_budget
                    sched = state.sched
                    if sched is not None:                          # --parallel N: the context above is that of every request
                        st = sched.status()
                        body.update(parallel=st["parallel"], slots_busy=st["slots_busy"], slots_idle=st["slots_idle"], queued=st["queued"])
                        if st["state"] != "ok":
                            code, body["status"] = 503, st["state"]
                            if st["error"]:
                                body["error"] = st["error"]
                return self._json(code, body)
            if self._unauthorized():
                return
            if path in ("/v1/models", "/v1/models/"):
                return self._json(200, dict(object="list", data=[model_json(state)]))
            if path == "/v1/models/" + cfg.model_id:
                return self._json(200, model_json(state))
            return self._error(404, "not found", "not_found")

        def do_POST(self):
            why = self._refused(True)
            if why:
                self._drain()
                return self._error(403, "forbidden: " + why, "forbidden")
            if self._unauthorized():
                self._drain()
                return
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._drain()
                return self._error(404, "not found", "not_found")
            runner = state.runner
            if runner is None:
                self._drain()
                return self._error(503, "the model is still loading", "unavailable")
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._error(413, "request body too large or length missing", "invalid_request_error", close=True)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                params = parse_chat_request(body, cfg, images=getattr(runner, "vision", None) is not None)
            except json.JSONDecodeError:
                return self._error(400, "the request body is not valid JSON")
            except RequestError as ex:
                return self._error(400, str(ex))
            self._chat(runner, body, params)

        def _chat(self, runner: ChatRunner, body: dict, params: ChatParams):
            if state.sched is not None:
                return self._chat_parallel(runner, body, params)
            # One request at a time.  With image input, prepare_checked also reads and encodes the images on the CPU, inside this
            # lock: while an image is encoded (about 15 s at the default limit of 1,024 image tokens on a 16-core CPU) no other
            # request is served and nothing is sent to this client yet.
            with state.gen_lock:
                try:
                    prepared = runner.prepare_checked(params)        # before any header: a request that cannot be served gets a plain 400
                except RequestError as ex:
                    return self._error(400, str(ex), err_code=getattr(ex, "code", None))
                except Exception as ex:                          # noqa: BLE001 - e.g. the image encoder failed: answer, do not drop
                    return self._error(500, str(ex)[:500] or type(ex).__name__, "server_error")
                self._reply(runner, body, params, prepared)

        def _chat_parallel(self, runner: ChatRunner, body: dict, params: ChatParams):
            """--parallel N: the scheduler generates; the template, the tokens and the length check are made here, outside any lock
            (only the images are read and encoded one request at a time).  A request that waits or runs without a stream is cancelled
            when its client goes away (the socket is looked at once a second)."""
            try:
                if has_images(params):
                    with state.image_lock:
                        prepared = runner.prepare_checked(params)
                else:
                    prepared = runner.prepare_checked(params)
            except RequestError as ex:
                return self._error(400, str(ex), err_code=getattr(ex, "code", None))
            except Exception as ex:                              # noqa: BLE001
                return self._error(500, str(ex)[:500] or type(ex).__name__, "server_error")
            self._reply(runner, body, params, prepared, sched=state.sched)

        def _reply(self, runner: ChatRunner, body: dict, params: ChatParams, prepared, sched=None):
            """Generate and answer.  ``sched`` (several requests at once): a stream's headers go out with its first piece of text (or
            at its end), so that a request that cannot be queued still gets a plain error status (503 when the queue is full)."""
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(body.get("stream"))
            started = [False]

            def chunk(delta=None, finish=None, usage=None, timings=None):
                o = dict(id=cid, object="chat.completion.chunk", created=created, model=cfg.model_id,
                         choices=[] if usage else [dict(index=0, delta=delta or {}, finish_reason=finish)])
                if usage:
                    o["usage"] = usage
                if timings:
                    o["timings"] = timings
                self.wfile.write(b"data: " + json.dumps(o, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.wfile.flush()

            def start():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    chunk(delta=dict(role="assistant"))

            eng = None
            closed = [False]
            if sched is not None:
                from .sched import ScheduledEngine
                # a running stream finds out by writing; until its first piece of text (and for a reply without a stream) the
                # socket is looked at
                eng = ScheduledEngine(sched, client_gone=lambda: not started[0] and self._client_gone())

            def generated(u):
                """The request generates nothing more: its view of the scheduler is closed before the end of the answer goes out (the
                client may send its next turn the moment it has the answer).  An answer with tool calls keeps its stream for that turn
                (the client sends the results at once), about as long as reading the conversation again would take: see
                ScheduledEngine.close and Scheduler.turn_keep_s."""
                if eng is not None and not closed[0]:
                    closed[0] = True
                    eng.close(next_turn=u.finish == "tool_calls")
            try:
                if stream:
                    if eng is None:
                        start()
                    n_calls = [0]

                    def emit(kind, item):
                        start()
                        if kind == "tool_call":
                            chunk(delta={"tool_calls": [tool_call_json(item, n_calls[0])]})
                            n_calls[0] += 1
                        else:
                            chunk(delta={"reasoning_content" if kind == "reasoning" else "content": item})

                    u = runner.run(params, emit, prepared) if eng is None else runner.run(params, emit, prepared, eng=eng)
                    generated(u)
                    start()
                    # the timings go with the last chunk, as with the llama.cpp server: the usage chunk when it is asked for,
                    # else the chunk with the finish reason
                    if (body.get("stream_options") or {}).get("include_usage"):
                        chunk(delta={}, finish=u.finish)
                        chunk(usage=usage_json(u), timings=timings_json(u))
                    else:
                        chunk(delta={}, finish=u.finish, timings=timings_json(u))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    parts = {"reasoning": [], "content": [], "tool_call": []}

                    def collect(kind, item):
                        parts[kind].append(item)
                    u = runner.run(params, collect, prepared) if eng is None else runner.run(params, collect, prepared, eng=eng)
                    generated(u)
                    msg = dict(role="assistant", content="".join(parts["content"]))
                    if parts["reasoning"]:
                        msg["reasoning_content"] = "".join(parts["reasoning"])
                    if parts["tool_call"]:
                        msg["tool_calls"] = [tool_call_json(c) for c in parts["tool_call"]]
                        if not msg["content"]:
                            msg["content"] = None
                    self._json(200, dict(id=cid, object="chat.completion", created=created, model=cfg.model_id,
                                         choices=[dict(index=0, message=msg, finish_reason=u.finish)], usage=usage_json(u),
                                         timings=timings_json(u)))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True                   # the client went away: the engine was already stopped
            except Exception as ex:                              # noqa: BLE001
                full = getattr(ex, "retry_after", None) is not None          # the scheduler's queue is full (QueueFull)
                code = 400 if isinstance(ex, RequestError) else (503 if full else 500)
                msg = str(ex)[:500] or type(ex).__name__
                try:
                    if stream and started[0]:
                        self.wfile.write(b"data: " + json.dumps(dict(error=dict(message=msg, type="server_error", code=code))).encode() + b"\n\n")
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        self.close_connection = True
                    elif full:
                        self._error(503, msg, "unavailable", headers={"Retry-After": str(int(ex.retry_after))})
                    else:
                        self._error(code, msg, "server_error" if code == 500 else "invalid_request_error")
                except Exception:                                # noqa: BLE001
                    self.close_connection = True
            finally:
                if eng is not None and not closed[0]:              # an error: nothing kept
                    closed[0] = True
                    eng.close()

    return Handler


def bind(state: ApiState) -> SingleServer:
    """Bind the port first (before loading the model): a second start finds the port taken and stops here."""
    cls = SingleServer
    if ":" in state.cfg.host:                                # an IPv6 address such as ::1
        class cls(SingleServer):                             # noqa: N801
            address_family = socket.AF_INET6
    return cls((state.cfg.host, state.cfg.port), make_handler(state))
