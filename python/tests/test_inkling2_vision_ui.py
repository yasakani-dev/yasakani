# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 Mini image input, the user interface (no GPU, no model weights): the server's content parts and every refusal with its
status and message, /v1/models and the usage block, `yasakani chat --image`, `yasakani info`, the options an Inkling-2 folder does
not take, and the image button of the quickstart page.  The engine is a stand-in; the image checks are those of the package.

Optional (else those tests skip):
  YASAKANI_TEST_INK2_SOURCE         the downloaded hf-internal-testing/tiny-inkling2-mini folder (tokenizer and chat template of
                                    Inkling-2 Mini): the prompt ids and the context check
  YASAKANI_TEST_INK2_VISION_REF     a folder with images/t1.png and golden/t1_prompt_ids.json (ids of the official processor)
"""
import argparse
import base64
import contextlib
import http.client
import io
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import cli
from yasakani import inkling2 as ink
from yasakani import inkling2_cli
from yasakani import inkling2_vision as V
from yasakani._native import YasakaniError
from yasakani.chat import ChatParams, RequestError, Usage, clean_messages
from yasakani.server import ApiState, ServerConfig, bind, parse_chat_request, usage_json

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

SOURCE = os.environ.get("YASAKANI_TEST_INK2_SOURCE")
HAVE_SOURCE = bool(SOURCE) and (Path(SOURCE) / "tokenizer.json").is_file() and (Path(SOURCE) / "chat_template.jinja").is_file()
REF = os.environ.get("YASAKANI_TEST_INK2_VISION_REF")
PAGE = Path(__file__).resolve().parents[2] / "quickstart" / "chat.html"
_FMT = {}


def fmt():
    if "f" not in _FMT:
        _FMT["f"] = ink.Ink2Formatter(SOURCE) if HAVE_SOURCE else None
    return _FMT["f"]


def png(w=64, h=48, color=(200, 40, 40), mode="RGB") -> bytes:
    b = io.BytesIO()
    Image.new(mode, (w, h), color if mode == "RGB" else 0).save(b, "PNG")
    return b.getvalue()


def gif2() -> bytes:
    """An animated GIF of two frames."""
    b = io.BytesIO()
    a, c = Image.new("RGB", (16, 16), (255, 0, 0)), Image.new("RGB", (16, 16), (0, 0, 255))
    a.save(b, "GIF", save_all=True, append_images=[c], duration=100, loop=0)
    return b.getvalue()


def uri(data: bytes, kind="png") -> str:
    return "data:image/%s;base64,%s" % (kind, base64.b64encode(data).decode("ascii"))


def img(url):
    return {"type": "image_url", "image_url": {"url": url}}


def text(s):
    return {"type": "text", "text": s}


def state(cap=1024, req=None):
    return ink.Ink2VisionState(kind="yskn", max_image_tokens=cap, max_request_tokens=req or 8 * cap, memory=(1, 2, 3),
                               cache=V.FeatureCache(1 << 20))


class FakeRunner(ink.Ink2Runner):
    """The package's runner on a stand-in engine: prepare and its checks are the real ones; run records what it got."""

    def __init__(self, vision=None, max_ctx=32768, problem=V.NO_TOWER):
        eng = SimpleNamespace(vision=vision, vision_problem=problem, max_ctx=max_ctx, _cancel=threading.Event())
        super().__init__(eng, fmt(), SimpleNamespace(stop_ids=[1]))
        self.got = []

    def run(self, p, emit, prepared=None, eng=None):
        prepared = prepared if prepared is not None else self.prepare_checked(p)
        ids = prepared[0]
        self.got.append(ids)
        emit("content", "ok")
        u = Usage(prompt_tokens=len(ids), completion_tokens=1)
        images = getattr(ids, "images", ())
        if images:
            u.vision = dict(images=len(images), image_tokens=int(ids.image_tokens), encoded=len(images), from_cache=0, reused=0)
        return u


@unittest.skipIf(Image is None, "Pillow is not installed")
class ServerTests(unittest.TestCase):
    """The HTTP server with an Inkling-2 runner: what is accepted, what is refused (status and message)."""

    def start(self, runner):
        cfg = ServerConfig(port=0, model_id="inkling-2-mini")
        cfg.keep_tool_ids = True
        cfg.vision_parts = True
        cfg.vision_problem = getattr(runner.engine, "vision_problem", "") if runner.vision is None else ""
        self.state = ApiState(cfg)
        self.state.runner = runner
        self.httpd = bind(self.state)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        threading.Thread(target=self.httpd.serve_forever, kwargs=dict(poll_interval=0.05), daemon=True).start()
        return runner

    def req(self, method, path, body=None):
        port = self.httpd.server_address[1]
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        h = {"Host": "127.0.0.1:%d" % port}
        data = None
        if body is not None:
            h["Content-Type"] = "application/json"
            data = json.dumps(body)
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        out = r.read()
        c.close()
        return r.status, json.loads(out)

    def chat(self, messages, **kw):
        return self.req("POST", "/v1/chat/completions", dict(messages=messages, max_tokens=4, **kw))

    def refused(self, messages, pattern, code=None):
        st, d = self.chat(messages)
        self.assertEqual(st, 400, d)
        self.assertRegex(d["error"]["message"], pattern)
        if code:
            self.assertEqual(d["error"].get("code"), code)
        return d

    def test_models_says_whether_images_are_read(self):
        self.start(FakeRunner(state()))
        st, d = self.req("GET", "/v1/models")
        self.assertEqual((st, d["data"][0]["id"], d["data"][0]["vision"]), (200, "inkling-2-mini", True))
        self.assertIs(self.req("GET", "/v1/models/inkling-2-mini")[1]["vision"], True)
        self.state.runner = FakeRunner(None)
        self.assertIs(self.req("GET", "/v1/models")[1]["data"][0]["vision"], False)

    def test_refusals_with_image_input_on(self):
        self.start(FakeRunner(state()))
        a = uri(png())
        cases = [
            ([{"role": "assistant", "content": [img(a)]}, {"role": "user", "content": "x"}],
             r"^messages\[0\]: images are supported only in user and tool messages$"),
            ([{"role": "system", "content": [img(a)]}, {"role": "user", "content": "x"}], r"^messages\[0\]: images are supported only in user"),
            ([{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}, text("x")]}],
             r"^messages\[0\]: video is not supported \(only still images\)$"),
            ([{"role": "user", "content": [{"type": "video", "video": "x"}]}], r"video is not supported"),
            ([{"role": "user", "content": [img("https://example.com/a.png"), text("x")]}],
             r"^messages\[0\]: image input: remote image URLs are not supported"),
            ([{"role": "user", "content": [img("C:/pictures/a.png"), text("x")]}],
             r"^messages\[0\]: image input: only data:image URIs are enabled; local files require --vision-allow-local-files"),
            ([{"role": "user", "content": [img("data:image/png;base64,@@@@"), text("x")]}], r"image input: the data:image URI is not valid base64"),
            ([{"role": "user", "content": [img(uri(b"not an image")), text("x")]}], r"^messages\[0\]: image input: the data is not a still image"),
            ([{"role": "user", "content": [img(uri(gif2(), "gif")), text("x")]}], r"image input: " + re.escape(V.STILL_ONLY)),
            ([{"role": "user", "content": [img(uri(png(402, 2))), text("x")]}], r"image input: image aspect ratio exceeds 200:1"),
            ([{"role": "user", "content": [img(a)] * 17 + [text("x")]}], r"^image input: at most 16 images are supported per request \(got 17\)$"),
            ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}], r"image_url needs a nonempty url string"),
            ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": a, "detail": "huge"}}]}], r"invalid image detail"),
        ]
        for messages, pattern in cases:
            with self.subTest(pattern=pattern):
                self.refused(messages, pattern)

    def test_too_many_pixels(self):
        self.start(FakeRunner(state()))
        big = io.BytesIO()
        Image.new("1", (8000, 5001)).save(big, "PNG")
        self.refused([{"role": "user", "content": [img(uri(big.getvalue())), text("x")]}], r"image input: image exceeds 40 million pixels")

    def test_too_many_image_tokens_per_request(self):
        self.start(FakeRunner(state(cap=64, req=100)))
        a = uri(png(400, 400))
        self.refused([{"role": "user", "content": [img(a), img(a), text("x")]}],
                     r"^image input: the images of this request take 128 image tokens; the limit per request is 100$")

    def test_image_input_off_names_the_reason(self):
        for problem in (ink.VISION_OFF, V.NO_TOWER, ink.VISION_NO_LIBRARY, ink.VISION_NO_GPU):
            with self.subTest(problem=problem):
                if getattr(self, "httpd", None):
                    self.httpd.shutdown()
                    self.httpd.server_close()
                self.start(FakeRunner(None, problem=problem))
                self.refused([{"role": "user", "content": [img(uri(png())), text("x")]}],
                             "^image input is not available: " + re.escape(problem) + "$")
                self.refused([{"role": "user", "content": [{"type": "input_image", "image_url": uri(png())}]}],
                             "^image input is not available: ")
                if HAVE_SOURCE:                          # a text request needs the chat template of the tiny folder
                    st, d = self.chat([{"role": "user", "content": [text("plain text parts still work")]}])
                    self.assertEqual(st, 200, d)

    @unittest.skipUnless(HAVE_SOURCE, "set YASAKANI_TEST_INK2_SOURCE to the tiny-inkling2-mini folder")
    def test_context_too_short_names_the_image_tokens(self):
        self.start(FakeRunner(state(), max_ctx=200))
        d = self.refused([{"role": "user", "content": [img(uri(png(640, 480))), text("Describe this image.")]}],
                         r"^the input is too long: \d+ tokens \(192 of them image tokens of 1 image\) exceeds the model's maximum context "
                         r"length of 200 tokens", code="context_length_exceeded")
        self.assertIn("image tokens", d["error"]["message"])

    @unittest.skipUnless(HAVE_SOURCE, "set YASAKANI_TEST_INK2_SOURCE to the tiny-inkling2-mini folder")
    def test_placeholder_in_text_is_refused(self):
        self.start(FakeRunner(state()))
        self.refused([{"role": "user", "content": [img(uri(png())), text("x <|unused_200054|> y")]}], r"image placeholder")

    @unittest.skipUnless(HAVE_SOURCE, "set YASAKANI_TEST_INK2_SOURCE to the tiny-inkling2-mini folder")
    def test_one_image_reaches_the_runner_and_counts_as_prompt_tokens(self):
        r = self.start(FakeRunner(state()))
        st, d = self.chat([{"role": "user", "content": [img(uri(png(640, 480))), text("Describe this image.")]}])
        self.assertEqual(st, 200, d)
        ids = r.got[-1]
        self.assertEqual(ids.count(V.IMAGE_TOKEN_ID), 192)                   # 12 x 16 tiles of 40 px
        self.assertEqual(len(ids.images), 1)
        self.assertEqual(d["usage"]["prompt_tokens"], len(ids))
        self.assertEqual(d["usage"]["prompt_tokens_details"]["image_tokens"], 192)
        self.assertEqual(d["usage"]["prompt_tokens_details"]["images"], 1)
        # in a tool message too, and the responses form input_image
        st, d = self.chat([{"role": "user", "content": "look"},
                           {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                               "function": {"name": "shot", "arguments": "{}"}}]},
                           {"role": "tool", "tool_call_id": "c1", "content": [{"type": "input_image", "image_url": uri(png())}]}],
                          tools=[{"type": "function", "function": {"name": "shot", "parameters": {"type": "object"}}}])
        self.assertEqual(st, 200, d)
        self.assertEqual(len(r.got[-1].images), 1)
        # text only: no image fields in usage, plain ids
        st, d = self.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(st, 200, d)
        self.assertNotIn("image_tokens", d["usage"]["prompt_tokens_details"])
        self.assertFalse(hasattr(r.got[-1], "images") and r.got[-1].images)

    @unittest.skipUnless(HAVE_SOURCE and REF and (Path(REF or ".") / "golden" / "t1_prompt_ids.json").is_file(),
                         "set YASAKANI_TEST_INK2_SOURCE and YASAKANI_TEST_INK2_VISION_REF")
    def test_request_builds_the_official_ids(self):
        r = self.start(FakeRunner(state()))
        t1 = (Path(REF) / "images" / "t1.png").read_bytes()
        st, d = self.chat([{"role": "user", "content": [img(uri(t1)), text("Describe this image.")]}])
        self.assertEqual(st, 200, d)
        g = json.loads((Path(REF) / "golden" / "t1_prompt_ids.json").read_text(encoding="utf-8"))
        want = [int(x) for x in (g["ids"] if isinstance(g, dict) else g)]
        self.assertEqual(list(r.got[-1]), want)
        self.assertEqual(d["usage"]["prompt_tokens"], 215)


class MessageRulesTests(unittest.TestCase):
    """clean_messages: the Inkling-2 rules only with vision_parts; the Qwen3.8 rules unchanged."""

    def test_qwen_rules_unchanged(self):
        a = img("data:image/png;base64,AAAA")
        with self.assertRaisesRegex(RequestError, r"only text content is supported"):
            clean_messages([{"role": "user", "content": [a]}])
        with self.assertRaisesRegex(RequestError, r"^messages\[0\]: images are supported only in user messages$"):
            clean_messages([{"role": "tool", "content": [a]}], images=True)
        with self.assertRaisesRegex(RequestError, r"only text content is supported"):
            clean_messages([{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}], images=True)
        self.assertEqual(clean_messages([{"role": "user", "content": [a]}], images=True)[0]["content"], [a])

    def test_inkling_parts_are_kept_in_order(self):
        parts = [text("a"), img("data:image/png;base64,AAAA"), {"type": "input_image", "image_url": "data:image/png;base64,BBBB"}, text("b")]
        out = clean_messages([{"role": "tool", "content": parts}], images=True, vision_parts=True)
        self.assertEqual([p["type"] for p in out[0]["content"]], ["text", "image_url", "input_image", "text"])
        cfg = ServerConfig()
        cfg.vision_parts, cfg.vision_problem = True, ink.VISION_OFF
        with self.assertRaisesRegex(RequestError, r"^image input is not available: image input is off \(--vision off\)$"):
            parse_chat_request({"messages": [{"role": "user", "content": parts}]}, cfg, images=False)

    def test_usage_without_images(self):
        d = usage_json(Usage(prompt_tokens=5, completion_tokens=2))
        self.assertEqual(d["prompt_tokens_details"], {"cached_tokens": 0})


class ChatImageTests(unittest.TestCase):
    """yasakani chat --image PATH for an Inkling-2 folder."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="yskn_ui_t_"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)

    def test_read_image_file_refusals(self):
        with self.assertRaisesRegex(YasakaniError, r"^image file not found: "):
            inkling2_cli.read_image_file(self.tmp / "missing.png")
        with self.assertRaisesRegex(YasakaniError, r"^image file not found: "):       # a folder is not a file
            inkling2_cli.read_image_file(self.tmp)
        (self.tmp / "empty.png").write_bytes(b"")
        with self.assertRaisesRegex(YasakaniError, r"is empty"):
            inkling2_cli.read_image_file(self.tmp / "empty.png")
        with open(self.tmp / "big.png", "wb") as f:
            f.truncate(V.MAX_IMAGE_BYTES + 1)
        with self.assertRaisesRegex(YasakaniError, r"is larger than 24 MiB"):
            inkling2_cli.read_image_file(self.tmp / "big.png")
        (self.tmp / "locked.png").write_bytes(b"x")
        with mock.patch.object(Path, "open", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaisesRegex(YasakaniError, r"^cannot read the image file .*Permission denied"):
                inkling2_cli.read_image_file(self.tmp / "locked.png")

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_prepare_image(self):
        r = FakeRunner(state(cap=64))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            part = inkling2_cli.prepare_image(r, png(640, 480), "t.png")
        self.assertEqual(part["type"], "image")
        self.assertTrue(part["image"].shrunk)
        self.assertLessEqual(part["image"].tokens, 64)
        self.assertIn("t.png: image 640x480 shrunk to", err.getvalue())
        with self.assertRaisesRegex(YasakaniError, r"^image bad.png: the data is not a still image"):
            inkling2_cli.prepare_image(r, b"nope", "bad.png")
        with self.assertRaisesRegex(YasakaniError, r"^--image needs image input, which is off: image input is off \(--vision off\)$"):
            inkling2_cli.prepare_image(FakeRunner(None, problem=ink.VISION_OFF), png(), "t.png")

    def args(self, *extra):
        a = cli.build_parser().parse_args(["chat", "--model-dir", str(self.tmp)] + list(extra))
        a.model_dir_path = self.tmp
        return a

    def test_missing_file_fails_before_the_load(self):
        with mock.patch.object(inkling2_cli, "open_stack") as op:
            with self.assertRaisesRegex(YasakaniError, r"image file not found"):
                inkling2_cli.cmd_chat(self.args("--image", str(self.tmp / "nope.png"), "--ask", "q"))
            op.assert_not_called()

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_images_go_first_in_the_given_order(self):
        (self.tmp / "a.png").write_bytes(png(80, 40, (255, 0, 0)))
        (self.tmp / "b.png").write_bytes(png(40, 80, (0, 0, 255)))
        r = FakeRunner(state())
        seen = []
        r.run = lambda p, emit, prepared=None, eng=None: (seen.append(p.messages), emit("content", "ok"), Usage(prompt_tokens=3))[2]
        pack = SimpleNamespace(model_id="inkling-2-mini", sampling=dict(temperature=1.0, top_p=1.0, top_k=0))
        eng = SimpleNamespace(close=lambda: None, reset=lambda: None, vision_problem="")
        with mock.patch.object(inkling2_cli, "open_stack", return_value=(pack, eng, None, r)) as op, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = inkling2_cli.cmd_chat(self.args("--image", str(self.tmp / "a.png"), "--image", str(self.tmp / "b.png"),
                                                 "--ask", "what is this?", "--vision", "on"))
        self.assertEqual(rc, 0)
        self.assertEqual(op.call_args.kwargs, dict(warm=True, images=True, image_option=True))
        content = seen[0][-1]["content"]
        self.assertEqual([q["type"] for q in content], ["image", "image", "text"])
        self.assertEqual([(q["image"].width, q["image"].height) for q in content[:2]], [(80, 40), (40, 80)])
        self.assertEqual(content[2]["text"], "what is this?")
        self.assertEqual(inkling2_cli.user_message("hi", []), {"role": "user", "content": "hi"})


class OptionTests(unittest.TestCase):
    def test_parser(self):
        a = cli.build_parser().parse_args(["chat", "--image", "a.png", "--image", "b.png", "--vision", "off",
                                           "--vision-max-image-tokens", "256"])
        self.assertEqual((a.image, a.vision, a.vision_max_image_tokens), (["a.png", "b.png"], "off", 256))
        a = cli.build_parser().parse_args(["serve", "--vision", "on", "--vision-max-request-image-tokens", "4096"])
        self.assertEqual((a.vision, a.vision_max_request_image_tokens), ("on", 4096))
        self.assertIsNone(cli.build_parser().parse_args(["serve"]).vision)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["bench", "--image", "a.png"])

    def test_vision_options(self):
        a = argparse.Namespace(ink2_images=False, vision="on")
        self.assertEqual(inkling2_cli.vision_options(a), ("off", None, None))        # selfcheck and bench: never loaded
        a = argparse.Namespace(ink2_images=True, vision=None, vision_max_image_tokens=1024, vision_max_request_image_tokens=None)
        with mock.patch.dict(os.environ, {"YASAKANI_INK2_VISION": "off"}):
            self.assertEqual(inkling2_cli.vision_options(a), ("off", 1024, 8192))
        a.vision_max_image_tokens = 9000
        with self.assertRaisesRegex(YasakaniError, r"--vision-max-image-tokens"):
            inkling2_cli.vision_options(a)

    def test_qwen_image_options_are_refused(self):
        for key, value in (("mmproj", "m.gguf"), ("llama_dir", "x"), ("vision_model", "q.gguf")):
            with self.subTest(key=key):
                with self.assertRaisesRegex(YasakaniError, r"are for Qwen3.8 model folders; Inkling-2 Mini reads images"):
                    inkling2_cli.refuse_unsupported(argparse.Namespace(**{key: value}))
        with self.assertRaisesRegex(YasakaniError, r"--vision-threads"):
            inkling2_cli.refuse_unsupported(argparse.Namespace(vision_threads=4))
        with self.assertRaisesRegex(YasakaniError, r"--vision-cache"):
            inkling2_cli.refuse_unsupported(argparse.Namespace(vision_cache="d"))
        with self.assertRaisesRegex(YasakaniError, r"--image is taken by yasakani chat only"):
            inkling2_cli.refuse_unsupported(argparse.Namespace(image=["a.png"]))
        inkling2_cli.refuse_unsupported(argparse.Namespace(image=["a.png"]), images=True)
        with mock.patch.dict(os.environ, {"YASAKANI_MMPROJ": "m.gguf"}), contextlib.redirect_stderr(io.StringIO()):
            inkling2_cli.refuse_unsupported(argparse.Namespace(mmproj="m.gguf"))     # from the environment: a note, not an error

    def test_vision_flag_refused_for_qwen_folders(self):
        a = cli.build_parser().parse_args(["serve", "--vision", "on"])
        with mock.patch.object(cli, "_model_dir", return_value="x"), \
                mock.patch.object(cli.ModelPack, "load", return_value=SimpleNamespace(is_hf_inkling2=False)):
            with self.assertRaisesRegex(YasakaniError, r"^--vision is for Inkling-2 Mini model folders"):
                cli._open_stack(a)

    def test_qwen_chat_takes_no_image_options(self):
        # a Qwen3.8 chat without images must start as before even with the Qwen image variables of serve set
        env = {"YASAKANI_MMPROJ": "D:/nope/mmproj.gguf", "YASAKANI_LLAMA_DIR": "D:/nope", "YASAKANI_VISION_MODEL": "D:/nope/q.gguf"}
        with mock.patch.dict(os.environ, env):
            a = cli.build_parser().parse_args(["chat", "--model-dir", "x"])
        for key in ("mmproj", "llama_dir", "vision_model", "vision_threads"):
            self.assertFalse(hasattr(a, key), key)
        self.assertIsNone(a.vision_cache)        # --vision-cache is read for a published Inkling-2 Mini folder only
        self.assertIsNone(cli._vision(a))
        cli._refuse_chat_images(a)
        for extra in (["--image", "a.png"], ["--vision", "on"], ["--vision-max-image-tokens", "256"], ["--vision-cache", "c"]):
            with self.subTest(extra=extra):
                a = cli.build_parser().parse_args(["chat", "--model-dir", "x"] + extra)
                with mock.patch.object(cli, "_open_stack", side_effect=AssertionError("loaded")), \
                        self.assertRaisesRegex(YasakaniError, r"image input in yasakani chat is for Inkling-2 Mini"):
                    cli.cmd_chat(a)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["chat", "--mmproj", "m.gguf"])

    def test_info_line(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "inkling_2_mini_vision.yskn").write_bytes(b"x" * 10)
            pack = SimpleNamespace(vision_entry={"file": "inkling_2_mini_vision.yskn"}, vision_file=root / "inkling_2_mini_vision.yskn",
                                   has_vision=True, weights=root / "w.yskn")
            self.assertRegex(inkling2_cli.vision_line(pack), r"^image tower inkling_2_mini_vision.yskn ")
            pack.vision_file = root / "gone.yskn"
            self.assertIn("missing", inkling2_cli.vision_line(pack))
            self.assertIn("text only", inkling2_cli.vision_line(SimpleNamespace(vision_entry={}, has_vision=False)))


@unittest.skipUnless(PAGE.is_file(), "quickstart/chat.html not found")
class PageTests(unittest.TestCase):
    """The image button of the quickstart page (string-level, as the other page tests)."""

    def setUp(self):
        self.page = PAGE.read_text(encoding="utf-8")

    def test_button_only_when_the_model_reads_images(self):
        p = self.page
        self.assertRegex(p, r'<button class="attach" id="attach" type="button" hidden></button>')
        self.assertIn('<input type="file" id="pick" accept="image/*" multiple hidden>', p)
        self.assertIn("$('attach').hidden = !vision;", p)
        self.assertIn("vision = !!(first && first.vision === true);", p)        # only an explicit true from /v1/models
        self.assertIn("if (up) await loadVision(); else { vision = false; clearImages(); }", p)
        self.assertIn("const images = vision ? attached.slice() : [];", p)
        for fn in ("addImages", "$('box').addEventListener('paste'", "$('composer').addEventListener('drop'"):
            self.assertIn(fn, p)
        self.assertIn("if (!vision || !files) return;", p)

    def test_size_limits_and_messages(self):
        p = self.page
        self.assertIn("const IMG_MAX = 11 * 1024 * 1024, IMG_COUNT = 4, BODY_MAX = 16 * 1024 * 1024;", p)
        self.assertLess(4 * ((11 * 1024 * 1024 + 2) // 3), 16 * 1024 * 1024)      # one image in base64 fits the launcher's body limit
        ja, en = p.index("  ja: {"), p.index("  en: {")
        for part in (p[ja:en], p[en:]):
            for key in ("attach", "attach_title", "img_remove", "img_big", "img_type", "img_many", "img_read", "img_total", "img_sent"):
                self.assertRegex(part, r"\b%s: '" % key)
        self.assertIn("readAsDataURL", p)                                       # the original bytes, no canvas re-encode
        self.assertNotIn("toDataURL", p)
        self.assertNotIn("<img", p)                                             # the page's CSP has no img-src: names, not thumbnails

    def test_launcher_body_limit_matches(self):
        qs = (PAGE.parent / "quickstart.py").read_text(encoding="utf-8")
        self.assertIn("MAX_BODY = 16 << 20", qs)


if __name__ == "__main__":
    unittest.main()
