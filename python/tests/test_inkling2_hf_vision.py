# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Inkling-2 native vision tests: message splitting, RGBA checkerboard compositing, fitting, expansion, caching, limits, and server."""
from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests import support
from yasakani import cli
from yasakani.chat import ChatRunner, ContextLengthError, RequestError
from yasakani.runtime import ModelPack
from yasakani.server import ApiState, ServerConfig, bind
from yasakani.textio import ChatFormatter
from yasakani.vision import (
    DEFAULT_IMAGE_TOKENS,
    INK2_HIDDEN,
    INK2_IMAGE_TOKEN,
    INK2_IMAGE_TOKEN_ID,
    MARKER,
    MAX_IMAGE_TOKENS,
    MAX_IMAGES,
    Ink2NativeVision,
    _fit_ink2_image,
    build_ink2_visual_prompt,
    ink2_grid_size,
    read_image,
    split_images,
    validate_ink2_features,
    validate_vision_limits,
)

INK2_DIR = Path("/home/au/models/Inkling-2-Mini")


def _need_pillow():
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("Pillow is not installed")


def _png_bytes(size, mode="RGB", color=(20, 120, 220)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _png_uri(size, mode="RGB", color=(20, 120, 220)):
    return "data:image/png;base64," + base64.b64encode(_png_bytes(size, mode, color)).decode("ascii")


class FakeInk2Formatter:
    hf_inkling2 = True

    def token_id(self, name: str) -> int:
        if name != INK2_IMAGE_TOKEN:
            raise KeyError(name)
        return INK2_IMAGE_TOKEN_ID

    def encode(self, text: str) -> np.ndarray:
        if text == INK2_IMAGE_TOKEN:
            return np.array([INK2_IMAGE_TOKEN_ID], dtype=np.int32)
        return np.array([1], dtype=np.int32)


class FakeInk2Engine:
    def __init__(self, hidden_size: int = INK2_HIDDEN, max_ctx: int = 8192):
        self.pack = SimpleNamespace(is_hf_inkling2=True, hidden_size=hidden_size, weights=Path(__file__), root=Path(__file__).parent)
        self.supports_ink2_vision = True
        self.supports_visual = True
        self.max_ctx = max_ctx
        self.encode_calls = []

    def encode_ink2_images(self, rgbs):
        self.encode_calls.append([tuple(r.shape) for r in rgbs])
        feats, grids = [], []
        for idx, rgb in enumerate(rgbs):
            rows, cols = ink2_grid_size(rgb.shape[1], rgb.shape[0])
            grids.append([rows, cols])
            val = 0x3F80 + ((int(rgb[0, 0, 0]) + idx) & 0x3F)
            feats.append(np.full((rows * cols, self.pack.hidden_size), val, dtype=np.uint16))
        return np.concatenate(feats, axis=0), np.asarray(grids, dtype=np.int32)


class Inkling2VisionSplitTests(unittest.TestCase):
    def test_split_images_inkling2_user_and_tool(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "check"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
            ]},
        ]
        out, sources = split_images(msgs, is_inkling2=True)
        self.assertEqual(sources, ["data:image/png;base64,AAA", "data:image/png;base64,BBB"])
        self.assertEqual(out[0]["content"], [{"type": "text", "text": "check"}, {"type": "image"}])
        self.assertEqual(out[1]["content"], [{"type": "image"}])
        self.assertEqual(msgs[0]["content"][1]["type"], "image_url")

    def test_split_images_default_keeps_qwen_marker_behavior(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:1"}}]}]
        out, sources = split_images(msgs)
        self.assertEqual(sources, ["data:1"])
        self.assertEqual(out[0]["content"], [{"type": "text", "text": MARKER}])
        with self.assertRaises(RequestError):
            split_images([{"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:1"}}]}])

    def test_split_images_rejects_assistant_and_too_many(self):
        with self.assertRaises(RequestError):
            split_images([{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:1"}}]}],
                         is_inkling2=True)
        with self.assertRaises(RequestError):
            split_images([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:1"}}] * (MAX_IMAGES + 1)}],
                         is_inkling2=True)


class Inkling2ImagePrepTests(unittest.TestCase):
    def test_rgba_checkerboard_compositing(self):
        _need_pillow()
        from PIL import Image
        im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        out_rgb = np.asarray(read_image(uri, rgba_checkerboard=True))
        self.assertEqual(out_rgb.shape, (16, 16, 3))
        self.assertEqual(tuple(out_rgb[0, 0].tolist()), (204, 204, 204))
        self.assertEqual(tuple(out_rgb[0, 8].tolist()), (153, 153, 153))
        self.assertEqual(tuple(out_rgb[8, 0].tolist()), (153, 153, 153))
        self.assertEqual(tuple(out_rgb[8, 8].tolist()), (204, 204, 204))

    def test_fit_ink2_image_respects_budget(self):
        _need_pillow()
        from PIL import Image
        small = Image.new("RGB", (333, 217))
        self.assertIs(_fit_ink2_image(small, DEFAULT_IMAGE_TOKENS), small)

        big = Image.new("RGB", (1920, 1080))
        fitted = _fit_ink2_image(big, 1024)
        r, c = ink2_grid_size(fitted.width, fitted.height)
        self.assertLessEqual(r * c, 1024)
        self.assertGreater(r * c, 900)

    def test_build_ink2_visual_prompt_multiframe(self):
        img1 = {"features": np.full((6, INK2_HIDDEN), 0x3F80, dtype=np.uint16), "grid": np.array([2, 3], np.int32)}
        img2 = {"features": np.full((4, INK2_HIDDEN), 0x4000, dtype=np.uint16), "grid": np.array([2, 2], np.int32)}
        prompt = [10, INK2_IMAGE_TOKEN_ID, 20, INK2_IMAGE_TOKEN_ID, 30]
        ids, vis = build_ink2_visual_prompt(prompt, [img1, img2], INK2_IMAGE_TOKEN_ID, max_tokens=100, max_ctx=164)
        self.assertEqual(ids.tolist(), [10] + [INK2_IMAGE_TOKEN_ID] * 6 + [20] + [INK2_IMAGE_TOKEN_ID] * 4 + [30])
        self.assertEqual(vis["slots"].tolist(), [1, 2, 3, 4, 5, 6, 8, 9, 10, 11])
        self.assertEqual(vis["n_prompt"], 13)
        self.assertEqual(vis["features"].shape, (10, INK2_HIDDEN))

        with self.assertRaises(ContextLengthError):
            build_ink2_visual_prompt(prompt, [img1, img2], INK2_IMAGE_TOKEN_ID, max_tokens=13, max_ctx=77)
        with self.assertRaises(ValueError):
            build_ink2_visual_prompt([10, 20], [img1], INK2_IMAGE_TOKEN_ID)

    def test_validate_ink2_features_rejects_bad_inputs(self):
        bad_nan = {"features": np.full((4, INK2_HIDDEN), 0x7F80, dtype=np.uint16), "grid": np.array([2, 2], np.int32)}
        with self.assertRaises(ValueError):
            validate_ink2_features(bad_nan)
        bad_shape = {"features": np.full((3, INK2_HIDDEN), 0x3F80, dtype=np.uint16), "grid": np.array([2, 2], np.int32)}
        with self.assertRaises(ValueError):
            validate_ink2_features(bad_shape)


class Ink2NativeVisionTests(unittest.TestCase):
    def setUp(self):
        _need_pillow()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)

    def test_multiframe_batching_and_cache(self):
        eng = FakeInk2Engine()
        vis = Ink2NativeVision(eng, cache=self.t / "cache", max_image_tokens=1024)
        vis.check()
        fmt = FakeInk2Formatter()
        u1 = _png_uri((80, 80), color=(10, 20, 30))
        u2 = _png_uri((120, 80), color=(40, 50, 60))
        ids, payload = vis.prepare([1, INK2_IMAGE_TOKEN_ID, 2, INK2_IMAGE_TOKEN_ID, 3], [u1, u2], fmt, 1000, 1064)
        self.assertEqual(len(eng.encode_calls), 1)
        self.assertEqual(eng.encode_calls[0], [(80, 80, 3), (80, 120, 3)])
        self.assertEqual(payload["features"].shape, (10, INK2_HIDDEN))
        self.assertEqual(payload["slots"].tolist(), list(range(1, 5)) + list(range(6, 12)))

        # Memory cache hit on repeat
        ids2, payload2 = vis.prepare([1, INK2_IMAGE_TOKEN_ID, 2, INK2_IMAGE_TOKEN_ID, 3], [u1, u2], fmt, 1000, 1064)
        self.assertEqual(len(eng.encode_calls), 1)
        np.testing.assert_array_equal(payload["features"], payload2["features"])

        # Corrupt one disk cache file and reload in fresh Ink2NativeVision
        files = sorted((self.t / "cache").glob("*.npz"))
        self.assertEqual(len(files), 2)
        files[0].write_bytes(b"corrupt")
        vis2 = Ink2NativeVision(eng, cache=self.t / "cache", max_image_tokens=1024)
        vis2.prepare([1, INK2_IMAGE_TOKEN_ID, 2, INK2_IMAGE_TOKEN_ID, 3], [u1, u2], fmt, 1000, 1064)
        self.assertEqual(len(eng.encode_calls), 2)
        self.assertEqual(len(eng.encode_calls[1]), 1)

    def test_limits_local_files_and_literal_marker_spoofing(self):
        from PIL import Image
        eng = FakeInk2Engine()
        fmt = FakeInk2Formatter()
        vis = Ink2NativeVision(eng, max_image_tokens=1024, max_request_image_tokens=1024)

        # Literal <|unused_200054|> spoofing rejected before reading source
        with self.assertRaises(RequestError):
            vis.prepare([1, INK2_IMAGE_TOKEN_ID, INK2_IMAGE_TOKEN_ID, 2], ["data:image/png;base64,!!"], fmt, 1000, 1064)

        # Request token budget enforced before encode_ink2_images
        big1 = _png_uri((1280, 720), color=(1, 2, 3))
        big2 = _png_uri((1280, 720), color=(4, 5, 6))
        with self.assertRaises(RequestError):
            vis.prepare([1, INK2_IMAGE_TOKEN_ID, INK2_IMAGE_TOKEN_ID, 2], [big1, big2], fmt, 10000, 10064)
        self.assertEqual(eng.encode_calls, [])

        # Context length limit enforced before encode_ink2_images
        with self.assertRaises(ContextLengthError):
            vis.prepare([1, INK2_IMAGE_TOKEN_ID, 2], [_png_uri((120, 80))], fmt, 7, 71)
        self.assertEqual(eng.encode_calls, [])

        # Local file security gating
        img_dir = self.t / "allowed"
        other_dir = self.t / "other"
        img_dir.mkdir()
        other_dir.mkdir()
        Image.new("RGB", (40, 40)).save(img_dir / "ok.png")
        Image.new("RGB", (40, 40)).save(other_dir / "nope.png")
        with self.assertRaises(ValueError):
            vis.prepare([1, INK2_IMAGE_TOKEN_ID, 2], [str(img_dir / "ok.png")], fmt, 1000, 1064)
        vis_dir = Ink2NativeVision(eng, allow_local_files=str(img_dir))
        vis_dir.check()
        vis_dir.prepare([1, INK2_IMAGE_TOKEN_ID, 2], [str(img_dir / "ok.png")], fmt, 1000, 1064)
        with self.assertRaises(ValueError):
            vis_dir.prepare([1, INK2_IMAGE_TOKEN_ID, 2], [str(other_dir / "nope.png")], fmt, 1000, 1064)


class Inkling2CliOptionTests(unittest.TestCase):
    def test_chat_image_args_and_validation(self):
        ap = cli.build_parser()
        a = ap.parse_args(["chat", "--image", "a.png", "--image", "b.png", "--vision-max-image-tokens", "512"])
        self.assertEqual(a.image, ["a.png", "b.png"])
        self.assertEqual(a.vision_max_image_tokens, 512)
        with self.assertRaises(Exception):
            validate_vision_limits(32, None)
        args = argparse.Namespace(mmproj="x.gguf", llama_dir=None, vision_model=None,
                                  vision_max_image_tokens=1024, vision_max_request_image_tokens=None,
                                  vision_allow_local_files=None, image=[])
        with self.assertRaises(Exception):
            cli._validate_ink2_vision_args(args)


@unittest.skipUnless((INK2_DIR / "tokenizer.json").is_file() and (INK2_DIR / "chat_template.jinja").is_file(),
                     "Inkling-2-Mini tokenizer files not present")
class Inkling2VisionServerStubTests(unittest.TestCase):
    def setUp(self):
        _need_pillow()
        self.stub = support.Stub()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = Path(tmp.name)
        for f in ("qwen38_27b_e3x.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead", "kvq_u5.bin"):
            (t / f).write_bytes(b"placeholder")
        for f in ("tokenizer.json", "tokenizer_config.json"):
            (t / f).write_bytes((INK2_DIR / f).read_bytes())
        self.engine = support.open_engine(t, self.stub, max_ctx=8192)
        self.fmt = ChatFormatter(INK2_DIR)
        self.calls = []
        self.encoded = []

        def fake_set_visual(e, feats, slots, n, pos, n_prompt, delta):
            self.calls.append((int(n), int(n_prompt), int(delta), pos))
            return 0

        def fake_encode_rgb(e, frames, widths, heights, n_frames, out_bf16, out_grids, out_n_tokens):
            w_arr = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_int32 * n_frames).from_address(widths))
            h_arr = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_int32 * n_frames).from_address(heights))
            g_out = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_int32 * (2 * n_frames)).from_address(out_grids)).reshape(n_frames, 2)
            tot = 0
            for i in range(n_frames):
                r, c = ink2_grid_size(int(w_arr[i]), int(h_arr[i]))
                g_out[i, 0], g_out[i, 1] = r, c
                tot += r * c
            buf = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_uint16 * (tot * INK2_HIDDEN)).from_address(out_bf16))
            buf[:] = 0x3F80
            if isinstance(out_n_tokens, int):
                np.ctypeslib.ctypes.c_int32.from_address(out_n_tokens).value = tot
            else:
                out_n_tokens._obj.value = tot
            self.encoded.append(tot)
            return 0

        self.engine.pack = ModelPack(root=INK2_DIR, weights=t / "qwen38_27b_e3x.yskn", max_ctx=8192,
                                     model_id="Inkling-2-Mini", arch="inkling2_loop", vocab_size=200058,
                                     padded_vocab_size=201024, eos_ids=(200006, 199999),
                                     hidden_size=INK2_HIDDEN, has_vision=True)
        self.engine._f = dict(self.engine._f, yskn_set_visual=fake_set_visual, yskn_ink2_encode_rgb=fake_encode_rgb)

        vp = Ink2NativeVision(self.engine)
        cfg = ServerConfig(port=0, model_id="Inkling-2-Mini", hf_inkling2=True)
        self.state = ApiState(cfg)
        self.state.runner = ChatRunner(self.engine, self.fmt, seed0=1, vision=vp)
        self.httpd = bind(self.state)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine.close()

    def _post(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("POST", "/v1/chat/completions", body=json.dumps(body),
                      headers={"Host": "127.0.0.1:%d" % self.port, "Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def test_multiframe_nonstream_and_stream_zh_tw(self):
        eos = 200006
        out_toks = self.fmt.encode("<|message_model|><|content_text|>紅色圓形與藍色方形<|end_message|>").tolist() + [eos]
        self.stub.push(out_toks)
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _png_uri((80, 80), color=(220, 20, 20))}},
                {"type": "image_url", "image_url": {"url": _png_uri((120, 80), color=(20, 20, 220))}},
                {"type": "text", "text": "這兩張圖有什麼？"},
            ]}],
            "max_tokens": 32,
            "temperature": 0,
            "reasoning_effort": "none",
        }
        status, raw = self._post(body)
        self.assertEqual(status, 200, raw)
        d = json.loads(raw)
        self.assertEqual(d["choices"][0]["message"]["content"], "紅色圓形與藍色方形")
        self.assertEqual(self.encoded, [10])
        self.assertEqual([(c[0], c[3]) for c in self.calls], [(10, None), (0, None)])

        # Streaming request hitting memory cache
        self.stub.push(out_toks)
        body["stream"] = True
        status, raw = self._post(body)
        self.assertEqual(status, 200, raw)
        self.assertEqual(self.encoded, [10])
        chunks = [
            json.loads(line[6:])
            for line in raw.decode("utf-8").splitlines()
            if line.startswith("data: {")
        ]
        streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))
        self.assertEqual(streamed, "紅色圓形與藍色方形")


if __name__ == "__main__":
    unittest.main()
