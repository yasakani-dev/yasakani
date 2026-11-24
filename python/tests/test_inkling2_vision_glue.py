# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 Mini image input, the glue between the package and the engine library (no GPU, no model weights): the prompt with
images (content parts, the placeholder expanded to rows x cols ids), the limits and their messages, the engine wrapper's prompt
reading with image positions (prefix reuse by (id, image key), two images of the same size, the host cache, truncation), copy
decoding next to image positions, the model folder's image tower, the context rule, and that a folder or library without image
input behaves as before.

Needs (else those tests skip):
  YASAKANI_TEST_INK2_SOURCE         the downloaded hf-internal-testing/tiny-inkling2-mini folder (tokenizer and chat template of
                                    Inkling-2 Mini)
  YASAKANI_TEST_INK2_VISION_REF     a folder with images/t1..t4.png and golden/t1_prompt_ids.json (ids of the official processor)
  YASAKANI_TEST_INK2_VISION_PROMPTS a prompts.json with the ids of reference prompts ([{"name", "image", "text", "ids"}, ...])
  YASAKANI_TEST_INK2_DLL            the host build of the engine library (it has no image input)
"""
import base64
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import _ink2_copy as copy_base
from yasakani import _ink2_native as native_mod
from yasakani import gpu as G
from yasakani import inkling2 as ink
from yasakani import inkling2_vision as V
from yasakani.chat import ChatParams, ContextLengthError, RequestError

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

SOURCE = os.environ.get("YASAKANI_TEST_INK2_SOURCE")
HAVE_SOURCE = bool(SOURCE) and (Path(SOURCE) / "tokenizer.json").is_file() and (Path(SOURCE) / "chat_template.jinja").is_file()
NEED_SOURCE = "set YASAKANI_TEST_INK2_SOURCE to the downloaded tiny-inkling2-mini folder"
REF = os.environ.get("YASAKANI_TEST_INK2_VISION_REF")
PROMPTS = os.environ.get("YASAKANI_TEST_INK2_VISION_PROMPTS")
DLL = os.environ.get("YASAKANI_TEST_INK2_DLL")
IMG = ink.IMAGE_TOKEN
_FMT = {}


def fmt():
    if "f" not in _FMT:
        _FMT["f"] = ink.Ink2Formatter(SOURCE)
    return _FMT["f"]


def image(w, h, seed):
    """A PreparedImage of random pixels (within the cap: no resize, no Pillow)."""
    rgb = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return V.prepare_rgb(rgb, 1024)


def state(cap=1024, req=None, cache_mib=64):
    return ink.Ink2VisionState(kind="yskn", max_image_tokens=cap, max_request_tokens=req or 8 * cap, memory=(0, 0, 0),
                               cache=V.FeatureCache(cache_mib << 20))


def runner(vision=None, max_ctx=32768, problem=V.NO_TOWER):
    eng = SimpleNamespace(vision=vision, vision_problem=problem, max_ctx=max_ctx, _cancel=threading.Event())
    return ink.Ink2Runner(eng, fmt(), SimpleNamespace(stop_ids=[fmt().ids[ink.T_STOP]]))


def user(*parts):
    return {"role": "user", "content": list(parts)}


def png_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ---- prompt ids ------------------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
@unittest.skipIf(Image is None, "Pillow is not installed")
class PromptIdsTests(unittest.TestCase):
    """prepare(): the ids of the official processor for the reference prompts."""

    def ids_of(self, img_path, text):
        r = runner(state())
        p = ChatParams(messages=[user({"type": "image_url", "image_url": {"url": png_uri(img_path)}}, {"type": "text", "text": text})])
        ids, in_answer = r.prepare_checked(p)
        self.assertFalse(in_answer)
        return ids

    def test_t1_golden(self):
        if not REF or not (Path(REF) / "golden" / "t1_prompt_ids.json").is_file():
            self.skipTest("YASAKANI_TEST_INK2_VISION_REF is not set")
        g = json.loads((Path(REF) / "golden" / "t1_prompt_ids.json").read_text(encoding="utf-8"))
        want = g["ids"] if isinstance(g, dict) else g
        ids = self.ids_of(Path(REF) / "images" / "t1.png", "Describe this image.")
        self.assertEqual(len(want), 215)
        self.assertEqual(list(ids), [int(x) for x in want])
        self.assertEqual(len(ids.images), 1)
        start, im = ids.images[0]
        self.assertEqual((start, im.rows, im.cols, im.tokens, ids.image_tokens), (14, 12, 16, 192, 192))

    def test_reference_prompts(self):
        if not REF or not PROMPTS or not Path(PROMPTS).is_file():
            self.skipTest("YASAKANI_TEST_INK2_VISION_REF / YASAKANI_TEST_INK2_VISION_PROMPTS are not set")
        for case in json.loads(Path(PROMPTS).read_text(encoding="utf-8")):
            with self.subTest(prompt=case["name"]):
                if case.get("image"):
                    ids = self.ids_of(Path(REF) / "images" / ("%s.png" % case["image"]), case["text"])
                    self.assertEqual(ids.images[0][0], case["image_span"][0])
                else:                                       # the text-only control: the path of a text-only engine
                    ids, _ = runner(state()).prepare_checked(ChatParams(messages=[{"role": "user", "content": case["text"]}]))
                    self.assertNotIsInstance(ids, ink.PromptIds)
                self.assertEqual(list(ids), [int(x) for x in case["ids"]])


# ---- limits and refusals -----------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class LimitTests(unittest.TestCase):
    def prep(self, messages, vision=None, **kw):
        return runner(vision if vision is not None else state(), **kw).prepare_checked(ChatParams(messages=messages))

    def test_expansion_and_order(self):
        a, b = image(80, 40, 1), image(40, 120, 2)               # 1 x 2 and 3 x 1 tiles
        msgs = [user({"type": "image", "image": a}, {"type": "text", "text": "and"}, {"type": "image", "image": b}),
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "name": "cam", "content": [{"type": "input_image", "image_url": {"url": "x"}}]}]
        with self.assertRaisesRegex(RequestError, r"messages\[2\]: image input: .*data:image"):
            self.prep(msgs)                                      # a local path without the opt-in
        msgs[2]["content"] = [{"type": "image", "image": a}]
        ids, _ = self.prep(msgs)
        self.assertEqual([(s, im.key) for s, im in ids.images][1:], [(ids.images[1][0], b.key), (ids.images[2][0], a.key)])
        self.assertEqual(ids.image_tokens, 2 + 3 + 2)
        self.assertEqual(sum(1 for t in ids if t == IMG), 7)
        for s, im in ids.images:
            self.assertEqual(ids[s - 1], fmt().tok.token_to_id("<|content_image|>"))
            self.assertEqual(ids[s:s + im.tokens], [IMG] * im.tokens)
            self.assertNotEqual(ids[s + im.tokens], IMG)

    def test_refusals(self):
        a = image(40, 40, 3)
        cases = [
            ([{"role": "assistant", "content": [{"type": "image", "image": a}]}], RequestError, r"messages\[0\]: images are supported only in user and tool"),
            ([{"role": "system", "content": [{"type": "image", "image": a}]}], RequestError, r"only in user and tool"),
            ([user({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}})], RequestError, r"video is not supported"),
            ([user({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}})], RequestError, r"remote image URLs"),
            ([user({"type": "image_url", "image_url": {"url": "data:image/png;base64,!!"}})], RequestError, r"not valid base64"),
            ([user({"type": "image_url", "image_url": {}})], RequestError, r"needs its image"),
            ([user(*[{"type": "image", "image": a}] * 17)], RequestError, r"at most 16 images"),
            ([user({"type": "image", "image": a}, {"type": "text", "text": "x <|unused_200054|> y"})], RequestError,
             r"contains the image placeholder"),
        ]
        for messages, err, rx in cases:
            with self.subTest(rx=rx):
                with self.assertRaisesRegex(err, rx):
                    self.prep(messages)

    def test_request_limit_names_the_tokens(self):
        big = image(320, 320, 4)                                 # 64 tokens
        with self.assertRaisesRegex(RequestError, r"take 128 image tokens; the limit per request is 100"):
            self.prep([user({"type": "image", "image": big}, {"type": "image", "image": big})], vision=state(cap=64, req=100))
        ids, _ = self.prep([user({"type": "image", "image": big})], vision=state(cap=64, req=100))
        self.assertEqual(ids.image_tokens, 64)

    def test_shrink_above_the_cap(self):
        if Image is None:
            self.skipTest("Pillow is not installed")
        big = image(640, 640, 5)                                 # 256 tokens > cap 64: shrunk before anything else
        ids, _ = self.prep([user({"type": "image", "image": big})], vision=state(cap=64))
        self.assertLessEqual(ids.images[0][1].tokens, 64)
        self.assertTrue(ids.images[0][1].shrunk)

    def test_context_error_names_the_image_tokens(self):
        a = image(400, 400, 6)                                   # 100 tokens
        with self.assertRaises(ContextLengthError) as cm:
            self.prep([user({"type": "image", "image": a}, {"type": "text", "text": "hi"})], max_ctx=120)
        self.assertRegex(str(cm.exception), r"\(100 of them image tokens of 1 image\)")

    def test_not_available(self):
        a = image(40, 40, 7)
        for problem in (V.NO_TOWER, ink.VISION_OFF, ink.VISION_NO_LIBRARY):
            r = runner(None, problem=problem)
            with self.assertRaisesRegex(RequestError, "image input is not available: " + problem.replace("(", r"\(").replace(")", r"\)")):
                r.prepare_checked(ChatParams(messages=[user({"type": "image", "image": a})]))
            self.assertIsNone(r.vision)

    def test_text_only_is_unchanged(self):
        """A request without image parts: the ids of the formatter, a plain list, the same with and without image input."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        want = fmt().prompt([{"role": "user", "content": "ab"}])
        for vision in (None, state()):
            got = runner(vision).prepare_checked(ChatParams(messages=[dict(m) for m in msgs]))
            self.assertEqual(got, want)
            self.assertIs(type(got[0]), list)
        with self.assertRaisesRegex(RequestError, "only text content"):
            runner(None).prepare_checked(ChatParams(messages=[user({"type": "audio", "x": 1})]))


# ---- the engine wrapper ----------------------------------------------------------------------------------------------------------
class FakeNative:
    """Records the calls read_prompt makes; image rows checked as the library does (ids, valid rows, increasing)."""

    def __init__(self, rollback=512):
        self.rollback = rollback
        self.calls = []
        self.valid = set()

    def reset(self):
        self.calls.append("reset")
        self.valid = set()

    def truncate(self, n):
        self.calls.append(("truncate", n))

    def feed_many(self, ids):
        assert all(t >= 0 for t in ids)
        self.calls.append(("feed", len(ids)))

    def feed_many_img(self, ids, rows):
        used = [r for r in rows if r >= 0]
        assert used == sorted(used) and len(set(used)) == len(used), rows
        for t, r in zip(ids, rows):
            assert (t == IMG) if r >= 0 else True
            assert r < 0 or r in self.valid, (r, sorted(self.valid))
        self.calls.append(("feed_img", len(ids), tuple(used)))

    def vision_encode(self, im, row0):
        self.calls.append(("encode", im.key[:8], row0))
        self.valid.update(range(row0, row0 + im.tokens))
        self.last = np.full((im.tokens, V.TEXT_HIDDEN), int(im.key[:4], 16), np.uint16)

    def vision_rows_bf16(self, row0, n):
        return self.last

    def vision_set_rows(self, feats, row0):
        self.calls.append(("upload", int(feats[0, 0]), row0))
        self.valid.update(range(row0, row0 + feats.shape[0]))

    def step(self, tok):
        assert tok != IMG
        self.calls.append("step")
        return np.zeros(4, np.float32)


def engine(rollback=512, vision=True, max_ctx=32768, **kw):
    e = ink.Ink2Engine.__new__(ink.Ink2Engine)
    e.max_ctx, e.native, e.fed, e.last_logits, e.last_vision = max_ctx, FakeNative(rollback), [], None, {}
    e.vision = state(**kw) if vision else None
    e.vision_problem = "" if vision else V.NO_TOWER
    return e


def prompt(parts):
    """parts: ints (text ids) and PreparedImage objects -> PromptIds."""
    out = ink.PromptIds()
    placed = []
    for x in parts:
        if isinstance(x, V.PreparedImage):
            placed.append((len(out), x))
            out.extend([IMG] * x.tokens)
        else:
            out.append(int(x))
    out.images = tuple(placed)
    out.image_tokens = sum(im.tokens for _, im in placed)
    return out


class EngineTests(unittest.TestCase):
    A, B = image(80, 80, 10), image(80, 80, 11)                  # 2 x 2 tiles each, same size, other pixels

    def test_prefix_reuse_by_image_key(self):
        e = engine()
        p1 = prompt([1, 2, self.A, 3, 4])
        _, cached = e.read_prompt(p1)
        self.assertEqual((e.native.calls, cached), (["reset", ("encode", self.A.key[:8], 0), ("feed_img", 7, (0, 1, 2, 3)), "step"], 0))
        self.assertEqual(e.last_vision["encoded"], 1)
        self.assertEqual(e.fed[2:6], [(IMG, self.A.key, k) for k in range(4)])
        # the next turn: the same image at the same place is neither encoded nor read again
        e.fed = e.fed + [7, 8]
        e.native.calls = []
        _, cached = e.read_prompt(prompt([1, 2, self.A, 3, 4, 7, 8, 9, 10]))
        self.assertEqual((e.native.calls, cached), ([("feed", 1), "step"], 10))
        self.assertEqual((e.last_vision["reused"], e.last_vision["encoded"], e.last_vision["from_cache"]), (1, 0, 0))
        # another image of the same size: not taken for the same prompt
        e.native.calls = []
        _, cached = e.read_prompt(prompt([1, 2, self.B, 3, 4]))
        self.assertEqual((e.native.calls, cached), ([("truncate", 2), ("encode", self.B.key[:8], 0), ("feed_img", 5, (0, 1, 2, 3)), "step"], 2))
        # back to A: its vectors come from the host cache, not from the tower
        e.native.calls = []
        e.read_prompt(prompt([1, 2, self.A, 3, 4]))
        self.assertEqual(e.native.calls[:2], [("truncate", 2), ("upload", int(self.A.key[:4], 16), 0)])
        self.assertEqual(e.last_vision["from_cache"], 1)

    def test_text_only_takes_the_text_path(self):
        e = engine()
        e.read_prompt([1, 2, 3, 4])
        self.assertEqual(e.native.calls, ["reset", ("feed", 3), "step"])
        e.native.calls = []
        e.read_prompt(prompt([1, 2, self.A, 3]))                 # after text: images read from the first difference
        e.native.calls = []
        e.read_prompt([1, 2, IMG, IMG, 9])                       # 200054 typed as text never matches an image position
        self.assertEqual(e.native.calls, [("truncate", 2), ("feed", 2), "step"])

    def test_two_images_and_the_same_image_twice(self):
        e = engine()
        e.read_prompt(prompt([1, self.A, 2, self.B, 3, self.A, 4]))
        enc = [c for c in e.native.calls if c[0] in ("encode", "upload")]
        self.assertEqual(enc, [("encode", self.A.key[:8], 0), ("encode", self.B.key[:8], 4), ("upload", int(self.A.key[:4], 16), 8)])
        self.assertIn(("feed_img", 15, tuple(range(12))), e.native.calls)

    def test_truncation_inside_an_image(self):
        """A shared prefix that ends inside an image: the kept positions stay, the rest of the image is read with its rows."""
        e = engine()
        e.read_prompt(prompt([1, self.A, 2]))
        e.fed = e.fed[:3] + [(IMG, "other", 2), (IMG, "other", 3), 2]     # positions 3, 4 hold other vectors
        e.native.calls = []
        _, cached = e.read_prompt(prompt([1, self.A, 2]))
        self.assertEqual(cached, 3)
        self.assertEqual(e.native.calls, [("truncate", 3), ("upload", int(self.A.key[:4], 16), 0), ("feed_img", 2, (2, 3)), "step"])

    def test_deep_truncation_reads_again(self):
        e = engine(rollback=2)
        e.read_prompt(prompt([1, self.A, 2, 3, 4, 5]))
        e.native.calls = []
        _, cached = e.read_prompt(prompt([1, self.A, 9]))       # 5 positions to drop > rollback 2: from the start
        self.assertEqual(cached, 0)
        self.assertEqual(e.native.calls[0], "reset")
        self.assertIn(("feed_img", 5, (0, 1, 2, 3)), e.native.calls)

    def test_refusals(self):
        with self.assertRaisesRegex(Exception, "image input: " + V.NO_TOWER[:20]):
            engine(vision=False).read_prompt(prompt([1, self.A, 2]))
        with self.assertRaisesRegex(Exception, "ends inside it"):
            engine().read_prompt(prompt([1, self.A]))           # the last position must be text
        with self.assertRaisesRegex(Exception, "need 8 feature rows; the engine holds 4"):
            engine(cap=64, req=4).read_prompt(prompt([1, self.A, self.B, 2]))

    def test_binding_never_feeds_image_positions_as_text(self):
        n = native_mod.Ink2Native.__new__(native_mod.Ink2Native)
        n.abi, n.rollback, n.num_seen = 2, 2, 5
        n.seen_ids = [1, native_mod.image_position(1), native_mod.image_position(2), 3, 4]
        with self.assertRaisesRegex(RuntimeError, "hold image positions"):
            n.truncate(1 + 1)                                    # deeper than the rollback, image positions kept
        with self.assertRaisesRegex(RuntimeError, "negative token id"):
            n.feed_many([1, -2])
        self.assertEqual(n.num_seen, 5)


# ---- copy decoding ---------------------------------------------------------------------------------------------------------------
class CopyTests(unittest.TestCase):
    def test_no_proposal_into_an_image(self):
        P = native_mod.image_position
        seen = [5, 6, 7, 8, P(4), P(5), 9, 10, 5, 6, 7]
        sp = copy_base.CopySpec()
        self.assertIsNone(sp.lookup(seen, 8))                    # the match 5 6 7 8 is followed by an image: nothing to copy
        seen2 = [5, 6, 7, 8, 11, P(5), 9, 5, 6, 7]
        src, chain, ln, cursor = sp.lookup(seen2, 8)
        self.assertEqual(chain, [11])                            # cut before the image position
        self.assertEqual(copy_base.text_chain([1, 2, -9, 3]), [1, 2])
        seen3 = [P(0), P(1), P(2), 1, 2, 3, 4, P(7), P(8), P(9)]
        self.assertIsNone(copy_base.CopySpec().lookup(seen3, 4))      # a window over image positions never matches


# ---- model folder, library, context ----------------------------------------------------------------------------------------------
class FolderTests(unittest.TestCase):
    def folder(self, d, fmt_="yskn", vision=None, vision_config=True, make_file=True):
        d = Path(d)
        for n in ("tokenizer.json", "chat_template.jinja", "w.bin"):
            (d / n).write_text("x", encoding="utf-8")
        (d / "config.json").write_text(json.dumps({"text_config": {}, **({"vision_config": {}} if vision_config else {})}),
                                       encoding="utf-8")
        meta = {"architecture": "inkling2", "weights": {"file": "w.bin", "format": fmt_, "version": 2}}
        if vision:
            meta["vision"] = vision
            if make_file:
                (d / vision["file"]).write_bytes(b"v")
        (d / "yasakani.json").write_text(json.dumps(meta), encoding="utf-8")
        return ink.Ink2Pack.load(d)

    def test_has_vision(self):
        entry = {"file": V.VISION_FILE, "format": "yskn", "version": 2, "sha256": "0" * 64, "size": 1}
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self.folder(d).has_vision)                          # converted, no vision entry: as before
        with tempfile.TemporaryDirectory() as d:
            pack = self.folder(d, vision=entry)
            self.assertTrue(pack.has_vision)
            self.assertEqual(pack.vision_file.name, V.VISION_FILE)
            self.assertEqual(pack.verify(), {V.VISION_FILE: False})               # listed for verify (the hash differs)
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self.folder(d, vision=entry, make_file=False).has_vision)
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self.folder(d, fmt_="hf-safetensors").has_vision)      # the published file holds the tower
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self.folder(d, fmt_="hf-safetensors", vision_config=False).has_vision)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(Exception, "vision entry"):
                self.folder(d, vision=dict(entry, file="../x.yskn"), make_file=False)

    def test_context_rule(self):
        self.assertAlmostEqual(G.ink2_vision_mib(1024, 8192) * G.MIB, 636_512_256 + 88_821_760 + 8192 * 4608)
        self.assertAlmostEqual(G.ink2_vision_mib(4096, 32768, weights=False) * G.MIB, 355_225_600 + 32768 * 4608)
        args = (20000, 15_052_423_172 + 209_688_040, G.ink2_bytes_per_position(65536, True), 1 << 20, 512)
        self.assertEqual(G.ink2_context(*args), G.ink2_context(*args, vision_mib=0.0))
        self.assertLess(G.ink2_context(*args, vision_mib=G.ink2_vision_mib(1024, 8192)), G.ink2_context(*args))
        self.assertEqual(G.ink2_need_mib(32768, *args[1:3], 512), G.ink2_need_mib(32768, *args[1:3], 512, vision_mib=0))
        entry = {"file": V.VISION_FILE, "format": "yskn", "version": 2}
        with tempfile.TemporaryDirectory() as d:
            pack = self.folder(d, vision=entry)
            self.assertEqual(ink.vision_plan_mib(pack, "off"), 0.0)
            self.assertEqual(ink.vision_plan_mib(pack, "auto", has_library=False), 0.0)
            self.assertAlmostEqual(ink.vision_plan_mib(pack, "auto", has_library=True), G.ink2_vision_mib(1024, 8192))
            self.assertAlmostEqual(ink.vision_plan_mib(pack, "on", 4096, 4096), G.ink2_vision_mib(4096, 4096))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ink.vision_plan_mib(self.folder(d), "on"), 0.0)           # no tower: plans as today
        with tempfile.TemporaryDirectory() as d:
            self.assertAlmostEqual(ink.vision_plan_mib(self.folder(d, fmt_="hf-safetensors"), "on"),
                                   G.ink2_vision_mib(1024, 8192, weights=False))

    def test_options(self):
        self.assertEqual(ink.vision_mode_of(None, {}), "auto")
        self.assertEqual(ink.vision_mode_of(None, {ink.VISION_ENV: "OFF"}), "off")
        with self.assertRaisesRegex(Exception, "--vision must be one of"):
            ink.vision_mode_of("maybe")
        self.assertEqual(ink.vision_limits(), (1024, 8192))
        with self.assertRaisesRegex(Exception, "--vision-max-image-tokens"):
            ink.vision_limits(5000)
        with self.assertRaisesRegex(Exception, "--vision-max-request-image-tokens"):
            ink.vision_limits(1024, 100)

    def test_set_vision_without_tower_or_capability(self):
        """auto: off with the reason, nothing loaded; on: an error naming the reason."""
        with tempfile.TemporaryDirectory() as d:
            pack = self.folder(d)
            e = ink.Ink2Engine.__new__(ink.Ink2Engine)
            e.max_ctx, e.backend, e.native = 32768, "cuda", SimpleNamespace(has_vision=lambda: False)
            self.assertEqual(e.set_vision(pack, "auto"), "image input: off (%s)" % V.NO_TOWER)
            self.assertIsNone(e.vision)
            pack = self.folder(d, vision={"file": V.VISION_FILE})
            self.assertEqual(e.set_vision(pack, "auto"), "image input: off (%s)" % ink.VISION_NO_LIBRARY)
            with self.assertRaisesRegex(Exception, "could not be turned on: " + ink.VISION_NO_LIBRARY[:30]):
                e.set_vision(pack, "on")
            e.backend = "host"
            self.assertEqual(e.set_vision(pack, "auto"), "image input: off (%s)" % ink.VISION_NO_GPU)
            self.assertEqual(e.set_vision(pack, "off"), "image input: off (--vision off)")

    @unittest.skipUnless(DLL and Path(DLL).is_file(), "set YASAKANI_TEST_INK2_DLL to the host build of the engine library")
    def test_host_library_has_no_image_input(self):
        lib = native_mod.open_dll(DLL)
        self.assertFalse(lib.has_vision)
        self.assertFalse(ink.library_has_vision(DLL))


if __name__ == "__main__":
    unittest.main()
