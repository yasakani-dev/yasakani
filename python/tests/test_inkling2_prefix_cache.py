# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Inkling-2 multi-turn prefix cache tests (zero-KV-copy rewind, SWA host restore, vis_sig isolation, root protection)."""
from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path

from yasakani.engine import Engine, Sampling
from yasakani.runtime import ModelPack
from yasakani.textio import ChatFormatter, Inkling2OutputStream
from yasakani.vision import Ink2NativeVision, split_images

INK2_DIR = Path("/home/au/models/Inkling-2-Mini")
REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(os.environ.get("YASAKANI_LIB", str(REPO_ROOT / "build/bin/libyskn.so")))
SAMPLING = Sampling(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=1)


def _has_ink2_runtime() -> bool:
    return (INK2_DIR / "config.json").is_file() and (INK2_DIR / "tokenizer.json").is_file() and LIB_PATH.is_file()


def _png_uri(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@unittest.skipUnless(_has_ink2_runtime(), "Inkling-2 checkpoint or libyskn.so not available")
class Inkling2PrefixCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = ModelPack.load(INK2_DIR, hf_inkling2=True)
        cls.fmt = ChatFormatter(INK2_DIR)
        cls.eng = Engine(cls.pack, max_ctx=8192, dll=str(LIB_PATH))
        cls.eng.set_sampling(SAMPLING)
        cls.vis = Ink2NativeVision(cls.eng)

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "eng", None) is not None:
            cls.eng.close()
            cls.eng = None

    def setUp(self) -> None:
        self.eng.reset()
        self.eng.set_sampling(SAMPLING)

    def test_reset_clears_snapshots(self) -> None:
        msgs = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is 6 times 7?"},
        ]
        p1, _ = self.fmt.prompt(msgs, reasoning_effort="low")
        r1 = self.eng.generate(p1, 64, stop_ids=self.pack.eos_ids)
        self.assertEqual(r1.snap_mode, 0)
        self.assertEqual(r1.cached_tokens, 0)
        self.eng.reset()
        self.assertEqual(self.eng.position(), 0)
        p2, _ = self.fmt.prompt(msgs + [{"role": "assistant", "content": "42"}, {"role": "user", "content": "Add 8."}], reasoning_effort="low")
        r2 = self.eng.generate(p2, 16, stop_ids=None)
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (0, 0))

    def test_mtp_eos_live_position_invariant(self) -> None:
        msgs = [
            {"role": "system", "content": "Reply in one short sentence."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        p1, _ = self.fmt.prompt(msgs, reasoning_effort="none")
        r1 = self.eng.generate(p1, 128, stop_ids=self.pack.eos_ids)
        self.assertEqual(r1.finish, "stop")
        self.assertEqual(self.eng.position(), len(p1) + len(r1.ids) - 1)

    def test_thought_stripped_turn2_zero_kv_copy_bit_exact(self) -> None:
        m1 = [
            {"role": "system", "content": "You are a precise mathematics tutor. Always check arithmetic step by step."},
            {"role": "user", "content": "Compute 17 * 23 and show your multiplication steps clearly."},
        ]
        p1, _ = self.fmt.prompt(m1, reasoning_effort="medium")
        r1 = self.eng.generate(p1, 512, stop_ids=self.pack.eos_ids)
        self.assertEqual(r1.finish, "stop")
        ev = Inkling2OutputStream(self.fmt, hidden_ids=self.pack.eos_ids).feed(r1.ids, final=True)
        reason = "".join(v for k, v in ev if k == "reasoning")
        content = "".join(v for k, v in ev if k == "content")
        self.assertGreater(len(reason), 0)
        self.assertGreater(len(content), 0)

        m2 = m1 + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "What is 391 plus 109? Reply with the exact integer result and state whether the sum is divisible by ten."},
        ]
        p2, _ = self.fmt.prompt(m2, reasoning_effort="medium")
        self.assertEqual(p2[:len(p1)].tolist(), p1.tolist())

        r2 = self.eng.generate(p2, 64, stop_ids=None)
        self.assertEqual(r2.snap_mode, 2)
        self.assertEqual(r2.cached_tokens, len(p1))

        self.eng.reset()
        r2_cold = self.eng.generate(p2, 64, stop_ids=None)
        self.assertEqual(r2.ids, r2_cold.ids)

    def test_thought_off_end_message_branch_rewind_bit_exact(self) -> None:
        base_m = [
            {"role": "system", "content": "You are a helpful geography assistant answering directly and concisely."},
            {"role": "user", "content": "Name the capital city of France and state what river runs through it."},
        ]
        p_raw, _ = self.fmt.prompt(base_m, reasoning_effort="none")
        p_t1 = p_raw.tolist() + self.fmt.encode("<|message_model|><|content_text|>").tolist()
        r_d1 = self.eng.generate(p_t1, 128, stop_ids=self.pack.eos_ids)
        self.assertEqual(r_d1.finish, "stop")
        t1_end = len(p_t1) + len(r_d1.ids) - 1
        self.assertEqual(self.eng.position(), t1_end)
        prefix_d1 = p_t1 + r_d1.ids[:-1]

        u2a = self.fmt.encode("<|content_model_end_sampling|><|message_user|><|content_text|>List three famous national museums located inside the city of Paris, numbered 1 through 3, and give a concise one-sentence description of each museum's primary collection.<|end_message|><|message_model|><|content_text|>").tolist()
        u2b = self.fmt.encode("<|content_model_end_sampling|><|message_user|><|content_text|>State the exact year the Eiffel Tower first opened to the public, name its principal structural engineer, and give its official current height in meters including the broadcast antenna.<|end_message|><|message_model|><|content_text|>").tolist()

        r_2a = self.eng.generate(prefix_d1 + u2a, 64, stop_ids=None)
        self.assertEqual((r_2a.snap_mode, r_2a.cached_tokens), (1, t1_end))

        r_2b = self.eng.generate(prefix_d1 + u2b, 64, stop_ids=None)
        self.assertEqual((r_2b.snap_mode, r_2b.cached_tokens), (2, t1_end))

        # Compare restored kEndMessage snapshot directly against uninterrupted Turn-1 -> Turn-2b live continuation
        # (both share the exact decoded Turn-1 KV prefix 0..t1_end; cold single-pass prefill uses a different BF16 GEMM path).
        self.eng.reset()
        r_d1_ref = self.eng.generate(p_t1, 128, stop_ids=self.pack.eos_ids)
        self.assertEqual(r_d1_ref.ids, r_d1.ids)
        r_2b_live = self.eng.generate(prefix_d1 + u2b, 64, stop_ids=None)
        self.assertEqual((r_2b_live.snap_mode, r_2b_live.cached_tokens), (1, t1_end))
        self.assertEqual(r_2b.ids, r_2b_live.ids)

    def test_multimodal_vis_sig_reuse_and_invalidation(self) -> None:
        from PIL import Image, ImageDraw
        im_red = Image.new("RGB", (320, 320), (245, 245, 245))
        ImageDraw.Draw(im_red).ellipse([60, 60, 260, 260], fill=(220, 25, 25))
        im_blue = Image.new("RGB", (320, 320), (245, 245, 245))
        ImageDraw.Draw(im_blue).rectangle([60, 60, 260, 260], fill=(25, 70, 225))
        uri_red, uri_blue = _png_uri(im_red), _png_uri(im_blue)

        def prep_mm(msgs):
            clean_m, srcs = split_images(msgs, is_inkling2=True)
            p_base, _ = self.fmt.prompt(clean_m, reasoning_effort="low")
            return self.vis.prepare(p_base, srcs, self.fmt, self.eng.max_ctx - 64, self.eng.max_ctx)

        mm_m1 = [
            {"role": "system", "content": "You are an accurate visual inspection assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri_red}},
                {"type": "text", "text": "Identify the color and shape shown in this image concisely."},
            ]},
        ]
        p1, v1 = prep_mm(mm_m1)
        with self.eng.visual_input(**v1):
            r1 = self.eng.generate(p1, 256, stop_ids=self.pack.eos_ids)
        ans1 = "".join(v for k, v in Inkling2OutputStream(self.fmt, hidden_ids=self.pack.eos_ids).feed(r1.ids, final=True) if k == "content")

        mm_m2 = mm_m1 + [
            {"role": "assistant", "content": ans1},
            {"role": "user", "content": "What background color surrounds that geometric object, and is the shape roughly centered in the frame?"},
        ]
        p2, v2 = prep_mm(mm_m2)
        with self.eng.visual_input(**v2):
            r2 = self.eng.generate(p2, 64, stop_ids=None)
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (2, len(p1)))

        self.eng.reset()
        with self.eng.visual_input(**v2):
            r2_cold = self.eng.generate(p2, 64, stop_ids=None)
        self.assertEqual(r2.ids, r2_cold.ids)

        # Identical prompt token IDs with a different image -> must invalidate (cached=0, snap_mode=0)
        self.eng.reset()
        with self.eng.visual_input(**v1):
            _ = self.eng.generate(p1, 64, stop_ids=None)
        mm_blue = [
            {"role": "system", "content": "You are an accurate visual inspection assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri_blue}},
                {"type": "text", "text": "Identify the color and shape shown in this image concisely."},
            ]},
        ]
        p_b, v_b = prep_mm(mm_blue)
        self.assertEqual(p_b.tolist(), p1.tolist())
        with self.eng.visual_input(**v_b):
            r_b = self.eng.generate(p_b, 64, stop_ids=None)
        self.assertEqual((r_b.snap_mode, r_b.cached_tokens), (0, 0))
        self.eng.reset()
        with self.eng.visual_input(**v_b):
            r_b_cold = self.eng.generate(p_b, 64, stop_ids=None)
        self.assertEqual(r_b.ids, r_b_cold.ids)

    def test_six_slot_eviction_preserves_shared_root(self) -> None:
        root_m = [
            {"role": "system", "content": "You are a precise mathematics tutor. Always check arithmetic step by step."},
            {"role": "user", "content": "Compute 17 * 23 and show your multiplication steps clearly."},
        ]
        p_root, _ = self.fmt.prompt(root_m, reasoning_effort="medium")
        r_root = self.eng.generate(p_root, 512, stop_ids=self.pack.eos_ids)
        ans_root = "".join(v for k, v in Inkling2OutputStream(self.fmt, hidden_ids=self.pack.eos_ids).feed(r_root.ids, final=True) if k == "content")

        hist = list(root_m) + [{"role": "assistant", "content": ans_root}]
        for turn in range(8):
            hist.append({"role": "user", "content": f"Turn {turn}: add {turn + 1} to the previous number and state the result."})
            pt, _ = self.fmt.prompt(hist, reasoning_effort="medium")
            rt = self.eng.generate(pt, 64, stop_ids=self.pack.eos_ids)
            at = "".join(v for k, v in Inkling2OutputStream(self.fmt, hidden_ids=self.pack.eos_ids).feed(rt.ids, final=True) if k == "content")
            hist.append({"role": "assistant", "content": at or f"Step {turn}."})

        p_branch, _ = self.fmt.prompt(
            root_m + [{"role": "assistant", "content": ans_root}, {"role": "user", "content": "What is 391 plus 109? Reply with the exact integer result and state whether the sum is divisible by ten."}],
            reasoning_effort="medium",
        )
        r_branch = self.eng.generate(p_branch, 64, stop_ids=None)
        self.assertEqual(r_branch.snap_mode, 2)
        self.assertGreaterEqual(r_branch.cached_tokens, len(p_root))

        self.eng.reset()
        r_branch_cold = self.eng.generate(p_branch, 64, stop_ids=None)
        self.assertEqual(r_branch.ids, r_branch_cold.ids)


@unittest.skipUnless(_has_ink2_runtime(), "Inkling-2 checkpoint or libyskn.so not available")
class Inkling2PrefixCacheSwaRestoreTests(unittest.TestCase):
    def test_swa_host_restore_and_wrapped_ring_bit_exact(self) -> None:
        script = r"""
import os, sys
from pathlib import Path
sys.path.insert(0, %r)
from yasakani.engine import Engine, Sampling
from yasakani.runtime import ModelPack
from yasakani.textio import ChatFormatter

pack = ModelPack.load(Path(%r), hf_inkling2=True)
fmt = ChatFormatter(Path(%r))
eng = Engine(pack, max_ctx=4096, dll=%r)
try:
    eng.set_sampling(Sampling(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=1))
    filler = " ".join(f"item_{i}={i*3+1}" for i in range(350))
    ml = [
        {"role": "system", "content": "Context log: " + filler},
        {"role": "user", "content": "What is item_10 plus item_20? Explain carefully."},
    ]
    pl, _ = fmt.prompt(ml, reasoning_effort="high")
    assert 1300 <= len(pl) < 2200, len(pl)
    rl1 = eng.generate(pl, 320, stop_ids=None)
    assert len(rl1.ids) == 320
    ml2 = ml + [
        {"role": "assistant", "content": "92"},
        {"role": "user", "content": "Now multiply that sum by 3 and show the steps concisely."},
    ]
    pl2, _ = fmt.prompt(ml2, reasoning_effort="high")
    rl2 = eng.generate(pl2, 64, stop_ids=None)
    assert rl2.snap_mode == 2 and rl2.cached_tokens == len(pl), (rl2.snap_mode, rl2.cached_tokens, len(pl))
    eng.reset()
    rl2_cold = eng.generate(pl2, 64, stop_ids=None)
    assert rl2.ids == rl2_cold.ids
    print("SWA_WRAPPED_RESTORE_64_EXACT")
finally:
    eng.close()
""" % (str(REPO_ROOT / "python"), str(INK2_DIR), str(INK2_DIR), str(LIB_PATH))
        env = os.environ.copy()
        env["YSKN_INK2_RING_ROWS"] = "1280"
        env["YSKN_INK2_SNAP_DEBUG"] = "1"
        cp = subprocess.run([sys.executable, "-c", script], env=env, text=True, capture_output=True)
        self.assertEqual(cp.returncode, 0, cp.stdout + "\n" + cp.stderr)
        self.assertIn("[ink2_snap] backup", cp.stderr)
        self.assertIn("mode=swa_host_restore", cp.stderr)
        self.assertIn("SWA_WRAPPED_RESTORE_64_EXACT", cp.stdout)


if __name__ == "__main__":
    unittest.main()
