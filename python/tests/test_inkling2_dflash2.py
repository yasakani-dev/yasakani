# SPDX-License-Identifier: Apache-2.0
import ctypes
import json
import math
import os
import pathlib
import random
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from yasakani import cli
from yasakani._native import NativeLib, YasakaniError
from yasakani.runtime import HF_INKLING2_ARCH, ModelPack, RuntimeProfile


def _load_accept_sparse():
    try:
        nlib = NativeLib()
    except YasakaniError as exc:
        raise unittest.SkipTest(f"native library not available: {exc}") from exc
    fn = getattr(nlib.dll, "yskn_ink2_accept_sparse", None)
    if fn is None:
        raise unittest.SkipTest("libyskn not built with YSKN_WITH_INK2=ON")
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint64,
        ctypes.c_int64,
    ]
    fn.restype = ctypes.c_int
    return fn


def _softmax(row, temp=1.0):
    m = max(row)
    w = [math.exp((v - m) / temp) for v in row]
    s = sum(w)
    return [x / s for x in w]


class TestInkling2DFlash2Cli(unittest.TestCase):
    def test_parser_accepts_dflash_dir(self):
        for cmd in ("serve", "chat", "bench"):
            args = cli.build_parser().parse_args([cmd, "--dflash-dir", "/tmp/some-dflash", "--draft-depth", "7"])
            self.assertEqual(args.dflash_dir, "/tmp/some-dflash")
            self.assertEqual(args.draft_depth, 7)

    def test_missing_dflash_dir_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "config.json").write_text('{"model_type":"inkling2_loop","text_config":{"model_type":"inkling2_loop_text"}}')
            (root / "tokenizer.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"")
            missing = root / "missing_dflash"
            args = cli.build_parser().parse_args(
                ["serve", "--model-dir", str(root), "--dflash-dir", str(missing)]
            )
            with mock.patch("yasakani.textio.ChatFormatter"), \
                 mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch("yasakani.cli.NativeLib") as lib_mock, \
                 mock.patch("yasakani.cli.lib_has_hf_inkling2", return_value=True):
                lib_mock.return_value = mock.MagicMock()
                with self.assertRaisesRegex(YasakaniError, "--dflash-dir"):
                    cli._open_stack(args, warm=False)

    def test_runtime_profile_preserves_dflash_env_for_hf_ink2(self):
        with tempfile.TemporaryDirectory() as td:
            pack = ModelPack(root=pathlib.Path(td), weights=pathlib.Path(td) / "model.safetensors", arch=HF_INKLING2_ARCH)
            with mock.patch.dict(os.environ, {"YSKN_INK2_DFLASH_DIR": "/opt/df", "YSKN_INK2_DFLASH_K": "5"}, clear=False):
                prof = RuntimeProfile.default(pack, {"YSKN_INK2_DFLASH": "1"})
            self.assertEqual(prof.env["YSKN_INK2_DFLASH_DIR"], "/opt/df")
            self.assertEqual(prof.env["YSKN_INK2_DFLASH_K"], "5")
            self.assertEqual(prof.env["YSKN_INK2_DFLASH"], "1")


class TestInkling2DFlash2Checkpoint(unittest.TestCase):
    def test_bad_draft_tensor_is_reported_before_selecting_cuda_device(self):
        try:
            lib = NativeLib()
        except YasakaniError as exc:
            self.skipTest(f"native library not available: {exc}")
        if not hasattr(lib.dll, "yskn_ink2_accept_sparse"):
            self.skipTest("libyskn not built with YSKN_WITH_INK2=ON")

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({
                "model_type": "inkling2_loop",
                "unpadded_vocab_size": 200058,
                "local_layer_ids": [i for i in range(24) if i % 6 != 5],
            }))
            (draft / "config.json").write_text(json.dumps({
                "hidden_size": 2304, "num_hidden_layers": 5, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 6912,
                "sliding_window": 2048, "rms_norm_eps": 1e-6, "vocab_size": 201024,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
                "num_target_layers": 48, "is_causal": False, "use_sliding_window": True,
                "hidden_act": "silu", "attention_bias": False, "layer_types": ["sliding_attention"] * 5,
                "dflash_config": {
                    "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                    "selector_rank": 256, "selector_top_k": 16, "mask_token_id": 200056,
                    "final_logit_softcapping": 30.0, "selector_softcap_mode": "after_correction",
                    "target_layer_ids": [5, 23, 41, 48],
                },
            }))
            header = json.dumps({
                "fc.weight": {"dtype": "BF16", "shape": [1, 1], "data_offsets": [0, 2]},
            }).encode()
            (draft / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with mock.patch.dict(os.environ, {"YSKN_INK2_DFLASH_DIR": str(draft), "YSKN_INK2_DFLASH": "1"}):
                # CUDA would reject this device even on a host with a GPU. Use the public entry point to prove ordering.
                handle = lib.fn["yskn_open"](os.fsencode(target), 8192, -1, 0)
                if handle:
                    lib.fn["yskn_close"](handle)
                    self.fail("Opened an invalid draft checkpoint")
                self.assertRegex(lib.last_error(), r"fc\.weight.*needs BF16")


class TestInkling2DFlash2AcceptSparse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accept_sparse = _load_accept_sparse()

    def test_greedy_exact_prefix_and_bonus(self):
        K, V, Ksel = 4, 8, 4
        draft = (ctypes.c_int32 * K)(1, 2, 3, 4)
        out = (ctypes.c_int32 * (K + 1))()
        v_arg = (ctypes.c_int32 * (K + 1))(1, 2, 6, 4, 5)
        a = self.accept_sparse(out, draft, K, V, None, None, None, Ksel, v_arg, 0.0, 0, 1.0, 0.0, 7, 0)
        self.assertEqual(a, 2)
        self.assertEqual([out[i] for i in range(a + 1)], [1, 2, 6])

        v_arg_all = (ctypes.c_int32 * (K + 1))(1, 2, 3, 4, 5)
        a_all = self.accept_sparse(out, draft, K, V, None, None, None, Ksel, v_arg_all, 0.0, 0, 1.0, 0.0, 7, 0)
        self.assertEqual(a_all, 4)
        self.assertEqual([out[i] for i in range(K + 1)], [1, 2, 3, 4, 5])

    def test_sampled_sparse_unbiased_distribution(self):
        K, V, Ksel = 1, 5, 3
        temp = 1.0
        p_logits = [0.5, 1.8, -0.3, 0.9, -1.2]
        p_prob = _softmax(p_logits, temp)
        # Sparse candidate distribution over only 3 tokens [0, 2, 3]
        cand_ids = [0, 2, 3]
        cand_probs = [0.60, 0.30, 0.10]

        v_buf = (ctypes.c_float * (2 * V))(*(p_logits + [0.0] * V))
        c_ids = (ctypes.c_int32 * Ksel)(*cand_ids)
        c_probs = (ctypes.c_float * Ksel)(*cand_probs)
        out = (ctypes.c_int32 * 2)()

        rng = random.Random(20261124)
        counts = [0] * V
        N = 25000
        for step in range(N):
            u = rng.random()
            cum = 0.0
            dt = cand_ids[-1]
            for tid, qi in zip(cand_ids, cand_probs):
                cum += qi
                if u < cum:
                    dt = tid
                    break
            draft = (ctypes.c_int32 * 1)(dt)
            self.accept_sparse(out, draft, K, V, v_buf, c_ids, c_probs, Ksel, None, temp, 0, 1.0, 0.0, 991 + step * 13, step)
            counts[out[0]] += 1

        for tid in range(V):
            obs = counts[tid] / N
            exp = p_prob[tid]
            se = math.sqrt(exp * (1.0 - exp) / N)
            self.assertLess(abs(obs - exp), 4.0 * se, f"tok {tid}: obs={obs:.4f} vs exp={exp:.4f}")


if __name__ == "__main__":
    unittest.main()
