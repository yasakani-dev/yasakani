# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests import support                                   # noqa: F401
from yasakani import chat, cli, engine, gpu, textio
from yasakani._native import YasakaniError
from yasakani.runtime import RuntimeProfile


def _fake_pack_with_sizes(root: Path, weights_mib: int, draft_mib: int = 1144, head_mib: int = 1335):
    pack = support.make_pack(root)
    sizes = {
        pack.weights: weights_mib * gpu.MIB,
        pack.draft: draft_mib * gpu.MIB,
        pack.draft_head: head_mib * gpu.MIB,
    }
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self in sizes:
            return types.SimpleNamespace(st_size=sizes[self])
        return real_stat(self, *args, **kwargs)

    return pack, fake_stat


class Yarn2CliTests(unittest.TestCase):
    def test_parser_accepts_yarn2_except_selfcheck(self):
        parser = cli.build_parser()
        for cmd in ("serve", "chat", "bench"):
            args = parser.parse_args([cmd, "--yarn2"])
            self.assertTrue(args.yarn2)
            args_default = parser.parse_args([cmd])
            self.assertFalse(args_default.yarn2)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                parser.parse_args(["selfcheck", "--yarn2"])
        self.assertEqual(cm.exception.code, 2)

    def test_selfcheck_refuses_programmatic_yarn2_before_load(self):
        with tempfile.TemporaryDirectory() as t:
            pack = support.make_pack(Path(t))
            args = cli.build_parser().parse_args(["selfcheck", "--model-dir", str(pack.root)])
            args.yarn2 = True
            with self.assertRaisesRegex(YasakaniError, "selfcheck does not support --yarn2"):
                cli._open_stack(args, warm=False, allow_yarn2=False)

    def test_yarn2_env_survives_profile_apply(self):
        with tempfile.TemporaryDirectory() as t:
            pack = support.make_pack(Path(t))
            prof = RuntimeProfile.default(pack, {"YSKN_Q35_YARN2": "1"})
            env = {"YSKN_Q35_YARN2": "0", "YSKN_STALE": "1"}
            removed = prof.apply(env)
            self.assertEqual(removed, ["YSKN_STALE"])
            self.assertEqual(env["YSKN_Q35_YARN2"], "1")

    def test_open_stack_rejects_yarn2_above_524288(self):
        with tempfile.TemporaryDirectory() as t:
            pack = support.make_pack(Path(t))
            args = cli.build_parser().parse_args(
                ["serve", "--model-dir", str(pack.root), "--max-ctx", "524289", "--yarn2"]
            )
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()):
                with self.assertRaisesRegex(YasakaniError, "524288"):
                    cli._open_stack(args, warm=False)

    def test_yarn2_without_max_ctx_plans_extended_context(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = support.make_pack(root)
            dll = root / "fake.so"
            dll.write_bytes(b"fake-engine")
            seen_ctx = []

            class FakeEngine:
                def __init__(self, _pack, *, max_ctx=None, device=0, dll=None, profile=None, wide=None, lib=None):
                    seen_ctx.append(max_ctx)
                    self.max_ctx = max_ctx or _pack.max_ctx
                    self.removed_env = []
                    self.lib = types.SimpleNamespace(path=dll)

                def close(self):
                    pass

            args = cli.build_parser().parse_args(["serve", "--model-dir", str(pack.root), "--dll", str(dll), "--yarn2"])
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch.object(engine, "Engine", FakeEngine), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()), \
                 mock.patch.object(chat, "ChatRunner", return_value=types.SimpleNamespace(vision=None)), \
                 mock.patch.object(cli.widecheck, "setup", return_value=None), \
                 contextlib.redirect_stderr(err):
                _, eng, _, _ = cli._open_stack(args, warm=False)
            self.assertEqual(seen_ctx, [None])
            self.assertEqual(eng.max_ctx, pack.max_ctx)
            self.assertIn("GPU free memory could not be read", err.getvalue())
            self.assertIn("--max-ctx", err.getvalue())
            self.assertIn("experimental", err.getvalue())
            self.assertIn("5-bit KV cache", err.getvalue())

            # With free GPU memory known (32 GB RTX 5090), --yarn2 without --max-ctx plans up to 524,288
            seen_ctx.clear()
            g5090 = gpu.GpuInfo(index=0, name="NVIDIA GeForce RTX 5090", uuid="GPU-1", compute_cap=(12, 0),
                                total_mib=32607, used_mib=1100, driver="590.0")
            det5090 = gpu.Detected(gpus=[g5090], candidates=[g5090], device=0, numbered=True)
            pack_q, stat_q = _fake_pack_with_sizes(root / "q32", weights_mib=11826)
            args_q = cli.build_parser().parse_args(["serve", "--model-dir", str(pack_q.root), "--dll", str(dll), "--yarn2"])
            err_q = io.StringIO()
            with mock.patch.object(gpu, "detect", return_value=det5090), \
                 mock.patch.object(Path, "stat", stat_q), \
                 mock.patch.object(engine, "Engine", FakeEngine), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()), \
                 mock.patch.object(chat, "ChatRunner", return_value=types.SimpleNamespace(vision=None)), \
                 mock.patch.object(cli.widecheck, "setup", return_value=None), \
                 contextlib.redirect_stderr(err_q):
                _, eng_q, _, _ = cli._open_stack(args_q, warm=False)
            self.assertEqual(seen_ctx, [gpu.YARN2_MAX_CTX])
            self.assertEqual(eng_q.max_ctx, gpu.YARN2_MAX_CTX)
            self.assertIn("2x YaRN", err_q.getvalue())

            # Explicit --max-ctx <= native 262144 notes that context is not extended
            args_short = cli.build_parser().parse_args(
                ["serve", "--model-dir", str(pack.root), "--dll", str(dll), "--max-ctx", "131072", "--yarn2"]
            )
            err_short = io.StringIO()
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch.object(engine, "Engine", FakeEngine), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()), \
                 mock.patch.object(chat, "ChatRunner", return_value=types.SimpleNamespace(vision=None)), \
                 mock.patch.object(cli.widecheck, "setup", return_value=None), \
                 contextlib.redirect_stderr(err_short):
                cli._open_stack(args_short, warm=False)
            self.assertIn("does not exceed the model's native", err_short.getvalue())

    def test_gpu_plan_context_with_yarn2_limit_respects_vram_on_32gb_and_24gb(self):
        g5090 = gpu.GpuInfo(index=0, name="NVIDIA GeForce RTX 5090", uuid="GPU-1", compute_cap=(12, 0),
                            total_mib=32607, used_mib=1100, driver="590.0")
        det5090 = gpu.Detected(gpus=[g5090], candidates=[g5090], device=0, numbered=True)
        g4090 = gpu.GpuInfo(index=0, name="NVIDIA GeForce RTX 4090", uuid="GPU-2", compute_cap=(8, 9),
                            total_mib=24564, used_mib=3500, driver="590.0")
        det4090 = gpu.Detected(gpus=[g4090], candidates=[g4090], device=0, numbered=True)

        with tempfile.TemporaryDirectory() as t:
            # Quality model (~11,826 MiB) on RTX 5090: fits 524,288 with draft head
            pack_q, stat_q = _fake_pack_with_sizes(Path(t) / "q", weights_mib=11826)
            with mock.patch.object(Path, "stat", stat_q):
                plain = gpu.plan_context(pack_q, det5090)
                self.assertIsNone(plain.max_ctx)
                plan_q = gpu.plan_context(pack_q, det5090, max_ctx_limit=gpu.YARN2_MAX_CTX)
                self.assertEqual(plan_q.max_ctx, 524288)
                self.assertNotIn("YSKN_BDR_SHEAD", plan_q.env)

                # Quality model on 24 GB card: fits <= 262,144 and explains --yarn2 gives no longer context
                plan_24 = gpu.plan_context(pack_q, det4090, max_ctx_limit=gpu.YARN2_MAX_CTX)
                assert plan_24.max_ctx is not None
                self.assertLessEqual(plan_24.max_ctx, pack_q.max_ctx)
                self.assertTrue(any("native" in n and "--yarn2" in n for n in plan_24.notes), plan_24.notes)

                # Parallel on 32 GB card: per-slot context fits <= 262,144, notes --yarn2 gives no longer context
                plan_par = gpu.plan_parallel(pack_q, det5090, 2, max_ctx_limit=gpu.YARN2_MAX_CTX)
                assert plan_par.max_ctx is not None
                self.assertLessEqual(plan_par.max_ctx, pack_q.max_ctx)
                self.assertTrue(any("native" in n and "--yarn2" in n for n in plan_par.notes), plan_par.notes)

            # Max-quality model (~14,973 MiB) on RTX 5090: auto-plans < 524,288 to fit; explicit 524,288 drops draft head
            pack_mq, stat_mq = _fake_pack_with_sizes(Path(t) / "mq", weights_mib=14973)
            with mock.patch.object(Path, "stat", stat_mq):
                plan_mq = gpu.plan_context(pack_mq, det5090, max_ctx_limit=gpu.YARN2_MAX_CTX)
                assert plan_mq.max_ctx is not None
                self.assertGreater(plan_mq.max_ctx, pack_mq.max_ctx)
                self.assertLess(plan_mq.max_ctx, gpu.YARN2_MAX_CTX)
                self.assertEqual(plan_mq.max_ctx % gpu.CONTEXT_STEP, 0)

                given_mq = gpu.plan_given_context(pack_mq, det5090, gpu.YARN2_MAX_CTX)
                self.assertEqual(given_mq.max_ctx, gpu.YARN2_MAX_CTX)
                self.assertEqual(given_mq.env.get("YSKN_BDR_SHEAD"), "0")

    def test_open_stack_sets_yarn2_env_and_isolates_parallel_check_key(self):
        with tempfile.TemporaryDirectory() as t:
            pack = support.make_pack(Path(t))
            dll = Path(t) / "fake.so"
            dll.write_bytes(b"fake-engine")
            seen_profiles = []

            class FakeEngine:
                def __init__(self, _pack, *, max_ctx=None, device=0, dll=None, profile=None, wide=None, lib=None):
                    self.max_ctx = max_ctx or _pack.max_ctx
                    self.removed_env = []
                    self.lib = types.SimpleNamespace(path=dll)
                    seen_profiles.append(profile)

                def close(self):
                    pass

            args = cli.build_parser().parse_args(
                ["serve", "--model-dir", str(pack.root), "--dll", str(dll), "--max-ctx", "524288", "--yarn2"]
            )
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch.object(engine, "Engine", FakeEngine), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()), \
                 mock.patch.object(chat, "ChatRunner", return_value=types.SimpleNamespace(vision=None)), \
                 mock.patch.object(cli.widecheck, "setup", return_value=None):
                _, eng, _, _ = cli._open_stack(args, warm=False)
            self.assertEqual(len(seen_profiles), 1)
            self.assertIsNotNone(seen_profiles[0])
            self.assertEqual(seen_profiles[0].env["YSKN_Q35_YARN2"], "1")

            g = gpu.GpuInfo(index=0, name="RTX 5090", uuid="GPU-1", compute_cap=(12, 0), total_mib=32607, used_mib=1000, driver="590.0")
            det = gpu.Detected(gpus=[g], candidates=[g], device=0, numbered=True)
            sched = types.SimpleNamespace(n_slots=2, rows=16, drafter=True)
            a_off = types.SimpleNamespace(gpu_detected=det, yarn2=False)
            a_on = types.SimpleNamespace(gpu_detected=det, yarn2=True)
            k_off = cli._check_key(a_off, pack, eng, sched)
            k_on = cli._check_key(a_on, pack, eng, sched)
            self.assertIsNotNone(k_off)
            self.assertIsNotNone(k_on)
            self.assertNotEqual(k_off, k_on)


if __name__ == "__main__":
    unittest.main()
