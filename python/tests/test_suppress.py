# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import io
import json
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from typing import List, Optional
from unittest import mock

from tests import support
from yasakani import chat, cli, engine, gpu, runtime, textio
from yasakani._native import YasakaniError
from yasakani.engine import Engine, GenResult, Sampling
from yasakani.runtime import (
    EOS_IDS,
    IM_START,
    MIN_UNSUPPRESSED_TOKENS,
    VOCAB_SIZE,
    suppress_fingerprint,
)
from yasakani.sched import ScheduledEngine, Scheduler, consistency_check


def _make_pack_with_tokenizer(tmp: Path) -> runtime.ModelPack:
    pack = support.make_pack(tmp)
    tok = {
        "model": {"vocab": {"hello": 1001, "world": 1002, "foo": 1003}},
        "added_tokens": [
            {"id": 248044, "content": "<|endoftext|>", "special": True},
            {"id": 248045, "content": "<|im_start|>", "special": True},
            {"id": 248046, "content": "<|im_end|>", "special": True},
            {"id": 248050, "content": "<|custom_special|>", "special": True},
            {"id": 248058, "content": "<tool_call>", "special": False},
            {"id": 248059, "content": "</tool_call>", "special": False},
            {"id": 248066, "content": "<tool_response>", "special": False},
            {"id": 248067, "content": "</tool_response>", "special": False},
            {"id": 248068, "content": "<think>", "special": False},
            {"id": 248069, "content": "</think>", "special": False},
            {"id": 1004, "content": "extra_tok", "special": False},
        ],
    }
    (pack.root / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    return pack


class SuppressLoaderTests(unittest.TestCase):
    def test_loads_all_supported_json_formats_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = _make_pack_with_tokenizer(root)
            tok_sha = hashlib.sha256((pack.root / "tokenizer.json").read_bytes()).hexdigest()

            f_list = root / "plain.json"
            f_list.write_text(json.dumps([1010, 1005, 1010]), encoding="utf-8")

            f_tc = root / "thakicloud.json"
            f_tc.write_text(json.dumps({
                "name": "zh-TW-t1",
                "target_token_ids": [1005, 1020],
                "n_target": 2,
                "vocab_size": VOCAB_SIZE,
                "tokenizer_sha256": tok_sha,
            }), encoding="utf-8")

            f_hf = root / "hf.json"
            f_hf.write_text(json.dumps({"suppress_tokens": [1030], "token_ids": [1031]}), encoding="utf-8")

            f_toks = root / "tokens.json"
            f_toks.write_text(json.dumps({"tokens": ["hello", "extra_tok"]}), encoding="utf-8")

            ids = pack.load_suppress_tokens([f_list, f_tc, f_hf, f_toks])
            self.assertEqual(ids, (1001, 1004, 1005, 1010, 1020, 1030, 1031))

    def test_rejects_bad_integrity_fields_types_and_ranges(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = _make_pack_with_tokenizer(root)

            cases = [
                ("bad_n_target.json", {"target_token_ids": [10, 11], "n_target": 3}, "n_target"),
                ("bad_vocab.json", {"token_ids": [10], "vocab_size": 128000}, "vocab_size"),
                ("bad_sha.json", {"token_ids": [10], "tokenizer_sha256": "0" * 64}, "SHA-256"),
                ("bad_bool.json", [True, 10], "integer"),
                ("bad_range.json", [VOCAB_SIZE], "out of range"),
                ("bad_neg.json", [-1], "out of range"),
                ("bad_tok.json", {"tokens": ["not_in_vocab"]}, "not found"),
                ("bad_obj.json", {"unrelated": [1, 2]}, "expected a JSON array"),
            ]
            for name, payload, pattern in cases:
                p = root / name
                p.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(case=name):
                    with self.assertRaisesRegex(YasakaniError, pattern):
                        pack.load_suppress_tokens([p])

    def test_loads_utf8_bom_mask_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = _make_pack_with_tokenizer(root)
            f_arr = root / "bom_arr.json"
            f_arr.write_bytes(b"\xef\xbb\xbf" + json.dumps([1001, 1002]).encode("utf-8"))
            f_obj = root / "bom_obj.json"
            f_obj.write_bytes(b"\xef\xbb\xbf" + json.dumps({"tokens": ["foo"]}).encode("utf-8"))
            self.assertEqual(pack.load_suppress_tokens([f_arr, f_obj]), (1001, 1002, 1003))

    def test_protects_special_control_and_eos_tokens_unless_overridden(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = _make_pack_with_tokenizer(root)
            protected = (
                EOS_IDS[0], EOS_IDS[1], IM_START, 248050,
                248058, 248059, 248066, 248067, 248068, 248069,
            )
            for bad_id in protected:
                p = root / f"special_{bad_id}.json"
                p.write_text(json.dumps([1001, bad_id]), encoding="utf-8")
                with self.subTest(bad_id=bad_id):
                    with self.assertRaisesRegex(YasakaniError, "allow-suppress-special"):
                        pack.load_suppress_tokens([p])
                    self.assertEqual(pack.load_suppress_tokens([p], allow_special=True), (1001, bad_id))

    def test_requires_minimum_remaining_unmasked_tokens(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = _make_pack_with_tokenizer(root)
            too_many = root / "too_many.json"
            too_many.write_text(json.dumps(list(range(VOCAB_SIZE - MIN_UNSUPPRESSED_TOKENS + 1))), encoding="utf-8")
            with self.assertRaisesRegex(YasakaniError, f"at least {MIN_UNSUPPRESSED_TOKENS}"):
                pack.load_suppress_tokens([too_many], allow_special=True)

            at_limit = root / "at_limit.json"
            at_limit.write_text(json.dumps(list(range(VOCAB_SIZE - MIN_UNSUPPRESSED_TOKENS))), encoding="utf-8")
            ids = pack.load_suppress_tokens([at_limit], allow_special=True)
            self.assertEqual(len(ids), VOCAB_SIZE - MIN_UNSUPPRESSED_TOKENS)


class SuppressStubEngineTests(unittest.TestCase):
    stub: support.Stub
    tmp: tempfile.TemporaryDirectory[str]
    e: Engine

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.stub.clear_scripts()
        self.e = support.open_engine(Path(self.tmp.name), self.stub)
        self.stub.model(self.e, True, eos_every=0, vocab=32)

    def tearDown(self):
        self.e.close()
        self.tmp.cleanup()
        self._env.stop()

    def test_single_stream_suppress_and_clear(self):
        self.e.set_sampling(Sampling(temperature=0.0))
        baseline = self.e.generate([5, 6, 7], 40, stop_ids=None).ids
        banned = set(baseline[:10])
        self.assertTrue(self.e.supports_suppress)
        self.assertEqual(self.e.set_suppress(sorted(banned)), len(banned))
        masked = self.e.generate([5, 6, 7], 40, stop_ids=None).ids
        self.assertEqual(banned.intersection(masked), set())
        self.assertNotEqual(masked, baseline)

        self.assertEqual(self.e.set_suppress(None), 0)
        restored = self.e.generate([5, 6, 7], 40, stop_ids=None).ids
        self.assertEqual(restored, baseline)

    def test_stub_enforces_minimum_remaining_unmasked_tokens(self):
        too_many = list(range(VOCAB_SIZE - MIN_UNSUPPRESSED_TOKENS + 1))
        with self.assertRaisesRegex(YasakaniError, f"at least {MIN_UNSUPPRESSED_TOKENS}"):
            self.e.set_suppress(too_many)
        at_limit = list(range(VOCAB_SIZE - MIN_UNSUPPRESSED_TOKENS))
        self.assertEqual(self.e.set_suppress(at_limit), len(at_limit))
        self.e.set_suppress(None)

    def test_parallel_scheduler_matches_single_stream_and_honors_suppression(self):
        banned = set(range(1000, 1016))
        self.e.set_suppress(sorted(banned))
        sched = Scheduler(self.e, 3)
        try:
            prompts = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
            samplings = [
                Sampling(temperature=0.0),
                Sampling(temperature=0.7, top_p=0.95, top_k=20, seed=11),
                Sampling(temperature=0.7, top_p=0.95, top_k=20, seed=22),
            ]
            res = consistency_check(sched, prompts, max_new=32, samplings=samplings)
            self.assertTrue(res.ok, res.line())

            outs: List[Optional[GenResult]] = [None, None, None]
            bar = threading.Barrier(3)

            def run_one(i: int) -> None:
                bar.wait(10)
                with ScheduledEngine(sched) as se:
                    se.set_sampling(samplings[i])
                    outs[i] = se.generate(prompts[i], 32, stop_ids=None)

            threads = [threading.Thread(target=run_one, args=(i,)) for i in range(3)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            for i, out in enumerate(outs):
                assert out is not None
                self.assertEqual(banned.intersection(out.ids), set(), f"slot {i} emitted suppressed IDs")
        finally:
            sched.close()


class SuppressCliTests(unittest.TestCase):
    def test_parser_only_accepts_suppress_on_serve_and_chat(self):
        parser = cli.build_parser()
        for cmd in ("serve", "chat"):
            args = parser.parse_args([cmd, "--suppress-tokens", "mask.json", "--allow-suppress-special"])
            self.assertEqual(args.suppress_tokens, [Path("mask.json")])
            self.assertTrue(args.allow_suppress_special)
        for cmd in ("selfcheck", "bench"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    parser.parse_args([cmd, "--suppress-tokens", "mask.json"])
            self.assertEqual(cm.exception.code, 2)

    def test_open_stack_fails_before_engine_on_bad_mask_or_old_dll_and_isolates_check_key(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = _make_pack_with_tokenizer(root)
            dll = root / "fake.so"
            dll.write_bytes(b"fake-engine")
            bad_mask = root / "bad.json"
            bad_mask.write_text(json.dumps([EOS_IDS[0]]), encoding="utf-8")
            good_mask = root / "good.json"
            good_mask.write_text(json.dumps([1001, 1002]), encoding="utf-8")

            opened = []
            applied_ids = []

            class FakeEngine:
                def __init__(self, _pack, *, max_ctx=None, device=0, dll=None, profile=None, wide=None, lib=None):
                    opened.append(True)
                    self.max_ctx = max_ctx or _pack.max_ctx
                    self.removed_env = []
                    self.lib = types.SimpleNamespace(path=dll)

                def set_suppress(self, ids):
                    applied_ids.append(tuple(ids))
                    return len(ids)

                def close(self):
                    pass

            bad_args = cli.build_parser().parse_args(
                ["serve", "--model-dir", str(pack.root), "--dll", str(dll), "--suppress-tokens", str(bad_mask)]
            )
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch.object(engine, "Engine", FakeEngine), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()):
                with self.assertRaisesRegex(YasakaniError, "allow-suppress-special"):
                    cli._open_stack(bad_args, warm=False)
            self.assertEqual(opened, [])

            # Older engine library without yskn_set_suppress must fail before Engine opens the model
            old_lib = types.SimpleNamespace(supports_suppress=False, supports_parallel=False, path=dll)
            old_args = cli.build_parser().parse_args(
                ["serve", "--model-dir", str(pack.root), "--dll", str(dll), "--suppress-tokens", str(good_mask)]
            )
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch.object(cli, "NativeLib", return_value=old_lib), \
                 mock.patch.object(engine, "Engine", FakeEngine), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()):
                with self.assertRaisesRegex(YasakaniError, "yskn_set_suppress"):
                    cli._open_stack(old_args, warm=False)
            self.assertEqual(opened, [])

            new_lib = types.SimpleNamespace(supports_suppress=True, supports_parallel=True, path=dll)
            env_args = cli.build_parser().parse_args(["serve", "--model-dir", str(pack.root), "--dll", str(dll)])
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none", "YASAKANI_SUPPRESS_TOKENS": str(good_mask)}), \
                 mock.patch.object(cli, "NativeLib", return_value=new_lib), \
                 mock.patch.object(engine, "Engine", FakeEngine), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()), \
                 mock.patch.object(chat, "ChatRunner", return_value=types.SimpleNamespace(vision=None)), \
                 mock.patch.object(cli.widecheck, "setup", return_value=None):
                _, eng, _, _ = cli._open_stack(env_args, warm=False)
            self.assertEqual(applied_ids, [(1001, 1002)])
            self.assertEqual(env_args.suppress_hash, suppress_fingerprint((1001, 1002)))

            g = gpu.GpuInfo(index=0, name="RTX 5090", uuid="GPU-1", compute_cap=(12, 0), total_mib=32607, used_mib=1000, driver="590.0")
            det = gpu.Detected(gpus=[g], candidates=[g], device=0, numbered=True)
            sched = types.SimpleNamespace(n_slots=2, rows=16, drafter=True)
            k_none = cli._check_key(types.SimpleNamespace(gpu_detected=det, suppress_hash=None), pack, eng, sched)
            k_mask = cli._check_key(types.SimpleNamespace(gpu_detected=det, suppress_hash=env_args.suppress_hash), pack, eng, sched)
            self.assertNotEqual(k_none, k_mask)

    def test_selfcheck_and_bench_ignore_env_mask_with_note(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pack = _make_pack_with_tokenizer(root)
            dll = root / "fake.so"
            dll.write_bytes(b"fake-engine")
            good_mask = root / "good.json"
            good_mask.write_text(json.dumps([1001, 1002]), encoding="utf-8")

            applied_ids = []

            class FakeEngine:
                def __init__(self, _pack, *, max_ctx=None, device=0, dll=None, profile=None, wide=None, lib=None):
                    self.max_ctx = max_ctx or _pack.max_ctx
                    self.removed_env = []
                    self.lib = types.SimpleNamespace(path=dll)

                def set_suppress(self, ids):
                    applied_ids.append(tuple(ids))
                    return len(ids)

                def close(self):
                    pass

            for cmd in ("selfcheck", "bench"):
                args = cli.build_parser().parse_args([cmd, "--model-dir", str(pack.root), "--dll", str(dll)])
                err = io.StringIO()
                with self.subTest(cmd=cmd), \
                     mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none", "YASAKANI_SUPPRESS_TOKENS": str(good_mask)}), \
                     mock.patch.object(engine, "Engine", FakeEngine), \
                     mock.patch.object(textio, "ChatFormatter", return_value=object()), \
                     mock.patch.object(chat, "ChatRunner", return_value=types.SimpleNamespace(vision=None)), \
                     mock.patch.object(cli.widecheck, "setup", return_value=None), \
                     contextlib.redirect_stderr(err):
                    cli._open_stack(args, warm=False, allow_suppress=False)
                self.assertEqual(applied_ids, [])
                self.assertIsNone(args.suppress_hash)
                self.assertIn("ignoring YASAKANI_SUPPRESS_TOKENS", err.getvalue())

            # Programmatic suppression path on selfcheck/bench is refused before model load
            args = cli.build_parser().parse_args(["selfcheck", "--model-dir", str(pack.root), "--dll", str(dll)])
            args.suppress_tokens = [good_mask]
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}):
                with self.assertRaisesRegex(YasakaniError, "does not support --suppress-tokens"):
                    cli._open_stack(args, warm=False, allow_suppress=False)


if __name__ == "__main__":
    unittest.main()
