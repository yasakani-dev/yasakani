# SPDX-License-Identifier: Apache-2.0
"""Metal adapter contracts without a GPU, model download, or tokenizer fixture."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from yasakani import cli
from yasakani._native import YasakaniError
from yasakani.engine import CONTEXT_MARGIN, Sampling
from yasakani.metal_engine import MetalEngine
from yasakani.runtime import ModelPack


class FakeModel:
    def __init__(self, logits=(0.0, 1.0, 2.0, 0.5), hook=None):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.hook = hook
        self.position = 0
        self.calls = []

    def reset(self):
        self.position = 0

    def forward(self, token, *, logits=True):
        if not 0 <= token < len(self.logits):
            raise ValueError("token is outside the model vocabulary")
        self.calls.append(token)
        self.position += 1
        if self.hook:
            self.hook(self.position)
        return self.logits.copy() if logits else None

    def state_hash(self):
        return np.asarray([self.position], dtype=np.uint64)


class MetalEngineTests(unittest.TestCase):
    def engine(self, model=None, **kwargs):
        engine = MetalEngine(SimpleNamespace(max_ctx=262144), model=model or FakeModel(), **kwargs)
        self.addCleanup(engine.close)
        return engine

    def run_blocked(self, engine, model, block_at, *, close=False):
        entered, release = threading.Event(), threading.Event()
        result, errors = [], []

        def hook(position):
            if position == block_at:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release the forward call")

        def generate():
            try:
                result.append(engine.generate([0, 1], 50, stop_ids=None))
            except BaseException as exc:
                errors.append(exc)

        model.hook = hook
        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        closer = None
        try:
            self.assertTrue(entered.wait(3), "generation did not reach the requested phase")
            if close:
                closer = threading.Thread(target=engine.close, daemon=True)
                closer.start()
                self.assertTrue(engine._cancel.wait(3), "close did not request cancellation")
            else:
                engine.cancel()
        finally:
            release.set()
            worker.join(3)
            if closer:
                closer.join(3)
            model.hook = None
        self.assertFalse(worker.is_alive(), "generation left a running worker")
        if closer:
            self.assertFalse(closer.is_alive(), "close did not release the engine")
        self.assertEqual(errors, [])
        self.assertEqual(len(result), 1)
        return result[0]

    def test_cancel_during_prefill_emits_no_tokens_and_can_restart(self):
        model = FakeModel()
        engine = self.engine(model)
        result = self.run_blocked(engine, model, 1)
        self.assertEqual((result.finish, result.ids), ("cancelled", []))
        self.assertEqual(model.calls, [0])
        engine.set_sampling(Sampling(temperature=0))
        self.assertEqual(engine.generate([0], 2, stop_ids=None).ids, [2, 2])

    def test_cancel_during_decode_keeps_only_committed_tokens(self):
        model = FakeModel()
        engine = self.engine(model)
        engine.set_sampling(Sampling(temperature=0))
        result = self.run_blocked(engine, model, 3)
        self.assertEqual((result.finish, result.ids), ("cancelled", [2]))
        self.assertEqual(model.calls, [0, 1, 2])

    def test_callback_failure_propagates_without_running_more_tokens(self):
        model = FakeModel()
        engine = self.engine(model)
        engine.set_sampling(Sampling(temperature=0))
        received = []

        def fail(ids):
            received.append(ids.tolist())
            raise RuntimeError("consumer disconnected")

        with self.assertRaisesRegex(RuntimeError, "consumer disconnected"):
            engine.generate([0, 1], 1000, stop_ids=None, on_tokens=fail)
        self.assertEqual(received, [[2]])
        self.assertEqual(model.calls, [0, 1])
        self.assertEqual(engine.generate([0], 2, stop_ids=None).ids, [2, 2])

    def test_callback_cancel_stops_before_next_forward(self):
        model = FakeModel()
        engine = self.engine(model)
        result = engine.generate([0], 30, stop_ids=None, on_tokens=lambda _: engine.cancel())
        self.assertEqual(result.finish, "cancelled")
        self.assertEqual(len(result.ids), 1)
        self.assertEqual(model.calls, [0])

    def test_close_during_generation_cancels_and_waits(self):
        engine = self.engine()
        result = self.run_blocked(engine, engine.model, 1, close=True)
        self.assertEqual(result.finish, "cancelled")
        with self.assertRaisesRegex(YasakaniError, "closed"):
            engine.generate([0], 1)

    def test_closed_lifecycle_and_context_manager(self):
        engine = self.engine()
        with engine:
            engine.generate([0], 1)
        engine.close()  # repeated close is safe
        engine.cancel()
        for operation in (engine.reset, engine.position, engine.state_hash,
                          lambda: engine.set_sampling(Sampling()), lambda: engine.generate([0], 1)):
            with self.subTest(operation=operation), self.assertRaisesRegex(YasakaniError, "closed"):
                operation()

    def test_reset_and_each_request_start_with_fresh_state(self):
        engine = self.engine()
        engine.generate([0, 1], 3, stop_ids=None)
        self.assertEqual(engine.position(), 4)
        self.assertEqual(engine.state_hash().tolist(), [4])
        result = engine.generate([0], 1, stop_ids=None)
        self.assertEqual((engine.position(), result.cached_tokens, result.snap_mode), (1, 0, 0))
        engine.reset()
        self.assertEqual(engine.state_hash().tolist(), [0])

    def test_seed_repeats_across_requests_and_instances(self):
        engine = self.engine(FakeModel(logits=(0, 0, 0, 0)))
        engine.set_sampling(Sampling(temperature=1, top_p=1, seed=123))
        first = engine.generate([0], 80, stop_ids=None).ids
        self.assertEqual(first, engine.generate([0], 80, stop_ids=None).ids)
        other = self.engine(FakeModel(logits=(0, 0, 0, 0)))
        other.set_sampling(Sampling(temperature=1, top_p=1, seed=123))
        self.assertEqual(first, other.generate([0], 80, stop_ids=None).ids)
        engine.set_sampling(Sampling(temperature=1, top_p=1, seed=456))
        self.assertNotEqual(first, engine.generate([0], 80, stop_ids=None).ids)

    def test_sampling_filters_and_stable_greedy_ties(self):
        engine = self.engine(FakeModel(logits=(2, 2, 0, -1)))
        engine.set_sampling(Sampling(temperature=0))
        self.assertEqual(engine.generate([3], 10, stop_ids=None).ids, [0] * 10)
        engine = self.engine(FakeModel(logits=(0, 1, 2, 0.5)))
        for sampling in (Sampling(top_k=1), Sampling(top_p=0.01), Sampling(min_p=0.99)):
            with self.subTest(sampling=sampling):
                engine.set_sampling(sampling)
                self.assertEqual(engine.generate([0], 10, stop_ids=None).ids, [2] * 10)

    def test_sampling_validation_and_invalid_logits(self):
        engine = self.engine()
        for sampling in (Sampling(temperature=float("nan")), Sampling(temperature=float("inf")),
                         Sampling(top_p=0), Sampling(top_p=1.1), Sampling(min_p=-0.1), Sampling(min_p=1.1)):
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                engine.set_sampling(sampling)
        for logits in ((0, float("nan")), (0, float("inf")), [[0, 1]]):
            with self.subTest(logits=logits):
                engine.model.logits = np.asarray(logits)
                with self.assertRaisesRegex(YasakaniError, "invalid logits"):
                    engine.generate([0], 1)

    def test_stop_and_streamed_prefixes(self):
        engine = self.engine()
        engine.set_sampling(Sampling(temperature=0))
        seen = []
        result = engine.generate([0, 1], 20, stop_ids=(2,), on_tokens=lambda ids: seen.append(ids.tolist()))
        self.assertEqual((result.finish, result.ids, seen), ("stop", [2], [[2]]))
        seen.clear()
        result = engine.generate([0], 3, stop_ids=None, on_tokens=lambda ids: seen.append(ids.tolist()))
        self.assertEqual((result.finish, result.ids), ("length", [2, 2, 2]))
        self.assertEqual(seen, [[2], [2, 2], [2, 2, 2]])

    def test_numpy_stop_vector_is_accepted_without_truth_value_coercion(self):
        engine = self.engine()
        engine.set_sampling(Sampling(temperature=0))
        result = engine.generate([0], 20, stop_ids=np.asarray([1, 2], dtype=np.int32))
        self.assertEqual((result.finish, result.ids), ("stop", [2]))
        result = engine.generate([0], 2, stop_ids=np.asarray([], dtype=np.int32))
        self.assertEqual((result.finish, result.ids), ("length", [2, 2]))

    def test_context_defaults_and_bounds(self):
        self.assertEqual(self.engine().max_ctx, 4096)
        for context in (CONTEXT_MARGIN, 262145):
            with self.subTest(context=context), self.assertRaises(YasakaniError):
                self.engine(max_ctx=context)
        engine = self.engine(max_ctx=CONTEXT_MARGIN + 6)
        result = engine.generate([0] * 4, 100, stop_ids=None)
        self.assertEqual(len(result.ids), 3)
        self.assertLessEqual(engine.position(), engine.max_ctx)
        with self.assertRaisesRegex(ValueError, "no room"):
            engine.generate([0] * 6, 1)

    def test_invalid_prompt_and_generation_options(self):
        engine = self.engine()
        for prompt in ([], [[0, 1]], [0.1], [True], ["0"]):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                engine.generate(prompt, 1)
        for maximum in (0, -1):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                engine.generate([0], maximum)
        with self.assertRaisesRegex(ValueError, "speculative"):
            engine.generate([0], 1, spec={"gamma": 5})
        with self.assertRaisesRegex(YasakaniError, "device"):
            self.engine(device=1)


class MetalRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("qwen38_27b_e3x.yskn", "tokenizer.json", "tokenizer_config.json"):
            (self.root / name).write_bytes(b"fixture")

    def test_metal_pack_needs_no_cuda_drafter_or_kv_profile(self):
        pack = ModelPack.load(self.root, require_aux=False)
        self.assertEqual(pack.weights, self.root / "qwen38_27b_e3x.yskn")
        with self.assertRaisesRegex(YasakaniError, "missing"):
            ModelPack.load(self.root)
        (self.root / "tokenizer.json").unlink()
        with self.assertRaisesRegex(YasakaniError, "tokenizer.json"):
            ModelPack.load(self.root, require_aux=False)

    def test_backend_selection_preserves_explicit_cuda_library(self):
        cases = [("darwin", "auto", None, "metal"), ("linux", "auto", None, "cuda"),
                 ("win32", "auto", None, "cuda"), ("darwin", "auto", "stub.so", "cuda"),
                 ("darwin", "cuda", None, "cuda"), ("darwin", "metal", "metal.dylib", "metal")]
        for platform, backend, dll, expected in cases:
            with self.subTest(platform=platform, backend=backend, dll=dll), mock.patch.object(cli.sys, "platform", platform):
                self.assertEqual(cli._backend(SimpleNamespace(backend=backend, dll=dll)), expected)

    def test_open_stack_routes_to_metal_without_cuda_auxiliaries(self):
        args = cli.build_parser().parse_args(["serve", "--backend", "metal", "--model-dir", str(self.root),
                                              "--max-ctx", "512", "--dll", "metal.dylib"])
        engine = mock.Mock(max_ctx=512, removed_env=[])
        with mock.patch("yasakani.textio.ChatFormatter") as formatter, \
                mock.patch("yasakani.metal_engine.MetalEngine", return_value=engine) as metal, \
                mock.patch("yasakani.engine.Engine") as cuda, \
                mock.patch("yasakani.chat.ChatRunner") as runner, \
                mock.patch.object(cli, "_vision", return_value=None), \
                mock.patch.object(cli, "_gpu_memory_warning") as nvidia, mock.patch.object(cli, "_say"):
            pack, opened, fmt, chat = cli._open_stack(args, warm=False)
        metal.assert_called_once_with(pack, max_ctx=512, device=0, dll="metal.dylib")
        cuda.assert_not_called()
        nvidia.assert_not_called()
        self.assertIs(opened, engine)
        self.assertIs(fmt, formatter.return_value)
        self.assertIs(chat, runner.return_value)
        runner.return_value.warmup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
