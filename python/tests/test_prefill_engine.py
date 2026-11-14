# SPDX-License-Identifier: Apache-2.0
"""Generation preserves cancellation and bounded prefill at the engine boundary."""
import unittest
import numpy as np
from tests import test_metal_engine as fixtures
from yasakani.engine import Sampling


class PrefillModel(fixtures.FakeModel):
    def __init__(self):
        super().__init__()
        self.batches = []
        self.on_prefill = None

    def prefill(self, tokens, *, cancel=None, logits=True):
        self.batches.append((len(tokens), logits))
        if self.on_prefill:
            self.on_prefill()
        if cancel and cancel():
            return None
        self.position += len(tokens)
        return self.logits.copy() if logits else None


class PrefillEngineTests(unittest.TestCase):
    engine = fixtures.MetalEngineTests.engine

    def test_bounded_prefill_only_projects_final_logits(self):
        model = PrefillModel()
        engine = self.engine(model)
        engine.set_sampling(Sampling(temperature=0))
        result = engine.generate([0] * 70, 3, stop_ids=None)
        self.assertEqual(model.batches, [(32, False), (32, False), (6, True)])
        self.assertEqual(result.ids, [2, 2, 2])
        self.assertEqual(model.calls, [2, 2])
        self.assertEqual(result.prompt_tokens, 70)
        self.assertEqual(result.cached_tokens, 0)
        self.assertEqual(model.position, 72)

    def test_cancel_in_prefill_emits_nothing_and_skips_later_chunks(self):
        model = PrefillModel()
        engine = self.engine(model)
        model.on_prefill = engine.cancel
        result = engine.generate([0] * 70, 10, stop_ids=np.array([2]))
        self.assertEqual(result.finish, "cancelled")
        self.assertEqual(result.ids, [])
        self.assertEqual(model.batches, [(32, False)])
        self.assertEqual(model.calls, [])
        model.on_prefill = None
        engine.set_sampling(Sampling(temperature=0))
        self.assertEqual(engine.generate([0], 1).ids, [2])

    def test_prefill_failure_propagates_without_decode(self):
        model = PrefillModel()
        engine = self.engine(model)
        def failure():
            raise RuntimeError("GPU error")
        model.on_prefill = failure
        with self.assertRaisesRegex(RuntimeError, "GPU error"):
            engine.generate([0] * 40, 10)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
