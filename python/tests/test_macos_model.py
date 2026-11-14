# SPDX-License-Identifier: Apache-2.0
"""Numerical checks against independent dense/reference calculations; no GPU needed."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from yasakani.macos_model import Model, delta_step, grouped_attention, rotary
from yasakani.yskn_format import Container, FormatError, NumpyOps, checked, fwht512, rotated_rows, vector


ROOT = Path(__file__).resolve().parents[2]


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("yskn_decode_reference", ROOT / "tools/yskn_decode.py")
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)
        cls.path = ROOT / "tools/testdata/tiny.yskn"

    def test_compressed_operations_match_independent_dense_decoder(self):
        c = Container(self.path)
        reference = self.reference.Container(self.path)
        ops = NumpyOps(c, block_rows=37)  # deliberately cross warp/tile boundaries
        rng = np.random.default_rng(73)
        for name, desc in c.tensors.items():
            with self.subTest(name=name):
                t = checked(desc)
                expected = self.reference.decode(reference, self.reference.checked(desc))
                x = rng.normal(size=t["cols"]).astype(np.float32)
                np.testing.assert_allclose(ops.linear(name, x), expected @ x, rtol=3e-5, atol=3e-6)
                for lo, hi in ((0, 1), (17, 38), (t["rows"] - 1, t["rows"])):
                    np.testing.assert_array_equal(rotated_rows(c, t, lo, hi),
                                                  self.reference.rotated_rows(reference, self.reference.checked(desc), lo, hi))
        for name, desc in c.vectors.items():
            np.testing.assert_array_equal(vector(c, desc), self.reference.vector(reference, desc))

    def test_hadamard_is_normalized_involution(self):
        x = np.random.default_rng(8).normal(size=(3, 1024)).astype(np.float32)
        np.testing.assert_allclose(fwht512(fwht512(x)), x, atol=8e-7)
        np.testing.assert_allclose(np.linalg.norm(fwht512(x)), np.linalg.norm(x), rtol=1e-6)

    def test_embedding_adds_both_residual_tables(self):
        c = Container(self.path)
        desc = c.tensors["tiny.embed"]
        c.tensors = {name: dict(desc, name=name) for name in ("embed_tokens", "embed_tokens.res", "embed_tokens.res2")}
        ops = NumpyOps(c)
        reference = self.reference.Container(self.path)
        expected = self.reference.decode(reference, self.reference.checked(desc))
        for token in (0, 31, 95):
            np.testing.assert_allclose(ops.embedding(token), 3 * expected[token], rtol=3e-5, atol=3e-7)
        for token in (-1, 96):
            with self.assertRaises(ValueError):
                ops.embedding(token)

    def test_truncated_container_is_rejected_before_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.yskn"
            path.write_bytes(self.path.read_bytes()[:64])
            with self.assertRaises(FormatError):
                Container(path)

    def test_bad_offsets_and_unsupported_representation_fail(self):
        c = Container(self.path)
        for offset, count in ((-1, 1), (0, -1), (c.size, 1)):
            with self.assertRaises(FormatError):
                c.array(offset, "<f4", count)
        for key, value in (("start", "carry"), ("bits", 3), ("cols", 513), ("stream_bytes", 0)):
            t = dict(next(iter(c.tensors.values())), **{key: value})
            with self.assertRaises(FormatError):
                checked(t)


class TinyOps:
    """Dense injected operations exercise the network at intentionally tiny shapes.

    The synthetic header describes shapes; compressed payload validity is checked
    separately against the upstream fixture by FormatTests.
    """
    def __init__(self, directory):
        rng = np.random.default_rng(120)
        self.weights, self.vectors, self.calls = {}, {}, []
        self.embeddings = rng.normal(0, .3, (8, 4)).astype(np.float32)
        self.config = dict(arch="qwen3_5", layers=2, hidden=4, ffn=6, heads=2, kv_heads=1, head_dim=4,
                           vocab=8, eps=1e-5, rope_theta=10000, rotary_dim=2, tie_embeddings=False,
                           layer_types=["linear_attention", "full_attention"], linear_num_key_heads=1,
                           linear_num_value_heads=2, linear_key_head_dim=3, linear_value_head_dim=3,
                           linear_conv_kernel_dim=4, attn_output_gate=True, norm_offset=1)
        def matrix(name, shape):
            self.weights[name] = rng.normal(0, .2, shape).astype(np.float32)

        def vec(name, size, base=0):
            # Match the actual values represented by the fixture's f16 vectors.
            self.vectors[name] = (base + rng.normal(0, .1, size)).astype(np.float16).astype(np.float32)

        matrix("embed_tokens", (8, 4))
        matrix("lm_head", (8, 4))
        vec("norm", 4)
        for layer in range(2):
            p = "layers.%d." % layer
            vec(p + "input_layernorm", 4)
            vec(p + "post_attention_layernorm", 4)
            matrix(p + "mlp.gate_proj", (6, 4))
            matrix(p + "mlp.up_proj", (6, 4))
            matrix(p + "mlp.down_proj", (4, 6))
        p = "layers.0.linear_attn."
        matrix(p + "in_proj_qkv", (12, 4))
        matrix(p + "in_proj_z", (6, 4))
        matrix(p + "out_proj", (4, 6))
        vec(p + "in_proj_a", 8)
        vec(p + "in_proj_b", 8)
        vec(p + "conv1d", 48, base=.3)
        vec(p + "A_log", 2)
        vec(p + "dt_bias", 2)
        vec(p + "norm", 3, base=1)
        p = "layers.1.self_attn."
        matrix(p + "q_proj", (16, 4))
        matrix(p + "k_proj", (4, 4))
        matrix(p + "v_proj", (4, 4))
        matrix(p + "o_proj", (4, 8))
        vec(p + "q_norm", 4)
        vec(p + "k_norm", 4)
        payload, vectors = bytearray(), []
        for name, value in self.vectors.items():
            vectors.append(dict(name=name, n=len(value), off=8192 + len(payload), dtype="f16"))
            payload.extend(value.astype("<f2").tobytes())
        self.header = dict(model=self.config, tensors=[dict(name=name, rows=w.shape[0], cols=w.shape[1],
                           role="embedding" if name == "embed_tokens" else "body", rot_out=None)
                           for name, w in self.weights.items()], vectors=vectors)
        self.path = Path(directory) / "tiny_model.yskn"
        self._payload = payload
        self.write_header()

    def write_header(self):
        header = json.dumps(self.header).encode()
        assert len(header) + 32 < 8192
        self.path.write_bytes(struct.pack("<4sIQQQ", b"YSKN", 1, len(header), 8192, 8192 + len(self._payload))
                              + header + bytes(8192 - 32 - len(header)) + self._payload)

    def linear(self, name, x):
        self.calls.append(name)
        return self.weights[name] @ x

    def embedding(self, token):
        return self.embeddings[token]

    def vector(self, name):
        return self.vectors[name]


def dense_oracle(ops, tokens):
    """Independent float64 sequence computation with explicit head/state loops."""
    w = {k: v.astype(np.float64) for k, v in ops.weights.items()}
    vec = {k: v.astype(np.float64) for k, v in ops.vectors.items()}
    embeddings = ops.embeddings.astype(np.float64)
    history, state, keys, values = [], np.zeros((2, 3, 3)), [], []
    outputs = []
    def norm(x, name, offset=1):
        return x * (vec[name] + offset) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    for position, token in enumerate(tokens):
        x = embeddings[token].copy()
        for layer in range(2):
            p = "layers.%d." % layer
            z = norm(x, p + "input_layernorm")
            if layer == 0:
                a = p + "linear_attn."
                history.append(w[a + "in_proj_qkv"] @ z)
                convolved = np.zeros(12)
                kernel = vec[a + "conv1d"].reshape(12, 4)
                for lag in range(4):
                    if position - lag >= 0:
                        convolved += history[position - lag] * kernel[:, 3 - lag]
                convolved *= sigmoid(convolved)
                q, k, v = convolved[:3], convolved[3:6], convolved[6:].reshape(2, 3)
                q = q / np.sqrt(q @ q + 1e-6) / np.sqrt(3)
                k = k / np.sqrt(k @ k + 1e-6)
                gate_a = vec[a + "in_proj_a"].reshape(2, 4) @ z
                gate_b = vec[a + "in_proj_b"].reshape(2, 4) @ z
                decay = -np.exp(vec[a + "A_log"]) * np.log1p(np.exp(gate_a + vec[a + "dt_bias"]))
                core = np.zeros((2, 3))
                for head in range(2):
                    state[head] *= np.exp(decay[head])
                    delta = (v[head] - k @ state[head]) * sigmoid(gate_b[head])
                    state[head] += np.outer(k, delta)
                    core[head] = q @ state[head]
                gate_z = (w[a + "in_proj_z"] @ z).reshape(2, 3)
                attended = norm(core, a + "norm", 0) * gate_z * sigmoid(gate_z)
                x += w[a + "out_proj"] @ attended.ravel()
            else:
                a = p + "self_attn."
                qg = (w[a + "q_proj"] @ z).reshape(2, 8)
                q = norm(qg[:, :4], a + "q_norm")
                k = norm((w[a + "k_proj"] @ z)[None, :], a + "k_norm")[0]
                for value in (q[0], q[1], k):
                    left, right = value[0], value[1]
                    value[0] = left * np.cos(position) - right * np.sin(position)
                    value[1] = right * np.cos(position) + left * np.sin(position)
                keys.append(k)
                values.append(w[a + "v_proj"] @ z)
                attended = np.zeros((2, 4))
                for head in range(2):
                    scores = np.array([key @ q[head] / 2 for key in keys])
                    probability = np.exp(scores - scores.max())
                    probability /= probability.sum()
                    for probability_t, value in zip(probability, values):
                        attended[head] += probability_t * value
                attended *= sigmoid(qg[:, 4:])
                x += w[a + "o_proj"] @ attended.ravel()
            z = norm(x, p + "post_attention_layernorm")
            gate, up = w[p + "mlp.gate_proj"] @ z, w[p + "mlp.up_proj"] @ z
            x += w[p + "mlp.down_proj"] @ (gate * sigmoid(gate) * up)
        outputs.append(w["lm_head"] @ norm(x, "norm"))
    return np.asarray(outputs)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ops = TinyOps(self.temp.name)

    def test_complete_hybrid_model_matches_independent_sequence_oracle(self):
        model = Model(self.ops.path, 8, self.ops)
        tokens = [3, 1, 5, 2, 4]
        expected = dense_oracle(self.ops, tokens)
        actual = np.array([model.forward(t) for t in tokens])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(model.position, len(tokens))
        digest = model.state_hash()
        model.reset()
        again = np.array([model.forward(t) for t in tokens])
        np.testing.assert_array_equal(actual, again)
        np.testing.assert_array_equal(model.state_hash(), digest)

    def test_prefill_without_logits_commits_identical_state(self):
        full = Model(self.ops.path, 8, self.ops)
        fast = Model(self.ops.path, 8, self.ops)
        full.forward(2)
        self.ops.calls.clear()
        self.assertIsNone(fast.forward(2, logits=False))
        self.assertNotIn("lm_head", self.ops.calls)
        np.testing.assert_array_equal(full.state_hash(), fast.state_hash())
        np.testing.assert_array_equal(full.forward(6), fast.forward(6))

    def test_context_and_token_errors_leave_state_usable(self):
        model = Model(self.ops.path, 1, self.ops)
        before = model.state_hash()
        with self.assertRaises(ValueError):
            model.forward(8)
        np.testing.assert_array_equal(before, model.state_hash())
        model.forward(0)
        with self.assertRaisesRegex(ValueError, "context"):
            model.forward(0)
        self.assertEqual(model.position, 1)
        model.reset()
        self.assertEqual(model.position, 0)

    def test_interrupted_forward_requires_reset(self):
        model = Model(self.ops.path, 8, self.ops)
        saved = self.ops.weights["layers.1.self_attn.q_proj"].copy()
        self.ops.weights["layers.1.self_attn.q_proj"][:] = np.nan
        with self.assertRaises(FloatingPointError):
            model.forward(0)
        self.ops.weights["layers.1.self_attn.q_proj"][:] = saved
        with self.assertRaisesRegex(RuntimeError, "reset"):
            model.forward(0)
        model.reset()
        self.assertTrue(np.isfinite(model.forward(0)).all())

    def test_unknown_architecture_and_shapes_are_rejected_early(self):
        original = copy.deepcopy(self.ops.header)
        self.ops.header["model"]["arch"] = "qwen2"
        self.ops.write_header()
        with self.assertRaisesRegex(FormatError, "qwen3_5"):
            Model(self.ops.path, 8, self.ops)
        self.ops.header = original
        self.ops.header["tensors"][1]["cols"] = 99
        self.ops.write_header()
        with self.assertRaisesRegex(FormatError, "shape"):
            Model(self.ops.path, 8, self.ops)

    def test_attention_cache_growth_preserves_prefix(self):
        model = Model(self.ops.path, 70, self.ops)
        tokens = [i % 8 for i in range(66)]
        actual = np.array([model.forward(t) for t in tokens])
        np.testing.assert_allclose(actual, dense_oracle(self.ops, tokens), rtol=3e-5, atol=3e-6)


class PrimitiveTests(unittest.TestCase):
    def test_delta_rule_does_not_discard_previous_state(self):
        state = np.array([[[1., 2.], [3., 4.]], [[-.5, 1.], [2., -.1]]], np.float32)
        previous = state.copy()
        q, k = np.array([[.2, .7]], np.float32), np.array([[.4, -.3]], np.float32)
        v = np.array([[2., 1.], [-1., 3.]], np.float32)
        beta, decay = np.array([.3, .7]), np.array([-.2, -.9])
        expected = []
        for h in range(2):
            s = previous[h] * np.exp(decay[h])
            s += np.outer(k[0], beta[h] * (v[h] - k[0] @ s))
            expected.append(q[0] @ s)
        np.testing.assert_allclose(delta_step(state, q, k, v, beta, decay), expected, rtol=1e-6)

    def test_rope_preserves_tail_and_norm(self):
        x = np.array([[1., 2., 3., 4., 5., 6.]], np.float32)
        out = rotary(x, 7, 4, 10000.)
        np.testing.assert_array_equal(out[:, 4:], x[:, 4:])
        np.testing.assert_allclose(np.linalg.norm(out), np.linalg.norm(x), rtol=1e-6)

    def test_grouped_heads_use_their_own_key_value_head(self):
        q = np.ones((4, 2), np.float32)
        keys = np.zeros((1, 2, 2), np.float32)
        values = np.array([[[1., 2.], [3., 4.]]], np.float32)
        np.testing.assert_array_equal(grouped_attention(q, keys, values), [[1, 2], [1, 2], [3, 4], [3, 4]])


if __name__ == "__main__":
    unittest.main()
