# SPDX-License-Identifier: Apache-2.0
import ctypes
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from yasakani._native import NativeLib, YasakaniError


def _load_accept():
    try:
        nlib = NativeLib()
    except YasakaniError as exc:
        raise unittest.SkipTest(f"native library not available: {exc}") from exc
    fn = getattr(nlib.dll, "yskn_ink2_accept", None)
    if fn is None:
        raise unittest.SkipTest("libyskn not built with YSKN_WITH_INK2=ON")
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
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


def _softmax(row, temp, top_k=0, top_p=1.0, min_p=0.0):
    V = len(row)
    items = sorted(((float(row[i]), i) for i in range(V)), key=lambda x: (-x[0], x[1]))
    if top_k > 0:
        items = items[:top_k]
    m = items[0][0]
    w = [math.exp((v - m) / temp) for v, _ in items]
    tot = sum(w)
    if 0.0 < top_p < 1.0:
        run = 0.0
        cut = len(items)
        for idx, wi in enumerate(w):
            run += wi
            if run >= top_p * tot:
                cut = idx + 1
                break
        items = items[:cut]
        w = w[:cut]
    if min_p > 0.0:
        cut = 1
        while cut < len(w) and w[cut] >= min_p:
            cut += 1
        items = items[:cut]
        w = w[:cut]
    s = sum(w)
    out = [0.0] * V
    for (_, tid), wi in zip(items, w):
        out[tid] = wi / s
    return out


class TestInkling2Accept(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accept = _load_accept()

    def test_greedy_exact_prefix_and_bonus(self):
        K, V = 4, 8
        draft = (ctypes.c_int32 * K)(1, 2, 3, 4)
        out = (ctypes.c_int32 * (K + 1))()

        # Prefix match up to index 2 (mismatch at index 2 -> token 7)
        v_arg = (ctypes.c_int32 * (K + 1))(1, 2, 7, 4, 5)
        a = self.accept(out, draft, K, V, None, None, v_arg, 0.0, 20, 0.95, 0.0, 42, 0)
        self.assertEqual(a, 2)
        self.assertEqual([out[i] for i in range(a + 1)], [1, 2, 7])

        # All K accepted -> emits K draft tokens + bonus argmax[K]
        v_arg_all = (ctypes.c_int32 * (K + 1))(1, 2, 3, 4, 6)
        a_all = self.accept(out, draft, K, V, None, None, v_arg_all, 0.0, 20, 0.95, 0.0, 42, 0)
        self.assertEqual(a_all, 4)
        self.assertEqual([out[i] for i in range(K + 1)], [1, 2, 3, 4, 6])

        # Zero accepted -> emits first mismatch token argmax[0]
        v_arg_0 = (ctypes.c_int32 * (K + 1))(5, 2, 3, 4, 6)
        a_0 = self.accept(out, draft, K, V, None, None, v_arg_0, 0.0, 20, 0.95, 0.0, 42, 0)
        self.assertEqual(a_0, 0)
        self.assertEqual(out[0], 5)

    def test_sampled_unbiased_residual_distribution(self):
        # Craft Q biased toward token 0, P biased toward token 1 & 2, draft K=1
        K, V = 1, 4
        temp = 1.0
        q_logits = [2.0, -1.0, -2.0, -3.0]
        p_logits = [0.2, 1.5, 0.8, -0.5]
        q_prob = _softmax(q_logits, temp)
        p_prob = _softmax(p_logits, temp)

        v_buf = (ctypes.c_float * (2 * V))(*(p_logits + [0.0, 0.0, 0.0, 0.0]))
        d_buf = (ctypes.c_float * V)(*q_logits)
        out = (ctypes.c_int32 * 2)()

        # Independent deterministic draft sampler using standard splitmix/LCG surrogate over trials
        import random
        rng = random.Random(12345)
        counts = [0] * V
        N = 20000
        for step in range(N):
            # Sample candidate dt ~ q
            u = rng.random()
            cum = 0.0
            dt = V - 1
            for tid, qi in enumerate(q_prob):
                cum += qi
                if u < cum:
                    dt = tid
                    break
            draft = (ctypes.c_int32 * 1)(dt)
            self.accept(out, draft, K, V, v_buf, d_buf, None, temp, 0, 1.0, 0.0, 777 + step * 17, step)
            counts[out[0]] += 1

        # Exact target marginal equivalence check: observed proportions match P within 4 sigma
        for tid in range(V):
            obs = counts[tid] / N
            exp = p_prob[tid]
            se = math.sqrt(exp * (1.0 - exp) / N)
            self.assertLess(abs(obs - exp), 4.0 * se, f"tok {tid}: obs={obs:.4f} vs exp={exp:.4f}")

    def test_top_k_top_p_min_p_matching(self):
        # Verify draft and target respect top_k / top_p / min_p masks so out-of-support tokens are never emitted
        K, V = 1, 6
        p_logits = [3.0, 2.5, 0.0, -10.0, -10.0, -10.0]
        q_logits = [-10.0, -10.0, 3.0, 2.0, 1.0, 0.0]
        v_buf = (ctypes.c_float * (2 * V))(*(p_logits + p_logits))
        d_buf = (ctypes.c_float * V)(*q_logits)
        out = (ctypes.c_int32 * 2)()
        draft = (ctypes.c_int32 * 1)(4)  # token 4 not in top_k=2 of P!
        for step in range(50):
            a = self.accept(out, draft, K, V, v_buf, d_buf, None, 1.0, 2, 0.99, 0.1, 99, step)
            self.assertEqual(a, 0)
            self.assertIn(out[0], (0, 1))


if __name__ == "__main__":
    unittest.main()
