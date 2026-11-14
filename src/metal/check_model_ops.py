#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Bounded-memory real-model Metal/reference comparison; no full weight expansion.

python src/metal/check_model_ops.py LIB MODEL.yskn [TENSOR ...]
"""
import ctypes as C
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import yskn_decode as ref


def default_tensors(container):
    names = ["layers.0.linear_attn.in_proj_qkv", "layers.3.self_attn.k_proj", "layers.0.mlp.down_proj"]
    # Exercise the high host binding offset separately from the uint32 shader
    # offset. Prefer a late down projection to keep CPU reference memory bounded.
    high = [t for t in container.tensors.values() if t.get("role") == "body" and t["stream_off"] > 2**32]
    preferred = [t for t in high if t["name"].endswith(".mlp.down_proj")]
    if high:
        name = max(preferred or high, key=lambda t: t["stream_off"])["name"]
        if name not in names:
            names.append(name)
    else:
        print("INFO: no body stream above 4 GiB; large-offset coverage unavailable", flush=True)
    return names


def main():
    lib = C.CDLL(str(Path(sys.argv[1]).resolve()))
    lib.yskn_metal_open.argtypes, lib.yskn_metal_open.restype = [C.c_char_p], C.c_void_p
    lib.yskn_metal_close.argtypes = [C.c_void_p]
    lib.yskn_metal_last_error.restype = C.c_char_p
    lib.yskn_metal_linear.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    lib.yskn_metal_linear_batch.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int]
    lib.yskn_metal_embedding.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    c = ref.Container(sys.argv[2])
    h = lib.yskn_metal_open(str(Path(sys.argv[2]).resolve()).encode())
    if not h:
        raise RuntimeError(lib.yskn_metal_last_error().decode())
    def check(rc):
        if rc:
            raise RuntimeError(lib.yskn_metal_last_error().decode())
    try:
        t = ref.checked(c.tensors["embed_tokens"])
        for token in (0, 42, min(248045, t["rows"] - 1)):
            expected = ref.decode_rows(c, t, token, token + 1)[0]
            for extra in ("embed_tokens.res", "embed_tokens.res2"):
                if extra in c.tensors:
                    expected += ref.decode_rows(c, ref.checked(c.tensors[extra]), token, token + 1)[0]
            actual = np.empty(t["cols"], np.float32)
            check(lib.yskn_metal_embedding(h, token, actual.ctypes.data, actual.size))
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
            print("PASS embedding", token, "max error", float(np.max(np.abs(actual - expected))), flush=True)
        names = sys.argv[3:] or default_tensors(c)
        rng = np.random.default_rng(261114)
        for name in names:
            started = time.monotonic()
            t = ref.checked(c.tensors[name])
            inputs = rng.standard_normal((5, t["cols"])).astype(np.float32)
            rotated_x = ref.fwht512(inputs * ref.signs_of(t["rot_in"], t["cols"]))
            units = ref.unit_words(c, t)
            lut = ref.value_table(c, t)
            scales = c.array(t["scale_off"], "<f2", t["physical_rows"]).astype(np.float32)
            expected = np.empty((5, t["physical_rows"]), np.float32)
            # Only 128 decoded rows at a time (under 9 MiB for these matrices).
            for first in range(0, t["physical_rows"], 128):
                last = min(first + 128, t["physical_rows"])
                block = lut[ref.states(units[first:last], t["bits"])].reshape(last - first, t["cols"])
                expected[:, first:last] = (rotated_x @ block.T) * scales[first:last]
            if t["rot_out"] is not None:
                expected = ref.fwht512(expected) * ref.signs_of(t["rot_out"], t["physical_rows"])
            expected = expected[:, :t["rows"]]
            single = np.empty(t["rows"], np.float32)
            check(lib.yskn_metal_linear(h, name.encode(), inputs[0].ctypes.data, t["cols"], single.ctypes.data, single.size))
            np.testing.assert_allclose(single, expected[0], atol=2e-3, rtol=2e-4)
            actual = np.empty((5, t["rows"]), np.float32)
            check(lib.yskn_metal_linear_batch(h, name.encode(), inputs.ctypes.data, 5, t["cols"], actual.ctypes.data, t["rows"]))
            np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-4)
            print("PASS linear and batch=5", name, "max error", float(np.max(np.abs(actual - expected))),
                  "stream offset", t["stream_off"], "seconds", round(time.monotonic() - started, 3), flush=True)
        print("PASS: real released-model embedding and matrix operations (not full generation)")
    finally:
        lib.yskn_metal_close(h)


if __name__ == "__main__":
    main()
