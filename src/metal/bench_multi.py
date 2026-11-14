#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Compare separate projection calls with one multi-projection submission.

python src/metal/bench_multi.py LIB.dylib MODEL.yskn --out report.json
Both paths use the same kernels and weights; only submission grouping differs.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import struct
import time

import numpy as np

from bench_metal import Ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with args.model.open("rb") as f:
        header = f.read(32)
        metadata = json.loads(f.read(struct.unpack_from("<Q", header, 8)[0]))
    tensors = {t["name"]: t for t in metadata["tensors"]}
    groups = [("layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z"),
              ("layers.3.self_attn.q_proj", "layers.3.self_attn.k_proj", "layers.3.self_attn.v_proj"),
              ("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj")]
    report = {"kind": "same-kernel separate/multi projection calls (not generation)",
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "model_bytes": args.model.stat().st_size, "repeats": args.repeats, "groups": []}
    engine = Ops(args.library, args.model)
    fn = engine.lib.yskn_metal_linear_multi
    fn.argtypes = [C.c_void_p, C.POINTER(C.c_char_p), C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    fn.restype = C.c_int
    try:
        rng = np.random.default_rng(20261115)
        for names in groups:
            n = tensors[names[0]]["cols"]
            assert all(tensors[name]["cols"] == n for name in names)
            x = rng.standard_normal(n).astype(np.float32)
            single = [np.empty(tensors[name]["rows"], np.float32) for name in names]
            together = np.empty(sum(tensors[name]["rows"] for name in names), np.float32)
            packed = (C.c_char_p * len(names))(*(name.encode() for name in names))
            def run_separate():
                start = time.perf_counter()
                for name, out in zip(names, single):
                    engine.linear(name, x, out)
                return time.perf_counter() - start
            def run_multi():
                start = time.perf_counter()
                rc = fn(engine.handle, packed, len(names), x.ctypes.data, n, together.ctypes.data, len(together))
                elapsed = time.perf_counter() - start
                if rc:
                    raise RuntimeError(engine.lib.yskn_metal_last_error().decode())
                return elapsed
            calls = (run_separate, run_multi)
            for call in calls:
                call()
            samples = [[], []]
            for repeat in range(args.repeats):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    samples[index].append(calls[index]())
            np.testing.assert_array_equal(together, np.concatenate(single))
            entry = dict(names=names, separate_seconds=samples[0], multi_seconds=samples[1],
                         separate_median=statistics.median(samples[0]), multi_median=statistics.median(samples[1]),
                         bit_identical=True)
            entry["speedup"] = entry["separate_median"] / entry["multi_median"]
            print(json.dumps(entry), flush=True)
            report["groups"].append(entry)
    finally:
        engine.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
