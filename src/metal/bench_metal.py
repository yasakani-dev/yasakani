#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Paired warmed matrix timings for two real Metal libraries, using identical input.

python src/metal/bench_metal.py BASELINE.dylib CANDIDATE.dylib MODEL.yskn --out report.json
This measures matrix operations only, not end-to-end generation performance.
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


class Ops:
    def __init__(self, path, model):
        self.lib = C.CDLL(str(path.resolve()))
        self.lib.yskn_metal_open.argtypes, self.lib.yskn_metal_open.restype = [C.c_char_p], C.c_void_p
        self.lib.yskn_metal_close.argtypes = [C.c_void_p]
        self.lib.yskn_metal_last_error.restype = C.c_char_p
        self.lib.yskn_metal_linear.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int]
        self.batch = getattr(self.lib, "yskn_metal_linear_batch", None)
        if self.batch is not None:
            self.batch.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int]
        self.handle = self.lib.yskn_metal_open(str(model.resolve()).encode())
        if not self.handle:
            raise RuntimeError(self.lib.yskn_metal_last_error().decode())

    def linear(self, name, x, out):
        start = time.perf_counter()
        rc = self.lib.yskn_metal_linear(self.handle, name.encode(), x.ctypes.data, x.size, out.ctypes.data, out.size)
        elapsed = time.perf_counter() - start
        if rc:
            raise RuntimeError(self.lib.yskn_metal_last_error().decode())
        return elapsed

    def linear_many(self, name, x, out):
        start = time.perf_counter()
        if self.batch is not None and len(x) > 1:
            rc = self.batch(self.handle, name.encode(), x.ctypes.data, len(x), x.shape[1], out.ctypes.data, out.shape[1])
            if rc:
                raise RuntimeError(self.lib.yskn_metal_last_error().decode())
        else:
            for row, result in zip(x, out):
                self.linear(name, row, result)
        return time.perf_counter() - start

    def close(self):
        if self.handle:
            self.lib.yskn_metal_close(self.handle)
            self.handle = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--names", nargs="+", default=["layers.0.linear_attn.in_proj_qkv", "layers.3.self_attn.k_proj",
                                                     "layers.0.mlp.down_proj", "lm_head"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(b < 1 or b > 32 for b in args.batch_sizes):
        parser.error("--repeats must be positive; batch sizes must be between 1 and 32")
    with args.model.open("rb") as f:
        head = f.read(32)
        header = json.loads(f.read(struct.unpack_from("<Q", head, 8)[0]))
    tensors = {t["name"]: t for t in header["tensors"]}
    report = {"kind": "paired warmed Metal matrix operations (not generation)", "model_bytes": args.model.stat().st_size,
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(), "repeats": args.repeats, "matrices": []}
    engines = []
    try:
        engines = [Ops(args.baseline, args.model)]
        engines.append(Ops(args.candidate, args.model))
        rng = np.random.default_rng(20261114)
        for name in args.names:
            t = tensors[name]
            for batch in args.batch_sizes:
                x = rng.standard_normal((batch, t["cols"])).astype(np.float32)
                outputs = [np.empty((batch, t["rows"]), np.float32), np.empty((batch, t["rows"]), np.float32)]
                for engine, out in zip(engines, outputs):
                    engine.linear_many(name, x, out)  # excludes mapping/page-in and initial pipeline use
                timings = [[], []]
                for repeat in range(args.repeats):
                    for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        timings[index].append(engines[index].linear_many(name, x, outputs[index]))
                np.testing.assert_allclose(outputs[1], outputs[0], atol=2e-3, rtol=2e-4)
                entry = dict(name=name, shape=[t["rows"], t["cols"]], bits=t["bits"], batch=batch,
                             baseline_seconds=timings[0], candidate_seconds=timings[1],
                             baseline_median=statistics.median(timings[0]), candidate_median=statistics.median(timings[1]),
                             max_abs_difference=float(np.max(np.abs(outputs[1] - outputs[0]))))
                entry["speedup"] = entry["baseline_median"] / entry["candidate_median"]
                report["matrices"].append(entry)
                print(json.dumps(entry), flush=True)
    finally:
        for engine in engines:
            engine.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
