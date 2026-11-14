#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Record bit-exact Metal decode diagnostics for the Qwen3.8 speed model.

This model-specific diagnostic makes no timing claim.

Run baseline and candidate separately with their own PYTHONPATH and --library.
This tool targets the Qwen3.8 speed model and its bundled chat template.
The fixed counting prompt must format to exactly 27 tokens with
enable_thinking=False; other models/templates are not supported.
Eight generated tokens mean one prefill prediction plus seven forward calls;
the final state contains the prompt and the first seven generated tokens.
Only token IDs, hashes and metadata are saved, never logits or model data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


PROMPT = "Write the numbers from 1 to 20, separated by commas."
PROMPT_TOKENS = 27
MAX_CTX = 512


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(args, report):
    # Explicit PYTHONPATH/installed packages win, as in macos/benchmark.py.
    if importlib.util.find_spec("yasakani") is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import numpy as np
    from yasakani.engine import CONTEXT_MARGIN, Sampling
    from yasakani.metal_engine import MetalEngine, find_metal_library
    from yasakani.runtime import ModelPack
    from yasakani.textio import ChatFormatter

    if PROMPT_TOKENS + args.steps - 1 > MAX_CTX - CONTEXT_MARGIN:
        raise ValueError("--steps exceeds the fixed 512-token context budget")
    pack = ModelPack.load(args.model_dir, require_aux=False)
    library = find_metal_library(args.library)
    formatter = ChatFormatter(pack.root)
    prompt, _ = formatter.prompt([{"role": "user", "content": PROMPT}], enable_thinking=False)
    if len(prompt) != PROMPT_TOKENS:
        raise ValueError("fixed diagnostic requires 27 prompt tokens; formatter produced %d" % len(prompt))
    package_dir = Path(importlib.import_module("yasakani").__file__).resolve().parent
    weights = pack.weights.resolve()
    report.update({
        "source": {"package_directory": str(package_dir), "runner_sha256": file_hash(__file__),
                   "package_sha256": {p.name: file_hash(p) for p in sorted(package_dir.glob("*.py"))}},
        "library": {"path": str(library), "sha256_actual": file_hash(library)},
        "model": {"id": pack.model_id, "weights": str(weights), "size_bytes": weights.stat().st_size,
                  "mtime_ns": weights.stat().st_mtime_ns,
                  "sha256_expected_from_manifest": pack.sha256.get(str(pack.weights.relative_to(pack.root))),
                  "checksum_verification": "not_performed_by_this_run"},
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                 "python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
        "settings": {"prompt": PROMPT, "prompt_ids": [int(token) for token in prompt],
                     "prompt_tokens": len(prompt), "enable_thinking": False, "max_ctx": MAX_CTX,
                     "generated_tokens_requested": args.steps, "fixed_length": True, "stop_ids": None,
                     "sampling_requested": {"temperature": 0.0, "seed": 42}, "prefix_cache": False},
    })
    write_report(args.output, report)
    with MetalEngine(pack, max_ctx=MAX_CTX, dll=str(library)) as engine:
        report["settings"]["sampling_effective"] = asdict(engine.set_sampling(Sampling(temperature=0, seed=42)))
        report["settings"]["seed_note"] = "Greedy sampling ignores the seed."
        model = engine.model
        prefill = model.prefill
        forward = model.forward

        def capture(operation, inputs, logits):
            values = np.asarray(logits)
            if values.ndim != 1 or values.dtype != np.dtype(np.float32) or not np.all(np.isfinite(values)):
                raise ValueError("expected a finite one-dimensional float32 logits vector")
            # View, never numerically cast float values: preserve every float32 bit.
            words = np.ascontiguousarray(values).view(np.uint32).astype("<u4", copy=False)
            row = {"index": len(report["logits"]), "operation": operation,
                   "input_token_ids": [int(token) for token in inputs], "position": engine.position(),
                   "dtype": "float32", "shape": list(values.shape), "hash_encoding": "little-endian uint32 bytes",
                   "sha256_uint32_bytes": hashlib.sha256(words.tobytes(order="C")).hexdigest(),
                   "argmax_id": int(values.argmax())}
            report["logits"].append(row)
            write_report(args.output, report)
            print("%s %d: position=%d argmax=%d sha256=%s" %
                  (operation, row["index"], row["position"], row["argmax_id"], row["sha256_uint32_bytes"]),
                  file=sys.stderr, flush=True)
            return logits

        def measured_prefill(tokens, *positional, **keyword):
            return capture("prefill", tokens, prefill(tokens, *positional, **keyword))

        def measured_forward(token, *positional, **keyword):
            return capture("forward", [token], forward(token, *positional, **keyword))

        model.prefill = measured_prefill
        model.forward = measured_forward
        try:
            result = engine.generate(prompt, args.steps, stop_ids=None)
        finally:
            model.prefill = prefill
            model.forward = forward
        generated = [int(token) for token in result.ids]
        if result.finish != "length" or len(generated) != args.steps or result.cached_tokens:
            raise RuntimeError("generation did not complete the requested fixed-length uncached decode")
        expected_operations = ["prefill"] + ["forward"] * (args.steps - 1)
        if [row["operation"] for row in report["logits"]] != expected_operations:
            raise RuntimeError("expected one prefill and steps - 1 forward calls")
        if [row["argmax_id"] for row in report["logits"]] != generated:
            raise RuntimeError("generated tokens differ from the captured greedy argmax sequence")
        if [row["position"] for row in report["logits"]] != list(range(len(prompt), len(prompt) + args.steps)):
            raise RuntimeError("captured model positions do not match prefill plus sequential decode")
        state = np.asarray(engine.state_hash(), dtype="<u8")
        report["generated"] = {"token_ids": generated, "text": formatter.decode(generated), "finish": result.finish}
        report["final_state"] = {
            "position": engine.position(), "expected_position": len(prompt) + args.steps - 1,
            "model_state_hash_hex": state.tobytes(order="C").hex(),
            "model_state_hash_uint64": [int(word) for word in state],
            "consumed_generated_token_ids": generated[:-1], "last_prediction_not_consumed": generated[-1],
            "scope": "input prompt plus the first steps - 1 generated tokens; final predicted token has not been forwarded",
        }
        # Deliberately discard result.prefill_ms/gen_ms: hashing and report writes
        # ran inside generate and make its timings unsuitable for speed comparison.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--library", required=True, help="explicit native library for this baseline/candidate process")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8, help="fixed generated token count, including the prefill prediction (default: 8)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    report = {"schema_version": 1, "mode": "exact_decode_diagnostic", "status": "running", "logits": [],
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "instrumented": True,
              "speed_measurement": False,
              "timing_policy": "CPU logits hashing and report I/O are instrumentation; all timing metrics are intentionally omitted"}
    code = 0
    try:
        run(args, report)
        report["status"] = "complete"
    except KeyboardInterrupt:
        report.update(status="interrupted", error={"type": "KeyboardInterrupt", "message": "diagnostic interrupted"})
        code = 130
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        print("diagnostic failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        code = 1
    finally:
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_report(args.output, report)
    print(str(args.output.resolve()), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
