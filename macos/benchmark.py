#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Reproduce Metal generation timings with the same model and prompt.

Run from a checkout with its runtime dependencies installed::

    build/venv/bin/python macos/benchmark.py --model-dir /path/to/model \
        --library build/bin/libyskn_metal.dylib --output build/metal-after.json

The selected PYTHONPATH wins, allowing the exact same runner to measure a saved
baseline checkout and library. --runs counts warm repetitions AFTER one cold-first
generation. Every request resets model state; "warm" never means a cached prefix.
"Cold" means the first request of this process, not flushed filesystem/GPU caches.

--profile adds overlapping wall-time instrumentation and optional --cprofile FILE.
--smoke runs just one output token. Both are labelled separately from benchmarks;
neither is a formal speed measurement. No model is downloaded by this script.
"""
from __future__ import annotations

import argparse
import cProfile
import datetime as dt
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


DEFAULT_PROMPT = "What is 2 + 2? Answer briefly."


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    """Keep completed measurements readable even if a later request fails."""
    path = Path(path)
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


def command_value(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def host_info():
    info = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "macos_version": platform.mac_ver()[0], "python": platform.python_version(),
            "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        info["hardware_model"] = command_value(["/usr/sbin/sysctl", "-n", "hw.model"])
        info["chip"] = command_value(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
        memory = command_value(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        info["memory_bytes"] = int(memory) if memory and memory.isdigit() else None
    info["packages"] = {}
    for name in ("numpy", "tokenizers", "Jinja2"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    info["thread_environment"] = {name: os.environ[name] for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS") if name in os.environ}
    return info


def source_info():
    """Hash the imported package, not a potentially different working checkout."""
    package = importlib.import_module("yasakani")
    package_dir = Path(package.__file__).resolve().parent
    source_root = package_dir.parent.parent
    hashes = {"python/yasakani/" + path.name: sha256_file(path) for path in sorted(package_dir.glob("*.py"))}
    for relative in ("CMakeLists.txt", "src/metal/yskn_metal.mm", "src/metal/yskn_metal.h", "src/metal/kernels.hpp"):
        path = source_root / relative
        if path.is_file():
            hashes[relative] = sha256_file(path)
    # A source snapshot under another checkout must not inherit its parent's HEAD.
    has_git = (source_root / ".git").exists()
    return {"package_directory": str(package_dir), "version": getattr(package, "__version__", None),
            "git_head": command_value(["git", "-C", str(source_root), "rev-parse", "HEAD"]) if has_git else None,
            "git_dirty": bool(command_value(["git", "-C", str(source_root), "status", "--porcelain"])) if has_git else None,
            "sha256_by_file": hashes, "runner_sha256": sha256_file(__file__)}


class OperationProfile:
    """Inclusive timings: attention/GDN overlap their nested linear operations."""
    def __init__(self, engine):
        self.patches = []
        self.totals = {}
        ops = getattr(engine, "_ops", None) or getattr(engine.model, "ops", None)
        for target, method, label in ((ops, "linear", "ops.linear"), (ops, "linear_batch", "ops.linear_batch"),
                                       (ops, "linear_multi", "ops.linear_multi"),
                                       (ops, "embedding", "ops.embedding"), (engine.model, "prefill", "model.prefill"),
                                       (engine.model, "_attention", "model.attention"), (engine.model, "_gdn", "model.gdn")):
            if target is None or not hasattr(target, method):
                continue
            original = getattr(target, method)
            had_attribute = method in getattr(target, "__dict__", {})
            previous = target.__dict__.get(method) if had_attribute else None

            @functools.wraps(original)
            def timed(*args, _fn=original, _label=label, **kwargs):
                start = time.perf_counter()
                try:
                    return _fn(*args, **kwargs)
                finally:
                    elapsed = (time.perf_counter() - start) * 1000
                    labels = [_label]
                    if _label in ("ops.linear", "ops.linear_batch") and args:
                        labels.append(("linear_tensor:" if _label == "ops.linear" else "linear_batch_tensor:") + str(args[0]))
                    if _label == "ops.linear_multi":
                        names = args[0] if args else kwargs.get("names")
                        if names is not None:
                            labels.append("linear_multi_tensors:" + json.dumps(list(names), separators=(",", ":")))
                    for key in labels:
                        row = self.totals.setdefault(key, {"calls": 0, "total_ms": 0.0})
                        row["calls"] += 1
                        row["total_ms"] += elapsed

            setattr(target, method, timed)
            self.patches.append((target, method, had_attribute, previous))

    def reset(self):
        self.totals.clear()

    def snapshot(self):
        return {name: dict(row, mean_ms=row["total_ms"] / row["calls"])
                for name, row in sorted(self.totals.items())}

    def restore(self):
        for target, method, had_attribute, previous in reversed(self.patches):
            if had_attribute:
                setattr(target, method, previous)
            else:
                delattr(target, method)
        self.patches.clear()


def timing_summary(rows):
    summary = {}
    for key in ("prefill_ms", "gen_ms", "total_ms", "decode_tokens_per_s"):
        values = [row[key] for row in rows]
        if values:
            summary[key] = {"mean": statistics.mean(values), "median": statistics.median(values),
                            "min": min(values), "max": max(values)}
    return summary


def run(args, report):
    # An explicit PYTHONPATH or installed package takes priority, including a
    # baseline snapshot. A plain source checkout works without pip install -e.
    if importlib.util.find_spec("yasakani") is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from yasakani.engine import CONTEXT_MARGIN, Sampling
    from yasakani.metal_engine import MetalEngine, find_metal_library
    from yasakani.runtime import EOS_IDS, ModelPack
    from yasakani.textio import ChatFormatter

    pack = ModelPack.load(args.model_dir, require_aux=False)
    library = find_metal_library(args.library)
    weights = pack.weights.resolve()
    tokenizer_start = time.perf_counter()
    formatter = ChatFormatter(pack.root)
    prompt, _ = formatter.prompt([{"role": "user", "content": args.prompt}], enable_thinking=False)
    tokenizer_load_ms = (time.perf_counter() - tokenizer_start) * 1000
    max_new = 1 if args.smoke else args.max_new
    if len(prompt) + max_new - 1 > args.max_ctx - CONTEXT_MARGIN:
        raise ValueError("prompt plus requested output exceeds the context; choose --max-ctx explicitly")
    report.update({"host": host_info(), "source": source_info(),
                   "library": {"path": str(library), "size_bytes": library.stat().st_size,
                               "sha256_actual": sha256_file(library)},
                   "model": {"id": pack.model_id, "weights": str(weights), "size_bytes": weights.stat().st_size,
                             "mtime_ns": weights.stat().st_mtime_ns,
                             "sha256_expected_from_manifest": pack.sha256.get(str(pack.weights.relative_to(pack.root))),
                             "sha256_actual": None, "checksum_verification": "not_performed_by_this_run"},
                   "tokenizer_load_ms": tokenizer_load_ms,
                   "settings": {"prompt": args.prompt, "prompt_ids": [int(x) for x in prompt], "prompt_tokens": len(prompt),
                                "enable_thinking": False, "max_ctx": args.max_ctx, "max_new": max_new,
                                "fixed_length": args.fixed_length, "stop_ids": None if args.fixed_length else list(EOS_IDS),
                                "sampling_requested": {"temperature": 0.0, "seed": 42},
                                "warm_repetitions": 0 if args.smoke else args.runs,
                                "prefix_cache": False, "reset_state_each_request": True,
                                "profiling_overhead": args.profile,
                                "cold_definition": "first request after model load; OS and driver caches are not flushed"}})
    # This explicit option costs a full model read; a manifest checksum alone is
    # never represented as a verified or actually measured checksum.
    if args.hash_weights:
        actual = sha256_file(weights)
        expected = report["model"]["sha256_expected_from_manifest"]
        report["model"]["sha256_actual"] = actual
        report["model"]["checksum_verification"] = "matches_manifest" if expected == actual else (
            "mismatches_manifest" if expected else "computed_without_expected_checksum")
        if expected and expected != actual:
            raise ValueError("model weights do not match the manifest SHA-256")
    write_report(args.output, report)
    start = time.perf_counter()
    engine = MetalEngine(pack, max_ctx=args.max_ctx, dll=str(library))
    report["model_load_ms"] = (time.perf_counter() - start) * 1000
    profiler = None
    cpu_profile = None
    try:
        sampling = engine.set_sampling(Sampling(temperature=0.0, seed=42))
        report["settings"]["sampling_effective"] = asdict(sampling)
        report["settings"]["seed_note"] = "Greedy sampling ignores the seed; normalization may reset it to zero."
        if args.profile:
            profiler = OperationProfile(engine)
        if args.cprofile:
            cpu_profile = cProfile.Profile()
        for index in range(1 if args.smoke else args.runs + 1):
            if profiler:
                profiler.reset()
            if cpu_profile:
                cpu_profile.enable()
            try:
                start = time.perf_counter()
                result = engine.generate(prompt, max_new, stop_ids=None if args.fixed_length else EOS_IDS)
                total_ms = (time.perf_counter() - start) * 1000
            finally:
                if cpu_profile:
                    cpu_profile.disable()
            ids = [int(token) for token in result.ids]
            row = {"index": index, "phase": "cold_first" if index == 0 else "warm",
                   "prefill_ms": result.prefill_ms, "gen_ms": result.gen_ms, "total_ms": total_ms,
                   "prompt_tokens": result.prompt_tokens, "generated_tokens": len(ids),
                   "cached_tokens": result.cached_tokens, "finish": result.finish,
                   "decode_tokens_per_s": max(0, len(ids) - 1) * 1000 / result.gen_ms if result.gen_ms > 0 else 0.0,
                   "prefill_tokens_per_s": result.prompt_tokens * 1000 / result.prefill_ms if result.prefill_ms > 0 else 0.0,
                   "token_ids": ids, "text": formatter.decode(ids)}
            if profiler:
                row["operation_profile"] = profiler.snapshot()
            report["runs"].append(row)
            if result.cached_tokens:
                raise RuntimeError("unexpected prefix-cache reuse: this benchmark requires fresh requests")
            if result.finish == "cancelled":
                raise RuntimeError("generation was cancelled; this is not a completed timing")
            if args.fixed_length and len(ids) != max_new:
                raise RuntimeError("fixed-length generation did not produce the requested number of tokens")
            write_report(args.output, report)
            print("%s %d: prefill %.1f ms, decode %.1f ms, %d tokens, %.3f decode token/s" %
                  (row["phase"], index, row["prefill_ms"], row["gen_ms"], len(ids), row["decode_tokens_per_s"]),
                  file=sys.stderr, flush=True)
    finally:
        if profiler:
            profiler.restore()
        engine.close()
        if cpu_profile:
            output = Path(args.cprofile)
            output.parent.mkdir(parents=True, exist_ok=True)
            cpu_profile.dump_stats(str(output))
            report["cprofile_file"] = str(output.resolve())
    report["outputs_identical_across_runs"] = all(row["token_ids"] == report["runs"][0]["token_ids"] for row in report["runs"])
    # Keep profile/smoke results visible but outside the formal timing summary.
    report["warm_summary"] = timing_summary([row for row in report["runs"] if row["phase"] == "warm"]) if report["mode"] == "benchmark" else None
    if args.profile:
        report["profile_note"] = "Inclusive wall timings overlap: attention/GDN contain linear calls. Do not sum categories or compare profiled runs with uninstrumented timings."


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model-dir", required=True, type=Path)
    result.add_argument("--library", help="exact libyskn_metal.dylib; use the baseline library with a baseline PYTHONPATH")
    result.add_argument("--output", required=True, type=Path, help="JSON report, updated after each completed request")
    result.add_argument("--runs", type=int, default=3, help="warm repetitions after one cold-first request (default: 3)")
    result.add_argument("--prompt", default=DEFAULT_PROMPT)
    result.add_argument("--max-new", type=int, default=8)
    result.add_argument("--max-ctx", type=int, default=512)
    result.add_argument("--fixed-length", action="store_true", help="ignore EOS and generate exactly --max-new tokens")
    result.add_argument("--profile", action="store_true", help="instrument operations; mark results as profile, not formal benchmark")
    result.add_argument("--cprofile", type=Path, help="write generation cProfile stats; requires --profile")
    result.add_argument("--smoke", action="store_true", help="one request with one output token; not a formal benchmark")
    result.add_argument("--hash-weights", action="store_true", help="read all model bytes to compute and check SHA-256 (optional, slow)")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.runs < 0 or args.max_new < 1 or args.max_ctx < 65:
        arguments.error("--runs must be nonnegative; --max-new positive; --max-ctx at least 65")
    if not args.prompt.strip():
        arguments.error("--prompt must not be empty")
    if args.cprofile and not args.profile:
        arguments.error("--cprofile requires --profile")
    if args.smoke and args.profile:
        arguments.error("choose either --smoke or --profile")
    report = {"schema_version": 1, "status": "running",
              "mode": "smoke" if args.smoke else "profile" if args.profile else "benchmark",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "runs": [],
              "metric": "decode_tokens_per_s = max(generated_tokens - 1, 0) / (gen_ms / 1000); the first token is predicted by prefill"}
    code = 0
    try:
        run(args, report)
        report["status"] = "complete"
    except KeyboardInterrupt:
        report.update(status="interrupted", error={"type": "KeyboardInterrupt", "message": "benchmark interrupted"})
        code = 130
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        print("benchmark failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        code = 1
    finally:
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_report(args.output, report)
    print(str(args.output.resolve()), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
