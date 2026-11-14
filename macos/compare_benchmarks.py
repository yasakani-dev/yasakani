#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Compare one matching before/after pair from macos/benchmark.py.

Requires completed, uninstrumented benchmarks with at least two warm runs, the
same host/runtime/settings/weights identity, and identical generated token IDs.
Manifest-only weight identity is explicitly distinguished from measured hashes.
Speedups use warm medians; individual runs and cold-first timing remain separate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import sys
from pathlib import Path


SETTINGS = ("prompt", "prompt_ids", "prompt_tokens", "enable_thinking", "max_ctx", "max_new", "fixed_length",
            "stop_ids", "sampling_requested", "sampling_effective", "warm_repetitions", "prefix_cache",
            "reset_state_each_request", "profiling_overhead", "cold_definition")
HOST = ("system", "release", "machine", "macos_version", "python", "logical_cpus", "hardware_model", "chip",
        "memory_bytes", "packages", "thread_environment")


def at(report, *keys):
    current = report
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError("missing field: " + ".".join(keys))
        current = current[key]
    return current


def checksum(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(label + " is not a SHA-256 checksum")
    return value.lower()


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be a finite positive number")
    return value


def validate(report, side):
    if at(report, "schema_version") != 1:
        raise ValueError(side + ": unsupported benchmark schema")
    if at(report, "mode") != "benchmark" or at(report, "status") != "complete":
        raise ValueError(side + ": only complete benchmark-mode reports can be compared")
    settings = at(report, "settings")
    if settings.get("profiling_overhead") is not False or settings.get("prefix_cache") is not False \
            or settings.get("reset_state_each_request") is not True:
        raise ValueError(side + ": requires uninstrumented requests with reset state and no prefix cache")
    prompt = at(settings, "prompt_ids")
    if not isinstance(prompt, list) or not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError(side + ": invalid prompt token IDs")
    if at(settings, "prompt_tokens") != len(prompt):
        raise ValueError(side + ": prompt token count disagrees with IDs")
    checksum(at(report, "library", "sha256_actual"), side + " library hash")
    if not at(report, "source", "sha256_by_file"):
        raise ValueError(side + ": source hashes are required")
    rows = at(report, "runs")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict) or rows[0].get("phase") != "cold_first":
        raise ValueError(side + ": expected a cold-first run followed by warm runs")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(side + ": invalid run record")
    warm = [row for row in rows if row.get("phase") == "warm"]
    if len(warm) < 2 or len(rows) != len(warm) + 1:
        raise ValueError(side + ": at least two warm runs and exactly one cold-first run are required")
    if at(settings, "warm_repetitions") != len(warm):
        raise ValueError(side + ": incomplete warm repetition count")
    reference_ids = at(rows[0], "token_ids")
    for index, row in enumerate(rows):
        label = "%s run %d" % (side, index)
        ids = at(row, "token_ids")
        if not isinstance(ids, list) or len(ids) < 2 or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(label + ": at least two generated tokens are required to measure decoding")
        if ids != reference_ids:
            raise ValueError(side + ": generated token IDs differ across repetitions")
        if at(row, "generated_tokens") != len(ids) or at(row, "prompt_tokens") != len(prompt):
            raise ValueError(label + ": token counts disagree with IDs")
        if at(row, "cached_tokens") != 0 or at(row, "finish") not in ("stop", "length"):
            raise ValueError(label + ": cached or incomplete generation cannot be compared")
        if len(ids) > at(settings, "max_new") or (settings["fixed_length"] and len(ids) != settings["max_new"]):
            raise ValueError(label + ": generated length disagrees with the requested limit")
        for metric in ("prefill_ms", "gen_ms", "total_ms"):
            positive(at(row, metric), label + " " + metric)
        rate = (len(ids) - 1) * 1000 / row["gen_ms"]
        if not math.isclose(positive(at(row, "decode_tokens_per_s"), label + " decode rate"), rate, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(label + ": decode rate does not match token count and duration")
    return warm


def model_identity(before, after):
    evidence = {}
    for side, report in (("before", before), ("after", after)):
        model = at(report, "model")
        expected = model.get("sha256_expected_from_manifest")
        actual = model.get("sha256_actual")
        expected = checksum(expected, side + " expected model hash") if expected is not None else None
        actual = checksum(actual, side + " actual model hash") if actual is not None else None
        if not (expected or actual):
            raise ValueError(side + ": no model checksum identity is available")
        if expected and actual and expected != actual:
            raise ValueError(side + ": actual weights differ from the manifest")
        declared = at(model, "checksum_verification")
        if declared == "mismatches_manifest":
            raise ValueError(side + ": model verification failed")
        if declared in ("matches_manifest", "computed_without_expected_checksum") and actual is None:
            raise ValueError(side + ": declared model verification has no measured checksum")
        evidence[side] = {"sha256_expected_from_manifest": expected, "sha256_actual": actual,
                          "declared_verification": declared, "verification_by_comparison": "not_performed"}
    left, right = evidence["before"], evidence["after"]
    if (left["sha256_actual"] or left["sha256_expected_from_manifest"]) != (right["sha256_actual"] or right["sha256_expected_from_manifest"]):
        raise ValueError("model checksum identity differs between reports")
    if left["sha256_expected_from_manifest"] != right["sha256_expected_from_manifest"]:
        raise ValueError("model expected checksums differ between reports")
    actual_count = sum(evidence[side]["sha256_actual"] is not None for side in evidence)
    evidence["basis"] = ("actual_hash_in_both_reports" if actual_count == 2 else
                         "actual_hash_in_one_report_and_matching_manifest" if actual_count == 1 else
                         "matching_manifest_expectations_only")
    evidence["note"] = "This tool compares recorded identities; it does not read or rehash model files. Manifest expectations alone do not prove that the measured files match those expectations."
    return evidence


def series(values):
    return {"samples": values, "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def compare(before, after):
    before_warm, after_warm = validate(before, "before"), validate(after, "after")
    for section, keys in (("settings", SETTINGS), ("host", HOST), ("model", ("id", "size_bytes"))):
        for key in keys:
            if at(before, section, key) != at(after, section, key):
                raise ValueError("comparison conditions differ: %s.%s" % (section, key))
    positive(at(before, "host", "memory_bytes"), "host memory size")
    for key in ("hardware_model", "chip", "packages"):
        if not at(before, "host", key):
            raise ValueError("host." + key + " is required")
    identity = model_identity(before, after)
    if before["runs"][0]["token_ids"] != after["runs"][0]["token_ids"]:
        raise ValueError("generated token IDs differ between before and after")
    reference_text = at(before["runs"][0], "text")
    if any(at(row, "text") != reference_text for row in before["runs"] + after["runs"]):
        raise ValueError("decoded text differs within or between benchmark reports")
    metrics = {}
    for metric in ("prefill_ms", "gen_ms", "total_ms", "decode_tokens_per_s"):
        left, right = series([row[metric] for row in before_warm]), series([row[metric] for row in after_warm])
        speedup = right["median"] / left["median"] if metric.endswith("per_s") else left["median"] / right["median"]
        metrics[metric] = {"before": left, "after": right, "speedup": speedup}
    return {"status": "comparable", "schema_version": 1,
            "compared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "basis": "warm medians; speedup above 1 means faster; cold-first runs are excluded",
            "warm_runs_per_side": len(before_warm), "metrics": metrics,
            "model_identity": identity, "host": before["host"], "settings": before["settings"],
            "outputs": {"token_ids_identical": True, "token_ids": before["runs"][0]["token_ids"], "text": reference_text},
            "provenance": {side: {"source": report["source"], "library": report["library"]}
                           for side, report in (("before", before), ("after", after))},
            "cold_first": {side: {key: report["runs"][0][key] for key in ("prefill_ms", "gen_ms", "total_ms", "decode_tokens_per_s")}
                           for side, report in (("before", before), ("after", after))},
            "limit": "A speedup for this prompt, generated sequence and machine; it is not a general model-quality or throughput claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="also write comparison JSON to this file")
    args = parser.parse_args(argv)
    try:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        report = compare(before, after)
        code = 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {"schema_version": 1, "status": "rejected", "error": str(exc)}
        code = 1
    report["inputs"] = {"before": str(args.before.resolve()), "after": str(args.after.resolve())}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
