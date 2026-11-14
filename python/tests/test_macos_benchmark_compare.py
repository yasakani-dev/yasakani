# SPDX-License-Identifier: Apache-2.0
"""The optional release gate must reject a decode regression, even if total improves."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "macos" / "compare_benchmarks.py"


def benchmark(prefill_ms, gen_ms, total_ms, *, cold_scale=2):
    ids = list(range(16))
    rows = []
    for index, scale in enumerate((cold_scale, 0.9, 1.1)):
        rows.append({
            "phase": "cold_first" if index == 0 else "warm",
            "prefill_ms": prefill_ms * scale, "gen_ms": gen_ms * scale,
            "total_ms": total_ms * scale, "decode_tokens_per_s": 15_000 / (gen_ms * scale),
            "prompt_tokens": 2, "generated_tokens": 16, "token_ids": ids,
            "cached_tokens": 0, "finish": "length", "text": "fixture completion",
        })
    return {
        "schema_version": 1, "mode": "benchmark", "status": "complete", "runs": rows,
        "settings": {
            "prompt": "fixture prompt", "prompt_ids": [21, 22], "prompt_tokens": 2,
            "enable_thinking": False, "max_ctx": 512, "max_new": 16, "fixed_length": True,
            "stop_ids": None, "sampling_requested": {"temperature": 0, "seed": 42},
            "sampling_effective": {"temperature": 0, "seed": 0}, "warm_repetitions": 2,
            "prefix_cache": False, "reset_state_each_request": True, "profiling_overhead": False,
            "cold_definition": "first request after model load",
        },
        "host": {
            "system": "Darwin", "release": "fixture", "machine": "arm64", "macos_version": "fixture",
            "python": "fixture", "logical_cpus": 12, "hardware_model": "fixture", "chip": "fixture",
            "memory_bytes": 32 * 1024**3, "packages": {"numpy": "fixture"}, "thread_environment": {},
        },
        "model": {"id": "fixture", "size_bytes": 1024, "sha256_expected_from_manifest": "a" * 64,
                  "sha256_actual": None, "checksum_verification": "not_performed_by_this_run"},
        "library": {"sha256_actual": "b" * 64},
        "source": {"sha256_by_file": {"fixture.py": "c" * 64}},
    }


class PerformanceGateTests(unittest.TestCase):
    def compare(self, before, after, *, require_improvement=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, report in (("before", before), ("after", after)):
                (root / (name + ".json")).write_text(json.dumps(report), encoding="utf-8")
            command = [sys.executable, str(SCRIPT), "--before", str(root / "before.json"),
                       "--after", str(root / "after.json"), "--output", str(root / "comparison.json")]
            if require_improvement:
                command.append("--require-improvement")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertFalse(result.stderr, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report, json.loads((root / "comparison.json").read_text(encoding="utf-8")))
            return result.returncode, report

    def test_all_warm_latencies_improve_despite_slower_cold_run(self):
        code, report = self.compare(benchmark(100, 40, 145), benchmark(60, 30, 95, cold_scale=10))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "comparable")
        gate = report["performance_gate"]
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(set(gate["metrics"]), {"prefill_ms", "gen_ms", "total_ms"})
        self.assertTrue(all(check["passed"] for check in gate["metrics"].values()))

    def test_decode_regression_or_tie_blocks_gate_even_when_total_improves(self):
        before = benchmark(100, 40, 145)
        for decode_ms in (50, 40):
            with self.subTest(decode_ms=decode_ms):
                after = benchmark(60, decode_ms, 115)
                code, report = self.compare(before, after)
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "comparable")
                self.assertEqual(report["performance_gate"]["status"], "failed")
                checks = report["performance_gate"]["metrics"]
                self.assertFalse(checks["gen_ms"]["passed"])
                self.assertTrue(checks["prefill_ms"]["passed"])
                self.assertTrue(checks["total_ms"]["passed"])
                code, report = self.compare(before, after, require_improvement=False)
                self.assertEqual(code, 0)
                self.assertNotIn("performance_gate", report)

    def test_incomparable_reports_do_not_evaluate_performance(self):
        after = benchmark(60, 30, 95)
        after["settings"]["prompt_ids"] = [21, 23]
        code, report = self.compare(benchmark(100, 40, 145), after)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("comparison conditions differ", report["error"])
        self.assertEqual(report["performance_gate"]["status"], "not_evaluated")
        self.assertNotIn("metrics", report["performance_gate"])


if __name__ == "__main__":
    unittest.main()
