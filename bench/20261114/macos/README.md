# Metal real-model latency comparison

Measured on 2026-11-14–15 JST, Apple M2 Max / 32 GiB / macOS 26.6.2,
Python 3.14.10, NumPy 2.5.4 (Apple Accelerate). These are physical-device
generation measurements, not CUDA figures or matrix-only throughput.

The model is `yasakani-dev/Qwen3.8-27B-YASAKANI-speed`, revision
`cd78f90d205677f8b3d2ce95eae82a64e278b910`. The unchanged 10,311,344,128-byte
weights have SHA-256
`3ba17a76871a2b5e86fb1d01ab677e6b35ff5ccc27d4f03ad941501e3319d1ca`.
The arithmetic runs rehashed the weights on both sides. Counting runs recorded
the same manifest expectation and file size/mtime; they did not independently
rehash the file.

Baseline: frozen source from `004d5d0014bb5a855081b644f3654c3309b94b58` and
its packaged Release library. Candidate: the source file hashes in the `after`
reports, built with CMake Release (`-O3`, arm64, macOS target 14). Measurements
precede the performance commit, so the candidate's recorded Git HEAD is still
the baseline with a dirty working tree. Source hashes identify the tested code.

## Results

Warm medians of three fresh-state requests, after one separately recorded
first request. No prefix cache, greedy sampling, context 512, same tokenizer and
input IDs. Models ran in separate processes, with no other model benchmark or
YASAKANI server running. Ordinary desktop activity was not disabled.

| Request | Measure | Before | After | Speedup |
|---|---|---:|---:|---:|
| Arithmetic, 23 input / 2 output tokens | Total | 14.147 s | 7.748 s | 1.83× |
| Arithmetic | Prefill | 13.557 s | 6.627 s | 2.05× |
| Count, 27 input / 16 output tokens | Total | 25.506 s | 17.070 s | 1.49× |
| Count | Prefill | 16.189 s | 6.798 s | 2.38× |
| Count | Decode, excluding first token | 1.610 token/s | 1.461 token/s | 0.91× |

**This establishes lower total latency for these two requests, not faster
sustained decoding.** Counting decode throughput regressed by about 9% at the
median, and the arithmetic request's one decode step increased from 0.597 s to
1.389 s. The three candidate counting totals were 17.070 s, 29.890 s and 14.546 s:
there is substantial variability, including a slower run. All samples are retained.
First-request totals were arithmetic 24.362 → 17.791 s and count 36.218 → 21.920 s;
“first” does not mean flushed operating-system or GPU caches.

All four repetitions per side produced exactly the same token IDs and text:

- `What is 2 + 2? Answer briefly.` → `4<|im_end|>` (including the stop token).
- `Write the numbers from 1 to 20, separated by commas.` →
  `1, 2, 3, 4, 5, 6` (fixed 16-token limit, not a completed answer).

No quality model, long-context quality, broader language evaluation, CUDA parity
or other Apple hardware was tested. Small floating-point differences from the
new reduction order remain possible for other inputs.

## Reproduce

Use `macos/benchmark.py` with the matching package and library:

```sh
PYTHONPATH=/path/to/baseline/python build/venv/bin/python macos/benchmark.py \
  --model-dir /path/to/model --library /path/to/baseline/libyskn_metal.dylib \
  --output build/before.json --runs 3 --hash-weights
PYTHONPATH=python build/venv/bin/python macos/benchmark.py \
  --model-dir /path/to/model --library build/perf-release/bin/libyskn_metal.dylib \
  --output build/after.json --runs 3 --hash-weights
build/venv/bin/python macos/compare_benchmarks.py \
  --before build/before.json --after build/after.json
```

For the counting task, add to both commands:
`--prompt 'Write the numbers from 1 to 20, separated by commas.' --max-new 16 --fixed-length`.
The comparison tool rejects different settings, output IDs or incomplete runs.
The JSON files here retain timings, settings, outputs and source/library hashes;
only local absolute paths have been replaced by placeholders.

## What changed and was checked

- Aligned 6/8-bit warp kernels reuse packed words and decode value pairs once,
  with independent accumulation chains. Generic layouts retain their decoder.
- Prefill batches at most 32 tokens by layer, using GPU tiles of up to eight.
  Recurrent state and causal attention advance in token order.
- Decode submits independent GDN inputs, Q/K/V and MLP gate/up projections
  together: 401 → 257 matrix submissions per token for this model.
- 195 Python tests: 162 passed, 33 optional image-dependency skips. Coverage
  includes continuation across chunk/KV boundaries, cancellation and reset,
  grouped projections, and comparison with an independent sequence oracle.
- 92 real-GPU fixture/error checks and allocation-failure/retry checks passed.
  Three released-model body matrices with five inputs matched the independent
  decoder (maximum absolute error `1.55e-5`). Grouped projection results were
  bit-identical to separate calls with the same new kernel.

Exploratory timings were used to select the implementation, not mixed into the
table. SIMD lane splitting and a smaller eight-token model chunk were rejected
after regressions. A two-model simultaneous diagnostic was stopped because its
additional GPU resources changed the memory conditions. The final runs above
use one model and three preselected warm repetitions per task.
