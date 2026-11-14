# Metal decode regression fix — 2026-11-15 JST

The per-tensor mapping in `f342895` registered up to 1,206 weight-buffer views
for the released speed model. The fix shares one no-copy Metal buffer for the
model file when it fits `maxBufferLength`. Scratch buffers remain separate.
Page-aligned host binding offsets preserve addresses above 4 GiB; shader
offsets remain page-local uint32 values. The kernels, projection arithmetic,
Python model, sampling and prefill chunking are unchanged.

This comparison demonstrates faster sustained decoding as well as lower
prefill and total latency. It supersedes the decode regression documented in
[the first optimization report](../../20261114/macos/README.md), which is
retained as historical evidence.

## Real-model results

Apple M2 Max, 32 GiB, macOS 26.6.2; Python 3.14.10, NumPy 2.5.4 (Apple
Accelerate). Same public `yasakani-dev/Qwen3.8-27B-YASAKANI-speed` checkpoint,
revision `cd78f90d205677f8b3d2ce95eae82a64e278b910`. Baseline is a frozen
`f342895b3105aa34c1b877273a0228469c4339a4` checkout and its packaged Release
library. Candidate is CMake Release, arm64/macOS 14 target; the report's source
hashes identify the tested working tree before the fix commit.

Each process makes one first request and three warm requests. Every request
starts with fresh recurrent/KV state, without prefix caching. Greedy sampling,
thinking disabled, context 512, and fixed output lengths are identical on both
sides. One model process runs at a time, with the launcher server stopped.
Normal desktop activity is not disabled. The 16-token case ran baseline then
candidate; the 64-token case ran candidate then baseline to vary measurement order.

Warm medians (each metric is summarized independently):

| Output length | Metric | Before (`f342895`) | After | Speedup |
|---|---|---:|---:|---:|
| 16 tokens | Decode | 1.846 token/s | 2.476 token/s | 1.341× |
| 16 tokens | Prefill | 6.104 s | 4.537 s | 1.345× |
| 16 tokens | Total | 13.527 s | 10.637 s | 1.272× |
| 64 tokens | Decode | 2.184 token/s | 2.474 token/s | 1.133× |
| 64 tokens | Prefill | 7.030 s | 4.609 s | 1.525× |
| 64 tokens | Total | 37.395 s | 30.074 s | 1.243× |

Decode throughput counts `(output_tokens - 1) / generation_seconds` because
the first token comes from prefill. The warm decode ranges were:

- 16 tokens: before 1.724–2.306, after 2.378–2.545 token/s.
- 64 tokens: before 1.844–2.198, after 2.458–2.478 token/s.

Every candidate warm decode rate exceeded every baseline warm rate in its case.
First-request totals also improved: 21.927 → 16.306 s (16 tokens) and
45.877 → 35.742 s (64 tokens). “First” means first request in that process;
filesystem/GPU caches were not flushed. Model-load timing is recorded separately
in each raw report. These are bounded device measurements, not confidence
intervals or a guarantee for other prompts/hardware.

All four repetitions per side produced **identical output token IDs and text**.
Prompts are `Write the numbers from 1 to 20, separated by commas.` and
`Write the numbers from 1 to 100, separated by commas.` respectively (27 and
28 input tokens). The fixed limits intentionally truncate the requested lists;
they are performance probes, not completed-answer quality evaluations.

## Acceptance and correctness

Both comparisons pass `macos/compare_benchmarks.py --require-improvement`:
matching inputs/settings/model identity/output IDs, plus strictly improved warm
median **prefill, decode and total**. The optional gate returns failure for a
decode regression even if total latency improves. The previous real-model
regression report is rejected by this gate; tests also cover ties and mismatched
conditions. All repetitions and gate results are included here, without removing
outliers. Local absolute paths are the only redactions.

- Python: 198 tests, 165 passed and 33 optional image-dependency skips.
- Physical GPU: 92 fixture/reference/error checks passed.
- Allocation fault injection: whole-map open failure/reopen, forced per-region
  fallback failure/retry, scratch growth failure/retry, and bit-identical
  shared/fallback fixture results passed.
- Independent format reader: three embedding rows and four body matrices passed,
  including `layers.63.mlp.down_proj` at byte offset **9,111,902,208**. Single
  and five-input batch operations matched; maximum absolute error `1.57e-5`.
- After timed runs, the 10,311,344,128-byte model was rehashed and matched
  `3ba17a76871a2b5e86fb1d01ab677e6b35ff5ccc27d4f03ad941501e3319d1ca`.
  Its size and mtime match all four run reports. See `weights-verification.json`;
  the individual timed runs themselves only recorded the manifest expectation.

The quality checkpoint, other Macs, actual macOS 14 hardware, long-context
quality and CUDA parity remain unverified. A file larger than the device's
maximum buffer length retains the previous per-region path; this report does
not establish a speedup for that fallback. No macOS 15-only API is required.

## Reproduce

Build the native library in Release mode. Run each side in a separate process
with the matching Python package and library; keep other model servers stopped.

```sh
PYTHONPATH=/path/to/f342895/python build/venv/bin/python macos/benchmark.py \
  --model-dir /path/to/model --library /path/to/f342895/libyskn_metal.dylib \
  --output build/before16.json --runs 3 \
  --prompt 'Write the numbers from 1 to 20, separated by commas.' --max-new 16 --fixed-length
PYTHONPATH=python build/venv/bin/python macos/benchmark.py \
  --model-dir /path/to/model --library build/perf-release/bin/libyskn_metal.dylib \
  --output build/after16.json --runs 3 \
  --prompt 'Write the numbers from 1 to 20, separated by commas.' --max-new 16 --fixed-length
build/venv/bin/python macos/compare_benchmarks.py \
  --before build/before16.json --after build/after16.json --require-improvement
```

For the 64-token case, replace `20` with `100`, `--max-new 16` with
`--max-new 64`, and use distinct output filenames. Preserve failed comparisons
instead of replacing them with successful measurements from different settings.
