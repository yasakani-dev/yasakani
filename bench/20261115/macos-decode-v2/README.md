# Metal decode optimization, second stage — 2026-11-15 JST

This change builds on the shared model mapping in `143dd06`. The new scalar
6/8-bit kernels load the value table into threadgroup memory, with
constant-address reads for the inputs and table source. The threadgroup cache
is used only for single-input tiles; multi-input prefill retains its existing
batched kernels. Unsupported scalar pipelines fall back to the previous fast
kernels, and other tensor formats retain the generic path.

Grouped projections reuse the H512-transformed input only when their width,
sign seed, offset and sign rule match. Different rotations still have separate
inputs, and every projection retains its own partial and output buffers. The
NumPy sigmoid selects its numerator before division, avoiding an unused branch.
The reduction order and the sequential generation policy are retained.

## Measurement protocol

Apple M2 Max, 32 GiB unified memory, macOS 26.6.2; Python 3.14.10 and NumPy
2.5.4. The public model is `yasakani-dev/Qwen3.8-27B-YASAKANI-speed`, revision
`cd78f90d205677f8b3d2ce95eae82a64e278b910`. The baseline package, native source
and library are frozen at `143dd06d250f0f7c298f6cb562fa589887e00b5f`. The candidate
is a CMake Release build targeting arm64/macOS 14. Raw reports identify the
exact imported Python files and native library by SHA-256.

Each case runs one first request followed by three warm requests in the same
process. State is reset on every request; prefix caching is disabled. Both
sides use greedy sampling, thinking disabled, a 512-token context and fixed
output lengths that ignore EOS. The 16-token prompt is `Write the numbers from
1 to 20, separated by commas.` (27 input tokens); the 64-token prompt changes
`20` to `100` (28 input tokens). These deliberately truncated lists measure
generation work, not completed-answer quality.

Only one model process runs at a time. Measurement order was baseline 16,
baseline 64, exploratory variants, final candidate 16, then final candidate 64.
A repeat of the frozen baseline 64 ran immediately after the final candidate
as a separate reverse-order confirmation. The primary table retains
the original baseline rather than replacing it with a later measurement.
Normal desktop activity remained enabled. This sequential comparison does
not control all changes in system load, thermal state or filesystem/GPU caches.

## Real-model results

Warm medians, summarized independently for each metric:

| Output length | Metric | Before (`143dd06`) | Final candidate | Speedup |
|---|---|---:|---:|---:|
| 16 tokens | Decode | 2.472 token/s | 3.978 token/s | 1.609× |
| 16 tokens | Prefill | 4.633 s | 4.308 s | 1.075× |
| 16 tokens | Total | 10.631 s | 8.081 s | 1.316× |
| 64 tokens | Decode | 2.503 token/s | 3.270 token/s | 1.307× |
| 64 tokens | Prefill | 4.783 s | 4.681 s | 1.022× |
| 64 tokens | Total | 29.959 s | 23.947 s | 1.251× |

Both primary comparisons pass the strict prefill/decode/total improvement gate, with
identical token IDs and text in all four repetitions on both sides. Warm decode
ranges are 2.438–2.522 → 3.948–4.032 token/s for 16 tokens, and
2.431–2.520 → 3.261–3.410 token/s for 64 tokens. First-request total latency is
16.701 → 13.698 s and 35.120 → 30.183 s, respectively; first requests are
excluded from the warm medians.

The additional reverse-order 64-token comparison confirms faster decode
(2.406 → 3.270 token/s, 1.359×) and lower total latency (30.887 → 23.947 s).
However, prefill is 4.622 → 4.681 s, **1.27% slower**, so
`confirm-comparison64.json` records `performance_gate.status: failed`.
It is not counted as an all-metric pass or substituted into the primary table.
Its baseline prefill range, 4.619–4.703 s, overlaps the candidate's
4.478–4.904 s. The supported result is faster sustained decoding with matching
outputs; a stable prefill improvement is not established.

Decode throughput is `(generated_tokens - 1) / generation_seconds`: prefill
predicts the first token. “First” means the first request after loading the
model in that process, with no cache flush. No outlier is removed. This is a
device-specific measurement, not a confidence interval or a guarantee for
other models, prompts, context lengths or hardware.

An earlier constant-read/cache-only candidate reached 0.998× for 64-token
decode (2.503 → 2.498 token/s), failing the gate despite lower prefill and total
latency; it was not adopted. Fused reduction and other exploratory variants
are also absent from the final implementation. Decoding was evaluated with
complete generation requests; matrix-only speedups were not sufficient.

## Correctness and evidence

- Python suite: 199 tests, 166 passed, 33 optional image-dependency skips.
- Physical Metal GPU: 104 fixture/reference/error checks passed.
- Allocation-failure/retry and forced fallback checks passed.
- Independent format decoder: three embedding rows and four body matrices
  passed, with single-input and five-input batch products. Coverage includes
  `layers.63.mlp.down_proj` at byte offset **9,111,902,208**; maximum absolute
  error was `1.567602e-5`.
- Separate exact-decode diagnosis: all eight full-vocabulary logits hashes,
  output token IDs/text and the final model state hash matched the frozen
  baseline. This covers one 27-token prefill plus seven single-token forwards.
  The final state contains the prompt and the first seven generated tokens;
  the eighth prediction has not been forwarded.
- After all timed runs, a complete-file SHA-256 check matched
  `3ba17a76871a2b5e86fb1d01ab677e6b35ff5ccc27d4f03ad941501e3319d1ca`.
  The 10,311,344,128-byte file's size and mtime match every published run;
  `weights-verification.json` records the evidence. This check is outside
  the timers; the timed reports themselves only record the manifest expectation.

The exact diagnostic hashes the float32 logits viewed as little-endian uint32
bytes. It records no latency metrics: CPU hashing and report writes occur
inside generation and would contaminate a speed comparison. Agreement on this
bounded sequence does not prove general language quality or CUDA parity.

The raw result set uses these filenames:

- `count16-before.json`, `count16-after.json`, `count16-comparison.json`.
- `count64-before.json`, `count64-after.json`, `count64-comparison.json`.
- `confirm-before64.json`, `confirm-comparison64.json` preserve the additional
  reverse-order baseline and its comparison against the same final candidate.
- `cache-only-after16.json`, `cache-only-comparison16.json`,
  `cache-only-after64.json`, `cache-only-comparison64.json` preserve the earlier
  unadopted candidate; their baselines are the corresponding `count*-before.json`.
- `exact-before.json`, `exact-after.json`, `exact-comparison.json`.
- `weights-verification.json` records the separate model verification; timed
  runs only record the manifest's expected checksum.

The quality checkpoint, other Macs, actual macOS 14 hardware, long-context
behavior and broad prompt coverage remain unverified. The previous
[shared-mapping report](../macos-decode/README.md) and
[initial optimization report](../../20261114/macos/README.md) remain as history.

## Reproduce

Keep a frozen checkout of `143dd06` and its matching Release library. Build the
candidate library, using the same Python environment for both sides:

```sh
cmake -S . -B build/perf-release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build/perf-release --target yskn_metal
```

Run each side in a separate process and wait for it to exit before starting
the next. Keep other model servers stopped. Explicit `PYTHONPATH` selects the
matching Python implementation; `--library` selects the native implementation.

```sh
PYTHONPATH=/path/to/143dd06/python build/venv/bin/python macos/benchmark.py \
  --model-dir /path/to/model --library /path/to/143dd06/libyskn_metal.dylib \
  --output build/before16.json --runs 3 --max-ctx 512 \
  --prompt 'Write the numbers from 1 to 20, separated by commas.' --max-new 16 --fixed-length
PYTHONPATH=python build/venv/bin/python macos/benchmark.py \
  --model-dir /path/to/model --library build/perf-release/bin/libyskn_metal.dylib \
  --output build/after16.json --runs 3 --max-ctx 512 \
  --prompt 'Write the numbers from 1 to 20, separated by commas.' --max-new 16 --fixed-length
build/venv/bin/python macos/compare_benchmarks.py \
  --before build/before16.json --after build/after16.json \
  --output build/comparison16.json --require-improvement
```

For 64 tokens, replace `20` with `100`, set `--max-new 64`, and use separate
`before64`, `after64` and `comparison64` output files. The gate requires
comparable settings and outputs, plus strictly lower warm-median prefill,
decode and total latency. A decode tie or regression fails even if total time
improves. Preserve every repetition and rejected result.

Run the independent bitwise diagnostic separately from speed measurements:

```sh
PYTHONPATH=/path/to/143dd06/python build/venv/bin/python macos/check_decode.py \
  --model-dir /path/to/model --library /path/to/143dd06/libyskn_metal.dylib \
  --output build/exact-before.json --steps 8
PYTHONPATH=python build/venv/bin/python macos/check_decode.py \
  --model-dir /path/to/model --library build/perf-release/bin/libyskn_metal.dylib \
  --output build/exact-after.json --steps 8
```

Compare completed reports without comparing the intentionally different
package/library provenance:

```sh
build/venv/bin/python - <<'PY'
import json
from pathlib import Path
before = json.loads(Path('build/exact-before.json').read_text())
after = json.loads(Path('build/exact-after.json').read_text())
assert before['status'] == after['status'] == 'complete'
for key in ('settings', 'model', 'logits', 'generated', 'final_state'):
    assert before[key] == after[key], key
print('PASS: logits, outputs and final state match for this sequence')
PY
```

This diagnostic is specific to the Qwen3.8 speed model's bundled template and
rejects a prompt that does not format to exactly 27 tokens.
