# Experimental Metal compressed operations

`yskn_metal.mm` implements matrix-vector multiplication and embedding-row decoding
for the public YSKN v1 format. It is used by the macOS Python model implementation,
not by the CUDA ABI. An Apple Silicon Mac with macOS 13 or later is required.

The kernels read the compressed trellis stream directly. Input and output H512
transforms run on the CPU; the matrix products and compressed row decoding run on
Metal. The aligned 6/8-bit warp path reads packed words and half-precision value pairs directly; other valid layouts retain the byte-addressed decoder.
For single-token projections, 128 threads cooperatively cache the 16 KiB value
table in threadgroup memory before using the same accumulation order. Batched
projections keep their original kernels. Devices that cannot create or support
the cached pipeline retain the original fast scalar path.
The model file is privately memory-mapped and exposed through page-aligned
Metal buffers without expanding or copying the complete weights. When the whole
file fits the device's `maxBufferLength`, all tensors share one buffer. A
page-aligned binding offset and a separate page-local shader offset preserve
addressing above 4 GiB. This avoids registering many overlapping buffer views
while preserving the existing kernels and projection arithmetic.
When the file exceeds the device limit, buffers are created lazily per tensor.
A failed whole-file allocation is reported as an open error; it does not
silently fall back or retry. The caller can retry opening a new handle. Keep
the file unchanged while a handle is open. These APIs also support macOS 14,
the minimum version for the distributed application.

Supported storage: 4–8 bits per step, row/warp layouts, zero start, H512 rotations,
mulberry32/splitmix64 signs, physical row padding, optional embedding residuals, and plain BF16/F16 embedding vectors.
Unsupported encodings are rejected. `yskn_metal_row` requires no output rotation.
The backend uses the stored half-precision value table and float32 arithmetic,
without the CUDA engine's activation/table quantization, speculative decoding, or
KV codec; CUDA bit-for-bit agreement and CUDA speed figures do not apply.

Build with ARC and link Foundation and Metal. `kernels.hpp` embeds the shader
source, so installed users do not need Xcode or standalone shader files.
`yskn_metal.h` defines the C ABI; functions returning an integer return 0 on
success and -1 on failure, with a thread-local message from
`yskn_metal_last_error`. Calls on one handle are serialized; do not close a handle
while another thread is using it.

`yskn_metal_linear_batch` accepts row-major `x[batch, cols]` and
`y[batch, rows]`. It rotates each input and processes at most eight tokens per
GPU tile, reusing a decoded weight pair across those tokens. Scratch is bounded
by the tile size, not the prompt length. The generic encoding path evaluates
one input at a time. Floating-point reduction order can differ from individual
matrix calls; numerical agreement is tested against the independent reader.

`yskn_metal_linear_multi` applies one input to one, two, or three matrices
with matching column counts and returns their logical outputs concatenated.
It validates every shape and reserves disjoint partial/output ranges, then
encodes all projections into one command buffer and waits once. Matrices with
identical input rotations share the prepared input range; different rotations
keep separate input ranges. Each matrix retains its own input/output rotation,
including different signs and padded
physical rows. This reduces submission waits for Q/K/V and gate/up projections
without changing their mathematical operations.

From the repository root, with NumPy installed:

```sh
python src/metal/check_metal.py build/bin/libyskn_metal.dylib
```

This runs real Metal operations against the independent decoder in
`tools/yskn_decode.py` using the official tiny fixture, plus malformed-file and
shape-error checks. It fails if Metal is unavailable (including sandbox denial);
it does not substitute CPU results or count unavailable hardware as a pass.

The standalone allocation regression injects a failed whole-file allocation and
retries opening, then forces the device buffer-size limit to exercise per-tensor
fallback. It checks that a failed tensor initialization and failed scratch growth
leave subsequent calls usable, and that fallback output matches the shared map:

```sh
clang++ -std=c++17 -O3 -fobjc-arc -Ithird_party src/metal/check_allocation.mm \
  -framework Foundation -framework Metal -o build/check_metal_allocation
build/check_metal_allocation tools/testdata/tiny.yskn
```

For an installed released model, compare selected embeddings and body matrices
with bounded row-block decoding. The default selection also includes a late
body matrix whose stream starts above 4 GiB, when present, to check large binding
offsets in both single and batched operations:

```sh
python src/metal/check_model_ops.py build/bin/libyskn_metal.dylib /path/to/model.yskn
```

This check validates individual operations; it does not establish whole-model
generation quality or CUDA equivalence.

For paired warmed timings with identical inputs (and alternating run order):

```sh
python src/metal/bench_metal.py baseline.dylib candidate.dylib /path/to/model.yskn --batch-sizes 1 8 23 --out timings.json
```

The report records both library hashes, every measured sample, the median and
output differences. Batch sizes greater than one use repeated single calls for
a baseline library that lacks the batch ABI. These measurements cover matrix
operations only; report generation latency separately with the same prompts,
sampling, model and hardware.
