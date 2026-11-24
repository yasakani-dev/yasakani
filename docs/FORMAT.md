# The YASAKANI model file format (`.yskn`), version 1

日本語：[FORMAT.ja.md](FORMAT.ja.md)

This document says what a `.yskn` file means, byte by byte, so that anyone can read one without the engine, check it against the
original weights, or write one.

* A reader that follows this text and nothing else: [`tools/yskn_decode.py`](../tools/yskn_decode.py) (Python and numpy, no GPU).
* A small test file and the values a reader must get from it: [`tools/testdata/`](../tools/testdata/).
* The reader of the engine: [`src/format.hpp`](../src/format.hpp) and [`src/loader.cpp`](../src/loader.cpp).
* Version 2, for Inkling-2 models (NVFP4, MXFP8 and bfloat16 tensors): [section 10](#10-version-2-inkling-2-files).

The program that made the released files from the original weights is in [`tools/convert/`](../tools/convert/) (added on
November 14, 2026); its method is described in [TECHNOLOGY.md](TECHNOLOGY.md), section 3.2. See [section 9](#9-writing-a-file).

## 1. Overview

A file holds a JSON header and, after it, blocks of data that the header points to.

A weight matrix is not stored as numbers. It is stored as

* a **code stream**: 3 or 4 bits per weight,
* a **value table**: 4,096 pairs of numbers, one table per matrix,
* a **scale**: one number per row,

and all of this in **rotated coordinates**: the rows and columns of the matrix are mixed by a fixed Hadamard transform with
pseudo-random signs before they are coded. The signs are not stored; the header holds the seed they are made from.

Small parameters (normalisation weights and the like) are stored as plain 16-bit floating-point numbers and are called
**vectors**.

All integers are little-endian.

## 2. File layout

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | the bytes `YSKN` |
| 4 | 4 | version, unsigned 32-bit: `1` |
| 8 | 8 | length of the JSON header in bytes, unsigned 64-bit |
| 16 | 8 | offset of the first data block, unsigned 64-bit (for information) |
| 24 | 8 | length of the whole file in bytes, unsigned 64-bit |
| 32 | (length) | the header: one JSON object, UTF-8 |

The bytes between the end of the header and the first data block are zero. Every data block is found through an absolute offset
(from the start of the file) in the header. The format does not ask for any alignment; in the released files every block starts at
a multiple of 256 bytes.

## 3. The header

The JSON object has three members: `model`, `tensors`, `vectors`.

### 3.1 `model`

The dimensions of the model (`arch`, `layers`, `hidden`, `ffn`, `heads`, `kv_heads`, `head_dim`, `vocab`, `eps`, `rope_theta`,
`tie_embeddings`), further members that depend on the architecture, and `source`: where the original weights came from (for the
released models the Hugging Face repository and revision). How the model computes with its tensors is defined by the original model
and by the engine source for that `arch`; this document covers only how the tensors are stored.

### 3.2 `tensors`

A list. Each entry describes one matrix with `rows` rows and `cols` columns (`y = W x`: `cols` inputs, `rows` outputs).

| Member | Meaning |
|---|---|
| `name` | for the released models: the parameter name of the original checkpoint, shortened (section 6) |
| `role` | `body`, `head` (the output matrix), `embedding`, or `draft` (a matrix of the helper model) |
| `rows`, `cols` | the shape. `cols` is a multiple of 512 |
| `physical_rows` | rows that are stored; default `rows`. Rows beyond `rows` are padding |
| `l` | bits of the state; default 12. This document and the engine cover 12 only |
| `d` | weights per step; default 2, the only value |
| `t` | weights per sequence; default 256, the only value |
| `bits` | bits per step: 6 (3 bits per weight) or 8 (4 bits per weight) in the quality and fastest models, 10 (5 bits per weight) in the max-quality model; the reader takes 4 to 8 and 10. Sections 4.1 to 4.3 define 10 the same way (a sequence is 40 words, a symbol may cross a word). The engine reads 10 for matrices (body and output), not for the embedding table |
| `start` | `zero` (default), or `carry` (not used by the released files and not defined here) |
| `layout` | how the code stream is ordered: `warp` (default) or `row` (section 4.3) |
| `stream_off`, `stream_bytes` | the code stream |
| `lut_off` | the value table: 8,192 half-precision floats (16,384 bytes) |
| `scale_off` | the scales: `physical_rows` half-precision floats |
| `rot_in`, `rot_out` | the rotations of the input side (columns) and the output side (rows), section 3.3 |
| `rar` | `null`, or an object for per-tile bit widths (not used by the released files and not defined here; see `src/loader.cpp`) |
| `bias_off`, `proxy`, `encoder` | not used by a reader. `bias_off` is 0; `proxy` is a number the conversion recorded; `encoder` is a text |

"Half-precision float" always means IEEE 754 binary16.

### 3.3 Rotations

`rot_in` and `rot_out` are objects:

| Member | Meaning |
|---|---|
| `kind` | `H512` (sections 4.5 and 4.6). `none` is allowed for `rot_out`. `AYA` is a second kind that the released files do not use and this document does not define |
| `m` | number of blocks of 512 (`cols / 512` for `rot_in`, `rows / 512` for `rot_out`) |
| `sign_seed`, `sign_offset` | where the signs come from (section 4.5) |
| `rule` | `splitmix64` (default) or `mulberry32_7919_13` |
| `qm_off` | used by `AYA` only; 0 otherwise |

If `rot_out` is `null` or has the kind `none`, the tensor has no output rotation. The reader also has defaults for files that leave
the members out (the released files always write them): `rot_in` absent means `H512`, seed 7, offset 0, rule `mulberry32_7919_13`;
`rot_out` absent means no output rotation for the roles `head` and `embedding`, and otherwise `H512`, seed 7, offset `cols`, rule
`mulberry32_7919_13`.

### 3.4 `vectors`

A list of `{ "name", "off", "n", "dtype" }`: `n` 16-bit numbers at the offset `off`. `dtype` is `f16` (default; half-precision
float) or `bf16` (bfloat16: the upper 16 bits of an IEEE 754 single-precision float; to read it, put the 16 bits into the upper
half of a 32-bit float and zeros into the lower half).

## 4. Decoding a matrix

### 4.1 Sequences and steps

Each row is cut into `S = cols / 256` **sequences** of 256 weights. A sequence is coded in 128 **steps**; a step yields two
weights and takes `bits` bits of code. A sequence therefore has `128 * bits` bits of code: `W = 4 * bits` 32-bit words.

Within a sequence, bit number `p` (counted from 0) is bit `p mod 32` of word `p div 32`, where bit 0 of a word is its least
significant bit.

### 4.2 From bits to values

For step `j` (0 to 127) let `e = bits * (j + 1)`, the position just after the bits of this step. The **state** of the step is the
12 bits that end there:

    state = sum over i = 0..11 of  b[e - 12 + i] * 2^i          (b[p] = bit p of the sequence; b[p] = 0 for p < 0)

So the newest bits sit at the top of the state, and each step shifts the older ones down. Only the first step reaches before the
start of the sequence; there the bits count as zero.

The value table has 4,096 pairs. Step `j` yields

    v[2 j]     = table[2 * state]
    v[2 j + 1] = table[2 * state + 1]

and with the scale of the row, the stored matrix in rotated coordinates is

    M[r][256 * s + k] = scale[r] * v[k]           (row r, sequence s, k = 0..255)

Because each state shares 12 - `bits` bits with the state before it, the value of a step depends on the bits of the step before as
well: this is a trellis code. Any bit string is a valid code.

### 4.3 Where the words of a sequence are

`stream_off` points to `physical_rows * S * W` 32-bit words. Word `w` (0 to `W` - 1) of sequence `s` of row `r` is the word
number

* layout `row`:  `(r * S + s) * W + w`
* layout `warp`: with `q = s * physical_rows + r`, `g = q div 32`, `lane = q mod 32`, `c = w div 4`:
  `((g * bits + c) * 32 + lane) * 4 + (w mod 4)`

In words: `warp` puts the sequences in the order "all rows of sequence 0, all rows of sequence 1, ..." and interleaves every 32
of them in pieces of four words (16 bytes), so that 32 GPU threads that decode 32 rows read neighbouring memory. `warp` needs
`physical_rows * S` to be a multiple of 32 (the engine asks for a multiple of 64 rows).

### 4.4 The Hadamard transform

`H` is the 512 x 512 Hadamard matrix in natural (Sylvester) order: `H[a][b] = +1` if the number of one-bits of `a AND b` is even,
`-1` if it is odd. It is used with the factor `1 / sqrt(512)`; `H / sqrt(512)` is symmetric and its own inverse. A vector whose
length is a multiple of 512 is transformed block by block.

### 4.5 Signs

`signs(rule, seed, offset, n)` is the list of `n` values `+1` / `-1` at the positions `offset .. offset + n - 1` of an endless
stream. With 64-bit and 32-bit unsigned arithmetic that wraps around:

    splitmix64, position k (from 0):
        z = seed + (k + 1) * 0x9E3779B97F4A7C15
        z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z XOR (z >> 27)) * 0x94D049BB133111EB
        z = z XOR (z >> 31)
        sign = +1 if the top bit of z is set, else -1

    mulberry32_7919_13, position k (from 0):
        s0 = (seed mod 2^32) * 7919 + 13            (if this is 0: 0x9E3779B9)
        t = s0 + (k + 1) * 0x6D2B79F5
        t = (t XOR (t >> 15)) * (t OR 1)
        t = t XOR (t + (t XOR (t >> 7)) * (t OR 61))
        t = t XOR (t >> 14)
        sign = -1 if t < 2^31, else +1

The input side uses `s_in = signs(rot_in.rule, rot_in.sign_seed, rot_in.sign_offset, cols)`, the output side
`s_out = signs(rot_out.rule, rot_out.sign_seed, rot_out.sign_offset, physical_rows)`.

### 4.6 Back to the coordinates of the original model

With `M` from section 4.2:

1. Input side, for every row: transform the row with `H / sqrt(512)` (blocks of 512 columns), then multiply column `i` by `s_in[i]`.
2. Output side, only if the tensor has an output rotation, for every column: transform the column with `H / sqrt(512)` (blocks of
   512 rows), then multiply row `n` by `s_out[n]`.

The result `W` is the matrix the file stands for, in the coordinates of the original model. As a formula, with `D_in` and `D_out`
the diagonal matrices of the signs and `Hn` the block-wise `H / sqrt(512)`:

    W = D_out * Hn * M * Hn * D_in          (without an output rotation: W = M * Hn * D_in)

The engine never builds `W`. It computes `y = W x` as `x' = Hn (D_in x)`, `z = M x'`, `y = D_out (Hn z)`, with `M` taken straight
from the codes.

## 5. What the engine does with it

This is not part of the format, but it explains small differences between a decoded matrix and what the engine computes.

* For matrices (the roles `body`, `head`, `draft`) the engine rounds the value table to 8 bits when it loads the file:
  `gamma = (largest absolute value of the table) / 127`, `q = round(value / gamma)` limited to -127 .. 127, and the scale of row `r`
  becomes `scale[r] * gamma`. `yskn_decode.py --engine-table` decodes with this table. The embedding is read with the 16-bit table.
* The input of a matrix, after its rotation, is rounded to 8-bit integers with one scale per vector, so that the product runs on
  integers ([TECHNOLOGY.md](TECHNOLOGY.md), section 4).
* At load time the engine reorders the code stream once more for its kernels (`src/repack.cu`). This does not change the meaning.

## 6. The released files

| File | Tensors | Vectors |
|---|---|---|
| quality model, `qwen38_27b_quality.yskn` | 400 `body` (288 with 8 bits per step, 112 with 6), `head` (8), `embedding` (6) | 449 |
| fastest model, `qwen38_27b_speed.yskn` | 400 `body` (all with 6 bits per step), `head` (8), `embedding` (6) | 449 |
| helper model, `bdraft2_k8.yskn` | 46 `draft` (8 bits per step, no output rotation) | 35 |

All of them use `l` = 12, `start` = `zero`, the kind `H512` with the rule `mulberry32_7919_13` and seed 7, the layout `warp` for
matrices and `row` for the embedding, and `bf16` vectors. None uses `rar`, `AYA` or `carry`.

The names follow the original checkpoint (`Qwen/Qwen3.8-27B`): the tensor or vector `layers.N.X` is the parameter
`model.language_model.layers.N.X.weight` (or `model.language_model.layers.N.X` where the original has no `.weight`), `embed_tokens`
is `model.language_model.embed_tokens.weight`, `norm` is `model.language_model.norm.weight`, `lm_head` is `lm_head.weight`. A vector
holds the numbers of the original parameter in their original order (a parameter with several axes is flattened). The parameters of the
original checkpoint outside `model.language_model.*` and `lm_head.weight` (the vision part `model.visual.*` among them) are not in the
files.

## 7. Checking a file yourself

    python tools/yskn_decode.py info    qwen38_27b_quality.yskn
    python tools/yskn_decode.py compare qwen38_27b_quality.yskn --original <folder with the *.safetensors of Qwen/Qwen3.8-27B> --json out.json
    python tools/yskn_decode.py decode  qwen38_27b_quality.yskn layers.3.self_attn.k_proj --out k_proj.npy

`compare` decodes every tensor, compares it with the original and prints the relative error
`|W_decoded - W_original| / |W_original|` (Frobenius norm); for the vectors it checks that they are bit for bit the original
numbers. It needs the original checkpoint (about 54 GB) and about half an hour on a desktop CPU.

What we get for the released files (the full lists are in [`tools/records/`](../tools/records/)):

| Model | Tensors | Bits per weight | Relative error: median (smallest to largest) |
|---|---|---|---|
| quality model | 288 body matrices | 4 | 0.092 (0.085 to 0.106) |
| quality model | 112 body matrices | 3 | 0.177 (0.166 to 0.199) |
| fastest model | 400 body matrices | 3 | 0.177 (0.163 to 0.202) |
| both models | the output matrix | 4 | 0.087 |
| both models | the embedding | 3 | 0.146 |

All 449 vectors of each model are bit for bit the numbers of the original checkpoint.

A relative error of this size is what 3 or 4 bits per weight give; it says how far the stored weights are from the original ones,
not how good the model's output is. The conversion chooses the codes so that the output of the model stays close to the original
([TECHNOLOGY.md](TECHNOLOGY.md), section 3.2), which is not the same as making this number as small as possible. The quality of the
output is measured separately ([MEASUREMENTS.md](MEASUREMENTS.md)).

## 8. Test data

[`tools/testdata/tiny.yskn`](../tools/testdata/tiny.yskn) (468,304 bytes, SHA-256
`3f4a48866f6a1751beabc3b9a8d06088672a3db0aa0a4c3f96e1877cb448d393`) is a container with four small tensors and two vectors: 6 and 8
bits per step, both layouts, both sign rules, with and without an output rotation, both vector types. It holds no model: streams,
tables and scales are pseudo-random. [`tools/yskn_fixture.py`](../tools/yskn_fixture.py) writes the same file again, byte for byte.

[`tools/testdata/tiny_expected.json`](../tools/testdata/tiny_expected.json) lists what a reader must get: the header numbers, the
SHA-256 of several sign streams (as signed bytes), and for every tensor the SHA-256 of the states (unsigned 16-bit, little-endian,
rows x sequences x 128) and of the values in rotated coordinates (32-bit floats, little-endian, rows x cols; the product of the
table value and the scale, both converted to 32-bit floats first), some weights at given positions, and the Frobenius norm of the
weights. `python tools/yskn_decode.py selftest` checks the reader of this repository against it, and also reads a few rows a
second time with a slow, bit-by-bit routine written from section 4.

## 9. Writing a file

A writer has to produce, for every matrix, a value table, a scale per row and a code stream such that the decoded matrix is
close to the one it wants to store, and then lay the data out as in sections 2 to 4. Since every bit string is a valid code,
`tools/yskn_fixture.py` is a complete (if useless) writer: it fills the streams with pseudo-random bits.

Choosing good codes is the hard part. The program that made the released files is in [`tools/convert/`](../tools/convert/), with
the data it needs; its method is described in [TECHNOLOGY.md](TECHNOLOGY.md), section 3.2. Run again on the PC that made the
released files, it gave the same codes for every matrix of the body and for the embedding table, and with the output matrix
taken from a released file it gave the released files byte for byte. Whether another PC gives the same codes depends on its
arithmetic ([`tools/convert/README.md`](../tools/convert/README.md), "What we got" and "Limits").

## 10. Version 2: Inkling-2 files

Version 2 holds the weights of an Inkling-2 model (Inkling-2 Mini) as they are stored by the model's own formats: bfloat16 and
float32 tensors as published, and matrices in the two block formats of the published experts, NVFP4 and MXFP8. It has nothing
to decode beyond these formats (no trellis code, no rotation). The program that writes it is
[`tools/convert/inkling2_convert.py`](../tools/convert/inkling2_convert.py) (`--scheme quality`); the reader of the package is
[`python/yasakani/ink2_yskn.py`](../python/yasakani/ink2_yskn.py).

A reader of version 1 refuses a version 2 file at the version field (`tools/yskn_decode.py` and `src/format.hpp` take version
`1` only), so it never misreads one.

### 10.1 File layout

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | the bytes `YSKN` |
| 4 | 4 | version, unsigned 32-bit: `2` |
| 8 | 8 | length of the JSON header in bytes, unsigned 64-bit |
| 16 | 8 | offset of the first data block, unsigned 64-bit |
| 24 | 8 | length of the whole file in bytes, unsigned 64-bit (a reader compares it with the file) |
| 32 | 4 | CRC-32 (the one of zlib) of the JSON header |
| 36 | 4 | zero |
| 40 | (length) | the header: one JSON object, ASCII |

The first 32 bytes mean what they mean in version 1. Every data block (a **part**, section 10.3) starts at a multiple of the
`alignment` of the header (4,096 bytes: a memory page), so that the file can be memory-mapped and every part handed on without
copying. The bytes between the parts are zero.

### 10.2 The header

| Member | Meaning |
|---|---|
| `format`, `version` | `"yskn"`, `2` |
| `alignment` | 4096 |
| `checksum` | how the parts are checked: `crc32 (zlib) of every part` |
| `model` | `arch` (`inkling2`), `model_id`, `source` (the Hugging Face repository and revision, the name and SHA-256 of the published weights file, the licence and the acceptable-use policy), `scheme` (its name, the rules in words, the search settings, the quantiser), `text_config` (the `text_config` of the model's config.json), `converter` |
| `tensors` | a list, section 10.3 |

### 10.3 Tensors and parts

Each entry of `tensors` has `name`, `encoding`, `shape` (the logical shape: the values the tensor stands for), `role` (`dense`,
`vector` or `experts`), `source` (a text: copied or quantised) and `parts`. Each part has `part`, `dtype` (`bf16`, `f32`, `u8`
or `e4m3`, all little-endian), `shape`, `offset` (absolute), `nbytes` and `crc32` (8 hexadecimal digits of the CRC-32 of its
bytes).

| `encoding` | Parts (shape for a logical shape `[..., R, C]`) | Value |
|---|---|---|
| `bf16` | `data` bf16, the shape | as stored |
| `f32` | `data` f32, the shape | as stored |
| `nvfp4` | `weight` u8 `[..., R, C/2]`; `scale` e4m3 `[..., R, C/16]`; `amax` f32 `[n]`; `denominator` f32 `[1]` | `(e2m1[code] * e4m3[scale]) * (amax / denominator)` |
| `mxfp8` | `weight` e4m3 `[..., R, C]`; `scale` u8 `[..., R, C/32]` | `e4m3[weight] * 2^(scale - 127)` |

* NVFP4: each byte of `weight` holds two 4-bit codes, the element with the even column number in the low four bits. A code
  is an E2M1 number: bit 3 the sign, the others the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6. `scale` holds one E4M3 number for
  every 16 consecutive values of a row. `amax` holds one number per matrix: `n` = 1 for a matrix, the number of experts for an
  expert tensor; `denominator` is 2688 (6 times 448). The products are float32, in the order of the formula.
* MXFP8: `weight` holds one E4M3 number per value, `scale` one byte for every 32 consecutive values of a row.
* E4M3 (`float8_e4m3fn`): sign, 4 exponent bits with bias 7, 3 mantissa bits; exponent 0 is subnormal (`m / 8 * 2^-6`); the
  largest value is 448; `0x7F` and `0xFF` are not numbers and are not written.

These are the formats of the experts of the published Inkling-2 Mini checkpoint, and the parts are laid out as there, so the
engine's expert kernels read them as they are: an expert tensor (`role` `experts`, logical shape `[experts, R, C]`) is stored
exactly like the published one, the rows of `w13` interleaved as published (gate row `i` at `2i`, up row `i` at `2i + 1`). A
dense matrix in NVFP4 or MXFP8 has the same layout as a single expert.

The names are the parameter names of the published checkpoint; an expert tensor is named without the suffix
(`model.llm.layers.N.mlp.experts.w13`; its parts are the published `..._weight`, `..._weight_scale`, `..._weight_amax` and
`..._weight_outer_scale_denominator`). `model.llm.unembed.weight` holds only the rows of the unpadded vocabulary. The file
holds no tensor of the vision tower (`model.visual.*`) and none of the drafter; each can be a file of its own next to it (the
image tower: section 10.7; a converted drafter: section 10.6).

### 10.4 The quality scheme

| Tensors | Encoding |
|---|---|
| `attn.wq_du`, `attn.gate_attn`, `unembed` | NVFP4, one factor per matrix |
| `attn.wk_dv`, `attn.wv_dv`, `attn.wo_ud`, `attn.wr_du`, `attn.attn_sink_proj`, the `weight_proj` of the key and value convolutions, `controllers.*`, `embed` | MXFP8 |
| the experts | NVFP4: as published where the published file has NVFP4; re-packed from MXFP8 (layers 20 to 23) with one factor per expert |
| everything else: the router, the norms, the convolutions, the clip scales, `rel_logits_proj`, the attention-residual queries | as published (bf16 or f32) |

The quantiser ([`tools/convert/inkling2_quant.py`](../tools/convert/inkling2_quant.py)) chooses the block scale of NVFP4 among
eight E4M3 codes around the plain rule, and the exponent of MXFP8 among two, by the smallest squared error of the decoded block.
Against the published model, on 43 text windows (14,024 positions), measured on the decoded values of the written file: KL
0.0110, top-1 agreement 96.0 %. The file of Inkling-2 Mini has 15.05 GB; the published file has 19.5 GB (with the vision tower,
0.64 GB).

### 10.5 Checking a file

`yasakani verify` compares the SHA-256 of every file of the model folder with yasakani.json and then the CRC-32 of every part.
The package checks the CRC-32 of every part before it loads the file, and names the tensor whose bytes differ.

### 10.6 Drafter files

A drafter file holds the drafter of an Inkling-2 model (the published `mtp.safetensors`: three transformer layers and the
recurrent head that proposes up to seven tokens) in the layout of this version: the same prefix, header and parts. It is a
file of its own next to the model file, named in `yasakani.json` by the `drafter` entry with `"format": "yskn"`, `"version": 2`
and the scheme (the published file is `"format": "hf-safetensors"`). It never goes into the model file: a package that knows
only the published drafter refuses the entry by its format and generates without drafting, and the model file stays as it was
(its SHA-256 does not change). `tools/convert/inkling2_convert.py --scheme quality` writes `mtp_d4.yskn` (`--drafter-scheme d4`,
the default; `d8` writes `mtp_d8.yskn`, `published` places the published file instead).

* `model.arch` is `inkling2-drafter`; `model.source` names the published drafter file (name, SHA-256, size); `model.scheme`
  names the scheme; `model.mtp_config` is the `mtp_config` of the model's config.json.
* The tensors keep the names of the published file. Every matrix (2 axes) larger than 64 KB in bfloat16, except
  `markov_head.hidden_out.weight`, is stored in the encoding of the scheme; everything else (norm gains, biases, `fuse_weight`,
  `mask_embed`, and `markov_head.hidden_out.weight`) is `bf16` as published.
* Two more tensors: `fold.weight`, the folded head `U @ markov_head.hidden_out.weight` (`[V, 256]`, `V` the unpadded vocabulary,
  `U` the output projection of the model file as the engine multiplies it; float64 products of the decoded values, rounded to
  float32, then encoded as the scheme's matrices), and `fold.bias` = `U @ markov_head.hidden_out.bias` (`f32`, `[V]`). `model.fold`
  names the `U` they were made with: `unembed_encoding` (`bf16`, `nvfp4` or `mxfp8`) and `unembed_sha256`, the SHA-256 of the
  encoding name, a zero byte, and the bytes of `U` cut to `V` rows (bfloat16: the values; packed: `weight`, `scale`, and for
  NVFP4 the float32 number `amax / denominator`). The `drafter` entry of `yasakani.json` repeats them as `folded_with`. An
  engine whose `U` has another fingerprint folds the head again from `markov_head.hidden_out` and does not use `fold.weight`.

| Scheme | The large matrices and `fold.weight` | File (Inkling-2 Mini) |
|---|---|---|
| `d4` | NVFP4, one factor per matrix (the encoding and the quantiser of the quality scheme's NVFP4 matrices) | 209.7 MB |
| `d8` | MXFP8 (the quantiser of the quality scheme's MXFP8 matrices) | 382.2 MB |

The published drafter file has 635.8 MB. The drafter cannot change what the model generates: every proposed token is checked by
the model, and greedy output is the same token for token with any drafter file; the scheme changes how many proposals are
accepted, the time of a draft call, memory and load time. An engine library that multiplies these matrices as they are stored
says so by bit 4 of `ink2_capabilities()` and by drafter interface version 2 (`ink2_draft_abi()`); with an older library the
package generates without drafting. `yasakani verify` checks the SHA-256 of the file and the CRC-32 of every part, as for the
model file.

### 10.7 Image-tower files

Since {{VISION_VERSION}}, a converted Inkling-2 Mini folder can hold the image tower of the model (the published tensors
`model.visual.*`) in a file of its own next to the model file, `inkling_2_mini_vision.yskn`. The program that writes it is
[`tools/convert/inkling2_vision_convert.py`](../tools/convert/inkling2_vision_convert.py); the reader of the package is
[`python/yasakani/inkling2_vision.py`](../python/yasakani/inkling2_vision.py) (`vision_weights`, `iter_vision_tensors`, with
the version 2 reader of `ink2_yskn.py`). A folder whose weights are the published safetensors file needs no such file: the
image tower is read from the `model.visual.*` tensors of that file.

The file has the layout of this version: the same prefix (version `2`), header and parts, every part at a multiple of 4,096
bytes with its CRC-32. The header has the members of section 10.2 (`model`: `arch` `inkling2_vision`, `model_id`,
`converter`) and these:

| Member | Meaning |
|---|---|
| `kind` | `"inkling2_vision"`; the package refuses any other value |
| `vision_config` | the `vision_config` of the model's config.json. The package needs these values and refuses the file, naming the key, when one is missing or differs: `model_type` `inkling2_vision`, `hidden_size` 1024, `num_hidden_layers` 23, `num_attention_heads` 16, `mlp_ratio` 4.0, `native_patch_size` 20, `adapter_merge_size` 2, `temporal_patch_size` 2, `posemb_grid_size` 32, `projection_hidden_dim` 4096, `projector_bias` false, `layer_norm_eps` 1e-5, `text_hidden_size` 2304 (and, where present, `hidden_act` `gelu`, `num_channels` 3, `rope_theta` 10000) |
| `text_hidden_size` | 2304: the values per image token handed to the language model; the package refuses any other value |
| `source` | where the bytes come from: `file` (the name of the published weights file), `range` (the first byte of the image tower in that file and the byte after its last, absolute), `revision` (the revision of the Hugging Face repository), `sha256` (the SHA-256 of the bytes of that range) |

* `tensors` lists the 422 `model.visual.*` tensors of the published weights file, with their published names and shapes, in
  the order of that file. Each has `encoding` `bf16`, `role` `vision` and one part `data` (`bf16`, the shape of the tensor)
  that holds the bytes of the published file unchanged, 636,512,256 bytes in all; an entry has no `source` member (the
  header's `source` covers them all). The names are `model.visual.backbone.embeddings.*` (4 tensors),
  `model.visual.backbone.layers.N.*` for N = 0 to 22 (18 each), `model.visual.backbone.ln_pre.*` (2),
  `model.visual.proj_in.weight` (`[4096, 4096]`) and `model.visual.proj_out.weight` (`[2304, 4096]`); the full table with the
  shapes is `VISION_TENSORS` in `inkling2_vision.py`.
* Before it uses the file, the package reads the header as for every version 2 file (the CRC-32 of the header, the file length,
  every part inside the file, aligned, with no overlap), then checks `kind`, `vision_config` and `text_hidden_size`, then all
  422 entries: the set of names equals the table (none missing, none unknown), `encoding` `bf16`, and the shape and byte count
  of every tensor as in the table. `yasakani verify` checks the file by its size and SHA-256.
* The file of Inkling-2 Mini has 637,136,896 bytes.

`yasakani.json` names the file in the entry `vision` and also lists it in its `sha256` and `sizes` maps:

    "vision": {"file": "inkling_2_mini_vision.yskn", "format": "yskn", "version": 2,
               "sha256": "<SHA-256 of the file>", "size": <bytes>, "note": "<a text for people>"}

* Without a `vision` entry (and with weights other than the published safetensors file) the folder has no image tower: the
  model works with text only. The same holds when the entry names a file that is not in the folder.
* An entry with another `format` or `version`, or a file that the checks above refuse, turns image input off and says why; the
  model works with text only.
* With `--vision on` the package stops with the reason instead of working with text only.
* Adding the file changes neither the model file nor its SHA-256. A package without image input does not read the entry; its
  `yasakani verify` checks the file through the `sha256` map.
