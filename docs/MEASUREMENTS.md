# Measurement conditions and records

日本語は [MEASUREMENTS.ja.md](MEASUREMENTS.ja.md)。

The conditions, the alternative tallies and the limits of the numbers in [README.md](../README.md). The content is based on the same
records as the section "技術情報と測定条件" of the CAMPFIRE project page. The launch settings, prompts, measurement scripts and the raw
records themselves are in [`bench/`](../bench/).

## Which builds were measured

* The speed tables are the saved records of the quality model (12.37 GB), the fastest model (10.31 GB) and the eight other engines.
  YASAKANI's thinking-off rows were measured on November 15, 2026 with the `yskn.dll` of 0.1.4 (sha256
  `bae77ae6e73369dee5d33490b1ff2093f88ab57c02efa06912db6246efffaa09`; 0.1.5, 0.1.6 and the 0.1.7 test build have the same GPU code) and its Python package; its thinking-on rows on November 8 with the
  `yskn.dll` built on November 7 (sha256 `0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb`, the engine of 0.1.0 to 0.1.2).
  The thinking-on table was not measured again with 0.1.4.
* The quality table evaluates the same two models with an FP16 key-value (KV) cache. That is not the KV condition of the speed tables, which
  use KV5. YASAKANI's two rows of that table were measured with a research build of the engine, which can use an FP16 KV cache and return
  the distribution over the whole vocabulary; the weights are the same data as in the released model files. The `yskn.dll` of this
  release was not used for the quality table. It was measured with the released `yskn.dll` afterwards (November 14, 2026): see
  "The released engine" at the end of the section on quality.
* "16.78 GiB" comes from the long-context test of the fastest model.
* The code-edit demonstration (the uncut screen recording) was made with the fastest model on November 8: 5.56 s from sending the input to the
  arrival of the last output.

## Conditions and tallies of the speed tables

One RTX 5090 at its stock 600 W, no overclocking. One request at a time on the same PC. The tasks are 12 published prompts (code, story,
translation and structured output, 3 each) and 3 additional tasks (code edit, JSONL edit, long JSONL). The output limit is 4,096 or 8,192
tokens.

Thinking off is greedy, 2 runs each. The table in the README is the average after runs with a repetition loop or an unrequested tool call are
left out by one saved rule shared by all engines. The exclusion can change the number of tasks in a category. For the fastest model's "Code",
the value is 1,089 tok/s before the exclusion and 512 tok/s after it (0.1.4; in the records of November 8, 900 and 529). We do not use the
former as a representative figure for practical code generation. The table before the exclusion is kept below.

On November 14, 2026 the rule for repetition was tightened and the table of the README was made again: for story and translation, a run is
also left out when lines that occur more than once make up 20% or more of its lines, for every engine alike. Only the "Story" column
changed (fastest model 285 -> 201, mkyt 204 -> 181, vLLM 150 -> 148, llama.cpp 139 -> 133). The table is the output of
[`tools/speed/tables_strict.py`](../tools/speed/tables_strict.py);
[`tools/speed/SPEED_TABLES_STRICT.md`](../tools/speed/SPEED_TABLES_STRICT.md) also lists the 12 prompts one by one. The first tally is kept in
[`bench/20261108/tables/`](../bench/20261108/tables/).

On November 15, 2026 YASAKANI's thinking-off rows were measured again with 0.1.4, with the same clients and rules (records in
[`bench/20261115/`](../bench/20261115/); the table is the output of [`tools/speed/tables_0114.py`](../tools/speed/tables_0114.py),
[`tools/speed/SPEED_TABLES_0114.md`](../tools/speed/SPEED_TABLES_0114.md)). The other engines' rows are the records of November 3 and 6, unchanged.
0.1.2 and 0.1.3 were measured in the same sitting; see the last section.

Thinking on uses temperature 1.0, top-p 0.95, top-k 20, min-p 0. It is a comparison of records: the fastest run of each task is chosen and
averaged per category. YASAKANI's value is chosen from 5 runs and the other engines' from 2, so the number of tries is not equal. The table
tallied by plain averages is also given below. Many runs ended at the output limit, so this is not a comparison of the time to reach an answer
or of accuracy.

Speed is (output tokens - 1) / (time from the first to the last content chunk). Reading the input is not included. Weights, quantization, KV
cache and speed-up settings are each engine's own; this is not a comparison with every setting made equal. For an engine measured with
several settings, the fastest setting of each category is used.

The engines compared are MegaCapybara 1.39 Tiny, mkyt v0.3.2, ignis v0.9.6, the November 2 build of NInfer, vLLM 0.31.0, SGLang 0.5.22,
llama.cpp b11700 and TensorFold 0.6.6. MegaCapybara with thinking off was measured on November 3; its thinking-on runs and the other engines on
November 6. This does not rank the latest version of every program, or every use.

## Thinking off: the table of the README (average without repetition loops etc., tok/s)

YASAKANI: 0.1.4, November 15, 2026.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 512 | 195 | 479 | 882 | 2,025 | 2,298 | 1,214 |
| YASAKANI quality model | 468 | 172 | 464 | 892 | 2,008 | 2,243 | 1,115 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 181 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 148 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 133 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

## Thinking on: the table of the README (fastest run of each task, tok/s)

YASAKANI's values are chosen from 5 runs, the others' from 2. YASAKANI: November 8, 2026, the engine of 0.1.0 to 0.1.2 (the build of November 7; not
measured again with 0.1.4).

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 337 | 287 | 497 | 536 | 667 | 1,108 | 704 |
| YASAKANI quality model | 299 | 242 | 428 | 480 | 743 | 1,053 | 634 |
| MegaCapybara 1.39 Tiny | 290 | 206 | 308 | 429 | 550 | 663 | 516 |
| mkyt v0.3.2 | 244 | 196 | 348 | 384 | 378 | 603 | 454 |
| ignis v0.9.6 | 211 | 130 | 276 | 277 | 278 | 485 | 327 |
| NInfer (November 2 build) | 172 | 140 | 258 | 263 | 263 | 404 | 322 |
| vLLM 0.31.0 | 131 | 112 | 197 | 195 | 187 | 369 | 253 |
| SGLang 0.5.22 | 157 | 124 | 206 | 230 | 257 | 402 | 282 |
| llama.cpp b11700 | 140 | 122 | 184 | 197 | 206 | 349 | 221 |
| TensorFold 0.6.6 | 134 | 79 | 167 | 183 | 439 | 360 | 268 |

## Alternative tally: thinking off, average before the exclusion (tok/s)

These values include the runs with repetition loops and unrequested tool calls. YASAKANI: 0.1.4, November 15, 2026.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 1,089 | 281 | 479 | 882 | 2,025 | 2,298 | 1,214 |
| YASAKANI quality model | 468 | 262 | 464 | 892 | 2,008 | 2,243 | 1,115 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 438 | 204 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 282 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 312 | 150 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 270 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 143 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

## Alternative tally: thinking on, average of all runs (tok/s)

Here too YASAKANI has 5 runs and the others 2 (YASAKANI: November 8, the engine of 0.1.0 to 0.1.2). Both tables are kept so that the table of records and the table of averages are not confused.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 324 | 273 | 467 | 471 | 627 | 1,073 | 657 |
| YASAKANI quality model | 282 | 234 | 407 | 432 | 626 | 1,013 | 568 |
| MegaCapybara 1.39 Tiny | 285 | 192 | 298 | 418 | 508 | 662 | 508 |
| mkyt v0.3.2 | 242 | 193 | 340 | 361 | 357 | 597 | 448 |
| ignis v0.9.6 | 208 | 128 | 268 | 273 | 270 | 472 | 316 |
| NInfer (November 2 build) | 167 | 136 | 245 | 252 | 254 | 396 | 280 |
| vLLM 0.31.0 | 128 | 109 | 192 | 189 | 168 | 365 | 246 |
| SGLang 0.5.22 | 153 | 122 | 202 | 217 | 250 | 400 | 274 |
| llama.cpp b11700 | 136 | 122 | 180 | 184 | 197 | 346 | 219 |
| TensorFold 0.6.6 | 127 | 77 | 152 | 175 | 320 | 356 | 232 |

## Quality on thinking text

Twenty thinking transcripts of our own, not used for the statistics with which the models were made, were used as windows of 2,048 tokens
each; the last 1,023 positions of every window (20,460 positions in total) were compared with the same reference, the original BF16 model. The
evaluation covers our own thinking text, and we do not generalize it to quality for every use or to the rate of correct answers on tasks.

| Engine / model | File size, GB | KV cache | KL on thinking text (smaller is closer) | Top-1 agreement (higher is closer) |
|---|---|---|---|---|
| YASAKANI quality model | 12.37 | FP16 | 0.0122 | 95.24% |
| NInfer, official NVFP4 | 23.72 | FP8 | 0.0178 | 94.67% |
| Unsloth UD-Q3_K_XL (llama.cpp) | 12.80* | FP16 | 0.0204 | 94.32% |
| llama.cpp, GSQ-RCO IQ3_S | 12.12 | FP16 | 0.0229 | 93.64% |
| SGLang, NVIDIA NVFP4 | 21.92 | FP8 (e4m3) | ≥0.0283 | 92.70% |
| YASAKANI fastest model | 10.31 | FP16 | 0.0304 | 93.01% |
| MegaCapybara, Tiny | 16.59 | 6/4 | 0.0321 | 92.80% |
| TensorFold, MLX 4-bit | 16.05 | BF16 | 0.0353 | 92.05% |
| mkyt, gdnfp4 | 19.50 | NVFP4 | ≥0.0362 | 92.17% |
| Unsloth UD-IQ3_XXS (llama.cpp) | 10.93 | FP16 | 0.0458 | 91.32% |
| ignis, NVFP4full-v2 | 19.41 | hq-e8-2b | 0.0473 | 91.09% |
| vLLM, NVFP4 LMHead4 | 18.77 | FP8 | ≥0.0509 | 90.22% |
| llama.cpp, GSQ-RCO IQ3_XXS | 10.44 | FP16 | 0.0518 | 91.22% |

KL is the divergence of the predicted distribution; top-1 agreement is how often the most likely next token is the same. The quality model's
KL is 0.0122198812 and that of the GSQ-RCO IQ3_S (llama.cpp) configuration is 0.02288, so the former is 46.59% smaller. The KV cache of YASAKANI
and of llama.cpp is FP16.

For SGLang, mkyt and vLLM the KL is that of a distribution made of the top 100 tokens (64 for mkyt) plus everything else as one bucket, which
is a lower bound of the KL over the whole vocabulary. SGLang's lower bound of 0.0283 alone cannot settle its order against the fastest model's
0.0304. We did not use the approximation gap observed on YASAKANI as an error bound for the other engines. Because the quality model's value
is smaller than every configuration's lower bound, the conclusion that it is the smallest of the thirteen listed configurations stands.

For MegaCapybara the distribution over the whole vocabulary was obtained at all 20,460 positions, giving KL 0.0321 and top-1 agreement 92.80%
against the same BF16 reference. In a check of the read-out, the top candidate of the distribution agreed with the server's actual reply at
all 20,289 positions where the two could be compared. That number of positions belongs to this check and is separate from the 20,460 positions
of the quality figures. The feature that lowers the probability of certain words during thinking (thinking penalty) was turned off so that the
model's own distribution was evaluated.

The KV format is as given in each row. YASAKANI is evaluated on its verification path in units of 16 rows; the other configurations mostly on
their prompt-processing path. SGLang's row in the quality table is NVIDIA NVFP4, not the LMHead4 configuration of the speed tables. Read the
table as a comparison of whole configurations (weights, KV cache and arithmetic), not as a ranking of weight-quantization methods alone.

Sizes are in GB (10^9 bytes) and are, as a rule, the whole model file that is loaded. Only UD-Q3_K_XL, marked *, shows the size of the main
model without its auxiliary layers. Some configurations include a vision part or auxiliary weights, so the sizes are not the compression
ratio of the main model alone, nor the VRAM needed. The GSQ-RCO files used here include about 0.35 GB of auxiliary layers; the main model
alone is 11.77 GB for IQ3_S and 10.09 GB for IQ3_XXS. As that main model is itself smaller than YASAKANI's, we do not describe this as a
compression comparison at strictly equal size. In an additional evaluation on general text, the new conversion of the quality model showed a
KL 3-10% larger than the earlier conversion in some results; this is stated together with the result on thinking text.

The two Unsloth configurations were also measured with llama.cpp and an FP16 KV cache. The 12.80 GB* of UD-Q3_K_XL is the main model without
auxiliary layers; the whole file is 13.15 GB. For UD-IQ3_XXS the whole file, 10.93 GB, is shown; the main model is 10.58 GB.

The score of every configuration, the logs, the scoring scripts and how the distribution of each engine was obtained are in
[`bench/20261108/quality/`](../bench/20261108/quality/), together with two more Unsloth files that are not in the table above. The twenty
transcripts are in [`tools/quality/`](../tools/quality/) (added on November 14, 2026), together with a program that makes the reference from
the original model and a program that measures a released model with the released engine. The reference file of this table itself (about
10 GB) is not published; a reference made with that program from the original model gives the same distributions (KL divergence between
the two: less than 0.000001).

### The released engine (added November 14, 2026)

The two YASAKANI rows above were measured with a research build and an FP16 KV cache. The released `yskn.dll` always rounds the KV
cache with KV5 and has no FP16 cache, so those rows cannot be repeated with the released engine; a measurement build of the source with an
FP16 cache repeats them (below). On the
same 20 windows and against the same reference, the engine of release 0.1.1 gives:

| Model | KV cache | KL on thinking text↓ | Top-1 agreement↑ |
|---|---|---|---|
| quality model | KV5 (the engine as the package starts it) | 0.0126 | 95.35% |
| quality model | KV5 values kept unpacked in 8-bit form (`YSKN_TB_CAP=0`) | 0.0125 | 95.23% |
| fastest model | KV5 (the engine as the package starts it) | 0.0309 | 92.98% |
| fastest model | KV5 values kept unpacked in 8-bit form (`YSKN_TB_CAP=0`) | 0.0307 | 93.06% |

In the rows with `YSKN_TB_CAP=0` the keys and values are still rounded by KV5; they are only kept unpacked, in 8-bit form, and read by
another attention kernel. They are not a measurement of a plain 8-bit cache, which the released engine cannot run. (Until 0.1.3 these
rows were labelled "8-bit, without the KV5 compression", which was not correct.)

Against a reference made from the original model with Hugging Face transformers (`tools/quality/make_reference.py`), the two KV5 rows
are 0.0126 / 95.35% and 0.0309 / 92.98%. How to repeat this: [`tools/quality/README.md`](../tools/quality/README.md).

**For comparison: a plain 8-bit and an FP16 cache (added November 15, 2026).** The released `yskn.dll` cannot run either of them. We
measured them with two measurement builds: `YSKN_KVX_MODE=2` (a plain rotated 8-bit cache) and `YSKN_KVX_MODE=0` (FP16). Since 0.1.4 the
published source builds both ([BUILDING.md](../BUILDING.md), "Measurement builds"; up to 0.1.3 they also needed a three-line change in
`src/engine35.cpp`). Builds of the 0.1.2 source with that change and builds of the 0.1.4 source gave the same values to every digit. The
builds are not released. Same 20 windows, same reference, a context of 2,048 tokens:

| Model | KV cache | Engine | KL on thinking text↓ | Top-1 agreement↑ |
|---|---|---|---|---|
| quality model | KV5 (default) | released `yskn.dll` (0.1.0 to 0.1.2) | 0.0126 | 95.35% |
| quality model | plain 8-bit | measurement build (`YSKN_KVX_MODE=2`) | 0.0122 | 95.31% |
| quality model | FP16 | measurement build (`YSKN_KVX_MODE=0`) | 0.0122 | 95.24% |
| fastest model | KV5 (default) | released `yskn.dll` (0.1.0 to 0.1.2) | 0.0309 | 92.98% |
| fastest model | plain 8-bit | measurement build (`YSKN_KVX_MODE=2`) | 0.0304 | 93.14% |
| fastest model | FP16 | measurement build (`YSKN_KVX_MODE=0`) | 0.0304 | 93.01% |

With KV5 the KL is about 3% higher than with the plain 8-bit cache for the quality model (+0.0004) and about 2% higher for the fastest
model (+0.0005); with both models KV5 was higher in 18 of the 20 windows. The plain 8-bit and the FP16 cache differ only within the noise of
this measurement. The FP16 build gives the values of the two YASAKANI rows of the table above, to every digit. Longer contexts were not
compared this way. The records of these runs are not in the repository. The KV5 rows were measured with the `yskn.dll` of 0.1.0 to
0.1.2; in our checks on the RTX 5090 the `yskn.dll` of 0.1.3 wrote the same tokens (see the 0.1.3 fix below).

## KV5 (the compressed key-value cache) with the fastest model

With the same 10.31 GB weights, four fixed sequences of 131,072 tokens (thinking and answers joined) were evaluated. The last 1,023 positions
at each context length, 4,092 positions in total, were compared with the BF16 reference. The input is fixed text that the engine reads; it is
not an independent corpus that was never used before.

The FP16 KV cache was measured with a research build and KV5 with a distribution build, so the difference includes the difference of
implementation paths. Over the 7 bands from 2K to 128K, the mean KL of KV5 minus that of FP16 was about 0.000034 to 0.000516. At 128K, FP16
was 0.02646 and KV5 0.02687; the mean difference at the same positions was +0.000412 (95% interval +0.000177 to +0.000674). We do not treat
a small difference as proof of equivalence.

With 256K reserved, the engine's memory is 27.12598 GiB with FP16 and 16.78320 GiB with KV5. Per token the KV cache takes 64.25 KiB against
21.875 KiB, about 66% less; for the whole engine it is about 38% less. These are the amounts reserved at start; they are not the peak during
generation, not the whole setup with the helper model, and not a measurement of quality at 256K. A plain 8-bit (INT8) KV
cache was measured only at a context of 2,048 tokens (see above), not at long contexts.

## Supported environment

This release covers text chat with Qwen3.8-27B on Windows 11. Every measurement on this page is from the RTX 5090. Since 0.1.3 the engine
is also built to run on other RTX 30, 40 and 50 series cards with 16 GB or more, but it is not optimized or tested on them: nothing on this page was
measured on them, and neither speed nor operation is promised there. On those cards the context is chosen from the free GPU memory with
a rule calculated from RTX 5090 measurements (`python/yasakani/gpu.py`). Image input is optional since 0.1.2 (a community contribution, [#4](https://github.com/yasakani-dev/yasakani/pull/4)); we have not measured image quality, and it is not part of the measurements on this page. Tool calling through the API is in the package since 0.1.1; it is not part of the measurements on this page.
Which configurations and context lengths really work on each 24 GB and 16 GB card will be checked on the actual card; operation is not
promised from the file size alone, and the conversation history and working memory are included in that check. The executables are not code-signed, so
Windows may show a warning.

## The uncut screen recording of November 8 (fastest model)

The same fastest model as in the comparison tables (10.311344128 GB) and the `yskn.dll` built on November 7 (the engine of 0.1.0 to 0.1.2) were used. One RTX 5090 at stock
600 W, maximum context 262,144, thinking off, temperature 0, output limit 8,192. The target input was sent once to a new process that had
been warmed up twice with a different short text. The prompt cache was empty.

In the 19:22 recording with a clock, the first output arrived after 1.281498 s, the last output after 5.555668 s, and the transfer completed
after 5.556058 s. The output was 4,712 tokens and the generation speed 1,102.202299 tok/s. Speed is (output tokens - 1) / (arrival of the
last output - arrival of the first). These are values measured while the screen was recorded, the clock was displayed and the CPU was
encoding; they do not replace the values of the speed tables. The effect of the recording load was not measured separately. The earlier
recording of 19:01 is also kept as an original record.

The generated file matched the expected file (apart from the final newline). It passed the syntax comparison and the behaviour comparison on
17,701 inputs, with 0 mismatches against the original. Python's own standard test suite was not run because it is not available in the
environment. The result of this one task is not treated as a general rate of correct code generation.

The video is a direct recording of one screen that shows the application connected to the API and a command prompt running as a separate
process. 1800x1000, 30 fps, no sound, 17.000 s, 510 frames. No cuts, no speed-up and no composited numbers were applied after recording, and
no dropped frames were detected. JST is the current time, ELAPSED the seconds running on from the start of the clock, REQUEST the seconds
from sending to completion. The clock log, the receive log and the hashes of the original recording, the output and the verification results
can be checked against each other. The clock is that of the same PC, and these records do not replace a third party's presence or
replication.

## Checks on this release (November 10, 2026)

Checked with the files that are published: the `yskn.dll` above, the Python package and the two model folders.

* `yasakani verify`: in both folders every file matches the checksums in `yasakani.json`.
* `yasakani selfcheck`: the reference (`selfcheck.json`) was written, then reproduced in a new process with the default context length and
  with 16,384 (both models).
* `yasakani bench` (256 tokens x 2 runs; explain / code / story / proof): fastest model 358 / 600 / 239 / 633 tok/s, quality model
  353 / 575 / 203 / 480 tok/s.
* The same request repeated (`repeat_probe.py` in `bench/`: one engine in one process, the same request generated 8 or 12 times, 256 tokens
  each, greedy and seed 1234, with the default settings and with speculative decoding switched off, one token per pass):
  * with nothing else using the GPU (utilisation 0% for 30 s before the start): **240 of 240 runs** gave the same tokens as the other runs
    of their group (fastest model; 48 runs with the default settings, 192 with speculative decoding off);
  * while other programs were using the GPU (the desktop was drawing; utilisation 3-5% before the start): with the default settings
    1 of 190 runs differed from the most common output of its group, with speculative decoding off 21 of 432 (both models; none of 16 further runs with only part of it switched off). A run that differs
    has the same tokens up to one position and continues differently from there; where the same position came up more than once, it was
    the same pair of tokens each time. While another process used the GPU without pause, it was 8 of 60 runs;
  * the three requests of `yasakani selfcheck` were identical in 180 of 180 repeats, and the command passed 12 of 12 times in new
    processes (both while other programs were using the GPU).
* Speculative decoding on against off (`onoff_eq.py` in `bench/`; 34 inputs, greedy and seed 1234, 62 cases, on / three kinds of off / on
  again; run while other programs were using the GPU): fastest model 171 of 174 comparisons identical, quality model 170 of 174. The
  differences: fastest model - one greedy case (first different token #143) and two sampled cases (#152, #97); quality model - one sampled
  case (#110) and one greedy case in which all three kinds of off differed from the first run at token #67. A repeated run with the same
  settings differed from the first run in one case per model (#115; #67). With speculative decoding off the engine needed at least 1.5 times as
  many passes in all 62 cases, so "off" really ran differently.
* What this meant for 0.1.0 to 0.1.2: when nothing else used the GPU, the same request gave the same tokens every time. When other
  programs used the GPU at the same time, a token now and then came out differently, more often in plain decoding (one token per pass)
  than with the default settings. It was not the hardware: the same computations repeated in PyTorch on the same GPU and compared bit for
  bit (about 540,000 repeats, including four minutes at the 600 W power limit with two processes at once) showed no difference. The cause
  and the fix in 0.1.3 are in the next section.

## The cause, and the fix in 0.1.3 (added for 0.1.3)

* **The cause** (found on November 11, 2026). `q35_attn_post_kernel` and `wk_attn_post_kernel` apply the position rotation to queries and
  keys. The rotation reads values of other threads from shared memory, and the next step (`kvx_rotate`) wrote into the same memory
  without waiting for those reads: one `__syncthreads()` was missing. When the timing of the threads shifted, for example because another
  program was using the GPU, a key could be computed from a value that had already been overwritten. The first layer that differed was
  always one of the 16 attention layers (177 of 177 cases that could be traced).
* **Passes compared bit for bit** (the same pass repeated, its result compared with a run alone; the shipping settings; another process
  kept the GPU busy): without the fix 6 of 3,200 differed (the released engine 5 of 2,400, an unchanged rebuild 1 of 800); with the fix
  0 of 2,800.
* **Repeated generation** (November 11 and 12, one engine, the desktop drawing): the engine of 0.1.0 to 0.1.2 differed in 10 of 593
  runs (11 at most: one test kept the result of only one of two repeats); builds with the fix 0 of 272 each. An unchanged rebuild of the same source also gave 0 of 272 in this test,
  so this test alone does not show the fix; the comparison of passes above does.
* With the fix, the text written alone was the same as with the released engine (8 runs of 256 tokens), and a one-row pass took about
  as long (8.01 ms against 7.98 ms).
* These tests used test builds of the engine with the fix (November 11 and 12, 2026), not the 0.1.3 release files. We do not promise
  bit-identical output in every condition.
* **The `yskn.dll` of 0.1.3** (tested on the RTX 5090 on November 15, 2026). 0.1.3 also gives every engine its own scratch buffer for
  the KV5 encoder; before, two engines in one process shared it. With two engines writing at the same time in one process (fastest
  model, greedy), the 0.1.3 engine gave the same text as one engine alone in 32 of 32 runs, the engine of 0.1.0 to 0.1.2 in 0 of 8.
  In 240 repeated runs of one engine (fastest model, 768 tokens each) neither engine differed this time, so that test shows no
  difference between them. With both models the 0.1.3 engine wrote the same tokens as the engine of 0.1.0 to 0.1.2 (`yasakani
  selfcheck`, 48 runs of the look-ahead check, 12 benchmark texts), and its speed was the same within our measurement noise.

## 0.1.4: wide copy passes and saved prompt states (added for 0.1.4)

Measured on November 15, 2026 on the RTX 5090 of the other measurements, with the same model files: 0.1.2, 0.1.3 and 0.1.4, each with its own
Python package and `yskn.dll`, `yasakani serve` with its default settings. One server at a time; before each step the GPU was checked to be idle.

### The answers stay the same

* **Passes.** On compute capability 12.x the passes of 17 to 64 rows have matrix kernels of their own in 0.1.4 (the same integer sums as
  the 16-row kernels, added in the same order). A single pass of 17 to 64 rows (12 sizes) gave the same chosen tokens, the same NLL bits
  and, after confirming, the same state as passes of 16 rows over the same tokens, with these kernels and with them switched off;
  confirming 0 to 63 tokens after a 64-row pass, the last part of a prompt read in 17, 40 or 64 rows, and passes of 32, 48 and 64 rows with
  a fixed sampling seed gave the same too: 90 of 90 checks with each model (contexts of 2K, 16K and 40K). The integer sums of the new
  kernels were the same as those of the 16-row kernels, bit for bit, for every kind of matrix of both models (27 matrices, 12 sizes, three
  kernels, two ways of splitting the work: 1,944 of 1,944 per model). A second check with other sizes and splits (7,200 of 7,200 per model)
  and with prompts read, partly confirmed and sampled in passes of 17 to 64 rows (25 of 25 and 21 of 21) found no difference either.
* **Generations.** 297 of 297 runs with copying gave the same tokens as decoding one token per pass (58 of them used passes wider than 16
  rows, up to 64). Fastest model: 21 tasks (the 12 published prompts, the 3 tasks of the tables, a longer JSONL edit and 5 tasks with
  thinking on), greedy and with a fixed seed, with the 0.1.4 settings twice, with the new kernels switched off, with 16-row copy passes,
  with the copy lookup of 0.1.3, and with an earlier build of 0.1.4 without the new kernels (252 runs). Quality model: 15 tasks, greedy, with
  the 0.1.4 settings, with the new kernels switched off and with the earlier build (45 runs).
* **The speed measurement below.** Its 32 answers per model were the same text with 0.1.3 and 0.1.4 (64 of 64), and 63 of 64 with 0.1.2:
  the second run of `scenario_code_python` of the quality model with 0.1.2 differs from character 1,348 on (its first run is the same text),
  the rare change part-way through that 0.1.3 fixed (see the 0.1.3 section).
* **Saved prompt states.** A prompt continued from a saved state gave the same output and the same state hash as the same prompt read from
  the start: 37 of 37 cases (documents of 5K, 51K and 229K tokens: the same input three times, a new question, the document in the system
  message, a continued conversation, an edited last message, an unrelated request in between), with the first build of 0.1.4; with the
  final build 34 of 34 cases of the 5K and 51K documents.
* `yasakani selfcheck` 3 of 3 with both models; tool calls through the server 8 of 8 with both models (streaming and not, thinking on and off).
* **The self-check on other GPUs** (new in 0.1.4). The package was shown a stand-in nvidia-smi that names an RTX 4090; the engine ran
  on the RTX 5090. Numbers: mean / largest NLL difference at the 195 positions of the reference output, and the share of the 285
  positions with the same most likely token. Passed: both models as released (exit code 0, no difference); the 8-bit key-value cache
  instead of KV5 (quality model 0.0056 / 0.081 / 99.6 %, fastest model 0.0051 / 0.060 / 99.3 %; the output was the same, exit code 0);
  a copy of the quality model with one of its 400 matrices scaled by 1.01, as a stand-in for a small difference in the computation
  (the output differed from token 29 on; 0.0048 / 0.077 / 99.3 %, exit code 3). Failed (exit code 1): the quality model's weights
  against the fastest model's reference and the other way round (0.030 / 0.53 / 96.8 % and 0.028 / 0.53 / 97.2 %), a copy whose 64
  `mlp.down_proj` matrices were scaled by 1.1 (the output was the same; 0.019 / 0.41 / 96.8 %) and a copy with one bit changed in
  every 64th byte of those matrices (0.028 / 0.91 / 96.8 %). As the RTX 5090 (the real nvidia-smi) the same runs gave exit code 0 when
  the output was the same and 1 when it differed, as before. The tolerances were chosen on the RTX 5090, and the comparison has not been
  run on another GPU. It is not sensitive to everything: one matrix scaled by 1.25, all 400 layer matrices scaled by 1.02, and one bit
  changed in every 4,096th byte of the 64 `mlp.down_proj` matrices passed it.
* **The self-consistency check of every GPU** (new in 0.1.4, also on the RTX 5090). The self-check then generates the answer of a prompt
  that copies its input (60 short JSON lines with two values changed) and reads prompt + answer again in passes of 16 rows: every generated
  token must be the most likely token there, and where wider passes are used, passes of 64 rows must give the same most likely tokens and
  NLL bits as passes of 16. With both models on the RTX 5090 (also with a stand-in nvidia-smi naming an RTX 4090, and an RTX 5080 for the
  fastest model): prompt 1,127 tokens, answer 1,080 tokens in 27 passes (15 of them wider than 16 rows, up to 64), every generated token
  the most likely one, and the same predictions in passes of 64 and 16 at all 2,206 positions; 3.5 s (fastest model) and 3.7 s (quality model).
* **Requests cut by the thinking budget.** 0.1.4 makes the cut depend only on the generated tokens. Fastest model: three requests with a
  thinking budget of 64, 200 and 600 tokens and one whose thinking ends before the budget (also compared with no budget), greedy and with a
  fixed seed, with the settings of 0.1.4 (twice), 16-row copying and one token per pass: the same answers in 8 of 8 groups (42 runs).

### Speed, thinking off

The clients and rules of the speed tables (2 runs per task, greedy, the stricter rule for repetition). The long JSONL edit is an extra task
(160 records returned with three fields changed, [`bench/20261115/prompts/cases_plus.json`](../bench/20261115/prompts/cases_plus.json)).
Order: fastest model 0.1.2, 0.1.3, 0.1.4, then quality model 0.1.4, 0.1.3, 0.1.2 (14:51 to 15:17), with the final `yskn.dll` of 0.1.4. tok/s:

| | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL | Long JSONL edit |
|---|---|---|---|---|---|---|---|---|
| fastest model, 0.1.2 | 515 | 195 | 499 | 911 | 1,210 | 1,125 | 1,218 | 1,141 |
| fastest model, 0.1.3 | 514 | 195 | 479 | 878 | 1,188 | 1,124 | 1,221 | 1,137 |
| fastest model, 0.1.4 | 512 | 195 | 479 | 882 | 2,025 | 2,298 | 1,214 | 2,099 |
| 0.1.4 / 0.1.2 | 0.99 | 1.00 | 0.96 | 0.97 | 1.67 | 2.04 | 1.00 | 1.84 |
| quality model, 0.1.2 | 464 | 175 | 478 | 902 | 1,128 | 1,146 | 1,155 | 1,096 |
| quality model, 0.1.3 | 468 | 172 | 465 | 893 | 1,089 | 1,103 | 1,117 | 1,059 |
| quality model, 0.1.4 | 468 | 172 | 464 | 892 | 2,008 | 2,243 | 1,115 | 2,056 |
| 0.1.4 / 0.1.2 | 1.01 | 0.98 | 0.97 | 0.99 | 1.78 | 1.96 | 0.97 | 1.88 |

* One by one, the 9 published prompts of the fastest model's table were 0.958 to 1.003 times as fast with 0.1.4 as with 0.1.2 (mean -2.4%);
  the 11 prompts of the quality model: 0.961 to 1.030 times (mean -1.3%).
* In this sitting 0.1.3 was also slower than 0.1.2 on the tasks without copying (0.96 to 1.01 times; translation and structured output most), and
  0.1.4 ran at 0.99 to 1.00 times the speed of 0.1.3 there. In the morning of the same day (03:42 to 04:07, the same files for 0.1.2 and 0.1.3, the
  first build of 0.1.4) the three versions were within 1% of each other on these tasks. We do not know the cause; the order of the
  versions was reversed for the quality model, and 0.1.2 was faster on translation and structured output in both orders.
* 0.1.4 checks copied text in passes of up to 64 rows with kernels of their own (TECHNOLOGY.md, 5.4). Measured with the first build of
  0.1.4, which had these passes without the new kernels (thinking off, 03:42 to 04:07 the same day): code edit 1.36 and 1.46 times, JSONL
  edit 1.55 and 1.56 times as fast as 0.1.2 (fastest and quality model).
* Measured with the first build of 0.1.4 only, not again: with a fixed sampling seed (temperature 1.0) one structured prompt of the fastest
  model (CSV) ran at 0.93 times the speed of 0.1.3 (both runs; as fast as 0.1.3 with the new copy lookup switched off; the cause in detail is
  not known). With thinking on, the first 1,536 tokens of the edit tasks (mostly thinking) ran at 0.99 to 1.02 times; the answer part was
  not measured.
* The code edit of the quality model differs from the expected file in one place with all three versions (as on November 8); the other edit
  answers are the expected files.
* Not measured: the thinking-on table, the speed of writing after a long prompt, GPUs other than the RTX 5090.

### Reading a prompt again

Through the server, fastest model, thinking off, greedy, 32 tokens per answer. The document is Python source text cut to the given number of
tokens. Seconds from sending the request to the first token of the answer (that token included); in brackets
`usage.prompt_tokens_details.cached_tokens`:

| Request after a 229,376-token document | 0.1.1 | 0.1.2 | 0.1.3 | 0.1.4 |
|---|---|---|---|---|
| first request | 115.69 (0) | 115.58 (0) | 115.83 (0) | 115.64 (0) |
| the same request again | 0.35 (229,398) | 0.36 (229,398) | 0.37 (229,398) | 0.35 (229,398) |
| the conversation continued | 0.38 (229,435) | 0.38 (229,435) | 0.38 (229,435) | 0.36 (229,435) |
| the same document, a different question | 115.63 (0) | 115.61 (0) | 115.73 (0) | **0.37 (229,376)** |
| with `tools`, first request | 116.15 (0) | 115.98 (0) | 116.15 (0) | 115.87 (0) |
| the tool result sent back | 0.38 (229,701) | 0.38 (229,701) | 0.38 (229,701) | 0.38 (229,701) |

With documents of 65,536 and 8,192 tokens it is the same: the first request about 20.0 s and 2.0 s; the same request, the continued
conversation and the tool result 0.04 to 0.15 s; a different question 20.0 s and 1.96 s with 0.1.1 to 0.1.3, 0.12 s and 0.09 s with 0.1.4.
The answers of a continued read and of a fresh read were the same. The GPU memory after loading did not change between the versions.
A different question reuses the document when the document comes before it, the question part is up to about 4,000 tokens and no other
request comes in between.

