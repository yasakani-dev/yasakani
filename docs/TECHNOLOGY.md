# How YASAKANI works — why it is fast

Technical notes by the author, written as a map for people who read the source code in this repository.
日本語: [TECHNOLOGY.ja.md](TECHNOLOGY.ja.md)

* Scope: versions 0.1.0 to 0.1.6. The engine library is the same in 0.1.0 to 0.1.2; 0.1.3 adds builds for the RTX 30 and 40 series and
  one missing synchronization (chapter 7); 0.1.4 adds kernels for passes of 17 to 64 rows on compute capability 12.x and changes host
  code (5.4 and chapter 8); 0.1.5 and 0.1.6 change host code only (fixes; the GPU code is that of 0.1.4). Windows 11, one NVIDIA GeForce RTX 5090, Qwen3.8-27B. Other GPUs: chapter 10.
* Unless noted otherwise, the numbers here were measured on 12 November 2026 on the author's computer with the public `yskn.dll` and the
  fastest model (context 16,384, temperature 0, 768 tokens each, four tasks). Comparisons with other engines, and their conditions, are in
  [MEASUREMENTS.md](MEASUREMENTS.md).
* "measured" marks a measured value; "calc." marks a value computed from file sizes or from measured values.
* Most of the methods described here stand on published research ([9. Research this builds on](#9-research-this-builds-on)). What is new is
  that the compression, the file format, the GPU computation and the look-ahead were **designed together** for one model, and tuned on one card.

## 1. Speed is set by bytes read and memory bandwidth

### 1.1 What sets the ceiling

To produce one token, all the weights of the model are read once and computed with. For a 27B model this reading is what takes the time.

* The original weights (BF16) are 53.8 GB.
* The memory bandwidth of the RTX 5090 (the most it can read per second) is 1,792 GB/s (GDDR7, 512 bits, 28 Gbps).
* 53.8 GB ÷ 1,792 GB/s = 30 ms. However fast the arithmetic, the original weights allow at most 33 tokens per second (calc.).

The ceiling is set by how fast memory can be read, not by how fast the arithmetic is.

### 1.2 The speed equation

```
speed (tok/s)    = tokens confirmed per pass / time of one pass
time of one pass = bytes read per pass / (memory bandwidth × share of it actually used)
```

A "pass" is one run through the whole 27B model (64 layers). YASAKANI works on the three numbers in these equations.

| | what is done | chapter |
|---|---|---|
| ① read less | the weights are stored as 3- to 4-bit codes; the cache for long contexts as 5-bit codes | [3](#3-weights-as-3--to-4-bit-codes), [6](#6-kv5-the-cache-for-long-contexts) |
| ② use the memory bandwidth | the codes are in a form the GPU decodes in a few instructions | [4](#4-computing-on-the-compressed-form) |
| ③ confirm more tokens per pass | drafts are checked together in one pass of the main model | [5](#5-look-ahead-speculative-decoding) |

### 1.3 Checking it with numbers

| | bytes read per pass (calc.) | time of one pass (measured) | read rate (calc.) | tokens confirmed (measured) | speed (measured) |
|---|---|---|---|---|---|
| no look-ahead (one token per pass) | about 10.1 GB | 7.9 ms | about 1,270 GB/s (71% of the ceiling) | 1.00 | **126 tok/s** |
| prose (an essay) | about 11.5 GB | 9.7 to 9.9 ms | about 1,170 GB/s (65%) | 2.71 | **279 tok/s** |
| with thinking (an explanation) | about 11.5 GB | 9.8 to 9.9 ms | about 1,170 GB/s (65%) | 2.91 | **298 tok/s** |
| code (a web page) | about 11.5 GB | about 10.2 ms | about 1,130 GB/s (63%) | 4.31 | **about 420 tok/s** |
| structured (CSV to JSON) | — | 11.9 to 12.1 ms | — | 15.04 | **1,264 tok/s** |

* What is read without look-ahead: 9.76 GB of codes (the figure the engine prints when it loads the model), about 0.15 GB of state of the
  state-carrying layers, and a little more. With look-ahead, the draft model (about 1.1 GB) and the re-advance of the state for the
  confirmed rows (about 0.3 GB) are added. With a long context the cache adds to it (chapter 6). The byte counts come from the contents of
  the files and are partly estimates. The bytes of the task with mostly 16 rows (CSV to JSON) were not counted.
* The "read rate" is bytes read divided by time. The time of one pass is accounted for by the bytes read and 60 to 70% of the memory bandwidth.
* The lower four rows are the shipping settings. **The text that is written is the same as in the first row**
  ([7. Does the output change?](#7-does-the-output-change)).
* The more predictable the content, the more tokens are confirmed. This is why the numbers in the speed table of the README differ so much
  from task to task.
* **Check it yourself.** `quickstart/lookahead_check.py` (item 2 of the quickstart menu) writes the four tasks of this table on your PC,
  without look-ahead and with the shipping settings, and prints the time of one pass, the tokens confirmed per pass, the speed, and
  whether the two texts are the same.

## 2. The shape of the model

Qwen3.8-27B has 64 layers. Sixteen are ordinary attention layers; forty-eight are "state-carrying" layers (linear attention, Gated DeltaNet).

* An attention layer keeps a key and a value (KV) for every past token. The longer the context, the more memory it uses and the more it
  reads. Only these 16 layers have a KV cache.
* A state-carrying layer keeps only a summary of the past (about 151 MB for one conversation), which does not grow with the context.
  In exchange, the state must be advanced token by token, in order. For look-ahead (chapter 5) this is the hardest part.

## 3. Weights as 3- to 4-bit codes

### 3.1 The file format (`.yskn`)

The original weights (BF16, 53.8 GB) are stored as three things:

* **codes**: 3 or 4 bits per weight,
* **small dictionaries**: tables that turn codes into values,
* **one scale per row** of each matrix.

The codes are **trellis codes**. Two weights make one step, and each step writes 6 bits (= 3 bits per weight) or 8 bits (= 4 bits per
weight). The reader keeps a 12-bit state (4,096 values); at each step the new bits enter the state, and the state indexes a table that
gives the two values. Because earlier codes decide how the next values can be chosen, the error is smaller than with rounding each weight
on its own at the same number of bits. Before rounding, the weights are rotated with a signed Hadamard transform over 512 elements, which
makes their distribution bell-shaped.

| | quality model | fastest model |
|---|---|---|
| the 400 matrices of the body | 288 matrices at 4 bits, 112 at 3 bits | all 400 at 3 bits |
| input embedding table | 3-bit codes | 3-bit codes |
| output matrix | 4 bits | 4 bits |
| file | 12.37 GB | 10.31 GB |

The code that reads the format is in `src/format.hpp` and `src/loader.cpp`.

### 3.2 How the codes are made (the conversion)

The conversion program is in [`tools/convert/`](../tools/convert/) (added on November 14, 2026), with the data it needs and what
we got when we ran it again. This is how it works.

1. **Collect statistics.** Calibration text is run through the original model, and two statistics are taken for every matrix:
   * input side: the second-order statistics of the values that enter the matrix;
   * output side: how much an error in the output of the matrix moves the prediction of the next token. It is built from gradients,
     with tokens drawn from the original model's own prediction as the targets.
2. **Align the coordinates.** The weights and the statistics go through the same Hadamard rotation and are divided by the size of each row.
3. **Choose the path through the trellis.** Viterbi (dynamic programming) picks the sequence of codes with the smallest error.
4. **Feed the error back.** The error made by rounding is taken up by the weights that have not been rounded yet. This is done in the
   direction of the inputs (columns) and in the direction of the outputs (rows).
5. On one RTX 5090 the 64 layers (the statistics and the rounding of the 400 matrices) took about 75 minutes with one bit width per
   matrix, as the quality model has them (November 7, 2026, with the research scripts). The program in `tools/convert/` rounds every
   matrix at both bit widths, from which both model files are assembled; that took 1 hour 55 minutes (November 14, 2026, see its README).

The calibration text is 96 windows of WikiText and 96 windows of **the model's own thinking transcripts**. Thinking text is included
because this model spends most of its output on thinking. With the output-side statistics and the thinking text together, and with the
same size and the same format, the KL on thinking text fell to 0.58 times that of our earlier conversion (quality model, 0.0210 to 0.0122).
On general web text it rose by 3 to 10%. This choice is why the quality numbers in the README are measured on thinking text.
The 20 thinking transcripts used for that measurement were not used for the statistics.

### 3.3 The helper model

The small model used for look-ahead (chapter 5) is built on a published draft model, converted to the same kind of 4-bit codes and
**trained again to match the compressed main model**. Its origin is given in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## 4. Computing on the compressed form

Most of the time of one pass through a 27B model is **the time to read the weights from memory**. So if the amount to read goes from
53.8 GB to about 10 GB, and the codes are used for the computation right where they are read, one pass becomes shorter. If the codes were
unpacked to the original size first, this gain would be lost.

| mechanism | what it does | source |
|---|---|---|
| codes laid out in reading order | at load time the codes are rearranged into the order in which the GPU threads read them; during a pass they are only read front to back | `src/lnord.cu` |
| table lookup to int8 | code → small table (8 KB) → int8 weight; the table sits in the shared memory of the GPU | `src/lnord.cu` |
| int8 matrix product | the inputs are made int8 too (one scale per 512 values, the same Hadamard rotation) and multiplied on the tensor cores; weights decoded once are reused for 8 rows of input | `src/a8_prologue.cuh`, `src/lnord.cu` |
| exact integer sums | products are added as integers into 64 bits; the result does not depend on the order or the split of the sum, so **computing many rows together gives the same bits as computing one row** | |
| shapes by row count | the work is split differently for 1 row, 2 rows and 3 to 8 rows; the inputs of 8 rows are placed next to each other in the order the weights are read | |
| one call per pass | one pass (several hundred kernels) is a CUDA graph and is launched with one call | `src/engine35.cpp` |
| early launch of kernels | the next kernel is started before the previous one ends and prepares its table and its first reads | `src/pdl.cuh` |
| merged post-processing | residual, normalisation and the int8 conversion for the next layer in one kernel | |

"The same bits however many rows" is the foundation of the look-ahead in the next chapter: if the result of checking several rows together
differed from computing them one by one, the text would change with look-ahead on and off.

### 4.1 Why the memory bandwidth can be used

Going to 3 bits is not enough. One pass decodes about 25.6 billion weights and multiplies them with the input. To finish in 7.9 ms that is
about 3 trillion weights per second (calc.). Only a few instructions per weight are affordable. If decoding costs more than that, the GPU
runs out of compute and memory bandwidth is left unused.

The YASAKANI format is built so that one weight is decoded in about 2 to 3 instructions (counted from the source by the author; when 8 rows are computed together).

* The codes are laid out in the order in which the GPU threads read them. One tile of codes (1.5 KB) is read with three 16-byte loads per thread.
* The table that turns codes into values is 8 KB and sits in the fast memory inside the GPU (shared memory). It holds signed values, so no
  sign handling is needed.
* The multiplication runs on the int8 tensor cores. Once a weight is decoded, its products with 8 rows of input take one instruction.

The code format and the GPU computation were designed together. A code that was only easy to decode was tried as well; its quality fell
below the bar, so it was not used.

### 4.2 More rows add almost no time

| rows checked in one pass | time of one pass (measured) |
|---|---|
| 1 row (no look-ahead) | 7.9 ms |
| 2 rows | 9.3 ms |
| 3 rows | 9.4 to 9.5 ms |
| 4 rows | 9.3 to 9.4 ms |
| 6 rows | 9.6 to 9.9 ms |
| 8 rows | 9.8 to 10.1 ms |
| a task with mostly 16 rows (CSV to JSON) | 11.9 to 12.1 ms |

From 2 rows on, the time includes making the drafts. The difference to one row (1.3 to 1.4 ms) is the draft model and the rows after the first.
More rows cost little more because the weights are decoded only once.

## 5. Look-ahead (speculative decoding)

### 5.1 The loop

1. Prepare candidates for what comes next (drafts).
2. The main model checks "the last token + the drafts" together in one pass (8 or 16 rows in all; since 0.1.4 up to 64 rows when the draft
   is copied from the input, see 5.4).
3. The tokens the main model chose are compared with the drafts from the front, and everything up to the first difference is confirmed.
   At the position of the difference the main model's token is taken. So one pass confirms at most 8 or 16 tokens (64 in a wide copy pass).
4. The state of the model is advanced to the confirmed position.

A draft is only a proposal; only tokens the main model agrees with are kept. The control code is in `src/specx.cpp`.

### 5.2 Where the drafts come from

| source | what it does | source file |
|---|---|---|
| draft model | takes intermediate values of five layers of the main model as input and produces 8 rows (or 16 rows) of continuation **in one shot** | `src/q35_bdraft.cu` |
| copy from the input | if the last few tokens occurred before (in the prompt or in the answer so far), what followed them is used as the draft; since 0.1.4 an index of 4-token sequences finds the place, and a cursor keeps copying from it while the answer follows it (5.4) | `src/specx.cpp`, `src/copy_index.hpp` |
| next-token table | remembers which tokens the main model chose before | |

In the tasks of the table above almost all confirmed tokens came from the draft model.

| setting | code | prose | CSV to JSON | with thinking |
|---|---|---|---|---|
| no look-ahead | 126 tok/s | 126 | 126 | 126 |
| copy and table only | 159 | 130 | 375 | 151 |
| draft model only | 422 | 274 | 1,252 | 298 |
| shipping settings | about 420 | 279 | 1,264 | 298 |
| without the switch to 16 rows | 423 | 276 | 807 | 294 |

### 5.3 Making drafts cheap and likely to be accepted

* **The draft model is stored as codes too.** The draft model (3.85 GB originally) is converted to the same kind of 4-bit codes (1.17 GB).
  About 1.1 GB of it is read per pass (calc.).
* **A light output head.** Scoring the whole vocabulary (about 248,000 tokens) in the draft model would be heavy, so a small approximation
  first narrows it down to 512 tokens and only those are scored exactly. Instead of reading the output matrix (0.64 GB) again, a small
  table (about 0.17 GB) is read (calc.).
* **Automatic switch between 8 and 16 rows.** When all 8 rows keep being accepted it goes to 16 rows, and comes back when that stops paying.
* **Drafts are accepted with sampling too.** When tokens are chosen with a temperature, the random number is derived from the seed, the
  position and the token. The main model and the draft draw the same random numbers, so drafts are accepted, and the chosen tokens are
  the same with and without look-ahead (`src/q35_sample.cu`, `src/smp_hash.cuh`).
* **Side candidates.** With sampling, the second and third candidates are placed next to low-confidence positions. If the main chain
  misses but a side candidate hits, one more token is confirmed (`src/lfrow_kernels.cuh`, `src/hcap_tables.inc`).
* **The state-carrying layers.** The checking pass is provisional. At confirmation the state is advanced again, but intermediate results
  of the checking pass are kept, and the re-advance runs on a second stream, overlapping with the next draft. The state (about 0.15 GB) is
  read once per checking pass; it is not copied for every candidate row.

### 5.4 Wide copy passes (0.1.4)

When an answer copies its input (an edit that returns most of a file, records returned with a few changes), most drafts are right, but up
to 0.1.3 one pass confirmed at most 16 tokens. 0.1.4 lets a copy go further:

* **Finding the place.** An index of the 4-token sequences of the prompt and of the answer so far points to earlier places with the same
  tokens; building it and one lookup take about 1 ms after a 229,000-token prompt. A cursor remembers where the copy is: while the answer
  follows the source, the next draft continues from there. When the answer leaves the source, the nearest match after the point where it
  left is preferred, so that look-alike lines (JSONL records, for example) are not mixed up (`src/copy_index.hpp`, `src/specx.cpp`).
* **Wider passes.** After two full 16-row copy passes in a row, the next copy pass checks 63, 47 or 31 tokens of the source at once: 64, 48
  or 32 rows. On compute capability 12.x these passes have matrix kernels of their own: the weights are read once for all rows, and each
  kernel adds the same integer sums as the 16-row kernels, so the results are the same bit for bit (`src/pfa.cu`, `src/lnord.cu`; the
  output layer of such a pass is one matrix product as well). A 64-row pass takes about 2.3 times as long as a 16-row pass (24.9 against
  11.0 ms with the fastest model and a 4K context on the RTX 5090; a 32-row pass 15.2 ms), so it pays only when most of it is accepted. If a run of wide passes took longer than 16-row passes would have, measured by time, the engine waits for twice as
  many fully accepted 16-row passes before it widens again (from 2 up to 16; current defaults, not tuned).
* **Confirming.** For passes of up to 16 rows the provisional results of the state-carrying layers are kept (5.3). A wider pass does not
  keep them; at confirmation the state is computed again from the inputs of the pass (`src/engine35.cpp`).
* **Where.** Wide passes are used on compute capability 12.x (RTX 50 series) only; other GPUs copy with passes of up to 16 rows. Only the
  RTX 5090 has been measured.
* **Per GPU (0.1.7, a test build).** What decides is the code the GPU runs (the PTX version of a probe kernel: sm_86, sm_89 or sm_120),
  not its compute capability alone. On sm_86 / sm_89 the wide passes are off unless asked for (`--wide on`, engine setting `YSKN_WIDE=1`);
  then no width is open until a check opens it. The check (`python/yasakani/widecheck.py`, once per GPU, driver, engine library, package
  and model) reads about 2,400 tokens in passes of 17, 31, 32, 33, 47, 48, 63 and 64 rows and again in passes of 16, and requires the
  same most likely ids and NLL bit for bit; then it times copy cycles (the pass, the commit of all rows, the drafter reading them) of 16
  and of W rows in turn at a 4K context, and opens a width W when a cycle takes at most 0.8 × W / 16 times a 16-row one. On 12.x every
  width stays open unless every round of the check is over that limit. The measured cycle times are also where the time-based back-off
  starts (they were the RTX 5090's on every GPU before). When the correctness check finds a difference, no width is used, passes of 17
  to 64 rows (such as the ends of prompt reads) go back to the 16-row sums also on 12.x, and `yasakani selfcheck` fails. The check reads
  its text in pieces of at most 1,023 tokens, so that the engine does not set up the work area of long prompt reads (about 580 MiB).
* **Time-based back-off (0.1.7).** During generation the copy loop (`src/specx.cpp`) adds up the measured time of each wide pass (at most
  twice its running mean, so that a stall of the OS counts little; before: the running mean) and checks every 4 wide passes, also inside
  a run of fully accepted ones (before: only when a run ended), whether the tokens of these passes per millisecond reach those of a fully
  accepted 16-row pass. A run that ends with fewer than 4 wide passes since the last check is judged together with the 4 before it, so a
  partly accepted pass at the end of a copy is not judged alone. If they do not reach it, the copy goes back to 16 rows at once and the
  number of fully accepted 16-row passes needed before the next wide pass doubles (from 2 up to 16); if they do, it is 2 again. The
  generated tokens do not depend on this (a wide pass gives the same choices as 16-row passes); the number of passes can. When the
  engine goes back in at least 10 of every 100 wide passes, counted over at least 128 of them in one process, the package marks the kept
  result of the check, and the next start checks again.

The answers do not change: only tokens the main model chose are kept, and a pass of 17 to 64 rows gives bit for bit the same choices and
values as passes of 16 rows (chapter 7). Thinking off, measured one after the other against 0.1.2: code edit 1.67 times (fastest model) and
1.78 times (quality model), JSONL edit 2.04 and 1.96 times, a longer JSONL edit 1.84 and 1.88 times. The other tasks ran at 0.96 to 1.01 times
(against 0.1.3 in the same sitting: 0.99 to 1.00 times; thinking off, greedy; [MEASUREMENTS.md](MEASUREMENTS.md)). The code edit of the fastest model needed 110 passes with 0.1.4 and 335 with the
copy lookup of 0.1.3 (one run each). The following were measured with the first build of 0.1.4, before the kernels of the wide passes, and
not again: with a fixed sampling seed one structured task (CSV) was about 7% slower than with 0.1.3 (both runs; as fast as 0.1.3 with the
new copy lookup switched off; the cause in detail is not known); with thinking on, the thinking copies little and is not faster (the first
1,536 tokens of the edit tasks: 0.99 to 1.02 times; the answer part was not measured); with passes of 16 rows only, the index and the
cursor alone made the edit tasks 1.07 to 1.18 times as fast (one run each).

## 6. KV5 (the cache for long contexts)

KV5 stores the keys and values of the attention layers at 5 bits per value. Per token it takes about a third of the memory of an FP16
cache and about two thirds of an 8-bit one, which leaves more memory for long contexts. With the fastest model and 262,144 tokens
reserved, the engine takes 16.78 GiB with KV5 and 27.13 GiB with an FP16 cache (a research build; [MEASUREMENTS.md](MEASUREMENTS.md)).

* **How it is made.** Keys, values and queries get the same Hadamard rotation (the result of attention does not change), and each row of
  256 values is rounded with a trellis code of the same kind as the weights (in chains of 64 values, with an integer-only Viterbi).
* **Rules it keeps.** Every row is encoded on its own. Writing one token at a time or 16 rows together gives the same bits. Rows have a
  fixed size. This is needed so that rows that were "checked" and rows that were "confirmed" by look-ahead never disagree.
* **Reading.** The rows are read at 5 bits and turned back into values inside the attention computation; no 8-bit copy is made.
* **What is packed.** The keys of all 16 layers and the values of 13 layers. The values of 3 layers stay at 8 bits.
* **Size.** Per token: about 64 KiB with FP16, about 33 KiB with 8 bits, about 22 KiB with KV5.
* **Quality and speed.** With the fastest model on fixed texts of 2K to 128K tokens, the KL with KV5 was at most 0.00052 higher than with
  FP16 on average (the FP16 side was a research build). Against a plain 8-bit cache (a measurement build of the 0.1.2 source, context
  2,048 tokens) the KL with KV5 was about 3% higher for the quality model and about 2% higher for the fastest model
  ([MEASUREMENTS.md](MEASUREMENTS.md)). The time of one pass was about the same as with the KV5-rounded values kept unpacked in 8-bit
  form: on the four tasks at a context of 16,384 tokens the difference was within 1% for prose, the thinking task and CSV to JSON, and
  KV5 took about 3% longer for code. Only CSV to JSON gave the same text with both settings (chapter 7). The released engine has not
  been compared with a plain 8-bit cache for speed.

Source: the rotation is in `src/kvx_codec.cuh`, the encoder in `src/kvq_fast.cuh` and `src/kvq_vit2.cuh`, the reader in `src/q35_atxpk.cu`,
`src/q35_atxpk3.cuh` and `src/q35_atxpk4.cuh`. The code table is `kvq_u5.bin` in the model folder.

### 6.1 A comparison at the same number of bits (research measurement)

The codes of KV5 follow the same idea as the weights (earlier codes decide how the next values are chosen). After the same rotation, they
were compared with the usual way of rounding to uniform steps. These values were measured in October 2026 with a research build (only the
values of the cache were re-rounded). What ships is the 5-bit KV5 only.

Quality drift (increase in perplexity against the FP16 cache):

| format | evaluation text (20 windows) | the model's own thinking text (16 windows) |
|---|---|---|
| 8 bit, uniform steps | +0.01% | −0.01% |
| **5 bit, YASAKANI codes (the method of KV5)** | +0.07% | +0.10% |
| 4 bit, YASAKANI codes | +0.15% | +0.17% |
| 4 bit, uniform steps | +0.52% | +0.54% |
| 3 bit, YASAKANI codes | +0.60% | +0.60% |

At the same 4 bits the drift is about 1/3.5. YASAKANI codes at 3 bits are about level with uniform steps at 4 bits.

In a long context (five places in one long text, 1,023 tokens each; increase in loss in nat, 0 = same as FP16; one point is good to
about ±0.001 to 0.002):

| position in the context | 4 bit, uniform steps | 4 bit, YASAKANI codes | 8 bit, uniform steps |
|---|---|---|---|
| 1K | +0.0040 | +0.0008 | −0.0030 |
| 8K | +0.0057 | −0.0006 | +0.0006 |
| 32K | +0.0055 | +0.0006 | −0.0008 |
| 65K | +0.0020 | −0.0005 | +0.0012 |
| 130K | +0.0072 | −0.0016 | +0.0015 |

With uniform 4-bit steps the drift persists out to 130,000 tokens. In this measurement the 4-bit YASAKANI codes stayed near 0 (within
the error) at every position.

## 7. Does the output change?

The mechanisms of chapters 4 to 6 are built to change only the speed. To check this, four tasks were generated with 28 settings, each with
one mechanism switched off. Comparing the first run of each setting, **all 112 texts were the same as with the shipping settings** (and the
same as with all look-ahead switched off).

Up to 0.1.2, while other programs used the GPU at the same time, a run occasionally turned into a different text part-way through. The
cause was found on 11 November 2026. `q35_attn_post_kernel` (`src/q35_kernels.cu`) and `wk_attn_post_kernel` (`src/lfrow_kernels.cuh`)
apply the position rotation to queries and keys; the rotation reads values of other threads from shared memory, and the next step
(`kvx_rotate`) wrote into the same memory without waiting for those reads. 0.1.3 adds the missing `__syncthreads()`. In our tests with
builds that have it, the change did not appear again ([MEASUREMENTS.md](MEASUREMENTS.md)). We still do not promise output that is
identical to the last character in every condition.

0.1.4 adds wide copy passes, kernels of their own for them on compute capability 12.x, and saved prompt states (5.4 and chapter 8). They
were checked against plain decoding on the RTX 5090. The new kernels add the same integer sums as the 16-row kernels, in the same order, and
their sums were the same bit for bit for every kind of matrix of both models (1,944 of 1,944 per model, and 7,200 of 7,200 in a second
check). A single pass of 17 to 64 rows gave the same chosen tokens, NLL bits and confirmed state as passes of 16 rows, also after
confirming 0 to 63 tokens of a 64-row pass, for the last part of a prompt and with a fixed sampling seed (90 of 90 checks per model;
contexts of 2K, 16K and 40K). In real generations 297 of 297 runs with copying (58 of them with wide passes) gave the same tokens as decoding
one token per pass (fastest model: 21 tasks, greedy and with a fixed seed, the settings of 0.1.4 twice, the new kernels switched off,
16-row copying, the copy lookup of 0.1.3, and an earlier build without the new kernels; quality model: 15 tasks, greedy). A prompt
continued from a saved state gave the same output and the same state hash as the same prompt read from the start (37 of 37 cases with the
first build, documents of 5K, 51K and 229K tokens; 34 of 34 with the final build). Wide passes after a long prompt also gave the same tokens
as decoding one token per pass with the final build (fastest model, a code edit; about 60,000 tokens, greedy and with a fixed seed, and
about 177,000 tokens, greedy: 3 of 3). The self-check of every GPU repeats a small version of this check (python/README.md).

0.1.4 also makes the point where `thinking_budget` or the loop guard cuts the thinking depend only on the generated tokens
(`python/yasakani/loopguard.py`). Up to 0.1.3 the cut could move by the tokens confirmed before it was noticed, so the answer could depend on
the look-ahead. Three requests cut by the thinking budget and one whose thinking ended before it gave the same answers with the settings of
0.1.4 (twice), with 16-row copying and with one token per pass (greedy and with a fixed seed).

KV5 rounds the cached values, so it changes the numbers. The released engine has no unrounded cache to switch to: with `YSKN_TB_CAP=0`
it keeps the KV5-rounded values unpacked in 8-bit form and reads them with another attention kernel, and that can give a different text too.

## 8. What each part contributes

The time of one pass with one part switched off (shipping: prose 9.70 to 9.90 ms, CSV to JSON 11.89 to 12.07 ms).

| switched off | prose | CSV to JSON | effect on speed |
|---|---|---|---|
| fast encoder of KV5 | 10.63 ms | 13.67 ms | 9 to 13% |
| early launch of kernels | 10.53 ms | 12.76 ms | 7 to 8% |
| light output head of the draft model | 10.08 ms | 12.33 ms | 3% |
| kernel shapes by row count | 9.98 ms | 12.07 ms | 2 to 3% |
| inputs placed in reading order | 9.93 ms | 12.09 ms | 1 to 3% |
| confirmation on a second stream | 9.90 ms | 12.37 ms | 1 to 3% |

When a long prompt is read (17 or more rows together), the weights of one matrix at a
time are expanded to int8 and reused (`src/q35_g6.cu`). A request that starts with the same text as the previous one reads only the new part.

**Saved prompt states (0.1.4).** The engine keeps up to 8 saved states along the prompt of one conversation in pinned host memory (187 MiB
each, about 1.5 GiB in all; no more GPU memory; 8, 8,192 and 4,096 are current defaults, not tuned): one after the last `<|im_start|>` of a prompt, and for prompts of 8,192 tokens or more also at multiples of 1,024
in the 4,096 tokens before it and at multiples of 65,536. All of them are prefixes of one sequence. A new request continues from the live
state if that is a prefix of it, otherwise from the latest saved state inside the part it shares with the previous prompt, otherwise it is
read from the start. The prompt is read in pieces that end at multiples of 1,024, so a request restored from a saved state is computed in
the same pieces as when it is read from the start. The same 229,376-token document with a new question is therefore read again only from
the last multiple of 1,024 before the question: 0.37 s to the first token instead of 115.6 s. A prompt is read from the start when an
unrelated request came in between (it replaces the sequence), when thinking is switched on or off, after a request with an image, and when
a prompt shorter than 8,192 tokens differs part-way. If pinned memory cannot be allocated, the engine goes on with the states it has. An older package that does not pass the number of states
gets one, which works as in 0.1.3. The
settings are the `YSKN_SNAP_*` entries of `DEFAULT_ENV` in `python/yasakani/runtime.py`; the code is in `src/engine35.cpp` and `src/specx.cpp`.

## 9. Research this builds on

* Evening out distributions with a Hadamard rotation: QuIP# (arXiv:2402.04396), QuaRot (arXiv:2404.00456).
* Redistributing rounding error with statistics: GPTQ (arXiv:2210.17323), LDLQ in QuIP (arXiv:2307.13304).
* Trellis codes for weights: QTIP (arXiv:2406.11235).
* Rounding with output-side statistics: YAQA (arXiv:2505.22988).
* Speculative decoding: arXiv:2211.17192, arXiv:2302.01318.
* The draft model: the published model named in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## 10. What this release does not have, and limits

* Tested on the RTX 5090, Windows 11 and Qwen3.8-27B only. The kernels are written for the shape of this model and tuned on this card.
  Since 0.1.3 they are also built for compute capability 8.6 and 8.9 (RTX 30 and 40 series); the RTX 50 series shares 12.0 with the
  RTX 5090. Two kinds of instructions need compute capability 9.0 or higher: the early launch of kernels (`griddepcontrol`, chapter 8) and
  a bulk prefetch into the L2 cache (off by default); on the RTX 30 and 40 series they are left out. Block sizes and the split of the work
  were chosen on the RTX 5090; on cards with fewer SMs and less memory bandwidth they are not tuned, and the speed there has not been
  measured. On cards with less than 32 GB the package chooses a shorter context ([python/README.md](../python/README.md#gpus-other-than-the-rtx-5090)).
* AMD GPUs and Macs are not supported: the kernels are CUDA.
* The converter ([`tools/convert/`](../tools/convert/)) knows the layers of Qwen3.8-27B only; another model needs changes to it and to
  the engine.
* Wide copy passes (5.4) are used by default on compute capability 12.x only, and were measured on the RTX 5090 only. The 0.1.7 test build
  lets RTX 30 / 40 cards use them after a check on the card (`--wide on`, 5.4); their machine code for 8.6 / 8.9 has not run on such a card.
  Their kernels have not run on another RTX 50 card; there, `yasakani selfcheck` compares passes of 64 rows with passes of 16 where wide passes are open, and fails when the check at start found that they compute differently ([python/README.md](../python/README.md#the-self-check)). Their answers were checked
  against single passes up to a context of 40,000 tokens, and in generations after a prompt of about 177,000 tokens. Their speed after a long
  prompt (more than about 40,000 tokens) has not been measured. A stop string is acted on after a pass, so up to 64 tokens (16 up to 0.1.3)
  can be generated past it; the text is cut at the stop string, but those tokens count in `usage.completion_tokens`.
* The saved prompt states take up to about 1.5 GiB of pinned host memory.
* One request is generated at a time.
* The numbers here come from one computer. They change with the task, the context length and what else is using the GPU.
