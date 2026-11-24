# YASAKANI

English · [日本語](README.ja.md)

<a name="what-it-is"></a>

YASAKANI is an engine that runs the Qwen3.8-27B language model on a Windows PC with one NVIDIA RTX GPU. The developer has tested it on the RTX 5090 only. After setup you can chat and edit code offline. You can also connect your own development tools through its OpenAI-compatible API. Since 0.3.1 it also runs Inkling-2 Mini by Thinking Machines Lab, as an experimental model (text only at first, still images since 0.3.5). Since 0.3.2 you can choose it in the start screen ([details](#inkling-2-mini)). 0.3.3 makes it faster on RTX 50 cards, most of all with long inputs (measured on the RTX 5090; [details](#inkling-2-mini-033)).

**[Download for Windows](https://github.com/yasakani-dev/yasakani/releases/download/v0.3.5/yasakani-quickstart-0.3.5.zip)** · [Compare eight engines](docs/MEASUREMENTS.md#thinking-off-the-table-of-the-readme-average-without-repetition-loops-etc-toks) · [Watch an uncut code edit](https://github.com/yasakani-dev/yasakani/releases/download/v0.3.1/fast_live_uncut.mp4)

<a name="two-models"></a>
<a name="what-you-need"></a>
<a name="limits"></a>

## Models and hardware

![One RTX 5090, one request at a time, thinking off. On our own JSONL edit task (return 60 JSONL lines with one record changed) the recommended quality model (12.37 GB) wrote 2,243 tok/s and the fastest model (10.31 GB) 2,298 tok/s. v0.1.4, November 15, 2026, mean of 2 runs, prompt reading excluded. Context up to 262,144 tokens.](docs/assets/hero-en.svg)

JSONL edit is our own task. The model gets 60 JSONL lines, changes one record and returns all the lines, so almost all of the answer is a copy of the input. tok/s means tokens generated per second. Thinking off means the model answers without first writing out its reasoning. Conditions: one RTX 5090, one request at a time, thinking off, v0.1.4 (November 15, 2026), mean of 2 runs, prompt reading excluded. Other GPUs are slower.

For everyday use, choose the **quality model**. Its predictions are close to the original model. Choose the **fastest model** if you want a smaller file and higher speed in most tasks. The **max-quality model** (15.72 GB) is the closest to the original model of the three. It needs a card with 24 GB or more and generates at about 0.8 to 0.9 times the speed of the quality model ([details](#max-quality-model)). [Quality model download](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) · [Fastest model download](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed) · [Max-quality model download](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-max-quality)

A fourth choice in the start screen, since 0.3.2: **Inkling-2 Mini**, another model (experimental, text and, since 0.3.5, still images, one request at a time, a card with 24 GB or more; [details](#inkling-2-mini)).

| Your GPU | Model choice | Windows testing |
|---|---|---|
| RTX 5090 (32 GB) | Quality, fastest or max-quality | Tested by the developer |
| RTX 3090, 3090 Ti, or 4090 (24 GB) | Quality, fastest or max-quality (max-quality with a shorter context; one user report, [#24](https://github.com/yasakani-dev/yasakani/issues/24)) | Built for these cards; not tested or tuned by the developer |
| Listed RTX 40 and 50 series cards (16 GB) | Fastest, with context sized to free memory | Built for these cards; not tested or tuned by the developer |

All speed numbers in this repository were measured on the RTX 5090. Other cards are slower. We do not guarantee their speed or that they work. The edit numbers are high because one pass of the model checks many tokens copied from the input (wide copy passes, up to 64 tokens per pass). Since 0.1.7 the engine uses them on RTX 30, 40 and 50 cards after a check on the card at the first start: a width is used only if it gives the same predictions as 16-token passes, bit for bit, and is faster on that card. Normal prose is not faster; the gain is on answers that copy the input, such as edits.

**Prompt reading (new in 0.2.0):** on the RTX 5090 the engine reads prompts faster, with the same answers ([What 0.2.0 changed](#what-020-changed)). By default cards with 30,000 MiB of memory or more use the new reading. Since 0.3.4 a card of 22,000 to 30,000 MiB uses it too when you give the context with `--max-ctx` and at least 1,800 MiB of GPU memory stays free beside it for the work area (on our RTX 5090 set up as a 24 GB card the start-up line said `prompt reading: new (setting auto: 24564 MiB card, 4202 MiB free beside --max-ctx 131072; --prefill old for the 0.1.7 one)`); this follows a user's measurement in [#24](https://github.com/yasakani-dev/yasakani/issues/24), and we have not run it on a real 24 GB card. Other cards read prompts exactly as 0.1.7 did. `--prefill new` tries the new reading on a smaller card. This is experimental. On the RTX 5090 set up as a 24 GB card it used the new reading. A user measured it on an RTX 3090 Ti (24 GB, Windows 11, 0.3.2, quality model, context given with `--max-ctx`): prompts of 8,224 and 32,797 tokens were read 1.39 and 1.40 times as fast, with the same self-check numbers; the work area fit up to a context of 131,072, not at 262,144 ([#24](https://github.com/yasakani-dev/yasakani/issues/24); the reporter's numbers). We have not run it on a real card under 30,000 MiB ([smaller cards](docs/FAQ.md#prompt-reading-smaller-cards)).

**Wide copy passes on other cards:** users reported the 0.1.7 test build and later versions in [#24](https://github.com/yasakani-dev/yasakani/issues/24). On an RTX 4090 (Windows 10) the check opened all three widths, and the three edit tasks ran 1.85 to 1.99 times as fast as with wide passes off (770.0 → 1,420.9, 822.5 → 1,634.9 and 793.2 → 1,545.6 tok/s), with the same output. On an RTX 5060 Ti only the 64-token width opened (215.1 → 279.4, 226.9 → 300.4 and 218.8 → 280.6 tok/s), with the same output. On an RTX 3090 (Linux, 0.1.7 built from source) and an RTX 3090 Ti (Windows 11, 0.3.2) the check opened all three widths, and the three edit tasks ran 1.52 to 1.61 times as fast, with the same output. These are the reporters' numbers. We have run the code for RTX 30 / 40 only on the RTX 5090. [How the check works](docs/FAQ.md#wide-copy-passes-per-gpu)

Contributors have tested a Linux build from source. AMD and Mac are future targets. [GPU list and requirements](docs/FAQ.md#which-gpus-which-systems) · [Build on Linux](BUILDING.md#linux-building-from-source)

**Several GPUs:** one process uses one GPU. `--device N` picks GPU N as `yasakani info` and nvidia-smi number them. With 0.1.5, `--device 1` on a PC with two GPUs could stop with `graph instantiate failed` ([#20](https://github.com/yasakani-dev/yasakani/issues/20)). 0.1.6 includes a fix. On November 18, 2026 the reporter of [#20](https://github.com/yasakani-dev/yasakani/issues/20) confirmed that it works on a PC with two GPUs (RTX A2000 and RTX 5060 Ti). We have not tried two GPUs ourselves.

## Quick start

You need 64-bit Windows 11, Python 3.10 or later, NVIDIA driver 580 or later, the Visual C++ 2015–2022 x64 runtime, and 20 GB of free disk space.

1. Extract the v0.3.5 Windows quickstart ZIP to a path with only English letters and digits, such as `C:\yasakani`. [Release files and SHA-256 checksums](https://github.com/yasakani-dev/yasakani/releases/tag/v0.3.5)
2. Open **`START.cmd`** and choose a model (1 to 3: the Qwen3.8-27B models; 4: Inkling-2 Mini, experimental). On the first run it downloads the engine and model files, checks them, and runs a GPU self-check.
3. A menu comes last. Press Enter and the page opens in your browser. It shows the model loading; when that is done, type your first prompt. The page has four tabs: chat, a speed test (a rough figure measured in the browser; `yasakani bench` is the standard measurement), the API address for other apps, and model and status (GPU memory, and a switch between models when two or more are downloaded). It also has a quit button: it stops the model, and the black window then closes by itself.
4. From the second run on (once the model has passed the checks), `START.cmd` opens this page directly, with no model question and no menu. For the menu (measure the speed, compare look-ahead on and off, chat in the terminal, start the server), open a command prompt in that folder and type `START.cmd --menu`.

To use YASAKANI from another app, set the app's API address to the one on the API tab of the page, normally `http://127.0.0.1:8765/v1`. The menu item “start the server” gives the same API without the page.

In terminal chat, type `/effort medium` to make the model think for a shorter time, or `/think off` to turn thinking off. [Manual installation](python/README.md#install) · [API guide](python/README.md#the-server)

**Updates.** Since 0.1.7, the quickstart can update itself (from v0.1.6 or older, download the 0.3.5 quickstart ZIP by hand once). At every start, before it shows the menu or before it opens the browser page directly, it asks GitHub whether there is a newer release and says so first. If there is one, it shows the old and new version. Before the menu it then asks, and it changes nothing without a yes. The browser page shows the same offer with an **Update** button. When the page opens directly, nothing is asked in the black window, and nothing changes until you press that button. You can also choose “check for a new version” in the menu, run `START.cmd --action update`, or type `yasakani update`. Since 0.3.5 the automatic check runs at every start, at most once an hour (0.3.4 and older: at most once a day). A start within an hour of the last check does not ask again (within a day on 0.3.4 or older), and a quickstart that keeps running does not check by itself, so right after a release the button can be missing. To ask at once: since 0.3.4 the Model tab of the page has **Check for updates**, which asks when you press it; on any version, run `START.cmd --action update` in the quickstart folder. The models stay; the engine and the Python package are installed again. `START.cmd --action rollback` goes back to the version before. To turn the automatic check off: `START.cmd --no-update-check`. [What the check sends, and how the files are checked](docs/FAQ.md#how-does-the-update-work)

<a name="speed"></a>

## Speed and quality

The fastest model was faster than each of the eight other engines in all seven categories. All engines ran on the same PC with one stock RTX 5090 (600 W) and generated one request at a time. With thinking off we compare means that leave out repetition loops and similar failures. With thinking on we compare the best run per task. The big gaps are in our own three tasks. Two are edits whose answers mostly copy the input, and one is a long, very regular JSONL output. On these, with thinking off, the fastest model is 1.8 to 4.6 times as fast as the next engine (code edit: 2,025 tok/s with v0.1.4 on November 15, against 441 tok/s for MegaCapybara 1.39 Tiny on November 3). On the 12 published prompts it is 1.1 to 1.5 times as fast as the next engine. Three of those 12 (CUDA code, TypeScript code, the English mystery story) are left out of the comparison. On them the fastest model's answer fell into a repetition loop or made a tool call that was not asked for.

<a name="what-020-changed"></a>

**What 0.2.0 changed** (RTX 5090, one request at a time, against the released 0.1.7):

* **Reading a prompt is faster.** The quality model read a prompt of 8,192 tokens at 5,153 input tokens/s (0.1.7: 4,248, 1.21×) and one of 32,768 tokens at 4,647 (0.1.7: 3,774, 1.23×). The fastest model: 5,102 (4,259) and 4,601 (3,774). This is the first read, with nothing reused; thinking off, median of 2 runs per version, context 69,632. With thinking on, the quality model was 1.21× and 1.25× as fast at these two lengths. [Details](docs/FAQ.md#prompt-reading-020)
* **A known long prompt changed in the middle is read again from the last saved state before the change.** 0.1.7 read it from the start. A prompt of 32,768 tokens changed at 25, 50 or 75 % of its length was read again 1.51, 2.05 and 4.21 times as fast with the quality model (fastest model: 1.49, 2.04 and 4.21 times; same conditions).
* **The answers are the same.** In our checks the quality and fastest models gave the same tokens as 0.1.7. Generation itself is not faster: on the 15 tasks of the speed tables (thinking off, sum of all tasks, 3 runs per version, edit copy on as by default) the quality model was +0.37 % and the fastest model +0.45 % against 0.1.7. The edit copy does not speed up these tasks beyond noise.
* **It needs more GPU memory.** The engine keeps a work area from the first long prompt on: at the peak 1,385 to 1,388 MiB more than 0.1.7 with the quality model, 1,476 to 1,479 MiB with the fastest model. When the GPU does not have that room, the engine prints `[q35_layer] capacity fallback; no context reduction` once and reads prompts pass by pass without the work area (the other changes of 0.2.0 stay on; same answers). By default (`--prefill auto`) the new reading is used only on cards with 30,000 MiB or more; smaller cards read prompts exactly as 0.1.7 did.
* **Edit copy (new, on by default).** When an edit makes the same short replacement again and again, such as a key or a column renamed in every line, the look-ahead puts that replacement into its drafts once the model has written it at two places. The model still checks every drafted token, so the answer does not change. Renaming a key in every line of a JSONL file (80 lines), quality model: 1,151.4 → 2,039.8 tok/s (1.77×); renaming a column in every view of an SQL file (50 views): 1,168.5 → 2,101.0 tok/s (1.80×). Fastest model: 1.61× and 1.60×. Context 32,768, temperature 1.0, thinking off, the first 1,024 tokens of the answer, mean of 2 runs with the edit copy off and 2 with it on, measured in turn. These two tasks are not part of `yasakani bench --edit`, and their prompts are not in this release. The edit copy works where one request is generated at a time; the shared passes of several requests in `serve --parallel N` do not use it. [Details](docs/FAQ.md#edit-copy-020)

Both are settings of `chat`, `serve`, `bench` and `selfcheck`: `--prefill auto|new|old` (or `YASAKANI_PREFILL`) and `--edit-copy on|off` (or `YASAKANI_EDIT_COPY`). One line at start says which prompt reading is used and why. [The options](python/README.md#prompt-reading-and-edit-copy) · [Measurements](docs/MEASUREMENTS.md#measurements-020)

**New model in 0.2.0: the max-quality model.** The same Qwen3.8-27B with one bit per weight more than the quality model in each part (15.72 GB). On 20 thinking texts it is closer to the original model than the quality model (KL 0.0040 against 0.0126, default KV5, RTX 5090), and it generates at about 0.8 to 0.9 times its speed. It needs a card with 24 GB or more and is tested on the RTX 5090 only. In the quickstart it is choice 3 of the model question, which comes on the first run and with `START.cmd --menu`; `START.cmd --model max` picks it without the question. [Details](#max-quality-model)

<a name="what-014-changed"></a>

**What 0.1.4 changed** (thinking off, 2 runs per task, measured one after the other against 0.1.2 on November 15, 2026; RTX 5090; tok/s):

| | Code edit | JSONL edit | Long JSONL edit (160 records) | Other tasks |
|---|---|---|---|---|
| fastest model | 1,210 → 2,025 (1.67×) | 1,125 → 2,298 (2.04×) | 1,141 → 2,099 (1.84×) | 0.96 to 1.00× |
| quality model | 1,128 → 2,008 (1.78×) | 1,146 → 2,243 (1.96×) | 1,096 → 2,056 (1.88×) | 0.97 to 1.01× |

Both versions gave the same answer text, with one exception: in one 0.1.2 run the text changed part-way through. That was the rare problem fixed in 0.1.3. 0.1.5 and 0.1.6 have the same GPU code as 0.1.4. 0.1.7 adds GPU code for [several requests at once](#several-requests-at-once). [Details](docs/MEASUREMENTS.md#speed-thinking-off)

To measure the speed on your own PC, run `yasakani bench`. To run the three edit tasks above (the code edit, the JSONL edit and the long JSONL edit), run `yasakani bench --edit`. It prints the speed, the tokens per pass (how many tokens one pass of the model produced) and whether each answer is the expected one. [How to measure the edit speed](docs/FAQ.md#how-do-i-measure-the-edit-speed-on-my-pc)

<details>
<summary>See all seven categories with thinking off and on</summary>

**Thinking off:** the mean of two runs per task. Runs with a repetition loop or an unrequested tool call are left out, by the same rule for all engines. YASAKANI was measured again on November 15 with v0.1.4 (0.1.5 and 0.1.6 have the same GPU code). Numbers are generation tok/s, without prompt processing.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI fastest model** | **512** | **195** | **479** | 882 | **2,025** | **2,298** | **1,214** |
| **YASAKANI quality model** | 468 | 172 | 464 | **892** | 2,008 | 2,243 | 1,115 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 181 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 148 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 133 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

**Thinking on:** the fastest run of each task, averaged within each category. YASAKANI was measured on November 8 with the engine of 0.1.0 to 0.1.2 (the build of November 7), with five runs per task. It was not measured again with 0.1.4. The other engines had two runs per task.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI fastest model** | **337** | **287** | **497** | **536** | 667 | **1,108** | **704** |
| **YASAKANI quality model** | 299 | 242 | 428 | 480 | **743** | 1,053 | 634 |
| MegaCapybara 1.39 Tiny | 290 | 206 | 308 | 429 | 550 | 663 | 516 |
| mkyt v0.3.2 | 244 | 196 | 348 | 384 | 378 | 603 | 454 |
| ignis v0.9.6 | 211 | 130 | 276 | 277 | 278 | 485 | 327 |
| NInfer (November 2 build) | 172 | 140 | 258 | 263 | 263 | 404 | 322 |
| vLLM 0.31.0 | 131 | 112 | 197 | 195 | 187 | 369 | 253 |
| SGLang 0.5.22 | 157 | 124 | 206 | 230 | 257 | 402 | 282 |
| llama.cpp b11700 | 140 | 122 | 184 | 197 | 206 | 349 | 221 |
| TensorFold 0.6.6 | 134 | 79 | 167 | 183 | 439 | 360 | 268 |

The other engines were measured on November 3 and 6. Each engine used its own weights, quantization, KV cache and acceleration settings. [Measurement conditions and all-run averages](docs/MEASUREMENTS.md) · [Records and reproduction scripts](bench/README.md)

</details>

<a name="one-edit-start-to-finish"></a>

In the uncut November 8 recording, the fastest model edited a Python file and returned the whole file **5.56 seconds** after the request was sent. On 17,701 test inputs, the edited code gave the same results as the original program. [Recording and verification records](bench/20261108/demo/)

<a name="quality"></a>

The quality model picks the same most likely next token as the original BF16 model at **95.24%** of positions, and the fastest model at **93.01%**. The test used 20 thinking texts that were not used in the conversion (20,460 positions). KL divergence measures how far the predicted probabilities are from the original model's; lower is closer. It was 0.0122 for the quality model and 0.0304 for the fastest model. The similarly sized GSQ-RCO IQ3_S agreed at 93.64%. These numbers are from November 8, with a research build and an FP16 KV cache. Measured again on November 14 with the released engine (0.1.0 to 0.1.2) and its KV5 cache, the results were 95.35% (KL 0.0126) and 92.98% (KL 0.0309). This measures agreement with the original model on thinking text, not task accuracy. The models are not identical to the original. [Quality results for 13 configurations and methodology](docs/MEASUREMENTS.md#quality-on-thinking-text) · [Measure it yourself](tools/quality/)

<a name="max-quality-model"></a>

**Max-quality model.** The same Qwen3.8-27B, converted with more bits: 5 and 4 bits per weight in the body, 5 bits in the output matrix, 4 bits in the embedding table (15.72 GB; one bit per weight more than the quality model in each part). Measured the same way with the default KV5 cache (RTX 5090, November 20, 2026), it picks the same most likely next token as the original model at **97.40%** of positions, with KL 0.0040. The quality model, measured the same way: 95.35% and 0.0126. Against the quality model on the same PC, reading a prompt and the three edit tasks ran at 0.97 to 1.00 times its speed. The four `yasakani bench` tasks ran at 0.92 times with thinking off (403.4 against 439.3 tok/s) and 0.81 times with thinking on (317.9 against 392.4 tok/s). It is tested on the RTX 5090 only. On a 24 GB card the package would choose a context of 163,840 tokens when the desktop uses 500 MiB (196,608 with nothing else on the GPU): that is a calculation, not checked by us on such a card. A user ran it on an RTX 3090 Ti (Windows 11, 0.3.2) with the desktop using about 1.1 GB: the package chose 147,456 tokens, the wide-pass check was bit for bit at 463 positions, the self-check passed, and it generated at 0.95 times the speed of the quality model ([#24](https://github.com/yasakani-dev/yasakani/issues/24); the reporter's numbers). If the start-up line of the wide copy passes shows little free memory (that report: 153 MiB with the max-quality model on a 24 GB card), a `--max-ctx` one step shorter (16,384 tokens) leaves about 350 MiB more. It does not fit on 16 GB cards. It needs YASAKANI 0.2.0 or newer; earlier versions stop with an error when they open its 5-bit codes. [Measurements](docs/MEASUREMENTS.md#max-quality-model)

<a name="long-contexts"></a>

With room reserved for the maximum context of 262,144 tokens, GPU memory use was about 21.2 GiB with the quality model and 19.1 GiB with the fastest model. This includes the helper model. The max-quality model used 25,977 MiB after loading with the 0.2.0 package (quality model: 22,795 MiB, same run). The work area of the new prompt reading of 0.2.0 comes on top ([above](#what-020-changed)). We measured long-context quality up to 128K tokens. [KV5 measurements](docs/MEASUREMENTS.md#kv5-the-compressed-key-value-cache-with-the-fastest-model)

<a name="several-requests-at-once"></a>

## Several requests at once (experimental)

New in 0.1.7. The server can generate several requests at the same time on one GPU. Each request gets its own context, and one pass of the model works on all running requests together.

```bat
yasakani serve --model-dir D:\models\qwen38-quality --parallel 4
```

The default is `--parallel 1`: one request at a time, as before. We checked this with a development build that has the same GPU code as this release (RTX 5090, both models, thinking off): `--parallel 1` gave the same tokens as 0.1.5 on 35 prompts per model, and its speed on 15 tasks was within 1% of 0.1.5.

With `--parallel N`, each answer has the same tokens as when that conversation is sent alone to a new server with the same settings. This holds when the request has its own `seed` and its prompt is read the same way (`timings.snap_mode`).

Limits:

* Experimental. Tested on the RTX 5090 (32 GB) only.
* Edits get slower. On the code edit and the JSONL edit, 2 to 16 requests at once were 0.42 to 0.59 times as fast as the same requests sent one by one. The 12 published prompts were 1.1 to 6.6 times as fast with 16 at once (fastest model, thinking off).
* While a prompt is read, or a request with images runs, all other requests pause. Reading 16,533 tokens takes about 4.1 seconds.
* Contexts get shorter. With the quality model and N = 4, each request had 114,688 tokens on the RTX 5090.
* The new prompt reading of 0.2.0: with a given `--max-ctx` that leaves room (`--parallel 4 --max-ctx 16384` on the RTX 5090) the engine used it. When `--parallel N` chooses the contexts itself, they take the free GPU memory, and the engine may find no room and read prompts without the work area. We have not measured this.
* Without a `seed`, a request's result depends on the order in which requests arrive.

[Questions and measurements](docs/FAQ.md#several-requests-at-once-experimental)

<a name="inkling-2-mini"></a>

## Inkling-2 Mini (experimental, new in 0.3.1)

Inkling-2 Mini is a language model that Thinking Machines Lab published on November 20, 2026 ([thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)). It has 26.4 billion weights in a mixture of experts: for every token each layer picks 8 of its 128 experts. It is published under Apache-2.0, and its model card also names the [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/), which applies to the weights and to modified versions. YASAKANI runs it with a second engine library in the engine zip; the Qwen3.8-27B models run on `yskn.dll` as before (with the two new options of #26 and #27, off by default).

<a name="inkling-2-mini-033"></a>

**The fast path (new in 0.3.3).** 0.3.3 adds a faster way to compute this model: an 8-bit key/value cache (MXFP8, an open format) for the 8 layer visits that attend to the whole context, attention that splits a long context into parts, a new attention for prompt reading, and, on RTX 50 cards, prompt passes of the experts on the 4-bit (NVFP4) tensor cores. The default is `--fast auto`. Since 0.3.4 it turns the fast path on for RTX 30, 40 and 50 cards (compute capability 8.6, 8.9 and 12.0; in 0.3.3 only for RTX 50 cards), and on RTX 30 and 40 cards the fast path runs without the RTX 50-only 4-bit part. We changed this after a user's measurement of 0.3.3 on an RTX 3090 Ti (below; [#29](https://github.com/yasakani-dev/yasakani/issues/29)). The output of the fast path is close to, but not the same as, that of the standard path (below), so on RTX 30 and 40 cards the default output changes with 0.3.4. `--fast off` (or `YASAKANI_INK2_FAST=off`) turns the fast path off on any card and computes as before, and `--fast on` turns it on for other GPUs too. With the fast path off, the engine computes as 0.3.2 did. On the RTX 5090, with the fast path off the engine gave the same logits and tokens as 0.3.2, bit for bit; in one comparison in the same GPU job it was 1 to 2 % slower (after 4,096 tokens 160.4 against 163.7 tokens/s without the drafter, 257.8 against 261.0 with it). We have not measured this on other cards.

Speed, 0.3.2 and 0.3.3 with the fast path: one RTX 5090, Windows 11, November 23, 2026, the `--scheme quality` folder, one request at a time, greedy decoding. The text is 53 public source files, without repetition. Prompts were read from an empty context (median of 2 reads; 131,072 tokens: one read). Generation is 128 tokens with `--draft off` and 256 tokens with `--draft auto`, right after an input of the given length (median of 3 runs in each of 2 rounds, 0.3.2 and 0.3.3 side by side in one GPU job; at 131,072 tokens 1 round).

| | 0.3.2 | 0.3.3, fast path |
|---|---|---|
| Prompt reading, 4,096 tokens | 900 tokens/s | 1,826 tokens/s |
| Prompt reading, 32,768 tokens | 633 tokens/s | 1,755 tokens/s |
| Prompt reading, 131,072 tokens | 314 tokens/s (418 s) | 1,377 tokens/s (95 s) |
| Generation after 256 / 4,096 / 32,768 / 131,072 tokens, `--draft auto` | 230 / 257 / 88 / 58 tokens/s | 268 / 351 / 183 / 132 tokens/s |
| Generation after 256 / 4,096 / 32,768 / 131,072 tokens, `--draft off` | 202 / 162 / 93 / 38 tokens/s | 222 / 209 / 169 / 109 tokens/s |
| GPU memory in use, context 33,024 / 131,072 tokens, with the drafter | 19.9 / 28.5 GiB | 18.5 / 20.9 GiB |

The default context is 32,768 tokens; for longer inputs set a longer context with `--max-ctx` (the 131,072-token numbers need a context of at least that length). GPU memory is the use of the whole GPU after loading minus the use before the start. Without the RTX 50-only part (the fast path of an RTX 30 or 40 card, measured on the RTX 5090 with that part off) prompt reading was 1,284, 1,236 and 1,012 tokens/s at 4,096, 32,768 and 131,072 tokens; generation does not use that part. We have not run the fast path on an RTX 30 or 40 card ourselves. A user measured 0.3.3 on an RTX 3090 Ti (24 GB, Windows 11; the reporter's numbers, [#29](https://github.com/yasakani-dev/yasakani/issues/29)): `selfcheck` passed with the fast path on and with it off; `bench` averaged 131.0 tokens/s with it on against 101.6 with it off; prompt reading was 421.2 against 256.3 tokens/s at 4,096 tokens and 405.0 against 171.1 at 32,768; a 131,072-token prompt was read in 356 s; at a 32K context the engine used 17.87 GB of GPU memory against 19.47 GB. No RTX 40 card has been reported.

The fast path computes in a slightly different way (the 8-bit cache, other kernels, another order of sums), so its output is not the same as that of 0.3.2. We compared it with the standard path (32-bit cache) on the text above at 128 positions after 4,096 and after 32,768 tokens. The most likely next token was the same at 126 and 123 of the 128 positions, with a mean KL divergence of 0.0035 and 0.0059. On another text it was 128 of 128 at 32,768 tokens (KL 0.0043). Greedy output therefore parts from that of the standard path after some tokens: in these runs the 64 greedy tokens first differed at token 5 (after 4,096) and token 31 (after 32,768). We have not measured task accuracy. On the fast path the drafter and copy decoding (below) still do not change the output: a code edit and a JSONL edit gave the same tokens with and without them, greedy and with seeded sampling at temperature 1.0.

Same PC, compared with llama.cpp b12600: the GGUF `Inkling-2-Mini-IQ4_NL.gguf` of ggml-org (17.63 GB), `llama-server -ngl 999 -fa on -np 1` with an 8-bit key/value cache (`-ctk q8_0 -ctv q8_0`), the same token ids as input, median of 2 runs (131,072 tokens: 1 run with `-ub 256`; with the default `-ub 512` the 16-bit cache did not fit, and the 8-bit run kept `-ub 256`). llama.cpp reads short prompts faster: 2,119 against our 1,826 tokens/s at 4,096 tokens. From 32,768 tokens our fast path reads faster: 1,755 against 1,467, and 1,377 against 639 tokens/s at 131,072. Generation without our drafter (llama.cpp has none here): 222 / 209 / 169 / 109 against 91 / 90 / 80 / 67 tokens/s after 256 / 4,096 / 32,768 / 131,072 tokens. The weights differ (our 15.05 GB file against the 17.63 GB GGUF), and so does the timing: llama.cpp reports its own server timings, we measure the wall time inside the process. GPU memory in use at 131,072 tokens: llama.cpp 25.1 GiB, our fast path 20.9 GiB with the drafter.

**Copy decoding for edits (new in 0.3.3, off by default).** `--copy auto` turns it on. When an answer copies long parts of the prompt, as in an edit, the model checks a copied span in one pass and keeps only the tokens it would have chosen itself, so the output does not change. With the fast path on the RTX 5090 (one run each): a JSONL edit ran at 208 tokens/s without the drafter, 508 with the drafter and 791 with the drafter and copy decoding; a code edit at 203, 442 and 716 tokens/s. With seeded sampling at temperature 1.0: 191, 414 and 583, and 187, 367 and 548 tokens/s. On text that does not copy the prompt it costs a little: on 12 short prompts a development build was 1.4 % slower on average and 15 % slower at worst.

What you need: 64-bit Windows 11, an NVIDIA RTX card with 24 GB or more (cards under 24 GB cannot hold the model), the YASAKANI quickstart or package of 0.3.1 or newer (0.3.2 or newer to choose it in the start screen, 0.3.5 or newer for images, which also need the Pillow library: the quickstart of 0.3.5 installs it, and with the package alone use `pip install ".[vision]"`), and about 20 GB of free disk for the download. The converter links the weights file into the new folder when both are on the same drive, so the folder takes almost no more space there (on another drive the converter copies the file). The steps below are for Windows. On Linux there are two ways, both built from source and not part of an official release ([BUILDING.md](BUILDING.md#linux-inkling-2)): the engine inside `libyskn.so` (`-DYSKN_WITH_INK2=ON`, by @audreyt in [#28](https://github.com/yasakani-dev/yasakani/pull/28), [#30](https://github.com/yasakani-dev/yasakani/pull/30) and [#32](https://github.com/yasakani-dev/yasakani/pull/32)) reads the downloaded folder without conversion, is faster since 0.3.5, reads images and has a prefix cache; `libyskn_inkling2.so` (`build_linux.sh`, a build contributed by @Albrod in [#33](https://github.com/yasakani-dev/yasakani/pull/33)) reads a converted folder with the fast path, the drafter and images.

**The quickest way: the start screen (new in 0.3.2).** Choose 4 at the model question of `START.cmd` (or start `START.cmd --model inkling2`). The quickstart downloads our converted copy from Hugging Face (about 15.9 GB with the image file of 0.3.5, no conversion; a folder of 0.3.2 to 0.3.4 is asked once whether to add that file, about 0.64 GB: Enter alone downloads it, `n` keeps text only, `START.cmd --image-file` adds it later), checks it with `info`, `verify` and `selfcheck` like the other models. Then the menu comes: press Enter and the page opens in your browser. The model question comes on the first run and with `START.cmd --menu`; cards under 24 GB are not offered the model. The model switch of the page lists it once it is downloaded. The ways below stay for a folder of your own.

**By hand: our converted copy.** We publish the `--scheme quality` conversion described below on Hugging Face: [yasakani-dev/Inkling-2-Mini-YASAKANI-quality](https://huggingface.co/yasakani-dev/Inkling-2-Mini-YASAKANI-quality) (15.9 GB, with the 4-bit drafter and, since 0.3.5, the image file). Download it and use the folder as it is, without converting. In a quickstart folder, run `yasakani_env.cmd` first; it puts `hf`, `python` and `yasakani` on the path.

```bat
hf download yasakani-dev/Inkling-2-Mini-YASAKANI-quality --local-dir D:\models\inkling2-mini-quality
yasakani verify --model-dir D:\models\inkling2-mini-quality
START.cmd --model-dir D:\models\inkling2-mini-quality
```

To convert the published model yourself instead:

1. Download the published model with the Hugging Face command line tool (`hf`, from the Python package `huggingface_hub`), at the revision we checked.
2. Convert it with `tools/convert/inkling2_convert.py` from the source of this release (the source code ZIP on the release page, or a clone of this repository). By default the converter does not change the weights: it writes the files our package needs next to them (the manifest `yasakani.json` with the SHA-256 of every file, `selfcheck.json` and `NOTICE`). The image part stays in that weights file, and image input reads it from there (since 0.3.5; we have not run this way on Windows). The drafter (`mtp.safetensors`) is linked or copied into the folder in the same way as the weights (with `--scheme quality`, below, the converter writes a 4-bit copy of it instead).
3. Check the folder, then chat, start the server or open the browser page.

In a quickstart folder, run `yasakani_env.cmd` first in the command prompt; it puts the quickstart's `python` and `yasakani` on the path.

```bat
cd /d C:\yasakani
yasakani_env.cmd
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir D:\models\Inkling-2-Mini-hf
python C:\src\yasakani-0.3.5\tools\convert\inkling2_convert.py --source D:\models\Inkling-2-Mini-hf --out D:\models\inkling2-mini
yasakani verify --model-dir D:\models\inkling2-mini
yasakani selfcheck --model-dir D:\models\inkling2-mini
yasakani chat --model-dir D:\models\inkling2-mini
yasakani serve --model-dir D:\models\inkling2-mini
START.cmd --model-dir D:\models\inkling2-mini
```

The converter is run with the quickstart's Python (the `python` that `yasakani_env.cmd` puts on the path): it uses numpy, tokenizers and
jinja2, which are installed there with the package. The commands find the engine library `yskn_inkling2.dll` by themselves: it is next to
`yskn.dll` in the engine folder of the quickstart, and `yasakani_env.cmd` names that `yskn.dll`. Give `--dll` only for a library in
another place (the path of `yskn_inkling2.dll`, or of the `yskn.dll` it is next to).

`selfcheck` runs the prompt of the official text test of the model's code in the transformers library and compares the 30 greedy tokens with the ones that test expects. `serve` gives the same OpenAI-compatible API as for the Qwen models, with the model id `inkling-2-mini`. `START.cmd --model-dir` opens the browser page with this folder. It first runs `info`, `verify` and `selfcheck` on the folder at every
start (in our run the checks and the start of the page took 35 to 40 s with the files in the Windows file cache); the other options of
`START.cmd` work as before.

**Speed with the fast path off** (as in 0.3.2; one RTX 5090, Windows 11, November 22, 2026, the published weights, one request at a time): generation with greedy decoding, thinking off and 256 tokens ran at 161.2 tokens/s with `yasakani bench --draft off` and at 212.5 tokens/s with the default `--draft auto` (the drafter, below; the mean of the 4 prompts of `bench`, median of 3 such runs after one warm-up run). Chat in the terminal with thinking on at low effort ran at 155.0 tokens/s (one question in Japanese, 375 tokens, `--draft auto`); the server streamed 173 tokens/s (a short question in Japanese, 145 tokens with thinking), with the first piece 0.08 s after the request. A long conversation is slower (measured with the engine before the changes of 0.3.1: from position 1,024 on, a token took 7.3 to 7.5 ms, 133 to 138 tokens/s). Prompt reading through the package: 1,063 tokens/s for a prompt of 1,079 tokens and 961 tokens/s for 4,900 tokens (median of 3 runs each, from an empty context). When a prompt starts like the one before (the next turn of a chat), only the new part is read. The model was ready 6.4 s after the start without the drafter and 8.5 s with it (files in the Windows file cache). With the default context of 32,768 tokens the engine allocated 22.7 GB of GPU memory (2.15 GB of it the key/value cache), 23.8 GB with the drafter; `--max-ctx` changes the context, and the cache grows with it. Since 0.3.2, without `--max-ctx` the package takes 32,768 tokens when they fit the free GPU memory, else the longest context that fits; a line at start says which ([Commands and settings](python/README.md#inkling-2-mini-experimental)).

**Answers.** On the RTX 5090 our engine gave the 30 tokens that the official test expects, and it agreed with our own CPU implementation of the model (hidden states within 1e-4, the same experts chosen). We measured agreement with the published model only, not accuracy on tasks.

**Drafter.** Since 0.3.1 the package uses the drafter that comes with the model (`mtp.safetensors`, a few small layers published with the weights). The drafter guesses the next few tokens, and the model checks all of them in one pass. A guessed token that the model would not have chosen itself is thrown away, so the output does not change: greedy decoding gives the same tokens, and sampling picks each token with the same probabilities. `--draft` chooses how it is used: `auto` (the default) uses it only while it makes generation faster, `on` always uses it, and `off` does not load it (the environment variable `YASAKANI_DRAFT` does the same). With `auto` the drafter guesses up to 7 tokens, and the model checks them only up to the first one the drafter is unsure of. `--draft-depth N` sets how many tokens it guesses per pass with `on` (1 to 7). The start-up lines say whether the drafter was loaded and which mode is used. A model folder without the file still works, without the drafter.

Speed with the drafter (one RTX 5090, Windows 11, November 22, 2026, greedy, 200 tokens per prompt, through the package: one process per folder and `--draft` setting, the prompts one after the other, median of 3 rounds; Japanese prose: 2 prompts, one with thinking; code: 2 prompts, one with thinking; JSON: 1 prompt; math: 1 prompt with thinking). The `--scheme quality` folder has the 4-bit drafter `mtp_d4.yskn`, the published one `mtp.safetensors`. Tokens per pass are those of `--draft auto`:

| text | published weights, `--draft off` | published weights, `--draft auto` | `--scheme quality`, `--draft off` | `--scheme quality`, `--draft auto` | tokens per pass (published / quality) |
|---|---|---|---|---|---|
| Japanese prose | 162.0 tokens/s | 174.5 tokens/s | 205.0 tokens/s | 201.2 tokens/s | 1.32 / 1.11 |
| code | 162.3 tokens/s | 227.7 tokens/s | 205.3 tokens/s | 225.9 tokens/s | 2.44 / 1.60 |
| JSON | 162.0 tokens/s | 293.2 tokens/s | 204.2 tokens/s | 266.7 tokens/s | 3.64 / 2.30 |
| math | 162.6 tokens/s | 211.8 tokens/s | 204.9 tokens/s | 233.9 tokens/s | 1.89 / 1.77 |

Over these and one English prompt (7 prompts), `--draft auto` gave 207.2 against 162.2 tokens/s with the published weights and 223.9 against 204.9 tokens/s with the `--scheme quality` weights. In every run the greedy output was the same, token for token, with and without the drafter. Where the guesses are seldom right, `auto` costs a little: with the `--scheme quality` weights, Japanese prose was 1.5 to 2.1 % slower. With `yasakani bench` (its 4 prompts, thinking off, 256 tokens; median of 3 runs): published weights 161.2 tokens/s with `--draft off` and 212.5 with `--draft auto`; `--scheme quality` 203.8 and 225.9 tokens/s.

The drafter helps most where the next tokens are easy to guess (code, JSON, math). `yasakani chat` and `yasakani bench` print the tokens per pass; the server gives them in `timings` (`tokens_per_pass`, `draft_passes`, `draft_n`, `draft_n_accepted`).

**Smaller converted weights (`--scheme quality`).** The converter can also write the weights again in fewer bits: with `--scheme quality` it stores most dense matrices in 4 or 8 bits and re-packs the 8-bit experts of layers 20 to 23 to 4 bits, in one `.yskn` file of 15.05 GB (the published file has 19.5 GB). Against the published model on 43 text windows (14,024 positions: the model's own thinking and answers, English and Japanese prose, code) the KL divergence was 0.0110, and the most likely next token was the same at 96.0 % of the positions. The converter also writes the drafter again in 4 bits, as `mtp_d4.yskn` next to the weights (209.7 MB instead of 635.8 MB; `--drafter-scheme published` keeps the published file; [FORMAT](docs/FORMAT.md) section 10.6). The engine multiplies all these matrices as they are stored, so the folder is faster and needs less GPU memory. Measured in one job on the RTX 5090 (`yasakani bench`, greedy, thinking off, 256 tokens, median of 3 runs): 203.8 tokens/s against 161.2 for the published weights with `--draft off`, and 225.9 against 212.5 tokens/s with the default `--draft auto`. At the default context of 32,768 tokens the engine allocated 18.9 instead of 22.7 GB (19.5 instead of 23.8 GB with the drafter: the 4-bit drafter takes 0.56 GB, the published one 1.10 GB). Prompt reading through the package was a little slower: 959 against 1,063 tokens/s for 1,079 tokens and 878 against 961 tokens/s for 4,900 tokens (median of 3 runs each). The model was ready 2.6 s after the start without the drafter and 3.0 s with it. The conversion takes about 4 minutes on the CPU; the new folder needs about 15 GB of disk. For image input add `--vision`: it writes the image part as a file of its own (`inkling_2_mini_vision.yskn`, 637 MB); without it this folder is text only.

```bat
python C:\src\yasakani-0.3.5\tools\convert\inkling2_convert.py --source D:\models\Inkling-2-Mini-hf --out D:\models\inkling2-mini-quality --scheme quality
yasakani verify --model-dir D:\models\inkling2-mini-quality
yasakani selfcheck --model-dir D:\models\inkling2-mini-quality
yasakani chat --model-dir D:\models\inkling2-mini-quality
START.cmd --model-dir D:\models\inkling2-mini-quality
```

`verify` compares every file with its SHA-256 and every tensor with its CRC-32. `selfcheck` runs three short prompts (one in English, one in Japanese, one asking for code) and compares 48 greedy tokens of each with the tokens that a reference implementation of the model gave on these converted weights, decoded exactly as the engine multiplies them. We chose prompts where, at each of the 48 tokens, the chosen token leads the next best one by at least 0.11 in the logits (in the reference and in our engine), so that small differences of arithmetic on other GPUs should not change the result; only the RTX 5090 was tested. The conversion on the CPU gives the same file every time: when the `.yskn` file has the SHA-256 of our conversion of revision `ed40e7da625f594fc9c4990f5f66be24873c22d7` (`44ef7ea5eade7a1e6cfb4f7ad3f7568a701ce9c621f7e4d00d38b624f4de138d`, 15,052,423,172 bytes), the converter writes those tokens into `selfcheck.json`, and `START.cmd --model-dir` runs its checks on this folder as on the published weights. A file that differs (for example one made from another revision) gets a `selfcheck.json` without cases: `selfcheck` then has nothing to compare, and `START.cmd --model-dir` needs `--skip-checks` for that folder (it stops at the self-check otherwise).

**Tool calls.** New in 0.3.1. `yasakani serve` with an Inkling-2 folder accepts `tools` and returns the calls of the model as `tool_calls` (finish reason `tool_calls`), with and without streaming. `tool_choice` takes `auto`, `none`, `required` or the name of a function. With `auto` one reply may hold several calls; with `required` (also among several tools) or a named function the model can only write one of the offered names as the call, and the reply ends after that one call. The model writes its calls in its own format and the server reads them; a block it cannot read comes back as text. Send the results back as `tool` messages with their `tool_call_id`. With tools and thinking off the model still thinks a little (34 to 85 tokens in our runs, returned in `reasoning_content`): the reply then starts at thinking effort 0, and the model writes the start of its answer or call itself. On one RTX 5090 with the `--scheme quality` folder we ran one call, a call with its result and the answer, two calls in one reply, streaming, and the four kinds of `tool_choice`.

**Limits.** Experimental in 0.3.1.

* Still images since 0.3.5 ([image input](python/README.md#inkling-2-mini-image-input)): no video or animated images, images in user and tool messages only, and no images from URLs. An image above 1,024 image tokens is made smaller first (our own rule, not the model's).
* One request at a time: no `--parallel`.
* Tool calls in `yasakani serve` only, not in terminal chat. A conversation with tools stops at the context of the server (32,768 tokens by default): a request that does not fit gets an error (HTTP 400), and a new conversation is needed. With thinking on, a long thinking can use up `max_tokens` before the call is written; `thinking_budget` (a request field, or the environment variable `YASAKANI_THINKING_BUDGET`) cuts the thinking after that many tokens.
* Most clients do not send the thinking of the last reply back, so the next request drops that reply. The engine can drop up to 512 positions without reading the conversation again (`YASAKANI_INK2_ROLLBACK`; 0 gives the engine's default of 64). This takes more GPU memory than 64 positions: with the `--scheme quality` folder and context 32,768 the engine allocated 0.6 GB more, and the GPU had 0.8 GB more in use (the allocations above include it).
* The drafter speeds up text that is easy to guess more than free prose. `--draft auto` stops using it where it does not pay, but its trial passes still cost a little there (up to 2.1 % on Japanese prose with the `--scheme quality` weights in our runs); `--draft off` avoids that.
* `bench --edit` is not available for this model.
* We ran the fast path on one RTX 5090 only; on RTX 30 cards there is the user report of an RTX 3090 Ti (above), and no RTX 40 card has been reported. Its output is close to, but not the same as, that of the standard path (above). `selfcheck` compares the standard path unless `--fast` is given.
* Windows: tested on one RTX 5090 under Windows 11 only; not on other GPUs. A user ran 0.3.1 on an RTX 3090 (24 GB, Windows 11): `selfcheck` passed, `bench` gave 105.6 tok/s, and 22.52 GB of GPU memory were in use at a 32K context, the display included ([#29](https://github.com/yasakani-dev/yasakani/issues/29); the reporter's numbers). Linux: built from source only, two ways ([BUILDING.md](BUILDING.md#linux-inkling-2)): the engine inside `libyskn.so` for the downloaded folder ([#28](https://github.com/yasakani-dev/yasakani/pull/28), [#30](https://github.com/yasakani-dev/yasakani/pull/30), [#32](https://github.com/yasakani-dev/yasakani/pull/32)) and `libyskn_inkling2.so` for a converted folder ([#33](https://github.com/yasakani-dev/yasakani/pull/33)). We ran both on one RTX 5090 under WSL2 only; the author of #28 ran that engine on an RTX 5090 and a DGX Spark, and the author of #33 the library on an RTX 3090 (before image input).
* Thinking off is our own setting ([FAQ](docs/FAQ.md#inkling-2-mini)).

[FAQ](docs/FAQ.md#inkling-2-mini) · [Commands and settings](python/README.md#inkling-2-mini-experimental)

## Community conversions

Files that other people made with the converter in [tools/convert/](tools/convert/). They are **not checked or maintained by this
project**; the person named below keeps them working, and a conversion may be removed from this list in a release where it has fallen
behind.

**Arete-27B** (by [@audreyt](https://github.com/audreyt)): a community conversion of [NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B),
Nous Research's fine-tune of Qwen3.8-27B, made with the quality model's settings and reusing its draft model unchanged. Files:
[audreyt/Arete-27B-YASAKANI-quality](https://huggingface.co/audreyt/Arete-27B-YASAKANI-quality/tree/659effbdf9dc103fcc053e3da7858bcff7f14cc9) at revision
`659effb`; the SHA-256 of every file is in its `SHA256SUMS` (model file `arete_quality.yskn`, 12,365,242,368 bytes,
`541c9206ede621bd92e78ff09f838fca5e4660d834fa883b3d20cdb4ab56880a`).

```bat
hf download audreyt/Arete-27B-YASAKANI-quality --revision 659effbdf9dc103fcc053e3da7858bcff7f14cc9 --local-dir D:\models\arete-quality
yasakani selfcheck --model-dir D:\models\arete-quality --dll D:\yasakani\yskn.dll
```

Then use it as in [Quick start](#quick-start) with `--model-dir D:\models\arete-quality`. Measured by the author of this conversion on an
RTX 5090:

* Closeness to its own BF16 original (`tools/quality/`, 20 thinking windows, KV5 cache, measured with 0.1.3): KL divergence 0.0116,
  same most likely token 95.64%. The quality model, measured the same way against Qwen3.8-27B: 0.0126 and 95.35%.
* With 0.1.4: `yasakani verify` 11 of 11 files, `yasakani selfcheck` 3 of 3 (the same outputs as with 0.1.3). Tokens per look-ahead
  pass: prose 2.40, thinking 2.48, code 4.68, CSV to JSON 15.04. On thinking text that is fewer than the quality model (2.48 against
  3.24 in the author's run of both on the same card with 0.1.3), so expect it to be slower there than the speed tables above.
* Further evaluations, with their settings and caveats, are on the [model card](https://huggingface.co/audreyt/Arete-27B-YASAKANI-quality/tree/659effbdf9dc103fcc053e3da7858bcff7f14cc9).

**Arete-27B-pi** (by [@audreyt](https://github.com/audreyt)): Arete-27B with the harness-hub pi coding adapter merged before quantization.
The [model card](https://huggingface.co/audreyt/Arete-27B-pi-YASAKANI-quality/tree/7a0670eaa25753ed82aa1e31f1d943d6ff582be0)
(revision `7a0670e`; model file `arete_pi_quality.yskn`, 12,365,242,368 bytes, `49676c4ad01a9714fc35f74c9c4f6bb18d9367911f2453eab6b44798d907f75b`)
includes pi setup instructions and results measured by the author of this conversion on an RTX 5090 (Linux) with YASAKANI 0.1.4:
`verify` matched all 11 files and `selfcheck` passed all 3 cases. Against the
merged BF16 model, the quality tools measured KL divergence 0.01127 and 95.57% top-1 agreement (20 thinking windows, 20,460 positions, KV5 cache).

<a name="how-it-works-briefly"></a>
<a name="documentation-and-contributions"></a>

## Development and support

YASAKANI’s weight format and GPU kernels were designed together. In an edit, the main model checks a whole span copied from the input in one pass. This cuts the time spent writing out unchanged lines. [Technical notes](docs/TECHNOLOGY.md) · [File format](docs/FORMAT.md) · [Conversion tools](tools/convert/)

<a name="use-it-as-an-agent"></a>

You can build from source and report results from your GPU. You can also try the optional browser agent. It needs 24 GB of GPU memory and comes with an older version of the server. [Build guide](BUILDING.md) · [Contribute](CONTRIBUTING.md) · [Agent setup](harness/HARNESS_README.txt)

<a name="support"></a>

This engine was developed in Japan. Support pays for test hardware and development time, first for 24 GB and 16 GB RTX cards. Backers who ask to be listed are credited in the repository and in the startup credits of a later release. [CAMPFIRE for backers in Japan](https://camp-fire.jp/projects/992233/view) · [GitHub Sponsors](https://github.com/sponsors/yasakani-dev)

<a name="license"></a>

Released under Apache License 2.0. [License](LICENSE) · [Copyright notices](NOTICE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Release history and contributors](docs/RELEASE_NOTES.md)
