# FAQ

日本語は [FAQ.ja.md](FAQ.ja.md)。

## What is YASAKANI?

An inference engine for running **Qwen3.8-27B** on your own Windows PC with **one NVIDIA RTX GPU** (tested on the RTX 5090). It is GPU kernels, a file format for the
compressed model, and a runtime, with an OpenAI-compatible server, a terminal chat, a benchmark and a self-check. One person in Japan built it. It is
not a general-purpose engine: one model, tuned on one card. Since 0.3.1 it also runs a second model, Inkling-2 Mini, as an experimental
model ([below](#inkling-2-mini)); it reads text, and since 0.3.5 still images.

## Which model should I download?

There are three conversions of Qwen3.8-27B, and you need only one. The **quality model** (12.37 GB) stays close to the original model and is the
one recommended for everyday use. The **fastest model** (10.31 GB) is smaller and writes faster in most kinds of task, and is a little further
from the original model. The **max-quality model** (15.72 GB) is the closest to the original model; it needs YASAKANI 0.2.0 or newer and a card with 24 GB or more, and it
generates at about 0.8 to 0.9 times the speed of the quality model (RTX 5090). The numbers are in the [README](../README.md) and in
[MEASUREMENTS.md](MEASUREMENTS.md#max-quality-model).

## How fast is it, and how was that measured?

On one RTX 5090 with Qwen3.8-27B, YASAKANI's fastest model wrote answers faster than each of the eight other engines we measured on the same PC,
in all seven task categories, with thinking off and with thinking on. For example, with thinking off it wrote 2,025 tokens per
second on the code-edit task (0.1.4, November 15), against 441 for MegaCapybara 1.39 measured on November 3 (about 4.6 times), and
1,214 on long JSONL.

"Speed" is the speed of **writing the answer**: (output tokens - 1) / (time from the first to the last content chunk). The time to read the prompt
is not included. Each engine ran with its own weights, quantization, key-value cache format and speed-up settings, one request at a time. With
thinking off, runs that fell into a repetition loop or made an unrequested tool call are left out of the average by one rule for all engines;
with thinking on, the fastest run of each task is used, chosen from 5 runs for YASAKANI and from 2 for the other engines. The tables are in the
[README](../README.md); the conditions, the tallies without these choices and what the tables do not show are in
[MEASUREMENTS.md](MEASUREMENTS.md). [`bench/`](../bench/) has what you need to check or repeat the comparison, and `yasakani bench` measures
your own PC.

0.2.0 reads prompts faster on the RTX 5090 ([below](#prompt-reading-020)). That does not change these tables, because they do not include
the time to read the prompt.

## Will I see the same speed?

Not exactly. The speed depends on what you ask: predictable text (code, structured output) is written faster than free-form prose, because the
engine proposes several words at once and keeps the ones the main model agrees with. Your GPU's clocks, cooling and the other programs using it
matter too. Run `yasakani bench` to see your own numbers.

## How do I measure the edit speed on my PC?

```bat
yasakani bench --edit --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

After the quickstart, open a command prompt in its folder, run `yasakani_env.cmd` (it sets the model folder and the DLL), then type
`yasakani bench --edit`. It runs the three edit tasks of the speed tables: the code edit and the JSONL edit of the README table, and the
long JSONL edit (160 records) of "What 0.1.4 changed", with the prompts, output limits and expected answers of [`bench/`](../bench/) (they
come with the package). As in the tables it is greedy with thinking off. `--think` turns thinking on, with temperature 1.0 and an output
limit of 32,768 tokens for every task (the server's default), so tok/s varies from run to run and the thinking part is not sped up much;
the thinking columns of the README tables are not what it reproduces. `--runs 2` repeats each task, `--json` prints a machine-readable
result. For each task it prints the tokens, the seconds, tok/s, the tokens per pass of the main
model, the number of wide copy passes, and whether the answer is the expected one. The first lines give the package version, the SHA-256 of
`yskn.dll`, the GPU (name, compute capability, driver), the model file and whether wide copy passes are used, so a pasted result says what
it was measured with. Since 0.2.0 the lines at start also say which prompt reading is used, and `edit copy: on` when the
[edit copy](#edit-copy-020) is on. It is on by default; the speed tables of the README were measured before it existed, so
`yasakani bench --edit --edit-copy off` is the run to compare with them.

The wide copy passes (since 0.1.4: a pass that copies the input checks up to 64 tokens) are used where the check on the card opened them
(since 0.1.7 on RTX 30, 40 and 50 cards; see [below](#wide-copy-passes-per-gpu)); we measured them on the RTX 5090 only. Speed here is (tokens - 1) / the engine's
time from the first to the last generated token, inside the process; the thinking-off tables were measured through the server's stream, so
the numbers are close to them, not the same. The file of the code edit is CPython's `textwrap.py` with one e-mail address replaced by
`<email>`, as in `bench/`. YASAKANI's thinking-off measurements of November 15 (the speed table and "What 0.1.4 changed") used this same
text; every earlier record of the code edit in `bench/` (November 3 to 8; among them the other engines' rows and YASAKANI's thinking-on
rows) had the address itself, 4 prompt tokens more. In the records of November 15 (0.1.4) every answer was the expected one except the
quality model's code edit (2 lines differ).

## Are the answers the same as the original model's?

No. The models are compressed (the original BF16 weights are 53.8 GB; the quality model is 12.37 GB and the fastest model 10.31 GB), so they
are **not identical** to the original. On 20 thinking transcripts that were not used when the files were made (20,460 positions, FP16 key-value
cache, measured November 8, 2026), the most likely next token is the same as the original BF16 model's **95.24%** of the time for the quality
model and **93.01%** for the fastest model, and the KL divergence is 0.0122 and 0.0304. These numbers say how close the next-token predictions
are; they are not the rate of correct answers on tasks, and they cover thinking text. On general web text an additional evaluation showed a KL
divergence 3-10% larger for the quality model than for our earlier conversion of the same size. The max-quality model (15.72 GB), measured the same
way with the default KV5 cache on November 20, 2026, agrees at **97.40%** with KL 0.0040; the quality model gives 95.35% and 0.0126 with KV5.

Does the speed-up itself change the answers? It is built not to: speculative decoding only keeps tokens that the main model agrees with.
Up to 0.1.2, while other programs were using the GPU at the same time, a few repeated runs had the same tokens as the others up to one
position and continued differently from there (November 10, 2026: 1 of 190 runs with the default settings, 21 of 432 with speculative
decoding switched off; 256 tokens each). With nothing else using the GPU, all 240 repeated runs were identical. The cause was a missing
synchronization in the kernels that apply the position rotation; 0.1.3 fixes it, and in our tests with the fix it did not appear again.
We still do not promise bit-identical text in every condition; the details are in [MEASUREMENTS.md](MEASUREMENTS.md).

The copying of 0.1.4 was checked against decoding one token per pass: 255 of 255 generations with copying (28 of them with wide passes of
up to 64 rows) gave the same tokens ([TECHNOLOGY.md](TECHNOLOGY.md), chapter 7).

The new prompt reading of 0.2.0 was compared with 0.1.7 on the RTX 5090, with both models. In 46 cases (prompts of 256 to 65,536 tokens,
589,088 positions) the predictions, the 8 most likely candidates with their scores and the state after the prompt were the same; this
comparison used a test build of the 0.2.0 engine. The released engine library was compared through the chat's own path: 22 cases (first
reads of 256 to 65,536 tokens, prompts changed in the middle, thinking on), 4 runs each, gave the same tokens and the same state every
time. 35 conversations per model, one request at a time, gave the same tokens as 0.1.7, with the edit copy off and with it on. The [edit copy](#edit-copy-020) of 0.2.0 only
proposes tokens; the model still checks every one of them, so the output does not change.

## Which GPUs, which systems?

We test on **Windows 11 and the RTX 5090** (32 GB). Since 0.1.3 the engine is also built to run on other NVIDIA RTX 30, 40 and 50 series cards with
compute capability 8.6 or higher and 16 GB or more: the RTX 3090, RTX 3090 Ti and RTX 4090 (24 GB; the quality and fastest models, and the max-quality model with a shorter context; one user report, [#24](https://github.com/yasakani-dev/yasakani/issues/24)) and the
RTX 4060 Ti 16GB, RTX 4070 Ti SUPER, RTX 4080, RTX 4080 SUPER, RTX 5060 Ti 16GB, RTX 5070 Ti, RTX 5080 (16 GB; the fastest model only, with a shorter context). The engine is
not optimized for them and we have not tested it on them, so neither its speed nor its operation is guaranteed; they are slower than the
RTX 5090. The quickstart says so and goes on. Since 0.1.4, however, its self-check compares the card's predictions with those of the
RTX 5090, and the quickstart stops when they differ by more than small differences in the computation, or when the card's passes of
different widths do not agree. The faster prompt reading of 0.2.0 is used by default on cards with 30,000 MiB or more (the RTX
5090) and, since 0.3.4, on cards of 22,000 MiB or more when `--max-ctx` leaves room for it; the other cards read prompts exactly as 0.1.7
did ([below](#prompt-reading-smaller-cards)).
Results from these cards are welcome (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
AMD GPUs, Macs, cards with less than 16 GB and cards older than the RTX 30 series are not supported. On Linux the engine can be built from source (a community contribution, [#3](https://github.com/yasakani-dev/yasakani/pull/3), checked on
WSL2 / Ubuntu 24.04 with an RTX 5090); a prebuilt Linux release with a launcher is one of the campaign's goals. Testing and tuning on the 24 GB and 16 GB RTX cards first, then AMD and Mac - that is what the
crowdfunding campaign is for; the order and the conditions are on the [campaign page](https://camp-fire.jp/projects/992233/view).

The NVIDIA driver must be 580 or newer. The engine is built with CUDA 13.2; 580 is the oldest driver NVIDIA lists for programs of the CUDA
13 toolkits under its minor version compatibility
([NVIDIA's documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)), with limits: features that
need a newer driver fail, and so does compiling PTX at load time. The engine carries machine code for each GPU generation it supports, so
it needs no PTX compilation. We have tested 616.92 only.
The quickstart checks the driver before it downloads the engine or a model, and if the driver is older it stops and shows where to get a
new one.

Two GPUs in one PC: one process uses one GPU, chosen with `--device N` (GPU N of `yasakani info` and nvidia-smi). With 0.1.5 this
could fail with `graph instantiate failed` while the other GPU was visible ([#20](https://github.com/yasakani-dev/yasakani/issues/20)).
0.1.6 has a fix for this. On November 18, 2026 the reporter of #20 confirmed that it works on a PC with two GPUs (RTX A2000 and
RTX 5060 Ti); we have not tried two GPUs ourselves. A `CUDA_VISIBLE_DEVICES` you set yourself is kept. See
[What you need](../README.md#what-you-need).

<a name="test-build-017"></a>
<a name="wide-copy-passes-per-gpu"></a>

## Are the wide copy passes used on my card?

Since 0.1.7, on RTX 30, 40 and 50 series cards, after a check on the card. The wide copy passes (since 0.1.4: a pass that copies the
input checks up to 64 tokens instead of 16; see [technical notes 5.4](TECHNOLOGY.md#54-wide-copy-passes-014)) make edits faster. Normal
prose is not faster: the gain is on answers that copy the input. Up to 0.1.6 they were used on RTX 50 cards only; the 0.1.7 test build
([v0.1.7-test1](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.7-test1)) let owners of other cards try them.

* **The check.** At the first start of `chat`, `serve`, `bench` or `selfcheck`, after the model is loaded, the package checks the card
  once. Passes of 17 to 64 tokens must give the same predictions as passes of 16, bit for bit (since 0.2.0 also passes of 64 tokens
  after position 4,096; 463 positions in all). If not, no wide pass is used and
  `yasakani selfcheck` fails: please report it in GitHub Issues. On RTX 30 / 40 cards a width (32, 48 or 64 tokens) is then used only if
  its copy steps take at most 0.8 × width / 16 times a 16-token step on this card (32: 1.60, 48: 2.40, 64: 3.20), measured while the GPU
  is idle. RTX 50 cards start with every width, as in 0.1.6, and the check leaves out a width only when every round is over that limit.
  Since 0.3.4 the limit on RTX 50 cards is 0.9 × width / 16 (32: 1.80, 48: 2.70, 64: 3.60): on an
  RTX 5060 Ti the old limit closed all three widths, although with every width open the three edit tasks ran 1.23 to 1.24 times as fast
  with the same output ([#31](https://github.com/yasakani-dev/yasakani/issues/31); the reporter's numbers). With the times of that
  report all three widths stay open under the new limit. Under it our RTX 5090 still opened all three widths (its copy steps took 1.37, 1.81
  and 2.05 times a 16-token step), and the three edit tasks gave the same token ids with wide passes off and on. On the RTX 5090 the check kept every width and took 21.8 to 22.9 seconds (0.1.7 test build, November 17, 2026); it takes longer on
  other cards.
* **When it cannot check.** When another program uses the GPU, when nvidia-smi does not answer, or when the context is shorter than 4,224
  tokens, nothing is measured. `serve --no-warmup` uses a kept result only. Without a measured or kept result, RTX 50 cards use every
  width and RTX 30 / 40 cards none.
* **Memory.** On RTX 30 / 40 cards, when less than 128 MiB of GPU memory is free after the graphs of the check are built, no width is
  used. A shorter `--max-ctx` frees memory. The rule is checked once, at start; when it closes the widths, the start line says so. In
  one report (RTX 3090 Ti, max-quality model, desktop about 1.1 GB) 153 MiB were free after the graphs
  ([#24](https://github.com/yasakani-dev/yasakani/issues/24); the reporter's number).
* **How often it checks.** The result is kept per GPU, driver, engine library, package and model (`wide_check.json` in
  `%LOCALAPPDATA%\yasakani`). After an update the check runs once more, because the engine library and the package change. It also runs
  again after 30 days, with `--wide-recheck`, or when during generation the engine went back from wide passes in at least 10 of every 100
  (counted over at least 128 in one run of the program). During generation the engine goes back to 16-token passes when the wide ones
  turn out slower (it checks this every 4 wide passes). The check of 0.2.0 is a new version of it, so a result kept by 0.1.7 is
  measured again once ([below](#first-start-020)).
* **Settings.** `--wide off` (or `YASAKANI_WIDE=off`) turns the wide passes off; `--wide force` uses every width without the check, to
  compare. Since 0.1.7 `--wide on` is the same as the default.
* **Several requests.** With `serve --parallel N`, a request that runs alone uses the widths the check opened. Passes shared by several
  requests carry at most 16 tokens together and do not use them.
* **What we checked.** We ran the code for RTX 30 / 40 (compute capability 8.6 and 8.9) only on the RTX 5090: compiled for the RTX 5090
  when it loads, it gave the same predictions in wide passes as in passes of 16. That does not show that the machine code of an RTX 30
  or 40 card is right, or that the wide passes are faster there.
* **What users reported.** Users reported the 0.1.7 test build and later versions in
  [#24](https://github.com/yasakani-dev/yasakani/issues/24). These are the reporters' numbers:
  * RTX 4090 (Windows 10): the check opened widths 32, 48 and 64. Wide passes were equal to 16-token passes bit for bit at 335 positions.
    The three edit tasks ran at 770.0 → 1,420.9, 822.5 → 1,634.9 and 793.2 → 1,545.6 tok/s (wide passes off → on), with the same output.
  * RTX 5060 Ti: only the 64-token width opened. The three edit tasks ran at 215.1 → 279.4, 226.9 → 300.4 and 218.8 → 280.6 tok/s, with
    the same output.
  * RTX 3090 (Linux, 0.1.7 built from source): the check opened widths 32, 48 and 64; bit for bit at 335 positions. The three edit
    tasks ran at 423.0 → 641.7, 449.0 → 712.1 and 431.1 → 670.8 tok/s, with the same output.
  * RTX 3090 Ti (Windows 11, 0.3.2, quality model): the check opened widths 32, 48 and 64; bit for bit at 463 positions. The three
    edit tasks ran at 443.8 → 683.7, 461.8 → 741.4 and 448.8 → 704.0 tok/s, with the same output.
  * Another RTX 5060 Ti (0.3.2 built from source, fastest model; [#31](https://github.com/yasakani-dev/yasakani/issues/31)): the check
    closed all three widths (time ratios 1.79, 2.54 and 3.21 against the limits 1.60, 2.40 and 3.20). With every width in use
    (`--wide force`) the three edit tasks ran at 356.6 → 440.6, 367.5 → 450.7 and 354.2 → 434.9 tok/s, with the same output.
* **How to report.** `yasakani report` checks the card again, runs the three edit tasks of `bench --edit` with the wide passes off, with
  the widths the check chose and with every width, then the self-check, and prints a Markdown report: GPU, driver, versions, the check,
  the speeds, and whether the outputs are the same. It took 85 to 89 seconds on the RTX 5090 (0.1.7 test build); it takes longer on other
  cards. Please paste it into a GitHub issue (the reports so far are in [#24](https://github.com/yasakani-dev/yasakani/issues/24)). It
  contains no UUID, path, user or computer name. Details: [python/README.md](../python/README.md#wide-copy-passes-per-gpu).

<a name="edit-copy-020"></a>

## What is the edit copy?

New in 0.2.0. Some edits make the same replacement again and again: a key renamed in every JSONL line, a column renamed in every SQL
view. In the copy passes the engine then proposes the replacement the model has already written at earlier places, instead of the
unchanged input. The model still checks every proposed token, so the output does not change; only such edits get faster. An edit that
changes one place, and text that is not an edit, are not faster. The edit copy works where one request is generated at a time; the shared
passes of several requests in `serve --parallel N` do not use it.

It is on by default. `--edit-copy off` (or `YASAKANI_EDIT_COPY=off`) turns it off, `--edit-copy on` turns it on; it works with `chat`,
`serve`, `bench` and `selfcheck`. When it is on, `edit copy: on` is printed at start, and `yasakani info` shows the setting. A wrong value
of `YASAKANI_EDIT_COPY` stops before the model is loaded.

Measured on one RTX 5090, context 32,768, temperature 1.0, thinking off, one request at a time, 2 runs with the edit copy off and 2 with
it on, measured in turn (tok/s, mean of the 2 runs, off → on; the tokens and the state were the same in all 4 runs of every task):

| Task | Quality model | Fastest model |
|---|---|---|
| JSONL: a key renamed in every line (80 lines) | 1,151.4 → 2,039.8 (1.77×) | 1,244.9 → 2,009.3 (1.61×) |
| SQL: a column renamed in every view (50 views) | 1,168.5 → 2,101.0 (1.80×) | 1,262.7 → 2,025.2 (1.60×) |
| Code edit (`bench --edit`) | 2,040.1 → 2,209.0 (1.08×) | 2,048.1 → 2,163.7 (1.06×) |
| JSONL edit (`bench --edit`) | 2,292.2 → 2,309.3 (1.01×) | 2,284.5 → 2,286.6 (1.00×) |
| Long JSONL edit (`bench --edit`) | 2,168.7 → 2,186.9 (1.01×) | 2,165.2 → 2,162.3 (1.00×) |

The two rename tasks are not part of `yasakani bench --edit`, and their prompts are not in this release. In each the request is to return
the whole file with the new name; the first 1,024 tokens of the answer are timed. Without the edit copy the copy passes cannot be used on
them, because every line differs from the input; with it the model needed 25 to 27 passes instead of 66 to 68 for the 1,024 tokens.

On the 15 tasks of the speed tables (thinking off, temperature 1.0, one request at a time, the sum of all tasks, one uncounted pass and
then 3 runs per version) 0.2.0 with the edit copy on was +0.37 % (quality model) and +0.45 % (fastest model) against 0.1.7, with the
same tokens: the edit copy does not speed up these tasks beyond noise. We measured it on the RTX 5090 only. Details:
[MEASUREMENTS.md](MEASUREMENTS.md#measurements-020).

<a name="first-start-020"></a>

## The first start after the update to 0.2.0 takes longer. Why?

At the first start the package checks the wide copy passes on the card ([above](#wide-copy-passes-per-gpu)). It does that after every
update, and 0.2.0 also checks passes of 64 tokens after position 4,096, so a result kept by 0.1.7 is measured again once. On the RTX
5090 the first start of 0.2.0 printed `model ready in 28 s` (quality model) and `model ready in 27 s` (fastest model); the first start
of 0.1.7 in the same measurement printed 27 s and 26 s (one start per version). Later starts use the kept result.

## How much memory and disk does it need?

One model folder is a download of about 13 GB (fastest model), 15 GB (quality model) or 18 GB (max-quality model); keep 20 GB free. The context length goes up to 262,144
tokens. With the fastest model and that full context reserved, the engine takes 16.78 GiB of VRAM (27.13 GiB if the key-value cache were kept
in FP16); this is the amount reserved at start, not the peak of the whole setup. 32 GB of system RAM is recommended.

Together with the helper model that speeds up generation, the GPU memory in use at the full 262,144-token context was about 19.1 GiB for the
fastest model and about 21.2 GiB for the quality model (about 14.3 GiB and 16.4 GiB with `--max-ctx 32768`; November 11, 2026, before
0.2.0). On the RTX 5090, 0.2.0 uses more ([below](#prompt-reading-memory)).

On a GPU with less than 32 GB, the context is chosen from the free GPU memory each time the model starts (at most 262,144 tokens;
`--max-ctx` sets it yourself). On 16 GB cards only the fastest model fits. The max-quality model used 25,977 MiB after loading on the RTX 5090
(quality model: 22,795 MiB); on a 24 GB card the package would choose a context of 163,840 tokens when the desktop uses 500 MiB (196,608 with nothing else on the GPU; a calculation, not checked by us on such a card). A user ran it on an RTX 3090 Ti (Windows 11, 0.3.2) with the desktop using about 1.1 GB: the package chose 147,456 tokens, the wide-pass check was bit for bit at 463 positions, the self-check passed, and it generated at 0.95 times the speed of the quality model ([#24](https://github.com/yasakani-dev/yasakani/issues/24); the reporter's numbers). If the start-up line of the wide copy passes shows little free memory (that report: 153 MiB with the max-quality model on a 24 GB card), a `--max-ctx` one step shorter (16,384 tokens) leaves about 350 MiB more. The rule is in
[python/README.md](../python/README.md#gpus-other-than-the-rtx-5090).

## It is much slower than the tables

The tables were measured on an RTX 5090; other GPUs are slower, and we have not measured them. Check the GPU memory first. When other programs (games, other LLM servers) use GPU memory and it runs out, there is no error: generation just
becomes several times slower. In that case a warning that the GPU memory is almost full is printed at start. Close the other programs, or
lower `--max-ctx`. The speed also depends on what you ask (see "Will I see the same speed?" above).

Other programs that draw on the same GPU (games, videos, animated pages) also take time from the engine; the monitors themselves cost
little. The reporter of [#18](https://github.com/yasakani-dev/yasakani/issues/18) measured this on an RTX 5070 Ti (0.1.5, Windows 11,
both monitors on the card at 1440p 180 Hz and 1080p 240 Hz, hardware-accelerated GPU scheduling off, other programs closed, median of
5 runs; [the table](https://github.com/yasakani-dev/yasakani/issues/18#issuecomment-6934692757)):

| On the screen | One pass, look-ahead off | Prose / thinking / code / CSV to JSON (tok/s) |
|---|---|---|
| Idle desktop | 14.16 ms | 146 / 157 / 221 / 632 |
| A video in Chrome, full screen | 14.63 ms | 141 / 152 / 217 / 623 |
| A WebGL page redrawn every frame, maximized | 18.68 ms | 110 / 118 / 165 / 473 |

With the monitors on the CPU's integrated graphics a pass took 14.23 ms. These are the reporter's numbers; we have not measured this
on other cards. By the reporter's later measurement, the 1.5 to 2.1 times of the first report probably came from such programs, not
from the monitors. If a program keeps the card busy, close it, or, on a PC with integrated graphics, let it draw there.

## It only thinks and never answers

By default the model thinks before it answers, at its longest thinking setting. On requests that ask for a certain number of characters, or for
a whole program in one go, the thinking alone can take thousands to tens of thousands of tokens. The output limit (max tokens) includes the
thinking, so with a small limit the request ends in the middle of the thinking and the answer is empty. Raise the limit (the server's default is
32,768) or shorten the thinking: `/effort medium`, `/effort low` or `/think off` in `yasakani chat`; `reasoning_effort` (`medium`, `low`) or
`{"enable_thinking": false}` in `chat_template_kwargs` through the API.

## Is a long prompt reused?

When a request repeats the previous prompt, continues the conversation or sends back a tool result, the standard server reuses what
it has read and reads only the new part: after a 229,376-token document the first token came after 0.35 to 0.38 s, against 116 s the first
time (measured with 0.1.1 to 0.1.4). Since 0.1.4 the same document followed by a different question is reused too (0.37 s), when the
document comes before the question and no other request comes in between. The prompt
is read from the start after an unrelated request, when thinking is switched on or off, after an image, and when a prompt under 8,192
tokens changes part-way. `usage.prompt_tokens_details.cached_tokens` shows how many tokens were reused; the details are in
[MEASUREMENTS.md](MEASUREMENTS.md).

Since 0.2.0, with the new prompt reading ([below](#prompt-reading-020)), more states are saved inside a long prompt the first time it is
read (prompts of 32,768 tokens and more in our measurements). When a known prompt then changes in the middle, it is read again from the last saved state before the change, not from the start.
0.1.7 read such a prompt from the start.

<a name="prompt-reading-020"></a>

## Prompt reading in 0.2.0

### How fast is reading a prompt in 0.2.0?

On the RTX 5090, about 1.04 to 1.25 times as fast as 0.1.7, more for longer prompts. 0.2.0 reads the prompt layer by layer in larger
pieces, computes the output layer only where it is needed, and saves more states inside a long prompt (32,768 tokens and more in our measurements). With the quality model it read
an 8,192-token prompt at 5,153 input tokens per second (0.1.7: 4,248) and a 32,768-token prompt at 4,647 (0.1.7: 3,774).

Time to read a prompt the first time (nothing reused). One RTX 5090, default settings, context 69,632, one request at a time, thinking
off unless the row says on, median of 2 runs per side:

| Prompt (tokens) | Quality model: 0.1.7 → 0.2.0 (input tok/s) | Time | Ratio | Fastest model: 0.1.7 → 0.2.0 (input tok/s) | Ratio |
|---|---|---|---|---|---|
| 256 | 2,728 → 2,846 | 93.8 → 89.9 ms | 1.04 | 2,906 → 3,020 | 1.04 |
| 2,048 | 4,398 → 5,007 | 465.6 → 409.1 ms | 1.14 | 4,434 → 4,959 | 1.12 |
| 8,192 | 4,248 → 5,153 | 1.929 → 1.590 s | 1.21 | 4,259 → 5,102 | 1.20 |
| 32,768 | 3,774 → 4,647 | 8.683 → 7.051 s | 1.23 | 3,774 → 4,601 | 1.22 |
| 65,536 | 3,288 → 4,081 | 19.93 → 16.06 s | 1.24 | 3,277 → 4,051 | 1.24 |
| 8,192, thinking on | 4,223 → 5,128 | 1.940 → 1.597 s | 1.21 | 4,210 → 5,101 | 1.21 |
| 32,768, thinking on | 3,763 → 4,703 | 8.707 → 6.968 s | 1.25 | 3,744 → 4,660 | 1.24 |

A known 32,768-token prompt changed at 25, 50 or 75% of its length and read again (same conditions):

| Change at | Quality model: 0.1.7 → 0.2.0 | Ratio | Tokens not read again | Fastest model: 0.1.7 → 0.2.0 | Ratio |
|---|---|---|---|---|---|
| 25% | 8.701 → 5.776 s | 1.51 | 6,144 | 8.725 → 5.840 s | 1.49 |
| 50% | 8.702 → 4.238 s | 2.05 | 14,336 | 8.696 → 4.260 s | 2.04 |
| 75% | 8.707 → 2.070 s | 4.21 | 24,576 | 8.732 → 2.074 s | 4.21 |

The answers are the same as with 0.1.7 ([above](#are-the-answers-the-same-as-the-original-models)). The speed of writing the answer
does not change: "speed" in the tables of the README is still the speed of writing the answer, without the time to read the prompt. On
the 15 tasks of the speed tables (thinking off, temperature 1.0, one request at a time, the sum of all tasks, one uncounted pass and then
3 runs per version) 0.2.0 with its default settings was +0.37% (quality model) and +0.45% (fastest model) against 0.1.7, with the same
tokens. With the [edit copy](#edit-copy-020) off it wrote at 622.4 tok/s against 621.5 for 0.1.7 with the quality model (+0.15%) and
716.8 against 718.3 with the fastest model (-0.21%). The details are in [MEASUREMENTS.md](MEASUREMENTS.md#measurements-020).

### What do the lines `prompt reading: ...` at start mean? What do `--prefill` and `YASAKANI_PREFILL` do?

At start one line says which prompt reading is used and why, for example:

```
prompt reading: new (setting auto: 32607 MiB card)
prompt reading: new (setting auto: 24564 MiB card, 4202 MiB free beside --max-ctx 131072; --prefill old for the 0.1.7 one)
prompt reading: as in 0.1.7 (setting auto: 24564 MiB card and no --max-ctx; --prefill new or YASAKANI_PREFILL=new to try it)
```

`--prefill` (on `chat`, `serve`, `bench` and `selfcheck`) or the environment variable `YASAKANI_PREFILL` chooses it:

* `auto` (the default): the new reading on a card with 30,000 MiB or more, else the reading of 0.1.7. Since 0.3.4 also on a card of
  22,000 to 30,000 MiB when the context is given with `--max-ctx` and at least 1,800 MiB of GPU memory stays free beside it for the work area
  (from a user's measurement in
  [#24](https://github.com/yasakani-dev/yasakani/issues/24); we have not run it on a real 24 GB card). A card whose memory nvidia-smi
  cannot tell also reads as 0.1.7 did.
* `new`: the new reading on any card. Under 30,000 MiB this is experimental ([below](#prompt-reading-smaller-cards)).
* `old`: the reading of 0.1.7 on any card, including its passes of 64 tokens. On the RTX 5090, `--prefill old` gave the tokens, the
  states, the number of saved states and the GPU memory of 0.1.7.

A wrong value of `YASAKANI_PREFILL` stops before the model is loaded. `yasakani info` shows the setting and what it chooses on each GPU,
and `yasakani report` includes the line.

<a name="prompt-reading-smaller-cards"></a>

### I have an RTX 30 / 40 card or a 16 GB card. What changes for me?

By default nothing changes in prompt reading: cards with less than 30,000 MiB read prompts exactly as 0.1.7 did (since 0.3.4 except a
card of 22,000 MiB or more with `--max-ctx` given, [above](#prompt-reading-020)). We checked this on the
RTX 5090 shown to the package as a 24,564 MiB card (quality model, context 262,144) and as a 16,384 MiB card (fastest model, context
98,304) with a 32,768-token prompt (one run each): the peak GPU memory was within 9 MiB of 0.1.7, the output was the same, and the prompt
took 9.04 s (24,564 MiB) and 9.05 s (16,384 MiB) against 9.00 s and 9.02 s with 0.1.7. An earlier run also found the same tokens, state
and number of saved states. For the rule of 0.3.4 we set the RTX 5090 up as a 24,564 MiB card again (quality model): with
`--max-ctx 131072` `auto` took the new reading, `selfcheck` gave the same lines as with `--prefill old`, and a prompt of 32,000 tokens
was read in 7.2 s against 8.5 s with the same 16 output ids (one run each); with `--max-ctx 262144` `auto` stayed with the reading of
0.1.7. The
[edit copy](#edit-copy-020) is used on every card; we measured it on the RTX 5090 only.

`--prefill new` (or `YASAKANI_PREFILL=new`) on such a card is experimental:

* The package keeps the prompt sizes of 0.1.7 and chooses the context with 1,800 MiB more left free; one line says so. When that would
  cost more than half of the context (on 16 GB cards), it does not leave the room, says so, and the engine decides by its own check. A
  context you give with `--max-ctx` is used as it is, and a note says how much free memory the new reading needs.
* The engine makes the work area of the new reading (about 579 MiB at these sizes) only when the GPU has that much free plus 1 GiB.
  Otherwise it prints `[q35_layer] capacity fallback; no context reduction` once and, until the model is closed, reads prompts pass by
  pass without the work area (the other changes of 0.2.0 stay on, and the output is the same). This is not an error.
* We set up the RTX 5090 to look like a smaller card in both ways: the package saw the smaller memory, and another process held GPU
  memory so that only the free memory of that card was left. A 32,768-token prompt, one run each, the same output as 0.1.7 every time:
  * As a 24 GB card (24,564 MiB, quality model) the package chose a context of 229,376 instead of 262,144, and the engine made its work
    area and read the prompt in 7.56 s (0.1.7: 9.00 s). With 700 MiB less free it found no room and read the prompt without the work area.
  * As a 16 GB card (16,384 MiB, fastest model) keeping the room would have cost more than half of the context, so the context stayed
    98,304. The engine found no room and read the prompt without the work area. So at the automatic context a 16 GB card does not use the new
    reading; a smaller `--max-ctx` leaves room for it (not measured).
* We have not run it on a real RTX 30 or 40 card, or on a real 16 GB or 24 GB card. A user ran it on an RTX 3090 Ti (24 GB, Windows 11,
  0.3.2, quality model, `--max-ctx` given): 1.39 to 1.40 times as fast at 8,224 and 32,797 tokens, the same self-check numbers; the work
  area was made at contexts up to 131,072, and at 262,144 the engine printed the fallback line
  ([#24](https://github.com/yasakani-dev/yasakani/issues/24); the reporter's numbers). The context the package chooses itself with
  `--prefill new` has not been tried on a real card. How much free memory the driver reports differs between
  PCs, so we cannot say that it works on a given card. Results go in [GitHub issues](https://github.com/yasakani-dev/yasakani/issues).

<a name="prompt-reading-memory"></a>

### Why does 0.2.0 use more GPU memory on the RTX 5090?

The new prompt reading needs a work area. The engine makes it at the first long prompt and keeps it until the model is closed. On the
RTX 5090 with the default settings the peak GPU memory was 1,385 to 1,388 MiB higher than with 0.1.7 for the quality model, and 1,476
to 1,479 MiB higher for the fastest model (the peak while reading prompts of 8,192, 32,768 and 65,536 tokens at a context of 69,632).
After the model is loaded and before the first long prompt the difference is about 570 MiB. With the quality model at the full context of
262,144, the peak was 23,595 MiB against 22,218 MiB with 0.1.7 (above the level before the start, one run each). Details:
[MEASUREMENTS.md](MEASUREMENTS.md#measurements-020). The engine makes the area only when the GPU has enough free memory (the area plus 1 GiB);
otherwise it prints `[q35_layer] capacity fallback; no context reduction` once and reads prompts without the work area. `--prefill old` (or
`YASAKANI_PREFILL=old`) gives back the reading and the GPU memory of 0.1.7.

## How do I connect another app?

Any app that can talk to an OpenAI-compatible server: set the address to `http://127.0.0.1:8765/v1` and the model name to `qwen3.8-27b` (any
API key will do). If it does not connect:

* An app inside Docker, or anything outside this computer: by default only connections addressed to `127.0.0.1` are accepted. Start the server
  with `yasakani serve --host 0.0.0.0 --api-key YOUR_KEY` and give the app that key.
* Apps that call the server straight from a web page: calls from other pages are refused, for safety.
* Apps that depend on tool / function calling (coding agents, for example): supported since 0.1.1. The server takes `tools` with
  `tool_choice` `auto` or `none` and returns the calls as `tool_calls`. Since 0.1.4 it also accepts `required` or a named function, but does
  not enforce the call: the model decides, and an answer without a call comes back with `finish_reason` `stop`. The old `functions` field is
  answered with 400. Version 0.1.0 answered every request with `tools` with 400: if you use 0.1.0, put a newer quickstart (0.1.1 or later) into the same folder
  and run `START.cmd` again (the model is not downloaded again). Tool calling is new and has been tested far less than plain chat.
  An agent screen is in the harness (see "Development and support" in the [README](../README.md)).

## Several requests at once (experimental)

All numbers in this section were measured on one RTX 5090 (32 GB) with a development build of 0.1.7, November 17 and 18, 2026, except where 0.2.0 is named. Full conditions: [MEASUREMENTS.md](MEASUREMENTS.md#several-requests-at-once-experimental).

### How do I turn it on?

Start the server with `--parallel N` (or set `YASAKANI_PARALLEL=N`):

```bat
yasakani serve --model-dir D:\models\qwen38-quality --parallel 4
```

The default is 1: one request at a time, the same path as before. The first time the server starts with an engine library, a GPU and a driver, it runs N short requests at once and compares their tokens with the same requests sent one by one. If they differ, it serves one request at a time and says so. The result is saved, so later starts skip the check. `yasakani selfcheck --parallel N` runs the same check on demand. `yasakani bench --concurrency N` measures N requests at once on your GPU.

### When does it help, and when not?

It helps when several different requests write new text. It does not help with edits, where most of the answer is a copy of the input. One request alone checks up to 64 copied tokens per pass where the check on the card opened the wide copy passes, but a shared pass carries at most 16 rows for all requests together. We think this is why.

Fastest model, thinking off, temperature 1.0, top-p 0.95, top-k 20, seeds 1 to 16, `--parallel 16 --max-ctx 16384`. Each task was sent 16 times one by one, then N of those at once. The number is (all output tokens / time from the first send to the last end) at once, divided by the same for one by one. Mean of 2 runs, one uncounted request first.

| Task | N = 4 | N = 16 |
|---|---|---|
| Story: Chinese dialogue | 2.69 | 6.56 |
| Story: Chinese science fiction | 2.73 | 6.12 |
| Story: English mystery | 2.48 | 4.11 |
| Translation: Markdown | 2.18 | 3.02 |
| Code: Python | 2.14 | 2.98 |
| Code: CUDA | 1.96 | 2.63 |
| Translation: English to Chinese | 2.00 | 2.54 |
| Code: TypeScript | 1.67 | 2.53 |
| Structured: SQL | 1.94 | 2.19 |
| Translation: Chinese to English | 1.88 | 2.17 |
| Structured: CSV | 1.85 | 2.03 |
| Structured: JSONL | 1.17 | 1.11 |
| Long JSONL (own task) | 0.95 | 0.92 |
| Code edit (own task) | 0.54 | 0.53 |
| JSONL edit (own task) | 0.55 | 0.50 |
| Mixed: 4 code edits, 4 JSONL edits, 8 published prompts | | 0.99 (16 at once) |

Below 1 means slower than one by one. With N = 2 the edits were 0.42 and 0.59. Each single request is always slower than when it runs alone. The quality model was not measured this way.

### Are the answers the same as one at a time?

Yes, under two conditions. The request has its own `seed`, and its prompt is read the same way, which `timings.snap_mode` shows (0 from the start, 1 continuing the request before it, 2 from a saved state). With a stop string the answer is the same up to the stop. We compared the token ids, the finish reason, tool calls, text, thinking and the token counts.

### How much GPU memory does it need, and how long can the context be?

All requests get the same context length. Without `--max-ctx` the server picks it from the free GPU memory and prints it at start. It reserves N + 1 contexts of 22,400 bytes per token and about 256 MiB per request. For example, one 65,536-token context with its request is about 1,656 MiB. If not even 16,384 tokens fit, the server stops and names the largest N that fits.

Measured on the 32 GB RTX 5090, with the desktop using about 1.8 to 1.9 GB:

| N | Quality model | Fastest model |
|---|---|---|
| 1 | 262,144 | 262,144 |
| 2 | 196,608 | 229,376 |
| 4 | 114,688 | 131,072 |
| 8 | 65,536 | not measured |
| 16 | 16,384 | not measured |

With the quality model all four ran under load without running out of memory. The memory used stayed 297 to 524 MiB below the server's estimate.

### What do `--keep-turn` and `--max-wait` do?

When an answer ends with tool calls, the client usually sends the next turn with the tool results soon after. `--keep-turn` keeps that conversation's place for a while. The next turn then continues where the answer stopped and does not read the whole conversation again. While its place is kept, other requests cannot use it.

`--max-wait` limits how long a new request waits because of kept places and next turns. After that time it takes the next kept place, and that conversation reads its prompt again. The time it spends waiting behind requests that came before it does not count.

Both default to `auto`. `auto` keeps a place about as long as reading the conversation again would take, from 1 to 30 seconds (1 second for a short conversation, about 4 seconds for 16,000 tokens). The wait limit is that time plus 0.25 seconds. So a kept place never holds others back much longer than the time it saves. Use `--keep-turn off` to keep nothing, or give seconds.

Comparison on one load: fastest model, `--parallel 4 --max-ctx 65536`, thinking off. 6 agents read a file in 8 parts through a tool (the tool took 0.05, 0.5, 3 or 15 seconds, 2 times each), so their conversations grew to 8,719 to 40,957 tokens. At the same time 3 chat users sent short streamed requests every 1 to 6 seconds. Mean of 3 runs.

| Setting | All agents done | Conversations read again | Chat, first token, p95 | Total tok/s |
|---|---|---|---|---|
| keep nothing (`off`) | 178.1 s | 41 times, 154.0 s | 21.8 s | 22.6 |
| keep 2 seconds, no wait limit | 123.3 s | 24 times, 84.6 s | 25.8 s | 31.7 |
| `auto` (default) | 94.2 s | 19 times, 45.9 s | 17.2 s | 42.3 |
| `auto`, after the last fix of the wait limit | 100.6 s | 23 times, 56.5 s | 15.6 s | 40.6 |

The last row was measured later the same day, after the wait limit was fixed to count all waiting caused by next turns. The agents got 5% slower and the chats faster. The chat p95 stays high in all settings because reading a long prompt pauses everything. In a second load (`--parallel 2 --max-ctx 32768`, agents that call a tool 30 times in a row and send each next turn 20 ms after the answer, plus new requests) the wait limit cut the longest wait of a new request from 7,977 ms to 1,529 ms (fastest model) and from 7,448 ms to 1,322 ms (quality model).

### What happens when every place is busy?

New requests wait in a queue. When more than `--max-queue` requests wait (default 64), the server answers 503 with `Retry-After: 2`. `/health` shows `parallel`, `slots_busy`, `slots_idle` and `queued`. `timings` shows `queue_ms` (the time the request waited).

### Can I send images?

Yes. A request with images runs alone: the other requests pause until it is done. In our tests, all requests with images gave the same answers as one at a time.

### What happens when a client disconnects?

The request stops and frees its place. This works while it waits, while it streams, and while it generates without a stream. A request with a second part (after a thinking budget or a loop cut) keeps its place for that part for 2 seconds after the client got the first part, and up to 30 seconds for a slow client.

### Do seeds work?

Yes. A request with a `seed` gets the same tokens as alone. A request without a `seed` takes the next seed in the order requests arrive, as in one-at-a-time serving. Its result therefore depends on what else was sent.

### What was tested, and what was not?

Tested on the RTX 5090 only, with both models:

* One at a time (`--parallel 1`), with a development build that has the same GPU code as 0.1.7: the same tokens as 0.1.5 on 35 prompts per model, and within 1% of its speed on 15 tasks (thinking off).
* Several at once: about 35,000 requests compared with the same conversation sent alone. None differed. This includes requests that join and leave mid-run, contexts up to 229,376 tokens, tool calls, images, cancellation and engine failures.
* A 60-minute run with 10,178 mixed requests (text, tools and images, fastest model, N = 8): no difference, nothing stuck, and memory did not keep growing.
* With 0.2.0 (fastest model), with the [edit copy](#edit-copy-020) on (the default) and again with it off: requests that join and
  leave (`--parallel 4` and `8`, 123 requests), cancellation (21 requests) and engine failures (10). Every compared request gave the
  same output as the same conversation sent alone. A 30-minute run with mixed requests (text, tools and images, N = 8), edit copy on:
  5,473 requests, 4,293 the same as the conversation sent alone, 1,157 read another way with the same output, 23 not sent, none
  different; none stuck; the server's GPU memory grew by 6.0 MiB. The same run with the edit copy off: 5,573 requests, 4,364 the same,
  1,185 read another way with the same output, 24 not sent, none different; none stuck; 6.0 MiB. The shared passes of several requests
  do not use the edit copy, but a request that runs alone on the server (also with `--parallel N`, and every request with images) does.

Not tested:

* Other GPUs, including 24 GB and 16 GB cards. On them the start-up check decides.
* Speed with the quality model, and with thinking on.
* Linux with `--parallel`.
* Two GPUs.
* The new prompt reading of 0.2.0 when `--parallel N` chooses the contexts itself. The contexts then take the free GPU memory, and the
  engine may find no room for the work area of the new reading and read prompts without it. With a `--max-ctx` that leaves room
  (`--parallel 4 --max-ctx 16384` on the RTX 5090) it used the new reading.

Known limits:

* The 16-row pass is shared, so each request is slower than alone.
* Reading a prompt pauses all others. Reading 66,416 tokens took about 20 seconds (the development build of 0.1.7; 0.2.0 reads
  prompts faster on the RTX 5090, [above](#prompt-reading-020)).
* The next turn after tool calls must come within the kept time (1 second for a short conversation). Later, it reads the conversation again. The answer is the same.
* The wait limit counts only waiting caused by kept places and next turns. A next turn that runs alone (because no place is free) is not stopped for a new request, so a new request can wait longer than the limit.
* Over long runs the GPU memory of the whole card rose by up to 71 MiB. In a 30-minute run with a development build the server's own GPU memory did not change.

<a name="inkling-2-mini"></a>

## Inkling-2 Mini (experimental, new in 0.3.1)

Inkling-2 Mini is a language model by Thinking Machines Lab, published on November 20, 2026. Since 0.3.1 YASAKANI runs it as an
experimental model, with a second engine library. Since 0.3.2 it is choice 4 of the model question of `START.cmd` (or `START.cmd --model inkling2`):
the quickstart downloads our converted copy from Hugging Face (about 15.9 GB) and checks it like the other models. How to download, convert
and start it by hand: [README](../README.md#inkling-2-mini).

### Which cards can run it, and why 24 GB?

We ran it on one RTX 5090 (32 GB) under Windows 11 only. With the default context of 32,768 tokens the engine allocated 22.7 GB of GPU
memory (2.15 GB of it the key/value cache), and 23.8 GB with the drafter loaded (the default). The GPU as a whole had 25.0 GB in use
without and 27.2 GB with the drafter (as the package reports it at the start); with the converted weights of `--scheme quality` and their
4-bit drafter the engine allocated 18.9 / 19.5 GB, and the GPU had 21.2 / 21.9 GB in use. `--draft off` does not load the drafter. Cards under 24 GB cannot hold it. We have not run it on a 24 GB card (RTX 3090, 3090 Ti or 4090). Users ran 0.3.1 on an RTX 3090
and 0.3.3 on an RTX 3090 Ti, both under Windows 11: `selfcheck` passed, and on the RTX 3090 Ti a 131,072-token prompt fit with the fast
path ([#29](https://github.com/yasakani-dev/yasakani/issues/29); the reporters' numbers are in the
[README](../README.md#inkling-2-mini-033)). With the desktop also on that card
there is less room, and a smaller `--max-ctx` needs less memory. Since 0.3.2, without `--max-ctx` the package takes a shorter context when
32,768 tokens do not fit the free GPU memory, and stops before loading when not even 8,192 fit; a line at start says which. On Linux there are two ways, both built from source and not part of an official release
([BUILDING.md](../BUILDING.md#linux-inkling-2)): the engine inside `libyskn.so` for the downloaded folder (by @audreyt in
[#28](https://github.com/yasakani-dev/yasakani/pull/28), [#30](https://github.com/yasakani-dev/yasakani/pull/30) and [#32](https://github.com/yasakani-dev/yasakani/pull/32); it opens 32,768 tokens unless `--max-ctx` is given,
without this check), and `libyskn_inkling2.so` for a converted folder ([#33](https://github.com/yasakani-dev/yasakani/pull/33) by @Albrod). We ran both on one
RTX 5090 under WSL2 only.

### How fast is it, and under which conditions?

The fast path is on by default on RTX 50 cards since 0.3.3 and on RTX 30 and 40 cards since 0.3.4. On one RTX 5090 (Windows 11, November 23, 2026, the `--scheme quality`
folder, one request at a time, greedy, a text of 53 public source files without repetition) it read prompts of 4,096, 32,768 and 131,072
tokens at 1,826, 1,755 and 1,377 tokens/s (0.3.2: 900, 633 and 314), and generated 268, 351, 183 and 132 tokens/s with `--draft auto`
after 256, 4,096, 32,768 and 131,072 tokens (0.3.2: 230, 257, 88 and 58). The conditions, the numbers without the drafter and the
comparison with llama.cpp: [README](../README.md#inkling-2-mini-033).

The numbers below are those of the standard path (as in 0.3.2). One RTX 5090, Windows 11, November 22, 2026, the published weights, one request at a time:

* Generation with greedy decoding, thinking off, 256 tokens: `yasakani bench --draft off` 161.2 tokens/s (the mean of its 4 prompts,
  median of 3 runs), and 212.5 tokens/s with the default `--draft auto`. With the converted weights of `--scheme quality` and their
  4-bit drafter: 203.8 and 225.9 tokens/s in the same job. Speeds by kind of text with and without the drafter:
  [README](../README.md#inkling-2-mini).
* Terminal chat with thinking on at low effort and `--draft auto`: 155.0 tokens/s (one question in Japanese, 375 tokens). The server,
  streamed: 173 tokens/s (a short question in Japanese, 145 tokens with thinking), the first piece 0.08 s after the request.
* The speed drops as the conversation grows (measured with the engine before the changes of 0.3.1): 6.0 ms per token at position 88,
  6.7 ms at 536, 7.3 to 7.5 ms from position 1,024 on (133 to 138 tokens/s).
* Prompt reading through the package: 1,063 tokens/s for a prompt of 1,079 tokens, 961 tokens/s for 4,900 tokens (median of 3 runs each,
  from an empty context). A prompt that starts like the one before (the next turn of a chat) reads only the new part.
* Start until ready (files in the Windows file cache): published weights 6.4 s without the drafter and 8.5 s with it; converted
  weights 2.6 s and 3.0 s (their tensors are checked at load, 0.8 s).
* Prompt reading with the converted weights through the package: 959 tokens/s for 1,079 tokens, 878 tokens/s for 4,900 tokens.

`yasakani bench --model-dir <folder>` measures it on your PC (greedy, thinking off, 256 tokens per test). We have not measured other cards.

### How were the answers checked against the published model?

Two ways, both on the RTX 5090 with the published weights. Our engine gives the 30 tokens that the official text test of the model's code
in the transformers library expects for its prompt; `yasakani selfcheck` runs that same check on your PC. And we compared the engine with
our own CPU implementation of the model: the hidden states agreed within 1e-4 and the same experts were chosen. This measures agreement
with the published model, not accuracy on tasks; we have not measured task accuracy.

<a name="inkling-2-mini-fast-path"></a>

### What does the fast path of 0.3.3 change?

The way the engine computes, not the model files. It keeps the key/value cache of the 8 layer visits that attend to the whole context in
8 bits (MXFP8), splits long attention into parts, reads prompts with a new attention and, on RTX 50 cards, computes the experts of a
prompt on the 4-bit (NVFP4) tensor cores. Since 0.3.4 the default `--fast auto` turns it on for RTX 30, 40 and 50 cards (compute
capability 8.6, 8.9 and 12.0; 0.3.3: RTX 50 only) and off on other GPUs; on an RTX 30 or 40 card it runs without the 4-bit part. We made
this change after a user ran 0.3.3 on an RTX 3090 Ti ([#29](https://github.com/yasakani-dev/yasakani/issues/29)): `selfcheck` passed
with the fast path on and off, and generation and prompt reading were faster with it on (the reporter's numbers are in the
[README](../README.md#inkling-2-mini-033)). We have run it on the RTX 5090 only, and no RTX 40 card has been reported. On RTX 30 and 40
cards the default output therefore changes with 0.3.4 (below); `--fast off` (or `YASAKANI_INK2_FAST=off`) computes as before, that is
as 0.3.2 did. `selfcheck` compares the standard path unless `--fast` is given.

The output is close to that of the standard path, but not the same. At 128 positions after 4,096 and 32,768 tokens of one text, the most
likely next token was the same at 126 and 123 positions (mean KL 0.0035 and 0.0059); on another text at 128 of 128 after 32,768 tokens.
Greedy output parts from that of the standard path after some tokens. The drafter and copy decoding do not change the output of the fast
path. It needs less GPU memory at long contexts: 20.9 instead of 28.5 GiB in use at 131,072 tokens with the drafter (RTX 5090).
[Details](../README.md#inkling-2-mini-033)

### What are the converted weights of `--scheme quality`?

The converter can write the weights again in fewer bits (`inkling2_convert.py ... --scheme quality`, [README](../README.md#inkling-2-mini)). We publish this
conversion on Hugging Face, so you can download it instead of converting: [yasakani-dev/Inkling-2-Mini-YASAKANI-quality](https://huggingface.co/yasakani-dev/Inkling-2-Mini-YASAKANI-quality)
(15.9 GB with the image file of 0.3.5). What the conversion does:
most dense matrices in 4 or 8 bits, and the 8-bit experts of layers 20 to 23 re-packed to 4 bits, in one `.yskn` file of 15.05 GB
instead of 19.5 GB ([FORMAT](FORMAT.md)). Against the published model on 43 text windows (14,024 positions: the model's own thinking and
answers, English and Japanese prose, code) the KL divergence was 0.0110, and the most likely next token was the same at 96.0 % of the
positions. The drafter is written again in 4 bits as well (`mtp_d4.yskn`, 209.7 MB instead of 635.8 MB). The engine multiplies these
matrices as they are stored (interface version 3 of the library), so the folder is faster and needs less memory: `yasakani bench --draft
off` 203.8 against 161.2 tokens/s for the published weights in the same job on the RTX 5090, and 18.9 instead of 22.7 GB allocated at a
context of 32,768 tokens (with the drafter 19.5 instead of 23.8 GB). The conversion takes about 4 minutes on the CPU.
`yasakani verify` checks every file and every tensor. `yasakani selfcheck` compares the greedy output of three short prompts (English,
Japanese, code) with the tokens that a reference implementation of the model gave on the same converted weights, decoded exactly as the
engine multiplies them. At every one of the 48 tokens the chosen token leads the next best one by at least 0.11 in the logits, so small
differences of arithmetic on other GPUs should not change the result (only the RTX 5090 was tested). The conversion on the CPU gives the same
file every time, and the converter writes these tokens into `selfcheck.json` when its file has the SHA-256 of ours
(`44ef7ea5eade7a1e6cfb4f7ad3f7568a701ce9c621f7e4d00d38b624f4de138d`). `START.cmd --model-dir <folder>` then opens it after the
same checks as for the published weights. A file that differs gets no expected tokens; open that folder with `--skip-checks`.

### How do thinking on and off work for this model?

The model writes its reasoning in a thinking block before the answer, and its chat template takes a thinking effort (none, minimal, low,
medium, high, xhigh, max or a number from 0 to 0.99; the template's default is 0.9). In terminal chat, `/effort low` shortens the thinking;
in the API, `reasoning_effort`. The model card says that even the lowest effort does not fully turn the thinking off. Our "thinking off"
(`--no-think`, `/think off`, or `{"enable_thinking": false}` in `chat_template_kwargs`) is therefore our own setting: it sets the effort to 0
and starts the reply with an answer block, so that no thinking is written. The speed numbers with thinking off use this setting.

### Why is it slower than the Qwen models?

The work per token is different. Inkling-2 Mini has 24 layers, but a token passes through 48 layers, because layers 6 to 17 are visited
three times. In each visit the layer chooses 8 of its 128 experts for that token, so the weights that are read change from token to token.
The look-ahead also differs: with the Qwen models a small draft model proposes tokens that the main model checks several at a time,
and in edits whole copied spans are checked in one pass. Inkling-2 Mini uses the drafter that comes with the model (`--draft auto`, see
the [README](../README.md#inkling-2-mini)); it checks at most 8 tokens per pass and helps most with text that is
easy to guess. The output is the same with and without it. The speed tables of the Qwen models therefore do not apply to this model.

### What are its licence and its use policy?

The weights are published under Apache-2.0. The model card also names the
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/), which applies to the weights and to modified
versions. The model folder made by the converter has a `NOTICE` file that says where the files come from, the licence, the policy and what
the converter changed. Read them for your own case; this FAQ is not legal advice. YASAKANI is not made by, and not endorsed by, Thinking
Machines Lab.

### Does it support tool calls?

Since 0.3.1, in `yasakani serve`: the server accepts `tools` and `tool_choice` (`auto`, `none`, `required` or the name of a function) and
returns the calls as `tool_calls`, with and without streaming. With `required` (also among several tools) or a named function the model
can only write one of the offered names as the call, and the reply ends after that one call. Terminal chat has no tools. The limits (the context, the thinking budget and
the GPU memory for dropped replies) are in the [README](../README.md#inkling-2-mini).

### Can Inkling-2 Mini read images?

Yes, new in 0.3.5: still images, as the published model reads them. The image tower runs on the GPU inside the engine
library, on Windows and in both Linux builds from source ([BUILDING.md](../BUILDING.md#linux-inkling-2)). A converted folder needs one more file, `inkling_2_mini_vision.yskn` (637 MB, the published image tower unchanged). The
quickstart of 0.3.5 downloads it with the folder, and for a folder of 0.3.2 to 0.3.4 it asks once whether to add it
(about 0.64 GB; Enter alone downloads it, `n` keeps text only; later `START.cmd --image-file`). With the file, the quickstart page shows an image button, `yasakani serve` takes OpenAI `image_url` parts with `data:image/...;base64` URIs, and
`yasakani chat` takes `--image PATH`.

An image of H × W pixels becomes ceil(H / 40) × ceil(W / 40) image tokens, for example 576 for 1280 × 720. Images above 1,024 image
tokens (`--vision-max-image-tokens`) are shrunk to fit; that shrink is this project's choice, not part of the official method. The
image tower takes about 0.75 GB more GPU memory with the defaults (measured on the RTX 5090 at 32,768 tokens); `--vision off` does not load it, and then
the model is text only as before. Encoding one 1280 × 720 image took about 12.7 ms on the RTX 5090. Requests without images give the same
output as before. Not supported: video, several frames of one file, images in system or assistant messages, and images from URLs (they are never fetched).
Details: [python/README.md](../python/README.md#inkling-2-mini-image-input).

### What is not supported yet?

Video, several requests at once (`--parallel`), tools in terminal chat, `bench --edit`, a prebuilt Linux release (on Linux it is built
from source only), and any GPU other than the RTX 5090 (not tested). This model is experimental in 0.3.1.

## Can I use other models, for example Qwen4?

Not in this release: it runs Qwen3.8-27B and, since 0.3.1, Inkling-2 Mini ([above](#inkling-2-mini)). Each has its own converter in
[tools/convert/](../tools/convert/) and its own engine library, so another model needs changes in both.
Whether other models can be supported is decided one by one, and nothing is promised.

## Does it send my prompts anywhere?

No. The Python package makes no outgoing network connections (it only listens, on `127.0.0.1` unless you choose otherwise), and the engine library
imports no networking functions. After the download it needs no internet connection. The one exception is the update check: `yasakani update`
when you type it, and the quickstart's check for a new version, each send one request to GitHub that holds no prompt and nothing about your PC
([below](#how-does-the-update-work)). Details and the checks behind this statement are in [SECURITY.md](../SECURITY.md).

## How does the update work?

Since 0.1.7, the quickstart (`START.cmd`) can update itself; from 0.1.7 to 0.3.4 it offers 0.3.5. From v0.1.6 or older, download the quickstart ZIP of the current release (`yasakani-quickstart-0.3.5.zip`) by hand once and run `START.cmd`; the models are kept.

* **The check.** At every start, at most once an hour (since 0.3.5; 0.3.4 and older: at most once a day), before it shows the menu or
  before it opens the browser page directly (from the second run on,
  `START.cmd` does that without a menu), the quickstart first prints that it asks github.com, then sends one HTTPS request:
  `GET https://api.github.com/repos/yasakani-dev/yasakani/releases/latest`, with the User-Agent `yasakani-quickstart` and nothing else about
  your PC (no ID, no version of Windows, no GPU, no prompt). It waits at most 5 seconds; if GitHub cannot be reached, it says so in one line and
  goes on. Pre-releases are never offered. GitHub sees your IP address, as with any download.
* **Checking at once (new in 0.3.4).** A start within an hour of the last check does not ask again (within a day on 0.3.4 or older),
  and a quickstart that keeps running does not check by itself, so right after a release the **Update** button can be missing, and a
  double-click does not show the menu. The Model tab of the page has **Check for updates**: it makes the quickstart send the same
  single request when you press it, and only then (also with the automatic check turned off). On any version, including 0.3.3 or
  older, you can also open a command prompt in the quickstart folder and run `START.cmd --action update` (or `START.cmd --menu` and
  choose “check for a new version”).
* **Turning it off.** `START.cmd --no-update-check` turns the automatic check off and remembers it (in `quickstart_state.json`);
  `START.cmd --update-check` turns it on again. The environment variable `YASAKANI_NO_UPDATE_CHECK=1` also turns it off. The menu item
  “check for a new version” and `START.cmd --action update` still work when you choose them.
* **Nothing changes without a yes.** When there is a newer version, the quickstart shows the old and new version. Before the menu it
  also shows the release page and asks. When it opens the browser page directly, it asks nothing in the black window; it only prints
  how to update (the button of the page, or `START.cmd --action update`). In the browser page, a bar offers **Update** and **Later**,
  and nothing changes until you press **Update**. The page itself sends nothing to GitHub.
* **What an update does.** It downloads the quickstart ZIP and `SHA256SUMS.txt` of that release, only from
  `https://github.com/yasakani-dev/yasakani/releases/download/<tag>/`, compares the ZIP's SHA-256 with `SHA256SUMS.txt`, and accepts only
  the known start files (no folders, no other files). It copies the current start files to `backup\<old version>\`, puts the new ones in
  place and starts the new quickstart, which downloads its own engine ZIP and checks its SHA-256 as on the first run, then installs the
  Python package again. The models are kept.
* **Going back.** `START.cmd --action rollback` puts the start files of the version before the last update back; on the next start the
  engine of that version is installed again.
* **How far you can trust it.** The release files are not code-signed. The check rests on GitHub's HTTPS and on the release's
  `SHA256SUMS.txt`, which comes from the same release: it catches a broken or swapped download, not a release that was itself replaced on
  GitHub.
* **`yasakani update`** depends on how YASAKANI was installed. In a folder set up by the quickstart it asks GitHub nothing itself: it
  opens a new window that runs `START.cmd --action update` and ends; the quickstart then checks, asks and replaces the files (`--yes`
  skips the question). After a manual installation it asks GitHub once, prints the installed and the latest version, prints what to
  download and the command to run, and changes nothing. `--check` only prints, in both cases. `yasakani serve`, `chat` and `bench` never contact the network.

## Windows warns me about the download. Is that normal?

The release files are not code-signed, so Windows SmartScreen or an antivirus program may warn about them. Compare the SHA-256 checksum of what you
downloaded with the one on the release page, and run `yasakani verify` on the model folder. You can also build `yskn.dll` from the source.

## Can I use it from my own programs?

Yes: it serves the OpenAI chat-completions API on `http://127.0.0.1:8765/v1`. Supported request fields, the way the thinking text is returned and the
loop guard are described in [python/README.md](../python/README.md). It generates one request at a time; `n` above 1
and `logprobs` are not supported. Tool / function calling is supported since 0.1.1 (`tools` with `tool_choice` `auto` or `none`; since 0.1.4 also `required` or a named
function, accepted but not enforced).
Images are accepted since 0.1.2 through the optional image input (a community contribution, [#4](https://github.com/yasakani-dev/yasakani/pull/4); on Linux with llama.cpp
built from source, contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8) and not tried by us; see
[python/README.md](../python/README.md#image-input-optional)).

## Can I use it commercially?

YASAKANI is under the Apache License 2.0, which allows commercial use; Qwen3.8-27B (Alibaba Cloud) and the helper model are published under Apache-2.0
as well. Inkling-2 Mini (Thinking Machines Lab) is published under Apache-2.0, and its model card also names the
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/), which applies to the weights and to modified
versions ([above](#what-are-its-licence-and-its-use-policy)). The licenses, copyright notices and what was converted or re-trained are in [LICENSE](../LICENSE), [NOTICE](../NOTICE) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Read them for your own case; this FAQ is not legal advice.

## Who made the model, and what did you change?

Qwen3.8-27B is the work of Alibaba's Qwen team. YASAKANI is the engine and the file format that run it fast, and the model files are converted,
compressed versions of it. The small helper model that speeds up generation is built on a publicly released model: it was converted, compressed and
re-trained. [NOTICE](../NOTICE) says where each comes from. Inkling-2 Mini is the work of Thinking Machines Lab; YASAKANI runs its
published weights unchanged, or weights that the converter wrote again in fewer bits (`--scheme quality`), and the `NOTICE` file in the
model folder made by the converter says what the converter changed.

## How do I report a problem?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for bugs and test results, and [SECURITY.md](../SECURITY.md) for security problems.
