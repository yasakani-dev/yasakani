# FAQ

日本語は [FAQ.ja.md](FAQ.ja.md)。

## What is YASAKANI?

An inference engine for running **Qwen3.8-27B** on your own Windows PC with **one NVIDIA RTX GPU** (tested on the RTX 5090). It is GPU kernels, a file format for the
compressed model, and a runtime, with an OpenAI-compatible server, a terminal chat, a benchmark and a self-check. One person in Japan built it. It is
not a general-purpose engine: one model, tuned on one card.

## Which model should I download?

There are two conversions of Qwen3.8-27B, and you need only one. The **quality model** (12.37 GB) stays closest to the original model and is the
one recommended for everyday use. The **fastest model** (10.31 GB) is smaller and writes faster in most kinds of task, and is a little further
from the original model. The numbers for both are in the [README](../README.md).

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
it was measured with.

The wide copy passes (since 0.1.4: a pass that copies the input checks up to 64 tokens) are used on compute capability 12.x (RTX 50 series)
only, and were measured on the RTX 5090 only; other GPUs copy with passes of up to 16 tokens. Speed here is (tokens - 1) / the engine's
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
divergence 3-10% larger for the quality model than for our earlier conversion of the same size.

Does the speed-up itself change the answers? It is built not to: speculative decoding only keeps tokens that the main model agrees with.
Up to 0.1.2, while other programs were using the GPU at the same time, a few repeated runs had the same tokens as the others up to one
position and continued differently from there (November 10, 2026: 1 of 190 runs with the default settings, 21 of 432 with speculative
decoding switched off; 256 tokens each). With nothing else using the GPU, all 240 repeated runs were identical. The cause was a missing
synchronization in the kernels that apply the position rotation; 0.1.3 fixes it, and in our tests with the fix it did not appear again.
We still do not promise bit-identical text in every condition; the details are in [MEASUREMENTS.md](MEASUREMENTS.md).

The copying of 0.1.4 was checked against decoding one token per pass: 255 of 255 generations with copying (28 of them with wide passes of
up to 64 rows) gave the same tokens ([TECHNOLOGY.md](TECHNOLOGY.md), chapter 7).

## Which GPUs, which systems?

We test on **Windows 11 and the RTX 5090** (32 GB). Since 0.1.3 the engine is also built to run on other NVIDIA RTX 30, 40 and 50 series cards with
compute capability 8.6 or higher and 16 GB or more: the RTX 3090, RTX 3090 Ti and RTX 4090 (24 GB; both models) and the
RTX 4060 Ti 16GB, RTX 4070 Ti SUPER, RTX 4080, RTX 4080 SUPER, RTX 5060 Ti 16GB, RTX 5070 Ti, RTX 5080 (16 GB; the fastest model only, with a shorter context). The engine is
not optimized for them and we have not tested it on them, so neither its speed nor its operation is guaranteed; they are slower than the
RTX 5090. The quickstart says so and goes on. Since 0.1.4, however, its self-check compares the card's predictions with those of the
RTX 5090, and the quickstart stops when they differ by more than small differences in the computation, or when the card's passes of
different widths do not agree.
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
0.1.6 has a fix for this; it has not yet been tried on a PC with two GPUs. A `CUDA_VISIBLE_DEVICES` you set yourself is kept. See
[What you need](../README.md#what-you-need).

<a name="test-build-017"></a>

## Can I try the wide copy passes on my RTX 30 / 40 card? (0.1.7 test build)

Yes, with the test build [v0.1.7-test1](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.7-test1). It is a pre-release, not
the latest release: the latest release stays 0.1.6, and the quickstart installs 0.1.6. The test build is for owners of RTX 30 and 40
series cards and of RTX 50 cards other than the RTX 5090. It lets them try the wide copy passes of 0.1.4 (a pass that copies the input
checks up to 64 tokens instead of 16; see [technical notes 5.4](TECHNOLOGY.md#54-wide-copy-passes-014)) and tell us how they do. The
release page says how to install it next to 0.1.6 and how to go back.

* **What is opt-in.** On RTX 30 and 40 series cards (compute capability 8.6 and 8.9) nothing changes unless you ask for it: the engine
  copies with passes of up to 16 tokens, as in 0.1.6. `--wide on` (or `YASAKANI_WIDE=on`) with `chat`, `serve`, `bench` or `selfcheck`
  lets the card check the wide passes once. Passes of 17 to 64 tokens must give the same predictions as passes of 16, bit for bit. If
  not, no wide pass is used and `yasakani selfcheck` fails: please report it. A width (32, 48 or 64 tokens) is then used only if one of
  its copy steps takes at most 0.8 × width / 16 times a 16-token step on this card (32: 1.60, 48: 2.40, 64: 3.20), measured while the
  GPU is idle. On RTX 50 cards the wide passes stay on as in 0.1.6, but the same check leaves out a width that is slower on that card.
* **How often it checks.** The result is kept per GPU, driver, engine library, package and model (`wide_check.json` in
  `%LOCALAPPDATA%\yasakani`), so the check runs once. It took 21.8 to 22.9 seconds on the RTX 5090. It runs again after 30 days, with
  `--wide-recheck`, or when during generation the engine went back from wide passes in at least 10 of every 100 (counted over at least
  128 in one run of the program). `--wide off` turns the wide passes off; `--wide force` uses every width without the check, to compare.
* **What was checked.** On the RTX 5090 only. The kernels of the wide passes have been built for 8.6 and 8.9 since 0.1.4, but they have
  never run on such a card. On the RTX 5090 these checks passed: the same answers as 0.1.6; the check keeping every width there; and the
  engine's code for 8.6 and 8.9, compiled for the RTX 5090 when it loads, giving the same predictions in wide passes as in passes of 16.
  That does not show that the machine code of an RTX 30 or 40 card is right, or that the wide passes are faster there: that is what the
  test is for. During generation the engine goes back to 16-token passes when the wide ones turn out slower (it checks this every 4
  wide passes).
* **How to report.** `yasakani report` checks the card again, runs the three edit tasks of `bench --edit` with the wide passes off, with
  the widths the check chose and with every width, then the self-check, and prints a Markdown report: GPU, driver, versions, the check,
  the speeds, and whether the outputs are the same. It took 85 to 89 seconds on the RTX 5090; it takes longer on other cards. Please
  paste it into the issue for the test build on GitHub. It contains no UUID, path, user or computer name. Details:
  [python/README.md](../python/README.md#wide-copy-passes-per-gpu-017-test-build).

## How much memory and disk does it need?

One model folder is a download of about 13 GB (fastest model) or 15 GB (quality model); keep 20 GB free. The context length goes up to 262,144
tokens. With the fastest model and that full context reserved, the engine takes 16.78 GiB of VRAM (27.13 GiB if the key-value cache were kept
in FP16); this is the amount reserved at start, not the peak of the whole setup. 32 GB of system RAM is recommended.

Together with the helper model that speeds up generation, the GPU memory in use at the full 262,144-token context was about 19.1 GiB for the
fastest model and about 21.2 GiB for the quality model (about 14.3 GiB and 16.4 GiB with `--max-ctx 32768`; November 11, 2026).

On a GPU with less than 32 GB, the context is chosen from the free GPU memory each time the model starts (at most 262,144 tokens;
`--max-ctx` sets it yourself). On 16 GB cards only the fastest model fits. The rule is in
[python/README.md](../python/README.md#gpus-other-than-the-rtx-5090).

## It is much slower than the tables

The tables were measured on an RTX 5090; other GPUs are slower, and we have not measured them. Check the GPU memory first. When other programs (games, other LLM servers) use GPU memory and it runs out, there is no error: generation just
becomes several times slower. In that case a warning that the GPU memory is almost full is printed at start. Close the other programs, or
lower `--max-ctx`. The speed also depends on what you ask (see "Will I see the same speed?" above).

A monitor on the same GPU can also cost a lot. In one report ([#18](https://github.com/yasakani-dev/yasakani/issues/18)), an RTX 5070 Ti
that also drove two monitors (one of them at 240 Hz) generated 1.5 to 2.1 times as fast once the monitors were moved to the CPU's
integrated graphics: about half the speed of the RTX 5090, as its memory bandwidth suggests. The tokens per pass were the same; with
the monitors on the card each pass took longer. If your PC has integrated graphics, connecting the monitors to the motherboard leaves
the GPU to YASAKANI (`nvidia-smi --query-gpu=display_active --format=csv` then says `Disabled`). The cause is not known yet, and we
have not measured it on other cards.

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

## Can I use other models, for example Qwen4?

Not in this release: it runs Qwen3.8-27B only. The program that made the two model files ([tools/convert/](../tools/convert/), added on
November 14, 2026) knows the layers of Qwen3.8-27B only, and the engine runs this model only, so another model needs changes in both.
Whether other models can be supported is decided one by one, and nothing is promised.

## Does it send my prompts anywhere?

No. The Python package makes no outgoing network connections (it only listens, on `127.0.0.1` unless you choose otherwise), and the engine library
imports no networking functions. After the download it needs no internet connection. Details and the checks behind this statement are in
[SECURITY.md](../SECURITY.md).

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
as well. The licenses, copyright notices and what was converted or re-trained are in [LICENSE](../LICENSE), [NOTICE](../NOTICE) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Read them for your own case; this FAQ is not legal advice.

## Who made the model, and what did you change?

Qwen3.8-27B is the work of Alibaba's Qwen team. YASAKANI is the engine and the file format that run it fast, and the model files are converted,
compressed versions of it. The small helper model that speeds up generation is built on a publicly released model: it was converted, compressed and
re-trained. [NOTICE](../NOTICE) says where each comes from.

## How do I report a problem?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for bugs and test results, and [SECURITY.md](../SECURITY.md) for security problems.
