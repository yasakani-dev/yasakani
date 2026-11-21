# yasakani (Python package)

Run **Qwen3.8-27B** on one **NVIDIA RTX GPU** under **Windows** with the YASAKANI engine library (`yskn.dll`). It is tested on the RTX 5090.

This package is the part you talk to: a local **OpenAI-compatible server**, a **terminal chat**, a **benchmark**, and two checks that tell you
whether your files and your installation are the reference ones. The engine itself is the DLL; the package loads it, applies the
default settings the published numbers were measured with, handles the chat template and the tokenizer, and streams the answer.

Status: version `0.2.0`. Text, and optionally still images (new in 0.1.2; on Linux only with llama.cpp built from source, see [Image input](#image-input-optional)). One request is generated at a time, unless the server is started with `--parallel N` (experimental, see [Several requests at once](#several-requests-at-once-experimental)). Tool calling is new in 0.1.1. Since 0.1.3 the engine also runs on other RTX 30, 40 and 50 series cards, without tuning or tests (see [GPUs other than the RTX 5090](#gpus-other-than-the-rtx-5090)). 0.2.0 reads prompts faster on cards with 30,000 MiB or more and adds the edit copy, which is on by default (see [Prompt reading and edit copy](#prompt-reading-and-edit-copy)).

## What you need

| | |
|---|---|
| GPU | NVIDIA **RTX 5090** (tested), or since 0.1.3 another RTX 30, 40 or 50 series card with compute capability 8.6 or higher and 16 GB or more (not optimized or tested; [below](#gpus-other-than-the-rtx-5090)); NVIDIA driver 580 or newer (a driver for CUDA 13; tested with 616.92) |
| OS | **Windows 11**, 64-bit (the only version tested), with the *Microsoft Visual C++ 2015-2022 x64 runtime* ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe)) |
| Python | 3.10 or newer |
| Disk | about 15 GB for the quality model folder, about 13 GB for the fastest model folder, about 18 GB for the max-quality model folder |
| Engine library | `yskn.dll` from the release page |

You do **not** need the CUDA Toolkit, PyTorch, or a C++ compiler.

## Model folder

YASAKANI publishes three model folders; you need one (the max-quality model needs 0.2.0 or newer; earlier versions stop with an
error when they open its 5-bit codes). They hold the same files except for the weights:

```
qwen38_27b_quality.yskn       the weights of the quality model (12.4 GB)   - or -
qwen38_27b_speed.yskn         the weights of the fastest model (10.3 GB)   - or -
qwen38_27b_max_quality.yskn   the weights of the max-quality model (15.7 GB)
bdraft2_k8.yskn               small draft model used to speed up generation (1.2 GB)
bdraft2_k8.shead              its output table (1.4 GB)
kvq_u5.bin                    settings of the compressed key-value cache (17 KB)
tokenizer.json                } from the Qwen3.8-27B repository, unchanged
tokenizer_config.json         }
chat_template.jinja           }
yasakani.json                 file list with SHA-256 checksums; it also names the weights file
selfcheck.json                reference output for `yasakani selfcheck` (see below)
```

## Install

```bat
pip install .
yasakani info --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

`info` checks Python, the three libraries (numpy, tokenizers, jinja2), the DLL, the GPU and the model folder, and tells you what is missing.
Put `yskn.dll` in a `bin` folder next to the package, or point to it with `--dll` / the `YASAKANI_DLL` environment variable. The model folder
can be given with `--model-dir` or `YASAKANI_MODEL_DIR`.

## Commands

| command | what it does |
|---|---|
| `yasakani info` | checks this computer, the DLL and (with `--model-dir`) the model folder |
| `yasakani verify` | compares every model file with the checksums in `yasakani.json` (reads 13 to 15 GB, up to a minute or two) |
| `yasakani selfcheck` | runs three short fixed greedy prompts and compares the output with `selfcheck.json`: *does this installation reproduce the reference output?* |
| `yasakani chat` | chat in the terminal. `/think on\|off`, `/effort low\|medium\|xhigh`, `/reset`, `/exit`; Ctrl+C stops the answer being written |
| `yasakani serve` | starts the OpenAI-compatible server (below) |
| `yasakani bench` | speed measurement: greedy, fixed length, your own prompts with `--prompts-file`; `--json` for a machine-readable result; `--edit` runs the edit tasks of the speed tables ([below](#bench---edit-the-edit-tasks-of-the-speed-tables)); `--concurrency N` sends N requests at once (experimental) |
| `yasakani update` | looks for a newer release on GitHub; the only command that contacts the network ([below](#yasakani-update)) |
| `yasakani report` | checks the wide copy passes on this GPU and prints a report to paste on GitHub ([below](#wide-copy-passes-per-gpu)) |

Loading the model takes about a minute; `serve` answers `/health` with `503 loading` until it is ready.

## `yasakani update`

What `yasakani update` does depends on how the package was installed.

* In a folder set up by the quickstart (`START.cmd`), the update is the quickstart's job. The command sends nothing itself: it opens a
  new window that runs `START.cmd --action update` and ends at once, so that nothing of this Python environment is in use while the
  package is replaced. The quickstart checks for the latest release, asks before it changes anything (`--yes` skips the question) and
  keeps the start files of the old version for `START.cmd --action rollback`.
* After a manual installation it asks GitHub's API for the latest release (one HTTPS request, User-Agent `yasakani-update`, nothing
  about this computer), prints the installed and the latest version, and changes nothing. It prints the two steps: download and unpack the engine ZIP of the new release (its
  SHA-256 is in the release's `SHA256SUMS.txt`), then `pip install --upgrade` the unpacked folder and use the `yskn.dll` in it. On Linux:
  `git pull` and build again as in [BUILDING.md](../BUILDING.md).

No other command contacts the network: `serve`, `chat`, `bench`, `info`, `verify` and `selfcheck` do not check for updates.

## `bench --edit`: the edit tasks of the speed tables

`yasakani bench --edit` (since 0.1.6) runs the three edit tasks of the speed tables in the README of the repository: the code edit (CPython's
`textwrap.py`, about 4,800 prompt tokens, at most 8,192 tokens out), the JSONL edit (60 records, at most 8,192) and the long JSONL edit
(160 records, at most 12,000). Their prompts, output limits and expected answers are those of `bench/` and come with the package
(`yasakani/edit_tasks.json`), so the repository is not needed. As in the measurement of the tables, each task is greedy with thinking off
and runs until the model ends its answer. `--think` turns thinking on: those runs use temperature 1.0 (top-p 0.95, top-k 20, min-p 0,
one seed per run counted up from `--seed0`, default 1) and the output limit 32,768 for every task (the server's default), so their tok/s
varies from run to run, and the thinking part, which mostly does not copy the input, is not sped up much; the thinking columns of the
README tables are not what `--think` reproduces. `--runs N` repeats each task, `--json` prints a machine-readable result; `--n` and
`--prompts-file` do not apply.

Each row gives the prompt tokens; the tokens written (with thinking on, the thinking included); the seconds from the first to the last
generated token (the engine's own time, inside the process; reading the prompt is not included); tok/s = (tokens - 1) / seconds; the
tokens per pass of the main model (tokens / passes, as in `bench`); the number of wide copy passes; and whether the answer is the expected
one (white space around it and a Markdown code fence are ignored, as by the measurement client; otherwise the number of differing lines).
The first lines name the package version, the SHA-256 of the engine library, the GPU with its compute capability and driver, the model file
and whether wide copy passes are used, so that a pasted result says what it was measured with. A task whose prompt and output limit do not
fit the context (`--max-ctx`; the long JSONL edit needs 18,640 tokens with thinking off and 39,448 with `--think`) is skipped, and the
exit code is then 1.

Wide copy passes (since 0.1.4: a pass that copies the input checks up to 64 tokens) are used where the check on the card opened them
(since 0.1.7 on RTX 30, 40 and 50 cards; up to 0.1.6 on compute capability 12.x only), and were measured by us on the RTX 5090 only. The
header gives the widths the engine has open ([below](#wide-copy-passes-per-gpu)), and each row
of `--json` also has the SHA-256 of the generated ids, the back-offs of the copy loop and its final times of copy cycles. The
thinking-off tables were measured through the server's stream (2 runs per task), so these numbers are close to them, not the same. The
code-edit file has one e-mail address replaced by `<email>`, as in `bench/`. YASAKANI's thinking-off measurements of November 15 (in the
README: the speed table and "What 0.1.4 changed") used this same text; every earlier record of the code edit in `bench/` (November 3 to
8; among them the other engines' rows and YASAKANI's thinking-on rows) had the address itself, 4 prompt tokens more.

<a name="wide-copy-passes-per-gpu-017-test-build"></a>

## Wide copy passes per GPU

Wide copy passes (since 0.1.4: a pass that copies the input checks up to 64 tokens) were measured by us on the RTX 5090 only. Since 0.1.7
each GPU decides which widths (32, 48, 64 rows) it uses (`yasakani/widecheck.py`; first in the 0.1.7 test build). Only answers that copy
the input, such as edits, get faster; normal prose does not. With `serve --parallel N` a request that runs alone uses the opened widths;
passes shared by several requests carry at most 16 rows together and do not use them.

| `--wide` / `YASAKANI_WIDE` | RTX 50 (code sm_120) | RTX 30 / 40 (code sm_86 / sm_89) |
|---|---|---|
| `auto` (default) | on; a width the check finds slower is left out | only the widths the check finds correct and faster; none when it cannot check (up to 0.1.6: off) |
| `on` | as `auto` | as `auto` |
| `off` | off | off |
| `force` | every width, no check | every width, no check (to compare) |

`--wide` goes with `chat`, `serve`, `bench` and `selfcheck` (the command line wins over the variable); `--wide-recheck` checks again even
when a kept result fits. The package passes the setting to the engine as `YSKN_WIDE` (a `YSKN_WIDE` in your shell is removed like any
`YSKN_*` setting).

**The check** runs after the warm-up, once per GPU, driver, engine library (its SHA-256), package, model, and power limit, clocks,
driver mode of the GPU and whether a display is attached to it (nvidia-smi `display_attached`):

1. *Correctness.* The engine reads about 2,400 tokens of a fixed text (the prompt and answer of the code edit task) in passes of 17, 31,
   32, 33, 47, 48, 63 and 64 rows and again in passes of 16 over the same positions. Since 0.2.0 it then reads about 4,200 tokens of the
   same text and compares two passes of 64 rows that cross and follow position 4,096 in the same way (463 positions in all). The most
   likely ids and the NLL must be the same bit for bit, else no width is used, passes of 17 to 64 rows go back to the 16-row sums (`set_opt 31`, on RTX 50 too) and `yasakani
   selfcheck` fails ("mismatch"; please report it). The text is read in pieces of at most 1,023 tokens: one read of 1,024 tokens or more
   would make the engine set up the work area of long prompt reads and keep it (about 580 MiB at the sizes of 0.1.7, more with the new
   prompt reading of 0.2.0), which generation does not need.
2. *Time.* At 4,096 tokens of context it times copy cycles as the generation runs them (the pass, the commit of all its rows, the
   drafter reading them; `yskn_time_cycle`): 16 rows and W rows in turn, about 0.6 s each, 3 rounds greedy and one round sampled. The
   time of a width is the median of the means of pairs of cycles. W is used when it takes at most 0.8 × W / 16 times 16 rows (32: 1.60,
   48: 2.40, 64: 3.20). RTX 50 keeps a width unless every round is over the limit and the rounds agree within 10 %; RTX 30 / 40 opens a
   width only when every round is under it and they agree. The measured times are also where the engine's time-based back-off starts.
3. *Idle GPU.* Before and after, nvidia-smi is asked (by the GPU's UUID, which the engine gives) for the load and for other programs that
   hold 256 MiB or more of its memory (Windows lists every program that draws with the GPU and gives no memory per program, so there the
   load alone decides). If the GPU is busy or nvidia-smi does not answer for it, nothing is measured or kept: RTX 50 then uses every width
   (as 0.1.6), RTX 30 / 40 none.

The check of 0.2.0 is version 3 of it: a result kept by 0.1.7 is measured again once. The result is kept in `wide_check.json` (in `YASAKANI_CACHE_DIR`, else `%LOCALAPPDATA%\yasakani` on Windows, `$XDG_CACHE_HOME/yasakani` or
`~/.cache/yasakani`) for 30 days; the file holds no path and only a hash of the GPU's UUID. If during generation the engine goes back
from wide passes in at least 10 of every 100 of them, counted over at least 128 wide passes in one process, the next start checks again
(a few back-offs, for example where copies end, do not reach that rate; the rule is in docs/TECHNOLOGY.md 5.4). On RTX 30 / 40, when less than 128 MiB of GPU memory is free after the graphs of the
wide passes are built, the widths are closed again (a shorter `--max-ctx` leaves more memory free). One line at start says what was decided. `serve --no-warmup` uses a kept result
only. The check needs a context of at least 4,224 tokens.

**`yasakani report`** checks again, runs the three edit tasks of `bench --edit` with the wide passes off, with the widths of the check and
with every width (order off, checked, all, all, checked, off; `--rounds N` repeats it; after a mismatch only off), then the self-check
(the reference cases, and the self-consistency check with every width open), and prints Markdown to paste on GitHub (`--out FILE` also writes it, `--json` gives the raw
data, `--no-bench` and `--no-selfcheck` leave parts out). It names the GPU, compute capability, memory, SMs, driver and its mode, power
limit, clocks, display, the package, the engine library (file name, the first 16 hex digits of its SHA-256, `yskn_build_info`) and the
model, but no UUID, path, user or computer name. The columns "wide passes" and "back-offs" give the sums of the runs of each column. It
takes about 1.5 minutes on the RTX 5090; longer on other cards. The engine itself prints the full path of the draft head file to the
console when it loads; that line is not in the report, but leave it out when you paste the console output.

From Python: `Engine(pack, wide="on")`, then `engine.warmup(...)` with `wide_ids=yasakani.widecheck.check_ids(fmt)` (as `ChatRunner.warmup`
does); `engine.wide_result` says what was decided and `engine.wide_state()` what the engine has open. With an engine library older than the
0.1.7 test build (0.1.6 or older) nothing changes (and `--wide on` or `--wide force` says that it has no effect).

## Prompt reading and edit copy

0.2.0 adds two settings to `chat`, `serve`, `bench` and `selfcheck`. The command line wins over the environment variable. A wrong value in
the variable stops the command before the model is loaded (`YASAKANI_PREFILL / --prefill must be one of auto, new, old, not 'x'`, and
the same for `YASAKANI_EDIT_COPY / --edit-copy`). `yasakani info` prints both settings and, for each GPU, what the prompt reading setting
chooses there; `yasakani report` names both in its table (it follows the two variables).

| `--prefill` / `YASAKANI_PREFILL` | prompt reading |
|---|---|
| `auto` (default) | the new reading on a card with 30,000 MiB or more (nvidia-smi's total); on a smaller card, or when nvidia-smi cannot tell the memory, the reading of 0.1.7 |
| `new` | the new reading on any card; under 30,000 MiB (or of unknown memory) with the pass sizes of 0.1.7. Experimental there: never run on a real card under 30,000 MiB |
| `old` | the reading of 0.1.7 on any card, also the passes of 64 tokens as in 0.1.7 |

**The new prompt reading** computes the first read of a prompt layer by layer in larger pieces, computes the output layer only where it
is needed, and, for prompts of 8,192 tokens or more, puts saved states that are still free into the long stretches between the others
(within the 8 saved states of a conversation), so that a known prompt changed in the middle is read again from the last saved state
before the change. The answers do not change. On the RTX 5090 it read prompts of 8,192 and 32,768 tokens 1.21 and 1.23 times as fast as
0.1.7 (quality model, thinking off, the first read with nothing reused, median of 2 runs; the tables are in
[docs/MEASUREMENTS.md](https://github.com/yasakani-dev/yasakani/blob/main/docs/MEASUREMENTS.md#measurements-020)). It needs a work area on the GPU, which the engine makes at the
first long prompt and keeps until the model is closed: on the RTX 5090 the peak GPU memory was 1,385 to 1,388 MiB (quality model) and
1,476 to 1,479 MiB (fastest model) above 0.1.7. The engine makes the area only when the GPU has the area and 1 GiB more free (it tries smaller sizes of the area first).
When it does not find that much, it prints once, on stderr:

```
[q35_layer] capacity fallback; no context reduction
```

This is not an error: the context stays as it is, and until the model is closed the engine reads prompts pass by pass without the work
area. The other changes of 0.2.0 stay on, and the output is the same.

**The line at start** says which reading is used and why, for example:

```
prompt reading: new (setting auto: 32607 MiB card)
prompt reading: as in 0.1.7 (setting auto: 24564 MiB card, new only from 30000 MiB; --prefill new or YASAKANI_PREFILL=new to try it)
prompt reading: as in 0.1.7 (setting auto: card memory not known; --prefill new or YASAKANI_PREFILL=new to try it)
prompt reading: as in 0.1.7 (setting old)
prompt reading: new (setting new)
prompt reading: new (setting new; 24564 MiB card, so prompt sizes of 0.1.7)
```

When CUDA's numbering cannot be matched with nvidia-smi's, the card is given as `smallest of N GPUs: ... MiB`.

**Cards under 30,000 MiB.** On such a card the package keeps the pass sizes of 0.1.7 with every setting (`YSKN_Q35_PMAX` 1024,
`YSKN_Q35_LAYER_ROWS` 4096), because the memory rule of the context ([below](#gpus-other-than-the-rtx-5090)) was measured with them. With
`auto` the reading is that of 0.1.7: on the RTX 5090 shown to the package as a 24,564 MiB and a 16,384 MiB card, a prompt of 32,768 tokens
gave the same output as 0.1.7, with the peak GPU memory within 9 MiB of it (an earlier run also found the same ids, state and saved
states). With `--prefill new` the work area takes about
579 MiB, and the package leaves room for it when it chooses the context itself:

```
note: --prefill new: the context is chosen with 1800 MiB more left free (the new prompt reading keeps a work area of about 579 MiB from the first long prompt on; the engine makes it only with 1024 MiB to spare, and reads prompts without it when it finds less)
```

If keeping that room would cost more than half of the context (on 16 GB cards), the context is chosen without it, and the engine decides
by its own check:

```
note: --prefill new: keeping 1800 MiB free for the work area of the new prompt reading would cost more than half of the context, so the context is chosen without it; the engine reads prompts without that work area when it finds no room
```

With `--max-ctx` the context is used as given, and the line says how much must be free:

```
note: --prefill new: the new prompt reading needs about 2169 MiB free in nvidia-smi after the model is loaded (a work area of about 579 MiB, made only with 1024 MiB to spare); with less, prompts are read without that work area.  The context you gave is used as it is
```

With `serve --parallel N` the same room is left free, and the line that lowers N says `(besides 1800 MiB kept for the new prompt
reading)`. With `--parallel N` and no `--max-ctx`, on any card, the contexts take the free GPU memory, so the engine may find no room for
the work area and read prompts without it; this has not been measured.

What `--prefill new` did on the RTX 5090 set up to look like a smaller card in both ways (what the package sees, and only the free GPU
memory of that card left because another process held the rest), a prompt of 32,768 tokens, one run each, the same output as 0.1.7 every
time: as a 24 GB card (24,564 MiB, quality model) the package chose a context of 229,376 instead of 262,144, the engine made its work
area and read the prompt in 7.56 s (0.1.7: 9.00 s); with 700 MiB less free it found no room, printed the fallback line and read without
the work area. As a 16 GB card (16,384 MiB, fastest model) keeping the room would have cost more than half of the context, so the context stayed
98,304, and the engine found no room and read without the work area. So at the automatic context a 16 GB card does not use the new reading; a
smaller `--max-ctx` leaves room for it (not measured). No real card under 30,000 MiB has been measured, and how much free memory the
driver reports differs between PCs.

**Edit copy** (`--edit-copy on|off`, `YASAKANI_EDIT_COPY`, default `on`). When an answer copies the input with the same short replacement
again and again (a key or a column renamed in every line), the drafts of the copy passes put in that replacement once the model has
written it at two places. Every drafted token is still checked by the model, so the output does not change. It works where one request is
generated at a time; the shared passes of several requests in `serve --parallel N` do not use it. With `on` the commands print
the line `edit copy: on` at start. The package passes the setting to the engine as `YSKN_EDIT_COPY` (1 or 0). On the RTX 5090 it made a key
renamed in every line of a JSONL file 1.77 times and a column renamed in every view of an SQL file 1.80 times as fast (quality model,
thinking off, mean of 2 runs; [the measurements](https://github.com/yasakani-dev/yasakani/blob/main/docs/FAQ.md#edit-copy-020)).

The quickstart's "compare" script (`lookahead_check.py`) and `tools/quality/score_yasakani.py` use the same prompt reading and edit copy
as `yasakani chat` on GPU 0 (they follow `YASAKANI_PREFILL` and `YASAKANI_EDIT_COPY`) and print the same line at start.

## GPUs other than the RTX 5090

Since 0.1.3 the engine library is built for compute capability 8.6, 8.9 and 12.x. On any GPU other than the RTX 5090, `chat`, `serve`,
`bench` and `selfcheck` print a note at start: the engine runs there, but it is not optimized or tested, and its speed and operation are not
guaranteed. `yasakani info` shows what the package does on each GPU it finds. AMD GPUs and Macs are not supported.

**Context length.** The engine reserves the key-value cache for the whole context when it opens the model. When you do not give `--max-ctx`,
on a GPU with less than 30,000 MiB the package chooses the context before it opens the model:

```
room    = free GPU memory (nvidia-smi: total - used) - the three model files - 2,100 MiB (fastest model) or 2,300 MiB (quality and max-quality model) - 512 MiB
context = room / 22,400 bytes per token, rounded down to a multiple of 16,384, at most the model's 262,144
```

If that is less than 65,536 tokens and leaving out the output head of the draft model (`bdraft2_k8.shead`, 1.3 GiB) gives more, the head
is not loaded (`YSKN_BDR_SHEAD=0`; on the RTX 5090 each pass took about 3% longer and the output stayed the same). If not even 16,384 tokens
fit, the command stops before loading and says so; `--max-ctx` tries a given length anyway, and leaves out the output head when, by the same
numbers, it does not fit beside that length (without this the head was loaded whatever the length). On the RTX 5090, and on any card with 30,000 MiB
or more, nothing changes: the context stays 262,144. The 2,100 and 2,300 MiB were measured on the RTX 5090 under Windows; the rule has not
been checked on other cards. It is `yasakani.gpu.plan_context`; `Engine` itself does not apply it (pass `max_ctx`).

**Several GPUs.** nvidia-smi and `yasakani info` number the GPUs in PCI order; CUDA by default puts the fastest first. Since 0.1.5 the
package sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before it loads the engine library (unless it is already set), so `--device N` is GPU N of
`yasakani info` and the context is chosen for that GPU. On a PC with two GPUs, 0.1.5 was reported to fail with `graph instantiate
failed` on a GPU that was not CUDA's device 0 while the other one was visible ([#20](https://github.com/yasakani-dev/yasakani/issues/20)).
The likely cause, found by reading the code and with a stand-in test on one GPU (not yet confirmed on two GPUs): CUDA's current GPU
belongs to the thread, the engine chose its GPU only on the thread that opened it, and generation runs on other threads. Since 0.1.6
the engine makes its GPU current on every call, from any thread. The commands that load the model (`chat`, `serve`, `bench`,
`selfcheck`) also make that GPU the only one the engine sees: they set `CUDA_VISIBLE_DEVICES` to its UUID from nvidia-smi, open it as
device 0 and print one line that says so. When nvidia-smi gives no UUID for it, nothing is made invisible, and the line says that the
engine also sees the other GPUs. A `CUDA_VISIBLE_DEVICES` you set yourself is kept, and still renumbers the GPUs
(`--device` then counts the GPUs it names, in its order). When `CUDA_DEVICE_ORDER` is set to something else, it is not known which GPU
`--device` is: nothing is made invisible, and the context is chosen for the smallest of them. With one GPU nothing changes. `Engine`
itself makes no GPU invisible; to do what the commands do from Python, set `CUDA_VISIBLE_DEVICES` before the first `Engine`
(`yasakani.gpu.visible_only` names the GPU, but only once `CUDA_DEVICE_ORDER` is `PCI_BUS_ID`: call `yasakani._native.pci_bus_order()`
first). The quickstart sets `CUDA_VISIBLE_DEVICES` to the GPU it checked. Run one engine per process.

**The self-check** compares with output made on an RTX 5090. On another GPU tiny differences in the computation can make the greedy output
take another token part-way through, so since 0.1.4 `selfcheck` also lets the engine read the reference output there and compares its
predictions with those of the RTX 5090 ([The self-check](#the-self-check)). Within the tolerances it exits with 0 (the same output) or 3 (only
the output differs; the quickstart reports this and goes on); otherwise with 1, and the quickstart stops. Every GPU also checks itself against
its own predictions, with no tolerance (the same section).

`YASAKANI_NVIDIA_SMI` names the nvidia-smi to use; `none` turns the GPU check and the context rule off.

## The server

```bat
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

It listens on `127.0.0.1:8765` (this computer only). Point any OpenAI-compatible client at `http://127.0.0.1:8765/v1`; the model id is `qwen3.8-27b`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.reasoning_content)   # the thinking (the model thinks first by default)
print(r.choices[0].message.content)             # the answer
```

```bat
curl http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

Supported request fields: `messages` (text, and images only with the optional image input below; the roles `system`, `developer`, `user`, `assistant` and `tool`), `tools` with `tool_choice`
`auto`, `none`, `required` or a named function (see below), `max_tokens` / `max_completion_tokens`, `temperature` (0-2; 0 is greedy), `top_p`, `top_k` (at most 20),
`min_p`, `seed`, `stop` (up to 4 strings), `stream` with `stream_options.include_usage`, `reasoning_effort` (`low`, `medium`, `high`, `xhigh`),
`chat_template_kwargs` (for example `{"enable_thinking": false}`), `ignore_eos`, `loop_guard`. The defaults are the model's own:
temperature 1.0, top_p 0.95, top_k 20. Not supported (answered with 400): images when the server runs without the image input options, the old `functions` / `function_call` fields, `n` other than 1, `logprobs`.

Tool calling (new in 0.1.1): send `tools` (functions) with `tool_choice` `"auto"` (the default) or `"none"`. The model's own chat template
describes the tools to the model. The calls it writes come back as `message.tool_calls`, and `finish_reason` is `tool_calls`; in a stream each
finished call arrives as one chunk with the whole arguments. Send the results back as messages with the role `tool`. An input that does
not fit into the context is answered with 400 and the error code `context_length_exceeded` before any text is sent. Also accepted:
`thinking_budget` (a number of tokens after which the thinking is closed and the answer starts; 0 = no limit) and
`"thinking": {"type": "enabled"}` / `{"type": "disabled"}`. Tool calling is new and has been tested far less than plain chat.

Since 0.1.4 `tool_choice` `"required"` and a named function (`{"type": "function", "function": {"name": "get_weather"}}`) are accepted
too (contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8)), but not enforced: the engine has no constrained decoding, so the request runs as `"auto"` and the
model is left to make the call. An answer without a call comes back as text with `finish_reason` `stop`. With a named function only that
function is put into the prompt; the prompt then differs near its start, so a long conversation is read again from the start, and once more
by the next request with all the tools. A name that is not in `tools`, or `"required"` without `tools`, is answered with 400.

```python
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather of a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "How is the weather in Osaka?"}], tools=tools)
print(r.choices[0].message.tool_calls)
```

* The thinking comes back as `reasoning_content` (and as `delta.reasoning_content` when streaming), the answer as `content`.
* By default the model thinks first, at its longest thinking setting, and `max_tokens` counts the thinking too. With a small limit a request can
  end while the model is still thinking (`finish_reason` is `length` and `content` is empty). Raise the limit, or shorten the thinking with
  `reasoning_effort` (`medium`, `low`), or switch it off with `{"enable_thinking": false}` in `chat_template_kwargs`.
* A stream carries one chunk per token. Where one character is split over several tokens, its text comes with the last of them, so the number
  of chunks is slightly smaller than the number of tokens; the exact count is `usage.completion_tokens` (ask for it with
  `stream_options: {"include_usage": true}`). The chunks arrive in small groups, about 64 times a second.
* **Loop guard.** If the thinking falls into an endless repetition, the guard cuts it, asks the model to answer from what it has written, and reports this in
  `usage.loop_guard`. Pass `"loop_guard": false` to turn it off. Since 0.1.4 the guard and `thinking_budget` cut at a point that depends only on
  the generated tokens, so the answer does not depend on the look-ahead.
* Input, thinking and answer together are limited to the context length (262,144 tokens by default; `--max-ctx` lowers it and the memory it needs).
* Asking again with the earlier conversation in front of a new question does not recompute the earlier turns (`usage.prompt_tokens_details.cached_tokens`).
  Since 0.1.4 the same long document followed by a different question is not read again either (when the document comes before the question
  and no other request comes in between): the engine keeps up to 8 saved states along the prompt of one conversation in pinned host memory
  (about 1.5 GiB of RAM). The prompt is read from the start after an unrelated request, when thinking is
  switched on or off, after a request with an image, when the tools differ (a named function in `tool_choice`, above), and when a prompt under 8,192 tokens
  differs part-way. Since 0.2.0 the new prompt reading also keeps saved states inside the first read of a long prompt (where there is room: at least 4,096 tokens from the other saved states),
  so that a known prompt changed in the middle is read again from the last saved state before the change ([above](#prompt-reading-and-edit-copy)).
* Next to `usage`, the answer carries `timings` in the layout of the llama.cpp server (contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8)): `cache_n`, `prompt_n`,
  `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_token_ms`, `predicted_per_second`,
  and `draft_n` / `draft_n_accepted` (the tokens the look-ahead proposed and those the model confirmed; only when it proposed any). A stream
  carries them in its last chunk: the usage chunk with `stream_options.include_usage`, else the chunk with `finish_reason`. As with
  llama.cpp, `prompt_ms` runs until the first generated token and `predicted_ms` from it to the last one; the encoding of images is not in
  them. These times are measured inside the engine. The speed tables of this project are measured by the client, from the first to the
  last chunk ([docs/MEASUREMENTS.md](https://github.com/yasakani-dev/yasakani/blob/main/docs/MEASUREMENTS.md)), so their numbers differ from these.

### Output token suppression (`--suppress-tokens`)

`yasakani serve` and `yasakani chat` accept `--suppress-tokens PATH` (repeatable; or `YASAKANI_SUPPRESS_TOKENS`, paths separated by the
platform path separator) to mask output token IDs (`-inf` logits before greedy argmax and sampling; prompt tokens are unaffected).

* **Process-wide setting.** The mask is loaded once on the engine for the whole process and applies to every request; it is not per request.
* **Accepted JSON formats** (UTF-8, with or without BOM): a plain array of integer token IDs (`[1001, 1002]`), an object with
  `target_token_ids` (optional `n_target` length check), `suppress_tokens`, `token_ids`, or exact single-token strings in `tokens`
  (resolved against `tokenizer.json`). Optional `vocab_size` and `tokenizer_sha256` fields are verified against the model folder.
* **Safety guards.** Special/EOS tokens and structural control tokens (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
  `<tool_response>`, `</tool_response>`) are refused unless `--allow-suppress-special` is passed. At least 20 unmasked token IDs must
  remain so the sampler's top-k candidate lists stay valid.
* `yasakani selfcheck` and `yasakani bench` do not accept `--suppress-tokens` and ignore `YASAKANI_SUPPRESS_TOKENS` (with a note at
  start) so reference checks and speed measurements stay unmasked.

### Image input (optional)

Image input works on **Windows**, and on Linux with llama.cpp built from source (contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8); we have not tried it there,
see [Linux](#image-input-on-linux) below). The engine library already has the entry for image features (`yskn_set_visual`); the server feeds it
when it is started with three more files. Without them the server is text only, as before: a request with an image is answered with 400 at
once, without waiting for a running generation.

| | |
|---|---|
| `--mmproj` | the projector of Qwen3.8-27B: `mmproj-Qwen3.8-27B-BF16.gguf` from [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (931,145,888 bytes, sha256 `de2a49866988ea272c43dc2a43ee2662c25725a717ca5b9059ea420a97f53fe8`) |
| `--llama-dir` | Windows: a folder with the unpacked `llama-b10488-bin-win-cpu-x64.zip` of the [llama.cpp release b10488](https://github.com/ggml-org/llama.cpp/releases/tag/b10488) (18,490,962 bytes, sha256 `6c938f6d79aac96cb90fda673aade20cff9b1b6c1e97de04f4d5d60bca107082`); its multimodal library turns the image into features. Only this CPU build is accepted: at start the server compares its `llama.dll`, `mtmd.dll`, `ggml.dll` and `ggml-base.dll` by SHA-256 and refuses a folder with GPU backends (`ggml-cuda.dll` and the like). Linux: a folder with the shared libraries of llama.cpp b10488 built from source ([below](#image-input-on-linux)) |
| `--vision-model` | a Qwen3.8-27B GGUF. Only its settings and vocabulary are read, never its weights. We tested `Qwen3.8-27B-Q4_K_M.gguf` of [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) (17,442,399,968 bytes); other GGUF files of this model should work the same way, but we have not tried them |
| Pillow | install the package with `pip install ".[vision]"` instead of `pip install .`, or add Pillow to the Python that runs `yasakani` with `pip install Pillow` (in a quick-start folder: `venv\Scripts\python -m pip install Pillow`) |

```bat
pip install ".[vision]"
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll ^
  --mmproj D:\gguf\mmproj-Qwen3.8-27B-BF16.gguf --llama-dir D:\llama-b10488 --vision-model D:\gguf\Qwen3.8-27B-Q4_K_M.gguf
```

These options are checked at start, before the model loads; a missing file, the projector and the GGUF given the wrong way round, another
llama.cpp build or a missing Pillow stops the server with an error message. Relative paths are taken from the folder the server is started in.

Send the image as an OpenAI `image_url` part with a `data:image/...;base64,` URI, in a user message. Limits:

* at most 16 images per request. The whole request body is at most 32 MiB and base64 makes the data a third larger, so the images of one
  request can be about 24 MiB together;
* each image at most 40 million pixels, and its longer side at most 200 times its shorter side;
* each image is downsized to at most **1,024 image tokens** (about one million pixels), keeping its shape: a 1920 × 1080 screenshot
  becomes 1,008 image tokens and a 4032 × 3024 phone photo 972 (at full size they would take 2,040 and 4,015). Downsizing loses detail,
  so small text in a full-screen screenshot can become hard to read; crop the part you need, or raise the limit.
  `--vision-max-image-tokens N` sets it, from 64 to 4,096 (the projector's own limit: images up to about 4 million pixels keep their size).
  Below 256 a very long, thin image can stay a little above the limit (113 image tokens at 64);
* at most 8,192 image tokens per request, 8 times the per-image limit (`--vision-max-request-image-tokens N` changes it). The images
  of earlier turns count too, because a client sends the whole conversation again with every turn.

An image outside these limits, or one that cannot be decoded, is answered with 400 before it is encoded. When the images make the input longer than
the context, the answer is 400 with the error code `context_length_exceeded`, as for long text. `detail` is accepted and ignored.
`--vision-allow-local-files DIR` also accepts local file paths (and `file:` URLs) of files inside the folder DIR; remote URLs are never fetched.
`--vision-cache FOLDER` keeps the features of images seen before, one file per image: it holds information about every image sent to the
server, so delete the folder when you no longer want it.

* The image is encoded on the CPU (by default; on Linux see below); generation then runs on the GPU as usual. The encode time grows with the square of the image tokens.
  On our PC (AMD Ryzen 9 9950X3D2, 16 cores) with the default of one thread per physical core (16 threads): 1.2 s for a 512 × 512 image
  (256 image tokens), 12 s for 1280 × 720 (920), 12 to 15 s for any image downsized to the default limit (1920 × 1080, 2048 × 2048,
  4032 × 3024). At full size the same encoder took 51 s for a 1920 × 1080 screenshot with 16 threads and 5 1/2 minutes for a
  2048 × 2048 image with 8 threads. `--vision-threads N` sets the number of threads: with 32 (two per core) the same 1920 × 1080
  image took 10.9 s instead of 13.5 s, with the same features.
* **While an image is encoded, the server answers no other request**, and the client gets nothing back yet, not even the start of a
  stream: a request with eight large images waits about 2 minutes (8 × 14 s) for its first token. Give the client a long enough timeout.
  An image seen before is not encoded again: the server keeps the features of up to 512 MiB of images in memory (about 50 images at
  the default limit), and `--vision-cache FOLDER` also keeps them on disk.
* Image understanding was measured by the contributor of this feature (pull request #4) with their own test of 5 levels and 49 questions,
  which is not published: **87.8** with the quality model, against 89.8 for the original BF16 model in llama.cpp. With the projector on the GPU
  their score fell to 79.6 and 81.6 in two runs, which is why the projector runs on the CPU by default. We have not measured image quality ourselves; we checked that it works on an
  RTX 5090 with the quality model and a few simple test images.
* Image tokens get Qwen's three-axis positions (time, row, column); the text after an image continues from the largest of them.
* Requests without images are handled as before: the same prompt tokens and the same answers. After a request with an image the engine
  starts the next prompt from the beginning (no reuse of the earlier turns).

#### Image input on Linux

Contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8); its author ran it on native Ubuntu 26.04 with an RTX 5090. We have not run image input on Linux, and it is not
part of an official release. There is no llama.cpp release to compare with, so `--llama-dir` is a folder with the shared libraries of
llama.cpp b10488 built from source (`libllama.so`, `libmtmd.so`, `libggml.so`, `libggml-base.so` and the backends; the commands are in
[BUILDING.md](https://github.com/yasakani-dev/yasakani/blob/main/BUILDING.md#image-input-on-linux)). At start the server checks that they are there and that `libggml-base.so`
reports the version and the commit of b10488 (ggml 0.20.2, commit 9d77fa172; a build without git or from a changed ggml source is
refused). There is no checksum, so build b10488 unchanged; the structure layout is checked when the encoder opens.

By default the projector runs on the CPU, as on Windows. With the environment variable `YASAKANI_VISION_GPU=1` and a CUDA build of
llama.cpp it runs on the GPU **with flash attention switched off** (experimental, off by default). Measured by the contributor on an
RTX 5090, against the features of the CPU: with flash attention the relative error was 3% at 1,024 image tokens and 10% at 4,096;
without it 0.6 to 0.7%, at 0.15 s per 1,024-token image and 0.44 s per 2,048-token image instead of 7 s and 20 s on the CPU (14 threads).
Without flash attention the GPU memory of an encode grows with the square of the image tokens: about 4 GiB at 2,048, and at 4,096 (about
17 GiB) it did not fit next to the model, so with the variable set `--vision-max-image-tokens` can be at most 2,048. The projector
(0.93 GB) and its work memory then stay on the GPU next to the model. The image test above has not been repeated in this mode. The
variable is ignored on Windows.

### Several requests at once (experimental)

```bat
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll --parallel 4
```

`--parallel N` (or the environment variable `YASAKANI_PARALLEL`) lets the server generate up to N requests at once on the one engine.
The default is 1: one request at a time, as before. This is experimental: it has been tested on the RTX 5090 only, and far less than one
request at a time.

| Option | Default | What it sets |
|---|---|---|
| `--parallel N` | 1 (or `YASAKANI_PARALLEL`) | requests generated at once, 1 to 64 |
| `--max-queue Q` | 64 | requests that may wait for a free place; more are answered 503 |
| `--keep-turn auto\|off\|SECONDS` | `auto` (or `YASAKANI_KEEP_TURN`) | how long the place of an answer with tool calls is kept for its next turn. `auto`: about as long as reading the conversation again takes, 1 to 30 seconds. Seconds: 0 to 600 |
| `--max-wait auto\|SECONDS` | `auto` (or `YASAKANI_MAX_WAIT`) | how long a new request waits for next turns before it takes a kept place. `auto`: the reading time of the kept conversation (at most 30 seconds) and 0.25 seconds. Seconds: 0 to 600 |

`--max-queue`, `--keep-turn` and `--max-wait` only matter with `--parallel` 2 or more. `yasakani selfcheck --parallel N` and
`yasakani bench --concurrency N` are described below.

* **Context.** Every request has a context of its own, and all of them have the same length. Without `--max-ctx` it is chosen from the free
  GPU memory, on every card: N + 1 contexts of 22,400 bytes per token and about 256 MiB per request must fit beside the model (the same
  rounding and the same rule for `bdraft2_k8.shead` as above). The server prints the length at start and `/health` reports it as
  `max_context_tokens`. If not even 16,384 tokens fit, the server stops and names the largest N that fits.
* **The check at start.** The first time the server starts with an engine library, a GPU and a driver, it lets N short requests run at once
  and compares their tokens with those of the same requests one at a time; they must be the same. The result is kept (in
  `%LOCALAPPDATA%\yasakani\parallel_check.json`, or in the folder `YASAKANI_CACHE_DIR` names), so the next start skips the check. If the
  tokens differ, the server says so and serves one request at a time. It does the same when the engine library has no entries for several
  requests at once (an older `yskn.dll`) or when its settings do not allow them. `yasakani selfcheck --parallel N` runs the same check on
  demand.
* **The same answers.** A request gets the tokens that the same conversation gets when it is sent alone to a new server, provided its prompt
  is read the same way (`timings.snap_mode` and `timings.cache_n`, below) and it has its own `seed`. Without a `seed` each request takes the
  next one in the order the requests arrive, as before. With a stop string the answer is the same up to the stop.
* **What waits.** A request with images runs alone: the others pause until it is done. While a prompt is read (for a long one, seconds),
  the others pause too. A request whose generation goes on in a second part (after a thinking budget or a loop cut) keeps its place for
  that part, so that the part is read as when the request is alone: for 2 seconds after its client has received the first part, and for
  up to 30 seconds while a slow client is still receiving it. More than `--max-queue` waiting requests (default 64) are answered 503
  with `Retry-After: 2`. A request that is waiting, or that runs without a stream, is cancelled when its client closes the connection.
* **The next turn after tool calls.** An answer that ends with tool calls keeps its place for the next request of that conversation
  (with the tool results): that request continues it instead of reading the whole conversation again, and is read before the requests
  that wait. The place is kept from the end of the answer about as long as reading the conversation again would take, which is what
  keeping it saves: 1 second for a short conversation, about 4 seconds for 16,000 tokens and at most 30 seconds (times measured on
  the RTX 5090; a slower card reads slower). `--keep-turn SECONDS` (or the environment variable `YASAKANI_KEEP_TURN`) keeps it that
  long instead, `--keep-turn off` not at all. Once the next request has come, the place stays kept for it until it is read, also when
  the server is busy reading other prompts meanwhile. A request that was sent before the answer ended is not its next turn and does not
  take its place.
* **How long a new request waits for next turns.** Next turns come before new requests, so a new request that finds every place kept
  for a next turn, or taken by next turns that came after it, waits. It counts that time (not the time it waits for other requests),
  and once it has waited so as long as reading a kept conversation again would take and 0.25 seconds more, it takes the next place
  kept for a next turn, before the next turns that came after it, and that conversation's next turn reads the conversation again. So
  next turns do not hold a new request back much longer than the reading the keep saves: at most until the next turn that runs in a
  place when that time is up has ended. `--max-wait SECONDS` (or `YASAKANI_MAX_WAIT`) sets this time for every kept place. Places kept
  for the second part of a request (above) are never taken. The answers are the same either way; only how the next turn's prompt is
  read (`timings.snap_mode`) and the waiting change.
* **What you see.** `/health` adds `parallel`, `slots_busy`, `slots_idle` and `queued`. `timings` adds `queue_ms` (the time the request
  waited), `parallel_avg` (how many requests shared its passes on average) and `snap_mode` (how its prompt was read: 0 from the start,
  1 continuing the request before it, 2 from a saved state). A stream starts with its first piece of text.
* **Speed.** One pass of the engine carries at most 16 rows for all requests together, so each request is slower than alone, and the requests
  together are not always faster than one after another. When one request is left, it goes back to the speed of a single request.
  `yasakani bench --concurrency N` measures N requests at once on your GPU.

### Safety of the server

* By default only this computer can connect. The `Host` header must name the address the server is bound to, and a browser request from another web page
  (`Origin` / `Sec-Fetch-Site`) is refused, so a web page open in your browser cannot use your GPU.
* To serve other computers use `--host 0.0.0.0 --api-key YOUR_KEY` (or `YASAKANI_API_KEY`). A non-local address without a key is refused at start.
  The server speaks plain HTTP: put it behind a VPN or a TLS proxy when it leaves your LAN.

## The self-check

`selfcheck.json` holds, for three short fixed prompts, a hash of the prompt ids and a hash of the greedy output ids made with the reference installation.
`yasakani selfcheck` produces the same records on your computer and compares them. A *prompt ids differ* result means the tokenizer or template files are
not the reference ones; an *output differs* result means the weights, the DLL, or the GPU/driver behave differently from the reference. Send the result
with your report in that case. The reference was made on an RTX 5090.

On any other GPU (since 0.1.4) the self-check also lets the engine read prompt + reference output, 16 positions per pass, without generating
anything (teacher forcing). At each of the 285 positions it compares the most likely next token and the NLL of the token that follows with what
the RTX 5090 computed there. It passes when the most likely token is the same at 98 % of the positions or more and does not change where the
RTX 5090 preferred it by 2 nats or more, and when, at the 195 positions of the reference output, the NLL differs by 0.015 or less on average and
by 0.25 or less at any one position. This is checked also when the output is the same. Exit codes: 0 passed (on another GPU: the same
output and predictions within the tolerances), 1 failed, 2 no reference file, 3 another GPU whose output differs but whose predictions are
within the tolerances. On the RTX 5090 the comparison with the reference stays exact, as before. For scale, measured on the RTX 5090 by the same comparison
(`docs/MEASUREMENTS.md`): the 8-bit key-value cache instead of KV5 differs by 0.0056 on average and 0.081 at most (99.6 % the same token)
and passes; the fastest model's weights against the quality model's reference differ by 0.028 and 0.53 (97.2 %) and fail. The tolerances
were chosen on the RTX 5090; the check has not been run on another GPU yet, so a card that computes correctly could still fail it, or a
small fault could pass it: on the RTX 5090 it did not find one matrix scaled by 1.25, all 400 layer matrices scaled by 1.02, or one bit changed in
every 4,096th byte of 64 matrices.

Since 0.1.4 every GPU, the RTX 5090 included, then also checks itself, with no reference and no tolerance (the *self-consistency* lines).
It generates the answer of a prompt that copies most of its input (60 short JSON lines with two values changed, thinking off; where the check
opened the wide copy passes the copied part is checked in passes of up to 64 positions), reads prompt + answer again in passes of 16 positions, and
every generated token must be the most likely token at its position. Where passes of more than 16 positions are used, it also reads the
text in passes of 64, and every prediction (the most likely token and the NLL, bit for bit) must be the same as in passes of 16 (since
0.1.7: when the check of the wide copy passes found a difference at start, the self-consistency check fails). The engine
computes passes of every width with the same integer sums in the same order, so on a card that computes correctly both comparisons agree
exactly; any difference fails the self-check (exit code 1). This finds a card whose passes of different widths do not agree; damaged
weights give the same wrong answer in both readings, which is what `yasakani verify` and the comparison with the reference are for. On the
RTX 5090 it takes 3.5 to 3.7 s (prompt 1,127 tokens, answer 1,080 tokens, 15 of its 27 passes wider than 16 positions) and passed with both
models.

The predictions of the RTX 5090 for the two published `selfcheck.json` files are in the package (`yasakani/_selfcheck_ref.py`).
`selfcheck --write` writes the full output and the predictions into a new reference file, and the name of the GPU (information only). Write
reference files on an RTX 5090: other GPUs compare their predictions with the file as if it came from one (on another GPU `--write` prints a
note). A reference file written before 0.1.4 has neither; with it, another GPU passes only with the same output.

The three prompts are short on purpose. Output is not guaranteed to be bit-identical from run to run while other programs use the GPU at the
same time (see `docs/MEASUREMENTS.md` in the repository). In our checks the three self-check requests were identical in all 180 repeats and
the command passed 12 of 12 times, but if a case fails, run the command once more before you report it.

## Use it from Python

```python
from yasakani import Engine, ModelPack
from yasakani.chat import ChatParams, ChatRunner
from yasakani.textio import ChatFormatter

pack = ModelPack.load(r"D:\models\qwen38-quality")
with Engine(pack, max_ctx=32768, dll=r"D:\yasakani\yskn.dll") as eng:
    runner = ChatRunner(eng, ChatFormatter(pack.root))
    runner.warmup()
    params = ChatParams(messages=[{"role": "user", "content": "Hello!"}], max_tokens=512, template_kwargs={"enable_thinking": False})
    usage = runner.run(params, lambda kind, text: print(text, end="", flush=True))   # kind is "reasoning" or "content"
    print(usage.completion_tokens, "tokens")
```

`Engine.generate(prompt_ids, max_new, on_tokens=...)` works at the id level; `Engine.cancel()` stops a running generation from any thread.

## Settings and environment

The package sets the engine's environment variables itself (`yasakani.runtime.DEFAULT_ENV`) and removes any other `YSKN_*` variable left in your shell
before it opens the model, because a stray setting silently changes the engine. Since 0.1.7 (first in the 0.1.7 test build) `YSKN_TEST_*` variables are kept (with a note at
start): only an engine library built with the test exports reads them, such as `YSKN_TEST_CODE_ARCH=86`, which makes such a library take
the decisions of sm_86 code; the released library ignores them. The published results were measured with the defaults; changing them
means running an engine configuration that the self-check and the published numbers do not cover.

0.1.4 adds the settings of the saved prompt states to `DEFAULT_ENV` (`YSKN_SNAP_SLOTS` 8, `YSKN_SNAP_STEP` 1024, `YSKN_SNAP_TAIL` 4096,
`YSKN_SNAP_SPARSE` 65536, `YSKN_SNAP_MIN` 8192) and two look-ahead settings to `DEFAULT_SPEC`: `copy_rows` (64: copy passes of up to 64 rows,
used where the check on the card opened them; up to 0.1.6 on compute capability 12.x only) and `copy_index` (1: the copy index with the cursor). `copy_rows=16, copy_index=0` is the look-ahead of 0.1.3. These values are the current defaults; they
have not been tuned. An older package that does not set `YSKN_SNAP_SLOTS` gets one saved state, as with 0.1.3.

0.2.0 raises `YSKN_Q35_PMAX` from 1024 to 2048 and `YSKN_Q35_LAYER_ROWS` from 4096 to 8192 (the sizes of the new prompt reading), and
the commands add the settings of [Prompt reading and edit copy](#prompt-reading-and-edit-copy): on cards under 30,000 MiB the sizes of
0.1.7, with the reading of 0.1.7 also `YSKN_Q35_PF_V2=0`, and `YSKN_EDIT_COPY`. A value of these that you set in your shell is not used
and is named in the note on ignored settings.

## Troubleshooting

* *engine library not found / could not be loaded* - give `--dll`, and install the Visual C++ runtime linked above.
* *could not open the model ... not enough free GPU memory* - close other programs that use the GPU, or lower `--max-ctx`.
* *the model does not fit in the free GPU memory* (0.1.3, before loading) - close other programs that use the GPU; on a 16 GB card use the
  fastest model; `--max-ctx` tries a given length anyway.
* *much slower than the published numbers* - look for the warning *the GPU memory is almost full* at start. When other programs use GPU memory
  and it runs out, Windows keeps part of the model in ordinary memory; there is no error, but generation becomes several times slower.
  Close the other programs, or lower `--max-ctx`.
* *only thinking, no answer* - the length limit was reached while the model was still thinking; see the note on `max_tokens` above
  (`/effort medium`, `/effort low` or `/think off` in `yasakani chat`).
* *cannot listen on 127.0.0.1:8765* - another program (or a second server) uses the port; pick another with `--port`.
* *output differs in selfcheck* - run `yasakani verify` first (a damaged download is the most common cause), then run the self-check once more.
  On a GPU other than the RTX 5090, *self-check FAILED* after the comparison of the predictions, or after the self-consistency lines on any
  GPU, means, when `verify` passes, that the engine probably does not compute correctly on that GPU: please report it with the output of
  `yasakani info` and of the self-check.

## Tests

```
python -m unittest        # no GPU needed: the engine library is replaced by a small stand-in (tests/stub)
```

Set `YASAKANI_TEST_MODEL_DIR` to a folder containing the Qwen3.8 `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` to run the chat, server and
command-line tests as well.

## License

Apache License 2.0.
