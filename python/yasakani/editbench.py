# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""``yasakani bench --edit`` (0.1.6): the three edit tasks of the speed tables of the repository's README, run on this PC.

The tasks are package data (``edit_tasks.json``): the code edit and the JSONL edit of the README table (``bench/20261103/prompts/
cases.json``) and the edit of 160 JSONL records of "What 0.1.4 changed" (``bench/20261115/prompts/cases_plus.json``), each with its
output limit and its expected answer.  As in the measurement of the tables: greedy, thinking off, the limit of the task, until the
model ends its answer.  ``--think`` (thinking on) samples as the server does by default: temperature 1.0, top-p 0.95, top-k 20,
min-p 0, one seed per run counted up from ``--seed0`` (default 1), with the server's default output limit of 32,768 tokens for every
task.  So its tok/s varies from run to run, and the thinking part, which mostly does not copy the input, is not sped up much; the
thinking columns of the README tables were measured otherwise and are not what ``--think`` reproduces.

* Speed = (tokens - 1) / seconds, the seconds from the first to the last generated token (the engine's own time: the time to read the
  prompt is not included).  The tables were measured through the server's stream, so these numbers are close to them, not the same.
* Tokens per pass = tokens / passes of the main model, as in ``yasakani bench``.
* The answer is compared with the expected one as the measurement client did: white space around it and a Markdown code fence are
  ignored; otherwise it must be the same text.
* Wide copy passes (0.1.4: a pass that copies the input checks up to 64 tokens) are used on compute capability 12.x (RTX 50 series)
  only, and were measured on the RTX 5090 only.  The header says whether this GPU uses them; the ``wide`` column counts them.
  0.1.7 (test build): with an engine library of 0.1.7 the header gives the widths the engine has open on this GPU (``--wide``,
  yasakani/widecheck.py).
"""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ._native import YasakaniError

TASKS_FILE = Path(__file__).with_name("edit_tasks.json")
FORMAT = "yasakani-edit-tasks-1"
THINK_MAX_TOKENS = 32768                                       # --think: the output limit of every task (the server's default)
THINK_TEMPERATURE, THINK_TOP_P, THINK_TOP_K = 1.0, 0.95, 20    # --think: the package's default sampling (chat.DEFAULT_*), min-p 0


@dataclass(frozen=True)
class EditTask:
    name: str
    title: str
    max_tokens: int
    messages: List[dict]
    expected: str


def load_tasks(path=None) -> List[EditTask]:
    """The tasks of ``edit_tasks.json``; every expected answer is checked against its SHA-256."""
    p = Path(path) if path else TASKS_FILE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise YasakaniError("cannot read the edit tasks (%s): %s" % (p.name, ex))
    if data.get("format") != FORMAT:
        raise YasakaniError("%s: unknown format %r" % (p.name, data.get("format")))
    out = []
    for t in data.get("tasks") or []:
        exp = t["expected"]
        if hashlib.sha256(exp.encode("utf-8")).hexdigest() != t["expected_sha256"]:
            raise YasakaniError("%s: the expected answer of %s does not match its checksum" % (p.name, t["name"]))
        out.append(EditTask(name=t["name"], title=t["title"], max_tokens=int(t["max_tokens"]), messages=t["messages"], expected=exp))
    if not out:
        raise YasakaniError("%s holds no tasks" % p.name)
    return out


def answer_of(text: str) -> str:
    """The answer as the measurement client read it: white space around it and one Markdown code fence dropped."""
    got = text.strip()
    if got.startswith("```"):
        got = got.split("\n", 1)[1] if "\n" in got else got
        got = got.rsplit("```", 1)[0] if got.rstrip().endswith("```") else got
    return got.strip()


def compare(expected: str, text: str) -> Tuple[bool, int]:
    """(same as the expected answer, number of differing lines: removed + added)."""
    exp, got = expected.strip(), answer_of(text)
    if got == exp:
        return True, 0
    diff = [ln for ln in difflib.unified_diff(exp.split("\n"), got.split("\n"), lineterm="", n=0)
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    return False, len(diff)


def max_tokens_of(task: EditTask, think: bool) -> int:
    """The output limit of a run: the task's own (that of the tables) with thinking off, 32,768 for every task with ``--think``."""
    return THINK_MAX_TOKENS if think else task.max_tokens


def sampling_of(think: bool, seed: int = 0):
    """Greedy with thinking off; with ``--think`` temperature 1.0, top-p 0.95, top-k 20, min-p 0 and ``seed``."""
    from .engine import Sampling
    if not think:
        return Sampling(temperature=0.0)
    return Sampling(THINK_TEMPERATURE, THINK_TOP_P, THINK_TOP_K, 0.0, int(seed))


def wide_status(engine, gpu) -> Tuple[str, str]:
    """("on" | "off" | "unknown", one line): does the engine use wide copy passes on this GPU?  ``gpu`` is the GpuInfo of nvidia-smi
    (None when it is not known which GPU the engine uses)."""
    rows = int(engine.spec.get("copy_rows", 16))
    if "yskn_prepare_wide" not in engine.lib.fn or rows <= 16:
        return "off", "wide copy passes: off (this engine library or setting copies with passes of up to 16 tokens)"
    st = engine.wide_state() if callable(getattr(engine, "wide_state", None)) else None
    if st is not None:                                           # 0.1.7: what the engine has open (yasakani/widecheck.py)
        from .widecheck import widths_text
        mask = st.mask if st.allowed else 0
        if not mask:
            return "off", ("wide copy passes: off (code %s, compute capability %s); this GPU copies with passes of up to 16 tokens"
                           % (st.code_text, st.cc_text))
        return "on", ("wide copy passes: on, %s (code %s, compute capability %s); measured on the RTX 5090 only, checked on this GPU "
                      "by yasakani/widecheck.py unless --wide force" % (widths_text(mask), st.code_text, st.cc_text))
    cc = getattr(gpu, "compute_cap", None)
    if cc is None:
        return "unknown", ("wide copy passes: not known (nvidia-smi gave no compute capability); they are used on compute capability "
                           "12.x (RTX 50 series) only.  The column \"wide\" counts the passes that were wide")
    if cc[0] >= 12:
        return "on", ("wide copy passes: on (compute capability %d.%d).  They are used on compute capability 12.x (RTX 50 series) only, "
                      "and were measured on the RTX 5090 only" % cc)
    return "off", ("wide copy passes: off (compute capability %d.%d).  They are used on compute capability 12.x (RTX 50 series) "
                   "only; this GPU copies with passes of up to 16 tokens" % cc)


def header(version: str, pack, engine, gpu, gpu_lines: Sequence[str], think: bool, runs: int, seed0: int = 1) -> Dict[str, object]:
    """What a pasted result needs to say about itself: package, engine library, GPU, model, wide copy passes, settings (with
    ``--think``: the sampling, the first seed ``seed0`` and the output limit)."""
    dll = Path(engine.lib.path)
    status, line = wide_status(engine, gpu)
    g = None
    if gpu is not None:
        g = dict(index=gpu.index, name=gpu.name, compute_capability=gpu.cc_text if gpu.compute_cap else None, driver=gpu.driver)
    return dict(test="edit", version=version, engine=dict(file=dll.name, sha256=hashlib.sha256(dll.read_bytes()).hexdigest()),
                gpu=g, gpu_lines=list(gpu_lines), model=pack.model_id, weights=pack.weights.name, context=engine.max_ctx,
                wide_copy_passes=status, wide_copy_passes_note=line, thinking=bool(think), runs_per_task=int(runs), greedy=not think,
                sampling=None if not think else dict(temperature=THINK_TEMPERATURE, top_p=THINK_TOP_P, top_k=THINK_TOP_K, min_p=0.0,
                                                     first_seed=int(seed0)),
                output_limit=THINK_MAX_TOKENS if think else None)          # None: greedy, and each task's own limit (rows: max_tokens)


def header_lines(h: Dict[str, object]) -> List[str]:
    g = h["gpu"]
    if g:
        gl = "GPU %d: %s, compute capability %s, driver %s" % (g["index"], g["name"], g["compute_capability"] or "?", g["driver"] or "?")
    else:
        gl = "; ".join(h["gpu_lines"]) or "not known"
    if h["thinking"]:
        sm = h["sampling"]
        how = ("thinking on: temperature %s, top-p %s, top-k %d, seeds from %d (one per run), output limit %d tokens"
               % (sm["temperature"], sm["top_p"], sm["top_k"], sm["first_seed"], h["output_limit"]))
    else:
        how = "greedy, thinking off"
    n = h["runs_per_task"]
    return ["yasakani bench --edit: the 3 edit tasks of the speed tables (%s, %d run%s each)" % (how, n, "" if n == 1 else "s"),
            "package  yasakani %s" % h["version"],
            "engine   %s  sha256 %s" % (h["engine"]["file"], h["engine"]["sha256"][:16]),
            "gpu      " + gl,
            "model    %s  (context %d tokens)" % (h["weights"], h["context"]),
            h["wide_copy_passes_note"],
            "%-17s %3s %7s %7s %8s %8s %11s %5s  %s" % ("task", "run", "prompt", "tokens", "seconds", "tok/s", "tokens/pass", "wide",
                                                        "answer")]


def row_line(r: Dict[str, object]) -> str:
    ans = "= expected" if r["same"] else "differs (%d lines)" % r["diff_lines"]
    if r["finish"] != "stop":
        ans += ", hit the limit of %d tokens" % r["max_tokens"]
    return "%-17s %3d %7d %7d %8.3f %8.1f %11.2f %5d  %s" % (r["title"], r["run"], r["prompt_tokens"], r["tokens"], r["seconds"],
                                                           r["tokens_per_s"], r["tokens_per_pass"], r["wide_passes"], ans)


def run_task(runner, task: EditTask, think: bool, run: int) -> Dict[str, object]:
    """One generation of one task from a fresh engine state (the prompt is read in full): greedy with the task's limit, or with
    ``--think`` sampled with the next seed of the runner (``--seed0``, then one more per run) and the limit 32,768."""
    import numpy as np
    from .chat import ChatParams
    from .textio import OutputStream
    eng, fmt = runner.engine, runner.fmt
    limit, s = max_tokens_of(task, think), sampling_of(think, runner.next_seed() if think else 0)
    params = ChatParams(messages=[dict(m) for m in task.messages], max_tokens=limit, temperature=s.temperature, top_p=s.top_p,
                        top_k=s.top_k, min_p=s.min_p, seed=s.seed, loop_guard=False, template_kwargs={"enable_thinking": bool(think)})
    prompt, in_reasoning = runner.prepare_checked(params)
    eng.reset()
    eng.set_sampling(s)
    r = eng.generate(prompt, limit)
    stream = OutputStream(fmt, in_reasoning)
    content = "".join(t for kind, t in stream.feed(np.asarray(r.ids, np.int32), final=True) if kind == "content")
    n = len(r.ids)
    reasoning = stream.think_end_at if stream.think_end_at is not None else (n if stream.in_reasoning else 0)
    secs = r.gen_ms / 1000.0
    same, diff_lines = compare(task.expected, content)
    return dict(task=task.name, title=task.title, run=run, prompt_tokens=len(prompt), tokens=n, reasoning_tokens=int(reasoning),
                seconds=round(secs, 3), tokens_per_s=round((n - 1) / secs, 1) if (n > 1 and secs > 0) else 0.0,
                tokens_per_pass=round(n / max(1, r.passes), 2), passes=int(r.passes),
                wide_passes=int(r.stats.get("wide_passes", 0)), finish=r.finish, max_tokens=limit, seed=s.seed if think else None,
                same=same, diff_lines=diff_lines, ids_sha256=hashlib.sha256(np.asarray(r.ids, "<i4").tobytes()).hexdigest(),
                # 0.1.7 (0 with older engine libraries): back-offs of the copy loop and its final times of copy cycles (16..64 rows, ms)
                wide_backoffs=int(r.stats.get("wide_backoffs", 0)),
                t_copy=[round(float(r.stats.get("t_copy_%d" % k, 0.0)), 2) for k in (16, 32, 48, 64)])


def needed_context(runner, task: EditTask, think: bool) -> int:
    """Tokens of context the task needs: its prompt, its output limit (32,768 with ``--think``) and the engine's margin."""
    from .chat import ChatParams
    from .engine import CONTEXT_MARGIN
    prompt, _ = runner.prepare(ChatParams(messages=[dict(m) for m in task.messages], template_kwargs={"enable_thinking": bool(think)}))
    return len(prompt) + max_tokens_of(task, think) + CONTEXT_MARGIN


def run(runner, tasks: Sequence[EditTask], *, think: bool = False, runs: int = 1,
        on_row: Optional[Callable[[Dict[str, object]], None]] = None,
        on_skip: Optional[Callable[[EditTask, int], None]] = None) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Every task ``runs`` times.  A task whose prompt and output limit do not fit the engine's context is skipped.
    Returns (rows, skipped)."""
    rows, skipped = [], []
    for task in tasks:
        need = needed_context(runner, task, think)
        if need > runner.engine.max_ctx:
            skipped.append(dict(task=task.name, title=task.title, needs_context=need))
            if on_skip:
                on_skip(task, need)
            continue
        for k in range(max(1, int(runs))):
            row = run_task(runner, task, think, k + 1)
            rows.append(row)
            if on_row:
                on_row(row)
    return rows, skipped


def means(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Mean tok/s of each task (in the order of the rows)."""
    out: Dict[str, List[float]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(r["tokens_per_s"])
    return {k: round(sum(v) / len(v), 1) for k, v in out.items()}


FOOTER = ("tok/s = (tokens - 1) / seconds, the seconds from the first to the last generated token (reading the prompt not included).  "
          "The README tables were measured through the server's stream, 2 runs per task.")
FOOTER_THINK = ("tok/s = (tokens - 1) / seconds, the seconds from the first to the last generated token (reading the prompt not "
                "included), the thinking included.  With --think the model samples (temperature 1.0, output limit 32768 tokens), so "
                "tok/s varies from run to run, and the thinking part is not sped up much.  The thinking columns of the README tables "
                "were measured otherwise; this does not reproduce them.")


def footer(think: bool) -> str:
    """The last line: what the numbers are, and how the tables were measured (thinking off) or why they are not reproduced."""
    return FOOTER_THINK if think else FOOTER
