# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The self-check: three short fixed greedy prompts, compared with a reference file (``selfcheck.json`` in the model folder).

On the RTX 5090, the card the published references were made on, the output must be the same as the reference id for id.

On other GPUs the engine runs other code (it is built for compute capability 8.6, 8.9 and 12.x), and tiny differences in the
computation can make greedy decoding take another id where two ids are almost equally likely; from there on the output is a
different text.  So the output alone proves little there, either way.  Since 0.1.4 the self-check on those GPUs also lets the engine
read the reference output (teacher forcing: prompt + reference output are given, nothing is generated) and compares, position by
position, what it predicts with what the RTX 5090 predicted at the same positions:

* the most likely next id must be the same at 98 % of the positions or more, and it must not change at any position where the
  RTX 5090 preferred its id by 2 nats (a factor of e^2, about 7.4) or more;
* at the positions of the reference output, the negative log-likelihood (NLL) of the reference id may differ by 0.015 on average
  and by 0.25 at most, and the engine must give a number (not NaN) at every position.

A card that passes computes the same predictions as the RTX 5090 up to small differences: exit code 0 when its output is the same
as the reference, 3 when it differs.  A card that does not pass fails (exit code 1), also when its output happens to be the same.

For scale, measured on the RTX 5090 with the quality model (mean and largest NLL difference at the output positions, share of the
same most likely id): the 8-bit key-value cache (YSKN_TB_CAP=0) instead of KV5 gives 0.0056 / 0.081 / 99.6 % and passes; the
fastest model's weights read against the quality model's reference give 0.028 / 0.53 / 97.2 % and fail.

The predictions of the RTX 5090 are written into new reference files by ``yasakani selfcheck --write``.  For the two published
reference files (made before this check existed), the package carries them (``_selfcheck_ref.py``).

Since 0.1.4 every GPU, the RTX 5090 included, also checks itself, without a reference (``consistency``): it generates the answer of a
prompt that copies most of its input (greedy; on GPUs of compute capability 12.x such answers are checked in passes of up to 64
positions; 0.1.7 test build: on RTX 30 / 40 too with ``--wide on`` where the check of yasakani/widecheck.py opened them), then reads prompt + answer again in passes of 16 positions, and every generated id must be the most likely id at its
position.  Where passes wider than 16 positions can be used, it also reads the text in passes of 64, and every prediction (most likely
id and NLL, bit for bit) must be the same as with passes of 16.  The engine computes these passes with the same sums in the same
order, so on a card that computes correctly both comparisons are exact; no tolerance is involved.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

FORMAT = 1                     # new files add "ids" and "forced" to every case; readers of format 1 ignore them
ROWS = 16                      # positions per teacher-forced pass
WIDE_ROWS = 64                 # the widest pass of the engine (the self-consistency check)
TOLERANCE = dict(
    top1=0.98,                 # share of all positions where the most likely id is the RTX 5090's, at least
    sure_margin=2.0,           # nats: where the RTX 5090 preferred its id by this much, it must not change
    output_mean_nll=0.015,     # mean |NLL difference| at the positions of the reference output, at most
    output_max_nll=0.25,       # largest |NLL difference| there, at most
)
NLL_DIGITS, MARGIN_DIGITS = 5, 3


def cases() -> List[dict]:
    return [
        dict(name="short-answer", kind="chat", messages=[{"role": "user", "content": "Say hello in one short sentence."}],
             template_kwargs={"enable_thinking": False}, max_new=48),
        dict(name="thinking", kind="chat", messages=[{"role": "user", "content": "What is 17 times 23? Think it through step by step."}],
             template_kwargs={}, max_new=160),
        dict(name="plain-text", kind="raw", text="The capital of France is", max_new=32),
    ]


def ids_sha(ids) -> str:
    import numpy as np
    return hashlib.sha256(np.asarray(ids, "<i4").tobytes()).hexdigest()


def prompt_ids(fmt, c) -> List[int]:
    p = fmt.encode(c["text"]) if c["kind"] == "raw" else fmt.prompt(c["messages"], **c["template_kwargs"])[0]
    return [int(x) for x in p]


def run(eng, fmt) -> List[dict]:
    """Greedy runs of the fixed cases -> one record per case (prompt hash, output hash, first ids).  The prompt and output ids are
    kept under "_prompt" and "_ids" (not written)."""
    from .engine import Sampling
    eng.set_sampling(Sampling(temperature=0.0))
    out = []
    for c in cases():
        prompt = prompt_ids(fmt, c)
        eng.reset()
        r = eng.generate(prompt, c["max_new"])
        out.append(dict(name=c["name"], prompt_tokens=int(len(prompt)), prompt_sha256=ids_sha(prompt), tokens=len(r.ids),
                        sha256=ids_sha(r.ids), head=[int(x) for x in r.ids[:12]], _prompt=prompt, _ids=[int(x) for x in r.ids]))
    eng.reset()
    return out


# ---- reference files ------------------------------------------------------------------------------------------------------------
def forced_record(eng, prompt: Sequence[int], ids: Sequence[int]) -> dict:
    """What this installation predicts while it reads prompt + ids (written into reference files)."""
    f = eng.teacher_forced(list(prompt) + list(ids), rows=ROWS, margins=True)
    return dict(rows=ROWS, argmax=[int(x) for x in f.argmax], nll=[round(float(x), NLL_DIGITS) for x in f.nll],
                margin=[round(float(x), MARGIN_DIGITS) for x in f.margin])


def dumps(obj) -> str:
    """JSON with one key per line and every list of numbers on one line."""
    text = json.dumps(obj, indent=1)
    return re.sub(r"\[\s*([-+0-9.eE,\s]*?)\s*\]", lambda m: "[" + ",".join(x.strip() for x in m.group(1).split(",") if x.strip()) + "]", text)


def reference_record(eng, got: List[dict], model_id: str, version: str, gpu: Optional[str] = None) -> dict:
    """The reference file of this installation.  ``gpu`` (the name nvidia-smi gives, when known) is kept as information only."""
    cs = []
    for g in got:
        c = {k: v for k, v in g.items() if not k.startswith("_")}
        c["ids"] = list(g["_ids"])
        c["forced"] = forced_record(eng, g["_prompt"], g["_ids"])
        cs.append(c)
    rec = dict(format=FORMAT, model_id=model_id, yasakani=version)
    if gpu:
        rec["gpu"] = gpu
    rec["cases"] = cs
    return rec


def reference_key(ref: dict) -> str:
    """Names the published reference files by what they check (not by their bytes)."""
    lines = sorted("%s\t%s\t%s\t%s" % (c.get("name"), c.get("prompt_sha256"), c.get("tokens"), c.get("sha256")) for c in ref.get("cases", []))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:32]


def full_data(ref: dict) -> Dict[str, dict]:
    """name -> {"ids", "forced"} for every case of ``ref`` whose full output and RTX 5090 predictions are known: from the file itself
    (written by 0.1.4 or newer) or, for the published files, from the package.  Data that do not match the case's hashes are left out."""
    from ._selfcheck_ref import KNOWN
    builtin = (KNOWN.get(reference_key(ref)) or {}).get("cases", {})
    out = {}
    for c in ref.get("cases", []):
        for src in (c, builtin.get(c.get("name")) or {}):
            ids, forced = src.get("ids"), src.get("forced")
            if not isinstance(ids, list) or not isinstance(forced, dict):
                continue
            try:
                n = int(c.get("prompt_tokens", -1)) + len(ids) - 1
                ok = (len(ids) == c.get("tokens") and ids_sha(ids) == c.get("sha256") and forced.get("rows") == ROWS
                      and all(isinstance(forced.get(k), list) and len(forced[k]) == n for k in ("argmax", "nll", "margin")))
            except (TypeError, ValueError, OverflowError):
                ok = False
            if ok:
                out[c["name"]] = dict(ids=[int(x) for x in ids], forced=forced)
                break
    return out


# ---- the close check (GPUs other than the RTX 5090) --------------------------------------------------------------------------------
@dataclass
class CaseCompare:
    name: str
    positions: int
    same_top1: int
    output_positions: int
    output_mean_nll: float
    output_max_nll: float
    sure_changed: int
    nonfinite: int = 0                     # positions where the engine gave NaN or infinity as the NLL
    first_diff: Optional[int] = None       # first generated id that differs from the reference (0-based), None when the same


@dataclass
class CloseResult:
    cases: List[CaseCompare] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def positions(self) -> int:
        return sum(c.positions for c in self.cases)

    @property
    def top1(self) -> float:
        return sum(c.same_top1 for c in self.cases) / max(1, self.positions)

    @property
    def output_positions(self) -> int:
        return sum(c.output_positions for c in self.cases)

    @property
    def output_mean_nll(self) -> float:
        return sum(c.output_mean_nll * c.output_positions for c in self.cases) / max(1, self.output_positions)

    @property
    def output_max_nll(self) -> float:
        return max((c.output_max_nll for c in self.cases), default=0.0)

    @property
    def sure_changed(self) -> int:
        return sum(c.sure_changed for c in self.cases)

    @property
    def nonfinite(self) -> int:
        return sum(c.nonfinite for c in self.cases)

    def failures(self, tol: Optional[Dict[str, float]] = None) -> List[str]:
        """Empty when every tolerance holds; else one short text per tolerance that does not."""
        t = dict(TOLERANCE, **(tol or {}))
        bad = list(self.problems)
        if not self.cases:
            return bad or ["nothing was compared"]
        if self.nonfinite:
            bad.append("the engine gave no number (NaN or infinity) as the NLL at %d position(s)" % self.nonfinite)
        if self.top1 < t["top1"]:
            bad.append("the most likely id is the same at only %.1f %% of the positions (at least %.0f %% needed)" % (100 * self.top1, 100 * t["top1"]))
        if self.sure_changed:
            bad.append("the most likely id changed at %d position(s) where the RTX 5090 preferred its id by %.0f nats or more"
                       % (self.sure_changed, t["sure_margin"]))
        if self.output_mean_nll > t["output_mean_nll"]:
            bad.append("the NLL of the reference output differs by %.4f on average (at most %.3f allowed)" % (self.output_mean_nll, t["output_mean_nll"]))
        if self.output_max_nll > t["output_max_nll"]:
            bad.append("the NLL of the reference output differs by %.3f at one position (at most %.2f allowed)" % (self.output_max_nll, t["output_max_nll"]))
        return bad


def compare_case(name: str, n_prompt: int, ref_forced: dict, argmax, nll, sure_margin: float = TOLERANCE["sure_margin"],
                 first_diff: Optional[int] = None) -> CaseCompare:
    """This GPU's teacher-forced predictions (``argmax``, ``nll`` over prompt + reference output) against the RTX 5090's."""
    import numpy as np
    a0 = np.asarray(ref_forced["argmax"], np.int64)
    n0 = np.asarray(ref_forced["nll"], np.float64)
    m0 = np.asarray(ref_forced["margin"], np.float64)
    a1 = np.asarray(argmax, np.int64)
    n1 = np.asarray(nll, np.float64)
    if not (len(a0) == len(n0) == len(m0) == len(a1) == len(n1)):
        raise ValueError("%s: the predictions do not cover the same positions" % name)
    bad = ~np.isfinite(n1)
    d = np.abs(n1 - n0)
    d[bad] = np.inf                                          # a NaN from the GPU is the largest difference there is
    out = d[max(0, n_prompt - 1):]
    diff = a1 != a0
    return CaseCompare(name=name, positions=len(a0), same_top1=int((~diff).sum()), output_positions=len(out),
                       output_mean_nll=float(out.mean()) if len(out) else 0.0, output_max_nll=float(out.max()) if len(out) else 0.0,
                       sure_changed=int((diff & (m0 >= sure_margin)).sum()), nonfinite=int(bad.sum()), first_diff=first_diff)


def close_check(eng, got: List[dict], ref: dict) -> CloseResult:
    """Teacher forcing of every case's reference output on this GPU, compared with the RTX 5090's predictions."""
    res = CloseResult()
    data = full_data(ref)
    if not eng.supports_forced:
        res.problems.append("the engine library cannot read a given text (no yskn_verify_nll export)")
        return res
    for g in got:
        dat = data.get(g["name"])
        if dat is None:
            res.problems.append("%s: the reference file has no full output to compare with (write it again with yasakani 0.1.4 or newer)" % g["name"])
            continue
        ids = dat["ids"]
        mine = g.get("_ids") or []
        fd = None if mine == ids else next((i for i, (x, y) in enumerate(zip(mine, ids)) if x != y), min(len(mine), len(ids)))
        f = eng.teacher_forced(list(g["_prompt"]) + ids, rows=ROWS)
        res.cases.append(compare_case(g["name"], len(g["_prompt"]), dat["forced"], f.argmax, f.nll, first_diff=fd))
    return res


def close_lines(res: CloseResult) -> List[str]:
    t = TOLERANCE
    out = []
    for c in res.cases:
        out.append("  %-14s %3d positions: same most likely id at %d; reference output: NLL differs by %.4f on average, %.4f at most%s"
                   % (c.name, c.positions, c.same_top1, c.output_mean_nll, c.output_max_nll,
                      "" if c.first_diff is None else "  (generated output differs from id #%d)" % (c.first_diff + 1)))
    if res.cases:
        out.append("  all %d positions: same most likely id at %.1f %% (at least %.0f %%), changed where the RTX 5090 was sure (by %.0f nats "
                   "or more): %d (none allowed)" % (res.positions, 100 * res.top1, 100 * t["top1"], t["sure_margin"], res.sure_changed))
        out.append("  the %d positions of the reference output: NLL differs by %.4f on average (at most %.3f) and %.4f at most (at most %.2f)"
                   % (res.output_positions, res.output_mean_nll, t["output_mean_nll"], res.output_max_nll, t["output_max_nll"]))
    return out


# ---- the self-consistency check (every GPU, since 0.1.4) --------------------------------------------------------------------------
def copy_case() -> dict:
    """A prompt whose answer copies most of the input (60 short JSON lines, two values changed), thinking off."""
    recs = ['{"id":"R-%03d","qty":%d,"status":"%s"}' % (i, 1 + i % 7, ("open", "done", "hold")[i % 3]) for i in range(1, 61)]
    return dict(name="copy-edit", kind="chat", template_kwargs={"enable_thinking": False}, max_new=1400,
                messages=[{"role": "user", "content": "Return these 60 JSONL lines exactly, but set status to \"done\" in R-017 and "
                           "R-044. Output only the lines.\n\n" + "\n".join(recs)}])


@dataclass
class Consistency:
    name: str = "copy-edit"
    prompt_tokens: int = 0
    tokens: int = 0
    passes: int = 0
    wide_passes: int = 0                   # generation passes of more than 16 positions (engine statistics)
    max_rows: int = 0                      # the widest generation pass
    compared: int = 0                      # generated ids compared with the teacher-forced predictions
    mismatches: int = 0                    # generated ids that are not the most likely id when the text is read in passes of 16
    first_mismatch: Optional[int] = None   # 0-based index of the first such id
    wide_checked: bool = False             # the text was also read in passes of 64 and compared with the passes of 16
    wide_positions: int = 0
    wide_argmax_diff: int = 0
    wide_nll_diff: int = 0                 # positions whose NLL is not the same 32-bit float
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def failures(self) -> List[str]:
        bad = list(self.problems)
        if self.mismatches:
            bad.append("%d of the %d generated ids are not the most likely id when this GPU reads the same text again (first: id #%d)"
                       % (self.mismatches, self.tokens, self.first_mismatch + 1))
        if self.wide_checked and (self.wide_argmax_diff or self.wide_nll_diff):
            bad.append("passes of %d positions do not give the same predictions as passes of %d (most likely id differs at %d, NLL "
                       "at %d of %d positions)" % (WIDE_ROWS, ROWS, self.wide_argmax_diff, self.wide_nll_diff, self.wide_positions))
        return bad


def consistency(eng, fmt) -> Consistency:
    """Generate the answer of ``copy_case()`` (greedy) and check it against this GPU's own teacher-forced predictions (see the top of
    this module).  Needs the teacher-forcing exports (``eng.supports_forced``).  The state is reset before and after."""
    import numpy as np
    from .engine import Sampling, YasakaniError
    c = copy_case()
    res = Consistency(name=c["name"])
    prompt = prompt_ids(fmt, c)
    res.prompt_tokens = len(prompt)
    eng.set_sampling(Sampling(temperature=0.0))
    eng.reset()
    try:
        r = eng.generate(prompt, c["max_new"])
    finally:
        eng.reset()
    ids = [int(x) for x in r.ids]
    res.tokens, res.passes = len(ids), int(r.passes)
    res.wide_passes, res.max_rows = int(r.stats.get("wide_passes", 0)), int(r.stats.get("max_rows", 0))
    if not ids:
        res.problems.append("the copy prompt gave no output")
        return res
    seq = list(prompt) + ids
    try:
        f16 = eng.teacher_forced(seq, rows=ROWS)
    except ValueError as ex:                                 # prompt + answer do not fit the context: nothing to compare
        res.notes.append("not compared: %s" % ex)
        return res
    n0 = len(prompt)
    pred = [int(x) for x in f16.argmax[n0 - 1:n0 - 1 + len(ids)]]       # the prediction for ids[j] is made after reading ids[:j]
    mism = [j for j, (a, b) in enumerate(zip(pred, ids)) if a != b]
    res.compared = len(pred)
    res.mismatches = len(mism) + (len(ids) - len(pred))
    res.first_mismatch = mism[0] if mism else (len(pred) if res.mismatches else None)
    wr = getattr(eng, "wide_result", None)
    if wr is not None and getattr(wr, "status", "") == "mismatch":   # 0.1.7: the check at start found wide passes computing differently
        res.problems.append("the check of the wide copy passes at start found that passes of more than %d positions do not give the same "
                            "predictions as passes of %d on this GPU (%s); they are not used, but please report it on GitHub Issues"
                            % (ROWS, ROWS, "kept result of an earlier check" if getattr(wr, "cache_hit", False) else "checked now"))
        return res
    st = eng.wide_state() if callable(getattr(eng, "wide_state", None)) else None
    if st is not None and not (st.allowed and st.mask):         # 0.1.7: no wide copy pass is open on this GPU (--wide, widecheck)
        res.notes.append("this GPU does not use passes of more than %d positions (no wide copy pass is open)" % ROWS)
        return res
    try:
        f64 = eng.teacher_forced(seq, rows=WIDE_ROWS)
    except YasakaniError as ex:
        if res.max_rows > ROWS:                              # wide passes were used, so they must be readable
            res.problems.append("passes of %d positions were used in the generation, but reading the text in passes of %d failed: %s"
                                % (res.max_rows, WIDE_ROWS, ex))
        else:
            res.notes.append("this GPU does not use passes of more than %d positions" % ROWS)
        return res
    res.wide_checked = True
    res.wide_positions = len(f16.argmax)
    res.wide_argmax_diff = int((np.asarray(f64.argmax) != np.asarray(f16.argmax)).sum())
    res.wide_nll_diff = int((np.asarray(f64.nll, np.float32).view(np.uint32) != np.asarray(f16.nll, np.float32).view(np.uint32)).sum())
    return res


def consistency_lines(res: Consistency) -> List[str]:
    out = ["  %-14s prompt %d ids, answer %d ids in %d passes (%d of them wider than %d positions, the widest %d)"
           % (res.name, res.prompt_tokens, res.tokens, res.passes, res.wide_passes, ROWS, res.max_rows)]
    if res.compared:
        out.append("  read again in passes of %d: %s" % (ROWS, "every generated id is the most likely id (%d ids)" % res.compared
                                                         if not res.mismatches else
                                                         "%d generated id(s) are not the most likely id" % res.mismatches))
    if res.wide_checked:
        same = not (res.wide_argmax_diff or res.wide_nll_diff)
        out.append("  read again in passes of %d: %s" % (WIDE_ROWS, "the same predictions as in passes of %d, bit for bit (%d positions)"
                                                         % (ROWS, res.wide_positions) if same else
                                                         "most likely id differs at %d, NLL at %d of %d positions"
                                                         % (res.wide_argmax_diff, res.wide_nll_diff, res.wide_positions)))
    for n in res.notes:
        out.append("  note: " + n)
    return out
