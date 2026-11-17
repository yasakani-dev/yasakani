# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""``yasakani report`` (0.1.7 test build): how the wide copy passes do on this GPU, as Markdown to paste on GitHub.

1. Machine and versions: GPU, compute capability, memory, SMs, driver and its mode, power limit, clocks, whether a display is
   attached, OS, package, engine library (SHA-256 prefix and build info), the code the GPU runs, the model.
2. The check of yasakani/widecheck.py, measured again: correctness (passes of 17..64 rows against passes of 16, bit for bit) and the
   time of copy cycles of 16, 32, 48 and 64 rows against the limits.
3. The three edit tasks of ``yasakani bench --edit`` with the wide copy passes off, with the widths the check chose and with every
   width, in the order off, checked, all, all, checked, off (each column twice, so that a GPU that warms up or slows down affects
   every column alike), and whether every run gave the same ids.  A column that is the same setting as another is not run again.
4. The self-check: the output of the fixed prompts against the reference, and the self-consistency check with every width open
   (passes of 64 positions against 16, bit for bit).

Never in the report: the GPU's UUID, paths, the user or computer name, the environment.  The engine library is named by its file
name, the first 16 hex digits of its SHA-256 and its build info.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__, BUILD_NOTE
from . import widecheck
from ._native import YasakaniError

COLUMNS = ("off", "checked", "all")


def _say(msg: str) -> None:
    import sys
    print(msg, file=sys.stderr, flush=True)


def _abba(rounds: int) -> List[str]:
    return ["off", "checked", "all", "all", "checked", "off"] * max(1, int(rounds))


def _mismatch(res) -> bool:
    return res is not None and getattr(res, "status", "") == "mismatch"


def _switch(runner, mask: int, times) -> None:
    """Set the widths, build their graphs and run one short generation (its graphs too: on RTX 30 / 40 opening or closing every width
    changes how passes of 17..64 rows are computed, and the graphs are made again), then reset (no prompt snapshot is kept)."""
    eng = runner.engine
    eng.set_wide(mask, times)
    eng.prepare_wide()
    eng.reset()
    eng.generate(runner.fmt.encode("warm-up"), 33)
    eng.reset()


def bench_columns(eng, runner, tasks, res, st, rounds: int = 1, progress=None) -> Dict[str, object]:
    """The edit tasks with the widths of each column.  Returns {"masks", "rows", "same_as"}."""
    from . import editbench
    allm = 0 if _mismatch(res) else widecheck.mask_of(st.checkable_widths)    # after a mismatch no width is opened again
    masks = {"off": 0, "checked": res.mask if res is not None else allm, "all": allm}
    same_as = {}
    if masks["checked"] == masks["off"]:
        same_as["checked"] = "off"
    if masks["all"] == masks["checked"]:
        same_as["all"] = "checked"
    elif masks["all"] == masks["off"]:
        same_as["all"] = "off"
    times = (res.times_us or None) if res is not None else None
    rows = []
    for k, col in enumerate(_abba(rounds)):
        if col in same_as:
            continue
        _switch(runner, masks[col], times)
        for task in tasks:
            need = editbench.needed_context(runner, task, False)
            if need > eng.max_ctx:
                rows.append(dict(column=col, block=k, task=task.name, title=task.title, skipped=need))
                continue
            r = editbench.run_task(runner, task, False, k + 1)
            r.update(column=col, block=k)
            rows.append(r)
            if progress:
                progress("  %-8s %-17s %7.1f tok/s  %5d tokens  %3d wide passes  %s" % (
                    col, task.title, r["tokens_per_s"], r["tokens"], r["wide_passes"], "= expected" if r["same"] else "differs"))
    eng.set_wide(masks["checked"], times)
    eng.prepare_wide()
    return dict(masks=masks, rows=rows, same_as=same_as)


def selfcheck_part(eng, fmt, pack, st) -> Dict[str, object]:
    """The reference cases (same output or not, and on another GPU than the RTX 5090 the predictions within the tolerances) and the
    self-consistency check with every width open (none after a mismatch of the check: the self-consistency check then fails)."""
    from . import selfcheck as sc
    out: Dict[str, object] = {}
    got = sc.run(eng, fmt)
    ref_path = Path(pack.root) / "selfcheck.json"
    cases = []
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        want = {c["name"]: c for c in ref.get("cases", [])}
        differs = 0
        for g in got:
            w = want.get(g["name"])
            if w is None:
                cases.append(dict(name=g["name"], result="not in the reference"))
            elif w["sha256"] == g["sha256"] and w["tokens"] == g["tokens"]:
                cases.append(dict(name=g["name"], result="same", tokens=g["tokens"]))
            else:
                n = next((i for i, (a, b) in enumerate(zip(w["head"], g["head"])) if a != b), None)
                cases.append(dict(name=g["name"], result="differs", first_diff=None if n is None else n + 1))
                differs += 1
        out["reference"] = cases
        if differs and eng.supports_forced and len(sc.full_data(ref)) >= len(got):
            cr = sc.close_check(eng, got, ref)
            out["close"] = dict(lines=[ln.strip() for ln in sc.close_lines(cr)], failures=cr.failures())
    else:
        out["reference"] = [dict(name="-", result="no selfcheck.json next to the model files")]
    if eng.supports_forced and st is not None and st.allowed:
        keep = st.mask
        eng.set_wide(0 if _mismatch(eng.wide_result) else widecheck.mask_of(st.checkable_widths))
        try:
            c = sc.consistency(eng, fmt)
        finally:
            eng.set_wide(keep)
        out["consistency"] = dict(lines=[ln.strip() for ln in sc.consistency_lines(c)], failures=c.failures(),
                                  wide_checked=c.wide_checked, positions=c.wide_positions, argmax_diff=c.wide_argmax_diff,
                                  nll_diff=c.wide_nll_diff, wide_passes=c.wide_passes, max_rows=c.max_rows)
    return out


def machine(eng, st, res, pack) -> Dict[str, object]:
    smi = widecheck.smi_fields(st.uuid, widecheck.KEY_FIELDS + widecheck.LIVE_FIELDS) if st is not None else None
    smi = smi or {}
    dll = Path(eng.lib.path)
    return dict(
        package="yasakani %s%s" % (__version__, " (%s)" % BUILD_NOTE if BUILD_NOTE else ""),
        os="%s %s" % (platform.system(), platform.version() if platform.system() == "Windows" else platform.release()),
        python=platform.python_version(),
        library=dict(file=dll.name, sha256=widecheck.file_sha256(dll)[:16], build_info=eng.lib.build_info() or "(none: before 0.1.7)"),
        gpu=dict(name=smi.get("name"), cc=st.cc_text if st else None, total_mib=st.total_mib if st else None, sms=st.sms if st else None,
                 code=st.code_text if st else None, driver=smi.get("driver_version"), driver_model=smi.get("driver_model.current"),
                 power_limit_w=smi.get("power.limit"), max_sm_mhz=smi.get("clocks.max.sm"), max_mem_mhz=smi.get("clocks.max.memory"),
                 display_attached=smi.get("display_attached"), display_active=smi.get("display_active"), pstate=smi.get("pstate"),
                 temperature_c=smi.get("temperature.gpu"), sm_mhz=smi.get("clocks.sm"),
                 slowdown_reasons=smi.get("clocks_event_reasons.active"), free_mib_now=st.free_mib if st else None),
        model=dict(id=pack.model_id, weights=pack.weights.name, context=eng.max_ctx),
        wide=_wide_part(eng, res),
    )


def _wide_part(eng, res) -> Dict[str, object]:
    if res is None:
        return dict(setting=eng.wide_mode, status="-", mask=0, widths="-", reason="", cache="-", check_seconds=0.0, free_after_mib=None)
    cache = "kept result used" if res.cache_hit else "measured now" + ((", " + res.cache_note) if res.cache_note else "")
    return dict(setting=eng.wide_mode, status=res.status, mask=res.mask, widths=widecheck.widths_text(res.mask), reason=res.reason,
                cache=cache, check_seconds=round(res.check_seconds, 1), free_after_mib=res.free_after_mib)


def run_report(args) -> int:
    from . import editbench
    from .cli import _open_stack
    tasks = [] if args.no_bench else editbench.load_tasks()
    t0 = time.time()
    _say("yasakani report: the model loads, the wide copy passes are checked on this GPU, then%s the self-check (about 1.5 minutes "
         "on the RTX 5090; longer on other cards) ..." % ("" if args.no_bench else " the edit tasks run with the wide passes off and on,"))
    pack, eng, fmt, runner = _open_stack(args, warm=True)
    rep: Dict[str, object] = {}
    try:
        st = eng.wide_state()
        res = eng.wide_result
        rep["machine"] = machine(eng, st, res, pack)
        rep["check"] = res.summary() if res is not None else None
        if st is None:
            rep["note"] = "this engine library is older than 0.1.7: it cannot switch the wide copy passes, so there is nothing to compare"
        elif not st.allowed:
            rep["note"] = "wide copy passes are not allowed by this engine (YSKN_COPY_WIDE_OFF, a built-in draft head, or --wide off)"
        if st is not None and st.allowed and tasks:
            rep["bench"] = bench_columns(eng, runner, tasks, res, st, args.rounds, progress=_say)
        if not args.no_selfcheck:
            _say("self-check ...")
            rep["selfcheck"] = selfcheck_part(eng, fmt, pack, st)
        rep["seconds"] = round(time.time() - t0)
    finally:
        eng.close()
    text = json.dumps(rep, indent=1, default=str) if args.json else render(rep)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        _say("wrote %s" % Path(args.out).name)
    return 0


# ---- Markdown -----------------------------------------------------------------------------------------------------------------
def _v(x, unit: str = "") -> str:
    return "?" if x in (None, "") else "%s%s" % (x, unit)


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def render(rep: Dict[str, object]) -> str:
    m = rep["machine"]
    g, lib, mo, w = m["gpu"], m["library"], m["model"], m["wide"]
    L = ["### YASAKANI report (%s)" % m["package"], "", "| | |", "|---|---|",
         "| GPU | %s, compute capability %s, %s MiB, %s SMs |" % (_v(g["name"]), _v(g["cc"]), _v(g["total_mib"]), _v(g["sms"])),
         "| Driver | %s (%s) · %s · Python %s |" % (_v(g["driver"]), _v(g["driver_model"]), m["os"], m["python"]),
         "| GPU state | power limit %s W, max clocks %s / %s MHz, display attached %s (active %s), %s, %s °C, SM clock %s MHz, "
         "slowdown reasons %s |" % (_v(g["power_limit_w"]), _v(g["max_sm_mhz"]), _v(g["max_mem_mhz"]), _v(g["display_attached"]),
                                    _v(g["display_active"]), _v(g["pstate"]), _v(g["temperature_c"]), _v(g["sm_mhz"]),
                                    _v(g["slowdown_reasons"])),
         "| Engine | %s %s · %s · code %s |" % (lib["file"], lib["sha256"], lib["build_info"], _v(g["code"])),
         "| Model | %s (%s) · context %s tokens |" % (mo["id"], mo["weights"], mo["context"]),
         "| Wide passes | setting %s · result %s: %s%s · %s · free GPU memory after their graphs %s MiB |" % (
             w["setting"], w["status"], w["widths"], (" (" + w["reason"] + ")") if w["reason"] else "", w["cache"], _v(w["free_after_mib"])),
         ""]
    if rep.get("note"):
        L += ["**Note**: " + rep["note"], ""]
    c = rep.get("check")
    if c:
        gate = c.get("gate")
        if gate:
            ok = not gate["argmax_diff"] and not gate["nll_diff"] and not gate["error"] and gate["positions"]
            L += ["**Correctness** (passes of 17, 31, 32, 33, 47, 48, 63 and 64 rows against passes of 16 over the same positions): "
                  + ("same most likely ids and NLL bit for bit at %d positions" % gate["positions"] if ok else
                     "DIFFERENT: most likely id at %d, NLL at %d of %d positions %s%s" % (
                         gate["argmax_diff"], gate["nll_diff"], gate["positions"],
                         json.dumps(gate.get("per_rows") or {}), (" " + gate["error"]) if gate["error"] else "")), ""]
        t = c.get("timing")
        if c.get("widths"):
            L += ["**Copy cycles** at 4,096 tokens of context (pass + commit + drafter; median of batches, %d rounds%s; "
                  "limit = 0.8 × rows / 16; greedy, and one round sampled)" % (len((t or {}).get("rounds") or []),
                                                                               ", measured twice: rounds disagreed" if t and t.get("redone") else ""),
                  "", "| rows | ms | ratio to 16 (rounds) | sampled | limit | used |", "|---:|---:|---|---:|---:|---|"]
            t16 = None
            if t and t.get("rounds"):
                v = [x[0] for r in t["rounds"] for x in r.values()]
                t16 = sorted(v)[len(v) // 2] if v else None
            L.append("| 16 | %s | 1.00 | – | – | – |" % ("%.2f" % t16 if t16 else "?"))
            for x in c["widths"]:
                L.append("| %d | %s | %s | %s | %.2f | %s |" % (
                    x["rows"], "%.2f" % x["ms"] if x["ms"] else "?", " / ".join("%.2f" % r for r in x["ratios"]) or "?",
                    "%.2f" % x["sampled"] if x["sampled"] else "–", x["limit"],
                    ("yes" if x["used"] else "no") + (" (unsure)" if x["unsure"] else "")))
            L.append("")
    b = rep.get("bench")
    if b:
        L += _bench_lines(b)
    s = rep.get("selfcheck")
    if s:
        L += _selfcheck_lines(s)
    L += ["", "_%d s in all._" % rep.get("seconds", 0)]
    return "\n".join(L)


def _bench_lines(b: Dict[str, object]) -> List[str]:
    rows = [r for r in b["rows"] if "skipped" not in r]
    masks, same_as = b["masks"], b["same_as"]
    cols = [c for c in COLUMNS if c not in same_as]
    head = ["off"] + ["%s (%s)" % (c, widecheck.widths_text(masks[c])) for c in cols if c != "off"]
    L = ["**Edit speed** (`bench --edit` tasks; greedy, thinking off; tok/s, the mean of the runs; order off, checked, all, all, "
         "checked, off%s; wide passes and back-offs: the sums of the runs of each column)" % (
             "".join("; %s = %s" % (k, v) for k, v in same_as.items())), "",
         "| task | " + " | ".join(head) + " | wide passes (%s) | back-offs (%s) | same output |" % ("/".join(cols), "/".join(cols)),
         "|---|" + "---:|" * len(head) + "---:|---:|---|"]
    titles = []
    for r in rows:
        if r["title"] not in titles:
            titles.append(r["title"])
    for title in titles:
        rs = [r for r in rows if r["title"] == title]
        vals = []
        for c in cols:
            v = _mean([r["tokens_per_s"] for r in rs if r["column"] == c])
            vals.append("%.1f" % v if v is not None else "?")
        wp = "/".join(str(sum(r["wide_passes"] for r in rs if r["column"] == c)) for c in cols)
        bo = "/".join(str(sum(r.get("wide_backoffs", 0) for r in rs if r["column"] == c)) for c in cols)
        shas = {r.get("ids_sha256") for r in rs}
        exp = all(r["same"] for r in rs)
        L.append("| %s | %s | %s | %s | %s |" % (title, " | ".join(vals), wp, bo,
                                                 ("yes" if len(shas) == 1 else "NO (%d different)" % len(shas)) +
                                                 ("" if exp else ", not the expected answer")))
    for r in b["rows"]:
        if "skipped" in r:
            L.append("| %s | skipped in %s: needs a context of %d tokens | | | |" % (r["title"], r["column"], r["skipped"]))
    L.append("")
    return L


def _selfcheck_lines(s: Dict[str, object]) -> List[str]:
    ref = ", ".join("%s %s%s" % (c["name"], c["result"], (" (from token #%d)" % c["first_diff"]) if c.get("first_diff") else "")
                    for c in s.get("reference", []))
    L = ["**Self-check**: reference: %s" % ref]
    cl = s.get("close")
    if cl:
        L.append("  - predictions on the reference output: %s" % ("within the tolerances" if not cl["failures"] else
                                                                  "FAILED: " + "; ".join(cl["failures"])))
    co = s.get("consistency")
    if co:
        if co["wide_checked"]:
            same = not co["argmax_diff"] and not co["nll_diff"]
            L.append("  - self-consistency with every width open: %d wide passes (widest %d rows) in the generation; passes of 64 vs 16: %s"
                     % (co["wide_passes"], co["max_rows"], "same predictions bit for bit at %d positions" % co["positions"] if same else
                        "most likely id differs at %d, NLL at %d of %d positions" % (co["argmax_diff"], co["nll_diff"], co["positions"])))
        if co["failures"]:
            L.append("  - self-consistency FAILED: " + "; ".join(co["failures"]))
        elif not co["wide_checked"]:
            L.append("  - self-consistency: " + " / ".join(co["lines"][1:]))
    return L
