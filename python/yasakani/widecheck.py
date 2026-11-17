# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Wide copy passes per GPU (0.1.7, a test build).

Since 0.1.4 a pass that copies the input can check up to 64 tokens at once instead of 16 ("wide copy passes", docs/TECHNOLOGY.md
5.4).  They were measured on the RTX 5090 only, so up to 0.1.6 the engine used them on compute capability 12.x (RTX 50 series) only.
0.1.7 lets the owners of other cards try them.  A check on the card decides whether they are used there; it lowers the chance of a
slower or different result, but it cannot rule it out:

``--wide`` (or ``YASAKANI_WIDE``)
  * ``auto`` (default): RTX 30 / 40 (compute capability 8.6 / 8.9) as in 0.1.6, without wide passes.  RTX 50: wide passes as in
    0.1.6, except a width that the check below finds slower on this GPU.
  * ``on``: also on RTX 30 / 40, but only the widths that the check finds correct and faster; none when the check cannot be made.
  * ``off``: no wide copy passes.  ``force``: every width without the check (to compare).

The check (once per GPU, driver, engine library, package and model; kept in ``wide_check.json``, see :func:`cache_dir`):
  1. Correctness: the engine reads a text of about 2,400 tokens in passes of 17, 31, 32, 33, 47, 48, 63 and 64 rows and again in
     passes of 16; the most likely ids and the NLL must be the same bit for bit.  The engine computes both with the same integer
     sums, so any difference is an error on this GPU: then no wide pass is used, passes of 17..64 rows no longer use the exact-sum
     GEMV (set_opt 31 = 0, on RTX 50 too), and ``yasakani selfcheck`` fails.
  2. Time: at 4,096 tokens of context it runs copy cycles as the generation does (the pass, the commit of all its rows, the drafter
     reading them), 16 rows and W rows in turn, about 0.6 s each, 3 rounds, greedy, and one round with sampling.  A width W is used
     when one cycle of W rows takes at most 0.8 x W / 16 times one of 16 rows (32: 1.60, 48: 2.40, 64: 3.20): with every row
     accepted it then gives at least 1.25 times the tokens per second of 16 rows.  RTX 50 keeps a width unless every round is over
     the limit; RTX 30 / 40 opens a width only when every round is under it.  The measured times also become the copy loop's
     starting values (they were the RTX 5090's for every GPU before).
  3. The GPU must be idle: nvidia-smi shows its load before and after, and (where it can tell: not under Windows) whether another
     program holds GPU memory.  When it is busy nothing is measured or kept.

The check reads its text in pieces of at most 1,023 tokens (Engine.read_prompt): the engine sets up a large work area for a single
prompt read of 1,024 tokens or more and keeps it, which ordinary generation does not need.

During generation the engine still checks every 4 wide passes whether they are faster than passes of 16 rows and goes back to 16
when they are not (0.1.7: also inside a run of fully accepted passes; a run that ends with fewer than 4 wide passes since the last
check is judged together with the window before it).  If it goes back in at least RECHECK_RATE of every 100 wide passes, counted
over at least RECHECK_MIN_PASSES of them in one process, the kept result is marked and the next start checks again.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import gpu as gpus
from ._native import YasakaniError

MODES = ("auto", "on", "off", "force")
MODE_ENV = "YASAKANI_WIDE"
CACHE_ENV = "YASAKANI_CACHE_DIR"
CACHE_FILE = "wide_check.json"
CACHE_FORMAT = "yasakani-wide-check-1"
CHECK_VERSION = 2                 # of the measurement and the rule below (part of the key of a kept result; 2: the text read in
                                  # pieces, display_attached in the key)
CACHE_KEEP = 16                   # results kept in the file (newest first)
CACHE_DAYS = 30                   # a kept result older than this is measured again
WIDTHS = (32, 48, 64)
BIT = {32: 1, 48: 2, 64: 4}
MARGIN = 0.8                      # W rows may take at most MARGIN x W / 16 times 16 rows
CONTEXT = 4096                    # tokens of context for the time check
GATE_START = 2048                 # tokens read before the correctness check
GATE_ROWS = (17, 31, 32, 33, 47, 48, 63, 64)
ROUNDS = 3
SLICE_S = 0.6                     # seconds of cycles of one width in one round
WARM_S = 0.5
BATCH = 2                         # cycles per batch; a time is the median of the batch means
SPREAD_MAX = 1.10                 # rounds whose ratios differ more than this are measured once more, then "unsure"
BUSY_UTIL = 10                    # % GPU load (nvidia-smi) that counts as busy
MIN_FREE_MIB = 128                # RTX 30 / 40: free GPU memory needed after the graphs of the wide passes are built
RECHECK_RATE = 10                 # back-offs per 100 wide passes in real generations at which the next start checks again ...
RECHECK_MIN_PASSES = 128          # ... counted over at least this many wide passes in one process
TEXT_TASK = "edit_code"           # the text the checks read: this edit task's prompt and expected answer (edit_tasks.json)

# nvidia-smi fields of the key of a kept result and of the report.  Asked one by one when a driver does not know one of them.
KEY_FIELDS = ("name", "driver_version", "memory.total", "driver_model.current", "power.limit", "clocks.max.sm", "clocks.max.memory",
              "display_attached")             # not display_mode: deprecated (driver 616.92 answers "[Requested functionality ...]")
LIVE_FIELDS = ("utilization.gpu", "memory.used", "pstate", "temperature.gpu", "clocks.sm", "clocks_event_reasons.active",
               "display_active")


# ---- the setting --------------------------------------------------------------------------------------------------------------
def mode_of(value: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> str:
    """auto | on | off | force: ``value`` (the command line), else ``YASAKANI_WIDE``, else auto."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(MODE_ENV, "")
    v = (v or "auto").strip().lower()
    if v not in MODES:
        raise YasakaniError("%s / --wide must be one of %s, not %r" % (MODE_ENV, ", ".join(MODES), v))
    return v


def engine_env(mode: str) -> Dict[str, str]:
    """The engine setting for a mode: auto = none (the engine decides by the code the GPU runs)."""
    return {"auto": {}, "on": {"YSKN_WIDE": "1"}, "off": {"YSKN_WIDE": "0"}, "force": {"YSKN_WIDE": "2"}}[mode]


def widths_of(mask: int) -> List[int]:
    return [w for w in WIDTHS if mask & BIT[w]]


def mask_of(widths: Sequence[int]) -> int:
    m = 0
    for w in widths:
        m |= BIT[int(w)]
    return m


def limit(w: int, margin: float = MARGIN) -> float:
    return margin * w / 16.0


def widths_text(mask: int) -> str:
    w = widths_of(mask)
    return ", ".join(str(x) for x in w) + " rows" if w else "none"


# ---- the engine's state -------------------------------------------------------------------------------------------------------
@dataclass
class WideState:
    """yskn_wide_state (see include/yskn_abi.h)."""
    code_arch: int = 0
    cc: int = 0
    sms: int = 0
    allowed: bool = False
    xw: bool = False
    xw_rows: int = 16
    mask: int = 0
    pref: int = -1
    native: bool = False
    t_init_us: Tuple[int, int, int, int] = (0, 0, 0, 0)
    free_mib: int = 0
    total_mib: int = 0
    pci: Tuple[int, int, int] = (0, 0, 0)
    uuid: str = ""
    test_arch: bool = False
    sass_arch: int = 0
    builtin_draft: bool = False
    drafter: bool = True

    @classmethod
    def from_array(cls, a) -> "WideState":
        v = [int(x) for x in a] + [0] * 32
        u = b"".join(int(w & 0xFFFFFFFF).to_bytes(4, "big") for w in v[19:23]).hex()
        return cls(code_arch=v[1], cc=v[2], sms=v[3], allowed=bool(v[4]), xw=bool(v[5]), xw_rows=v[6], mask=v[7], pref=v[8],
                   native=bool(v[9]), t_init_us=tuple(v[10:14]), free_mib=v[14], total_mib=v[15], pci=tuple(v[16:19]),
                   uuid="GPU-%s-%s-%s-%s-%s" % (u[:8], u[8:12], u[12:16], u[16:20], u[20:32]) if any(v[19:23]) else "",
                   test_arch=bool(v[23]), sass_arch=v[24], builtin_draft=bool(v[25]), drafter=bool(v[26]) if len(a) > 26 else True)

    @property
    def cc_text(self) -> str:
        return "%d.%d" % (self.cc // 10, self.cc % 10) if self.cc else "?"

    @property
    def code_text(self) -> str:
        s = "sm_%d" % self.code_arch
        if self.test_arch:                                          # a test build replaced the engine's value: no word about JIT
            return s + (" (set by YSKN_TEST_CODE_ARCH; machine code sm_%d)" % self.sass_arch if self.sass_arch else
                        " (set by YSKN_TEST_CODE_ARCH)")
        if self.sass_arch and self.sass_arch != self.code_arch:
            s += " (compiled at load time for sm_%d)" % self.sass_arch
        return s

    @property
    def checkable_widths(self) -> List[int]:
        """32 rows only with the exact-sum wide GEMV."""
        return [w for w in WIDTHS if w != 32 or self.xw]


# ---- nvidia-smi ---------------------------------------------------------------------------------------------------------------
def _smi(args: Sequence[str], run: Callable, environ=None, timeout: float = 15) -> Optional[str]:
    exe = gpus.smi_path(environ)
    if not exe:
        return None
    try:
        r = run([exe] + list(args), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return r.stdout if r.returncode == 0 else None


def smi_fields(uuid: str, fields: Sequence[str], run: Callable = subprocess.run, environ=None) -> Optional[Dict[str, Optional[str]]]:
    """The values of ``fields`` for the GPU with this UUID (asked by its UUID, so the numbering of CUDA and nvidia-smi cannot be
    mixed up, issue #20), None when nvidia-smi does not answer for it.  A field the driver does not know is None."""
    if not uuid:
        return None

    def ask(fs):
        out = _smi(["--id=" + uuid, "--query-gpu=" + ",".join(fs), "--format=csv,noheader,nounits"], run, environ)
        if out is None:
            return None
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if len(lines) != 1:
            return None
        parts = [x.strip() for x in lines[0].split(",")]
        return dict(zip(fs, (gpus._value(x) for x in parts))) if len(parts) == len(fs) else None

    got = ask(fields)
    if got is not None:
        return got
    one = {f: (ask([f]) or {}).get(f) for f in fields}
    return one if any(v is not None for v in one.values()) else None


OTHER_APP_MIB = 256               # a program that holds this much GPU memory counts as computing on the GPU


def other_compute(uuid: str, run: Callable = subprocess.run, environ=None, pid: Optional[int] = None) -> Optional[List[str]]:
    """Programs other than this one that hold at least OTHER_APP_MIB of the GPU's memory (pids as text), None when nvidia-smi cannot
    tell.  Under Windows (WDDM) nvidia-smi lists every program that draws on the desktop with the GPU (explorer.exe, browsers) and
    gives no memory per program: then nothing can be told from the list, and the GPU load alone decides."""
    out = _smi(["--id=" + uuid, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], run, environ)
    if out is None:
        return None
    me = str(os.getpid() if pid is None else pid)
    found = []
    for ln in out.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) != 2 or parts[0] == me or not parts[0]:
            continue
        mib = gpus._int(gpus._value(parts[1]))
        if mib is not None and mib >= OTHER_APP_MIB:
            found.append(parts[0])
    return found


def busy_reason(uuid: str, run: Callable = subprocess.run, environ=None, sleep: Callable = time.sleep, reads: int = 2,
                wait_s: float = 1.1) -> Optional[str]:
    """Why the GPU is not idle (None = idle).  The load is read ``reads`` times, ``wait_s`` apart, the first one ``wait_s`` after this
    program's own last work (the load nvidia-smi shows is an average over up to one second)."""
    for i in range(reads):
        sleep(wait_s)
        v = smi_fields(uuid, ("utilization.gpu",), run, environ)
        u = gpus._int(v.get("utilization.gpu")) if v else None
        if u is None:
            return "nvidia-smi gives no GPU load"
        if u >= BUSY_UTIL:
            return "the GPU load is %d %% while this program does not use it" % u
    apps = other_compute(uuid, run, environ)
    if apps:
        return "%d other program(s) hold GPU memory on this GPU" % len(apps)
    return None


# ---- the text the checks read -------------------------------------------------------------------------------------------------
def check_ids(fmt, need: int = CONTEXT + 64) -> List[int]:
    """Token ids of a real text: the prompt of the code edit task (thinking off) and its expected answer, repeated to ``need``."""
    from . import editbench
    task = next((t for t in editbench.load_tasks() if t.name == TEXT_TASK), None)
    if task is None:
        raise YasakaniError("the edit task %s is missing" % TEXT_TASK)
    prompt, _ = fmt.prompt(task.messages, enable_thinking=False)
    ids = [int(x) for x in prompt] + [int(x) for x in fmt.encode(task.expected)]
    while len(ids) < need:
        ids = ids + ids
    return ids[:need]


# ---- measuring ----------------------------------------------------------------------------------------------------------------
def robust_ms(times: Sequence[float], batch: int = BATCH) -> float:
    """The median of the means of consecutive batches of cycles (one stall moves one batch, not the result)."""
    t = [float(x) for x in times]
    if not t:
        raise ValueError("no cycles were timed")
    if len(t) < 2 * batch:
        return statistics.median(t)
    means = [sum(t[i:i + batch]) / batch for i in range(0, len(t) - batch + 1, batch)]
    return statistics.median(means)


def _slice(engine, ids: Sequence[int], rows: int, seconds: float, chunk: int, clock: Callable = time.perf_counter) -> List[float]:
    out: List[float] = []
    t0 = clock()
    while True:
        out.extend(float(x) for x in engine.time_cycles(ids[:rows], chunk))
        if clock() - t0 >= seconds and len(out) >= 2 * BATCH:
            return out


@dataclass
class Timing:
    rounds: List[Dict[int, Tuple[float, float]]] = field(default_factory=list)    # per round: W -> (16-row ms, W-row ms)
    sampled: Dict[int, Tuple[float, float]] = field(default_factory=dict)         # one round with sampling
    redone: bool = False

    def ratios(self, w: int) -> List[float]:
        return [r[w][1] / r[w][0] for r in self.rounds if w in r and r[w][0] > 0]

    def sampled_ratio(self, w: int) -> Optional[float]:
        a = self.sampled.get(w)
        return a[1] / a[0] if a and a[0] > 0 else None

    def ms(self, rows: int) -> Optional[float]:
        """The median time of a copy cycle of ``rows`` rows (greedy) over the rounds."""
        if rows == 16:
            v = [x[0] for r in self.rounds for x in r.values()]
        else:
            v = [r[rows][1] for r in self.rounds if rows in r]
        return statistics.median(v) if v else None


def measure(engine, ids: Sequence[int], widths: Sequence[int] = WIDTHS, *, rounds: Optional[int] = None, slice_s: Optional[float] = None,
            warm_s: Optional[float] = None, sampled: bool = True, clock: Callable = time.perf_counter) -> Timing:
    """Time copy cycles of 16 and of W rows in turn (see the top of this module; None = ROUNDS, SLICE_S, WARM_S).  The engine's state
    is reset before and after; the widths must be open (``engine.set_wide``)."""
    from .engine import Sampling
    widths = list(widths)
    rounds = ROUNDS if rounds is None else int(rounds)
    slice_s = SLICE_S if slice_s is None else float(slice_s)
    warm_s = WARM_S if warm_s is None else float(warm_s)
    engine.read_prompt(ids[:CONTEXT])
    win = list(ids[CONTEXT:CONTEXT + 64])
    if len(win) < max(widths + [16]):
        raise ValueError("the check needs %d ids" % (CONTEXT + 64))
    chunk: Dict[int, int] = {}

    def warm():
        for rows in [16] + widths:                                  # graphs are captured on the first cycle of each shape
            t = engine.time_cycles(win[:rows], 2)
            chunk[rows] = max(2, min(32, int(0.1 / max(1e-4, float(t[-1]) / 1000.0))))
        for rows in [16] + widths:
            _slice(engine, win, rows, warm_s / (1 + len(widths)), chunk[rows], clock)

    def one_round(r: int) -> Dict[int, Tuple[float, float]]:
        rec = {}
        k = r % len(widths)
        for w in widths[k:] + widths[:k]:
            if r % 2 == 0:
                a = robust_ms(_slice(engine, win, 16, slice_s, chunk[16], clock))
                b = robust_ms(_slice(engine, win, w, slice_s, chunk[w], clock))
            else:
                b = robust_ms(_slice(engine, win, w, slice_s, chunk[w], clock))
                a = robust_ms(_slice(engine, win, 16, slice_s, chunk[16], clock))
            rec[w] = (a, b)
        return rec

    keep = engine.sampling
    res = Timing()
    try:
        engine.set_sampling(Sampling(temperature=0.0))
        warm()
        res.rounds = [one_round(r) for r in range(rounds)]
        if any(_spread(res.ratios(w)) > SPREAD_MAX for w in widths):
            res.redone = True
            res.rounds = [one_round(r) for r in range(rounds)]
        if sampled:
            engine.set_sampling(Sampling(1.0, 0.95, 20, 0.0, 1234))
            for rows in [16] + widths:
                engine.time_cycles(win[:rows], 2)
            res.sampled = one_round(0)
    finally:
        engine.reset()
        if keep is not None:
            engine.set_sampling(keep)
    return res


def _spread(r: Sequence[float]) -> float:
    return max(r) / min(r) if r and min(r) > 0 else 1.0


@dataclass
class Gate:
    """The correctness check: passes of W rows against passes of 16 rows over the same positions."""
    positions: int = 0
    argmax_diff: int = 0
    nll_diff: int = 0
    per_rows: Dict[int, Tuple[int, int]] = field(default_factory=dict)    # rows -> (argmax diffs, NLL diffs)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.positions > 0 and not self.argmax_diff and not self.nll_diff


def correctness(engine, ids: Sequence[int], rows: Sequence[int] = GATE_ROWS, start: int = GATE_START) -> Gate:
    """Read ``ids[:start]``, then for each M of ``rows``: one pass of M rows (not accepted) and the same M positions in passes of at
    most 16 rows (accepted).  The most likely ids and the NLL (as 32-bit floats) must be equal.  The state is reset afterwards."""
    import numpy as np
    from .engine import forced_passes
    g = Gate()
    need = start + sum(rows) + 1
    if len(ids) < need:
        raise ValueError("the correctness check needs %d ids" % need)
    try:
        engine.read_prompt(ids[:start])
        p = start
        for m in rows:
            a_am, a_nll = engine.verify_rows(ids[p:p + m], ids[p + 1:p + m + 1], commit=False)
            b_am, b_nll = [], []
            q = p
            for k in forced_passes(m, 16):
                am, nll = engine.verify_rows(ids[q:q + k], ids[q + 1:q + k + 1], commit=True)
                b_am.append(am)
                b_nll.append(nll)
                q += k
            b_am, b_nll = np.concatenate(b_am), np.concatenate(b_nll)
            da = int((a_am != b_am).sum())
            dn = int((a_nll.view(np.uint32) != b_nll.view(np.uint32)).sum())
            g.per_rows[m] = (da, dn)
            g.argmax_diff += da
            g.nll_diff += dn
            g.positions += m
            p += m
    except YasakaniError as ex:
        g.error = str(ex)
    finally:
        try:
            engine.reset()
        except YasakaniError:
            pass
    return g


# ---- the rule ----------------------------------------------------------------------------------------------------------------
@dataclass
class WidthResult:
    rows: int
    ratios: List[float]
    sampled: Optional[float]
    limit: float
    ms: Optional[float]
    used: bool
    unsure: bool

    @property
    def ratio(self) -> Optional[float]:
        return statistics.median(self.ratios) if self.ratios else None


def decide(timing: Timing, native: bool, widths: Sequence[int] = WIDTHS, margin: float = MARGIN) -> Tuple[int, List[WidthResult]]:
    """(mask, one WidthResult per width).  RTX 50 (``native``): a width is left out only when every greedy round and the sampled
    round are over the limit and the rounds agree.  RTX 30 / 40: a width is used only when every round is under the limit and the
    rounds agree (within SPREAD_MAX)."""
    out, mask = [], 0
    for w in widths:
        rs = timing.ratios(w)
        s = timing.sampled_ratio(w)
        lim = limit(w, margin)
        unsure = not rs or _spread(rs) > SPREAD_MAX
        every = rs + ([s] if s is not None else [])
        under = bool(rs) and all(x <= lim for x in every)
        over = bool(rs) and all(x > lim for x in every)
        used = (not (over and not unsure)) if native else (under and not unsure)
        if used:
            mask |= BIT[w]
        out.append(WidthResult(rows=w, ratios=rs, sampled=s, limit=lim, ms=timing.ms(w), used=used, unsure=unsure))
    return mask, out


def times_us(timing: Timing) -> List[int]:
    """The starting times of the copy loop (16, 32, 48, 64 rows; 0 = not measured: the built-in value)."""
    return [int(round(1000 * timing.ms(r))) if timing.ms(r) else 0 for r in (16,) + WIDTHS]


# ---- kept results -------------------------------------------------------------------------------------------------------------
def cache_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """``YASAKANI_CACHE_DIR``, else ``%LOCALAPPDATA%\\yasakani`` on Windows, ``$XDG_CACHE_HOME/yasakani`` or ``~/.cache/yasakani``."""
    env = os.environ if environ is None else environ
    if env.get(CACHE_ENV):
        return Path(env[CACHE_ENV])
    if sys.platform == "win32" and env.get("LOCALAPPDATA"):
        return Path(env["LOCALAPPDATA"]) / "yasakani"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "yasakani"
    return Path.home() / ".cache" / "yasakani"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def key_fields(engine, st: WideState, smi: Mapping[str, Optional[str]], version: str) -> Dict[str, object]:
    """What a kept result depends on.  The GPU's UUID only as a short hash; no paths."""
    pack = engine.pack
    wname = pack.weights.name
    wsha = (pack.sha256 or {}).get(wname) or ("size:%d" % pack.weights.stat().st_size)
    env = {k: v for k, v in sorted(engine.profile.env.items()) if k not in ("YSKN_BDR_SHEAD", "YSKN_TB_PROFILE")}
    return dict(check=CHECK_VERSION, margin=MARGIN, context=CONTEXT, package=version, library=file_sha256(engine.lib.path),
                build_info=engine.lib.build_info(), model=pack.model_id, weights=wname, weights_sha256=str(wsha)[:16],
                gpu=smi.get("name"), cc=st.cc, code=st.code_arch, sms=st.sms, total_mib=st.total_mib,
                uuid_sha=_sha(st.uuid.lower().encode())[:16], driver=smi.get("driver_version"), driver_model=smi.get("driver_model.current"),
                power_limit=smi.get("power.limit"), max_sm_mhz=smi.get("clocks.max.sm"), max_mem_mhz=smi.get("clocks.max.memory"),
                display=smi.get("display_attached"), env_sha=_sha(json.dumps(env, sort_keys=True).encode())[:16])


def key_of(fields: Mapping[str, object]) -> str:
    return _sha(json.dumps(fields, sort_keys=True).encode())


class Cache:
    """``wide_check.json``: {"format", "entries": [{"key", "fields", "time", "result", "recheck"}]}, newest first.  A file that cannot
    be read counts as empty; one that cannot be written is left alone (the check is then made at every start)."""

    def __init__(self, folder: Optional[Path] = None, now: Callable[[], float] = time.time):
        self.dir = Path(folder) if folder is not None else cache_dir()
        self.path = self.dir / CACHE_FILE
        self.now = now
        self.warning = ""

    def load(self) -> List[dict]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(d, dict) or d.get("format") != CACHE_FORMAT or not isinstance(d.get("entries"), list):
            return []
        return [e for e in d["entries"] if isinstance(e, dict) and isinstance(e.get("key"), str) and isinstance(e.get("result"), dict)]

    def get(self, key: str) -> Optional[dict]:
        for e in self.load():
            if e["key"] == key:
                age = self.now() - float(e.get("time", 0))
                if e.get("recheck") or age < 0 or age > CACHE_DAYS * 86400:
                    return None
                return e
        return None

    def _write(self, entries: List[dict]) -> bool:
        """Write to a temporary file and replace: two programs writing at once leave one whole file, never a mixed one."""
        tmp = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".wide_check.", suffix=".tmp", dir=str(self.dir))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(dict(format=CACHE_FORMAT, entries=entries[:CACHE_KEEP]), f, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
            return True
        except OSError as ex:
            self.warning = "the result of the check could not be kept (%s)" % type(ex).__name__
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return False

    def put(self, key: str, fields: Mapping[str, object], result: Mapping[str, object]) -> bool:
        entries = [e for e in self.load() if e["key"] != key]
        entries.insert(0, dict(key=key, fields=dict(fields), time=self.now(), result=dict(result), recheck=False))
        return self._write(entries)

    def mark_recheck(self, key: str) -> bool:
        entries = self.load()
        hit = False
        for e in entries:
            if e["key"] == key and not e.get("recheck"):
                e["recheck"] = True
                hit = True
        return self._write(entries) if hit else False


# ---- deciding at start --------------------------------------------------------------------------------------------------------
@dataclass
class WideResult:
    """What the start decided; ``status``: off | force | measured | cached | busy | unknown-gpu | mismatch | error | no-check |
    old-library | memory."""
    mode: str = "auto"
    status: str = "off"
    mask: int = 0
    native: bool = False
    code: str = ""
    cc: str = ""
    widths: List[WidthResult] = field(default_factory=list)
    gate: Optional[Gate] = None
    timing: Optional[Timing] = None
    times_us: List[int] = field(default_factory=list)
    reason: str = ""
    line: str = ""
    cache_hit: bool = False
    cache_note: str = ""
    key: str = ""
    smi: Dict[str, Optional[str]] = field(default_factory=dict)
    free_before_mib: Optional[int] = None
    free_after_mib: Optional[int] = None
    check_seconds: float = 0.0

    def summary(self) -> Dict[str, object]:
        """The part that is kept and reported (no UUID, no paths)."""
        return dict(status=self.status, mask=self.mask, reason=self.reason, times_us=list(self.times_us),
                    widths=[asdict(w) for w in self.widths],
                    gate=None if self.gate is None else dict(positions=self.gate.positions, argmax_diff=self.gate.argmax_diff,
                                                              nll_diff=self.gate.nll_diff, error=self.gate.error,
                                                              per_rows={str(k): list(v) for k, v in self.gate.per_rows.items()}),
                    timing=None if self.timing is None else dict(rounds=[{str(k): list(v) for k, v in r.items()} for r in self.timing.rounds],
                                                                  sampled={str(k): list(v) for k, v in self.timing.sampled.items()},
                                                                  redone=self.timing.redone),
                    check_seconds=round(self.check_seconds, 1))


def _restore(summary: Mapping[str, object]) -> Tuple[List[WidthResult], Optional[Gate], Optional[Timing]]:
    ws = [WidthResult(**w) for w in summary.get("widths") or []]
    g = summary.get("gate")
    gate = None if not g else Gate(positions=g["positions"], argmax_diff=g["argmax_diff"], nll_diff=g["nll_diff"], error=g.get("error", ""),
                                   per_rows={int(k): tuple(v) for k, v in (g.get("per_rows") or {}).items()})
    t = summary.get("timing")
    timing = None if not t else Timing(rounds=[{int(k): tuple(v) for k, v in r.items()} for r in t["rounds"]],
                                       sampled={int(k): tuple(v) for k, v in (t.get("sampled") or {}).items()}, redone=bool(t.get("redone")))
    return ws, gate, timing


def _closed_mask(st: WideState) -> int:
    """The widths when nothing could be checked: RTX 50 as 0.1.6 (all), RTX 30 / 40 none."""
    return 7 if st.native else 0


def describe(res: WideResult) -> str:
    """The one line printed at start."""
    head = "wide copy passes (code %s, compute capability %s): " % (res.code, res.cc)
    if res.status == "off":
        return head + "off" + ("" if res.mode == "off" else " (as in 0.1.6 on this GPU; --wide on checks whether they pay off here)")
    if res.status == "force":
        return head + "%s, not checked (--wide force)" % widths_text(res.mask)
    if res.status == "old-library":
        return head + res.reason
    if res.status in ("measured", "cached"):
        parts = []
        for w in res.widths:
            r = w.ratio
            if r is None:
                continue
            parts.append("%d %.2f%s%.2f%s" % (w.rows, r, "<=" if r <= w.limit else ">", w.limit, ", unsure" if w.unsure else ""))
        tail = "" if not parts else " (time of W rows / 16 rows vs limit: %s)" % "; ".join(parts)
        when = " - kept result of an earlier check" if res.cache_hit else ""
        return head + widths_text(res.mask) + tail + when
    return head + "%s (%s)" % (widths_text(res.mask), res.reason)


def setup(engine, ids: Optional[Sequence[int]], *, check: bool = True, cache: Optional[Cache] = None, run: Callable = subprocess.run,
          environ=None, version: Optional[str] = None, recheck: Optional[bool] = None,
          busy: Optional[Callable[[str], Optional[str]]] = None) -> Optional[WideResult]:
    """Decide the widths for this engine and open them (``engine.set_wide``): ``ids`` is the text the check reads (check_ids), with
    ``check=False`` only a kept result is used.  ``recheck`` (default: ``engine.wide_recheck``) measures even when a kept result fits.
    ``busy(uuid)`` replaces the nvidia-smi load check (tests).  Returns None for engine libraries without the 0.1.7 exports when wide
    passes are not asked for."""
    from . import __version__
    version = version or __version__
    t_start = time.perf_counter()
    mode = getattr(engine, "wide_mode", "auto")
    res = WideResult(mode=mode)
    st = engine.wide_state() if hasattr(engine, "wide_state") else None
    if st is None:
        if mode in ("on", "force"):
            res.status = "old-library"
            res.reason = "this engine library is older than 0.1.7 and cannot switch them per GPU, so --wide %s has no effect" % mode
            res.line = describe(res)
            return res
        return None
    res.native, res.code, res.cc, res.mask = st.native, st.code_text, st.cc_text, st.mask
    if not st.allowed or st.builtin_draft or not st.drafter or int(engine.spec.get("copy_rows", 16)) <= 16:
        res.status, res.mask = "off", 0
        if st.allowed and st.mask:
            engine.set_wide(0)
        res.line = describe(res)
        return res
    if st.pref == 2:                                                  # --wide force: every width, no check
        res.status = "force"
        res.line = describe(res)
        return res
    fallback = _closed_mask(st)
    smi = smi_fields(st.uuid, KEY_FIELDS, run, environ) or {}
    res.smi = dict(smi)
    if not smi.get("driver_version"):
        res.status, res.mask = "unknown-gpu", fallback
        res.reason = "nvidia-smi gives nothing for this GPU, so it cannot be checked; " + (
            "all widths as in 0.1.6" if st.native else "none on this GPU")
        engine.set_wide(fallback)
        res.line = describe(res)
        return res
    fields = key_fields(engine, st, smi, version)
    res.key = key_of(fields)
    cache = cache if cache is not None else Cache()
    recheck = bool(getattr(engine, "wide_recheck", False)) if recheck is None else recheck
    hit = None if recheck else cache.get(res.key)
    if hit is not None:
        r = hit["result"]
        res.widths, res.gate, res.timing = _restore(r)
        res.status, res.cache_hit = "cached", True
        res.mask = int(r.get("mask", fallback)) & 7
        res.times_us = [int(x) for x in (r.get("times_us") or [])]
        res.reason = str(r.get("reason", ""))
        if r.get("status") == "mismatch":
            res.status, res.mask = "mismatch", 0
        engine.set_wide(res.mask, res.times_us or None)
        if res.status == "mismatch":
            engine.set_exact_wide(False)                              # as right after the check (below)
        engine._wide_key, engine._wide_cache = res.key, cache
        res.line = describe(res)
        return res
    if check and ids is not None and engine.max_ctx < CONTEXT + 64 + 64:
        res.status, res.mask = "no-check", fallback
        res.reason = "the context of %d tokens is shorter than the check needs (%d); %s" % (
            engine.max_ctx, CONTEXT + 128, "all widths as in 0.1.6" if st.native else "none on this GPU")
        engine.set_wide(fallback)
        res.line = describe(res)
        return res
    if not check or ids is None or not engine.supports_wide_check:
        res.status, res.mask = "no-check", fallback
        res.reason = "not checked at this start; " + ("all widths as in 0.1.6" if st.native else "none on this GPU")
        engine.set_wide(fallback)
        res.line = describe(res)
        return res
    why = (busy or (lambda u: busy_reason(u, run, environ)))(st.uuid)
    if why:
        res.status, res.mask = "busy", fallback
        res.reason = "not checked: %s; %s" % (why, "all widths as in 0.1.6" if st.native else "none on this GPU")
        engine.set_wide(fallback)
        res.line = describe(res)
        return res
    widths = st.checkable_widths
    try:
        engine.set_wide(mask_of(widths))                              # open for the check (RTX 30 / 40: the exact-sum GEMV too)
        gate = correctness(engine, ids)
        res.gate = gate
        if not gate.ok:
            res.status, res.mask = "mismatch", 0
            res.reason = ("passes of more than 16 rows do not compute the same as passes of 16 on this GPU (most likely id differs "
                          "at %d, NLL at %d of %d positions%s): no wide pass is used; please report it on GitHub Issues"
                          % (gate.argmax_diff, gate.nll_diff, gate.positions, ("; " + gate.error) if gate.error else ""))
            engine.set_wide(0)
            engine.set_exact_wide(False)                              # passes of 17..64 rows (ends of prompt reads) as 16-row sums
        else:
            timing = measure(engine, ids, widths)
            res.timing = timing
            res.mask, res.widths = decide(timing, st.native, widths)
            res.times_us = times_us(timing)
            res.status = "measured"
            engine.set_wide(res.mask, res.times_us)
    except (YasakaniError, ValueError) as ex:
        res.status, res.mask = "error", fallback
        res.reason = "the check failed (%s); %s" % (ex, "all widths as in 0.1.6" if st.native else "none on this GPU")
        try:
            engine.reset()
            engine.set_wide(fallback)
        except YasakaniError:
            pass
        res.line = describe(res)
        return res
    except BaseException:
        try:
            engine.set_wide(fallback)                                 # e.g. Ctrl+C: never leave the widths of the check open
        except BaseException:                                         # noqa: BLE001
            pass
        raise
    after = busy_reason(st.uuid, run, environ, reads=1) if busy is None else busy(st.uuid)
    res.check_seconds = time.perf_counter() - t_start
    if after and res.status == "measured":
        res.cache_note = "not kept: %s after the check" % after
    elif res.status in ("measured", "mismatch"):
        if cache.put(res.key, fields, dict(res.summary(), status=res.status)):
            engine._wide_key, engine._wide_cache = res.key, cache
        else:
            res.cache_note = cache.warning
    res.line = describe(res) + (("; " + res.cache_note) if res.cache_note else "")
    return res


def after_prepare(engine, res: Optional[WideResult]) -> None:
    """After the graphs of the open widths are built: on RTX 30 / 40, close them again when the GPU memory is nearly full (Windows
    would then move memory to the PC's RAM and every pass would become slower)."""
    if res is None:
        return
    st = engine.wide_state()
    if st is None:
        return
    res.free_after_mib = st.free_mib
    if res.native or not res.mask:
        return
    if st.free_mib < MIN_FREE_MIB:
        engine.set_wide(0)
        res.status, res.mask = "memory", 0
        res.reason = "only %d MiB of GPU memory are free with their graphs (at least %d needed)" % (st.free_mib, MIN_FREE_MIB)
        res.line = describe(res)


def recheck_due(wide_passes: int, backoffs: int) -> bool:
    """Do the back-offs of real generations (counted in one process) call for a new check?  At least RECHECK_RATE back-offs per 100
    wide passes, over at least RECHECK_MIN_PASSES wide passes.  A few back-offs (e.g. where copies end) do not."""
    return wide_passes >= RECHECK_MIN_PASSES and 100 * backoffs >= RECHECK_RATE * wide_passes


def note_backoffs(engine) -> None:
    """Called after a generation that used wide passes.  When :func:`recheck_due`, the kept result is marked, so the next start checks
    again."""
    key, cache = getattr(engine, "_wide_key", ""), getattr(engine, "_wide_cache", None)
    if not key or cache is None or getattr(engine, "_wide_marked", False):
        return
    if recheck_due(getattr(engine, "_wide_passes_seen", 0), getattr(engine, "_wide_backoffs", 0)):
        engine._wide_marked = True
        cache.mark_recheck(key)
