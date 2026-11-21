# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The GPU the engine runs on, and a context length that fits its memory.

Since 0.1.3 the engine library is built for the RTX 30, 40 and 50 series (compute capability 8.6, 8.9 and 12.x).  It has been tested and
tuned on the RTX 5090 only; on other cards it runs without that tuning, and its speed and operation are not guaranteed.

The engine reserves the key-value cache for the whole context when the model is opened.  On a card with less memory than the RTX 5090,
the default context of 262,144 tokens may not fit next to the model: Windows then keeps part of it in ordinary memory and generation
becomes several times slower, without an error.  :func:`plan_context` therefore picks, before the model is opened, the longest context
that fits the free GPU memory.  On cards with 30,000 MiB or more (the RTX 5090) it changes nothing: the context stays the model's own.

The numbers below were measured on the RTX 5090 with the released engine (Windows, CUDA context included); on smaller cards the fixed
part is expected to be a little smaller, so the rule errs on the safe side.  nvidia-smi is the only source: nothing here starts CUDA.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ._native import YasakaniError

MIB = 1 << 20
KV_BYTES_PER_TOKEN = 22400            # KV5 cache 22,144 B + position table 256 B per token of context, reserved at open
CONTEXT_STEP = 16384                  # a chosen context is a multiple of this
MIN_CONTEXT = 16384                   # below this the model is not opened
HEAD_OFF_BELOW = 65536                # under this, the draft model's output head is left out to make room
MARGIN_MIB = 512                      # the desktop's drift, CUDA graphs, image input (at most about 84 MiB), rounding
OVERHEAD_MIB = {"fastest": 2100, "quality": 2300}    # GPU memory above the three model files (measured 2,090 and 2,291 MiB)
QUALITY_WEIGHTS_MIB = 11000           # weights larger than this are counted with the overhead of the quality model (also the
                                      # max-quality model: after load it used 3,182 MiB more than the quality model on the
                                      # RTX 5090, and its weights file is 3,203 MiB larger)
FULL_CONTEXT_MIN_TOTAL_MIB = 30000    # cards with this much memory keep the model's own context (as before 0.1.3)
YARN2_MAX_CTX = 524288                # experimental --yarn2 context limit (2x the model's native 262,144 tokens)
BIG_CARD_LOAD_EXTRA_MIB = 700         # extra load memory of runtime.py's 2048/8192 defaults over SMALL_CARD_ENV (about 0.6-0.7 GB)
# Cards with less than FULL_CONTEXT_MIN_TOTAL_MIB keep the prompt-reading sizes of 0.1.7: OVERHEAD_MIB was measured with them.  The
# larger defaults of runtime.py (2,048 rows per pass, 8,192 rows of layer work area) take about 0.6 to 0.7 GB more at load.
SMALL_CARD_ENV = {"YSKN_Q35_PMAX": "1024", "YSKN_Q35_LAYER_ROWS": "4096"}
# --prefill (or YASAKANI_PREFILL): auto (the default) = the new, faster prompt reading on a card whose memory is known and at least
# FULL_CONTEXT_MIN_TOTAL_MIB (the card it is tested on), else the reading of 0.1.7; new = the new reading on any card (under
# FULL_CONTEXT_MIN_TOTAL_MIB or of unknown memory with the sizes of SMALL_CARD_ENV); old = prompt reading and passes of 64 rows as in
# 0.1.7.  The engine reads YSKN_Q35_PF_V2=0 and then also keeps the sizes of 0.1.7; they are given here as well.
PREFILL_MODES = ("auto", "new", "old")
PREFILL_MODE_ENV = "YASAKANI_PREFILL"
OLD_PREFILL_ENV = dict(SMALL_CARD_ENV, YSKN_Q35_PF_V2="0")
NEW_PREFILL_AREA_MIB = 579            # the work area the new prompt reading makes at the sizes of SMALL_CARD_ENV, at the first long
                                      # prompt, and keeps until the model is closed (the engine reports 607,322,112 B; measured on an
                                      # RTX 5090 shown as a 24 GB and a 16 GB card: +569 and +607 MiB at the peak)
NEW_PREFILL_SPARE_MIB = 1024          # the engine makes that area only with this much to spare, else it reads prompts without it (slower, same output)
NEW_PREFILL_CUDA_GAP_MIB = 566        # CUDA reported this much less free memory than nvidia-smi on the PC of the measurements (the
                                      # RTX 5090, Windows 11); the engine asks CUDA, the planners ask nvidia-smi
NEW_PREFILL_RESERVE_MIB = 1800        # what the planners leave free for it: area + spare + gap - MARGIN_MIB, and about 100 MiB of room
# --edit-copy (or YASAKANI_EDIT_COPY): on | off, passed to the engine as YSKN_EDIT_COPY=1 / 0.  EDIT_COPY_DEFAULT is the default when
# neither is given; changing it is all it takes to change the default.
EDIT_COPY_MODES = ("on", "off")
EDIT_COPY_ENV = "YASAKANI_EDIT_COPY"
EDIT_COPY_DEFAULT = "on"
MIN_TOTAL_MIB = 15000                 # "16 GB": the smallest cards the released models fit on (fastest model only)
QUALITY_MIN_TOTAL_MIB = 23000         # "24 GB": the quality and the max-quality model need this much
HEAD_SLOWER = "about 3%"              # each pass without the output head, measured on the RTX 5090
# serve --parallel N (experimental): the carrier and every one of the N stream slots hold the KV of a whole context
POSITION_TABLE_BYTES = 256            # the carrier's position table per token of context (the rest of KV_BYTES_PER_TOKEN is its KV)
SLOT_ALLOC_MIB = 64                   # one slot takes this much more than the engine counts (yskn_ms_info): the driver rounds the
                                      # slot's many buffers up (measured 32 to 48 MiB per slot on the RTX 5090)
SLOT_MIB = 256                        # one slot besides its KV (recurrent states, histories, drafter memory): the engine reports
                                      # 187 MiB for the released models (yskn_ms_info), + SLOT_ALLOC_MIB, rounded up here
PARALLEL_MAX = 64                     # stream slots at most (the engine's limit)

QUERY_FIELDS = ("index", "uuid", "name", "compute_cap", "memory.total", "memory.used", "driver_version")
_TESTED = re.compile(r"(^|\s)RTX 5090$")


@dataclass(frozen=True)
class GpuInfo:
    """One row of nvidia-smi.  Values nvidia-smi could not give are None."""
    index: int
    name: str
    uuid: Optional[str] = None
    compute_cap: Optional[Tuple[int, int]] = None
    total_mib: Optional[int] = None
    used_mib: Optional[int] = None
    driver: Optional[str] = None

    @property
    def free_mib(self) -> Optional[int]:
        if self.total_mib is None or self.used_mib is None:
            return None
        return max(0, self.total_mib - self.used_mib)

    @property
    def tested(self) -> bool:
        """The card everything is tested and tuned on: the NVIDIA GeForce RTX 5090 (32 GB)."""
        return bool(_TESTED.search(self.name.strip())) and self.compute_cap in (None, (12, 0))

    @property
    def cc_text(self) -> str:
        return "%d.%d" % self.compute_cap if self.compute_cap else "?"

    @property
    def arch(self) -> str:
        """"ok" (8.6 to 8.9, 12.x: the released engine has code for it), "old" (below 8.6), "no-code" (9.x to 11.x, 13 and newer)
        or "unknown" (nvidia-smi did not tell)."""
        cc = self.compute_cap
        if cc is None:
            return "unknown"
        if cc < (8, 6):
            return "old"
        if cc < (9, 0) or cc[0] == 12:
            return "ok"
        return "no-code"


def smi_path(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The nvidia-smi to ask.  ``YASAKANI_NVIDIA_SMI`` names another one; set to ``none`` (or empty) nothing is asked."""
    env = os.environ if environ is None else environ
    if "YASAKANI_NVIDIA_SMI" in env:
        p = env["YASAKANI_NVIDIA_SMI"].strip()
        return p if p and p.lower() != "none" else None
    return shutil.which("nvidia-smi")


def _value(s: str) -> Optional[str]:
    s = s.strip()
    return None if (not s or s.startswith("[") or s.upper() == "N/A") else s


def _int(s: Optional[str]) -> Optional[int]:
    try:
        return int(round(float(s))) if s is not None else None
    except ValueError:
        return None


def _cc(s: Optional[str]) -> Optional[Tuple[int, int]]:
    m = re.fullmatch(r"(\d+)\.(\d+)", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse(text: str, fields: Sequence[str] = QUERY_FIELDS) -> List[GpuInfo]:
    """The output of ``nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits`` -> one GpuInfo per line.  Lines that do not
    have one value per field (an error text, "No devices were found") are skipped."""
    out = []
    for n, line in enumerate(ln for ln in text.splitlines() if ln.strip()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        v = dict(zip(fields, parts))
        name = _value(v.get("name", "")) or ""
        index = _int(_value(v.get("index", ""))) if "index" in v else n
        if not name or index is None:
            continue
        out.append(GpuInfo(index=index, name=name, uuid=_value(v.get("uuid", "")), compute_cap=_cc(_value(v.get("compute_cap", ""))),
                           total_mib=_int(_value(v.get("memory.total", ""))), used_mib=_int(_value(v.get("memory.used", ""))),
                           driver=_value(v.get("driver_version", ""))))
    return out


def query(environ: Optional[Mapping[str, str]] = None, run: Callable = subprocess.run, timeout: float = 15) -> Optional[List[GpuInfo]]:
    """Every GPU nvidia-smi lists, or None when it cannot be asked.  Drivers that do not know ``compute_cap`` are asked without it."""
    exe = smi_path(environ)
    if not exe:
        return None
    for fields in (QUERY_FIELDS, tuple(f for f in QUERY_FIELDS if f != "compute_cap")):
        try:
            r = run([exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=timeout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        if r.returncode == 0:
            return parse(r.stdout or "", fields)
    return None


@dataclass
class Detected:
    """What is known about the GPU the engine will use.  ``candidates`` has one entry when it is known which GPU that is, several when
    the PC has more than one and CUDA's numbering cannot be matched with nvidia-smi's, and none when nothing is known.  ``numbered`` is
    True when CUDA's device numbers are nvidia-smi's indexes (one GPU, or PCI bus order without ``CUDA_VISIBLE_DEVICES``)."""
    gpus: List[GpuInfo] = field(default_factory=list)
    candidates: List[GpuInfo] = field(default_factory=list)
    device: int = 0
    numbered: bool = False

    @property
    def gpu(self) -> Optional[GpuInfo]:
        return self.candidates[0] if len(self.candidates) == 1 else None

    @property
    def known(self) -> bool:
        return bool(self.candidates)

    @property
    def tested(self) -> bool:
        return bool(self.candidates) and all(g.tested for g in self.candidates)

    @property
    def untested(self) -> bool:
        """True only when the GPU is known and it is not the RTX 5090 (unknown is not "untested")."""
        return bool(self.candidates) and not self.tested

    def worst(self) -> Optional[GpuInfo]:
        """The GPU itself, or for several candidates the smallest memory and the least free memory of them."""
        if not self.candidates:
            return None
        if len(self.candidates) == 1:
            return self.candidates[0]
        totals = [g.total_mib for g in self.candidates]
        frees = [g.free_mib for g in self.candidates]
        if None in totals or None in frees:
            return None
        caps = [g.compute_cap for g in self.candidates if g.compute_cap]
        t, f = min(totals), min(frees)
        return replace(self.candidates[0], name="the smallest of %d GPUs" % len(self.candidates), uuid=None, total_mib=t,
                       used_mib=t - f, compute_cap=min(caps) if caps else None)


def _pci_order(env: Mapping[str, str]) -> bool:
    return (env.get("CUDA_DEVICE_ORDER") or "").strip().upper() == "PCI_BUS_ID"


def has_uuid(g: Optional[GpuInfo]) -> bool:
    """True when nvidia-smi gave the GPU's UUID (``GPU-...``), the name ``CUDA_VISIBLE_DEVICES`` is set to (:func:`visible_only`)."""
    return g is not None and bool(g.uuid) and g.uuid.upper().startswith("GPU-")


def numbering(environ: Optional[Mapping[str, str]] = None, gpus: Optional[Sequence[GpuInfo]] = None) -> str:
    """How ``--device`` relates to nvidia-smi's GPU numbers (one line for ``yasakani info`` on a PC with several GPUs).  With ``gpus``
    (nvidia-smi's list) it also says whether the commands that load the model let the engine see only that GPU (0.1.6): not for a GPU
    whose UUID nvidia-smi did not give."""
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" in env:
        return "--device counts only the GPUs in CUDA_VISIBLE_DEVICES=%s, in that order" % env.get("CUDA_VISIBLE_DEVICES")
    if not _pci_order(env):
        return "--device follows CUDA_DEVICE_ORDER=%s, not the GPU numbers above" % env.get("CUDA_DEVICE_ORDER")
    line = "--device N is GPU N above (PCI bus order)"
    if gpus is None:
        return line
    missing = [g.index for g in gpus if not has_uuid(g)]
    if not missing:
        return line + "; the engine then sees only that GPU"
    if len(missing) == len(gpus):
        return line + "; nvidia-smi gave no UUIDs, so the engine also sees the other GPUs"
    return line + "; the engine then sees only that GPU, except GPU %s (nvidia-smi gave no UUID): then it also sees the other GPUs" % (
        ", ".join(str(i) for i in missing))


def visible_only(det: Detected, environ: Optional[Mapping[str, str]] = None) -> Optional[GpuInfo]:
    """0.1.6 (issue #20): the GPU to make the only one CUDA sees, or None to leave CUDA's view of the GPUs as it is.

    On a PC with two GPUs, the engine was reported to fail ("graph instantiate failed") when it ran on a GPU that was not CUDA's
    device 0 while the other GPU was visible too; the same GPU passed when it was the only one visible (as the quickstart sets it).
    So when nvidia-smi lists more than one GPU, ``CUDA_VISIBLE_DEVICES`` is not set and it is known which GPU ``--device`` is (PCI
    bus order), the command line sets ``CUDA_VISIBLE_DEVICES`` to the UUID of that GPU before the engine library is loaded and opens
    the engine as device 0, as the quickstart does for the GPU it checked.  None on a PC with one GPU, when ``CUDA_VISIBLE_DEVICES``
    is set (even empty: the user's choice is kept), and when the GPU or its UUID is not known."""
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" in env or len(det.gpus) < 2 or not det.numbered:
        return None
    g = det.gpu
    return g if has_uuid(g) else None


def select(gpus: Sequence[GpuInfo], device: int = 0, environ: Optional[Mapping[str, str]] = None) -> List[GpuInfo]:
    """The GPU(s) that CUDA device ``device`` can be.  nvidia-smi numbers GPUs in PCI order; CUDA does too only with
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (its default puts the fastest first; since 0.1.5 the package sets PCI_BUS_ID when it is not set,
    :func:`yasakani._native.pci_bus_order`), and ``CUDA_VISIBLE_DEVICES`` renumbers them."""
    env = os.environ if environ is None else environ
    gpus = list(gpus)
    if not gpus or device < 0:
        return []
    pci_order = _pci_order(env)
    if "CUDA_VISIBLE_DEVICES" in env:
        entries = [e.strip() for e in (env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if e.strip()]
        if device >= len(entries):
            return []
        e = entries[device]
        if e.upper().startswith("GPU-"):
            m = [g for g in gpus if g.uuid and g.uuid.lower().startswith(e.lower())]
            return m if len(m) == 1 else gpus
        if e.isdigit() and (pci_order or len(gpus) == 1):
            return [g for g in gpus if g.index == int(e)]
        return gpus
    if len(gpus) == 1:
        return gpus if device == 0 else []
    if pci_order:
        return [g for g in gpus if g.index == device]
    return gpus


def detect(device: int = 0, environ: Optional[Mapping[str, str]] = None, gpus: Optional[Sequence[GpuInfo]] = None,
           run: Callable = subprocess.run) -> Detected:
    """Ask nvidia-smi once (unless ``gpus`` is given) and match CUDA device ``device``.  Never raises."""
    if gpus is None:
        gpus = query(environ, run=run) or []
    env = os.environ if environ is None else environ
    numbered = bool(gpus) and "CUDA_VISIBLE_DEVICES" not in env and (len(gpus) == 1 or _pci_order(env))
    return Detected(gpus=list(gpus), candidates=select(gpus, device, environ), device=device, numbered=numbered)


def describe(g: GpuInfo) -> str:
    """One line on what 0.1.3 does on this GPU."""
    if g.tested:
        return "RTX 5090: the card YASAKANI is tested and tuned on"
    if g.arch == "old":
        return "compute capability %s: below 8.6, the released engine has no code for it" % g.cc_text
    if g.arch == "no-code":
        return "compute capability %s: the released engine has no code for it (it is built for 8.6, 8.9 and 12.x)" % g.cc_text
    if g.total_mib is not None and g.total_mib < MIN_TOTAL_MIB:
        return "%d MiB: too small, the released models need a 16 GB card" % g.total_mib
    small = g.total_mib is not None and g.total_mib < QUALITY_MIN_TOTAL_MIB
    return ("runs since 0.1.3, but not optimized or tested (only the RTX 5090 is); speed and operation are not guaranteed"
            + ("; fastest model only" if small else ""))


def untested_note(det: Detected) -> Optional[str]:
    """The line printed at start on any GPU that is not the RTX 5090."""
    if not det.untested:
        return None
    g = det.gpu
    what = "%s (compute capability %s)" % (g.name, g.cc_text) if g else "this PC's GPUs (%s)" % ", ".join(sorted({x.name for x in det.candidates}))
    note = ("note: %s - YASAKANI runs on it since 0.1.3, but it is not optimized or tested there (it is tested on the RTX 5090 only); "
            "speed and operation are not guaranteed." % what)
    if g and g.arch in ("old", "no-code"):
        note += "  The released engine has no code for compute capability %s, so it will probably not start." % g.cc_text
    return note


@dataclass
class ContextPlan:
    """What :func:`plan_context` decided.  ``max_ctx`` None = the model's own context (unchanged); ``env`` = engine settings to add."""
    max_ctx: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    free_mib: Optional[int] = None
    parallel: int = 1                     # stream slots the context was chosen for (plan_parallel; 1 = one request at a time)
    prefill: str = ""                     # the prompt reading use_prefill chose: "new" or "old" ("" = not chosen yet)
    prefill_line: str = ""                # the one line on that choice, printed at start
    edit_copy: str = ""                   # use_edit_copy: "on" or "off" ("" = not chosen yet)
    reserve_mib: int = 0                  # GPU memory left free besides the context (prefill_reserve_mib; 0 = none)


def _small_card_env(plan: "ContextPlan", g) -> None:
    """On a card with less than FULL_CONTEXT_MIN_TOTAL_MIB: the prompt-reading sizes of 0.1.7 (SMALL_CARD_ENV), so that the overhead
    the context is chosen with stays true."""
    if g is not None and g.total_mib is not None and g.total_mib < FULL_CONTEXT_MIN_TOTAL_MIB:
        plan.env.update(SMALL_CARD_ENV)


def prefill_mode_of(value: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> str:
    """auto | new | old: ``value`` (the command line), else ``YASAKANI_PREFILL``, else auto."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(PREFILL_MODE_ENV, "")
    v = (v or "auto").strip().lower()
    if v not in PREFILL_MODES:
        raise YasakaniError("%s / --prefill must be one of %s, not %r" % (PREFILL_MODE_ENV, ", ".join(PREFILL_MODES), v))
    return v


def _card_total(g) -> Tuple[Optional[int], int]:
    """(total memory of the card in MiB or None when not known, how many GPUs it was chosen from).  ``g`` is a GpuInfo or a Detected;
    for several candidates (CUDA's numbering cannot be matched) the smallest total counts, when every total is known."""
    if isinstance(g, Detected):
        if len(g.candidates) > 1:
            totals = [c.total_mib for c in g.candidates]
            return (None if None in totals else min(totals)), len(totals)
        g = g.gpu
    return (g.total_mib if g is not None else None), 1


def prefill_choice(mode: str, g) -> Tuple[str, Dict[str, str], str]:
    """What ``mode`` (auto | new | old) means on the card ``g`` (a GpuInfo, a Detected or None): (reading, engine settings, line).
    reading is "new" or "old"; the line says which reading and why, printed once at start (as the line of the wide copy passes)."""
    total, n = _card_total(g)
    big = total is not None and total >= FULL_CONTEXT_MIN_TOTAL_MIB
    if total is None:
        card = "card memory not known"
    elif n > 1:
        card = "smallest of %d GPUs: %d MiB" % (n, total)
    else:
        card = "%d MiB card" % total
    if mode == "old":
        return "old", dict(OLD_PREFILL_ENV), "prompt reading: as in 0.1.7 (setting old)"
    if mode == "new":
        if big:
            return "new", {}, "prompt reading: new (setting new)"
        return "new", dict(SMALL_CARD_ENV), "prompt reading: new (setting new; %s, so prompt sizes of 0.1.7)" % card
    if big:
        return "new", {}, "prompt reading: new (setting auto: %s)" % card
    why = "%s, new only from %d MiB" % (card, FULL_CONTEXT_MIN_TOTAL_MIB) if total is not None else card
    return "old", dict(OLD_PREFILL_ENV), ("prompt reading: as in 0.1.7 (setting auto: %s; --prefill new or %s=new to try it)"
                                          % (why, PREFILL_MODE_ENV))


def prefill_reserve_mib(mode: str, g) -> int:
    """GPU memory the planners leave free for the prompt reading that ``mode`` gives on the card ``g``: NEW_PREFILL_RESERVE_MIB for
    the new reading on a card under FULL_CONTEXT_MIN_TOTAL_MIB or of unknown memory (only --prefill new gives that), else 0."""
    reading, env, _ = prefill_choice(mode, g)
    return NEW_PREFILL_RESERVE_MIB if reading == "new" and env else 0


NO_ROOM_NOTE = ("note: --prefill new: keeping %d MiB free for the work area of the new prompt reading would cost more than half of the "
                "context, so the context is chosen without it; the engine reads prompts without that work area when it finds no room")


def _reserve_note(plan: "ContextPlan", reserve_mib: int, chosen: bool) -> None:
    plan.reserve_mib = int(reserve_mib)
    if not reserve_mib:
        return
    if chosen:
        plan.notes.append("note: --prefill new: the context is chosen with %d MiB more left free (the new prompt reading keeps a work "
                          "area of about %d MiB from the first long prompt on; the engine makes it only with %d MiB to spare, and "
                          "reads prompts without it when it finds less)" % (reserve_mib, NEW_PREFILL_AREA_MIB, NEW_PREFILL_SPARE_MIB))
    else:
        plan.notes.append("note: --prefill new: the new prompt reading needs about %d MiB free in nvidia-smi after the model is loaded "
                          "(a work area of about %d MiB, made only with %d MiB to spare); with less, prompts are read without that work area.  "
                          "The context you gave is used as it is"
                          % (NEW_PREFILL_AREA_MIB + NEW_PREFILL_SPARE_MIB + NEW_PREFILL_CUDA_GAP_MIB, NEW_PREFILL_AREA_MIB,
                             NEW_PREFILL_SPARE_MIB))


def use_prefill(plan: "ContextPlan", mode: str, g=None) -> "ContextPlan":
    """Add the engine settings of :func:`prefill_choice` for ``mode`` on the card ``g`` to ``plan.env`` and keep the choice in
    ``plan.prefill`` and ``plan.prefill_line``.  old, and auto on a card under FULL_CONTEXT_MIN_TOTAL_MIB or of unknown memory, add
    OLD_PREFILL_ENV; auto on a larger card changes nothing; new adds SMALL_CARD_ENV unless the card is known to be that large."""
    reading, env, line = prefill_choice(mode, g)
    plan.env.update(env)
    plan.prefill, plan.prefill_line = reading, line
    return plan


def edit_copy_of(value: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> str:
    """on | off: ``value`` (the command line), else ``YASAKANI_EDIT_COPY``, else EDIT_COPY_DEFAULT."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(EDIT_COPY_ENV, "")
    v = (v or EDIT_COPY_DEFAULT).strip().lower()
    if v not in EDIT_COPY_MODES:
        raise YasakaniError("%s / --edit-copy must be one of %s, not %r" % (EDIT_COPY_ENV, ", ".join(EDIT_COPY_MODES), v))
    return v


def use_edit_copy(plan: "ContextPlan", mode: str) -> "ContextPlan":
    """Pass ``mode`` (on | off) to the engine as YSKN_EDIT_COPY=1 / 0 in ``plan.env``, and keep it in ``plan.edit_copy``."""
    if mode not in EDIT_COPY_MODES:
        raise YasakaniError("--edit-copy must be one of %s, not %r" % (", ".join(EDIT_COPY_MODES), mode))
    plan.env["YSKN_EDIT_COPY"] = "1" if mode == "on" else "0"
    plan.edit_copy = mode
    return plan


def overhead_mib(weights_bytes: int) -> int:
    return OVERHEAD_MIB["quality"] if weights_bytes / MIB > QUALITY_WEIGHTS_MIB else OVERHEAD_MIB["fastest"]


def fit_tokens(free_mib: float, files_bytes: int, overhead: int) -> int:
    """How many tokens of context fit: (free - files - overhead - margin) / bytes per token (0 when nothing fits)."""
    room = free_mib - files_bytes / MIB - overhead - MARGIN_MIB
    return max(0, int(room * MIB // KV_BYTES_PER_TOKEN))


def choose(free_mib: float, weights_bytes: int, draft_bytes: int, head_bytes: int, model_max: int) -> Tuple[int, bool]:
    """The rule: (context, head_off).  context is ``model_max`` when it fits, else a multiple of CONTEXT_STEP, 0 when even
    MIN_CONTEXT does not fit.  Under HEAD_OFF_BELOW the draft model's output head is left out when that gives more room."""
    over = overhead_mib(weights_bytes)

    def ctx(files):
        n = fit_tokens(free_mib, files, over)
        return model_max if n >= model_max else (n // CONTEXT_STEP) * CONTEXT_STEP

    with_head = ctx(weights_bytes + draft_bytes + head_bytes)
    if with_head >= min(HEAD_OFF_BELOW, model_max):
        return with_head, False
    without = ctx(weights_bytes + draft_bytes)
    if without > with_head:
        return without, True
    return with_head, False


def _which_gpu(det: Detected, sizes: Sequence[int], model_max: int) -> str:
    """For the message when the model does not fit, on a PC with several GPUs: which GPU that was, and another one with room
    ("" on a PC with one GPU, so that its message stays as before)."""
    if len(det.gpus) < 2:
        return ""
    if len(det.candidates) > 1:
        return ("This PC has %d GPUs and CUDA's device %d cannot be matched with nvidia-smi's list, so these are the numbers of the "
                "smallest of them (unset CUDA_DEVICE_ORDER or set it to PCI_BUS_ID to match them).  " % (len(det.candidates), det.device))
    g = det.gpu
    if g is None:
        return ""
    s = "That is nvidia-smi's GPU %d (%s)" % (g.index, g.name)
    if det.numbered:
        room = [o for o in det.gpus if o.index != g.index and o.arch == "ok" and o.free_mib is not None
                and choose(o.free_mib, sizes[0], sizes[1], sizes[2], model_max)[0] >= MIN_CONTEXT]
        if room:
            o = max(room, key=lambda o: (o.tested, o.free_mib, -o.index))
            s += "; GPU %d (%s, %d MiB free) has room for it: add --device %d" % (o.index, o.name, o.free_mib, o.index)
    return s + ".  "


def plan_context(pack, det: Detected, reserve_mib: int = 0, max_ctx_limit: Optional[int] = None) -> ContextPlan:
    """The context for ``pack`` on the detected GPU.  Unchanged (None) on cards with FULL_CONTEXT_MIN_TOTAL_MIB or more, and when
    nvidia-smi gave no answer (even with ``--yarn2``, so a card with 24 GB or less cannot silently spill into shared memory).  When free
    memory is known and ``max_ctx_limit`` exceeds ``pack.max_ctx`` (e.g. ``--yarn2``), the longer context is planned against free GPU memory
    on every card.  Raises YasakaniError when not even MIN_CONTEXT tokens fit.  ``reserve_mib`` (prefill_reserve_mib) is taken off the free
    memory before the context is chosen."""
    plan = ContextPlan()
    limit = pack.max_ctx if max_ctx_limit is None else int(max_ctx_limit)
    extended = limit > pack.max_ctx
    g = det.worst()
    _small_card_env(plan, g)
    if g is None or g.total_mib is None or g.free_mib is None:
        if extended:
            plan.notes.append("note: GPU free memory could not be read, so --yarn2 keeps the model's native %d-token context; "
                              "pass --max-ctx to set a longer context" % pack.max_ctx)
        if len(det.candidates) > 1:
            plan.notes.append("note: this PC has %d GPUs and their memory could not be read; the context stays %d tokens"
                              % (len(det.candidates), pack.max_ctx))
        return plan
    if not extended and g.total_mib >= FULL_CONTEXT_MIN_TOTAL_MIB:
        return plan
    plan.free_mib = g.free_mib
    sizes = [p.stat().st_size for p in (pack.weights, pack.draft, pack.draft_head)]
    extra = BIG_CARD_LOAD_EXTRA_MIB if (extended and g.total_mib >= FULL_CONTEXT_MIN_TOTAL_MIB) else 0
    n, head_off = choose(g.free_mib - reserve_mib - extra, sizes[0], sizes[1], sizes[2], limit)
    if reserve_mib:                              # not at the cost of more than half of the context: then the plan of the other
        plain = choose(g.free_mib - extra, sizes[0], sizes[1], sizes[2], limit)          # readings, and the engine decides
        if n < MIN_CONTEXT or 2 * n < plain[0]:
            n, head_off = plain
            if n >= MIN_CONTEXT:
                plan.notes.append(NO_ROOM_NOTE % reserve_mib)
            reserve_mib = 0
    if len(det.candidates) > 1:
        plan.notes.append("note: this PC has %d GPUs and CUDA's device %d cannot be matched with nvidia-smi's list, so the context is "
                          "chosen for the smallest of them (set CUDA_DEVICE_ORDER=PCI_BUS_ID to match them)" % (len(det.candidates), det.device))
    if n < MIN_CONTEXT:
        raise YasakaniError(
            "the model does not fit in the free GPU memory (%d of %d MiB free; it needs about %d MiB with a context of %d tokens).  "
            "%sClose other programs that use the GPU%s, or give --max-ctx to try anyway."
            % (g.free_mib, g.total_mib, int((sizes[0] + sizes[1]) / MIB + overhead_mib(sizes[0]) + MARGIN_MIB + MIN_CONTEXT * KV_BYTES_PER_TOKEN / MIB),
               MIN_CONTEXT, _which_gpu(det, sizes, limit), "" if sizes[0] / MIB <= QUALITY_WEIGHTS_MIB else ", or use the fastest model"))
    if head_off:
        plan.env["YSKN_BDR_SHEAD"] = "0"
        plan.notes.append("note: to leave room for the context, the output head of the draft model (%s, %.1f GiB) is not loaded; "
                          "each pass takes %s longer (measured on the RTX 5090)" % (pack.draft_head.name, sizes[2] / (1 << 30), HEAD_SLOWER))
    if extended:
        plan.max_ctx = n
        if n == limit:
            plan.notes.append("context: %d tokens (2x YaRN, experimental; %d of %d MiB free); --max-ctx sets it yourself"
                              % (n, g.free_mib, g.total_mib))
        elif n > pack.max_ctx:
            plan.notes.append("context: %d tokens (2x YaRN, experimental), chosen from the free GPU memory (%d of %d MiB free); "
                              "--max-ctx sets it yourself" % (n, g.free_mib, g.total_mib))
        else:
            plan.notes.append("context: %d tokens, chosen from the free GPU memory (%d of %d MiB free); this fits within the model's "
                              "native %d-token context, so --yarn2 does not give a longer context here (it only changes the outputs)"
                              % (n, g.free_mib, g.total_mib, pack.max_ctx))
    elif n < pack.max_ctx:
        plan.max_ctx = n
        plan.notes.append("context: %d tokens, chosen from the free GPU memory (%d of %d MiB free); --max-ctx sets it yourself"
                          % (n, g.free_mib, g.total_mib))
    _reserve_note(plan, reserve_mib, True)
    return plan


def plan_given_context(pack, det: Detected, max_ctx: int, reserve_mib: int = 0) -> ContextPlan:
    """The plan for a context given with --max-ctx.  The context stays as given (it is tried even where the rule says it does not fit),
    but on a card with less than FULL_CONTEXT_MIN_TOTAL_MIB (or when ``max_ctx`` exceeds ``pack.max_ctx``, e.g. ``--yarn2``) the draft
    model's output head is left out when it does not fit beside that context.  Without this the head is always loaded: the package names
    it in YSKN_BDR_SHEAD, and the engine loads a named head without the free-memory check it makes for one it finds by itself.
    Unchanged on larger cards for native contexts and when nvidia-smi gave no answer."""
    plan = ContextPlan(max_ctx=int(max_ctx))
    extended = plan.max_ctx > pack.max_ctx
    g = det.worst()
    _small_card_env(plan, g)
    if g is None or g.total_mib is None or g.free_mib is None or (not extended and g.total_mib >= FULL_CONTEXT_MIN_TOTAL_MIB):
        return plan
    plan.free_mib = g.free_mib
    sizes = [p.stat().st_size for p in (pack.weights, pack.draft, pack.draft_head)]
    extra = BIG_CARD_LOAD_EXTRA_MIB if (extended and g.total_mib >= FULL_CONTEXT_MIN_TOTAL_MIB) else 0
    _reserve_note(plan, reserve_mib, False)
    if fit_tokens(g.free_mib - reserve_mib - extra, sum(sizes), overhead_mib(sizes[0])) < plan.max_ctx:
        plan.env["YSKN_BDR_SHEAD"] = "0"
        plan.notes.append("note: the output head of the draft model (%s, %.1f GiB) is not loaded: it does not fit beside %d tokens of "
                          "context in the free GPU memory (%d of %d MiB free); each pass takes %s longer (measured on the RTX 5090)"
                          % (pack.draft_head.name, sizes[2] / (1 << 30), plan.max_ctx, g.free_mib, g.total_mib, HEAD_SLOWER))
        if extended and fit_tokens(g.free_mib - reserve_mib - extra, sizes[0] + sizes[1], overhead_mib(sizes[0])) < plan.max_ctx:
            plan.notes.append("warning: %d tokens of context may not fit in the free GPU memory (%d of %d MiB free); trying anyway.  "
                              "When the GPU memory overflows, Windows keeps part of it in ordinary memory and everything becomes several "
                              "times slower, without an error.  Lower --max-ctx if needed."
                              % (plan.max_ctx, g.free_mib, g.total_mib))
    return plan


# ---- several requests at once (serve --parallel N; experimental) ------------------------------------------------------------------
def parallel_fit(free_mib: float, files_bytes: int, overhead: int, n: int, kv_bytes: int = KV_BYTES_PER_TOKEN,
                 slot_bytes: int = SLOT_MIB * MIB) -> int:
    """How many tokens of context fit for the carrier and ``n`` stream slots, all with the same context:
    (n + 1) x context x kv_bytes + n x slot_bytes + files + overhead + margin <= free (0 when nothing fits)."""
    room = free_mib - files_bytes / MIB - overhead - MARGIN_MIB - n * slot_bytes / MIB
    return max(0, int(room * MIB // ((n + 1) * kv_bytes)))


def choose_parallel(free_mib: float, weights_bytes: int, draft_bytes: int, head_bytes: int, model_max: int, n: int,
                    kv_bytes: int = KV_BYTES_PER_TOKEN, slot_bytes: int = SLOT_MIB * MIB) -> Tuple[int, bool]:
    """:func:`choose` for ``n`` stream slots: (context, head_off).  The same steps and the same rule for the draft model's output head
    (left out under HEAD_OFF_BELOW when that gives more room)."""
    over = overhead_mib(weights_bytes)

    def ctx(files):
        k = parallel_fit(free_mib, files, over, n, kv_bytes, slot_bytes)
        return model_max if k >= model_max else (k // CONTEXT_STEP) * CONTEXT_STEP

    with_head = ctx(weights_bytes + draft_bytes + head_bytes)
    if with_head >= min(HEAD_OFF_BELOW, model_max):
        return with_head, False
    without = ctx(weights_bytes + draft_bytes)
    if without > with_head:
        return without, True
    return with_head, False


def plan_parallel(pack, det: Detected, n: int, reserve_mib: int = 0, max_ctx_limit: Optional[int] = None) -> ContextPlan:
    """The context for ``n`` requests at once (serve --parallel n): the carrier and every stream slot take the same context, the
    longest that fits the free GPU memory, a multiple of CONTEXT_STEP and at most ``max_ctx_limit`` (default: the model's own); on
    every card (the RTX 5090 included, where the model's own context does not fit n + 1 times).  The draft model's output head is left
    out by the rule of :func:`choose`.  Raises YasakaniError when not even MIN_CONTEXT tokens fit (naming the most streams that do).
    When nvidia-smi gives no free memory, the plan is for one request at a time (``parallel`` 1, with a note)."""
    n = int(n)
    limit = pack.max_ctx if max_ctx_limit is None else int(max_ctx_limit)
    extended = limit > pack.max_ctx
    plan = ContextPlan(parallel=n)
    g = det.worst()
    _small_card_env(plan, g)
    if g is None or g.total_mib is None or g.free_mib is None:
        plan.parallel = 1
        plan.notes.append("note: --parallel %d chooses the context from the free GPU memory, and nvidia-smi could not tell it; serving "
                          "one request at a time (give --max-ctx to choose the context yourself)" % n)
        return plan
    plan.free_mib = g.free_mib
    sizes = [p.stat().st_size for p in (pack.weights, pack.draft, pack.draft_head)]
    extra = BIG_CARD_LOAD_EXTRA_MIB if (extended and g.total_mib >= FULL_CONTEXT_MIN_TOTAL_MIB) else 0
    free = g.free_mib - reserve_mib - extra
    ctx, head_off = choose_parallel(free, sizes[0], sizes[1], sizes[2], limit, n)
    if reserve_mib:                              # not at the cost of more than half of the context (as plan_context)
        plain = choose_parallel(g.free_mib - extra, sizes[0], sizes[1], sizes[2], limit, n)
        if ctx < MIN_CONTEXT or 2 * ctx < plain[0]:
            ctx, head_off = plain
            if ctx >= MIN_CONTEXT:
                plan.notes.append(NO_ROOM_NOTE % reserve_mib)
            free, reserve_mib = g.free_mib - extra, 0
    if ctx < MIN_CONTEXT:
        fits = [m for m in range(n - 1, 1, -1)
                if choose_parallel(free, sizes[0], sizes[1], sizes[2], limit, m)[0] >= MIN_CONTEXT]
        raise YasakaniError(
            "--parallel %d does not fit in the free GPU memory (%d of %d MiB free): every request needs a context of at least %d tokens "
            "of its own.  %s" % (n, g.free_mib, g.total_mib, MIN_CONTEXT, ("--parallel %d fits." % fits[0]) if fits
                                 else "Even --parallel 2 does not fit: serve one request at a time."))
    if head_off:
        plan.env["YSKN_BDR_SHEAD"] = "0"
        plan.notes.append("note: to leave room for the contexts, the output head of the draft model (%s, %.1f GiB) is not loaded; "
                          "each pass takes %s longer (measured on the RTX 5090)"
                          % (pack.draft_head.name, sizes[2] / (1 << 30), HEAD_SLOWER))
    plan.max_ctx = ctx
    plan.notes.append("context: %d tokens for each of %d requests at once, chosen from the free GPU memory (%d of %d MiB free); "
                      "--max-ctx sets it yourself" % (ctx, n, g.free_mib, g.total_mib))
    if extended and ctx <= pack.max_ctx:
        plan.notes.append("note: %d tokens per request fits within the model's native %d-token context, so --yarn2 does not give a "
                          "longer context here (it only changes the outputs)" % (ctx, pack.max_ctx))
    _reserve_note(plan, reserve_mib, True)
    return plan


def plan_given_parallel(pack, det: Detected, max_ctx: int, n: int, reserve_mib: int = 0) -> ContextPlan:
    """The plan for ``n`` requests at once with a context given by --max-ctx: the context stays as given (as for one request, it is
    tried even where the rule says it does not fit, with a warning); the draft model's output head is left out when the contexts fit
    only without it.  Unchanged when nvidia-smi gave no answer."""
    plan = ContextPlan(max_ctx=int(max_ctx), parallel=int(n))
    g = det.worst()
    _small_card_env(plan, g)
    if g is None or g.total_mib is None or g.free_mib is None:
        return plan
    plan.free_mib = g.free_mib
    sizes = [p.stat().st_size for p in (pack.weights, pack.draft, pack.draft_head)]
    over = overhead_mib(sizes[0])
    _reserve_note(plan, reserve_mib, False)
    free = g.free_mib - reserve_mib
    if parallel_fit(free, sum(sizes), over, n) >= plan.max_ctx:
        return plan
    if parallel_fit(free, sizes[0] + sizes[1], over, n) >= plan.max_ctx:
        plan.env["YSKN_BDR_SHEAD"] = "0"
        plan.notes.append("note: the output head of the draft model (%s, %.1f GiB) is not loaded: it does not fit beside %d requests "
                          "with %d tokens of context each in the free GPU memory (%d of %d MiB free); each pass takes %s longer "
                          "(measured on the RTX 5090)" % (pack.draft_head.name, sizes[2] / (1 << 30), n, plan.max_ctx, g.free_mib,
                                                          g.total_mib, HEAD_SLOWER))
        return plan
    plan.notes.append("warning: %d requests with %d tokens of context each do not fit in the free GPU memory (%d of %d MiB free); "
                      "trying anyway.  When the GPU memory overflows, Windows keeps part of it in ordinary memory and everything becomes "
                      "several times slower, without an error.  Lower --max-ctx or --parallel."
                      % (n, plan.max_ctx, g.free_mib, g.total_mib))
    return plan


def parallel_need_mib(files_bytes: int, weights_bytes: int, ctx: int, n: int, kv_bytes: int, slot_bytes: int) -> float:
    """GPU memory for the model, the carrier's context and ``n`` stream slots by the engine's own numbers (yskn_ms_info: the KV bytes
    of one position of one sequence in its KV format, the bytes of one slot besides its KV): (n + 1) x ctx x kv_bytes + the carrier's
    position table + n x (slot_bytes + SLOT_ALLOC_MIB) + the files, the overhead and the margin."""
    return (files_bytes / MIB + overhead_mib(weights_bytes) + MARGIN_MIB
            + (ctx * (kv_bytes + POSITION_TABLE_BYTES) + n * (ctx * kv_bytes + slot_bytes + SLOT_ALLOC_MIB * MIB)) / MIB)


def check_parallel(pack, plan: Optional[ContextPlan], info, n: int) -> Tuple[int, Optional[str]]:
    """After the model is open: how many of ``n`` stream slots to make, by what the engine says of itself (``info``, a
    :class:`yasakani.engine.MsInfo`): 1 when the multi-stream pass does not take the engine's settings (e.g. an 8-bit KV), else the
    most slots up to ``n`` whose KV (in the engine's own KV format) fits the free memory the plan was made for.  Returns that number
    and a note (None when it is ``n``).  Without a known free memory the number stays ``n`` (a binding that does not fit fails, and the
    server then serves one request at a time)."""
    n = int(n)
    if info.unsupported:
        return 1, ("note: --parallel %d is off: the multi-stream pass does not take this engine's settings (%s); serving one request "
                   "at a time" % (n, info.reason))
    if plan is None or plan.free_mib is None:
        return n, None
    w = pack.weights.stat().st_size
    files = w + pack.draft.stat().st_size + (pack.draft_head.stat().st_size if info.draft_head else 0)
    m = n
    room = plan.free_mib - plan.reserve_mib      # --prefill new on a card under FULL_CONTEXT_MIN_TOTAL_MIB keeps its reserve
    kept = " (besides %d MiB kept for the new prompt reading)" % plan.reserve_mib if plan.reserve_mib else ""
    while m >= 2 and parallel_need_mib(files, w, info.max_ctx, m, info.kv_bytes_per_position, info.slot_bytes) > room:
        m -= 1
    if m == n:
        return n, None
    need = parallel_need_mib(files, w, info.max_ctx, n, info.kv_bytes_per_position, info.slot_bytes)
    if m >= 2:
        return m, ("note: by the engine's own numbers %d requests with %d tokens of context each need about %d MiB and %d MiB are free%s; "
                   "serving %d at a time" % (n, info.max_ctx, need, room, kept, m))
    return 1, ("note: by the engine's own numbers %d requests with %d tokens of context each need about %d MiB and %d MiB are free%s, and "
               "not even 2 fit; serving one request at a time" % (n, info.max_ctx, need, room, kept))
