# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Command line:  ``python -m yasakani  info | serve | chat | bench | selfcheck | verify | report``  (``yasakani`` when installed with
pip)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil                                    # noqa: F401  (gpu.smi_path looks nvidia-smi up through it; tests patch cli.shutil.which)
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__, BUILD_NOTE
from . import gpu as gpus
from . import widecheck
from ._native import NativeLib, YasakaniError, find_library, library_loaded, pci_bus_order
from .runtime import ModelPack, RuntimeProfile
from .selfcheck import FORMAT as SELFCHECK_FORMAT
from .selfcheck import cases as _selfcheck_cases, ids_sha as _ids_sha, run as _selfcheck_run    # noqa: F401  (names of 0.1.3: bench scripts use them)

DEFAULT_PORT = 8765
DEFAULT_MAX_QUEUE = 64                           # serve --parallel N: requests that may wait for a stream
CHECK_FILE = "parallel_check.json"               # results of the start-up check of --parallel (once per library, GPU and driver)


# ---- small helpers ----------------------------------------------------------------------------------------------------------------
def _say(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0
    return "%.1f GiB" % n


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and (os.name != "nt" or bool(os.environ.get("WT_SESSION") or os.environ.get("TERM")))


def _fix_console() -> None:
    """Never fail on a character the console cannot show (older Windows consoles)."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _model_dir(args) -> Path:
    d = getattr(args, "model_dir", None) or os.environ.get("YASAKANI_MODEL_DIR")
    if not d:
        raise YasakaniError("give the model folder with --model-dir (or set YASAKANI_MODEL_DIR)")
    return Path(d)


def _gpu_lines() -> List[str]:
    exe = gpus.smi_path()                       # nvidia-smi, or the one YASAKANI_NVIDIA_SMI names
    if not exe:
        return ["nvidia-smi not found (is the NVIDIA driver installed?)"]
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as ex:
        return ["nvidia-smi failed: " + type(ex).__name__]
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return lines or ["nvidia-smi reported no GPU"]


def _gpu_memory_warning(device: Optional[int] = 0) -> Optional[str]:
    """After loading: is the GPU's memory (almost) full?  When the model does not fit, Windows keeps part of it in ordinary memory
    and generation becomes several times slower, without any error.  None when all is well or when it cannot be told.  ``device`` is
    nvidia-smi's row of the GPU (None: not known which)."""
    exe = gpus.smi_path()                       # nvidia-smi, or the one YASAKANI_NVIDIA_SMI names
    if not exe or device is None:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        rows = [[float(x) for x in ln.split(",")] for ln in r.stdout.splitlines() if ln.strip()]
        total, used = rows[0] if len(rows) == 1 else rows[device]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    if total > 0 and used >= 0.95 * total:
        return ("warning: the GPU memory is almost full (%d of %d MiB in use).  When the model does not fit, Windows keeps part of it "
                "in ordinary memory and generation becomes several times slower, without an error.  Close other programs that use "
                "the GPU, or lower --max-ctx." % (used, total))
    return None


def _visible_gpu(det, device: int, environ=None, cuda_started: Optional[bool] = None) -> int:
    """0.1.6 (issue #20): on a PC with several GPUs, make the GPU of ``--device`` the only one the engine sees
    (``CUDA_VISIBLE_DEVICES`` = its UUID, :func:`yasakani.gpu.visible_only`) and say so in one line.  Returns the device number to open
    the engine with: 0 then, else ``device`` unchanged (one GPU, ``CUDA_VISIBLE_DEVICES`` set by the user, GPU not known).  When
    nvidia-smi gave no UUID for that GPU, nothing is changed and the line says that the engine also sees the other GPUs.  CUDA reads
    the variable once, when it starts, so nothing is changed once an engine library has been loaded in this process."""
    env = os.environ if environ is None else environ
    g = gpus.visible_only(det, env)
    if g is None:
        k = det.gpu
        if k is not None and len(det.gpus) > 1 and det.numbered and not gpus.has_uuid(k):
            _say("gpu: GPU %d of nvidia-smi (%s); nvidia-smi gave no UUID for it, so the engine also sees the other GPUs" % (k.index, k.name))
        return device
    if (library_loaded() if cuda_started is None else cuda_started):
        return device
    env["CUDA_VISIBLE_DEVICES"] = g.uuid
    _say("gpu: GPU %d of nvidia-smi (%s); the engine sees only this GPU (CUDA_VISIBLE_DEVICES is set to its UUID)" % (g.index, g.name))
    return 0


def _add_engine_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model-dir", help="folder with the model files (default: $YASAKANI_MODEL_DIR)")
    sp.add_argument("--dll", help="path of yskn.dll (default: $YASAKANI_DLL, or the bin folder next to the package)")
    sp.add_argument("--max-ctx", type=int, default=None,
                    help="context length in tokens: input + thinking + answer (default: the model's 262144; on a GPU with less than "
                         "30,000 MiB, the longest that fits the free GPU memory)")
    sp.add_argument("--device", type=int, default=0, help="the GPU to use, numbered as in `yasakani info` and nvidia-smi (default 0)")
    sp.add_argument("--seed0", type=int, default=1, help="seed of the first request; every request without its own seed uses the next one")
    sp.add_argument("--prefill", choices=gpus.PREFILL_MODES, default=None,
                    help="prompt reading: auto (default; the new, faster reading on a card with 30,000 MiB or more, else as in 0.1.7), "
                         "new (the new reading on any card; under 30,000 MiB with the prompt sizes of 0.1.7) or old (prompt reading and "
                         "passes of 64 tokens as in 0.1.7).  Default: $YASAKANI_PREFILL or auto")
    sp.add_argument("--edit-copy", choices=gpus.EDIT_COPY_MODES, default=None,
                    help="edit copy: on or off.  Default: $YASAKANI_EDIT_COPY or %s" % gpus.EDIT_COPY_DEFAULT)
    _add_wide_args(sp)


def _add_yarn2_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--yarn2", action="store_true",
                    help="experimental: 2x YaRN rotary scaling, planning context up to %d tokens (or --max-ctx up to %d).  "
                         "Outputs differ from the native rotary table at every length; quality beyond 262144 tokens and with the "
                         "5-bit KV cache has not been measured" % (gpus.YARN2_MAX_CTX, gpus.YARN2_MAX_CTX))


def _add_wide_args(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("wide copy passes")
    g.add_argument("--wide", choices=widecheck.MODES, default=None,
                   help="copy passes of up to 64 tokens: auto (default; RTX 30 / 40: only the widths a check on this GPU finds correct "
                        "and faster; RTX 50 on, a width this GPU finds slower left out), on (the same as auto since 0.1.7), off, "
                        "force (every width without the check).  Default: $YASAKANI_WIDE or auto")
    g.add_argument("--wide-recheck", action="store_true", help="check the wide copy passes again even when a kept result fits")


def _add_vision_args(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("image input (optional)")
    g.add_argument("--mmproj", default=os.environ.get("YASAKANI_MMPROJ"),
                   help="projector of Qwen3.8-27B (mmproj-Qwen3.8-27B-BF16.gguf); with --llama-dir and --vision-model it turns on image input")
    g.add_argument("--llama-dir", default=os.environ.get("YASAKANI_LLAMA_DIR"),
                   help="folder with the unpacked llama-b10488-bin-win-cpu-x64.zip (llama.cpp b10488, CPU build for Windows); "
                        "on Linux, with the shared libraries of llama.cpp b10488 built from source (see BUILDING.md)")
    g.add_argument("--vision-model", default=os.environ.get("YASAKANI_VISION_MODEL"),
                   help="a Qwen3.8-27B GGUF (tested: Qwen3.8-27B-Q4_K_M.gguf of bartowski/Qwen3.8-27B-GGUF); only its settings and "
                        "vocabulary are read, never its weights")
    g.add_argument("--vision-cache", default=None, help="folder to keep image features in (default: none)")
    # the defaults are those of yasakani/vision.py (DEFAULT_IMAGE_TOKENS, IMAGE_TOKENS_RANGE, REQUEST_IMAGE_TOKENS_FACTOR); the
    # numbers are written out here so that the help needs no import of numpy (tests/test_vision.py compares them)
    g.add_argument("--vision-threads", type=int, default=None,
                   help="CPU threads for encoding an image (default: one per physical CPU core)")
    g.add_argument("--vision-max-image-tokens", type=int, default=1024, metavar="N",
                   help="downsize every image to at most N image tokens, 64 to 4096 (default 1024; the encode time grows with the "
                        "square of N: about 15 s for 1024 on a 16-core CPU)")
    g.add_argument("--vision-max-request-image-tokens", type=int, default=None, metavar="N",
                   help="at most N image tokens in one request, the images of earlier turns included (default: 8 x the per-image limit)")
    g.add_argument("--vision-allow-local-files", metavar="DIR", default=None,
                   help="also accept local file paths in image_url, for files inside DIR only (default: only data:image URIs)")


def _vision(args):
    """VisionProcessor when all three of --mmproj, --llama-dir and --vision-model are given, else None (text only, as before).
    The options are checked here, before the model loads."""
    given = [getattr(args, k, None) for k in ("mmproj", "llama_dir", "vision_model")]
    if not any(given):
        return None
    if not all(given):
        raise YasakaniError("image input needs all three of --mmproj, --llama-dir and --vision-model")
    from .vision import DEFAULT_IMAGE_TOKENS, VisionProcessor, error_text
    try:                                         # the paths become absolute here, before anything changes the working folder
        vision = VisionProcessor(mmproj=args.mmproj, llama_dir=args.llama_dir, model=args.vision_model, cache=args.vision_cache,
                                 threads=args.vision_threads, allow_local_files=args.vision_allow_local_files or False,
                                 max_image_tokens=getattr(args, "vision_max_image_tokens", DEFAULT_IMAGE_TOKENS),
                                 max_request_image_tokens=getattr(args, "vision_max_request_image_tokens", None))
        vision.check()
    except (ValueError, OSError, RuntimeError, ImportError) as ex:
        raise YasakaniError("image input: %s" % error_text(ex)) from ex
    return vision


def _open_stack(args, *, warm: bool = True, parallel: int = 1, allow_yarn2: bool = True):
    """Model folder -> (pack, engine, formatter, runner).  The tokenizer is loaded first so a broken folder fails before the long load.
    ``parallel`` N >= 2 (serve --parallel N): the context is chosen for N requests at once (``gpu.plan_parallel``); afterwards
    ``args.parallel_planned`` is the N the engine was opened for (1 when the engine library has no multi-stream entries or the free GPU
    memory is not known) and ``args.context_plan`` the plan."""
    from .chat import ChatRunner
    from .engine import Engine
    from .textio import ChatFormatter
    prefill = gpus.prefill_mode_of(getattr(args, "prefill", None))       # a wrong YASAKANI_PREFILL or YASAKANI_EDIT_COPY fails first
    edit_copy = gpus.edit_copy_of(getattr(args, "edit_copy", None))
    yarn2 = bool(getattr(args, "yarn2", False))
    cmd = getattr(args, "cmd", None)
    if yarn2 and (not allow_yarn2 or cmd == "selfcheck"):
        raise YasakaniError("selfcheck does not support --yarn2: the reference files were made with the native rotary table")
    if yarn2 and args.max_ctx is not None and args.max_ctx > gpus.YARN2_MAX_CTX:
        raise YasakaniError("--yarn2 supports at most %d context tokens (got %d)" % (gpus.YARN2_MAX_CTX, args.max_ctx))
    pack = ModelPack.load(_model_dir(args))
    fmt = ChatFormatter(pack.root)
    vision = _vision(args)                       # wrong image input options fail here, before the long load
    det = gpus.detect(args.device)               # nvidia-smi only; CUDA is not started here
    args.gpu_detected = det
    note = gpus.untested_note(det)
    if note:
        _say(note)
    # several GPUs: the engine sees only the chosen one, as its device 0.  Before any engine library is loaded (--parallel loads it
    # below to look for the multi-stream entries): CUDA reads CUDA_VISIBLE_DEVICES only once.
    device = _visible_gpu(det, args.device)
    n, lib, plan = int(parallel or 1), None, None
    reserve = gpus.prefill_reserve_mib(prefill, det)     # --prefill new on a card under 30,000 MiB: room for its extra memory
    if n >= 2:                                   # several requests at once, if the engine library has the entries
        lib = NativeLib(args.dll)
        if not lib.supports_parallel:
            _say("note: this engine library has no multi-stream entries (yskn_ms_*), so --parallel %d is off: serving one request at "
                 "a time" % n)
            n = 1
    ctx_limit = gpus.YARN2_MAX_CTX if yarn2 else None
    if n >= 2:
        plan = (gpus.plan_parallel(pack, det, n, reserve, max_ctx_limit=ctx_limit) if args.max_ctx is None
                else gpus.plan_given_parallel(pack, det, args.max_ctx, n, reserve))
        if plan.parallel < 2:                    # the free memory is not known
            for line in plan.notes:
                _say(line)
            n, plan = 1, None
    if plan is None and args.max_ctx is None:    # 0.1.3: a context that fits the free GPU memory (unchanged on the RTX 5090)
        plan = gpus.plan_context(pack, det, reserve, max_ctx_limit=ctx_limit)
    elif plan is None:                           # a given context is used as it is; the draft model's output head only where it fits
        plan = gpus.plan_given_context(pack, det, args.max_ctx, reserve)
    gpus.use_prefill(plan, prefill, det)
    gpus.use_edit_copy(plan, edit_copy)
    args.parallel_planned, args.context_plan = n, plan
    for line in plan.notes:
        _say(line)
    if yarn2:
        _say("note: --yarn2 (experimental): 2x YaRN rotary scaling is on; outputs differ from the native rotary table at every length, "
             "and quality beyond %d tokens or with the 5-bit KV cache has not been measured" % pack.max_ctx)
        if args.max_ctx is not None and args.max_ctx <= pack.max_ctx:
            _say("note: --yarn2 with --max-ctx %d does not exceed the model's native %d-token context (outputs still differ from the "
                 "native rotary table)" % (args.max_ctx, pack.max_ctx))
    _say(plan.prefill_line)
    if plan.edit_copy == "on":
        _say("edit copy: on")
    max_ctx = plan.max_ctx
    extra_env = dict(plan.env) if plan.env else {}
    if yarn2:
        if max_ctx is not None and max_ctx > gpus.YARN2_MAX_CTX:
            raise YasakaniError("--yarn2 supports at most %d context tokens (got %d)" % (gpus.YARN2_MAX_CTX, max_ctx))
        extra_env["YSKN_Q35_YARN2"] = "1"
    profile = RuntimeProfile.default(pack, extra_env) if extra_env else None
    replaced = [k for k in extra_env if k in os.environ]     # set in the shell, but the plan's value is used: named with the removed ones
    wide = widecheck.mode_of(getattr(args, "wide", None))         # 0.1.7: a wrong YASAKANI_WIDE fails here, before the long load
    _say("loading the model (%s of weights) ..." % _human(pack.weights.stat().st_size))
    t0 = time.time()
    eng = Engine(pack, max_ctx=max_ctx, device=device, dll=args.dll, profile=profile, wide=wide, **({"lib": lib} if lib is not None else {}))
    try:
        ignored = sorted(set(eng.removed_env) | set(replaced))
        if ignored:
            _say("note: ignored %d YSKN_* setting(s) from the environment: %s" % (len(ignored), ", ".join(ignored)))
        kept = sorted(k for k in os.environ if k.startswith("YSKN_TEST_"))
        if kept:
            _say("note: kept %s from the environment (read only by an engine library built with the test exports)" % ", ".join(kept))
        eng.wide_recheck = bool(getattr(args, "wide_recheck", False))
        runner = ChatRunner(eng, fmt, seed0=args.seed0, vision=vision)
        if warm:
            runner.warmup(say=_say)
        else:                                                    # --no-warmup: a kept result of the wide check, nothing measured
            res = widecheck.setup(eng, None, check=False)
            eng.wide_result = res
            if res is not None and res.line:
                _say(res.line)
    except BaseException:
        eng.close()
        raise
    _say("model ready in %.0f s (context %d tokens)" % (time.time() - t0, eng.max_ctx))
    note = _gpu_memory_warning(det.gpu.index if det.gpu else (0 if len(det.gpus) < 2 else None))    # nvidia-smi's row, when it is known
    if note:
        _say(note)
    return pack, eng, fmt, runner


# ---- several requests at once (serve --parallel N; experimental) ------------------------------------------------------------------
def _parallel_count(value, env: bool = False) -> int:
    """--parallel N (serve: else $YASAKANI_PARALLEL), else 1."""
    if value is None and env:
        v = (os.environ.get("YASAKANI_PARALLEL") or "").strip()
        if v:
            try:
                value = int(v)
            except ValueError:
                raise YasakaniError("YASAKANI_PARALLEL must be a whole number, not %r" % v)
    n = 1 if value is None else int(value)
    if not 1 <= n <= gpus.PARALLEL_MAX:
        raise YasakaniError("--parallel must be between 1 and %d" % gpus.PARALLEL_MAX)
    return n


def _seconds_option(value, env: str, option: str, *, off: bool = False) -> Optional[float]:
    """An option given as "auto" (None), "off" (0; only where ``off``) or seconds; when it is not given, the environment variable
    ``env``; else "auto"."""
    from .sched import KEEP_TURN_LIMIT_S
    raw = value if value is not None else ((os.environ.get(env) or "").strip() or "auto")
    v = str(raw).strip().lower()
    if v == "auto":
        return None
    if off and v == "off":
        return 0.0
    try:
        x = float(v)
    except ValueError:
        x = float("nan")
    if not 0.0 <= x <= KEEP_TURN_LIMIT_S:                     # also NaN
        raise YasakaniError("%s (or %s) must be auto%s or seconds from 0 to %g, not %r"
                            % (option, env, ", off" if off else "", KEEP_TURN_LIMIT_S, raw))
    return x


def _turn_settings(args) -> dict:
    """--keep-turn and --max-wait (or $YASAKANI_KEEP_TURN, $YASAKANI_MAX_WAIT): the arguments of the scheduler for the next turn after
    tool calls."""
    return dict(keep_turn=_seconds_option(getattr(args, "keep_turn", None), "YASAKANI_KEEP_TURN", "--keep-turn", off=True),
                max_wait=_seconds_option(getattr(args, "max_wait", None), "YASAKANI_MAX_WAIT", "--max-wait"))


def _turn_line(sched) -> str:
    """What the server says at start about the stream kept for the next turn after tool calls."""
    from .sched import KEEP_TURN_MAX_S, KEEP_TURN_MIN_S, WAIT_FLOOR_S
    if sched.keep_turn == 0.0:
        return "after tool calls nothing is kept for the next turn"
    if sched.keep_turn is None:
        keep = "about as long as reading the conversation again takes (%g to %g s)" % (KEEP_TURN_MIN_S, KEEP_TURN_MAX_S)
    else:
        keep = "%g s" % sched.keep_turn
    if sched.max_wait is None:
        wait = "as long as reading that conversation again takes, and %g s" % WAIT_FLOOR_S
    else:
        wait = "%g s" % sched.max_wait
    return ("after tool calls a request's place is kept for its next turn %s; a new request waits for next turns at most %s"
            % (keep, wait))


def _cache_dir() -> Path:
    """Where the package keeps results between runs: $YASAKANI_CACHE_DIR, else the user's cache folder."""
    d = os.environ.get("YASAKANI_CACHE_DIR")
    if d:
        return Path(d)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "yasakani"
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "yasakani"


def _check_key(args, pack, eng, sched) -> Optional[str]:
    """What the start-up check of --parallel depends on: the engine library (its SHA-256), the GPU and its driver, the model, the
    package and the slots.  None when the GPU is not known (the check then runs at every start)."""
    det = getattr(args, "gpu_detected", None)
    g = det.gpu if det is not None else None
    if g is None:
        return None
    try:
        lib = hashlib.sha256(Path(eng.lib.path).read_bytes()).hexdigest()
    except (OSError, AttributeError):
        return None
    parts: dict[str, object] = dict(library=lib, gpu=g.name, uuid=g.uuid, compute_capability=g.cc_text, driver=g.driver, model=pack.model_id,
                 weights=pack.weights.stat().st_size, version=__version__, slots=sched.n_slots, rows=sched.rows, drafter=sched.drafter)
    if getattr(args, "yarn2", False):
        parts["yarn2"] = True
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _check_cache(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _check_remember(path: Path, key: str, record: dict) -> None:
    """Keep one result in the cache file (a file that cannot be written only costs the check at the next start)."""
    try:
        d = _check_cache(path)
        d[key] = record
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(tmp), str(path))
    except OSError:
        pass


def _parallel_check(args, pack, eng, fmt, sched, *, use_cache: bool = True) -> Optional[bool]:
    """The start-up check of --parallel: N short requests at once in the stream slots must give the ids each gives alone (the
    single path).  The result is kept per engine library, GPU and driver.  True passed, False differs, None could not be run."""
    from . import sched as S
    path = _cache_dir() / CHECK_FILE
    key = _check_key(args, pack, eng, sched)
    if use_cache and key is not None:
        rec = _check_cache(path).get(key)
        if isinstance(rec, dict) and "ok" in rec:
            if rec["ok"]:
                _say("parallel check: passed before with this engine library, GPU and driver")
                return True
            _say("parallel check: failed before with this engine library, GPU and driver (%s).  Delete %s to check again."
                 % (rec.get("detail"), path))
            return False
    _say("checking that %d requests at once give the same output as one at a time (once per engine library, GPU and driver) ..."
         % sched.n_slots)
    try:
        res = S.consistency_check(sched, S.check_prompts(fmt, sched.n_slots))
    except YasakaniError as ex:
        _say("parallel check could not run: %s" % ex)
        return None
    if res.compared == 0:
        _say("parallel check could not compare: %s" % res.line())
        return None
    if key is not None:
        _check_remember(path, key, dict(ok=res.ok, detail=res.line(), version=__version__,
                                        checked=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())))
    _say("parallel check %s: %s" % ("passed" if res.ok else "FAILED", res.line()))
    return res.ok


def _start_parallel(args, pack, eng, fmt, *, check: bool = True):
    """After the model is open (and warmed up): the scheduler for ``args.parallel_planned`` requests at once, or None (one request
    at a time; the reason is printed).  ``check`` (--parallel-check): "auto" the start-up check once per engine library, GPU and
    driver, "always" at every start, "off" none; True is "auto", False none."""
    from .sched import Scheduler
    n = int(getattr(args, "parallel_planned", 1) or 1)
    if n < 2:
        return None
    try:
        info = eng.ms_info()
    except YasakaniError as ex:
        _say("note: --parallel %d is off (%s); serving one request at a time" % (n, ex))
        return None
    m, note = gpus.check_parallel(pack, getattr(args, "context_plan", None), info, n)
    if note:
        _say(note)
    if m < 2:
        args.parallel_planned = 1
        return None
    turn = _turn_settings(args)
    try:
        sched = Scheduler(eng, m, rows=getattr(args, "parallel_rows", 16), max_queue=getattr(args, "max_queue", DEFAULT_MAX_QUEUE),
                          reuse=not getattr(args, "no_slot_reuse", False), **turn)
    except (YasakaniError, ValueError) as ex:
        _say("note: --parallel %d is off: the stream slots could not be made (%s); serving one request at a time" % (m, ex))
        args.parallel_planned = 1
        return None
    args.parallel_planned = m
    mode = getattr(args, "parallel_check", "auto") if check is True else ("off" if check is False else check)
    if mode != "off":
        ok = _parallel_check(args, pack, eng, fmt, sched, use_cache=mode != "always")
        if not ok:
            why = ("%d requests at once gave other output than one at a time on this GPU (please report it on GitHub Issues with the "
                   "output of yasakani info)" % m) if ok is False else "the start-up check did not run"
            _say("note: --parallel %d is off: %s; serving one request at a time" % (m, why))
            sched.close()
            eng.reset()                          # nothing of the check stays on the engine
            args.parallel_planned = 1
            return None
    return sched


# ---- info -------------------------------------------------------------------------------------------------------------------------
def cmd_info(args) -> int:
    ok = True
    print("yasakani", __version__, "(%s)" % BUILD_NOTE if BUILD_NOTE else "")
    print("python  ", platform.python_version(), platform.system(), platform.machine())
    for mod in ("numpy", "tokenizers", "jinja2"):
        try:
            m = importlib.import_module(mod)
            print("%-8s %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("%-8s MISSING  (pip install %s)" % (mod, mod))
            ok = False
    try:
        dll = find_library(args.dll)
        h = hashlib.sha256(dll.read_bytes()).hexdigest()
        print("engine   %s  %s  sha256 %s" % (dll.name, _human(dll.stat().st_size), h[:16]))
    except YasakaniError as ex:
        print("engine   NOT FOUND:", ex)
        ok = False
    for ln in _gpu_lines():
        print("gpu      " + ln)
    found = gpus.query() or []
    for g in found:
        print("         GPU %d: compute capability %s - %s" % (g.index, g.cc_text, gpus.describe(g)))
    if len(found) > 1:
        print("         " + gpus.numbering(gpus=found))
    for line in _wide_info_lines(found):
        print(line)
    for line in _prefill_info_lines(found):
        print(line)
    d = getattr(args, "model_dir", None) or os.environ.get("YASAKANI_MODEL_DIR")
    if d:
        try:
            pack = ModelPack.load(d)
            print("model    %s  (id %s, context up to %d tokens)" % (pack.root.name, pack.model_id, pack.max_ctx))
            for label, p in (("weights", pack.weights), ("draft", pack.draft), ("draft head", pack.draft_head), ("kv profile", pack.kv_profile)):
                print("  %-10s %-26s %s" % (label, p.name, _human(p.stat().st_size)))
            print("  checksums  %s" % ("%d files listed in yasakani.json" % len(pack.sha256) if pack.sha256 else "none listed (yasakani.json missing)"))
            from .textio import ChatFormatter
            ChatFormatter(pack.root)
            print("  tokenizer  ok (chat template found)")
        except (YasakaniError, ImportError, OSError, ValueError) as ex:
            print("model    PROBLEM:", ex)
            ok = False
    else:
        print("model    (give --model-dir to check the model folder)")
    return 0 if ok else 1


def _wide_info_lines(found) -> List[str]:
    """0.1.7: the setting of the wide copy passes and, per GPU, the latest kept result of the check (no paths)."""
    try:
        mode = widecheck.mode_of(None)
    except YasakaniError as ex:
        return ["wide     " + str(ex)]
    out = ["wide     copy passes of up to 64 tokens: setting %s (YASAKANI_WIDE / --wide)" % mode]
    entries = widecheck.Cache().load()
    for g in found:
        if not g.uuid:
            continue
        h = hashlib.sha256(g.uuid.lower().encode()).hexdigest()[:16]
        e = next((e for e in entries if (e.get("fields") or {}).get("uuid_sha") == h), None)
        if e is None:
            continue
        r = e["result"]
        when = time.strftime("%Y-%m-%d", time.localtime(float(e.get("time", 0))))
        out.append("         GPU %d: checked %s (driver %s, package %s): %s%s" % (
            g.index, when, (e.get("fields") or {}).get("driver"), (e.get("fields") or {}).get("package"),
            widecheck.widths_text(int(r.get("mask", 0))), " (%s)" % r.get("status") if r.get("status") != "measured" else ""))
    return out


def _prefill_info_lines(found) -> List[str]:
    """The setting of the prompt reading and, per GPU, what it chooses there (the line printed at start); the setting of the edit copy."""
    try:
        mode = gpus.prefill_mode_of(None)
        out = ["prefill  prompt reading: setting %s (YASAKANI_PREFILL / --prefill)" % mode]
        out += ["         GPU %d: %s" % (g.index, gpus.prefill_choice(mode, g)[2]) for g in found]
    except YasakaniError as ex:
        out = ["prefill  " + str(ex)]
    try:
        out.append("edit     edit copy: setting %s (YASAKANI_EDIT_COPY / --edit-copy)" % gpus.edit_copy_of(None))
    except YasakaniError as ex:
        out.append("edit     " + str(ex))
    return out


# ---- verify -----------------------------------------------------------------------------------------------------------------------
def cmd_verify(args) -> int:
    pack = ModelPack.load(_model_dir(args))
    if not pack.sha256:
        print("no checksums are listed for this folder (yasakani.json missing or without \"sha256\").")
        return 2
    print("checking %d files (reads about %s, one to two minutes) ..." % (len(pack.sha256), _human(sum((pack.root / n).stat().st_size for n in pack.sha256 if (pack.root / n).is_file()))))
    res = pack.verify(progress=lambda n, ok: print("  %-30s %s" % (n, "OK" if ok else "DIFFERENT"), flush=True))
    bad = [n for n, ok in res.items() if not ok]
    print("all files match." if not bad else "%d file(s) differ: %s  (download them again)" % (len(bad), ", ".join(bad)))
    return 0 if not bad else 1


# ---- serve ------------------------------------------------------------------------------------------------------------------------
def _stop_on_signals() -> None:
    """Ctrl+C, Ctrl+Break and a termination request all end the server the same orderly way."""
    def raise_interrupt(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, raise_interrupt)
            except (ValueError, OSError):
                pass


def cmd_serve(args) -> int:
    from .server import ApiState, ServerConfig, bind
    local = args.host in ("127.0.0.1", "localhost", "::1")
    if not local and not args.api_key:
        raise YasakaniError("refusing to listen on %s without --api-key: anyone who can reach this computer could use the GPU" % args.host)
    n = _parallel_count(getattr(args, "parallel", None), env=True)
    if getattr(args, "max_queue", DEFAULT_MAX_QUEUE) < 1:
        raise YasakaniError("--max-queue must be 1 or more")
    _turn_settings(args)                         # a wrong value stops here, before the model loads
    cfg = ServerConfig(host=args.host, port=args.port, api_key=args.api_key, default_max_tokens=args.default_max_tokens,
                       max_tokens_cap=args.max_tokens_cap)
    state = ApiState(cfg)
    state.version = __version__
    try:
        httpd = bind(state)
    except OSError as ex:
        raise YasakaniError("cannot listen on %s:%d (%s).  Is another server already using this port?  Use --port to pick another."
                            % (args.host, args.port, ex.strerror or type(ex).__name__))
    cfg.port = httpd.server_address[1]
    _stop_on_signals()
    threading.Thread(target=httpd.serve_forever, kwargs=dict(poll_interval=0.2), name="yasakani-http", daemon=True).start()
    base = "http://%s:%d" % ("[%s]" % args.host if ":" in args.host else args.host, cfg.port)
    _say("listening on %s  (answers /health while the model loads)" % base)
    eng = sched = None
    try:
        if n >= 2:
            pack, eng, fmt, runner = _open_stack(args, warm=not args.no_warmup, parallel=n)
            sched = _start_parallel(args, pack, eng, fmt)
            state.sched = sched
        else:
            pack, eng, fmt, runner = _open_stack(args, warm=not args.no_warmup)
        cfg.model_id = pack.model_id
        state.runner = runner
        _say("ready: OpenAI-compatible API at %s/v1   (model id %r%s%s)" % (base, cfg.model_id, ", API key required" if cfg.api_key else "",
                                                                         ", image input on" if runner.vision is not None else ""))
        if sched is not None:
            _say("serving up to %d requests at once (experimental), each with a context of %d tokens; up to %d more wait"
                 % (sched.n_slots, eng.max_ctx, sched.max_queue))
            _say(_turn_line(sched))
        if runner.vision is not None:
            _say(runner.vision.summary())
        _say("stop with Ctrl+C")
        while True:
            time.sleep(0.5)
            if sched is not None and sched.error is not None:     # every request got its error; the server does not wait on
                time.sleep(1.0)
                _say("error: the engine failed (%s); the server stops" % sched.error)
                return 1
    except KeyboardInterrupt:
        _say("stopping ...")
        return 0
    finally:
        runner_ = state.runner
        state.runner = None
        httpd.shutdown()
        if sched is not None:
            sched.close()
        if eng is not None:
            eng.cancel()
            with state.gen_lock:
                if runner_ is not None:
                    runner_.close()
                eng.close()
        httpd.server_close()


# ---- chat -------------------------------------------------------------------------------------------------------------------------
CHAT_HELP = """commands:  /exit  quit      /reset  forget the conversation      /think on|off  thinking on or off (default on)
           /effort low|medium|xhigh  how long the model thinks      /help  this text      Ctrl+C while it writes: stop the answer"""


def cmd_chat(args) -> int:
    from .chat import ChatParams, RequestError
    pack, eng, fmt, runner = _open_stack(args, warm=True)
    color = _use_color()
    dim = (lambda s: "\x1b[2m%s\x1b[0m" % s) if color else (lambda s: s)
    kw = {}
    if args.no_think:
        kw["enable_thinking"] = False
    if args.effort:
        kw["reasoning_effort"] = args.effort
    base: List[dict] = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)
    print("YASAKANI chat - type /help for commands, /exit to quit.")
    try:
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, rest = line[1:].partition(" ")
                rest = rest.strip().lower()
                if cmd in ("exit", "quit", "q"):
                    return 0
                if cmd == "reset":
                    messages = list(base)
                    print("(conversation cleared)")
                elif cmd == "think" and rest in ("on", "off"):
                    kw["enable_thinking"] = rest == "on"
                    print("(thinking %s)" % rest)
                elif cmd == "effort" and rest in ("low", "medium", "xhigh"):
                    kw["reasoning_effort"] = rest
                    print("(reasoning effort %s)" % rest)
                else:
                    print(CHAT_HELP)
                continue
            messages.append({"role": "user", "content": line})
            params = ChatParams(messages=[dict(m) for m in messages], max_tokens=args.max_tokens, temperature=args.temperature,
                                top_p=args.top_p, top_k=args.top_k, seed=args.seed, loop_guard=not args.no_loop_guard,
                                template_kwargs=dict(kw))
            parts = {"reasoning": [], "content": []}
            last = [None]

            def emit(kind: str, text: str) -> None:
                parts[kind].append(text)
                if kind == "reasoning" and args.hide_thinking:
                    return
                if last[0] != kind:
                    if kind == "reasoning":
                        print(dim("[thinking]"), flush=True)
                    elif last[0] == "reasoning":
                        print(dim("\n[/thinking]\n"), flush=True)
                    last[0] = kind
                print(dim(text) if kind == "reasoning" else text, end="", flush=True)

            try:
                u = runner.run(params, emit)
            except KeyboardInterrupt:
                print("\n[stopped]")
                messages.pop()
                continue
            except RequestError as ex:
                print("cannot do that: %s" % ex)
                messages.pop()
                continue
            except YasakaniError as ex:
                print("\nengine error: %s" % ex)
                messages.pop()
                try:
                    eng.reset()
                except YasakaniError:
                    return 1
                continue
            if parts["reasoning"] and not parts["content"] and u.finish != "stop":
                print(dim("\n(the limit of %d tokens was reached while the model was still thinking, so no answer was written.  "
                          "/effort medium or /effort low shortens the thinking, /think off skips it, --max-tokens raises the limit)"
                          % args.max_tokens))
            tps = (u.completion_tokens / (u.gen_ms / 1000.0)) if u.gen_ms > 0 else 0.0
            extra = ""
            if u.loop_guard:
                extra = ", loop guard cut the thinking"
            print(dim("\n\n[%d tokens, %.0f tokens/s, %s%s]" % (u.completion_tokens, tps, "complete" if u.finish == "stop" else "hit the length limit", extra)))
            msg = {"role": "assistant", "content": "".join(parts["content"]).strip()}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            messages.append(msg)
    finally:
        eng.close()


# ---- bench ------------------------------------------------------------------------------------------------------------------------
BUILTIN_PROMPTS = [
    ("explain", "Explain how a hash table works, including collisions and resizing, with a short example."),
    ("code", "Write a Python function that merges overlapping intervals and add a few tests for it."),
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("proof", "Prove that the square root of 2 is irrational."),
]


def _load_prompts(path: Optional[str]):
    if not path:
        return list(BUILTIN_PROMPTS)
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [ln for ln in text.splitlines() if ln.strip()]
    out = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            out.append(("test%d" % (i + 1), item))
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            out.append((str(item.get("name") or "test%d" % (i + 1)), item["prompt"]))
        else:
            raise YasakaniError("%s: entry %d is neither a string nor {\"name\":..., \"prompt\":...}" % (path, i + 1))
    if not out:
        raise YasakaniError("%s contains no prompts" % path)
    return out


def _bench_params(args, text):
    from .chat import ChatParams
    kw = {} if args.think else {"enable_thinking": False}
    return ChatParams(messages=[{"role": "user", "content": text}], max_tokens=args.n, temperature=0.0, ignore_eos=True,
                      loop_guard=False, template_kwargs=dict(kw))


def _bench_concurrent(args, n: int) -> int:
    """bench --concurrency N: the prompts x --runs as requests, N at a time on the scheduler of serve --parallel N."""
    from .sched import ScheduledEngine
    prompts = _load_prompts(args.prompts_file)
    pack, eng, fmt, runner = _open_stack(args, warm=True, parallel=n)
    sched = None
    jobs = [(name, text, run) for name, text in prompts for run in range(args.runs)]
    rows: List[Optional[dict]] = [None] * len(jobs)
    errors: List[BaseException] = []
    lock = threading.Lock()
    todo = list(range(len(jobs)))
    try:
        sched = _start_parallel(args, pack, eng, fmt)
        if sched is None:
            raise YasakaniError("--concurrency %d cannot run here (see the note above)" % n)
        _say("GPU: " + "; ".join(_gpu_lines()))

        def worker():
            while True:
                with lock:
                    if not todo or errors:
                        return
                    i = todo.pop(0)
                name, text, run = jobs[i]
                t0 = time.perf_counter()
                try:
                    with ScheduledEngine(sched) as se:
                        u = runner.run(_bench_params(args, text), lambda kind, t: None, eng=se)
                except BaseException as ex:          # noqa: BLE001  (reported on the main thread)
                    with lock:
                        errors.append(ex)
                    return
                sec = time.perf_counter() - t0
                rows[i] = dict(test=name, run=run + 1, prompt_tokens=u.prompt_tokens, tokens=u.completion_tokens, seconds=round(sec, 3),
                               tokens_per_s=round(u.completion_tokens / sec, 1) if sec > 0 else 0.0,
                               waited_s=round((u.queue_ms or 0.0) / 1000.0, 3), streams_per_pass=round(u.parallel_avg or 1.0, 2))
                if not args.json:
                    r = rows[i]
                    print("%-12s run %d  prompt %5d  generated %5d  %7.1f tokens/s  (%.1f s, waited %.1f s, %.1f streams per pass)" %
                          (name, r["run"], r["prompt_tokens"], r["tokens"], r["tokens_per_s"], r["seconds"], r["waited_s"],
                           r["streams_per_pass"]), flush=True)
        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, name="yasakani-bench-%d" % k, daemon=True) for k in range(min(n, len(jobs)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        if errors:
            raise errors[0]
    finally:
        if sched is not None:
            sched.close()
        eng.close()
    total = sum(r["tokens"] for r in rows)
    each = sum(r["tokens_per_s"] for r in rows) / len(rows)
    if args.json:
        print(json.dumps(dict(version=__version__, model=pack.model_id, gpu=_gpu_lines(), concurrency=sched.n_slots, tokens_per_test=args.n,
                              runs=rows, seconds=round(wall, 3), total_tokens=total, total_tokens_per_s=round(total / wall, 1),
                              mean_tokens_per_s=round(each, 1)), indent=1))
    else:
        print("%d requests, %d at a time: %d tokens in %.1f s = %.1f tokens/s in total, %.1f tokens/s per request on average   "
              "(greedy, thinking %s, %d tokens each; the times are measured here, from sending a request to its last token)"
              % (len(rows), sched.n_slots, total, wall, total / wall, each, "on" if args.think else "off", args.n))
    return 0


def cmd_bench(args) -> int:
    from .chat import ChatParams
    if getattr(args, "edit", False):
        if _parallel_count(getattr(args, "concurrency", None)) >= 2:
            raise YasakaniError("--edit runs one request at a time; leave out --concurrency")
        return _bench_edit(args)
    n_tokens = 512 if args.n is None else args.n
    n = _parallel_count(getattr(args, "concurrency", None))
    if n >= 2:
        args.n = n_tokens
        return _bench_concurrent(args, n)
    prompts = _load_prompts(args.prompts_file)
    pack, eng, fmt, runner = _open_stack(args, warm=True)
    rows = []
    try:
        _say("GPU: " + "; ".join(_gpu_lines()))
        kw = {} if args.think else {"enable_thinking": False}
        for name, text in prompts:
            for run in range(args.runs):
                eng.reset()
                p = ChatParams(messages=[{"role": "user", "content": text}], max_tokens=n_tokens, temperature=0.0, ignore_eos=True,
                               loop_guard=False, template_kwargs=dict(kw))
                u = runner.run(p, lambda kind, t: None)
                tps = u.completion_tokens / (u.gen_ms / 1000.0) if u.gen_ms > 0 else 0.0
                rows.append(dict(test=name, run=run + 1, prompt_tokens=u.prompt_tokens, tokens=u.completion_tokens, seconds=round(u.gen_ms / 1000.0, 3),
                                 tokens_per_s=round(tps, 1), tokens_per_pass=round(u.completion_tokens / max(1, u.passes), 2)))
                if not args.json:
                    r = rows[-1]
                    print("%-12s run %d  prompt %5d  generated %5d  %7.1f tokens/s  %.2f tokens/pass" %
                          (name, r["run"], r["prompt_tokens"], r["tokens"], r["tokens_per_s"], r["tokens_per_pass"]), flush=True)
    finally:
        eng.close()
    mean = sum(r["tokens_per_s"] for r in rows) / len(rows)
    if args.json:
        print(json.dumps(dict(version=__version__, model=pack.model_id, gpu=_gpu_lines(), tokens_per_test=n_tokens, runs=rows,
                              mean_tokens_per_s=round(mean, 1)), indent=1))
    else:
        print("mean over %d run(s): %.1f tokens/s   (greedy, thinking %s, %d tokens each)" % (len(rows), mean, "on" if args.think else "off", n_tokens))
    return 0


def _bench_edit(args) -> int:
    """0.1.6: ``bench --edit`` runs the three edit tasks of the speed tables (yasakani/editbench.py).  Exit code 1 when a task did not
    fit the context and was skipped."""
    from . import editbench
    if args.prompts_file or args.n is not None:
        raise YasakaniError("--prompts-file and --n do not apply to --edit: each edit task has its own prompt and output limit")
    tasks = editbench.load_tasks()                     # before the long load
    pack, eng, fmt, runner = _open_stack(args, warm=True)
    try:
        det = getattr(args, "gpu_detected", None)
        head = editbench.header(__version__, pack, eng, det.gpu if det is not None else None, _gpu_lines(), args.think, args.runs,
                                seed0=args.seed0)
        if not args.json:
            for line in editbench.header_lines(head):
                print(line, flush=True)

        def skip(task, need):
            if not args.json:
                print("%-17s skipped: needs a context of %d tokens, the engine has %d (--max-ctx)" % (task.title, need, eng.max_ctx), flush=True)

        rows, skipped = editbench.run(runner, tasks, think=args.think, runs=args.runs, on_skip=skip,
                                      on_row=None if args.json else (lambda r: print(editbench.row_line(r), flush=True)))
    finally:
        eng.close()
    mean = editbench.means(rows)
    if args.json:
        print(json.dumps(dict(head, runs=rows, skipped=skipped, mean_tokens_per_s=mean), indent=1))
    else:
        if args.runs > 1 and rows:
            print("mean tok/s: " + ", ".join("%s %.1f" % (t.title, mean[t.name]) for t in tasks if t.name in mean))
        print(editbench.footer(args.think))
    return 1 if skipped else 0


# ---- selfcheck --------------------------------------------------------------------------------------------------------------------
SELFCHECK_OTHER_GPU = ("the output differs from the reference for %d case(s).  The reference was made on an RTX 5090, and this GPU is not one: "
                       "on other GPUs tiny differences in the computation can make the output take another word part-way through.  So this "
                       "GPU now reads the reference output (nothing is generated), and its predictions are compared with the RTX 5090's at "
                       "the same positions:")
SELFCHECK_OTHER_SAME = ("this GPU is not an RTX 5090, on which the reference was made.  So its predictions are also compared with the RTX "
                        "5090's: it reads the reference output (nothing is generated) and they are compared at the same positions:")
SELFCHECK_PASSED = "self-check passed: this installation reproduces the reference output."
SELFCHECK_CLOSE_PASS = ("self-check passed within the tolerances: this GPU makes the same predictions as the RTX 5090, up to small "
                        "differences in the computation.  Exit code 3: only the generated output differs.")
SELFCHECK_CLOSE_FAIL = ("This is more than small differences in the computation explain.  Check the model files with yasakani verify.  When "
                        "they are fine, the engine probably does not compute correctly on this GPU: please report it on GitHub Issues with "
                        "the output of yasakani info and of this self-check.")
SELFCHECK_CONSISTENCY = ("self-consistency (every GPU, no reference): the answer of a prompt that copies its input is generated and then read "
                         "again by this GPU; generated and predicted ids must be the same, and passes of 64 positions must give the same "
                         "predictions as passes of 16, bit for bit:")
SELFCHECK_CONSISTENCY_FAIL = ("The engine computes these passes with the same sums in the same order, so they must agree exactly on any card.  "
                              "Check the model files with yasakani verify.  When they are fine, the engine probably does not compute "
                              "correctly on this GPU: please report it on GitHub Issues with the output of yasakani info and of this "
                              "self-check.")
SELFCHECK_WRITE_OTHER = ("note: this GPU is not known to be an RTX 5090.  The file will hold this GPU's output and predictions, and the "
                         "self-check on other GPUs will compare theirs with them as if they were the RTX 5090's.")


def _selfcheck_consistency(eng, fmt, rc: int) -> int:
    """0.1.4: the self-consistency check of every GPU (yasakani/selfcheck.py), after the comparison with the reference passed with
    exit code ``rc`` (0 or 3)."""
    from . import selfcheck as sc
    if not eng.supports_forced:
        print("note: the engine library cannot read a given text (no yskn_verify_nll export), so the self-consistency check is skipped.")
        return rc
    print(SELFCHECK_CONSISTENCY)
    res = sc.consistency(eng, fmt)
    for line in sc.consistency_lines(res):
        print(line)
    fails = res.failures()
    if fails:
        print("self-check FAILED: %s." % "; ".join(fails))
        print(SELFCHECK_CONSISTENCY_FAIL)
        return 1
    print("self-consistency passed.")
    return rc


SELFCHECK_PARALLEL = ("several requests at once (--parallel %d): short requests in the stream slots must give the ids each gives alone, "
                      "exactly:")


def _selfcheck_parallel(args, pack, eng, fmt, rc: int) -> int:
    """selfcheck --parallel N, after the checks of one request passed with exit code ``rc``: the start-up check of serve --parallel N,
    run now (its result is kept for serve).  Exit code 1 when the output differs, 2 when it cannot run here."""
    print(SELFCHECK_PARALLEL % getattr(args, "parallel", 0))
    sched = _start_parallel(args, pack, eng, fmt, check=False)
    if sched is None:
        print("  NOT RUN: the engine cannot serve several requests at once here (see the note above).")
        return 2
    try:
        ok = _parallel_check(args, pack, eng, fmt, sched, use_cache=False)
    finally:
        sched.close()
    if ok is None:
        print("  NOT RUN: the check could not run (see the note above).")
        return 2
    if not ok:
        print("self-check FAILED: %d requests at once gave other output than one at a time.  Please report it on GitHub Issues with the "
              "output of yasakani info and of this self-check." % sched.n_slots)
        return 1
    print("  PASS  (%d requests at once)" % sched.n_slots)
    return rc


def cmd_selfcheck(args) -> int:
    n = _parallel_count(getattr(args, "parallel", None))
    if n >= 2:
        pack, eng, fmt, runner = _open_stack(args, warm=True, parallel=n, allow_yarn2=False)
    else:
        pack, eng, fmt, runner = _open_stack(args, warm=True, allow_yarn2=False)
    try:
        rc = _selfcheck_one(args, pack, eng, fmt)
        if n >= 2 and not args.write and rc in (0, 3):
            rc = _selfcheck_parallel(args, pack, eng, fmt, rc)
        return rc
    finally:
        eng.close()


def _selfcheck_one(args, pack, eng, fmt) -> int:
    """The checks of one request (as up to 0.1.6)."""
    from . import selfcheck as sc
    got = sc.run(eng, fmt)
    if args.write:
        det = getattr(args, "gpu_detected", None)
        if det is None or not det.tested:
            print(SELFCHECK_WRITE_OTHER)
        rec = sc.reference_record(eng, got, pack.model_id, __version__, gpu=det.gpu.name if det is not None and det.gpu else None)
        Path(args.write).write_text(sc.dumps(rec) + "\n", encoding="utf-8")
        print("wrote %d reference record(s) to %s" % (len(got), args.write))
        return 0
    ref_path = Path(args.reference) if args.reference else pack.root / "selfcheck.json"
    if not ref_path.is_file():
        print("no reference file (%s).  Download selfcheck.json next to the model files, or pass --reference FILE." % ref_path)
        return 2
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    if ref.get("format") != SELFCHECK_FORMAT:
        print("the reference file has an unknown format (%r)" % ref.get("format"))
        return 2
    det = getattr(args, "gpu_detected", None)
    other = det is not None and det.untested           # a GPU that is known and is not the RTX 5090
    want = {c["name"]: c for c in ref.get("cases", [])}
    bad = differs = 0
    for g in got:
        w = want.get(g["name"])
        if w is None:
            print("  %-14s not in the reference file" % g["name"])
            bad += 1
        elif w["prompt_sha256"] != g["prompt_sha256"]:
            print("  %-14s FAIL  the prompt ids differ (tokenizer or chat template files are not the reference ones)" % g["name"])
            bad += 1
        elif w["sha256"] != g["sha256"] or w["tokens"] != g["tokens"]:
            n = next((i for i, (a, b) in enumerate(zip(w["head"], g["head"])) if a != b), None)
            print("  %-14s %s  the output differs%s" % (g["name"], "DIFF" if other else "FAIL",
                                                        "" if n is None else " (first different token: #%d)" % (n + 1)))
            differs += 1
        else:
            print("  %-14s PASS  (%d tokens)" % (g["name"], g["tokens"]))
    if bad or (differs and not other):
        print("self-check FAILED for %d case(s)." % (bad + differs))
        return 1
    if not other or (not differs and (len(sc.full_data(ref)) < len(got) or not eng.supports_forced)):
        print(SELFCHECK_PASSED)                         # the RTX 5090 (or an unknown GPU): the same output; as before 0.1.4
        return _selfcheck_consistency(eng, fmt, 0)
    # 0.1.4: on a GPU other than the RTX 5090 the predictions on the reference output decide (teacher forcing, yasakani/selfcheck.py)
    print(SELFCHECK_OTHER_GPU % differs if differs else SELFCHECK_OTHER_SAME)
    res = sc.close_check(eng, got, ref)
    for line in sc.close_lines(res):
        print(line)
    fails = res.failures()
    if not fails:
        print(SELFCHECK_CLOSE_PASS if differs else SELFCHECK_PASSED)
        return _selfcheck_consistency(eng, fmt, 3 if differs else 0)
    print("self-check FAILED: %s." % "; ".join(fails))
    print(SELFCHECK_CLOSE_FAIL)
    return 1


# ---- entry point ------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="yasakani", description="Run Qwen3.8-27B locally on one NVIDIA RTX GPU with the YASAKANI engine "
                                                           "(tested on the RTX 5090 under Windows).")
    ap.add_argument("--version", action="version", version="yasakani " + __version__ + (" (%s)" % BUILD_NOTE if BUILD_NOTE else ""))
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{info,serve,chat,bench,selfcheck,verify,update,report}")

    sp = sub.add_parser("info", help="check this computer, the engine library and (with --model-dir) a model folder")
    sp.add_argument("--model-dir")
    sp.add_argument("--dll")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("verify", help="check the model files against the checksums listed in yasakani.json")
    sp.add_argument("--model-dir")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("serve", help="start the OpenAI-compatible server")
    _add_engine_args(sp)
    _add_yarn2_args(sp)
    sp.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1: this computer only)")
    sp.add_argument("--port", type=int, default=DEFAULT_PORT)
    sp.add_argument("--api-key", default=os.environ.get("YASAKANI_API_KEY"), help="require this key (or set YASAKANI_API_KEY); needed for any non-local --host")
    sp.add_argument("--default-max-tokens", type=int, default=32768, help="answer limit when a request does not give one")
    sp.add_argument("--max-tokens-cap", type=int, default=None, help="upper limit for any request")
    sp.add_argument("--no-warmup", action="store_true", help="skip the short warm-up run at start")
    sp.add_argument("--parallel", type=int, default=None, metavar="N",
                    help="experimental: generate up to N requests at once (default 1, or $YASAKANI_PARALLEL).  Every request has a "
                         "context of its own, all of the same length, chosen from the free GPU memory unless --max-ctx gives it; at "
                         "the first start with an engine library, GPU and driver, a short check compares N requests at once with one "
                         "at a time")
    sp.add_argument("--max-queue", type=int, default=DEFAULT_MAX_QUEUE, metavar="Q",
                    help="with --parallel: requests that may wait for a free stream (default %d); more are answered 503 with "
                         "Retry-After" % DEFAULT_MAX_QUEUE)
    sp.add_argument("--keep-turn", default=None, metavar="auto|off|SECONDS",
                    help="with --parallel: after an answer with tool calls, how long the request's place (its stream) is kept for "
                         "the next turn with the tool results, which then continues it instead of reading the conversation again. "
                         "auto (the default, or $YASAKANI_KEEP_TURN): about as long as reading the conversation again takes, "
                         "from 1 to 30 seconds; off: not kept")
    sp.add_argument("--max-wait", default=None, metavar="auto|SECONDS",
                    help="with --parallel: the longest a new request waits for next turns (places kept that way, and next turns "
                         "that came after it and are read first); then it takes the next kept place, and that conversation's next turn "
                         "reads it again.  auto (the default, or $YASAKANI_MAX_WAIT): as long as reading the kept conversation again "
                         "takes, and 0.25 seconds.  Only the time in which next turns alone stood in its way counts")
    _add_parallel_hidden(sp)
    _add_vision_args(sp)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("chat", help="talk to the model in the terminal")
    _add_engine_args(sp)
    _add_yarn2_args(sp)
    sp.add_argument("--system", help="system prompt")
    sp.add_argument("--max-tokens", type=int, default=32768)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--top-p", type=float, default=0.95)
    sp.add_argument("--top-k", type=int, default=20)
    sp.add_argument("--seed", type=int, default=None, help="fixed seed (same seed and prompt give the same answer)")
    sp.add_argument("--no-think", action="store_true", help="answer without a thinking phase")
    sp.add_argument("--effort", choices=("low", "medium", "xhigh"), help="how long the model thinks (default xhigh)")
    sp.add_argument("--hide-thinking", action="store_true", help="do not print the thinking")
    sp.add_argument("--no-loop-guard", action="store_true")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("selfcheck", help="run a few fixed greedy prompts and compare the output with the reference file "
                                          "(exit code 0 same, 1 different, 2 no reference, 3 on a GPU other than the RTX 5090: the output "
                                          "differs, but the GPU's predictions on the reference output are within the tolerances; on "
                                          "such a GPU these predictions are checked also when the output is the same).  Every GPU then "
                                          "also checks that its own predictions agree with what it generated (exit code 1 if not)")
    _add_engine_args(sp)
    sp.add_argument("--reference", help="reference file (default: selfcheck.json in the model folder)")
    sp.add_argument("--write", metavar="FILE", help="write a new reference file instead of checking (for people who publish a model folder)")
    sp.add_argument("--parallel", type=int, default=None, metavar="N",
                    help="experimental: then also check that N requests at once (serve --parallel N) give the same output as one at "
                         "a time (exit code 1 if not, 2 if this engine cannot serve several requests at once)")
    _add_parallel_hidden(sp)
    sp.set_defaults(func=cmd_selfcheck)

    sp = sub.add_parser("bench", help="measure generation speed (greedy; fixed length, or with --edit the edit tasks of the speed tables; "
                                      "--edit --think samples)")
    _add_engine_args(sp)
    _add_yarn2_args(sp)
    sp.add_argument("--edit", action="store_true",
                    help="run the 3 edit tasks of the speed tables in the README (code edit, JSONL edit, long JSONL edit) until the model "
                         "ends its answer; prints tokens, seconds, tok/s, tokens per pass, wide copy passes and whether each answer is "
                         "the expected one")
    sp.add_argument("--prompts-file", help="JSON list of prompts (strings or {\"name\", \"prompt\"}) or a text file with one prompt per line")
    sp.add_argument("--n", type=int, default=None, help="tokens to generate per test (default 512; not with --edit)")
    sp.add_argument("--runs", type=int, default=1, help="repeats per test")
    sp.add_argument("--think", action="store_true",
                    help="let the model think first (default: answer directly).  With --edit: temperature 1.0 (seeds from --seed0) and "
                         "an output limit of 32768 tokens for every task, so tok/s varies from run to run")
    sp.add_argument("--json", action="store_true", help="print the result as JSON")
    sp.add_argument("--concurrency", type=int, default=None, metavar="N",
                    help="experimental: send the requests (the prompts x --runs) N at a time, as to serve --parallel N, and report "
                         "the tokens per second of all of them together")
    _add_parallel_hidden(sp)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("update", help="look for a newer release on GitHub (the only command that contacts the network).  In a folder "
                                       "set up by the quickstart its launcher does the update; otherwise the steps are printed")
    sp.add_argument("--check", action="store_true", help="only print the installed and the latest version")
    sp.add_argument("--yes", action="store_true", help="in a quickstart folder: update without asking")
    sp.set_defaults(func=cmd_update)
    sp = sub.add_parser("report", help="check the wide copy passes on this GPU and print a report to paste on GitHub "
                                       "(GPU, driver, versions, the check, the edit tasks with wide passes off / checked / all, the "
                                       "self-check); about 1.5 minutes on the RTX 5090, longer on other cards")
    sp.add_argument("--model-dir", help="folder with the model files (default: $YASAKANI_MODEL_DIR)")
    sp.add_argument("--dll", help="path of yskn.dll (default: $YASAKANI_DLL, or the bin folder next to the package)")
    sp.add_argument("--max-ctx", type=int, default=None, help="context length in tokens (default: as for serve)")
    sp.add_argument("--device", type=int, default=0, help="the GPU to use, numbered as in `yasakani info` and nvidia-smi (default 0)")
    sp.add_argument("--rounds", type=int, default=1, help="times each column of the edit table is run, in the order off, checked, all, "
                                                         "all, checked, off (default 1: two runs per column)")
    sp.add_argument("--no-bench", action="store_true", help="leave out the edit tasks (the check and the self-check only)")
    sp.add_argument("--no-selfcheck", action="store_true", help="leave out the self-check")
    sp.add_argument("--json", action="store_true", help="print the result as JSON instead of Markdown")
    sp.add_argument("--out", metavar="FILE", help="also write the report to FILE")
    sp.set_defaults(func=cmd_report, seed0=1, wide="on", wide_recheck=True)
    return ap


def cmd_update(args) -> int:
    from .update import cmd_update as run
    return run(args)


def cmd_report(args) -> int:
    from .report import run_report
    return run_report(args)


def _add_parallel_hidden(sp: argparse.ArgumentParser) -> None:
    """Options for tests of several requests at once (not in the help)."""
    sp.add_argument("--parallel-rows", type=int, default=16, help=argparse.SUPPRESS)        # rows of one pass at most (1..16)
    sp.add_argument("--no-slot-reuse", action="store_true", help=argparse.SUPPRESS)         # finished conversations do not keep their KV
    sp.add_argument("--parallel-check", choices=("auto", "always", "off"), default="auto", help=argparse.SUPPRESS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _fix_console()
    pci_bus_order()                              # 0.1.5: --device N is nvidia-smi's GPU N; set before anything can start CUDA
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except YasakaniError as ex:
        _say("error: %s" % ex)
        return 1
    except KeyboardInterrupt:
        return 130
