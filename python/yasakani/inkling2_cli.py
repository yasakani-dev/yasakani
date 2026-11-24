# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The commands info, verify, selfcheck, chat, serve and bench for an Inkling-2 model folder (yasakani/inkling2.py).  yasakani/cli.py
sends a command here when the folder's yasakani.json says ``"architecture": "inkling2"``; the Qwen3.8 path is not changed."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import List

from . import __version__
from . import inkling2 as ink
from ._native import YasakaniError

QWEN_VISION = ("--mmproj, --llama-dir and --vision-model are for Qwen3.8 model folders; Inkling-2 Mini reads images with the image "
               "tower of its own folder (--vision auto|on|off)")
QWEN_VISION_ENV = dict(mmproj="YASAKANI_MMPROJ", llama_dir="YASAKANI_LLAMA_DIR", vision_model="YASAKANI_VISION_MODEL")
UNSUPPORTED = {
    "parallel": "--parallel (several requests at once) is not supported for Inkling-2 models yet",
    "concurrency": "--concurrency is not supported for Inkling-2 models yet",
    "edit": "--edit (the edit tasks of the speed tables) is not supported for Inkling-2 models yet",
    "mmproj": QWEN_VISION,
    "llama_dir": QWEN_VISION,
    "vision_model": QWEN_VISION,
    "vision_threads": "--vision-threads is for the CPU image encoder of Qwen3.8 model folders; Inkling-2 Mini encodes images on the GPU",
    "vision_cache": "--vision-cache (a folder of image features) is for Qwen3.8 model folders; Inkling-2 Mini keeps the image vectors "
                    "of recent images in memory",
    "image": "--image is taken by yasakani chat only (serve reads the images inside the requests)",
    "write": "selfcheck --write is not supported for Inkling-2 models yet",
    "suppress_tokens": "--suppress-tokens is not supported for Inkling-2 models yet (it masks Qwen3.8 token IDs)",
    "yarn2": "--yarn2 is not supported for Inkling-2 models (it scales the rotary table of Qwen3.8 models)",
}


def _say(*a) -> None:
    from .cli import _say as say
    say(*a)


def _gb(n: float) -> str:
    return "%.2f GB" % (n / 1e9)


def refuse_unsupported(args, images: bool = False) -> None:
    """A YasakaniError for an option an Inkling-2 folder does not take.  ``images``: the command takes images (chat)."""
    for key, why in UNSUPPORTED.items():
        if key == "image" and images:
            continue
        v = getattr(args, key, None)
        env = QWEN_VISION_ENV.get(key)
        if env and v and v == os.environ.get(env):          # set for Qwen3.8 folders in the environment, not on this command line
            if not getattr(args, "qwen_vision_env_noted", False):
                args.qwen_vision_env_noted = True
                _say("note: ignoring %s: the Qwen3.8 image encoder is not used for Inkling-2 models" % env)
            continue
        if (v not in (None, 1)) if key in ("parallel", "concurrency") else bool(v):
            raise YasakaniError(why)
    if (os.environ.get("YASAKANI_SUPPRESS_TOKENS") or "").strip() and not getattr(args, "suppress_env_noted", False):
        args.suppress_env_noted = True           # serve and bench come here twice
        _say("note: ignoring YASAKANI_SUPPRESS_TOKENS: output token suppression is not supported for Inkling-2 models yet")


def context_of(args, pack: ink.Ink2Pack) -> int:
    """--max-ctx when it is given (256 to the model's limit), else (0.3.2) the folder's default when it fits the free GPU memory, or
    the longest context that fits (auto_context).  The choice is made once per command (serve asks before and after the load)."""
    n = getattr(args, "max_ctx", None)
    if n:
        if n < 256 or n > pack.max_ctx:
            raise YasakaniError("--max-ctx must be between 256 and %d (the model's limit)" % pack.max_ctx)
        return int(n)
    if getattr(args, "ink2_auto_ctx", None) is None:
        args.ink2_auto_ctx = auto_context(args, pack)
    return args.ink2_auto_ctx


def auto_context(args, pack: ink.Ink2Pack, det=None, say=None) -> int:
    """The context of the folder without --max-ctx, from nvidia-smi's free memory of the GPU the engine uses (yasakani/gpu.py
    ink2_context): the folder's default when it fits, else the longest multiple of 4,096 that fits, never more than the default.
    One line says what was chosen.  Raises YasakaniError when not even 8,192 positions fit; without an answer of nvidia-smi the
    folder's default is used."""
    from . import gpu as gpus
    say = say or _say
    default = int(pack.default_ctx)
    if det is None:
        det = gpus.detect(getattr(args, "device", 0) or 0)
    g = det.worst()
    if g is None or g.free_mib is None or g.total_mib is None:
        say("context: %d tokens (the folder's default; the free GPU memory could not be read); --max-ctx sets it yourself" % default)
        return default
    draft, _ = draft_options(args)
    drafter = draft != "off" and pack.drafter is not None
    args.ink2_compute_cap = g.compute_cap
    fast = ink.fast_expected(ink.fast_mode_of(getattr(args, "fast", None)), g.compute_cap)
    files = pack.weights.stat().st_size + (pack.drafter.stat().st_size if drafter else 0)
    per = gpus.ink2_bytes_per_position(ink.bytes_per_position(ink.read_config(pack.root), mxfp8=fast), drafter, fast=fast)
    rollback = ink.rollback_positions()
    vmib = vision_mib(args, pack)
    n = gpus.ink2_context(g.free_mib, files, per, default, rollback, fast=fast, vision_mib=vmib)
    if n <= 0:
        raise YasakaniError(
            "the model does not fit in the free GPU memory (%d of %d MiB free; with a context of %d tokens it needs about %d MiB).  "
            "Close other programs that use the GPU%s, or give a smaller --max-ctx to try anyway"
            % (g.free_mib, g.total_mib, gpus.INK2_MIN_CONTEXT,
               gpus.ink2_need_mib(gpus.INK2_MIN_CONTEXT, files, per, rollback, fast=fast, vision_mib=vmib),
               ", start with --draft off (the drafter needs memory too)" if drafter else "")
            + ("; --vision off gives back the %d MiB of the image tower" % round(vmib) if vmib else ""))
    if n == default:
        say("context: %d tokens (the folder's default; %d of %d MiB of GPU memory free); --max-ctx sets it yourself"
            % (n, g.free_mib, g.total_mib))
    else:
        say("context: %d tokens, chosen from the free GPU memory (%d of %d MiB free; the folder's default %d does not fit); "
            "--max-ctx sets it yourself" % (n, g.free_mib, g.total_mib, default))
    return n


def _select_gpu(args) -> None:
    """--device N on a PC with several GPUs: the engine sees only that GPU (as for Qwen models)."""
    if getattr(args, "device", 0) in (0, None):
        return
    from . import gpu as gpus
    from .cli import _visible_gpu
    _visible_gpu(gpus.detect(args.device), args.device)


def vision_options(args):
    """(mode, per-image cap, per-request limit) of the image options.  Commands without images (selfcheck, bench) run with "off";
    chat and serve take --vision (else $YASAKANI_INK2_VISION, else auto), --vision-max-image-tokens and
    --vision-max-request-image-tokens.  A wrong value fails here, before the long load."""
    if not getattr(args, "ink2_images", False):
        return "off", None, None
    mode = ink.vision_mode_of(getattr(args, "vision", None))
    cap, req = ink.vision_limits(getattr(args, "vision_max_image_tokens", None), getattr(args, "vision_max_request_image_tokens", None))
    return mode, cap, req


def vision_mib(args, pack: ink.Ink2Pack) -> float:
    """GPU memory (MiB) the image tower will take with these options (0: it will not be loaded), for planning the context."""
    mode, cap, req = vision_options(args)
    if mode == "off":
        return 0.0
    return ink.vision_plan_mib(pack, mode, cap, req, dll=getattr(args, "dll", None))


def draft_options(args):
    """(mode, depth) of --draft / --draft-depth, else $YASAKANI_DRAFT / $YASAKANI_DRAFT_DEPTH; a wrong value fails here."""
    return ink.draft_mode_of(getattr(args, "draft", None)), ink.draft_depth_of(getattr(args, "draft_depth", None))


def open_stack(args, warm: bool = True, images: bool = False, image_option: bool = False):
    """Folder -> (pack, engine, formatter, runner).  ``images``: the command reads images (chat, serve); the image tower is then
    loaded by the --vision rule and one line says whether image input is on (and why not); else (selfcheck, bench) it is not
    loaded.  ``image_option``: the command takes --image (chat)."""
    refuse_unsupported(args, image_option)
    args.ink2_images = images
    vmode, vcap, vreq = vision_options(args)                                    # wrong image options fail before the long load
    pack = ink.Ink2Pack.load(args.model_dir_path)
    if image_option and getattr(args, "image", None) and (vmode == "off" or not pack.has_vision):   # known before the long load
        from . import inkling2_vision as V
        raise YasakaniError("--image needs image input, which is off: %s" % (ink.VISION_OFF if vmode == "off" else V.NO_TOWER))
    ctx = context_of(args, pack)
    draft, depth = draft_options(args)
    fast = ink.fast_mode_of(getattr(args, "fast", None))
    cap = getattr(args, "ink2_compute_cap", None)                               # known when the context was planned
    if cap is None and fast == "auto":                                          # --max-ctx was given: ask nvidia-smi for this line,
        try:                                                                    # before --device N pins the engine's GPU (after
            from . import gpu as gpus                                           # that, index N no longer names it)
            g = gpus.detect(getattr(args, "device", 0) or 0).worst()
            cap = g.compute_cap if g is not None else None
        except Exception:
            cap = None
    _select_gpu(args)
    if cap is None:
        cap = ink.CAP_UNKNOWN                                                   # asked (or not needed): the engine does not ask again
    mxfp8 = ink.fast_expected(fast, cap)                                        # the same rule as the engine's (ink.fast_choice)
    kv = ink.bytes_per_position(ink.read_config(pack.root), mxfp8=mxfp8) * ctx
    _say("loading %s (%s of weights, context %d tokens, key/value cache %s) ..." % (pack.model_id, _gb(pack.weights.stat().st_size), ctx, _gb(kv)))
    t0 = time.time()
    try:
        fmt, eng, runner = ink.open_runner(pack, ctx, dll=getattr(args, "dll", None), seed0=getattr(args, "seed0", 1), warm=warm,
                                           draft=draft, draft_depth=depth, copy=getattr(args, "copy", None),
                                           copy_min=getattr(args, "copy_min", None), fast=fast, compute_cap=cap,
                                           vision=vmode, vision_max_image_tokens=vcap, vision_max_request_tokens=vreq)
    except YasakaniError as ex:
        raise YasakaniError("%s  (a smaller --max-ctx needs less GPU memory)" % ex) if "memory" in str(ex).lower() else ex
    dm = eng.device_memory()
    mem = "engine allocations %s" % _gb(eng.memory_used())
    if dm:
        mem += "; device %s used of %s" % (_gb(dm[1] - dm[0]), _gb(dm[1]))
    _say("model ready in %.1f s (engine load %.1f s, %s backend; %s)" % (time.time() - t0, eng.load_seconds, eng.backend, mem))
    if pack.weights_format == "yskn":
        _say("weights: %s" % eng.weights_note)
    _say(eng.fast_note)
    _say(eng.draft_note)
    _say(eng.copy_note)
    if images:
        _say(eng.vision_note)
        allow = getattr(args, "vision_allow_local_files", None)
        if allow:
            runner.vision_allow_local_files = str(Path(allow).resolve())
    return pack, eng, fmt, runner


# ---- info / verify ---------------------------------------------------------------------------------------------------------------
def cmd_info(args) -> int:
    from .cli import _gpu_lines, _human
    ok = True
    print("yasakani", __version__)
    print("python  ", platform.python_version(), platform.system(), platform.machine())
    for mod in ("numpy", "tokenizers", "jinja2"):
        try:
            m = importlib.import_module(mod)
            print("%-8s %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("%-8s MISSING  (pip install %s)" % (mod, mod))
            ok = False
    try:
        dll = ink.find_dll(getattr(args, "dll", None))
        binding = ink.find_binding(dll)
        mod = ink.load_binding(binding)
        lib = ink.open_library(mod, dll)
        lib_info = "%s, interface version %d" % (lib.ink2_build_info().decode("utf-8", "replace"), lib.abi)
        print("engine   %s  %s  sha256 %s  (%s; uses the %s backend)" % (dll.name, _human(dll.stat().st_size),
              hashlib.sha256(dll.read_bytes()).hexdigest()[:16], lib_info, ink.backend_of(lib_info)))
        print("binding  %s  (drafter calls: %s)" % (binding.name, "yes" if all(callable(getattr(mod.Ink2Native, n, None))
                                                                             for n in ink.DRAFT_CALLS) else "no, drafting not available"))
        caps = int(getattr(lib, "caps", 0) or 0)
        names = [n for bit, n in ((1, "packed dense matrices"), (2, "drafter"), (4, "converted drafter files"),
                                  (8, "3-bit drafter matrices"), (16, "8-bit key/value cache"),
                                  (32, "image input")) if caps & bit]
        print("         engine capabilities: %s" % (", ".join(names) if names else "none"))
    except (YasakaniError, OSError, AttributeError) as ex:
        print("engine   NOT FOUND:", ex)
        ok = False
    for ln in _gpu_lines():
        print("gpu      " + ln)
    try:
        pack = ink.Ink2Pack.load(args.model_dir_path)
        src = pack.meta.get("source") or {}
        print("model    %s  (id %s, architecture inkling2, context up to %d tokens, %d by default)"
              % (pack.root.name, pack.model_id, pack.max_ctx, pack.default_ctx))
        print("  source     %s @ %s, licence %s" % (src.get("repository"), str(src.get("revision"))[:12], src.get("license")))
        wmeta = pack.meta.get("weights") or {}
        print("  weights    %-36s %s  (%s%s)" % (pack.weights.name, _human(pack.weights.stat().st_size), pack.weights_format,
                                                ", version %s, scheme %s" % (wmeta.get("version", 2), wmeta.get("scheme"))
                                                if pack.weights_format == "yskn" else ""))
        d = pack.drafter_entry
        if not d:
            print("  drafter    none (generation without drafting)")
        else:
            print("  drafter    %-36s %s  (%s%s)%s" % (d["file"], _human((pack.root / d["file"]).stat().st_size) if pack.drafter else "-",
                                                   d.get("format", "hf-safetensors"),
                                                   ", version %s, scheme %s" % (d.get("version", 2), d.get("scheme"))
                                                   if pack.drafter_format == "yskn" else "",
                                                   "  PROBLEM: " + pack.drafter_problem if pack.drafter_problem else ""))
        print("  images     %s" % vision_line(pack))
        print("  sampling   temperature %g, top_p %g, top_k %d (generation_config.json); stop ids %s"
              % (pack.sampling["temperature"], pack.sampling["top_p"], pack.sampling["top_k"], pack.stop_ids))
        print("  checksums  %d files listed in yasakani.json" % len(pack.sha256))
        fmt = ink.Ink2Formatter(pack.root)
        ids, _ = fmt.prompt([{"role": "user", "content": "hello"}])
        print("  tokenizer  ok (chat template renders a %d-id prompt)" % len(ids))
        print("  selfcheck  %s" % ("%d case(s)" % len(ink.selfcheck_cases(pack)) if (pack.root / "selfcheck.json").is_file() else "no selfcheck.json"))
    except (YasakaniError, ImportError, OSError, ValueError) as ex:
        print("model    PROBLEM:", ex)
        ok = False
    return 0 if ok else 1


def vision_line(pack: ink.Ink2Pack) -> str:
    """What `yasakani info` says about the folder's image tower."""
    from .cli import _human
    if pack.vision_entry:
        f = pack.vision_file
        if f is None or not f.is_file():
            return "the image tower %s is listed in yasakani.json but missing (text only until it is there)" % pack.vision_entry["file"]
        return "image tower %s  %s  (still images; chat and serve read them unless --vision off)" % (f.name, _human(f.stat().st_size))
    if pack.has_vision:
        return "image tower inside %s  (still images; chat and serve read them unless --vision off)" % pack.weights.name
    return "none (text only: this folder has no image tower)"


def cmd_verify(args) -> int:
    from .cli import _human
    pack = ink.Ink2Pack.load(args.model_dir_path)
    if not pack.sha256:
        print("no checksums are listed for this folder (yasakani.json without \"sha256\").")
        return 2
    print("checking %d files (reads about %s) ..." % (len(pack.sha256), _human(sum(pack.sizes.values()))))
    res = pack.verify(progress=lambda n, ok: print("  %-36s %s" % (n, "OK" if ok else "DIFFERENT"), flush=True))
    bad = [n for n, ok in res.items() if not ok]
    print("all files match." if not bad else "%d file(s) differ: %s  (convert the folder again)" % (len(bad), ", ".join(bad)))
    if pack.weights_format == "yskn" and pack.weights.is_file():
        try:
            tbad = pack.check_tensors()
        except ValueError as ex:                          # ink2_yskn.FormatError: the header itself is damaged
            print("tensors: %s cannot be read: %s" % (pack.weights.name, ex))
            return 1
        if tbad:
            print("tensors: the CRC-32 of %d tensor(s) of %s differs: %s" % (len(tbad), pack.weights.name, ", ".join(tbad[:8])
                                                                            + (" ..." if len(tbad) > 8 else "")))
            return 1
        print("tensors: the CRC-32 of every tensor of %s matches." % pack.weights.name)
    if pack.drafter_format == "yskn" and pack.drafter is not None:
        try:
            dbad = pack.check_drafter()
        except ValueError as ex:                          # ink2_yskn.FormatError
            print("drafter: %s cannot be read: %s" % (pack.drafter.name, ex))
            return 1
        if dbad:
            print("drafter: the CRC-32 of %d tensor(s) of %s differs: %s" % (len(dbad), pack.drafter.name, ", ".join(dbad[:8])
                                                                            + (" ..." if len(dbad) > 8 else "")))
            return 1
        print("drafter: the CRC-32 of every tensor of %s matches." % pack.drafter.name)
    return 0 if not bad else 1


# ---- selfcheck --------------------------------------------------------------------------------------------------------------------
def selfcheck_fast(args) -> str:
    """The --fast value of the self-check: the one given on its command line, else "off".  The reference tokens are those of the
    standard path.  The fast path can pick another token at a near tie, and its order of sums depends on the GPU, so a self-check
    without --fast compares the standard path on every GPU, whatever $YASAKANI_INK2_FAST says."""
    v = getattr(args, "fast", None)
    return v if v not in (None, "") else "off"


def cmd_selfcheck(args) -> int:
    refuse_unsupported(args)
    pack = ink.Ink2Pack.load(args.model_dir_path)
    cases = ink.selfcheck_cases(pack, getattr(args, "reference", None))
    if not cases:
        print("the reference file has no cases")
        return 2
    if getattr(args, "fast", None) in (None, ""):
        args.fast = selfcheck_fast(args)
        _say("self-check: comparing the standard path (the reference is of this path; --fast on compares the fast path instead)")
    pack, eng, fmt, runner = open_stack(args, warm=False)
    try:
        rc = ink.run_selfcheck(eng, fmt, cases)
    finally:
        eng.close()
    print("self-check passed: the output equals the reference." if rc == 0 else "self-check FAILED.")
    return rc


# ---- chat -------------------------------------------------------------------------------------------------------------------------
CHAT_HELP = """commands:  /exit  quit      /reset  forget the conversation      /think on|off  thinking on or off (default on)
           /effort none|minimal|low|medium|high|xhigh|max  how long the model thinks (default: the template's 0.9)
           /image PATH  an image for the next message (when image input is on)
           /help  this text      Ctrl+C while it writes: stop the answer"""


def read_image_file(path) -> tuple:
    """--image PATH or /image PATH -> (file bytes, name).  A missing, unreadable or too large file is a YasakaniError naming it.
    Only the file the user names is read."""
    from . import inkling2_vision as V
    f = Path(str(path).strip().strip('"'))
    if not f.is_file():
        raise YasakaniError("image file not found: %s" % f)
    try:
        with f.open("rb") as fh:
            data = fh.read(V.MAX_IMAGE_BYTES + 1)
    except OSError as ex:
        raise YasakaniError("cannot read the image file %s (%s)" % (f, ex.strerror or type(ex).__name__))
    if len(data) > V.MAX_IMAGE_BYTES:
        raise YasakaniError("the image file %s is larger than %d MiB" % (f, V.MAX_IMAGE_BYTES >> 20))
    if not data:
        raise YasakaniError("the image file %s is empty" % f)
    return data, f.name


def prepare_image(runner, data: bytes, name: str):
    """File bytes -> the content part of one image for the next user message ({"type": "image", "image": PreparedImage}).  The
    image is decoded and checked here (and shrunk when it has more image tokens than the per-image cap; one line says so).
    A YasakaniError when image input is off or the file is not a usable still image."""
    from . import inkling2_vision as V
    st = getattr(runner, "vision", None)
    if st is None:
        raise YasakaniError("--image needs image input, which is off: %s"
                            % (getattr(runner.engine, "vision_problem", "") or V.NO_TOWER))
    try:
        im = V.prepare_image(data, st.max_image_tokens)
    except V.VisionError as ex:
        raise YasakaniError("image %s: %s" % (name, ex)) from None
    except ImportError as ex:                       # Pillow is missing
        raise YasakaniError("image %s: %s" % (name, ex)) from None
    if im.shrunk:
        _say("%s: %s" % (name, im.describe(st.max_image_tokens)))
    else:
        _say("%s: image %dx%d, %d image tokens" % (name, im.width, im.height, im.tokens))
    return {"type": "image", "image": im}


def user_message(text: str, images: list) -> dict:
    """The user message of the chat: the images first (in the order given), then the text; without images a plain string."""
    if not images:
        return {"role": "user", "content": text}
    return {"role": "user", "content": list(images) + [{"type": "text", "text": text}]}


def _chat_params(args, pack, messages, kw):
    from .chat import ChatParams
    s = pack.sampling
    return ChatParams(messages=[dict(m) for m in messages], max_tokens=args.max_tokens,
                      temperature=s["temperature"] if args.temperature is None else args.temperature,
                      top_p=s["top_p"] if args.top_p is None else args.top_p,
                      top_k=s["top_k"] if args.top_k is None else args.top_k, seed=args.seed, template_kwargs=dict(kw))


def cmd_chat(args) -> int:
    from .chat import RequestError
    from .cli import _use_color
    pending = [read_image_file(x) for x in (getattr(args, "image", None) or [])]     # a bad file fails before the long load
    pack, eng, fmt, runner = open_stack(args, warm=True, images=True, image_option=True)
    try:
        pending = [prepare_image(runner, data, name) for data, name in pending]
    except YasakaniError:
        eng.close()
        raise
    color = _use_color()
    dim = (lambda s: "\x1b[2m%s\x1b[0m" % s) if color else (lambda s: s)
    kw = {}
    if args.no_think:
        kw["enable_thinking"] = False
    if args.effort:
        kw["reasoning_effort"] = args.effort
    base: List[dict] = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)
    one_shot = getattr(args, "ask", None)
    if not one_shot:
        print("YASAKANI chat (%s) - type /help for commands, /exit to quit." % pack.model_id)
    try:
        while True:
            if one_shot:
                line = one_shot
            else:
                try:
                    line = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
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
                    elif cmd == "effort" and rest in ink.EFFORTS:
                        kw["reasoning_effort"] = rest
                        print("(reasoning effort %s)" % rest)
                    elif cmd == "image" and line[1:].partition(" ")[2].strip():
                        try:
                            data, name = read_image_file(line[1:].partition(" ")[2])
                            pending.append(prepare_image(runner, data, name))
                            print("(%d image%s for the next message)" % (len(pending), "" if len(pending) == 1 else "s"))
                        except YasakaniError as ex:
                            print("cannot do that: %s" % ex)
                    else:
                        print(CHAT_HELP)
                    continue
            messages.append(user_message(line, pending))
            sent, pending = pending, []
            params = _chat_params(args, pack, messages, kw)
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
                pending = sent
                eng.reset()
                if one_shot:
                    return 130
                continue
            except RequestError as ex:
                print("cannot do that: %s" % ex)
                messages.pop()
                if sent:                                 # a refused image is not sent again
                    print("(the image of that message was dropped)" if len(sent) == 1 else "(the images of that message were dropped)")
                if one_shot:
                    return 1
                continue
            if parts["reasoning"] and not parts["content"] and u.finish != "stop":
                print(dim("\n(the limit of %d tokens was reached while the model was still thinking, so no answer was written.  "
                          "/effort low shortens the thinking, /think off skips it, --max-tokens raises the limit)" % args.max_tokens))
            tps = (u.completion_tokens / (u.gen_ms / 1000.0)) if u.gen_ms > 0 else 0.0
            vis = getattr(u, "vision", None) or {}
            print(dim("\n\n[%d tokens, %.1f tokens/s%s, prompt %d tokens%s in %.2f s%s, %s]" % (
                u.completion_tokens, tps, ", %.2f tokens per pass" % u.spec["tokens_per_pass"] if u.spec else "",
                u.prompt_tokens, " (%d image tokens)" % vis["image_tokens"] if vis.get("image_tokens") else "", u.prompt_ms / 1000.0,
                " (%d kept from the last turn)" % u.cached_tokens if u.cached_tokens else "",
                "complete" if u.finish == "stop" else "hit the length limit")))
            if one_shot:
                return 0
            msg = {"role": "assistant", "content": "".join(parts["content"])}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"])
            messages.append(msg)
    finally:
        eng.close()


# ---- serve ------------------------------------------------------------------------------------------------------------------------
def cmd_serve(args) -> int:
    from .cli import _stop_on_signals
    from .server import ApiState, ServerConfig, bind
    refuse_unsupported(args)
    local = args.host in ("127.0.0.1", "localhost", "::1")
    if not local and not args.api_key:
        raise YasakaniError("refusing to listen on %s without --api-key: anyone who can reach this computer could use the GPU" % args.host)
    args.ink2_images = True
    vision_options(args)                         # wrong image options fail before the server starts
    pack = ink.Ink2Pack.load(args.model_dir_path)
    context_of(args, pack)
    cfg = ServerConfig(host=args.host, port=args.port, api_key=args.api_key, model_id=pack.model_id,
                       default_max_tokens=args.default_max_tokens, max_tokens_cap=args.max_tokens_cap)
    cfg.sampling_defaults = dict(pack.sampling)
    cfg.effort_aliases = dict(ink.EFFORT_ALIASES)
    cfg.keep_tool_ids = True                     # the chat template finds the function of a tool result through its tool_call_id
    cfg.vision_parts = True                      # image parts go to the runner, which checks them (and refuses them without the tower)
    state = ApiState(cfg)
    state.version = __version__
    try:
        httpd = bind(state)
    except OSError as ex:
        raise YasakaniError("cannot listen on %s:%d (%s).  Is another server already using this port?  Use --port to pick another."
                            % (args.host, args.port, ex.strerror or type(ex).__name__))
    cfg.port = httpd.server_address[1]
    _stop_on_signals()
    threading.Thread(target=httpd.serve_forever, kwargs=dict(poll_interval=0.2), name="http", daemon=True).start()
    base = "http://%s:%d" % ("[%s]" % args.host if ":" in args.host else args.host, cfg.port)
    _say("listening on %s  (answers /health while the model loads)" % base)
    eng = None
    try:
        pack, eng, fmt, runner = open_stack(args, warm=not args.no_warmup, images=True)
        cfg.vision_problem = getattr(eng, "vision_problem", "") or ""
        state.runner = runner
        _say("ready: OpenAI-compatible API at %s/v1   (model id %r%s; one request at a time, text, tool calls%s)"
             % (base, cfg.model_id, ", API key required" if cfg.api_key else "",
                " and images" if runner.vision is not None else ""))
        _say("stop with Ctrl+C")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _say("stopping ...")
        return 0
    finally:
        state.runner = None
        httpd.shutdown()
        if eng is not None:
            eng.cancel()
            with state.gen_lock:
                eng.close()
        httpd.server_close()


# ---- bench ------------------------------------------------------------------------------------------------------------------------
def bench_settings(eng) -> dict:
    """The fast and copy settings the engine runs with (bench --json; 0.3.4): the mode asked, whether the fast path and its NVFP4
    part run, the copy mode and its shortest match."""
    copying = bool(getattr(eng, "copying", False))
    return dict(fast=dict(mode=getattr(eng, "fast_mode", None), on=bool(getattr(eng, "fast", False)),
                          nvfp4_prompt=bool(getattr(eng, "fast_nvfp4", False)), compute_cap=_cc_text(getattr(eng, "compute_cap", None))),
                copy=dict(mode=getattr(eng, "copy_mode", "off"), on=copying,
                          min_match=getattr(eng, "copy_min", None) if copying else None))


def _cc_text(cc):
    return "%d.%d" % tuple(cc) if cc and tuple(cc) != ink.CAP_UNKNOWN else None


def cmd_bench(args) -> int:
    from .chat import ChatParams
    from .cli import _gpu_lines, _load_prompts
    refuse_unsupported(args)
    n_tokens = 256 if args.n is None else args.n
    prompts = _load_prompts(args.prompts_file)
    pack, eng, fmt, runner = open_stack(args, warm=True)
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
                rows.append(dict(test=name, run=run + 1, prompt_tokens=u.prompt_tokens, tokens=u.completion_tokens,
                                 seconds=round(u.gen_ms / 1000.0, 3), prompt_seconds=round(u.prompt_ms / 1000.0, 3), tokens_per_s=round(tps, 1)))
                if u.spec:
                    rows[-1].update(tokens_per_pass=round(u.spec["tokens_per_pass"], 2), draft_n=u.draft_n, draft_n_accepted=u.draft_n_accepted)
                if not args.json:
                    r = rows[-1]
                    print("%-12s run %d  prompt %5d (%.2f s)  generated %5d  %7.1f tokens/s%s" %
                          (name, r["run"], r["prompt_tokens"], r["prompt_seconds"], r["tokens"], r["tokens_per_s"],
                           "  %.2f tokens per pass" % r["tokens_per_pass"] if "tokens_per_pass" in r else ""), flush=True)
    finally:
        eng.close()
    mean = sum(r["tokens_per_s"] for r in rows) / len(rows)
    if args.json:
        print(json.dumps(dict(version=__version__, model=pack.model_id, gpu=_gpu_lines(), tokens_per_test=n_tokens, runs=rows,
                              mean_tokens_per_s=round(mean, 1), drafter=eng.draft_note, **bench_settings(eng)), indent=1))
    else:
        print("mean over %d run(s): %.1f tokens/s   (greedy, thinking %s, %d tokens each; %s)" % (
            len(rows), mean, "on" if args.think else "off", n_tokens, eng.draft_note))
    return 0


COMMANDS = dict(info=cmd_info, verify=cmd_verify, selfcheck=cmd_selfcheck, chat=cmd_chat, serve=cmd_serve, bench=cmd_bench)


def dispatch(name: str, args):
    """The Inkling-2 command for ``name`` when the model folder of ``args`` is an Inkling-2 folder, else None."""
    d = getattr(args, "model_dir", None) or os.environ.get("YASAKANI_MODEL_DIR")
    if name not in COMMANDS or not d or not ink.is_inkling2(d):
        return None
    args.model_dir_path = Path(d)
    return COMMANDS[name]
