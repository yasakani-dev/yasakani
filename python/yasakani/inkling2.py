# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Inkling-2 Mini (``"architecture": "inkling2"`` in ``yasakani.json``): model folder, chat template, output stream, sampling and
the chat runner on the Inkling-2 engine library.

One request at a time.  The engine library is a separate DLL (``yskn_inkling2.dll``, next to ``yskn.dll`` in the engine
zip) with its own C API; its ctypes binding is the package module ``_ink2_native.py`` (with the safetensors reader ``_ink2_ckpt.py``).
The library is found as ``find_dll`` says.  Model folders are made by ``tools/convert/inkling2_convert.py``.

Two weights formats (the ``weights`` entry of yasakani.json): ``hf-safetensors`` (the published file, read by the binding) and
``yskn`` (a re-quantised ``.yskn`` file of version 2, yasakani/ink2_yskn.py).  For a ``.yskn`` file the CRC-32 of every tensor is
checked first (``YASAKANI_INK2_SKIP_TENSOR_CHECK=1`` skips it), and the binding is given the file through an object with the interface of its safetensors reader: the experts as
stored, and the dense matrices that are stored packed as stored too when the engine (interface version 3) says through its
capability query that it takes them packed and the binding hands them over (``TAKES_PACKED_DENSE``); otherwise they are decoded
to bfloat16.  ``YASAKANI_INK2_DENSE=bf16`` decodes them even for such an engine (to compare).

How the model writes a reply (``response_template`` of its tokenizer_config.json): blocks ``<|message_model|>[name]<|content_X|>
text<|end_message|>``, where X is ``thinking`` (reasoning) or ``text`` (the answer), and ``<|content_model_end_sampling|>`` (the
stop token) at the end of the turn.  The prompt ends after the messages, without an assistant header: the model writes it.
Thinking controls: ``reasoning_effort`` goes to the template's "Thinking effort level" line (none, minimal, low, medium, high,
xhigh, max or a number from 0 to 0.99; default 0.9).  The model card says that even the lowest effort does not fully turn the
thinking off, so ``enable_thinking: false`` sets the effort to 0 and starts the reply with an answer block
(``<|message_model|><|content_text|>``) so that no thinking is written.

The drafter (the ``drafter`` entry of yasakani.json: the model's published ``mtp.safetensors``, or in a folder converted with
``--scheme quality`` the converted drafter file ``mtp_d4.yskn``, docs/FORMAT.md section 10.6) proposes a few tokens that the
model checks in one pass (speculative decoding).  The output does not change: greedy gives the same tokens, sampling the same
distribution (and with a fixed seed the same tokens, see ``Sampler.seek``).  ``--draft auto | on | off`` / ``$YASAKANI_DRAFT``:
auto (default) lets the engine use the drafter only while it makes generation faster, on always drafts, off does not load it.
A folder without the drafter, or an engine library without the drafter calls (or, for a converted drafter file, without the
capability to multiply it as stored), generates without drafting; with ``on`` that is an error.

The fast path (0.3.3; ``--fast auto | on | off`` / ``$YASAKANI_INK2_FAST``, see ``FAST_ENGINE_OPTIONS``): the keys and values of the
full-attention layer visits in an 8-bit cache (MXFP8 with BF16 copies of the first 8 and the last 128 positions), prompt passes
of 256 rows with tiled Flash-GQA attention on the tensor cores, the fast expert kernels, the Split-K attention for single steps
and short passes (every row of a pass with logits gets the bits of a single step), the arg-max on the GPU for greedy steps, the
Split-K attention of the drafter and its position-aware on / off rule, and on the RTX 50 series the expert products of prompt
passes on the NVFP4 tensor cores.  auto (default; 0.3.4) = on for the compute capabilities of ``FAST_AUTO_CAPS`` (8.6 RTX 30,
8.9 RTX 40, 12.0 RTX 50; nvidia-smi through yasakani/gpu.py, the GPU the engine uses) and where the library runs the NVFP4 prompt
products, else off (an unknown compute capability too); on = the fast path with what the GPU supports (the NVFP4 part where the
library reports it, RTX 50); off = the engine as in 0.3.2 (the same results bit for bit).  The fast
path gives other numbers than the 0.3.2 path (the 8-bit cache rounds the keys and values; other sum orders): greedy output can
differ after some tokens.  ``--copy auto`` / ``$YASAKANI_INK2_COPY=auto`` (default off) adds copy decoding for edits
(``_ink2_copyedit``): ids that repeat the context are proposed and checked in one pass; the output does not change.

Tool calls (OpenAI ``tools``; the parsing draws on PR #28 by @audreyt): the template declares the tools in a block
``<|message_system|>tool_declare<|content_xml|>[JSON specs]<|end_message|>`` and writes a call of an earlier turn as
``<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>``; a tool result is a
``<|message_tool|>NAME`` block, NAME taken from the message's ``name`` or found through its ``tool_call_id``.  The model writes its
calls the same way (``tool_calls`` of the response_template: ``content.name`` and ``content.args``), several in one reply when it
calls several tools.  ``OutputStream(tools=True)`` turns each such block into a ``("tool_call", {"name", "arguments"})`` item when it
ends; a block that is not a JSON object with a name comes back as answer text.  ``tool_choice``: "auto" as it is; "none" offers no
tools (the server); "required" with one function (a named function) writes the call header ``<|message_model|>NAME
<|content_invoke_tool_json|>`` for the model after its thinking; "required" with several writes ``<|message_model|>`` for it and
then lets it pick only ids that spell one of the offered names (``NameTrie``: the ids of each name as the template writes them, at
every position of the name) and, where a name is complete, ``<|content_invoke_tool_json|>``; so the header is always that of a call
of an offered function (masking the block openers only at the first position after ``<|message_model|>`` was not enough: the model
wrote another id there and its answer as the block's name, and no call came).  The arguments are the model's; the name of the
call is that of its header (a JSON object naming another function does not change it).  With required or a named function the
reply ends after the first call that can be read (the stop token is written for the model): once its first call was forced, the
model went on with thinking and text, and with thinking off in 2 of 5 runs with call after call until max_tokens (53 and 55
calls).  A call block that cannot be read is followed only by another call header of an allowed function or the stop token
(after <|message_model|> another kind of block ends the reply).  With ``enable_thinking: false`` and tools the effort is 0 and the
model writes its own block header: it may think briefly (the model card: the lowest effort does not turn the thinking fully off),
then answers or calls.  The answer block is not opened for it as without tools (that would rule out a call); masking only the
thinking opener after ``<|message_model|>`` was tried and made the model call a tool for 1 + ... + 5 in 1 of 7 runs (greedy and
6 seeds), the effort-0 start in none of the same 7.

Still images (``--vision auto | on | off`` / ``$YASAKANI_INK2_VISION``, see ``Ink2Engine.set_vision``): the image tower
(yasakani/inkling2_vision.py reads and checks it) runs inside the engine library.  It needs the library's capability bit
CAP_VISION (interface version 4) and a folder with the tower: the separate file of the manifest's ``vision`` entry, or the
published safetensors weights file, which holds ``model.visual.*``.  auto (default) loads it when both are there.  A request gives
images as content parts of user and tool messages; the template writes one placeholder (id 200054) per image, which ``prepare``
replaces by one id per 40 x 40 pixel tile of the image (rows x cols), so every image vector is one ordinary prompt position.
The prompt that ``prepare`` returns (``PromptIds``) carries its images and their positions.  ``Ink2Engine.read_prompt`` compares
positions by (id, image key, index in the image), so a new request goes on from the positions of the last one only where the
same images are at the same places; the vectors of an image whose positions must be read are computed once by the tower (or
taken from the host cache of earlier images, ``FeatureCache``) into the engine's feature rows and read with
``ink2_feed_many_img``.  A request without images takes exactly the path of a text-only engine.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._native import YasakaniError

IMAGE_TOKEN = 200054                   # the image placeholder of the template (<|unused_200054|>): one id per image position

ARCHITECTURE = "inkling2"
MANIFEST = "yasakani.json"
WEIGHTS_FORMATS = ("hf-safetensors", "yskn")
CONTEXT_MARGIN = 16                    # positions kept free at the end of the context
DRAFT_MODES = ("auto", "on", "off")
DRAFT_ENV, DRAFT_DEPTH_ENV = "YASAKANI_DRAFT", "YASAKANI_DRAFT_DEPTH"
ROLLBACK_ENV = "YASAKANI_INK2_ROLLBACK"
FAST_MODES = ("auto", "on", "off")
FAST_ENV = "YASAKANI_INK2_FAST"
# The fast path: (before the load) rows of the largest prompt pass and of a feed_many pass, the 8-bit cache and its BF16 copies
# of the last positions; (after it) the engine options in this order.  "nvfp4_prefill" only where nvfp4_prompt() says so.
FAST_LOAD_OPTIONS = (("max_rows", 256), ("chunk", 256), ("kv_mxfp8", 1), ("kv_hot", 128))
FAST_ENGINE_OPTIONS = (("packed_prefill", 1), ("nvfp4_prefill", 4), ("flash_attn", 3), ("rel_t", 1), ("fast", 1),
                       ("splitk_attn", 1), ("flash_min_rows", 9), ("splitk_max_rows", 8))
# positions the engine can drop without reading the prompt again (its option "rollback"; the engine's own default is 64).  The next
# request of a chat goes on from the ids already read only where it starts with them: a client that does not send the thinking of
# the last reply back (most OpenAI clients) or that writes a tool call in another form drops the whole last reply, which the engine
# can do without a new reading of the conversation up to this many positions.  Costs GPU memory for the rings of the sliding layers.
DEFAULT_ROLLBACK = 512


def rollback_positions(environ=None) -> int:
    """The rollback positions the engine will be loaded with ($YASAKANI_INK2_ROLLBACK, else DEFAULT_ROLLBACK; 0 = the engine's own),
    for planning the context.  A value that is not a number counts as the default here: the load refuses it with a message."""
    v = ((os.environ if environ is None else environ).get(ROLLBACK_ENV) or "").strip()
    try:
        return max(0, int(v)) if v else DEFAULT_ROLLBACK
    except ValueError:
        return DEFAULT_ROLLBACK


def fast_mode_of(value: Optional[str] = None, environ=None) -> str:
    """auto | on | off: ``value`` (--fast), else ``$YASAKANI_INK2_FAST``, else auto."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(FAST_ENV, "")
    v = (v or "auto").strip().lower()
    if v not in FAST_MODES:
        raise YasakaniError("%s / --fast must be one of %s, not %r" % (FAST_ENV, ", ".join(FAST_MODES), v))
    return v


# auto turns the fast path on for these compute capabilities (0.3.4: RTX 30 8.6, RTX 40 8.9, RTX 50 12.0; 0.3.3: 12.0 only).  The
# context planned before the load (fast_expected) and the engine (Ink2Engine through fast_choice) use this one rule;
# quickstart/quickstart.py keeps a copy of it (INK2_FAST_AUTO_CAPS).
FAST_AUTO_CAPS = ((8, 6), (8, 9), (12, 0))
CAP_UNKNOWN = (0, 0)                   # a caller asked nvidia-smi and got no answer: auto = off, and the engine does not ask again


def _cap(compute_cap):
    try:
        return tuple(int(x) for x in compute_cap) if compute_cap else None
    except (TypeError, ValueError):
        return None


def fast_expected(mode: str, compute_cap) -> bool:
    """For planning the context before the load: does ``mode`` turn the fast path on for a GPU of ``compute_cap`` (nvidia-smi,
    (major, minor) or None = unknown)?  auto: on for FAST_AUTO_CAPS (RTX 30, 40 and 50), off for others and when unknown."""
    return mode == "on" or (mode == "auto" and _cap(compute_cap) in FAST_AUTO_CAPS)


def engine_compute_cap(environ=None):
    """The compute capability of the GPU the engine opens (CUDA device 0 of what it sees; nvidia-smi through gpu.detect, which
    follows CUDA_VISIBLE_DEVICES), or None.  Never raises: an unknown GPU leaves auto off."""
    try:
        from . import gpu as gpus
        g = gpus.detect(0, environ).worst()
        return _cap(g.compute_cap) if g is not None else None
    except Exception:
        return None


def fast_choice(mode: str, compute_cap, nvfp4: bool, has_fast: bool, backend: str):
    """(fast, fast_nvfp4, start-up line) of the engine.  ``compute_cap``: (major, minor) or None (unknown); ``nvfp4``: the library
    runs the NVFP4 prompt products on this GPU; ``has_fast``: the library has the fast path (0.3.3 or later).  auto = on for
    FAST_AUTO_CAPS and where the library reports NVFP4.  Raises YasakaniError for --fast on with a library without the fast path."""
    cap = _cap(compute_cap)
    cuda = backend == "cuda"
    want = mode == "on" or (mode == "auto" and (fast_expected(mode, cap) or (cuda and bool(nvfp4))))
    if want and not has_fast:
        if mode == "on":
            raise YasakaniError("--fast on needs the engine library of 0.3.3 or later")
        return False, False, "fast path: off (this engine library has no fast path; the library of 0.3.3 or later has it)"
    if want and not cuda:
        return False, False, "fast path: off (it runs on the GPU only; the %s backend runs the standard path)" % backend
    if want:
        nv = cuda and bool(nvfp4)
        return True, nv, "fast path: on (8-bit key/value cache%s; --fast off for the standard path)" % (
            ", NVFP4 prompt products" if nv else "; this GPU has no NVFP4 prompt products")
    if mode == "off":
        return False, False, "fast path: off (--fast off or %s=off; auto turns it on for RTX 30, 40 and 50 cards)" % FAST_ENV
    if cap is None or cap == CAP_UNKNOWN:
        return False, False, ("fast path: off (the GPU's compute capability is unknown; auto turns it on for RTX 30, 40 and 50 "
                              "cards; --fast on turns it on)")
    return False, False, "fast path: off (auto turns it on for RTX 30, 40 and 50 cards; --fast on turns it on here)"


VISION_MODES = ("auto", "on", "off")
VISION_ENV = "YASAKANI_INK2_VISION"
CAP_VISION = 32                        # ink2_capabilities() bit: the image tower and ink2_feed_many_img (interface version 4)
VISION_OFF = "image input is off (--vision off)"
VISION_NO_LIBRARY = "this engine library has no image input (an older yskn_inkling2.dll: the library of this release has it)"
VISION_NO_GPU = "image input runs on the GPU only (cuda backend)"


def vision_mode_of(value: Optional[str] = None, environ=None) -> str:
    """auto | on | off: ``value`` (--vision), else ``$YASAKANI_INK2_VISION``, else auto."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(VISION_ENV, "")
    v = (v or "auto").strip().lower()
    if v not in VISION_MODES:
        raise YasakaniError("%s / --vision must be one of %s, not %r" % (VISION_ENV, ", ".join(VISION_MODES), v))
    return v


def vision_limits(max_image_tokens=None, max_request_tokens=None) -> Tuple[int, int]:
    """(image tokens per image, per request) of --vision-max-image-tokens (64 .. 4,096; default 1,024) and
    --vision-max-request-image-tokens (the per-image value .. 65,536; default 8 x the per-image value)."""
    from . import inkling2_vision as V
    cap = V.DEFAULT_IMAGE_TOKENS if max_image_tokens is None else max_image_tokens
    try:
        cap = V.check_image_cap(cap)
    except V.VisionError as ex:
        raise YasakaniError("--vision-max-image-tokens: %s" % ex) from None
    try:
        return cap, V.request_token_limit(cap, max_request_tokens)
    except V.VisionError as ex:
        raise YasakaniError("--vision-max-request-image-tokens: %s" % ex) from None


def library_has_vision(dll: Optional[str] = None) -> bool:
    """The engine library find_dll(dll) names reports image input (CAP_VISION) and is a cuda build.  Never raises."""
    try:
        from . import _ink2_native as mod
        path = find_dll(dll)
        lib = mod.open_dll(str(path))
        if not getattr(lib, "has_vision", False):
            return False
        return backend_of(lib.ink2_build_info().decode("utf-8", "replace")) == "cuda"
    except Exception:
        return False


def vision_plan_mib(pack: "Ink2Pack", mode: Optional[str] = None, max_image_tokens=None, max_request_tokens=None,
                    dll: Optional[str] = None, has_library: Optional[bool] = None) -> float:
    """GPU memory (MiB) the image tower will take when the model is opened with these options, for planning the context
    (gpu.ink2_context(..., vision_mib=...)): 0 when the mode is off, the folder has no tower, or (auto) the library has no image
    input (``has_library``: known already; None = ask the library).  The weights count only for a separate tower file: the
    published safetensors file holds them and is counted with the weights."""
    from . import gpu as gpus
    mode = vision_mode_of(mode)
    if mode == "off" or not pack.has_vision:
        return 0.0
    if mode == "auto" and not (library_has_vision(dll) if has_library is None else has_library):
        return 0.0
    cap, req = vision_limits(max_image_tokens, max_request_tokens)
    return gpus.ink2_vision_mib(cap, req, weights=bool(pack.vision_entry))


@dataclass
class Ink2VisionState:
    """Image input of a loaded engine (Ink2Engine.vision; the runner's ``vision``)."""
    kind: str                          # where the tower came from: "yskn" (separate file) or "hf-safetensors"
    max_image_tokens: int              # per image (an image above it is shrunk)
    max_request_tokens: int            # per request = feature rows of the engine
    memory: Tuple[int, int, int]       # (weights, scratch, feature buffer) bytes on the device
    cache: object                      # inkling2_vision.FeatureCache of image vectors in host memory
    load_seconds: float = 0.0


class PromptIds(list):
    """Prompt ids with the images of the prompt: ``images`` = [(first position, inkling2_vision.PreparedImage)], one entry per
    image in prompt order, each over ``image.tokens`` positions of id IMAGE_TOKEN; ``image_tokens`` = their sum."""
    images: Tuple = ()
    image_tokens: int = 0


MAX_DRAFT_DEPTH = 7
DRAFTER_FORMATS = ("hf-safetensors", "yskn")   # the published mtp.safetensors, or a converted drafter file (.yskn version 2)
CAP_DRAFTER_PACKED = 4                 # ink2_capabilities() bit: the engine multiplies a converted drafter file as stored
DRAFT_CALLS = ("draft_load", "draft_mode", "spec_step", "spec_stats")   # what the engine wrapper needs for drafting
# reasoning_effort values of the chat template (its effort_map); the server and the chat accept these names
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {e: e for e in EFFORTS}

# special tokens of the reply format (names; the ids are read from tokenizer.json)
T_MODEL, T_THINKING, T_TEXT, T_END, T_STOP = ("<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>",
                                              "<|content_model_end_sampling|>")
ANSWER_OPENERS = ("<|content_text|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>", "<|content_invoke_tool_json|>")
T_CALL = "<|content_invoke_tool_json|>"          # the block of one tool call (JSON {"name": ..., "args": {...}})


class Allow(frozenset):
    """Ids that are the only ones that may be picked (``masked`` keeps these and masks every other id); a plain list masks its ids."""


class NameTrie:
    """The names of the offered functions as a tree of their ids (each name encoded alone, as the template writes it between
    ``<|message_model|>`` and ``<|content_invoke_tool_json|>``).  A node maps an id to the next node; where a name is complete the
    node also holds ``<|content_invoke_tool_json|>`` (mapped to None), so a name that is the beginning of another name can end there
    or go on."""

    def __init__(self, fmt: "Ink2Formatter", names: Sequence[str]):
        self.end = fmt.ids[T_CALL]
        self.root: Dict[int, Optional[dict]] = {}
        for name in names:
            ids = fmt.encode(name)
            if not ids or any(t in fmt.special for t in ids):
                raise YasakaniError("the function name %r cannot be written as a call header" % name)
            node = self.root
            for t in ids:
                node = node.setdefault(t, {})
            node[self.end] = None

    def allowed(self, node: dict) -> Allow:
        return Allow(node)


def architecture_of(folder) -> Optional[str]:
    """The ``architecture`` of the folder's yasakani.json; None when there is no readable manifest or it names none (Qwen3.8)."""
    try:
        meta = json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return meta.get("architecture") if isinstance(meta, dict) else None


def is_inkling2(folder) -> bool:
    return bool(folder) and architecture_of(folder) == ARCHITECTURE


# ---- model folder -------------------------------------------------------------------------------------------------------------------
@dataclass
class Ink2Pack:
    root: Path
    meta: dict
    weights: Path
    weights_format: str
    model_id: str
    max_ctx: int
    default_ctx: int
    sampling: Dict[str, float]
    stop_ids: List[int]
    sha256: Dict[str, str] = field(default_factory=dict)
    sizes: Dict[str, int] = field(default_factory=dict)
    drafter_entry: dict = field(default_factory=dict)       # the "drafter" entry of yasakani.json ({} when there is none)
    drafter: Optional[Path] = None                          # its file, when it is there and of a known format
    drafter_problem: str = ""                               # why an entry that is listed is not used
    vision_entry: dict = field(default_factory=dict)        # the "vision" entry of yasakani.json ({} when there is none)

    @classmethod
    def load(cls, root) -> "Ink2Pack":
        root = Path(root)
        if not root.is_dir():
            raise YasakaniError("model folder not found: %s" % root)
        try:
            meta = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            raise YasakaniError("cannot read %s in %s: %s" % (MANIFEST, root, ex))
        if meta.get("architecture") != ARCHITECTURE:
            raise YasakaniError("%s does not describe an Inkling-2 model folder" % MANIFEST)
        w = meta.get("weights") or {}
        if not isinstance(w, dict) or not w.get("file"):
            raise YasakaniError("%s has no weights entry" % MANIFEST)
        fmt = str(w.get("format") or "hf-safetensors")
        if fmt not in WEIGHTS_FORMATS or (fmt == "yskn" and int(w.get("version") or 2) != 2):
            raise YasakaniError("weights format %r is not supported by this version of the package" % fmt)
        dr = meta.get("drafter")
        vi = meta.get("vision")
        named = [("weights", w.get("file")), ("drafter", dr.get("file") if isinstance(dr, dict) else None),
                 ("vision", vi.get("file") if isinstance(vi, dict) else None)]
        sums = meta.get("sha256")
        named += [("sha256", k) for k in sums] if isinstance(sums, dict) else []
        for what, f in named:                               # every file the manifest names must be inside the model folder
            if f and (not isinstance(f, str) or Path(f).is_absolute() or Path(f).drive or ".." in Path(f).parts):
                raise YasakaniError("%s: the %s entry %r is not a file inside the model folder" % (MANIFEST, what, f))
        need = [w["file"], "config.json", "tokenizer.json", "chat_template.jinja"]
        missing = [n for n in need if not (root / n).is_file()]
        if missing:
            raise YasakaniError("model folder %s is missing: %s" % (root, ", ".join(missing)))
        max_ctx = int(meta.get("max_ctx") or 32768)
        s = meta.get("sampling") or {}
        d = meta.get("drafter")
        d = dict(d) if isinstance(d, dict) and d.get("file") else {}
        drafter, problem = None, ""
        if d:
            if str(d.get("format") or "hf-safetensors") not in DRAFTER_FORMATS:
                problem = "drafter format %r is not supported by this version of the package" % d.get("format")
            elif d.get("format") == "yskn" and d.get("version", 2) != 2:
                problem = "drafter file version %r is not supported by this version of the package" % d.get("version")
            elif not (root / d["file"]).is_file():
                problem = "the drafter file %s is missing" % d["file"]
            else:
                drafter = root / d["file"]
        vision_entry = dict(vi) if isinstance(vi, dict) and vi.get("file") else {}
        return cls(root=root, drafter_entry=d, drafter=drafter, drafter_problem=problem, vision_entry=vision_entry, meta=meta, weights=root / w["file"], weights_format=fmt, model_id=str(meta.get("model_id") or "inkling-2"),
                   max_ctx=max_ctx, default_ctx=min(int(meta.get("default_ctx") or max_ctx), max_ctx),
                   sampling=dict(temperature=float(s.get("temperature", 1.0)), top_p=float(s.get("top_p", 1.0)),
                                 top_k=int(s.get("top_k", 0) or 0)),
                   stop_ids=[int(i) for i in (meta.get("stop_token_ids") or [])],
                   sha256=dict(meta.get("sha256") or {}), sizes={k: int(v) for k, v in (meta.get("sizes") or {}).items()})

    def verify(self, progress=None) -> Dict[str, bool]:
        """Size, then SHA-256 of every listed file (the weights file is about 20 GB: a minute or so)."""
        res = {}
        sums, sizes = dict(self.sha256), dict(self.sizes)
        for d in (self.drafter_entry, self.vision_entry):
            if d.get("sha256") and d["file"] not in sums:        # the drafter / vision entry carries its own checksum
                sums[d["file"]] = str(d["sha256"])
                if d.get("size") is not None:
                    sizes[d["file"]] = int(d["size"])
        for name, want in sorted(sums.items(), key=lambda kv: (sizes.get(kv[0], 0), kv[0])):
            p = self.root / name
            ok = p.is_file() and (name not in sizes or p.stat().st_size == sizes[name])
            if ok:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for b in iter(lambda: f.read(1 << 24), b""):
                        h.update(b)
                ok = h.hexdigest() == want
            res[name] = ok
            if progress:
                progress(name, ok)
        return res

    def check_tensors(self, threads: int = 8) -> List[str]:
        """A .yskn weights file: the CRC-32 of every tensor part -> the names of the tensors that differ (empty: all match)."""
        if self.weights_format != "yskn":
            return []
        from .ink2_yskn import YsknFile
        with YsknFile(self.weights) as yf:
            return sorted({n for n, _ in yf.check(threads=threads)})

    @property
    def vision_file(self) -> Optional[Path]:
        """The separate image-tower file of the ``vision`` entry (None without one; it may be missing on disk)."""
        return self.root / self.vision_entry["file"] if self.vision_entry else None

    @property
    def has_vision(self) -> bool:
        """The folder has an image tower: the file of its ``vision`` entry is there, or its weights are the published safetensors
        file and config.json has a ``vision_config``.  Reads no tensor (inkling2_vision.vision_weights checks them)."""
        if self.vision_entry:
            return self.vision_file.is_file()
        if self.weights_format != "hf-safetensors" or not self.weights.is_file():
            return False
        try:
            config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(config, dict) and isinstance(config.get("vision_config"), dict)

    @property
    def drafter_format(self) -> str:
        """hf-safetensors (the published drafter) or yskn (a converted drafter file); "" without a drafter entry."""
        return str(self.drafter_entry.get("format") or "hf-safetensors") if self.drafter_entry else ""

    def check_drafter(self, threads: int = 8) -> List[str]:
        """A converted drafter file (.yskn): the CRC-32 of every tensor part -> the names of the tensors that differ."""
        if self.drafter is None or self.drafter_format != "yskn":
            return []
        from .ink2_yskn import YsknFile
        with YsknFile(self.drafter) as yf:
            return sorted({n for n, _ in yf.check(threads=threads)})


# ---- chat template and tokenizer ---------------------------------------------------------------------------------------------------
_GENERATION_TAG = re.compile(r"\{%-?\s*(end)?generation\s*-?%\}")      # a tag of the transformers template environment (no effect)


def _raise_exception(message):
    from jinja2.exceptions import TemplateError
    raise TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class Ink2Formatter:
    """chat_template.jinja + tokenizer.json of one Inkling-2 folder."""

    def __init__(self, folder):
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        d = Path(folder)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        source = _GENERATION_TAG.sub("", (d / "chat_template.jinja").read_text(encoding="utf-8"))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        self.template = env.from_string(source)
        ids = {}
        for name in (T_MODEL, T_THINKING, T_TEXT, T_END, T_STOP) + ANSWER_OPENERS:
            i = self.tok.token_to_id(name)
            if i is None:
                raise YasakaniError("the tokenizer has no token %s (not an Inkling-2 tokenizer?)" % name)
            ids[name] = int(i)
        self.ids = ids
        cfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        self.special = frozenset(int(t["id"]) for t in cfg.get("added_tokens", []) if t.get("special"))
        self.answer_prefix = [ids[T_MODEL], ids[T_TEXT]]

    def render(self, messages: Sequence[dict], tools: Optional[Sequence[dict]] = None, **template_kwargs) -> str:
        return self.template.render(messages=list(messages), add_generation_prompt=True, tools=list(tools) if tools else None,
                                    **template_kwargs)

    def encode(self, text: str) -> List[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def prompt(self, messages: Sequence[dict], template_kwargs: Optional[dict] = None) -> Tuple[List[int], bool]:
        """(prompt ids, the reply starts inside the answer).  ``enable_thinking`` False: effort 0 and an answer block opened in the
        prompt (see the module text).  A wrong effort raises the template's error (jinja2 TemplateError)."""
        kw = dict(template_kwargs or {})
        thinking = kw.pop("enable_thinking", None)
        if thinking is False:
            kw["reasoning_effort"] = "none"
        ids = self.encode(self.render(messages, **kw))
        if thinking is False:
            return ids + self.answer_prefix, True
        return ids, False

    def call_header(self, name: str) -> List[int]:
        """The ids the model writes before the arguments of a call of ``name``: <|message_model|>NAME<|content_invoke_tool_json|>."""
        return [self.ids[T_MODEL]] + self.encode(name) + [self.ids[T_CALL]]

    def name_trie(self, tools: Sequence[dict]) -> NameTrie:
        """The NameTrie of the functions in ``tools`` (OpenAI form, as yasakani.chat.clean_tools gives them)."""
        return NameTrie(self, [t["function"]["name"] for t in tools])

    def tool_prompt(self, messages: Sequence[dict], tools: Sequence[dict], template_kwargs: Optional[dict] = None,
                    call: Optional[str] = None) -> Tuple[List[int], str, List[int]]:
        """A request with tools -> (prompt ids, start, mask of the first position: a list of masked ids, or an ``Allow`` of the
        only ids allowed).  ``call``: None (tool_choice auto), the name of the function that must be called, or "" (a call of any
        of the tools must be made).  ``start``: what the reply's first id continues: "free" (nothing: the model writes its block
        header; also with thinking off, at effort 0), "head" (after <|message_model|>, tool_choice required among several tools:
        the first id must begin one of the names, ``NameTrie``), "reasoning" (inside a thinking block the prompt opened) or "call"
        (inside the call block of ``call``; its header is in the prompt)."""
        kw = dict(template_kwargs or {})
        thinking = kw.pop("enable_thinking", None)
        if thinking is False:
            kw["reasoning_effort"] = "none"
        ids = self.encode(self.render(messages, tools=tools, **kw))
        i = self.ids
        if call is None:
            return ids, "free", []
        if thinking is False:
            if call:
                return ids + self.call_header(call), "call", []
            trie = self.name_trie(tools)
            return ids + [i[T_MODEL]], "head", trie.allowed(trie.root)
        return ids + [i[T_MODEL], i[T_THINKING]], "reasoning", []


class StreamDecoder:
    """Incremental detokenizer (as yasakani.textio.StreamDecoder): ``push`` returns the text that became complete."""

    def __init__(self, tok):
        self.tok = tok
        self.ids: List[int] = []
        self.prefix = 0
        self.read = 0

    def _dec(self, a: int, b: Optional[int] = None) -> str:
        return self.tok.decode(self.ids[a:b], skip_special_tokens=False)

    def push(self, new_ids) -> str:
        self.ids.extend(int(i) for i in new_ids)
        if len(self.ids) == self.read:
            return ""
        old = self._dec(self.prefix, self.read)
        new = self._dec(self.prefix)
        if len(new) > len(old) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(old):]
        return ""

    def flush(self) -> str:
        if len(self.ids) == self.read:
            return ""
        old = self._dec(self.prefix, self.read)
        new = self._dec(self.prefix)
        self.prefix = self.read = len(self.ids)
        return new[len(old):]


def parse_call(text: str, header_name: str = "") -> Optional[dict]:
    """The inside of one call block -> {"name", "arguments"}, or None when it is not a call.  The model writes
    ``{"name": NAME, "args": {...}}`` (the template's form); ``arguments`` instead of ``args``, arguments given as a JSON string,
    and a block without a name (the name is then the one in its header) are taken too."""
    try:
        obj = json.loads(text.strip())
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") if isinstance(obj.get("name"), str) and obj.get("name") else (header_name or "").strip()
    if not name:
        return None
    if "args" in obj or "arguments" in obj:
        args = obj.get("args", obj.get("arguments"))
    elif "name" not in obj:
        args = obj                                           # only the arguments, under a header with the name
    else:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


class OutputStream:
    """Reply ids -> ``("reasoning", text)`` / ``("content", text)`` pieces, one piece per id that completes some text, and with
    ``tools`` ``("tool_call", {"name", "arguments"})`` items.

    Text inside ``<|content_thinking|>`` blocks is reasoning; text inside ``<|content_text|>`` (and the other answer blocks) is the
    answer; the block headers (``<|message_model|>`` and the name after it), ``<|end_message|>``, the stop token and every other
    special token are not shown.  With ``tools`` a ``<|content_invoke_tool_json|>`` block is a call: it is read whole and becomes one
    item when the next special id (normally ``<|end_message|>``) or the end of the reply closes it; one that cannot be read as a call
    comes back as answer text (nothing the model wrote is lost).  Without ``tools`` it is answer text, as before.
    ``start``: "head" / "free" (between blocks, the default), "content" (``in_answer``), "reasoning" (the prompt opened a thinking
    block) or "call" (the prompt holds the header of a call of ``call_name``)."""

    def __init__(self, fmt: Ink2Formatter, in_answer: bool = False, tools: bool = False, start: Optional[str] = None,
                 call_name: str = "", pin_names: Sequence[str] = ()):
        self.fmt = fmt
        i = fmt.ids
        self.opens = {i[T_THINKING]: "reasoning"}
        self.opens.update({i[n]: "content" for n in ANSWER_OPENERS})
        self.tools = bool(tools)
        if self.tools:
            self.opens[i[T_CALL]] = "call"
        self.t_model, self.t_end = i[T_MODEL], i[T_END]
        start = start or ("content" if in_answer else "head")
        self.mode = {"free": "head"}.get(start, start)      # head: between blocks (headers and names are not shown)
        if self.mode not in ("head", "content", "reasoning", "call"):
            raise ValueError("unknown start %r" % start)
        self.dec = StreamDecoder(fmt.tok)
        self.n = 0
        self.reasoning_tokens = 0
        self.answer_at: Optional[int] = 0 if self.mode in ("content", "call") else None
        self.saw_reasoning = self.mode == "reasoning"
        self.head: List[int] = []                            # the ids after <|message_model|> (the name of the block)
        self.call_ids: List[int] = []
        self.call_name = call_name if self.mode == "call" else ""
        self.pin_names = frozenset(pin_names)               # a header with one of these names names the call (tool_choice forced)
        self.calls = 0

    def _end_call(self, out: List[Tuple[str, object]]) -> None:
        text = self.fmt.decode(self.call_ids)
        call = parse_call(text, self.call_name)
        if call is not None and self.call_name in self.pin_names:
            call["name"] = self.call_name
        if call is None:
            if text:
                out.append(("content", text))
        else:
            self.calls += 1
            out.append(("tool_call", call))
        self.call_ids, self.call_name = [], ""

    def feed(self, token: int) -> List[Tuple[str, object]]:
        token = int(token)
        out: List[Tuple[str, object]] = []
        k = self.n
        self.n += 1
        if token in self.fmt.special:
            if self.mode in ("reasoning", "content"):
                tail = self.dec.flush()
                if tail:
                    out.append((self.mode, tail))
            elif self.mode == "call":
                self._end_call(out)
            if token in self.opens:
                if self.opens[token] == "call":
                    self.call_name = self.fmt.decode(self.head).strip() if self.mode == "head" else ""
                    self.call_ids = []
                self.mode = self.opens[token]
                if self.mode == "reasoning":
                    self.saw_reasoning = True
                elif self.answer_at is None:
                    self.answer_at = k
            elif token in (self.t_model, self.t_end):
                self.mode = "head"
            elif self.mode == "call":
                self.mode = "head"                           # a call block closed by another special id
            self.head = []
            self.dec = StreamDecoder(self.fmt.tok)
            if self.answer_at is None:
                self.reasoning_tokens = self.n
            return out
        if self.answer_at is None:
            self.reasoning_tokens = self.n
        if self.mode in ("reasoning", "content"):
            piece = self.dec.push((token,))
            if piece:
                out.append((self.mode, piece))
        elif self.mode == "call":
            self.call_ids.append(token)
        elif self.mode == "head":
            self.head.append(token)
        return out

    def flush(self) -> List[Tuple[str, object]]:
        if self.mode in ("reasoning", "content"):
            tail = self.dec.flush()
            if tail:
                return [(self.mode, tail)]
        elif self.mode == "call":
            out: List[Tuple[str, object]] = []
            self._end_call(out)
            self.mode = "head"
            return out
        return []

    @property
    def reasoning_count(self) -> int:
        return self.reasoning_tokens if self.saw_reasoning else 0


# ---- sampling ---------------------------------------------------------------------------------------------------------------------
class Sampler:
    """Greedy (temperature 0) or temperature / top-k / top-p / min-p sampling with a seeded generator (numpy).  Exact: the
    vocabulary is cut only where the rule says so; the draw goes through block sums so that a token costs well under a
    millisecond on 200,000 entries."""

    BLOCK = 1024

    def __init__(self, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float = 0.0, seed: int = 0):
        self.t, self.top_p, self.top_k, self.min_p = float(temperature), float(top_p), int(top_k or 0), float(min_p or 0.0)
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        self._state0 = self.rng.bit_generator.state
        self._k = 0                                          # draws made since the seed

    def seek(self, k: int) -> None:
        """The next draw is draw number ``k`` of the seed (one draw per token): the token at output index k is drawn with the same
        random number whether earlier draws were made for drafted rows that were not kept, or not."""
        k = int(k)
        if k != self._k:
            bg = self.rng.bit_generator
            bg.state = self._state0
            if k:
                bg.advance(k)                                # PCG64: random() takes one 64-bit output
            self._k = k

    def _draw(self, p: np.ndarray) -> int:
        """Index drawn with probability p[i] / sum(p) (p >= 0, float32)."""
        n, B = p.size, self.BLOCK
        nb = -(-n // B)
        if nb * B != n:
            p = np.concatenate([p, np.zeros(nb * B - n, p.dtype)])
        cb = np.cumsum(p.reshape(nb, B).sum(axis=1, dtype=np.float64))
        r = self.rng.random() * cb[-1]
        self._k += 1
        b = min(int(np.searchsorted(cb, r, side="right")), nb - 1)
        while b > 0 and cb[b] == cb[b - 1]:                 # an empty block (rounding at its edge): the one before
            b -= 1
        r -= cb[b - 1] if b else 0.0
        blk = p[b * B:(b + 1) * B]
        c = np.cumsum(blk, dtype=np.float64)
        j = min(int(np.searchsorted(c, r, side="right")), B - 1)
        while j > 0 and blk[j] == 0.0:
            j -= 1
        return b * B + j

    def __call__(self, logits: np.ndarray) -> int:
        if self.t <= 0.0:
            return int(np.argmax(logits))
        x = np.asarray(logits, np.float32)
        idx = None
        if 0 < self.top_k < x.size:
            idx = np.argpartition(x, -self.top_k)[-self.top_k:]
            x = x[idx]
        p = np.exp((x - x.max()) * np.float32(1.0 / self.t))      # the most likely token has p = 1
        if self.min_p > 0.0:
            p[p < self.min_p] = 0.0
        if self.top_p < 1.0:
            total = float(p.sum(dtype=np.float64))
            want = self.top_p * total
            thr = 1e-4
            while True:                                      # every token above thr: if they hold top_p, the nucleus is among them
                cand = np.flatnonzero(p >= thr) if thr > 0 else np.arange(p.size)
                if thr <= 0 or float(p[cand].sum(dtype=np.float64)) >= want:
                    break
                thr = thr * 1e-4 if thr > 1e-30 else 0.0
            cand = cand[np.argsort(-p[cand], kind="stable")]
            cum = np.cumsum(p[cand], dtype=np.float64)
            keep = max(1, min(int(np.searchsorted(cum - p[cand], want, side="left")), cand.size))
            cand = cand[:keep]
            j = int(cand[self._draw(p[cand])])
        else:
            j = self._draw(p)
        return int(idx[j]) if idx is not None else j


# ---- engine ------------------------------------------------------------------------------------------------------------------------
LIBRARY = "yskn_inkling2.dll"          # the Inkling-2 engine library, next to yskn.dll in the engine zip
MAIN_LIBRARIES = ("yskn.dll", "libyskn.so")
REQUIRED_ABI = 3                       # ink2_abi_version() of the library this binding was written for (or newer)


def find_dll(path: Optional[str] = None) -> Path:
    """The Inkling-2 engine library, in this order: ``path`` (``--dll``) when it names a file other than the main library
    (yskn.dll), or a folder that holds yskn_inkling2.dll; ``$YASAKANI_INK2_DLL``; yskn_inkling2.dll next to the main library
    (``--dll`` naming yskn.dll, as the Windows launcher passes it, else ``$YASAKANI_DLL`` or the bin folder next to the package)."""
    tried = []
    main = None
    if path:
        p = Path(path)
        if p.is_dir():
            p = p / LIBRARY
        elif p.name.lower() in MAIN_LIBRARIES:
            main = p
            p = None
        if p is not None:
            if p.is_file():
                return p
            raise YasakaniError("Inkling-2 engine library not found: %s" % p)
    env = os.environ.get("YASAKANI_INK2_DLL")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise YasakaniError("Inkling-2 engine library not found: %s (YASAKANI_INK2_DLL)" % p)
    if main is None:
        from ._native import find_library
        try:
            main = find_library(None)
        except YasakaniError:
            main = None
    if main is not None:
        p = main.parent / LIBRARY
        if p.is_file():
            return p
        tried.append(str(p))
    raise YasakaniError("the Inkling-2 engine library (%s) was not found%s.  It is in the engine zip of YASAKANI 0.3.1 or newer, next "
                        "to yskn.dll: update the engine (START.cmd does this), or give its path with --dll or YASAKANI_INK2_DLL"
                        % (LIBRARY, " (looked for %s)" % "; ".join(tried) if tried else ""))


def find_binding(dll: Path) -> Path:
    """The binding of the engine library: the package module _ink2_native.py (nothing is loaded from outside the package)."""
    return Path(__file__).resolve().parent / "_ink2_native.py"


def load_binding(path: Optional[Path] = None):
    from . import _ink2_native
    return _ink2_native


def open_library(mod, dll: Path):
    """Opens the engine library through the binding and checks that it is the Inkling-2 library of a version this package can use."""
    try:
        lib = mod.open_dll(str(dll))
    except OSError as ex:
        from ._native import _VC_HINT
        raise YasakaniError("%s: %s%s" % (dll.name, _VC_HINT, ex)) from ex
    except AttributeError as ex:
        raise YasakaniError("%s is not the Inkling-2 engine library (%s)" % (dll, ex)) from ex
    abi = int(getattr(lib, "abi", 1))
    if abi < REQUIRED_ABI:
        raise YasakaniError("%s is too old for this package (interface version %d, %d or newer is needed): use the yskn_inkling2.dll "
                            "of the same release as the package" % (dll, abi, REQUIRED_ABI))
    return lib


def draft_mode_of(value: Optional[str] = None, environ=None) -> str:
    """auto | on | off: ``value`` (--draft), else ``$YASAKANI_DRAFT``, else auto."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(DRAFT_ENV, "")
    v = (v or "auto").strip().lower()
    if v not in DRAFT_MODES:
        raise YasakaniError("%s / --draft must be one of %s, not %r" % (DRAFT_ENV, ", ".join(DRAFT_MODES), v))
    return v


def draft_depth_of(value=None, environ=None) -> Optional[int]:
    """1 .. 7 tokens proposed per pass: ``value`` (--draft-depth), else ``$YASAKANI_DRAFT_DEPTH``, else None (the engine's)."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else (env.get(DRAFT_DEPTH_ENV) or "").strip()
    if v in (None, ""):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = 0
    if not 1 <= n <= MAX_DRAFT_DEPTH:
        raise YasakaniError("%s / --draft-depth must be a number from 1 to %d, not %r" % (DRAFT_DEPTH_ENV, MAX_DRAFT_DEPTH, v))
    return n


def backend_of(lib_info: str) -> str:
    b = (os.environ.get("YASAKANI_INK2_BACKEND") or "").strip().lower()
    if b:
        if b not in ("cuda", "host"):
            raise YasakaniError("YASAKANI_INK2_BACKEND must be cuda or host, not %r" % b)
        return b
    return "cuda" if "cuda" in lib_info else "host"


MAX_LAYERS = 64                  # the engine library takes at most this many layers (num_hidden_layers of config.json)


def read_config(root) -> dict:
    """config.json of a model folder, with the values the package and the engine rely on checked -> the parsed file.
    A value that the engine library does not take gives a YasakaniError here, before the library is called."""
    path = Path(root) / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise YasakaniError("cannot read config.json in %s: %s" % (Path(root), ex))
    t = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(t, dict):
        raise YasakaniError("config.json in %s has no text_config object" % Path(root))
    n = t.get("num_hidden_layers", 24)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_LAYERS:
        raise YasakaniError("config.json: num_hidden_layers is %r; this version of the engine takes 1 to %d layers" % (n, MAX_LAYERS))
    types = t.get("layer_types")
    if types is not None and (not isinstance(types, list) or len(types) != n or not all(isinstance(x, str) for x in types)):
        raise YasakaniError("config.json: layer_types is not a list of %d names (one per layer)" % n)
    if t.get("loop_config") is not None and not isinstance(t["loop_config"], dict):
        raise YasakaniError("config.json: loop_config is not an object")
    return config


class Ink2Engine:
    """The engine library with the state of one sequence: the ids it has read since the last reset, so that a prompt that only
    extends them (the next turn of a chat) reads only the new ids."""

    def __init__(self, pack: Ink2Pack, max_ctx: int, dll: Optional[str] = None, graph: bool = True, log=None,
                 fast: Optional[str] = None, compute_cap=None):
        """``fast``: auto | on | off (None: $YASAKANI_INK2_FAST, else auto); ``self.fast`` says what the engine runs.
        ``compute_cap``: the GPU's (major, minor) the caller planned the context with (CAP_UNKNOWN: asked, no answer); None: not
        given, then auto asks nvidia-smi here (engine_compute_cap) unless the library reports NVFP4.  The rule: fast_choice."""
        fast_mode = fast_mode_of(fast)
        read_config(pack.root)                               # values the library does not take stop here, with a message
        self.dll = find_dll(dll)
        self.binding_path = find_binding(self.dll)
        self.mod = load_binding(self.binding_path)
        lib = open_library(self.mod, self.dll)
        self.build_info = lib.ink2_build_info().decode("utf-8", "replace")
        self.backend = backend_of(self.build_info)
        self.max_ctx = int(max_ctx)
        nvfp4 = self.backend == "cuda" and bool(getattr(self.mod, "nvfp4_prompt", lambda _: 0)(lib))
        if compute_cap is None and fast_mode == "auto" and self.backend == "cuda" and not nvfp4:
            compute_cap = engine_compute_cap()
        self.fast_mode, self.compute_cap = fast_mode, _cap(compute_cap)
        has_fast = bool(lib.caps & 16 and hasattr(lib, "ink2_fast_path"))
        try:                                               # the CPU build of the tests: off whatever is asked
            self.fast, self.fast_nvfp4, self.fast_note = fast_choice(fast_mode, compute_cap, nvfp4, has_fast, self.backend)
        except YasakaniError as ex:
            raise YasakaniError("%s (%s)" % (ex, self.dll)) from ex
        self._cancel = threading.Event()
        t0 = time.perf_counter()
        self.weights_note = pack.weights_format
        self._yskn = None
        try:
            if pack.weights_format == "yskn":
                self.native = self._load_yskn(pack, lib, log)
            else:
                self.native = self.mod.Ink2Native().load(str(pack.root), backend=self.backend, max_positions=self.max_ctx,
                                                         dll=str(self.dll), weights=pack.weights.name, log=log,
                                                         **self._load_options())
            self.device_greedy = False
            if self.fast:
                self._fast_options(lib)
        except (RuntimeError, OSError, AssertionError, KeyError, ValueError) as ex:
            self.close()
            raise YasakaniError("the Inkling-2 engine could not load the model: %s" % ex) from ex
        if graph and self.backend == "cuda":
            try:
                self.native.set_option("graph", 1)
            except RuntimeError:
                pass
        self.load_seconds = time.perf_counter() - t0
        self.vocab = int(self.native.vocab)
        self.fed: List[int] = []
        self.last_logits: Optional[np.ndarray] = None
        self.drafting = False                                # a drafter is loaded (set_drafter)
        self.draft_mode = "off"
        self.draft_depth = 0
        self.draft_note = "drafter: not set up"
        self.copying = False                                 # copy decoding on (set_copy)
        self.copy_note = "copy: off"
        self.copy_mode, self.copy_min = "off", None
        self.vision: Optional[Ink2VisionState] = None        # image input loaded (set_vision)
        self.vision_note = "image input: not set up"
        self.vision_problem = "image input is not set up"
        self.last_vision: dict = {}                          # the images of the last prompt read (read_prompt)

    def _fast_options(self, lib) -> None:
        """The engine options of the fast path after the load (FAST_ENGINE_OPTIONS), the arg-max on the GPU for greedy steps
        (when the library has ink2_step_top1) and the drafter settings (``draft_fast``, read when the drafter is loaded)."""
        nat = self.native
        for name, value in FAST_ENGINE_OPTIONS:
            if name == "nvfp4_prefill" and not self.fast_nvfp4:
                continue
            nat.set_option(name, value)
        nat.flash_attn = dict(FAST_ENGINE_OPTIONS)["flash_attn"]   # copy decoding keeps its passes below the flash kernels
        if self.backend == "cuda" and hasattr(lib, "ink2_step_top1"):
            nat.set_option("device_greedy", 1)
            nat.device_greedy = self.device_greedy = True
        nat.draft_fast = True

    def _load_options(self) -> dict:
        kw = self._rollback_kw()
        if self.fast:
            parameters = inspect.signature(self.mod.Ink2Native.load).parameters
            for name, value in FAST_LOAD_OPTIONS:
                if name not in parameters:
                    raise YasakaniError("the fast path needs the binding of 0.3.3 or later (%s)" % self.binding_path)
                kw[name] = value
        return kw

    def _rollback_kw(self) -> dict:
        """{"rollback": n} for the binding's load (DEFAULT_ROLLBACK, or $YASAKANI_INK2_ROLLBACK; 0 = the engine's default)."""
        v = (os.environ.get(ROLLBACK_ENV) or "").strip()
        try:
            n = int(v) if v else min(DEFAULT_ROLLBACK, self.max_ctx)
        except ValueError:
            n = -1
        if not 0 <= n <= self.max_ctx:
            raise YasakaniError("%s must be a number from 0 to %d, not %r" % (ROLLBACK_ENV, self.max_ctx, v))
        try:
            parameters = inspect.signature(self.mod.Ink2Native.load).parameters
        except (TypeError, ValueError):
            parameters = {}
        return {"rollback": n} if n and "rollback" in parameters else {}

    # -- drafter ------------------------------------------------------------------------------------------------------------------
    def set_drafter(self, pack: "Ink2Pack", mode: str = "auto", depth: Optional[int] = None) -> str:
        """Load the model folder's drafter through the engine wrapper (mode auto or on) -> the line for the start-up output.
        Without a drafter in the folder, or with a wrapper that has no drafter calls, generation goes on without drafting; with
        ``mode == "on"`` a drafter that is there but cannot be loaded is an error."""
        self.drafting, self.draft_mode, self.draft_depth = False, "off", 0
        if mode == "off":
            note = "drafter: off (--draft off)"
        elif pack.drafter is None:
            note = "drafter: %s; generation without drafting" % (pack.drafter_problem or "none in this model folder")
        elif not all(callable(getattr(self.native, n, None)) for n in DRAFT_CALLS):
            note = "drafter: drafting is not available with this engine library (no drafter calls); generation without drafting"
        elif pack.drafter_format == "yskn" and not self.takes_converted_drafter():
            problem = ("%s is a converted drafter file, which this engine library cannot multiply (it needs a yskn_inkling2.dll of "
                       "0.3.1 or newer)" % pack.drafter.name)
            if mode == "on":
                raise YasakaniError("the drafter could not be loaded: " + problem)
            note = "drafter: %s; generation without drafting" % problem
        else:
            try:
                self.native.draft_load(str(pack.drafter))
                st = self.native.draft_mode(mode, depth)
            except (RuntimeError, OSError, ValueError) as ex:
                try:
                    if callable(getattr(self.native, "draft_loaded", None)) and self.native.draft_loaded():
                        self.native.draft_mode("off")
                    self.reset()
                except RuntimeError:
                    pass
                if mode == "on":
                    raise YasakaniError("the drafter %s could not be loaded: %s" % (pack.drafter.name, ex)) from ex
                note = "drafter: %s could not be loaded (%s); generation without drafting" % (pack.drafter.name, ex)
            else:
                self.drafting, self.draft_mode, self.draft_depth = True, str(st.get("mode", mode)), int(st.get("depth") or 0)
                self.reset()                                 # the drafter's context starts with the next prompt
                note = "drafter: %s loaded%s, mode %s, depth %d%s" % (
                    pack.drafter.name, self._drafter_form(pack), self.draft_mode, self.draft_depth,
                    " (used only while it makes generation faster)" if self.draft_mode == "auto" else "")
        self.draft_note = note
        return note

    def takes_converted_drafter(self) -> bool:
        """The engine library multiplies the matrices of a converted drafter file as stored (capability bit 4, drafter
        interface version 2 or newer)."""
        lib = getattr(self.native, "lib", None)
        if lib is None or not int(getattr(lib, "caps", 0) or 0) & CAP_DRAFTER_PACKED:
            return False
        try:
            return int(lib.ink2_draft_abi()) >= 2
        except (AttributeError, OSError):
            return False

    def _drafter_form(self, pack: "Ink2Pack") -> str:
        """" (converted, scheme d4, 30 matrices multiplied as stored)" for a converted drafter file, else ""."""
        if pack.drafter_format != "yskn":
            return ""
        n = 0
        try:
            n = int(self.native.draft_packed()) if callable(getattr(self.native, "draft_packed", None)) else 0
        except RuntimeError:
            pass
        return " (converted, scheme %s, %d matrices multiplied as stored)" % (pack.drafter_entry.get("scheme") or "?", n)

    # -- image input --------------------------------------------------------------------------------------------------------
    def set_vision(self, pack: "Ink2Pack", mode: Optional[str] = None, max_image_tokens=None, max_request_tokens=None,
                   cache_mib=None, log=None) -> str:
        """Load the folder's image tower (mode auto or on; None: $YASAKANI_INK2_VISION, else auto) -> the line for the start-up
        output.  auto loads it when the folder has the tower and the library has image input (cuda backend); otherwise text works
        as before and ``vision_problem`` says why images are refused.  ``on``: anything missing is a YasakaniError.  Feature rows
        = the per-request limit (at most the context).  ``cache_mib``: host cache of image vectors (None: 512 MiB; 0: none)."""
        from . import inkling2_vision as V
        self.vision = None
        mode = vision_mode_of(mode)
        cap, req = vision_limits(max_image_tokens, max_request_tokens)
        req = max(cap, min(req, self.max_ctx))

        def off(problem: str) -> str:
            if mode == "on":
                raise YasakaniError("image input could not be turned on: %s" % problem)
            self.vision_problem = problem
            self.vision_note = "image input: off (%s)" % ("--vision off" if problem == VISION_OFF else problem)
            return self.vision_note

        if mode == "off":
            return off(VISION_OFF)
        if not pack.has_vision:
            return off(V.NO_TOWER)
        if self.backend != "cuda":
            return off(VISION_NO_GPU)
        has = getattr(self.native, "has_vision", None)
        if not callable(has) or not has():
            return off(VISION_NO_LIBRARY)
        t0 = time.perf_counter()
        try:
            vw = V.vision_weights(pack.root)
            if vw is None:
                return off(V.NO_TOWER)
            memory = tuple(int(x) for x in self.native.vision_load(vw, cap, req, log=log))
        except V.VisionError as ex:
            return off(str(ex))
        except (RuntimeError, OSError, ValueError) as ex:
            return off("the image tower could not be loaded: %s" % ex)
        cache_bytes = V.MEMORY_CACHE_BYTES if cache_mib is None else max(0, int(cache_mib)) * (1 << 20)
        self.vision = Ink2VisionState(kind=vw.kind, max_image_tokens=cap, max_request_tokens=req, memory=memory,
                                      cache=V.FeatureCache(cache_bytes), load_seconds=time.perf_counter() - t0)
        self.vision_problem = ""
        mib = lambda n: (n + (1 << 20) - 1) >> 20
        self.vision_note = ("image input: on (image tower from %s, %d MiB, loaded in %.1f s; scratch %d MiB for %s image tokens per "
                            "image; feature buffer %d MiB for %s per request; host cache %d MiB)"
                            % (vw.path.name, mib(memory[0]), self.vision.load_seconds, mib(memory[1]), format(cap, ","),
                               mib(memory[2]), format(req, ","), cache_bytes >> 20))
        return self.vision_note

    # -- copy decoding ------------------------------------------------------------------------------------------------------
    def set_copy(self, mode: Optional[str] = None, min_match=None) -> str:
        """Copy decoding (--copy / $YASAKANI_INK2_COPY): "auto" proposes ids copied from the context (edits) through spec_step; the
        output does not change.  -> the line for the start-up output."""
        from . import _ink2_copyedit as ce
        try:
            mode, n = ce.mode_of(mode), ce.min_of(min_match)
        except ValueError as ex:
            raise YasakaniError(str(ex)) from ex
        self.copying = False
        self.copy_mode, self.copy_min = mode, n
        if mode == "off":
            ce.enable(self.native, "off")
            self.copy_note = "copy: off"
            return self.copy_note
        try:
            st = ce.enable(self.native, mode, n)
        except (RuntimeError, AttributeError) as ex:
            self.copy_note = "copy: not available with this engine library (%s)" % ex
            return self.copy_note
        self.copying = True
        self.copy_note = "copy: auto (matches of %d ids or more, passes up to %d ids; used only while it pays)" % (n, max(st["ladder"]))
        return self.copy_note

    def copy_stats(self) -> dict:
        if not self.copying:
            return {}
        from . import _ink2_copyedit as ce
        return ce.stats(self.native)

    def can_spec(self) -> bool:
        """A pass of spec_step fits: drafting, or copy decoding with room for the anchor."""
        return self.can_draft() or (self.copying and self.fed is not None and len(self.fed) + 1 <= self.max_ctx)

    def can_draft(self) -> bool:
        """A drafting pass fits: drafter loaded and room for the anchor and every proposed token in the context."""
        return self.drafting and self.fed is not None and len(self.fed) + 1 + self.draft_depth <= self.max_ctx

    def spec_step(self, anchor: int, pick=None) -> Tuple[List[int], int]:
        """Feed ``anchor`` and check the drafter's proposals in one pass -> (accepted tokens, next token; not fed yet)."""
        try:
            accepted, nxt, _ = self.native.spec_step(int(anchor), pick)
        except BaseException as ex:
            self.fed = None                                  # state unknown: the next prompt starts from a reset
            if isinstance(ex, RuntimeError):
                raise YasakaniError("engine step failed: %s" % ex) from ex
            raise
        accepted = [int(a) for a in accepted]
        self.fed.append(int(anchor))
        self.fed.extend(accepted)
        self.last_logits = None
        return accepted, int(nxt)

    def truncate(self, n: int) -> None:
        """Drop the positions from ``n`` on (the rest of a burst that was not emitted)."""
        if self.fed is None or n >= len(self.fed):
            return
        try:
            self.native.truncate(int(n))
        except RuntimeError as ex:
            self.fed = None
            raise YasakaniError("engine truncate failed: %s" % ex) from ex
        del self.fed[n:]
        self.last_logits = None

    def draft_stats(self) -> dict:
        if not self.drafting:
            return {}
        try:
            return dict(self.native.spec_stats())
        except RuntimeError:
            return {}

    def _load_yskn(self, pack: "Ink2Pack", lib, log):
        """Check the tensors of the .yskn file, then let the binding load it through a CheckpointView: the binding opens its weights
        with ``ink2_ckpt.Checkpoint(path)``, which is pointed at the view for this one path while ``load`` runs."""
        from . import ink2_yskn as Y
        t0 = time.perf_counter()
        yf = Y.YsknFile(pack.weights)
        self._yskn = yf
        if os.environ.get("YASAKANI_INK2_SKIP_TENSOR_CHECK") != "1":
            bad = yf.check(threads=8)
            if bad:
                raise ValueError("%s is damaged: the CRC-32 of %d tensor part(s) differs, first %s (%s); convert the folder again"
                                 % (pack.weights.name, len(bad), bad[0][0], bad[0][1]))
        t_check = time.perf_counter() - t0
        # packed dense matrices only when the engine says it takes them AND the binding says it hands them over
        packed = Y.engine_takes_packed_dense(lib) and bool(getattr(self.mod, "TAKES_PACKED_DENSE", False))
        want = (os.environ.get("YASAKANI_INK2_DENSE") or "auto").strip().lower()
        if want not in ("auto", "packed", "bf16"):
            raise ValueError("YASAKANI_INK2_DENSE must be auto, packed or bf16, not %r" % want)
        if want == "packed" and not packed:
            raise ValueError("YASAKANI_INK2_DENSE=packed, but this engine library or its binding does not take packed dense matrices"
                             " (engine interface version 3 with the packed-dense capability is needed)")
        if want == "bf16":
            packed = False
        view = Y.CheckpointView(yf, dense="packed" if packed else "bf16")
        kw = dict(backend=self.backend, max_positions=self.max_ctx, dll=str(self.dll), weights=pack.weights.name, log=log,
                  **self._load_options())
        try:
            takes_reader = "checkpoint" in inspect.signature(self.mod.Ink2Native.load).parameters
        except (TypeError, ValueError):
            takes_reader = False
        if takes_reader:                                     # a binding that accepts a reader object
            native = self.mod.Ink2Native().load(str(pack.root), checkpoint=view, **kw)
        else:                                                # else: its ink2_ckpt.Checkpoint gives the view for this one path
            ckpt_mod = getattr(self.mod, "ink2_ckpt", None)
            if ckpt_mod is None or not hasattr(ckpt_mod, "Checkpoint"):
                raise RuntimeError("this binding has no ink2_ckpt.Checkpoint to give the .yskn file to")
            original = ckpt_mod.Checkpoint
            target = os.path.normcase(os.path.abspath(str(pack.weights)))

            def checkpoint(path, *a, **k):
                if os.path.normcase(os.path.abspath(str(path))) == target:
                    return view
                return original(path, *a, **k)

            ckpt_mod.Checkpoint = checkpoint
            try:
                native = self.mod.Ink2Native().load(str(pack.root), **kw)
            finally:
                ckpt_mod.Checkpoint = original
        n_packed = int(getattr(native, "packed_dense", 0) or 0)
        if packed and n_packed == 0 and any(yf.entry(n)["encoding"] in ("nvfp4", "mxfp8") and yf.entry(n).get("role") != "experts"
                                            for n in yf.tensors):
            raise RuntimeError("the engine was given packed dense matrices but reports none held packed")
        self.weights_note = "yskn %s; tensor check %.1f s; dense matrices %s" % (
            (pack.meta.get("weights") or {}).get("scheme", "?"), t_check,
            ("packed (%d multiplied as stored)" % n_packed) if packed else "decoded to bfloat16")
        return native

    # -- state --------------------------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.native.reset()
        self.fed = []
        self.last_logits = None

    def cancel(self) -> None:
        self._cancel.set()

    def step(self, token: int) -> np.ndarray:
        if len(self.fed) >= self.max_ctx:
            raise YasakaniError("the context of %d positions is full" % self.max_ctx)
        try:
            logits = self.native.step(int(token))
        except RuntimeError as ex:
            self.fed = None                                  # state unknown: the next prompt starts from a reset
            raise YasakaniError("engine step failed: %s" % ex) from ex
        self.fed.append(int(token))
        self.last_logits = logits
        return logits

    def step_top1(self, token: int) -> int:
        if len(self.fed) >= self.max_ctx:
            raise YasakaniError("the context of %d positions is full" % self.max_ctx)
        try:
            result = self.native.step_top1(token)
        except RuntimeError as ex:
            self.fed = None
            raise YasakaniError("engine step failed: %s" % ex) from ex
        self.fed.append(int(token))
        self.last_logits = None
        return result

    def read_prompt(self, ids: Sequence[int]) -> Tuple[np.ndarray, int]:
        """Read the prompt; -> (logits after its last id, ids that did not have to be read again).

        The ids already read are kept as far as the new prompt starts with them: when the prompt goes on from all of them (the
        next turn of a chat) only the new ids are read; when it shares only a first part, the positions after that part are dropped
        (``truncate``, up to the engine's rollback) and the rest is read.  All ids but the last are read in passes of several rows
        (``feed_many``), the last one in a step that gives the logits."""
        images = tuple(getattr(ids, "images", None) or ())
        ids = [int(i) for i in ids]
        if images:
            return self._read_prompt_images(ids, images)
        if not ids:
            self.reset()
            return None, 0
        if len(ids) > self.max_ctx:
            raise YasakaniError("the context of %d positions is full" % self.max_ctx)
        fed = self.fed
        common = 0
        if fed is not None:
            limit = min(len(fed), len(ids) - 1)              # the last id is always read: its logits are needed
            while common < limit and fed[common] == ids[common]:
                common += 1
        try:
            if fed is None or common == 0:
                self.reset()
                common = 0
            elif common < len(fed):
                if len(fed) - common <= int(getattr(self.native, "rollback", 0)):
                    self.native.truncate(common)
                else:                                        # too far back for the caches: read again from the start
                    self.reset()
                    common = 0
            rest = ids[common:]
            self.fed = None                                  # state unknown until the reading ends
            self.last_logits = None
            if len(rest) > 1:
                self.native.feed_many(rest[:-1])
            logits = self.native.step(rest[-1])
        except (RuntimeError, ValueError) as ex:
            self.fed = None                                  # the next prompt starts from a reset
            raise YasakaniError("engine step failed: %s" % ex) from ex
        self.fed = ids
        self.last_logits = logits
        return logits, common

    def _read_prompt_images(self, ids: List[int], images) -> Tuple[np.ndarray, int]:
        """read_prompt of a prompt with images (PromptIds.images).  A position is compared as its id, an image position as (id,
        image key, index in the image): the ids already read are kept as far as the same text and the same images are at the
        same places.  The images with positions to read get feature rows from row 0 on, in prompt order; their vectors come
        from the host cache or from the tower (then kept in the cache).  The positions are read with ink2_feed_many_img; the
        last id (text: the end of a message) in a step.  Truncation and reset as for text (the engine's rollback; deeper: the
        prompt is read again from the start, images included)."""
        st = self.vision
        if st is None:
            raise YasakaniError("image input: %s" % self.vision_problem)
        if len(ids) > self.max_ctx:
            raise YasakaniError("the context of %d positions is full" % self.max_ctx)
        pos: List[object] = list(ids)
        owner: Dict[int, int] = {}
        for n, (start, im) in enumerate(images):
            start, cnt = int(start), int(im.tokens)
            if start < 0 or start + cnt > len(ids) - 1 or any(ids[start + k] != IMAGE_TOKEN for k in range(cnt)):
                raise YasakaniError("the prompt does not hold image %d at positions %d to %d (or ends inside it)" % (n + 1, start,
                                                                                                                start + cnt - 1))
            for k in range(cnt):
                if start + k in owner:
                    raise YasakaniError("images %d and %d of the prompt overlap" % (owner[start + k] + 1, n + 1))
                pos[start + k] = (IMAGE_TOKEN, im.key, k)
                owner[start + k] = n
        fed = self.fed
        common = 0
        if fed is not None:
            limit = min(len(fed), len(pos) - 1)              # the last id is always read: its logits are needed
            while common < limit and fed[common] == pos[common]:
                common += 1
        stats = dict(images=len(images), image_tokens=sum(int(im.tokens) for _, im in images), encoded=0, from_cache=0, reused=0,
                     encode_ms=0.0, upload_ms=0.0)
        try:
            if fed is None or common == 0:
                self.reset()
                common = 0
            elif common < len(fed):
                if len(fed) - common <= int(getattr(self.native, "rollback", 0)):
                    self.native.truncate(common)
                else:                                        # too far back for the caches: read again from the start
                    self.reset()
                    common = 0
        except (RuntimeError, ValueError) as ex:
            self.fed = None
            raise YasakaniError("engine step failed: %s" % ex) from ex
        need = [n for n, (start, im) in enumerate(images) if int(start) + int(im.tokens) > common]
        stats["reused"] = len(images) - len(need)
        row_of: Dict[int, int] = {}
        r = 0
        for n in need:
            row_of[n] = r
            r += int(images[n][1].tokens)
        self.fed = None                                      # state unknown until the reading ends (truncated or reset above)
        if r > st.max_request_tokens:
            raise YasakaniError("the images of this prompt need %d feature rows; the engine holds %d (the image token limit per "
                                "request)" % (r, st.max_request_tokens))
        self.last_logits = None
        self.last_vision = {}
        for n in need:
            im, r0 = images[n][1], row_of[n]
            t0 = time.perf_counter()
            feats = st.cache.get(im.key, int(im.tokens)) if st.cache is not None else None
            try:
                if feats is not None:
                    self.native.vision_set_rows(feats, r0)
                    stats["from_cache"] += 1
                    stats["upload_ms"] += (time.perf_counter() - t0) * 1000.0
                    continue
                self.native.vision_encode(im, r0)
                stats["encoded"] += 1
                stats["encode_ms"] += (time.perf_counter() - t0) * 1000.0
                if st.cache is not None and st.cache.max_bytes:
                    st.cache.put(im.key, self.native.vision_rows_bf16(r0, int(im.tokens)))
            except (RuntimeError, ValueError) as ex:
                raise YasakaniError("image encode failed: %s" % ex) from ex
        rest = ids[common:]
        rows = [-1] * (len(rest) - 1)
        for j in range(len(rest) - 1):
            n = owner.get(common + j)
            if n is not None:
                rows[j] = row_of[n] + (common + j - int(images[n][0]))
        try:
            if len(rest) > 1:
                if any(x >= 0 for x in rows):
                    self.native.feed_many_img(rest[:-1], rows)
                else:
                    self.native.feed_many(rest[:-1])
            logits = self.native.step(rest[-1])
        except (RuntimeError, ValueError) as ex:
            raise YasakaniError("engine step failed: %s" % ex) from ex
        self.fed = pos
        self.last_logits = logits
        self.last_vision = stats
        return logits, common

    # -- information ----------------------------------------------------------------------------------------------------------------
    def memory_used(self) -> int:
        return int(self.native.memory_used())

    def device_memory(self):
        return self.native.device_memory()

    def close(self) -> None:
        n = getattr(self, "native", None)
        if n is not None:
            n.close()
            self.native = None
        y = getattr(self, "_yskn", None)
        if y is not None:
            if n is not None:
                n.ckpt = None
            y.close()
            self._yskn = None


def bytes_per_position(config: dict, mxfp8: bool = False) -> int:
    """Key/value cache of the full-attention layer visits per position, from config.json: float32, or with ``mxfp8`` (the fast path)
    one byte per value and one scale byte per 32 values."""
    t = config.get("text_config") or config
    n = int(t.get("num_hidden_layers", 24))
    types = t.get("layer_types") or ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(n)]
    loop = t.get("loop_config") or {}
    a, b, passes = loop.get("core_start", 6), loop.get("core_end", 18), loop.get("num_passes", 3)
    visits = sum((passes if a <= i < b else 1) for i, ty in enumerate(types) if ty != "hybrid_sliding")
    values = visits * 2 * int(t.get("num_key_value_heads", 8)) * int(t.get("head_dim", 128))
    return values + values // 32 if mxfp8 else values * 4


# ---- generation loop -----------------------------------------------------------------------------------------------------------------
END, DROP = "end", "drop"        # on_token results: the reply ends with this token / ends before it (the token is not emitted)
NAME = "name"                    # Ink2Runner: after the next <|message_model|> the model writes the name of an offered function


def sampler_picks(sample: Sampler):
    """k -> a pick function for the rows of a pass whose first row gives output token k; one draw per token (Sampler.seek)."""
    def at(k: int):
        n = [int(k)]

        def pick(logits) -> int:
            sample.seek(n[0])
            n[0] += 1
            return sample(logits)
        return pick
    return at


def masked(logits: np.ndarray, ids) -> np.ndarray:
    """The logits with ``ids`` made impossible, or with an ``Allow`` every id but those (a copy; the engine's array is not changed)."""
    if isinstance(ids, Allow):
        keep = np.fromiter(ids, np.int64)
        x = np.full(np.shape(logits), -np.inf, np.float32)
        x[keep] = np.asarray(logits, np.float32)[keep]
        return x
    x = np.array(logits, np.float32, copy=True)
    x[list(ids)] = -np.inf
    return x


def generate(eng: Ink2Engine, first: int, on_token, pick_at=None, ban=None) -> int:
    """Emit tokens from ``first`` (chosen from the logits of the prompt) on, with the drafter when the engine has one.

    ``on_token(tok)`` emits a token and returns None (go on), END (the reply ends with it), DROP (the reply ends before it; it is
    not emitted) or a list of tokens that must come next (the thinking budget).  ``pick_at``: None = greedy, else ``k -> pick``
    (sampler_picks).  The engine state always matches what was emitted: every emitted token is fed except the last one, as in
    plain stepping, and the rest of a burst after the end is truncated away.  ``ban``: None, or a function that returns the ids that
    may not be picked next (asked after each on_token; while it returns some, the next token comes from a plain step whose logits
    have them masked, or with an ``Allow`` all other ids masked, not from a drafting pass; inside a drafting pass's accepted ids too:
    the rest of that pass is dropped).  Returns the passes of the model (the prompt's last pass counts as the one that
    gave ``first``)."""
    k, tok, forced, passes = 0, int(first), [], 1
    while True:
        r = on_token(tok)
        if r is END or r is DROP:
            return passes
        k += 1
        if r:
            forced = list(r)
        banned = ban() if ban is not None else None
        if not forced and not banned and (eng.can_spec() if getattr(eng, "copying", False) else eng.can_draft()):
            base = len(eng.fed)
            accepted, nxt = eng.spec_step(tok, pick_at(k) if pick_at else None)
            passes += 1
            for i, a in enumerate(accepted):
                r = on_token(a)
                if r is END or r is DROP:                    # the anchor and accepted[:i] stay fed
                    eng.truncate(base + 1 + i)
                    return passes
                k += 1
                if r:                                        # forced tokens after accepted[i]: it stays fed
                    eng.truncate(base + 2 + i)
                    forced = list(r)
                    break
                banned = ban() if ban is not None else None
                if banned:                                   # a mask from the next id on: the rest of the burst is not used,
                    eng.truncate(base + 1 + i)               # accepted[i] is emitted and not fed (a plain step from it below)
                    tok = a
                    break
            else:
                tok = nxt
                continue
            if forced:
                tok = forced.pop(0)
                continue
        if len(eng.fed) >= eng.max_ctx:
            return passes
        if getattr(eng, "device_greedy", False) and not forced and not banned and pick_at is None:
            tok = eng.step_top1(tok)
            passes += 1
            continue
        logits = eng.step(tok)
        passes += 1
        if forced:
            tok = forced.pop(0)
        else:
            if banned:
                logits = masked(logits, banned)
            tok = pick_at(k)(logits) if pick_at else int(np.argmax(logits))


def _spec_delta(a: dict, b: dict) -> Tuple[int, int]:
    return (int(b.get("proposed", 0)) - int(a.get("proposed", 0)), int(b.get("accepted", 0)) - int(a.get("accepted", 0)))


# ---- chat runner --------------------------------------------------------------------------------------------------------------------
IMAGE_PARTS = ("image", "input_image", "image_url")       # content parts that are images (the template's names)
VIDEO_PARTS = ("video", "input_video", "video_url")
PREPARED_KEEP = 8                                           # images read and prepared, kept by the hash of their file bytes


def _part_kind(q) -> str:
    t = q.get("type") if isinstance(q, dict) else None
    return "image" if t in IMAGE_PARTS else ("video" if t in VIDEO_PARTS else "")


class Ink2Runner:
    """Chat requests on one Inkling-2 engine, one at a time; the same calls as yasakani.chat.ChatRunner (prepare_checked, run,
    close), so the server and the terminal chat use it unchanged.  ``vision``: the engine's Ink2VisionState when it reads
    images, else None; ``vision_allow_local_files``: False (data URIs and bytes only), True, or the folder local image files
    must be in."""

    vision_allow_local_files = False

    @property
    def vision(self):
        return getattr(self.engine, "vision", None)

    def __init__(self, engine: Ink2Engine, fmt: Ink2Formatter, pack: Ink2Pack, seed0: int = 1):
        self.engine = engine
        self.fmt = fmt
        self.pack = pack
        self._seed = int(seed0)
        self._seed_lock = threading.Lock()
        self.stop_ids = frozenset(pack.stop_ids or [fmt.ids[T_STOP]])
        self._prepared: Dict[Tuple[str, int], object] = {}

    def next_seed(self) -> int:
        with self._seed_lock:
            s = self._seed
            self._seed += 1
            return s

    def warmup(self, say=None) -> None:
        ids, _ = self.fmt.prompt([{"role": "user", "content": "warm-up"}], {"enable_thinking": False})
        logits, _ = self.engine.read_prompt(ids)
        for _ in range(4):
            logits = self.engine.step(int(np.argmax(logits)))
        tok = int(np.argmax(logits))
        for _ in range(2):                                   # the drafting pass too (a CUDA graph is recorded on first use)
            if not self.engine.can_draft():
                break
            _, tok = self.engine.spec_step(tok)
        self.engine.reset()

    def prepare(self, p):
        """-> (prompt ids, the reply starts inside the answer) for a request without tools, as before; with ``p.tools``
        (prompt ids, start, plan) where plan says which call must be made and which ids are masked first (Ink2Formatter.tool_prompt)."""
        from .chat import RequestError
        if any(isinstance(m.get("content"), list) and any(_part_kind(q) for q in m["content"]) for m in p.messages):
            return self._prepare_images(p)
        for i, m in enumerate(p.messages):
            if isinstance(m.get("content"), list):
                if any(isinstance(q, dict) and q.get("type") not in ("text", None) for q in m["content"]):
                    raise RequestError("messages[%d]: only text content is supported for this model" % i)
                m = dict(m, content="".join(str(q.get("text", "")) for q in m["content"] if isinstance(q, dict)))
                p.messages[i] = m
            if m.get("tool_calls"):                          # the template reads the arguments as an object, not as JSON text
                calls = []
                for k, c in enumerate(m["tool_calls"]):
                    f = c.get("function") if isinstance(c, dict) else None
                    args = f.get("arguments") if isinstance(f, dict) else None
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except ValueError:
                            raise RequestError("messages[%d].tool_calls[%d]: arguments is not valid JSON" % (i, k))
                        c = dict(c, function=dict(f, arguments=args))
                    calls.append(c)
                p.messages[i] = m = dict(m, tool_calls=calls)
        try:
            from jinja2.exceptions import TemplateError
            if not p.tools:
                return self.fmt.prompt(p.messages, p.template_kwargs)
            call = None
            if getattr(p, "tool_choice", None) == "required":
                call = p.tools[0]["function"]["name"] if len(p.tools) == 1 else ""
            ids, start, ban_first = self.fmt.tool_prompt(p.messages, p.tools, p.template_kwargs, call)
            trie = self.fmt.name_trie(p.tools) if call == "" else None
            return ids, start, dict(call=call, ban_first=ban_first, trie=trie)
        except TemplateError as ex:
            raise RequestError("the chat template does not accept these messages: %s" % ex)
        except YasakaniError as ex:
            raise RequestError(str(ex))

    def _prepare_images(self, p):
        """prepare() of a request with image parts: the messages are rendered with their parts in order (an image part becomes
        the template's placeholder), every image is read, checked and, above the per-image cap, shrunk; then each placeholder
        is replaced by the image's rows x cols ids.  Nothing is encoded here.  -> as prepare, the ids being a PromptIds."""
        from .chat import RequestError
        from . import inkling2_vision as V
        messages, sources = [], []
        for i, m in enumerate(p.messages):
            content = m.get("content")
            if isinstance(content, list) and any(_part_kind(q) for q in content):
                if any(_part_kind(q) == "video" for q in content):
                    raise RequestError("messages[%d]: video is not supported (only still images)" % i)
                if m.get("role") not in ("user", "tool"):
                    raise RequestError("messages[%d]: images are supported only in user and tool messages" % i)
                parts = []
                for q in content:
                    if _part_kind(q) == "image":
                        sources.append((i, q))
                        parts.append({"type": "image"})
                    elif isinstance(q, str):
                        parts.append({"type": "text", "text": q})
                    elif isinstance(q, dict) and q.get("type") in ("text", "input_text", None):
                        parts.append({"type": "text", "text": str(q.get("text", ""))})
                    else:
                        raise RequestError("messages[%d]: content part %s is not supported for this model"
                                           % (i, repr(q.get("type") if isinstance(q, dict) else type(q).__name__)[:60]))
                m = dict(m, content=parts)
            elif isinstance(content, list):
                if any(isinstance(q, dict) and q.get("type") not in ("text", None) for q in content):
                    raise RequestError("messages[%d]: only text content is supported for this model" % i)
                m = dict(m, content="".join(str(q.get("text", "")) for q in content if isinstance(q, dict)))
            if m.get("tool_calls"):
                calls = []
                for k, c in enumerate(m["tool_calls"]):
                    f = c.get("function") if isinstance(c, dict) else None
                    args = f.get("arguments") if isinstance(f, dict) else None
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except ValueError:
                            raise RequestError("messages[%d].tool_calls[%d]: arguments is not valid JSON" % (i, k))
                        c = dict(c, function=dict(f, arguments=args))
                    calls.append(c)
                m = dict(m, tool_calls=calls)
            messages.append(m)
        st = self.vision
        if st is None:
            raise RequestError("image input is not available: %s" % getattr(self.engine, "vision_problem", V.NO_TOWER))
        if len(sources) > V.MAX_IMAGES:
            raise RequestError("image input: at most %d images are supported per request (got %d)" % (V.MAX_IMAGES, len(sources)))
        images = []
        for i, q in sources:
            try:
                images.append(self._read_image(q, st.max_image_tokens))
            except V.VisionError as ex:
                raise RequestError("messages[%d]: image input: %s" % (i, ex)) from None
            except ImportError as ex:
                raise YasakaniError(str(ex)) from ex
        try:
            total = V.check_request(images, st.max_image_tokens, st.max_request_tokens)
        except V.VisionError as ex:
            raise RequestError("image input: %s" % ex) from None
        try:
            from jinja2.exceptions import TemplateError
            if not p.tools:
                out = self.fmt.prompt(messages, p.template_kwargs)
            else:
                call = None
                if getattr(p, "tool_choice", None) == "required":
                    call = p.tools[0]["function"]["name"] if len(p.tools) == 1 else ""
                ids, start, ban_first = self.fmt.tool_prompt(messages, p.tools, p.template_kwargs, call)
                trie = self.fmt.name_trie(p.tools) if call == "" else None
                out = (ids, start, dict(call=call, ban_first=ban_first, trie=trie))
        except TemplateError as ex:
            raise RequestError("the chat template does not accept these messages: %s" % ex)
        except YasakaniError as ex:
            raise RequestError(str(ex))
        ids = out[0]
        if sum(1 for t in ids if t == IMAGE_TOKEN) != len(images):
            raise RequestError("the text of the messages contains the image placeholder <|unused_200054|>, which is reserved for "
                               "images")
        expanded = PromptIds()
        placed = []
        k = 0
        for t in ids:
            if t == IMAGE_TOKEN:
                im = images[k]
                k += 1
                placed.append((len(expanded), im))
                expanded.extend([IMAGE_TOKEN] * int(im.tokens))
            else:
                expanded.append(int(t))
        expanded.images = tuple(placed)
        expanded.image_tokens = total
        return (expanded,) + tuple(out[1:])

    def _read_image(self, q: dict, cap: int):
        """One image part -> inkling2_vision.PreparedImage (shrunk to ``cap`` tokens when it has more)."""
        from . import inkling2_vision as V
        t = q.get("type")
        x = q.get("image_url") if t == "image_url" else (q.get("image_url", q.get("image")) if t == "input_image"
                                                          else q.get("image", q.get("image_url")))
        if isinstance(x, dict):
            x = x.get("url")
        if isinstance(x, V.PreparedImage):
            return x if x.tokens <= cap else V.prepare_rgb(x.rgb, cap)
        if isinstance(x, (bytes, bytearray, memoryview)):
            data = bytes(x)
        elif isinstance(x, str) and x:
            data = V.source_bytes(x, allow_local_files=self.vision_allow_local_files)
        else:
            raise V.VisionError("an image part needs its image: a data:image/...;base64 URI, file bytes, or a local path (with %s)"
                                % V.LOCAL_FILES_HINT)
        key = (hashlib.sha256(data).hexdigest(), int(cap))
        im = self._prepared.pop(key, None)
        if im is None:
            im = V.prepare_image(data, cap)
        self._prepared[key] = im                             # the newest last; the oldest goes first
        while len(self._prepared) > PREPARED_KEEP:
            self._prepared.pop(next(iter(self._prepared)))
        return im

    def prepare_checked(self, p):
        from .chat import context_length_error
        prepared = self.prepare(p)
        if len(prepared[0]) >= self.engine.max_ctx - CONTEXT_MARGIN:
            images = getattr(prepared[0], "images", ())
            if images:
                from .chat import ContextLengthError
                raise ContextLengthError("the input is too long: %d tokens (%d of them image tokens of %d image%s) exceeds the "
                                         "model's maximum context length of %d tokens (input, thinking and answer together)"
                                         % (len(prepared[0]), prepared[0].image_tokens, len(images), "" if len(images) == 1 else "s",
                                            self.engine.max_ctx))
            raise context_length_error(len(prepared[0]), self.engine.max_ctx)
        return prepared

    def close(self) -> None:
        pass

    def run(self, p, emit: Callable[[str, str], None], prepared=None, eng=None):
        """Generate the reply, calling ``emit("reasoning" | "content", text)``; returns yasakani.chat.Usage."""
        from .chat import StopFilter, Usage
        eng = self.engine
        prepared = prepared if prepared is not None else self.prepare_checked(p)
        prompt, in_answer = prepared[0], prepared[1]
        plan = prepared[2] if len(prepared) > 2 else None    # a request with tools (prepare)
        eng._cancel.clear()
        seed = p.seed if p.seed is not None else self.next_seed()
        sample = Sampler(p.temperature, p.top_p, p.top_k, p.min_p, seed)
        stops = frozenset() if p.ignore_eos else self.stop_ids
        max_new = max(1, int(p.max_tokens))
        usage = Usage(prompt_tokens=len(prompt))
        ids = self.fmt.ids
        if plan is None:
            stream = OutputStream(self.fmt, in_answer)
        else:
            stream = OutputStream(self.fmt, tools=True, start=in_answer, call_name=plan["call"] or "",
                                  pin_names=[t["function"]["name"] for t in p.tools] if plan["call"] is not None else ())
        stopf = StopFilter(p.stop)
        budget = max(0, int(getattr(p, "thinking_budget", 0) or 0))
        ban_state = dict(pending=None, now=None)             # the mask of the pick after a forced <|message_model|>, and of the next
        trie = None                                          # tool_choice required / a named function: the names a call may take
        name = dict(node=None)                               # the node of the trie while the model writes the name of a call
        more = dict(stage=None)                              # after a call: "end" (its <|end_message|>), "head" (then <|message_model|>)
        if plan is not None and plan["call"] is not None:
            trie = plan.get("trie") or (self.fmt.name_trie(p.tools) if plan["call"] == "" else NameTrie(self.fmt, [plan["call"]]))
            if in_answer == "head":                          # thinking off, several: the prompt ends with <|message_model|>
                name["node"] = trie.root

        def deliver(events):
            for kind, text in events:
                if kind == "tool_call":
                    usage.tool_calls += 1
                    emit("tool_call", text)
                    continue
                if kind == "content":
                    text = stopf.push(text)
                if text:
                    emit(kind, text)

        t0 = time.perf_counter()
        logits, usage.cached_tokens = eng.read_prompt(prompt)
        t1 = time.perf_counter()
        usage.prefill_ms = usage.prompt_ms = (t1 - t0) * 1000.0
        if getattr(prompt, "images", None):
            usage.vision = dict(eng.last_vision)
        st0 = eng.draft_stats()
        pick_at = None if sample.t <= 0.0 else sampler_picks(sample)     # greedy: arg-max on the device, no logits copied
        if plan is not None and plan["ban_first"]:
            logits = masked(logits, plan["ban_first"])
        first = int(np.argmax(logits)) if pick_at is None else pick_at(0)(logits)
        state = dict(finish="length", n=0)
        force = [ids[T_END], ids[T_MODEL], ids[T_TEXT]]
        if plan is not None:                                 # after the thinking: a call (tool_choice required) or any block
            call = plan["call"]
            after_thinking = self.fmt.call_header(call) if call else [ids[T_MODEL]]
            after_ban = None if call else (NAME if call == "" else [ids[T_THINKING]])
            force = [ids[T_END]] + after_thinking

        def on_token(tok: int):
            if eng._cancel.is_set():
                return DROP
            if more["stage"] == "head":                      # after a call and <|message_model|>: another call, or the reply ends
                more["stage"] = None                         # (the template writes text and thinking before the calls, not after)
                if tok not in trie.root:
                    state["finish"] = "stop"
                    return DROP
            state["n"] += 1
            was = stream.mode
            ban_state["now"] = None
            if name["node"] is not None:                     # inside the name: the next id goes on with one of the names
                node = name["node"].get(tok) if tok != trie.end else None
                name["node"] = node
                if node is not None:
                    ban_state["now"] = trie.allowed(node)
            elif tok == ids[T_MODEL] and ban_state["pending"] is not None:
                if ban_state["pending"] is NAME:
                    name["node"] = trie.root
                    ban_state["now"] = trie.allowed(trie.root)
                else:
                    ban_state["now"] = ban_state["pending"]
                ban_state["pending"] = None
            deliver(stream.feed(tok))
            if more["stage"] == "end":
                more["stage"] = None
                if tok == ids[T_MODEL]:
                    more["stage"] = "head"
                    name["node"] = trie.root
                    ban_state["now"] = Allow(set(trie.root) | self.fmt.special)
            if tok in stops or stopf.hit:
                state["finish"] = "stop"
                return END
            if state["n"] >= max_new:
                return END
            if trie is not None and was == "call" and tok == ids[T_END]:
                if stream.calls:                             # the call is made: the turn ends (the template's stop token)
                    return [ids[T_STOP]]
                more["stage"] = "end"                        # it could not be read: only another call header or the stop token
                ban_state["now"] = Allow({ids[T_MODEL], ids[T_STOP]})
                return None
            if plan is not None and plan["call"] is not None and was == "reasoning" and tok == ids[T_END] and not stream.calls:
                ban_state["pending"] = after_ban             # tool_choice required: the thinking has ended, the call comes next
                return list(after_thinking)
            if plan is not None and plan["call"] is not None and was == "reasoning" and tok == ids[T_MODEL] and not stream.calls:
                if plan["call"]:                             # the thinking ended without <|end_message|>: the header goes on
                    return list(self.fmt.call_header(plan["call"])[1:])
                name["node"] = trie.root
                ban_state["now"] = trie.allowed(trie.root)
            if budget and stream.mode == "reasoning" and stream.answer_at is None and stream.n >= budget:
                if plan is not None:
                    ban_state["pending"] = after_ban
                return list(force)
            return None

        passes = generate(eng, first, on_token, pick_at, ban=(lambda: ban_state["now"]) if plan is not None else None)
        n_out, finish = state["n"], state["finish"]
        deliver(stream.flush())
        tail = stopf.flush()
        if tail:
            emit("content", tail)
        usage.gen_ms = (time.perf_counter() - t1) * 1000.0
        usage.completion_tokens = n_out
        usage.passes = n_out
        if eng.drafting:
            usage.passes = passes
            usage.draft_n, usage.draft_n_accepted = _spec_delta(st0, eng.draft_stats())
            usage.spec = dict(draft_mode=eng.draft_mode, draft_passes=passes,
                              tokens_per_pass=round(n_out / passes, 3) if passes else 0.0)
        usage.reasoning_tokens = stream.reasoning_count
        usage.finish = "tool_calls" if (finish == "stop" and usage.tool_calls) else finish
        return usage


def open_runner(pack: Ink2Pack, max_ctx: int, dll: Optional[str] = None, seed0: int = 1, say=None, warm: bool = True,
                draft: str = "auto", draft_depth: Optional[int] = None, copy: Optional[str] = None, copy_min=None,
                fast: Optional[str] = None, compute_cap=None, vision: Optional[str] = None, vision_max_image_tokens=None,
                vision_max_request_tokens=None, vision_cache_mib=None):
    """Model folder -> (formatter, engine, runner).  ``fast`` / ``compute_cap``: see Ink2Engine (without ``compute_cap``, auto asks
    nvidia-smi for the GPU the engine opens, as the command line does).  The tokenizer and template are read first, so a broken folder fails before
    the load.  ``draft``: auto | on | off (see the module text); ``eng.draft_note`` says what became of the drafter.  ``vision``:
    auto | on | off with its limits and host cache (Ink2Engine.set_vision); ``eng.vision_note`` says what became of it."""
    fmt = Ink2Formatter(pack.root)
    vision = vision_mode_of(vision)
    vision_limits(vision_max_image_tokens, vision_max_request_tokens)  # wrong values fail before the long load
    eng = Ink2Engine(pack, max_ctx, dll=dll, fast=fast, compute_cap=compute_cap)
    try:
        eng.set_vision(pack, vision, vision_max_image_tokens, vision_max_request_tokens, vision_cache_mib)
        eng.set_drafter(pack, draft, draft_depth)
        eng.set_copy(copy, copy_min)                         # off unless --copy / $YASAKANI_INK2_COPY says auto
        runner = Ink2Runner(eng, fmt, pack, seed0=seed0)
        if warm:
            runner.warmup(say)
    except BaseException:
        eng.close()
        raise
    return fmt, eng, runner


# ---- self-check ----------------------------------------------------------------------------------------------------------------------
def selfcheck_cases(pack: Ink2Pack, reference: Optional[str] = None) -> List[dict]:
    path = Path(reference) if reference else pack.root / "selfcheck.json"
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise YasakaniError("cannot read the reference file %s: %s" % (path, ex))
    if ref.get("format") != 1 or ref.get("architecture") != ARCHITECTURE:
        raise YasakaniError("%s is not an Inkling-2 reference file" % path)
    return list(ref.get("cases") or [])


def run_selfcheck(eng: Ink2Engine, fmt: Ink2Formatter, cases: Sequence[dict], say=print) -> int:
    """Greedy generation of every case against its expected tokens.  0 all pass, 1 a case fails."""
    bad = 0
    for c in cases:
        ids = fmt.encode(fmt.render(c["messages"], **(c.get("template_kwargs") or {})))
        if ids != list(c["prompt_ids"]):
            say("  %-10s FAIL  the prompt ids differ (tokenizer or chat template are not the reference ones)" % c["name"])
            bad += 1
            continue
        want = [int(t) for t in c["tokens"]]
        eng.reset()
        t0 = time.perf_counter()
        logits, _ = eng.read_prompt(ids)
        got: List[int] = []

        def on_token(tok: int, got=got, n=len(want)):
            got.append(int(tok))
            return END if len(got) >= n else None

        passes = generate(eng, int(np.argmax(logits)), on_token)
        dt = time.perf_counter() - t0
        eng.reset()
        per_pass = ", %.2f tokens per pass" % (len(got) / passes) if eng.drafting and passes else ""
        if got == want:
            say("  %-10s PASS  (%d prompt ids, %d tokens equal to the reference, %.2f s%s)" % (c["name"], len(ids), len(want), dt,
                                                                                            per_pass))
        else:
            n = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
            say("  %-10s FAIL  the output differs from token #%d on: %r" % (c["name"], n + 1, fmt.decode(got)))
            bad += 1
    return 1 if bad else 0
