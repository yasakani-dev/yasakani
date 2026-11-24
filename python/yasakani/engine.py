# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The engine: open the model, set sampling, generate with live token progress, cancel."""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
import locale
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._native import NativeLib, YasakaniError
from .runtime import (DEFAULT_SPEC, EOS_IDS, N_STATS, STAT_INDEX, STATE_FIELDS, VOCAB_SIZE, ModelPack, RuntimeProfile, spec_array)

# engine option keys (yskn_set_opt)
OPT_TEMPERATURE, OPT_TOP_P, OPT_TOP_K, OPT_SEED, OPT_SAMPLER, OPT_MIN_P = 10, 11, 12, 13, 16, 29
OPT_WIDE_MASK, OPT_WIDE_T16 = 38, 39       # 0.1.7: widths of copy passes (bit 0 = 32 rows, 1 = 48, 2 = 64); 39..42 cycle times in us
_NO_STOP = 0x7FFFFFFF                      # a stop id no token can have (ignore_eos)
CONTEXT_MARGIN = 64                        # ids kept free at the end of the context
READ_CHUNK = 1023                          # 0.1.7: Engine.read_prompt reads in pieces of at most this many ids (the engine sets up a
                                           # large work area for one prompt read of 1,024 ids or more and keeps it)
OPT_EXACT_WIDE = 31                        # passes of 17..64 rows with the exact-sum GEMV (0 = off; the engine's default is on where it can)

# ---- the multi-stream pass (experimental entries of the engine library, 0.1.7) --------------------------------------------------
# One engine (the carrier) holds stream slots: each holds the state of one request.  A request is read by the carrier (generate with
# max_new=1), then ms_load copies that state into a slot; a pass carries up to 16 rows of several slots.  The rows of a slot are next to
# each other: its frontier (the token it emitted last, not in its KV yet) first, then its drafts.  After every pass, ms_commit says how
# many drafts of each slot the pass confirmed (ms_accept counts them).  A stream gives the same ids as the same request alone.
MS_ROWS = 16                               # rows of one pass at most
MS_TOP = 8                                 # entries of a row's top list
CFG_FRONTIER = 46                          # the 47th value of the generation call: 1 = the frontier form (only that form sends it)
MS_UNSUPPORTED = {                         # yskn_ms_info out[1]: why the multi-stream pass does not take this engine's settings
    1: "the KV is kept in 16 bits (the multi-stream pass is built for KV5 only)",
    2: "the KV is kept in 8 bits (the multi-stream pass is built for KV5 only)",
    3: "image input is on",
    4: "the built-in draft head is loaded",
    5: "the attention settings are not the package's",
    6: "the exact-sum GEMV is off",
    7: "a debug skip mask is set",
    8: "the kept conv outputs are off",
    9: "the weights are not shared",
    10: "the pass tables are not made for 16 rows",
    11: "the attention settings of this process (environment) are not the package's",
}


@dataclass(frozen=True)
class Sampling:
    """Seeded sampling.  ``temperature <= 0`` is greedy.  top_k is capped at 20 by the engine.  The speculation is built so that the
    result for a given seed is the same as with plain decoding; what was measured is in docs/MEASUREMENTS.md of the repository."""
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    seed: int = 0
    mode: int = 3

    def normalized(self) -> "Sampling":
        if self.temperature <= 0:
            return Sampling(0.0, 0.95, 20, 0.0, 0, self.mode)
        k = int(self.top_k) if self.top_k else 20
        if k <= 0 or k > 20:
            k = 20
        return Sampling(float(self.temperature), float(self.top_p), k, float(self.min_p or 0.0), int(self.seed), self.mode)

    def normalized_full(self) -> "Sampling":
        """For the Inkling-2 engine of #28, which samples over the whole vocabulary: top_k 0 stays off (no cap of 20), so the model's
        own defaults (top_p 1.0, top_k off) are what it uses, as on the 0.3.0 Inkling-2 path."""
        if self.temperature <= 0:
            return Sampling(0.0, 1.0, 1, 0.0, 0, self.mode)
        return Sampling(float(self.temperature), float(self.top_p), max(0, int(self.top_k or 0)), float(self.min_p or 0.0),
                        int(self.seed), self.mode)


@dataclass
class GenResult:
    ids: List[int]                      # generated ids: the first one (predicted by the prompt pass) and all that followed, a stop id last
    finish: str                         # "stop" (a stop id was produced) | "length" | "cancelled"
    prompt_tokens: int
    cached_tokens: int                  # prompt ids the engine did not have to compute again
    snap_mode: int                      # 0 fresh prefill, 1 continued the live state, 2 restored the snapshot (engine's report)
    passes: int
    prefill_ms: float
    gen_ms: float
    stats: Dict[str, float] = field(default_factory=dict)
    draft_n: int = 0                    # tokens the look-ahead proposed (its three sources together)
    draft_n_accepted: int = 0           # of these, the tokens the model's own pass confirmed

    @property
    def completion_tokens(self) -> int:
        return len(self.ids)


@dataclass
class Forced:
    """What :meth:`Engine.teacher_forced` measured.  Index ``i`` is the prediction made after ``ids[:i + 1]``."""
    argmax: np.ndarray                  # int32: the most likely next id
    nll: np.ndarray                     # float32: -log p(ids[i + 1]), natural log
    margin: Optional[np.ndarray] = None # float32: logit of the most likely id minus the logit of the second one (margins=True only)


@dataclass(frozen=True)
class MsInfo:
    """What the engine library says about the multi-stream pass of an engine (yskn_ms_info; works before :meth:`Engine.ms_open`)."""
    unsupported: int                    # 0 = the pass takes this engine's settings; else a key of MS_UNSUPPORTED
    kv_format: int                      # 5 = KV5, 8 = 8-bit, 16 = 16-bit
    kv5_bits: int                       # bits of a packed KV5 code (0 when not KV5)
    kv_bytes_per_position: int          # KV of one position of one sequence; the carrier and every slot take max_ctx positions of it
    slot_bytes: int                     # one slot besides its KV (recurrent states and histories, the drafter memory)
    max_ctx: int                        # positions of the carrier and of every slot
    max_rows: int                       # rows of one pass at most
    max_slots: int
    drafter: bool                       # the drafter is loaded
    draft_head: bool                    # the drafter's output head is loaded
    slots_made: int
    slots_bound: int
    snapshot_bytes: int                 # pinned host memory of one saved prompt state
    snapshot_slots: int                 # saved prompt states at most

    @property
    def reason(self) -> str:
        """Why the pass does not take these settings ("" when it does)."""
        if not self.unsupported:
            return ""
        return MS_UNSUPPORTED.get(self.unsupported, "unsupported settings (code %d)" % self.unsupported)


@dataclass
class MsPass:
    """One multi-stream pass: ``toks[r]`` = the token of row r (drafts that the drafter made filled in), ``argmax[r]`` = the token the
    pass gave after row r (sampled when the row's slot samples).  The top lists (``tops=True``) hold MS_TOP entries per row."""
    argmax: List[int]
    toks: List[int]
    top_ids: Optional[np.ndarray] = None   # int32 (rows, MS_TOP)
    top_vals: Optional[np.ndarray] = None  # float32 (rows, MS_TOP)


def stop_pair(stop_ids: Optional[Sequence[int]], spec: Dict[str, int]) -> Tuple[int, ...]:
    """The stop ids a generation call ends at, as the engine takes them (the single path and the scheduler both decide here):
    ``None`` = none (``()``); an empty sequence = the engine's own two (``eos1`` and ``eos2`` of the speculation settings ``spec``:
    the model's end ids unless changed); else the first two ids given (the first one twice when one is given; a third and later ones
    are not stop ids)."""
    if stop_ids is None:
        return ()
    s = [int(x) for x in stop_ids]
    if not s:
        return (int(spec["eos1"]), int(spec["eos2"]))
    return (s[0], s[1] if len(s) > 1 else s[0])


def ms_accept(rows: Sequence[int], argmax: Sequence[int], stops=()) -> Tuple[int, List[int]]:
    """The drafts of one stream that a pass confirmed.  ``rows`` = the stream's rows of the pass (its frontier, then its drafts),
    ``argmax`` = the tokens the pass gave after them.  A draft is confirmed while it equals the token the pass gave after the row before
    it, from the first draft on.  A confirmed draft that is a stop id ends the stream there: it is the stream's last id and is not
    committed (the last id of a stream is never in its KV, as on the single path).  Returns the number of drafts to commit
    (:meth:`Engine.ms_commit`) and the stream's new ids: the committed drafts and the token after the last of them."""
    n = len(rows) - 1
    if n < 0 or len(argmax) != n + 1:
        raise ValueError("a stream has one token after each of its rows")
    a = 0
    while a < n and rows[a + 1] == argmax[a] and rows[a + 1] not in stops:
        a += 1
    return a, [int(x) for x in rows[1:a + 1]] + [int(argmax[a])]


class _MsBuffers:
    """Arrays the multi-stream calls write and read (made once; their addresses are kept)."""

    def __init__(self):
        self.slots, self.toks, self.dev, self.am, self.rt = (np.zeros(MS_ROWS, np.int32) for _ in range(5))
        self.cs, self.ca = np.zeros(64, np.int32), np.zeros(64, np.int32)
        self.ti = np.zeros((MS_ROWS, MS_TOP), np.int32)
        self.tv = np.zeros((MS_ROWS, MS_TOP), np.float32)
        self.pos = C.c_int(0)
        for k in ("slots", "toks", "dev", "am", "rt", "cs", "ca", "ti", "tv"):
            setattr(self, "p_" + k, getattr(self, k).ctypes.data)


def forced_passes(n: int, rows: int) -> List[int]:
    """Rows of each pass for ``n`` positions: ``rows`` at a time, and never a last pass of one row (a one-row pass can take a path of
    its own in the engine)."""
    if n < 1 or rows < 2:
        raise ValueError("teacher forcing needs at least one position and passes of two rows or more")
    sizes = [rows] * (n // rows) + ([n % rows] if n % rows else [])
    if len(sizes) > 1 and sizes[-1] == 1:
        sizes[-2] -= 1
        sizes[-1] = 2
    return sizes


def _narrow_paths_as_utf8(paths) -> bool:
    """Windows: the engine library gets file names as UTF-8 bytes, but parts of it read narrow file names in the code page of the C
    runtime's locale (the system code page by default), so a folder name with characters outside ASCII could not be opened.  When a
    path has such characters, the character type of the C locale is switched to UTF-8 for this process, so that every part agrees.
    Paths that are plain ASCII (the configuration all published checks were run with) change nothing.  Returns True when switched."""
    if sys.platform != "win32" or all(str(p).isascii() for p in paths):
        return False
    try:
        locale.setlocale(locale.LC_CTYPE, ".UTF-8")
        return True
    except locale.Error:
        return False


def _put_narrow_env_utf8(env: Dict[str, str]) -> None:
    """Store the path settings in the C runtime's narrow environment as UTF-8 bytes (``os.environ`` stores the narrow copy in the
    system code page; with the UTF-8 locale above the engine library would then read a file name it cannot open)."""
    crt = C.CDLL("ucrtbase")                    # the C runtime that Python and the engine library share
    crt._putenv_s.argtypes = [C.c_char_p, C.c_char_p]
    crt._putenv_s.restype = C.c_int
    for k, v in env.items():
        if crt._putenv_s(k.encode("ascii"), str(v).encode("utf-8")) != 0:
            raise YasakaniError("could not pass %s to the engine library" % k)



def _runtime_mod():
    from . import runtime
    return runtime

class Engine:
    """One model on one GPU.  Calls are serialized; ``cancel()`` may be called from any thread."""

    def __init__(self, pack: ModelPack, *, max_ctx: Optional[int] = None, device: int = 0, dll: Optional[str] = None,
                 profile: Optional[RuntimeProfile] = None, lib: Optional[NativeLib] = None, wide: Optional[str] = None):
        from . import widecheck
        self.pack = pack
        self.max_ctx = int(max_ctx or pack.max_ctx)
        self.profile = profile or RuntimeProfile.default(pack)
        # 0.1.7: wide copy passes (yasakani/widecheck.py): auto | on | off | force, from ``wide``, else YASAKANI_WIDE, else auto
        self.wide_mode = widecheck.mode_of(wide)
        for k, v in widecheck.engine_env(self.wide_mode).items():
            self.profile.env.setdefault(k, v)
        self.wide_recheck = False                 # measure again even when a kept result fits (--wide-recheck, yasakani report)
        self.wide_result = None                   # what the check decided (widecheck.WideResult), after warmup()
        self._wide_backoffs = 0                   # back-offs and wide passes of the copy loop in this process (widecheck.recheck_due)
        self._wide_passes_seen = 0
        self.utf8_paths = _narrow_paths_as_utf8([p for p in (pack.weights, pack.draft, pack.draft_head, pack.kv_profile) if p is not None])
        self.removed_env = self.profile.apply()
        if self.utf8_paths:
            _put_narrow_env_utf8({k: v for k, v in self.profile.env.items() if k in ("YSKN_BDR_SHEAD", "YSKN_TB_PROFILE")})
        self.lib = lib or NativeLib(dll)
        self._f = self.lib.fn
        self._lock = threading.RLock()
        self._cancelled = False
        self._busy = False
        if pack.is_hf_inkling2 and not _runtime_mod().lib_has_hf_inkling2(self.lib):
            raise _runtime_mod().hf_inkling2_error(pack.root)
        open_target = pack.root if pack.is_hf_inkling2 else pack.weights
        self._e = self._f["yskn_open"](str(open_target).encode("utf-8"), self.max_ctx, int(device), 0)
        if not self._e:
            msg = self.lib.last_error()
            hint = ""
            if "memory" in msg.lower():
                hint = "  (not enough free GPU memory: close other programs that use the GPU, or lower --max-ctx)"
            raise YasakaniError("could not open the model: %s%s" % (msg or "unknown error", hint))
        if not pack.is_hf_inkling2:
            try:
                self._check(self._f["yskn_load_bdraft"](self._e, str(pack.draft).encode("utf-8")))
            except Exception:
                self.close()
                raise
        self.spec: Dict[str, int] = dict(DEFAULT_SPEC)
        self.sampling: Optional[Sampling] = None
        self.ms_slots = 0                                   # slots of the current multi-stream binding (ms_open)
        self.ms_drafter = False                             # the drafter makes the drafts of the slots
        self._slot_sampling: List[Optional[Sampling]] = []  # the sampling each slot made so far holds (None = not known)
        self._msb: Optional[_MsBuffers] = None
        self.suppressed_count = 0

    # ---- plumbing --------------------------------------------------------------------------------------------------------------
    def _check(self, rc: int) -> None:
        if rc != 0:
            raise YasakaniError(self.lib.last_error() or "engine call failed (code %d)" % rc)

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_e", None):
                self._f["yskn_close"](self._e)
                self._e = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _need_open(self):
        if not self._e:
            raise YasakaniError("the engine is closed")

    # ---- state -----------------------------------------------------------------------------------------------------------------
    def position(self) -> int:
        self._need_open()
        return int(self._f["yskn_position"](self._e))

    def reset(self) -> None:
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_reset"](self._e))

    def state_hash(self) -> np.ndarray:
        """Hashes of the whole decode state (for checking that two runs are bit-identical)."""
        with self._lock:
            self._need_open()
            out = np.zeros(512, np.uint64)
            n = C.c_int(0)
            self._check(self._f["yskn_state_hash"](self._e, out.ctypes.data, len(out), C.byref(n)))
            return out[:n.value].copy()

    def cancel(self) -> None:
        """Ask a running generation to stop at its next pass boundary (the state stays valid)."""
        if self._e:
            self._cancelled = True
            self._f["yskn_cancel"](self._e)

    # ---- sampling --------------------------------------------------------------------------------------------------------------
    @property
    def supports_visual(self) -> bool:
        return "yskn_set_visual" in self._f

    @property
    def supports_ink2_vision(self) -> bool:
        return bool(self.pack.is_hf_inkling2 and "yskn_set_visual" in self._f and "yskn_ink2_encode_rgb" in self._f)

    def encode_ink2_images(self, rgbs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode one or more contiguous RGB uint8 frames [H_i, W_i, 3] on GPU -> (features_u16, grids_i32)."""
        if not self.supports_ink2_vision:
            raise YasakaniError("the engine library does not support native Inkling-2 vision encoding")
        if not rgbs:
            raise ValueError("encode_ink2_images requires at least one frame")
        arrs = [np.ascontiguousarray(np.asarray(im, dtype=np.uint8)) for im in rgbs]
        widths = np.empty(len(arrs), dtype=np.int32)
        heights = np.empty(len(arrs), dtype=np.int32)
        tot_tok = 0
        for i, a in enumerate(arrs):
            if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] <= 0 or a.shape[1] <= 0:
                raise ValueError("each image must be a non-empty (H, W, 3) uint8 array")
            h, w = int(a.shape[0]), int(a.shape[1])
            heights[i], widths[i] = h, w
            tot_tok += ((h + 39) // 40) * ((w + 39) // 40)
        hidden = int(getattr(self.pack, "hidden_size", 2304))
        out_u16 = np.empty((tot_tok, hidden), dtype=np.uint16)
        grids = np.empty((len(arrs), 2), dtype=np.int32)
        ptrs = (C.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
        n_tok = C.c_int(0)
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ink2_encode_rgb"](
                self._e, C.cast(ptrs, C.c_void_p), widths.ctypes.data, heights.ctypes.data,
                len(arrs), out_u16.ctypes.data, grids.ctypes.data, C.byref(n_tok)))
        if int(n_tok.value) != tot_tok:
            raise YasakaniError("ink2 vision encoder token count mismatch (%d != %d)" % (n_tok.value, tot_tok))
        return out_u16, grids

    @property
    def supports_suppress(self) -> bool:
        return "yskn_set_suppress" in self._f

    def set_suppress(self, token_ids: Optional[Sequence[int]]) -> int:
        with self._lock:
            self._need_open()
            fn = self._f.get("yskn_set_suppress")
            if not token_ids:
                if fn is not None:
                    self._check(fn(self._e, None, 0))
                self.suppressed_count = 0
                return 0
            if fn is None:
                raise YasakaniError("this engine library does not support output token suppression (yskn_set_suppress)")
            arr = np.ascontiguousarray(token_ids, dtype=np.int32)
            self._check(fn(self._e, arr.ctypes.data, int(arr.size)))
            self.suppressed_count = int(arr.size)
            return self.suppressed_count

    @contextmanager
    def visual_input(self, features, slots, positions=None, delta=0, n_prompt=None):
        """Keep visual input and generation under the same lock; always clear it afterwards.

        Features are bfloat16 bits, not float16 values. The arrays stay alive until generation finishes.
        Clearing keeps prefix tracking safe across subsequent requests.
        """
        if not self.supports_visual:
            raise YasakaniError("the engine library has no yskn_set_visual export")
        features, slots = np.asarray(features), np.asarray(slots)
        if self.pack.is_hf_inkling2 or positions is None:
            hidden = int(getattr(self.pack, "hidden_size", 2304)) if self.pack.is_hf_inkling2 else 5120
            if features.dtype != np.uint16 or features.ndim != 2 or features.shape != (slots.size, hidden):
                raise ValueError("visual features must be bfloat16 bits with shape (image tokens, %d)" % hidden)
            if np.any((features & 0x7f80) == 0x7f80):
                raise ValueError("non-finite visual features")
            if slots.dtype != np.int32 or slots.ndim != 1 or not slots.size:
                raise ValueError("visual slots must be a nonempty int32 vector")
            n_prompt_i = int(n_prompt) if n_prompt is not None else (int(slots.max()) + 1)
            if n_prompt_i <= 0 or n_prompt_i >= self.max_ctx or np.any(slots < 0) or np.any(slots >= n_prompt_i) or len(np.unique(slots)) != len(slots):
                raise ValueError("invalid visual slots or prompt length")
            features, slots = np.ascontiguousarray(features), np.ascontiguousarray(slots)
            with self._lock:
                self._need_open()
                f = self._f["yskn_set_visual"]
                try:
                    self._check(f(self._e, features.ctypes.data, slots.ctypes.data, len(slots), None, n_prompt_i, 0))
                    yield
                finally:
                    self._check(f(self._e, None, None, 0, None, 0, 0))
            return
        positions = np.asarray(positions)
        if features.dtype != np.uint16 or features.ndim != 2 or features.shape != (slots.size, 5120):
            raise ValueError("visual features must be bfloat16 bits with shape (image tokens, 5120)")
        if slots.dtype != np.int32 or slots.ndim != 1 or not slots.size:
            raise ValueError("visual slots must be a nonempty int32 vector")
        if positions.dtype != np.int32 or positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("visual positions must be int32 with shape (3, prompt tokens)")
        n_prompt_i = positions.shape[1]
        if n_prompt_i >= self.max_ctx or np.any(slots < 0) or np.any(slots >= n_prompt_i) or len(np.unique(slots)) != len(slots):
            raise ValueError("invalid visual slots or prompt length")
        if np.any(positions < 0) or np.any(positions >= self.max_ctx) or int(delta) != int(positions.max()) + 1 - n_prompt_i:
            raise ValueError("invalid visual positions or delta")
        features, slots, positions = map(np.ascontiguousarray, (features, slots, positions))
        with self._lock:
            self._need_open()
            f = self._f["yskn_set_visual"]
            try:
                self._check(f(self._e, features.ctypes.data, slots.ctypes.data, len(slots), positions.ctypes.data,
                              n_prompt_i, int(delta)))
                yield
            finally:
                self._check(f(self._e, None, None, 0, None, 0, 0))

    def set_sampling(self, s: Sampling) -> Sampling:
        s = s.normalized_full() if self.pack.is_hf_inkling2 else s.normalized()
        with self._lock:
            self._need_open()
            f, e = self._f["yskn_set_opt"], self._e
            self._check(f(e, OPT_TOP_P, int(round(s.top_p * 1000))))
            self._check(f(e, OPT_TOP_K, int(s.top_k)))
            self._check(f(e, OPT_SEED, int(s.seed)))
            self._check(f(e, OPT_TEMPERATURE, int(round(s.temperature * 1000))))
            self._check(f(e, OPT_SAMPLER, int(s.mode)))
            self._check(f(e, OPT_MIN_P, int(round(s.min_p * 1000))))
            self.sampling = s
        return s

    # ---- generation ------------------------------------------------------------------------------------------------------------
    def plan(self, n_prompt: int, max_new: int) -> int:
        """Number of ids to ask for after the first one; raises when the prompt does not fit."""
        if n_prompt < 1:
            raise ValueError("the prompt is empty")
        room = self.max_ctx - n_prompt - CONTEXT_MARGIN
        if room < 1:
            raise ValueError("the prompt (%d tokens) leaves no room in the context of %d tokens" % (n_prompt, self.max_ctx))
        return max(0, min(int(max_new) - 1, room))

    def generate(self, prompt: Sequence[int], max_new: int, *, stop_ids: Optional[Sequence[int]] = EOS_IDS,
                 on_tokens: Optional[Callable[[np.ndarray], None]] = None, spec: Optional[Dict[str, int]] = None,
                 poll_s: float = 0.001, frontier: bool = False) -> GenResult:
        """Generate up to ``max_new`` ids (the first one comes from the prompt pass).  ``on_tokens(ids)`` is called on the calling
        thread with all ids produced so far whenever new ones are committed.  If it raises, the generation is cancelled and the
        exception is re-raised after the engine has stopped.  ``stop_ids=None`` ignores stop ids (always runs to ``max_new``); the
        other forms: :func:`stop_pair`.

        ``frontier=True`` (engine libraries with the multi-stream pass): the last prompt id is the token the carrier's stream emitted
        last (right after :meth:`ms_unload`, or after a generation of this engine), it is not read again: the first pass checks it with
        its drafts, as the stream's next pass would.  The prompt without that id must be the engine's live sequence; else the engine
        refuses the call and nothing changes.  The result has the same form as without it (the first id is the one after the prompt)."""
        prompt_arr = np.ascontiguousarray(np.asarray(prompt, dtype=np.int32))
        n_gen = self.plan(len(prompt_arr), max_new)
        cfg = dict(self.spec)
        for k, v in (spec or {}).items():
            if k in STATE_FIELDS:
                raise ValueError("%s decides how the engine keeps its state and is not a per-request setting" % k)
            if k not in cfg:
                raise ValueError("unknown speculation setting: " + k)
            cfg[k] = int(v)
        stops = stop_pair(stop_ids, cfg)
        cfg["eos1"], cfg["eos2"] = stops if stops else (_NO_STOP, _NO_STOP)
        cfa = spec_array(cfg)
        if frontier:                                         # the 47th value; every other call sends the 46 values of 0.1.x
            if not self.supports_parallel:
                raise YasakaniError("the engine library has no multi-stream entries (the frontier form needs them)")
            if int(cfg["snap"]) != 1:
                raise ValueError("the frontier form needs prompt snapshots (snap = 1)")
            cfa = np.ascontiguousarray(np.append(cfa, np.int32(1)).astype(np.int32))
            assert len(cfa) == CFG_FRONTIER + 1
        out = np.full(n_gen + 1, -1, np.int32)
        st = np.zeros(N_STATS, np.float64)
        tm = np.zeros(2, np.float64)
        box: Dict[str, object] = {}
        with self._lock:
            self._need_open()
            self._cancelled = False
            self._busy = True
            e = self._e
            fn = self._f["yskn_spec_generate"]

            def work():
                try:
                    rc = fn(e, prompt_arr.ctypes.data, len(prompt_arr), n_gen, cfa.ctypes.data, len(cfa), out.ctypes.data,
                            st.ctypes.data, tm.ctypes.data)
                    box["rc"] = rc
                    if rc != 0:
                        box["err"] = self.lib.last_error()          # thread-local in the engine: read it here
                except BaseException as ex:                          # noqa: BLE001
                    box["exc"] = ex

            th = threading.Thread(target=work, name="yasakani-generate", daemon=True)
            th.start()
            seen = 0
            failure: Optional[BaseException] = None
            try:
                while True:
                    alive = th.is_alive()
                    n = seen
                    while n < len(out) and out[n] >= 0:
                        n += 1
                    if n > seen:
                        seen = n
                        if on_tokens is not None and failure is None:
                            try:
                                on_tokens(out[:n])
                            except BaseException as ex:              # noqa: BLE001  (a consumer that fails stops the engine)
                                failure = ex
                                self.cancel()
                    if not alive and seen >= n:
                        break
                    th.join(poll_s) if alive else None
            except BaseException:                                    # e.g. Ctrl+C: stop the engine, wait for it, then re-raise
                self.cancel()
                raise
            finally:
                th.join()
                self._busy = False
            if failure is not None:
                raise failure
        if "exc" in box:
            raise box["exc"]                                       # type: ignore[misc]
        if box.get("rc") != 0:
            raise YasakaniError(str(box.get("err") or "engine call failed"))
        if st[STAT_INDEX["wide_passes"]] > 0:                      # 0.1.7: how often the copy loop found wide passes slower than 16 rows
            self._wide_passes_seen += int(st[STAT_INDEX["wide_passes"]])
            self._wide_backoffs += int(st[STAT_INDEX["wide_backoffs"]])
            from . import widecheck
            widecheck.note_backoffs(self)
        produced = int(st[STAT_INDEX["produced"]])
        gen = min(produced + 1, len(out))
        ids = [int(x) for x in out[:gen]]
        stopped = int(st[STAT_INDEX["eos"]]) == 1
        finish = "stop" if stopped else ("cancelled" if self._cancelled else "length")
        stats = {k: float(st[i]) for k, i in STAT_INDEX.items()}
        return GenResult(ids=ids, finish=finish, prompt_tokens=len(prompt_arr), cached_tokens=int(st[STAT_INDEX["cached_tokens"]]),
                         snap_mode=int(st[STAT_INDEX["snap_mode"]]), passes=int(st[STAT_INDEX["passes"]]),
                         prefill_ms=float(tm[0]), gen_ms=float(tm[1]), stats=stats,
                         draft_n=int(sum(st[STAT_INDEX["drafted_%d" % s]] for s in range(3))),
                         draft_n_accepted=int(sum(st[STAT_INDEX["accepted_%d" % s]] for s in range(3))))

    # ---- teacher forcing -------------------------------------------------------------------------------------------------------
    @property
    def supports_forced(self) -> bool:
        if self.pack.is_hf_inkling2:                # the Inkling-2 engine of #28 has no teacher-forcing pass (yskn_verify_nll refuses it)
            return False
        return all(k in self._f for k in ("yskn_verify_nll", "yskn_commit"))

    def teacher_forced(self, ids: Sequence[int], *, rows: int = 16, margins: bool = False) -> Forced:
        """Let the engine read ``ids`` in passes of ``rows`` positions without generating anything (teacher forcing) and return, for
        every position but the last, the most likely next id and the negative log-likelihood of the id that really follows.
        ``margins=True`` also reads the logits of the whole vocabulary to give the margin of the most likely id.  The state is reset
        before and after; the sampling settings are not used."""
        if not self.supports_forced or (margins and not all(k in self._f for k in ("yskn_logits_rows", "yskn_verify"))):
            raise YasakaniError("the engine library has no exports for teacher forcing (yskn_verify_nll, yskn_commit)")
        seq = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        n = len(seq) - 1
        if n + 1 > self.max_ctx - CONTEXT_MARGIN:
            raise ValueError("%d ids do not fit the context of %d tokens" % (len(seq), self.max_ctx))
        sizes = forced_passes(n, rows)
        am = np.empty(n, np.int32)
        nll = np.empty(n, np.float32)
        mg = np.empty(n, np.float32) if margins else None
        f = self._f
        with self._lock:
            self._need_open()
            e = self._e
            self._check(f["yskn_reset"](e))
            done = False
            try:
                pos = 0
                for m in sizes:
                    x = np.ascontiguousarray(seq[pos:pos + m])
                    t = np.ascontiguousarray(seq[pos + 1:pos + m + 1])
                    a = np.empty(m, np.int32)
                    v = np.empty(m, np.float32)
                    self._check(f["yskn_verify_nll"](e, x.ctypes.data, t.ctypes.data, m, 1, a.ctypes.data, v.ctypes.data))
                    if margins:
                        lg = np.empty((m, VOCAB_SIZE), np.float32)
                        self._check(f["yskn_logits_rows"](e, 0, m, lg.ctypes.data))
                        if not np.array_equal(lg.argmax(1), a):        # the logits may not be kept by this pass: make it again
                            b = np.empty(m, np.int32)
                            self._check(f["yskn_verify"](e, x.ctypes.data, m, 1, b.ctypes.data, None, None))
                            self._check(f["yskn_logits_rows"](e, 0, m, lg.ctypes.data))
                            if not (np.array_equal(b, a) and np.array_equal(lg.argmax(1), a)):
                                raise YasakaniError("teacher forcing: the logits of the rows do not belong to the pass")
                        top2 = np.partition(lg, -2, axis=1)[:, -2:]
                        mg[pos:pos + m] = top2.max(1) - top2.min(1)
                    self._check(f["yskn_commit"](e, m - 1))
                    am[pos:pos + m] = a
                    nll[pos:pos + m] = v
                    pos += m
                done = True
            finally:
                rc = f["yskn_reset"](e)
                if done:                                             # an earlier error is the one to report
                    self._check(rc)
        return Forced(argmax=am, nll=nll, margin=mg)

    # ---- multi-stream pass -----------------------------------------------------------------------------------------------------
    # Experimental entries of the engine library (see the notes at MS_ROWS).  The calls of one engine are made on one thread (the
    # engine's last error is per thread); only cancel() may come from another one.  Rules of the engine library that callers keep:
    #   - ms_open once (a new binding drops the carrier's live sequence: the next read of a request is not a continuation);
    #   - read a request with generate(prompt, 1) and call ms_load right after it (nothing in between), so that the slot knows its
    #     start and frontier and ms_unload can hand the stream back for a continuation;
    #   - ms_commit after every pass (also with nothing accepted) before any other call of the carrier;
    #   - back to one request at a time: ms_unload, then right away generate(ids + [frontier]) (frontier=True or not).
    @property
    def supports_parallel(self) -> bool:
        """The engine library has the multi-stream entries (yskn_ms_*)."""
        return bool(getattr(self.lib, "supports_parallel", False))

    def _need_ms(self) -> None:
        if not self.supports_parallel:
            raise YasakaniError("the engine library has no multi-stream entries (yskn_ms_*): serve one request at a time")

    def _need_bound(self) -> _MsBuffers:
        self._need_ms()
        if self._msb is None or self.ms_slots < 1:
            raise YasakaniError("the multi-stream slots are not set up (call ms_open first)")
        return self._msb

    def ms_info(self) -> MsInfo:
        """What the engine library says about the multi-stream pass of this engine (also before :meth:`ms_open`)."""
        self._need_ms()
        out = np.zeros(15, np.int64)
        n = C.c_int(0)
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_info"](self._e, out.ctypes.data, len(out), C.byref(n)))
        v = [int(x) for x in out[:n.value]] + [0] * (15 - n.value)
        if v[0] != 1:
            raise YasakaniError("the engine library describes its multi-stream pass in a form this package does not know (%d)" % v[0])
        return MsInfo(unsupported=v[1], kv_format=v[2], kv5_bits=v[3], kv_bytes_per_position=v[4], slot_bytes=v[5], max_ctx=v[6],
                      max_rows=v[7], max_slots=v[8], drafter=bool(v[9]), draft_head=bool(v[10]), slots_made=v[11], slots_bound=v[12],
                      snapshot_bytes=v[13], snapshot_slots=v[14])

    def ms_open(self, n_slots: int, *, drafter: bool = True) -> MsInfo:
        """Make (or bind again) ``n_slots`` stream slots on this engine and let the drafter make their drafts (``drafter=False``:
        only drafts the caller gives).  Call it once: a new binding drops the live sequence of the carrier.  Slots made before keep their
        streams and sampling; a new slot starts with the carrier's sampling.  Raises when the library or the settings do not allow the
        pass (the reason is in the message)."""
        self._need_ms()
        n = int(n_slots)
        with self._lock:
            info = self.ms_info()
            if info.unsupported:
                raise YasakaniError("the multi-stream pass does not take this engine's settings: " + info.reason)
            if not 1 <= n <= info.max_slots:
                raise ValueError("stream slots: 1..%d" % info.max_slots)
            rc = self._f["yskn_ms_bind2"](self._e, None, 0, n)
            if rc != 0:                                     # a failure after the checks (out of memory) leaves no binding
                self.ms_slots = 0
                self._check(rc)
            while len(self._slot_sampling) < n:
                self._slot_sampling.append(self.sampling)
            self.ms_slots = n
            if self._msb is None:
                self._msb = _MsBuffers()
            self.ms_drafter = False
            self._check(self._f["yskn_ms_opt"](self._e, 1, 1 if drafter else 0))
            self.ms_drafter = bool(drafter)
            return self.ms_info()

    def ms_load(self, slot: int) -> None:
        """Copy the carrier's state (the request it read last) into ``slot``, with the carrier's sampling (set the slot's own values
        with :meth:`ms_set_sampling` after this).  Call it right after the read (generate with max_new=1)."""
        self._need_bound()
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_load"](self._e, int(slot), self._e))
            self._slot_sampling[int(slot)] = self.sampling

    def ms_unload(self, slot: int, ids: Sequence[int]) -> None:
        """Hand the stream of ``slot`` back to the carrier.  ``ids`` = the tokens the slot holds (its position = len(ids)): the prompt
        it was loaded with and the ids it emitted, all but the last one (its frontier).  The carrier takes the slot's sampling too
        (:attr:`sampling` follows).  Continue it right away with ``generate(ids + [frontier], ...)``."""
        self._need_bound()
        arr = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_unload"](self._e, int(slot), arr.ctypes.data, len(arr)))
            self.sampling = self._slot_sampling[int(slot)]

    def live_drop(self) -> None:
        """Drop the carrier's live sequence: the next read of a request does not continue what the carrier holds (its saved prompt
        states stay)."""
        self._need_ms()
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_live_drop"](self._e))

    def ms_set_sampling(self, slot: int, s: Sampling) -> Sampling:
        """The sampling of one slot from its next pass on (the values :meth:`set_sampling` gives the carrier)."""
        self._need_bound()
        s = s.normalized()
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_set_sampling"](self._e, int(slot), int(round(s.temperature * 1000)), int(round(s.top_p * 1000)),
                                                        int(s.top_k), int(round(s.min_p * 1000)), int(s.mode), int(s.seed) & 0xFFFFFFFF))
            self._slot_sampling[int(slot)] = s
        return s

    def slot_sampling(self, slot: int) -> Optional[Sampling]:
        """The sampling ``slot`` holds as far as this object knows (None = not known)."""
        return self._slot_sampling[int(slot)] if 0 <= int(slot) < len(self._slot_sampling) else None

    def ms_pass(self, slots: Sequence[int], toks: Sequence[int], dev: Optional[Sequence[int]] = None, *, tops: bool = False) -> MsPass:
        """One pass of up to 16 rows.  ``slots[r]`` = the slot of row r (the rows of a slot next to each other), ``toks[r]`` = its token:
        the slot's frontier at its first row, then the caller's drafts.  ``dev`` (needs the drafter, see :meth:`ms_open`): ``dev[r] !=
        0`` at the first row of a slot lets the drafter fill the slot's other rows (their values in ``toks`` are not used).  Call
        :meth:`ms_commit` right after it."""
        b = self._need_bound()
        m = len(slots)
        if not 1 <= m <= MS_ROWS or len(toks) != m or (dev is not None and len(dev) != m):
            raise ValueError("a pass has 1..%d rows, with one slot and one token (and one dev flag) per row" % MS_ROWS)
        b.slots[:m] = slots
        b.toks[:m] = toks
        g = 1 if int(self.spec.get("use_graph", 1)) else 0
        with self._lock:
            self._need_open()
            f, e = self._f, self._e
            if dev is not None:
                b.dev[:m] = dev
                self._check(f["yskn_ms_draft"](e, m, b.p_slots, b.p_toks, b.p_dev, g))
                tp = None
            else:
                tp = b.p_toks
            self._check(f["yskn_ms_step2"](e, m, b.p_slots, tp, g, b.p_am, b.p_rt, b.p_ti if tops else None, b.p_tv if tops else None))
            res = MsPass(argmax=b.am[:m].tolist(), toks=b.rt[:m].tolist())
            if tops:
                res.top_ids, res.top_vals = b.ti[:m].copy(), b.tv[:m].copy()
        return res

    def ms_commit(self, slots: Sequence[int], n_accept: Sequence[int]) -> None:
        """Right after every :meth:`ms_pass` (also with no slot): ``n_accept[i]`` = how many drafts of ``slots[i]`` the pass confirmed
        (:func:`ms_accept`).  The engine checks every item first and refuses drafts the pass did not confirm (nothing changes)."""
        b = self._need_bound()
        n = len(slots)
        if len(n_accept) != n or n > len(b.cs):
            raise ValueError("one accepted count per slot, at most %d slots" % len(b.cs))
        b.cs[:n] = slots
        b.ca[:n] = n_accept
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_commit"](self._e, n, b.p_cs, b.p_ca))

    def ms_flush(self) -> None:
        """Finish the pending work of the slots (a pass that is not committed counts as no draft accepted)."""
        self._need_ms()
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_flush"](self._e))

    def ms_slot_pos(self, slot: int) -> int:
        """The position of ``slot``: the number of tokens it holds."""
        b = self._need_bound()
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_slot_pos"](self._e, int(slot), C.byref(b.pos)))
            return int(b.pos.value)

    def ms_slot_hash(self, slot: int) -> np.ndarray:
        """The hashes :meth:`state_hash` would give for an engine in the slot's state."""
        self._need_bound()
        out = np.zeros(512, np.uint64)
        n = C.c_int(0)
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_ms_slot_hash"](self._e, int(slot), out.ctypes.data, len(out), C.byref(n)))
        return out[:n.value].copy()

    # ---- warm-up ---------------------------------------------------------------------------------------------------------------
    def warmup(self, prompts: Sequence[Sequence[int]], n_gen: int = 32, wide_ids: Optional[Sequence[int]] = None,
               say: Optional[Callable[[str], None]] = None) -> None:
        """Run short generations so that the CUDA graphs and the prompt-snapshot buffer exist before the first real request.  Give one
        plain prompt and one prompt that contains the chat start marker; the state is reset afterwards.  0.1.7: then decide the widths
        of the wide copy passes on this GPU (yasakani/widecheck.py; ``wide_ids`` is the text the check reads, None = use a kept result
        only) and build their graphs.  ``say`` gets the one-line summary."""
        if self.pack.is_hf_inkling2:
            self.set_sampling(Sampling(0.0, 1.0, 1, 0.0, 0, 3))
            if prompts:
                self.reset()
                self.generate(prompts[0], 4)
            self.reset()
            return
        from . import widecheck
        self.set_sampling(Sampling(1.0, 0.95, 20, 0.0, 999, 3))
        for p in prompts:
            self.reset()
            self.generate(p, n_gen + 1)
        self.reset()
        self.wide_result = widecheck.setup(self, wide_ids, check=wide_ids is not None)
        self.prepare_wide()
        widecheck.after_prepare(self, self.wide_result)
        if say is not None and self.wide_result is not None and self.wide_result.line:
            say(self.wide_result.line)

    # ---- wide copy passes (0.1.7) ---------------------------------------------------------------------------------------------
    @property
    def supports_wide_check(self) -> bool:
        if self.pack.is_hf_inkling2:
            return False
        return all(k in self._f for k in ("yskn_time_cycle", "yskn_wide_state", "yskn_prefill", "yskn_verify_nll", "yskn_commit"))

    def wide_state(self):
        """What the engine says about its wide copy passes (widecheck.WideState), or None for engine libraries before 0.1.7."""
        from .widecheck import WideState
        f = self._f.get("yskn_wide_state")
        if f is None:
            return None
        with self._lock:
            self._need_open()
            out = np.zeros(32, np.int32)
            n = C.c_int(0)
            self._check(f(self._e, out.ctypes.data, len(out), C.byref(n)))
            return WideState.from_array(out[:n.value])

    def set_wide(self, mask: int, times_us: Optional[Sequence[int]] = None) -> None:
        """Open the widths of copy passes in ``mask`` (bit 0 = 32 rows, 1 = 48, 2 = 64) and close the others; ``times_us`` = the times
        of copy cycles of 16 / 32 / 48 / 64 rows the copy loop starts from (0 = the built-in values of the RTX 5090)."""
        with self._lock:
            self._need_open()
            f = self._f["yskn_set_opt"]
            self._check(f(self._e, OPT_WIDE_MASK, int(mask) & 7))
            if times_us is not None:
                for i, t in enumerate(list(times_us)[:4]):
                    self._check(f(self._e, OPT_WIDE_T16 + i, max(0, min(10000000, int(round(t))))))

    def read_prompt(self, ids: Sequence[int]) -> int:
        """Read ``ids`` from a fresh state (nothing generated); returns the predicted next id.  For the checks of widecheck.  Read in
        pieces of at most READ_CHUNK ids, so that the engine does not set up (and keep) the work area of long prompt reads."""
        a = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        first = C.c_int32(0)
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_reset"](self._e))
            for k in range(0, len(a), READ_CHUNK):
                b = np.ascontiguousarray(a[k:k + READ_CHUNK])
                self._check(self._f["yskn_prefill"](self._e, b.ctypes.data, len(b), C.byref(first)))
        return int(first.value)

    def set_exact_wide(self, on: bool) -> None:
        """Passes of 17..64 rows with the exact-sum GEMV (set_opt 31; on by default where the engine can).  Turned off when the check
        finds that wide passes compute differently on this GPU."""
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_set_opt"](self._e, OPT_EXACT_WIDE, 1 if on else 0))

    def time_cycles(self, ids: Sequence[int], n: int) -> np.ndarray:
        """``n`` copy cycles of ``len(ids)`` rows at the current position (yskn_time_cycle): wall time of each, ms."""
        a = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        out = np.zeros(int(n), np.float32)
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_time_cycle"](self._e, a.ctypes.data, len(a), int(n), out.ctypes.data))
        return out

    def verify_rows(self, ids: Sequence[int], targets: Sequence[int], commit: bool):
        """One teacher-forced pass of ``len(ids)`` rows at the current position: (most likely ids, NLL of ``targets``); with
        ``commit`` the rows are then accepted (the position moves on)."""
        x = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        t = np.ascontiguousarray(np.asarray(targets, dtype=np.int32))
        m = len(x)
        a = np.empty(m, np.int32)
        v = np.empty(m, np.float32)
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_verify_nll"](self._e, x.ctypes.data, t.ctypes.data, m, 1, a.ctypes.data, v.ctypes.data))
            if commit:
                self._check(self._f["yskn_commit"](self._e, m - 1))
        return a, v

    def prepare_wide(self) -> None:
        """Build the graphs of the wide copy passes for greedy and for sampled generation (engine libraries of 0.1.4 or newer; nothing
        runs, the state is unchanged).  The sampling settings are restored afterwards."""
        f = self._f.get("yskn_prepare_wide")
        rows = int(self.spec.get("copy_rows", 16))
        if f is None or rows <= 16:
            return
        with self._lock:
            self._need_open()
            keep = self.sampling
            self._check(self._f["yskn_set_opt"](self._e, OPT_TEMPERATURE, 0))
            self._check(f(self._e, rows))
            self._check(self._f["yskn_set_opt"](self._e, OPT_TEMPERATURE, 1000))
            self._check(f(self._e, rows))
            if keep is not None:
                self.set_sampling(keep)
            else:
                self._check(self._f["yskn_set_opt"](self._e, OPT_TEMPERATURE, 0))
