# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Experimental macOS Metal backend for the original YSKN weights.

Metal evaluates compressed matrices; NumPy evaluates recurrent state and attention.
No CUDA speculation, quantized KV cache or CUDA output equivalence is implied.
"""
from __future__ import annotations

import ctypes as C
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ._native import YasakaniError
from .engine import CONTEXT_MARGIN, GenResult, Sampling
from .runtime import EOS_IDS
from .yskn_format import Container, vector


def find_metal_library(explicit=None):
    if explicit or os.environ.get("YASAKANI_METAL_LIBRARY"):
        path = Path(explicit or os.environ["YASAKANI_METAL_LIBRARY"]).expanduser().resolve()
        if not path.is_file():
            raise YasakaniError("Metal library not found: %s" % path)
        return path
    here = Path(__file__).resolve().parent
    for directory in (here / "bin", here.parent / "bin", here.parents[1] / "build" / "bin"):
        path = directory / "libyskn_metal.dylib"
        if path.is_file():
            return path
    raise YasakaniError("libyskn_metal.dylib not found; build with CMake on macOS, "
                        "or use the macOS app bundle (see docs/MACOS.md)")


class MetalOps:
    def __init__(self, path, library=None):
        if sys.platform != "darwin":
            raise YasakaniError("the Metal backend requires macOS")
        self.container = Container(path)
        self._vectors = {}
        self.handle = None
        try:
            self.lib = C.CDLL(str(find_metal_library(library)))
            signatures = {
                "open": ([C.c_char_p], C.c_void_p),
                "close": ([C.c_void_p], None),
                "linear": ([C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int], C.c_int),
                "embedding": ([C.c_void_p, C.c_int, C.c_void_p, C.c_int], C.c_int),
                "last_error": ([], C.c_char_p),
            }
            for name, (args, result) in signatures.items():
                fn = getattr(self.lib, "yskn_metal_" + name)
                fn.argtypes, fn.restype = args, result
            self._batch = getattr(self.lib, "yskn_metal_linear_batch", None)
            if self._batch is not None:
                self._batch.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int]
                self._batch.restype = C.c_int
            self._multi = getattr(self.lib, "yskn_metal_linear_multi", None)
            if self._multi is not None:
                self._multi.argtypes = [C.c_void_p, C.POINTER(C.c_char_p), C.c_int, C.c_void_p,
                                        C.c_int, C.c_void_p, C.c_int]
                self._multi.restype = C.c_int
            self.handle = self.lib.yskn_metal_open(os.fsencode(path))
            if not self.handle:
                raise YasakaniError(self.error())
        except (OSError, AttributeError) as exc:
            raise YasakaniError("could not load the Metal backend: %s" % exc) from exc

    def error(self):
        message = self.lib.yskn_metal_last_error()
        return message.decode("utf-8", "replace") if message else "Metal operation failed"

    def close(self):
        if self.handle:
            self.lib.yskn_metal_close(self.handle)
            self.handle = None

    def _check(self, result):
        if result:
            raise YasakaniError(self.error())

    def linear(self, name, x):
        if not self.handle:
            raise YasakaniError("the Metal backend is closed")
        desc = self.container.tensors[name]
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (desc["cols"],):
            raise ValueError("%s: input shape mismatch" % name)
        out = np.empty(desc["rows"], dtype=np.float32)
        self._check(self.lib.yskn_metal_linear(self.handle, name.encode(), x.ctypes.data, len(x), out.ctypes.data, len(out)))
        return out

    def embedding(self, token):
        if not self.handle:
            raise YasakaniError("the Metal backend is closed")
        if not 0 <= token < self.container.model["vocab"]:
            raise ValueError("token is outside the model vocabulary")
        out = np.empty(self.container.model["hidden"], dtype=np.float32)
        self._check(self.lib.yskn_metal_embedding(self.handle, int(token), out.ctypes.data, len(out)))
        return out

    def linear_batch(self, name, x):
        """A bounded group of tokens sharing the same compressed weight matrix."""
        if not self.handle:
            raise YasakaniError("the Metal backend is closed")
        desc = self.container.tensors[name]
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != desc["cols"] or not 1 <= len(x) <= 32:
            raise ValueError("%s: expected 1 to 32 rows with %d columns" % (name, desc["cols"]))
        if self._batch is None:
            return np.stack([self.linear(name, row) for row in x])
        out = np.empty((len(x), desc["rows"]), dtype=np.float32)
        self._check(self._batch(self.handle, name.encode(), x.ctypes.data, len(x), x.shape[1],
                                out.ctypes.data, out.shape[1]))
        return out

    def linear_multi(self, names, x):
        """Independent projections of one input, submitted together to Metal."""
        if not self.handle:
            raise YasakaniError("the Metal backend is closed")
        names = tuple(names)
        if not 1 <= len(names) <= 3:
            raise ValueError("expected 1 to 3 projection names")
        x = np.ascontiguousarray(x, dtype=np.float32)
        descriptors = [self.container.tensors[name] for name in names]
        if x.ndim != 1 or any(x.shape != (desc["cols"],) for desc in descriptors):
            raise ValueError("multi-projection input shape mismatch")
        if self._multi is None:
            return [self.linear(name, x) for name in names]
        sizes = [desc["rows"] for desc in descriptors]
        out = np.empty(sum(sizes), dtype=np.float32)
        encoded = (C.c_char_p * len(names))(*(name.encode() for name in names))
        self._check(self._multi(self.handle, encoded, len(names), x.ctypes.data, len(x),
                                out.ctypes.data, len(out)))
        return list(np.split(out, np.cumsum(sizes)[:-1]))

    def vector(self, name):
        if name not in self._vectors:
            self._vectors[name] = vector(self.container, self.container.vectors[name])
        return self._vectors[name]


class MetalEngine:
    """ChatRunner-compatible sequential engine; each request starts with a fresh state."""
    supports_visual = False

    def __init__(self, pack, *, max_ctx=None, device=0, dll=None, model=None):
        if device != 0:
            raise YasakaniError("Metal uses the system default GPU; --device must be 0")
        self.pack = pack
        self.max_ctx = min(4096, pack.max_ctx) if max_ctx is None else int(max_ctx)
        if not CONTEXT_MARGIN < self.max_ctx <= pack.max_ctx:
            raise YasakaniError("--max-ctx must be between %d and %d" % (CONTEXT_MARGIN + 1, pack.max_ctx))
        self.removed_env = []
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._closed = False
        self._ops = None
        self.sampling = Sampling().normalized()
        try:
            if model is None:
                from .macos_model import Model
                self._ops = MetalOps(pack.weights, dll)
                model = Model(pack.weights, self.max_ctx, ops=self._ops)
            self.model = model
        except Exception as exc:
            if self._ops:
                self._ops.close()
            if isinstance(exc, YasakaniError):
                raise
            raise YasakaniError("could not open the Metal model: %s" % exc) from exc

    def _need_open(self):
        if self._closed:
            raise YasakaniError("the engine is closed")

    def close(self):
        self.cancel()
        with self._lock:
            if self._ops:
                self._ops.close()
                self._ops = None
            self.model = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def cancel(self):
        self._cancel.set()

    def reset(self):
        with self._lock:
            self._need_open()
            self.model.reset()

    def position(self):
        self._need_open()
        return self.model.position

    def state_hash(self):
        with self._lock:
            self._need_open()
            return self.model.state_hash()

    def set_sampling(self, sampling):
        normalized = sampling.normalized()
        if (not all(np.isfinite(v) for v in (normalized.temperature, normalized.top_p, normalized.min_p))
                or not 0 < normalized.top_p <= 1 or not 0 <= normalized.min_p <= 1):
            raise ValueError("invalid sampling parameters")
        with self._lock:
            self._need_open()
            self.sampling = normalized
        return normalized

    def _sample(self, logits, rng):
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 1 or not np.all(np.isfinite(logits)):
            raise YasakaniError("the model produced invalid logits")
        s = self.sampling
        if s.temperature <= 0:
            return int(logits.argmax())
        k = min(s.top_k, len(logits))
        # Stable tie ordering, independent of the platform's partial-sort implementation.
        order = np.argsort(-logits, kind="stable")[:k]
        weights = np.exp((logits[order] - logits[order[0]]) / s.temperature)
        keep = weights >= weights[0] * s.min_p
        order, weights = order[keep], weights[keep]
        weights /= weights.sum()
        count = min(len(weights), int(np.searchsorted(np.cumsum(weights), s.top_p)) + 1)
        weights = weights[:count]
        return int(rng.choice(order[:count], p=weights / weights.sum()))

    def generate(self, prompt, max_new, *, stop_ids=EOS_IDS, on_tokens=None, spec=None, poll_s=0.001):
        if spec:
            raise ValueError("speculative decoding options are unavailable on Metal")
        ids = np.asarray(prompt)
        if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("the prompt must be a nonempty integer token vector")
        if max_new < 1:
            raise ValueError("max_new must be positive")
        room = self.max_ctx - len(ids) - CONTEXT_MARGIN
        if room < 1:
            raise ValueError("the prompt leaves no room in the context")
        limit = min(int(max_new), room + 1)
        stopped = set(() if stop_ids is None else stop_ids)
        with self._lock:
            self._need_open()
            self._cancel.clear()
            self.model.reset()
            rng = np.random.default_rng(self.sampling.seed & ((1 << 64) - 1))
            out = []
            start = time.perf_counter()
            logits = None
            if hasattr(self.model, "prefill"):
                for first in range(0, len(ids), 32):
                    if self._cancel.is_set():
                        break
                    last = min(first + 32, len(ids))
                    logits = self.model.prefill(ids[first:last], cancel=self._cancel.is_set, logits=last == len(ids))
            else:
                for index, token in enumerate(ids):
                    if self._cancel.is_set():
                        break
                    logits = self.model.forward(int(token), logits=index == len(ids) - 1)
            prefill = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            finish = "length"
            for _ in range(limit):
                if self._cancel.is_set():
                    finish = "cancelled"
                    break
                token = self._sample(logits, rng)
                out.append(token)
                if on_tokens:
                    on_tokens(np.asarray(out, dtype=np.int32))
                if token in stopped:
                    finish = "stop"
                    break
                if self._cancel.is_set():
                    finish = "cancelled"
                    break
                if len(out) < limit:
                    logits = self.model.forward(token)
            return GenResult(out, finish, len(ids), 0, 0, len(out), prefill,
                             (time.perf_counter() - start) * 1000,
                             {"passes": float(len(out)), "cached_tokens": 0.0})

    def warmup(self, prompts, n_gen=32):
        # Metal pipelines are compiled on open; a single real token is sufficient.
        self.set_sampling(Sampling(temperature=0))
        for prompt in prompts[:1]:
            self.generate(prompt[:1], 1)
        self.reset()
