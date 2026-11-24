# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""ctypes binding of the engine library (yskn.dll).

Only the exports the Python layer needs are bound.  The library is loaded by absolute path; it needs the Visual C++ runtime and an
NVIDIA driver, nothing else (the CUDA runtime is linked in statically).
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

P = C.c_void_p
I = C.c_int


class YasakaniError(RuntimeError):
    """An engine or setup error with a message meant for the person running the program."""


# name -> (argument types, result type).  Pointers are passed as plain integers (numpy ``arr.ctypes.data``).
SIGNATURES: Dict[str, Tuple[list, object]] = {
    "yskn_open": ([C.c_char_p, I, I, I], P),
    "yskn_close": ([P], None),
    "yskn_reset": ([P], I),
    "yskn_prefill": ([P, P, I, P], I),
    "yskn_decode": ([P, P, I], I),
    "yskn_logits": ([P, P], I),
    "yskn_position": ([P], I),
    "yskn_nll": ([P, P, I, P], I),
    "yskn_last_error": ([], C.c_char_p),
    "yskn_cancel": ([P], I),
    "yskn_set_opt": ([P, I, I], I),
    "yskn_load_bdraft": ([P, C.c_char_p], I),
    "yskn_spec_generate": ([P, P, I, I, P, I, P, P, P], I),
    "yskn_state_hash": ([P, P, I, P], I),
}

# The multi-stream pass (several requests in one pass; 0.1.7): experimental entries that are not in include/yskn_abi.h.  They are
# looked up at run time; a library without all of them serves one request at a time (``NativeLib.supports_parallel`` is False).
MS_SIGNATURES: Dict[str, Tuple[list, object]] = {
    "yskn_ms_info": ([P, P, I, P], I),
    "yskn_ms_bind2": ([P, P, I, I], I),
    "yskn_ms_opt": ([P, I, I], I),
    "yskn_ms_load": ([P, I, P], I),
    "yskn_ms_unload": ([P, I, P, I], I),
    "yskn_live_drop": ([P], I),
    "yskn_ms_set_sampling": ([P, I, I, I, I, I, I, C.c_uint32], I),
    "yskn_ms_draft": ([P, I, P, P, P, I], I),
    "yskn_ms_step2": ([P, I, P, P, I, P, P, P, P], I),
    "yskn_ms_commit": ([P, I, P, P], I),
    "yskn_ms_flush": ([P], I),
    "yskn_ms_slot_pos": ([P, I, P], I),
    "yskn_ms_slot_hash": ([P, I, P, I, P], I),
}

_VC_HINT = ("The engine library could not be loaded.  On Windows it needs the Microsoft Visual C++ 2015-2022 x64 runtime "
            "(https://aka.ms/vs/17/release/vc_redist.x64.exe) and a recent NVIDIA driver.  Details: ")


def _candidates(explicit: Optional[str]):
    if explicit:
        yield Path(explicit)
        return
    env = os.environ.get("YASAKANI_DLL")
    if env:
        yield Path(env)
        return
    name = "yskn.dll" if sys.platform == "win32" else "libyskn.so"
    here = Path(__file__).resolve().parent
    for d in (here / "bin", here.parent / "bin", here.parents[1] / "build" / "bin", here.parents[1] / "build" / "Release",
              here.parents[1] / "bin"):
        yield d / name


def find_library(explicit: Optional[str] = None) -> Path:
    """The engine library: the given path, ``YASAKANI_DLL``, or a ``bin`` / ``build`` folder next to the package."""
    tried = []
    for p in _candidates(explicit):
        tried.append(str(p))
        if p.is_file():
            return p.resolve()
    raise YasakaniError("engine library (yskn.dll) not found; set YASAKANI_DLL or pass --dll.  Looked in: " + "; ".join(tried))


def pci_bus_order(environ=None) -> None:
    """Let CUDA number the GPUs the way nvidia-smi and ``yasakani info`` do: in PCI bus order (CUDA's own default puts the fastest
    first, so on a PC with two GPUs ``--device 0`` could be nvidia-smi's GPU 1).  CUDA reads ``CUDA_DEVICE_ORDER`` once, when it
    starts in the process; the engine starts it at the first call of ``yskn_open``, so this is called before the library is loaded.
    A value that is already set is kept.  Nothing changes on a PC with one GPU."""
    env = os.environ if environ is None else environ
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


_LOADED = []                                     # paths of the engine libraries loaded in this process


def library_loaded() -> bool:
    """True once an engine library has been loaded in this process.  CUDA may have started then, and it does not read
    ``CUDA_DEVICE_ORDER`` or ``CUDA_VISIBLE_DEVICES`` again."""
    return bool(_LOADED)


class NativeLib:
    """The loaded library with typed entry points in ``fn``.  ``last_error()`` is thread-local in the engine: call it on the thread
    that made the failing call."""

    def __init__(self, path: Optional[str] = None):
        self.path = find_library(path)
        pci_bus_order()                          # before CUDA starts in this process: --device counts as nvidia-smi does
        try:
            self.dll = C.CDLL(str(self.path))
        except OSError as ex:
            raise YasakaniError(_VC_HINT + str(ex)) from ex
        _LOADED.append(str(self.path))
        self.fn = {}
        missing = []
        for name, (argtypes, restype) in SIGNATURES.items():
            f = getattr(self.dll, name, None)
            if f is None:
                missing.append(name)
                continue
            f.argtypes = argtypes
            f.restype = restype
            self.fn[name] = f
        if missing:
            raise YasakaniError("the engine library is missing exports: " + ", ".join(missing))
        # Optional so text-only users can keep using older engine libraries.
        f = getattr(self.dll, "yskn_set_visual", None)
        if f is not None:
            f.argtypes, f.restype = [P, P, P, I, P, I, I], I
            self.fn["yskn_set_visual"] = f
        f = getattr(self.dll, "yskn_ink2_encode_rgb", None)
        if f is not None:
            f.argtypes, f.restype = [P, P, P, P, I, P, P, P], I
            self.fn["yskn_ink2_encode_rgb"] = f
        f = getattr(self.dll, "yskn_set_suppress", None)
        if f is not None:
            f.argtypes, f.restype = [P, P, I], I
            self.fn["yskn_set_suppress"] = f
        self.supports_suppress = "yskn_set_suppress" in self.fn
        f = getattr(self.dll, "yskn_prepare_wide", None)      # 0.1.4 and newer
        if f is not None:
            f.argtypes, f.restype = [P, I], I
            self.fn["yskn_prepare_wide"] = f
        # 0.1.7: wide copy passes per GPU (yasakani/widecheck.py); older libraries have none of them
        for name, sig in (("yskn_time_cycle", ([P, P, I, I, P], I)), ("yskn_wide_state", ([P, P, I, P], I)),
                          ("yskn_build_info", ([], C.c_char_p))):
            f = getattr(self.dll, name, None)
            if f is not None:
                f.argtypes, f.restype = sig
                self.fn[name] = f
        # teacher forcing (the self-check on GPUs other than the RTX 5090; every released engine library has them)
        for name, sig in (("yskn_verify_nll", ([P, P, P, I, I, P, P], I)), ("yskn_verify", ([P, P, I, I, P, P, P], I)),
                          ("yskn_commit", ([P, I], I)), ("yskn_logits_rows", ([P, I, I, P], I))):
            f = getattr(self.dll, name, None)
            if f is not None:
                f.argtypes, f.restype = sig
                self.fn[name] = f
        # the multi-stream pass (optional, as yskn_set_visual): all of its entries or none are used
        found = {name: getattr(self.dll, name, None) for name in MS_SIGNATURES}
        self.supports_parallel = all(f is not None for f in found.values())
        if self.supports_parallel:
            for name, f in found.items():
                f.argtypes, f.restype = MS_SIGNATURES[name]
                self.fn[name] = f

    def build_info(self) -> str:
        """``cuda=13020;wide=1`` (0.1.7 and newer), "" for older libraries."""
        f = self.fn.get("yskn_build_info")
        if f is None:
            return ""
        m = f()
        return m.decode("utf-8", "replace") if m else ""

    def last_error(self) -> str:
        m = self.fn["yskn_last_error"]()
        return m.decode("utf-8", "replace") if m else ""
