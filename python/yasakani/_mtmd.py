# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""ctypes bindings to the multimodal library of llama.cpp (``mtmd.dll``), checked against the b10488 release for Windows.

The projector (mmproj) turns an image into features of 5120 values per image token.  The library also needs the text model of
llama.cpp, for its settings and vocabulary only: the model is opened with ``no_alloc``, so its weights are never read and any
Qwen3.8-27B GGUF works.  The images are encoded on the CPU: on the GPU the features differ slightly and the image score of our test
fell (87.8 on the CPU against 79.6 and 81.6 in two GPU runs).
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import time

import numpy as np

P, I, S, B = C.c_void_p, C.c_int32, C.c_size_t, C.c_bool


class _ModelParams(C.Structure):
    _fields_ = [("devices", P), ("tensor_buft_overrides", P), ("n_gpu_layers", I),
                ("split_mode", I), ("load_mode", I), ("main_gpu", I), ("tensor_split", P),
                ("progress_callback", P), ("progress_callback_user_data", P), ("kv_overrides", P),
                ("vocab_only", B), ("check_tensors", B), ("use_extra_bufts", B),
                ("no_host", B), ("no_alloc", B), ("load_mtp", B)]


class _ContextParams(C.Structure):
    _fields_ = [("use_gpu", B), ("print_timings", B), ("n_threads", I), ("image_marker", C.c_char_p),
                ("media_marker", C.c_char_p), ("flash_attn_type", I), ("warmup", B),
                ("image_min_tokens", I), ("image_max_tokens", I), ("cb_eval", P),
                ("cb_eval_user_data", P), ("batch_max_tokens", I), ("progress_callback", P),
                ("progress_callback_user_data", P)]


class _InputText(C.Structure):
    _fields_ = [("text", C.c_char_p), ("text_len", S), ("add_special", B), ("parse_special", B)]


class _DecoderPos(C.Structure):
    _fields_ = [("t", C.c_uint32), ("x", C.c_uint32), ("y", C.c_uint32), ("z", C.c_uint32)]


def _bind(lib, name, args, result):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, result
    return fn


def to_bfloat16(values) -> np.ndarray:
    """float32 -> bfloat16 bits, rounded to nearest even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


class MtmdEncoder:
    HIDDEN = 5120

    def __init__(self, llama_dir, mmproj, model, *, threads: int = 8):
        llama_dir, mmproj, model = Path(llama_dir), Path(mmproj), Path(model)
        for p in (llama_dir / "llama.dll", llama_dir / "mtmd.dll", llama_dir / "ggml.dll", mmproj, model):
            if not p.is_file():
                raise FileNotFoundError("vision: missing %s" % p)
        self.ctx = self.model = None
        self._dll_dir = os.add_dll_directory(str(llama_dir))
        # ggml_backend_load_all_from_path loads the backends with a plain LoadLibrary: without the folder on PATH their
        # dependencies are not found and the backends are skipped without an error.
        if str(llama_dir).lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = str(llama_dir) + os.pathsep + os.environ.get("PATH", "")
        self.llama = C.CDLL(str(llama_dir / "llama.dll"))
        self.lib = C.CDLL(str(llama_dir / "mtmd.dll"))
        ggml = C.CDLL(str(llama_dir / "ggml.dll"))
        _bind(ggml, "ggml_backend_load_all_from_path", [C.c_char_p], None)(str(llama_dir).encode())
        _bind(self.llama, "llama_backend_init", [], None)()
        mp = _bind(self.llama, "llama_model_default_params", [], _ModelParams)()
        mp.no_alloc, mp.n_gpu_layers, mp.load_mode = True, 0, 0     # settings and vocabulary only; no_alloc cannot memory-map
        self.model = _bind(self.llama, "llama_model_load_from_file", [C.c_char_p, _ModelParams], P)(str(model).encode(), mp)
        if not self.model:
            raise RuntimeError("vision: cannot open the text model %s" % model)
        hidden = _bind(self.llama, "llama_model_n_embd_inp", [P], I)(self.model)
        if hidden != self.HIDDEN:
            self.close()
            raise RuntimeError("vision: the text model has %d input dimensions, expected %d (Qwen3.8-27B)" % (hidden, self.HIDDEN))
        sig = {
            "mtmd_context_params_default": ([], _ContextParams),
            "mtmd_init_from_file": ([C.c_char_p, P, _ContextParams], P),
            "mtmd_free": ([P], None), "mtmd_default_marker": ([], C.c_char_p),
            "mtmd_bitmap_init": ([C.c_uint32, C.c_uint32, P], P), "mtmd_bitmap_free": ([P], None),
            "mtmd_input_chunks_init": ([], P), "mtmd_input_chunks_free": ([P], None),
            "mtmd_input_chunks_size": ([P], S), "mtmd_input_chunks_get": ([P, S], P),
            "mtmd_input_chunk_get_type": ([P], I), "mtmd_input_chunk_get_n_tokens": ([P], S),
            "mtmd_input_chunk_get_n_pos": ([P], I), "mtmd_input_chunk_get_tokens_image": ([P], P),
            "mtmd_image_tokens_get_decoder_pos": ([P, I, S], _DecoderPos),
            "mtmd_tokenize": ([P, P, C.POINTER(_InputText), C.POINTER(P), S], I),
            "mtmd_encode_chunk": ([P, P], I), "mtmd_get_output_embd": ([P], C.POINTER(C.c_float)),
        }
        self.f = {k: _bind(self.lib, k, *v) for k, v in sig.items()}
        cp = self.f["mtmd_context_params_default"]()
        if (cp.n_threads, cp.image_min_tokens, cp.image_max_tokens, cp.batch_max_tokens) != (4, -1, -1, 1024):
            self.close()
            raise RuntimeError("vision: mtmd.dll is not the expected llama.cpp release (b10488): structure layout differs")
        cp.use_gpu, cp.warmup, cp.n_threads = False, False, int(threads)
        self.ctx = self.f["mtmd_init_from_file"](str(mmproj).encode(), self.model, cp)
        if not self.ctx:
            self.close()
            raise RuntimeError("vision: cannot load the projector %s" % mmproj)

    def encode(self, image) -> dict:
        """RGB PIL image -> features (bfloat16 bits, (n, 5120)), positions ([time, row, column], (3, n)), grid (rows, columns)."""
        rgb = np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
        f = self.f
        bitmap = f["mtmd_bitmap_init"](rgb.shape[1], rgb.shape[0], rgb.ctypes.data)
        chunks = f["mtmd_input_chunks_init"]()
        t0 = time.perf_counter()
        try:
            marker = f["mtmd_default_marker"]()
            text = _InputText(marker, len(marker), False, True)
            if f["mtmd_tokenize"](self.ctx, chunks, C.byref(text), (P * 1)(bitmap), 1):
                raise ValueError("vision: image preprocessing failed")
            chunk = [f["mtmd_input_chunks_get"](chunks, i) for i in range(f["mtmd_input_chunks_size"](chunks))]
            chunk = [c for c in chunk if f["mtmd_input_chunk_get_type"](c) == 1]
            if len(chunk) != 1:
                raise ValueError("vision: expected one image chunk, got %d" % len(chunk))
            chunk = chunk[0]
            n = f["mtmd_input_chunk_get_n_tokens"](chunk)
            tokens = f["mtmd_input_chunk_get_tokens_image"](chunk)
            pos = [f["mtmd_image_tokens_get_decoder_pos"](tokens, 0, i) for i in range(n)]
            positions = np.array([(p.t, p.y, p.x) for p in pos], dtype=np.int32).T.copy()   # .y is the row, .x the column
            h, w = int(positions[1].max()) + 1, int(positions[2].max()) + 1
            expected = np.stack([np.zeros(n, np.int32), np.arange(n, dtype=np.int32) // w, np.arange(n, dtype=np.int32) % w])
            if n != h * w or not np.array_equal(positions, expected) or f["mtmd_input_chunk_get_n_pos"](chunk) != max(h, w):
                raise ValueError("vision: unexpected image positions from mtmd")
            if f["mtmd_encode_chunk"](self.ctx, chunk):
                raise RuntimeError("vision: mtmd encode failed")
            feats = np.ctypeslib.as_array(f["mtmd_get_output_embd"](self.ctx), shape=(n * self.HIDDEN,)).reshape(n, self.HIDDEN).copy()
            if not np.isfinite(feats).all():
                raise RuntimeError("vision: non-finite image features")
            return dict(features=to_bfloat16(feats), positions=positions, grid=np.array([h, w], dtype=np.int32),
                        seconds=np.array(time.perf_counter() - t0))
        finally:
            f["mtmd_input_chunks_free"](chunks)
            f["mtmd_bitmap_free"](bitmap)

    def close(self) -> None:
        if self.ctx:
            self.f["mtmd_free"](self.ctx)
            self.ctx = None
        if self.model:
            _bind(self.llama, "llama_model_free", [P], None)(self.model)
            self.model = None
