#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""vision_stage_check.py - check the GPU image encoder of Inkling-2 Mini stage by stage against reference arrays.

  python vision_stage_check.py --lib yskn_inkling2.dll --model MODEL_DIR --weights ORIGINAL_DIR --golden DIR
         IMAGE.png [IMAGE.png ...] [--time IMAGE.png ...] [--cap 4096] [--memory-caps 1024,4096] [--reps 3]
         [--python PACKAGE_DIR] [--json OUT.json]

  --lib      the engine library (yskn_inkling2.dll); only its exported calls are used (ink2_vision_*, ink2_set_option)
  --model    a converted Inkling-2 Mini model folder; the text model is loaded through the yasakani package, because
             the image tower lives next to a loaded text model (ink2_vision_create comes after ink2_finalize)
  --weights  the original model folder: config.json and the *.safetensors file(s) holding the 422 model.visual.* tensors
             (only those byte ranges are read, through the file header and offsets); default: --model
  --golden   folder of reference arrays NAME_{pixel_values,patch_proj,embeddings,block1,block23,merged,output}.npy
             (float32, sequence order of the reference implementation), NAME = the image file name without extension
  --python   folder that holds the yasakani package (default: the python folder of this repository)
  IMAGE.png  images compared stage by stage; --time: images only timed (mean of --reps encodes after a warm-up)

What it checks, in one run:
  stages 0..6 of every image against the reference (pixel_values exact; per-token cosine for patch_proj, embeddings
  and block1; global relative L2, token cosines and max abs for block23, merged and output) and the BF16 feature rows
  read back with ink2_vision_get_rows; two encodes of the same image give the same bits; a wrong config value, a wrong
  dtype, a wrong tensor shape, an unknown name, a wrong byte count, a duplicate and a missing tensor are refused with a
  message; an image that does not fit the feature buffer is refused and writes nothing; time per encode; device memory
  of the weights, the scratch and the feature buffer at the given caps. Exit code 0 when every check passes, 1 otherwise.
"""
import argparse
import ctypes as C
import glob
import json
import os
import struct
import sys
import time

import numpy as np
from PIL import Image

STAGES = ["pixel_values", "patch_proj", "embeddings", "block1", "block23", "merged", "output"]
HIDDEN_OUT = 2304
# Tolerances against an FP32 reference (encoder with BF16 matrix inputs and FP32 sums and residual stream).
TOL_TOKEN_COS = {"patch_proj": 0.9999, "embeddings": 0.9999, "block1": 0.9995}
TOL_OUT = {"global_rel_l2": 0.06, "mean_cos": 0.998, "p1_cos": 0.98, "min_cos": 0.85, "max_abs": 0.25}


class Cfg(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("hidden", "layers", "heads", "head_dim", "inter", "patch", "merge", "temporal",
                                        "grid", "proj_hidden", "text_hidden", "projector_bias")] + \
               [("eps", C.c_float), ("rope_theta", C.c_float), ("max_image_tokens", C.c_int),
                ("max_feature_rows", C.c_int)]


def make_cfg(config, cap, rows=None):
    v, t = config["vision_config"], config.get("text_config", {})
    hidden, heads = int(v["hidden_size"]), int(v["num_attention_heads"])
    th = int(v["text_hidden_size"])
    if t.get("hidden_size", th) != th:
        raise SystemExit("config: vision_config.text_hidden_size %d differs from text_config.hidden_size %s" % (th, t.get("hidden_size")))
    if v.get("hidden_act", "gelu") != "gelu" or v.get("num_channels", 3) != 3:
        raise SystemExit("config: hidden_act / num_channels not supported")
    return Cfg(hidden, int(v["num_hidden_layers"]), heads, hidden // heads, int(round(hidden * float(v["mlp_ratio"]))),
               int(v["native_patch_size"]), int(v["adapter_merge_size"]), int(v["temporal_patch_size"]),
               int(v["posemb_grid_size"]), int(v["projection_hidden_dim"]), th, 1 if v["projector_bias"] else 0,
               float(v["layer_norm_eps"]), float(v.get("rope_theta", 10000.0)), cap, rows or cap)


def visual_entries(model_dir):
    """name -> (file, absolute offset, nbytes, dtype, shape) for every model.visual.* tensor."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100 << 20:
                raise SystemExit("%s: header of %d bytes, not a safetensors file" % (path, n))
            hdr = json.loads(f.read(n))
        for k, e in hdr.items():
            if not k.startswith("model.visual."):
                continue
            if k in out:
                raise SystemExit("tensor %s appears twice" % k)
            a, b = e["data_offsets"]
            out[k] = (path, 8 + n + a, b - a, e["dtype"], e["shape"])
    return out


def read_bytes(entry):
    path, off, nb = entry[:3]
    with open(path, "rb") as f:
        f.seek(off)
        data = f.read(nb)
    if len(data) != nb:
        raise SystemExit("%s: file ends early" % path)
    return data


def bind(L):
    vp, cp, ll, ci = C.c_void_p, C.c_char_p, C.c_longlong, C.c_int
    L.ink2_last_error.restype = cp
    L.ink2_last_error.argtypes = [vp]
    L.ink2_set_option.argtypes = [vp, cp, ci]
    L.ink2_vision_create.argtypes = [vp, C.POINTER(Cfg)]
    L.ink2_vision_set_tensor.argtypes = [vp, cp, vp, ll]
    L.ink2_vision_set_tensor_shaped.argtypes = [vp, cp, cp, C.POINTER(ll), ci, vp, ll]
    L.ink2_vision_finalize.argtypes = [vp]
    L.ink2_vision_encode.argtypes = [vp, vp, ci, ci, ll, ci, ci, ci]
    L.ink2_vision_get_rows.argtypes = [vp, vp, ci, ci, ll]
    L.ink2_vision_stage.argtypes = [vp, ci, vp, ll]
    L.ink2_vision_memory.argtypes = [vp, C.POINTER(ll), C.POINTER(ll), C.POINTER(ll)]
    L.ink2_vision_unload.argtypes = [vp]
    L.ink2_device_memory.argtypes = [vp, C.POINTER(ll), C.POINTER(ll)]


class Tower:
    """The image tower of one loaded engine, through the exported calls."""

    def __init__(self, L, h):
        self.L, self.h = L, h

    def msg(self):
        m = self.L.ink2_last_error(self.h)
        return m.decode("utf-8", "replace") if m else ""

    def must(self, rc, what):
        if rc != 0:
            raise RuntimeError("%s: %s" % (what, self.msg()))

    def create(self, cfg, capture):
        self.must(self.L.ink2_set_option(self.h, b"vision_capture", 1 if capture else 0), "vision_capture")
        return self.L.ink2_vision_create(self.h, C.byref(cfg))

    def set(self, name, data, shape, dtype="BF16"):
        sh = (C.c_longlong * len(shape))(*shape)
        return self.L.ink2_vision_set_tensor_shaped(self.h, name.encode(), dtype.encode(), sh, len(shape), data, len(data))

    def upload(self, entries):
        for name, e in sorted(entries.items(), key=lambda kv: (kv[1][0], kv[1][1])):
            self.must(self.set(name, read_bytes(e), e[4], e[3]), "set_tensor")

    def encode(self, rgb, row0=0):
        h, w, _ = rgb.shape
        rows, cols = -(-h // 40), -(-w // 40)
        buf = np.ascontiguousarray(rgb, dtype=np.uint8)
        return self.L.ink2_vision_encode(self.h, buf.ctypes.data, w, h, buf.nbytes, rows, cols, row0), rows * cols

    def rows(self, n, row0=0):
        a = np.empty((n, HIDDEN_OUT), np.float32)
        self.must(self.L.ink2_vision_get_rows(self.h, a.ctypes.data, row0, n, a.size), "get_rows")
        return a

    def stage(self, k, n):
        size = {0: n * 4800, 1: 4 * n * 1024, 2: (1 + 4 * n) * 1024, 3: (1 + 4 * n) * 1024, 4: (1 + 4 * n) * 1024,
                5: n * 4096, 6: n * HIDDEN_OUT}[k]
        a = np.empty(size, np.float32)
        self.must(self.L.ink2_vision_stage(self.h, k, a.ctypes.data, a.size), "stage %d" % k)
        return a

    def memory(self):
        w, s, f = C.c_longlong(), C.c_longlong(), C.c_longlong()
        self.must(self.L.ink2_vision_memory(self.h, C.byref(w), C.byref(s), C.byref(f)), "memory")
        return w.value, s.value, f.value

    def time(self, rgb, reps):
        """Milliseconds per encode (RGB upload included): reps encodes queued back to back, then one row read back,
        which waits for them."""
        self.must(self.encode(rgb)[0], "encode")
        self.rows(1)
        t = time.perf_counter()
        for _ in range(reps):
            self.must(self.encode(rgb)[0], "encode")
        self.rows(1)
        return (time.perf_counter() - t) * 1000.0 / reps

    def unload(self):
        self.must(self.L.ink2_vision_unload(self.h), "unload")


def read_rgb(path):
    """uint8 [H, W, 3]: transparency composited on the 8-pixel 204 / 153 checkerboard, else converted to RGB."""
    with Image.open(path) as im:
        im.load()
        if im.has_transparency_data:
            a = np.asarray(im.convert("RGBA"), dtype=np.int64)
            yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
            bg = np.where((yy // 8 + xx // 8) % 2 == 0, 204, 153)[:, :, None]
            al = a[:, :, 3:4]
            return ((a[:, :, :3] * al + bg * (255 - al) + 127) // 255).astype(np.uint8)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def stats(ours, ref):
    a, b = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    a, b = a.reshape(b.shape[0], -1), b.reshape(b.shape[0], -1)
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
    return {"max_abs": float(np.abs(a - b).max()), "global_rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
            "min_cos": float(cos.min()), "p1_cos": float(np.percentile(cos, 1)), "mean_cos": float(cos.mean())}


def judge(stage, s):
    if stage == "pixel_values":
        return s["max_abs"] == 0.0
    if stage in TOL_TOKEN_COS:
        return s["min_cos"] >= TOL_TOKEN_COS[stage]
    t = TOL_OUT
    return (s["global_rel_l2"] <= t["global_rel_l2"] and s["mean_cos"] >= t["mean_cos"] and s["p1_cos"] >= t["p1_cos"]
            and s["min_cos"] >= t["min_cos"] and s["max_abs"] <= t["max_abs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--python", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
    ap.add_argument("--time", nargs="*", default=[])
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--memory-caps", default="1024,4096")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args()
    weights = a.weights or a.model
    with open(os.path.join(weights, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    entries = visual_entries(weights)
    if len(entries) != 422:
        raise SystemExit("found %d model.visual.* tensors, expected 422" % len(entries))
    sys.path.insert(0, os.path.abspath(a.python))
    from yasakani import inkling2 as ink  # the text model is loaded the way the package loads it

    t0 = time.perf_counter()
    eng = ink.Ink2Engine(ink.Ink2Pack.load(a.model), 4096, dll=os.path.abspath(a.lib))
    L, h = eng.native.lib, eng.native.h
    bind(L)
    tw = Tower(L, h)
    rep = {"images": {}, "checks": {}, "timing_ms": {}, "memory": {}, "text_load_s": time.perf_counter() - t0}
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        rep["checks"][name] = {"pass": bool(cond), "detail": detail}
        ok &= bool(cond)
        print("%-22s %s  %s" % (name, "PASS" if cond else "FAIL", detail))

    # 1. a wrong config value is refused with a message naming it
    bad = make_cfg(config, a.cap)
    bad.layers = 22
    rc = tw.create(bad, False)
    check("config_layers_22", rc != 0 and "layers" in tw.msg(), tw.msg())
    if rc == 0:
        tw.unload()

    # 2. tensor refusals (before the real upload), then all 422 and finalize with the stages kept
    t0 = time.perf_counter()
    tw.must(tw.create(make_cfg(config, a.cap), True), "create")
    pe = "model.visual.backbone.embeddings.patch_embedding.weight"
    shape = entries[pe][4]
    pe_bytes = read_bytes(entries[pe])
    rc = tw.set(pe, pe_bytes, [shape[1], shape[0]])
    check("wrong_shape", rc != 0 and pe in tw.msg() and "shape" in tw.msg(), tw.msg())
    rc = tw.set(pe, pe_bytes, shape, "F16")
    check("wrong_dtype", rc != 0 and "BF16" in tw.msg(), tw.msg())
    rc = tw.set("model.visual.backbone.layers.23.ln_1.weight", pe_bytes[:2048], [1024])
    check("unknown_name", rc != 0 and "unknown" in tw.msg(), tw.msg())
    rc = L.ink2_vision_set_tensor(h, pe.encode(), pe_bytes, len(pe_bytes) - 2)
    check("wrong_nbytes", rc != 0 and "bytes" in tw.msg(), tw.msg())
    rc = L.ink2_vision_finalize(h)
    check("finalize_missing", rc != 0 and "missing" in tw.msg(), tw.msg())
    tw.upload(entries)
    rc = tw.set(pe, pe_bytes, shape)
    check("duplicate", rc != 0 and "twice" in tw.msg(), tw.msg())
    tw.must(L.ink2_vision_finalize(h), "finalize")
    rep["tower_load_s"] = time.perf_counter() - t0

    # 3. stages against the reference
    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        rgb = read_rgb(path)
        rc, n = tw.encode(rgb)
        tw.must(rc, "encode " + name)
        rec = {"width": rgb.shape[1], "height": rgb.shape[0], "tokens": n, "stages": {}}
        for k, st in enumerate(STAGES):
            ref = np.load(os.path.join(a.golden, "%s_%s.npy" % (name, st)))
            ours = tw.stage(k, n).reshape(ref.shape)
            s = stats(ours, ref)
            s["pass"] = judge(st, s)
            ok &= s["pass"]
            rec["stages"][st] = s
        ref = np.load(os.path.join(a.golden, "%s_output.npy" % name))
        s = stats(tw.rows(n), ref)
        s["pass"] = judge("output", s)
        ok &= s["pass"]
        rec["stages"]["output_bf16"] = s
        rep["images"][name] = rec
        for st, s in rec["stages"].items():
            print("%s %-13s %s max_abs %.3g rel_l2 %.4g cos min %.6f p1 %.6f mean %.6f" % (
                name, st, "PASS" if s["pass"] else "FAIL", s["max_abs"], s["global_rel_l2"], s["min_cos"], s["p1_cos"],
                s["mean_cos"]))

    # 4. same bits twice; 5. an image past the end of the feature buffer is refused and writes nothing
    if a.images:
        rgb = read_rgb(a.images[0])
        tw.must(tw.encode(rgb)[0], "encode")
        n = -(-rgb.shape[0] // 40) * -(-rgb.shape[1] // 40)
        o1, s1 = tw.rows(n).copy(), tw.stage(4, n).copy()
        tw.must(tw.encode(rgb)[0], "encode")
        o2, s2 = tw.rows(n), tw.stage(4, n)
        check("same_bits_twice", o1.tobytes() == o2.tobytes() and s1.tobytes() == s2.tobytes(),
              "%s: output %d rows and block23 compared bytewise" % (os.path.basename(a.images[0]), n))
        last6 = tw.stage(6, n).copy()
        rc, _ = tw.encode(rgb, row0=a.cap - n + 1)
        msg = tw.msg()
        untouched = tw.rows(n).tobytes() == o1.tobytes() and tw.stage(6, n).tobytes() == last6.tobytes()
        check("past_buffer_end", rc != 0 and "outside the buffer" in msg and untouched,
              msg + "; rows and last stages unchanged: %s" % untouched)
    tw.unload()

    # 6. time per encode and memory (no stage capture); feature buffer = 8 x the cap, as the package's default
    caps = sorted({int(c) for c in a.memory_caps.split(",") if c} | {a.cap}, reverse=True)
    free0, total = C.c_longlong(), C.c_longlong()
    L.ink2_device_memory(h, C.byref(free0), C.byref(total))
    rep["memory"]["device_free_without_tower"] = free0.value
    for cap in caps:
        tw.must(tw.create(make_cfg(config, cap, 8 * cap), False), "create")
        tw.upload(entries)
        tw.must(L.ink2_vision_finalize(h), "finalize")
        w, s, f = tw.memory()
        free1 = C.c_longlong()
        L.ink2_device_memory(h, C.byref(free1), C.byref(total))
        rep["memory"][str(cap)] = {"weights": w, "scratch": s, "features": f, "max_feature_rows": 8 * cap,
                                   "device_free_drop": free0.value - free1.value}
        print("cap %d / %d rows: weights %d B, scratch %d B, features %d B; device free drops by %d B" % (
            cap, 8 * cap, w, s, f, free0.value - free1.value))
        for path in (a.images[:2] + a.time if cap == a.cap else []):
            rgb = read_rgb(path)
            n = -(-rgb.shape[0] // 40) * -(-rgb.shape[1] // 40)
            ms = tw.time(rgb, a.reps)
            rep["timing_ms"][os.path.basename(path)] = {"tokens": n, "ms": ms, "cap": cap}
            print("time %s (%d tokens): %.1f ms" % (os.path.basename(path), n, ms))
        tw.unload()
    eng.close()
    rep["pass"] = bool(ok)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
