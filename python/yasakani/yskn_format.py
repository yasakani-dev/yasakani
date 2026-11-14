# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Bounded, CPU-only reader for the released YSKN v1 weight representation.

The decoding equations follow ``tools/yskn_decode.py`` and ``docs/FORMAT.md``.
This module is packaged with the runtime so the macOS backend needs no checkout.
It deliberately rejects representations not defined for the released models.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np


class FormatError(ValueError):
    pass


class Container:
    def __init__(self, path):
        self.path = os.fspath(path)
        self.size = os.path.getsize(path)
        if self.size < 32:
            raise FormatError("truncated YSKN header")
        with open(path, "rb") as stream:
            magic, version, length, data_offset, file_length = struct.unpack("<4sIQQQ", stream.read(32))
            if magic != b"YSKN" or version != 1:
                raise FormatError("expected a YSKN version 1 container")
            if length > min(self.size - 32, 64 * 1024 * 1024):
                raise FormatError("invalid YSKN JSON header length")
            if data_offset < 32 + length or data_offset > self.size or file_length != self.size:
                raise FormatError("invalid YSKN file length or data offset")
            self.header = json.loads(stream.read(length).decode("utf-8"))
        self.model = self.header["model"]
        self.tensors = self._index(self.header["tensors"], "tensor")
        self.vectors = self._index(self.header["vectors"], "vector")
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    @staticmethod
    def _index(items, kind):
        result = {}
        for item in items:
            name = item["name"]
            if not isinstance(name, str) or name in result:
                raise FormatError("invalid or duplicate %s name" % kind)
            result[name] = item
        return result

    def array(self, offset, dtype, count):
        dtype = np.dtype(dtype)
        if not isinstance(offset, int) or not isinstance(count, int) or offset < 0 or count < 0:
            raise FormatError("invalid data offset or count")
        if offset + count * dtype.itemsize > self.size:
            raise FormatError("YSKN data extends past the end of the file")
        return np.ndarray((count,), dtype=dtype, buffer=self.mm, offset=offset)


def checked(t):
    d = dict(t)
    for key, default in (("physical_rows", d["rows"]), ("l", 12), ("d", 2), ("t", 256),
                         ("start", "zero"), ("layout", "warp")):
        d.setdefault(key, default)
    if (d["rows"] < 1 or d["cols"] < 1 or d["physical_rows"] < d["rows"] or d["cols"] % 512
            or d["l"] != 12 or d["d"] != 2 or d["t"] != 256 or d["bits"] not in range(4, 9)
            or d["start"] != "zero" or d.get("rar") or d["layout"] not in ("row", "warp")):
        raise FormatError("%s: unsupported tensor representation" % d["name"])
    default = dict(kind="H512", sign_seed=7, qm_off=0, rule="mulberry32_7919_13")
    d.setdefault("rot_in", dict(default, m=d["cols"] // 512, sign_offset=0))
    d.setdefault("rot_out", None if d["role"] in ("head", "embedding") else
                 dict(default, m=d["physical_rows"] // 512, sign_offset=d["cols"]))
    if d["rot_out"] is not None and d["rot_out"].get("kind") == "none":
        d["rot_out"] = None
    if d["rot_in"] is None:
        raise FormatError("%s: missing input rotation" % d["name"])
    for rot, n in ((d["rot_in"], d["cols"]), (d["rot_out"], d["physical_rows"])):
        if rot is not None and (rot.get("kind") != "H512" or n % 512 or rot["m"] != n // 512
                                or rot.get("rule", "splitmix64") not in ("splitmix64", "mulberry32_7919_13")):
            raise FormatError("%s: unsupported rotation" % d["name"])
    sequences = d["physical_rows"] * (d["cols"] // 256)
    if d["stream_bytes"] != sequences * d["bits"] * 16 or (d["layout"] == "warp" and sequences % 32):
        raise FormatError("%s: invalid code stream size/layout" % d["name"])
    return d


def sign_stream(rule, seed, offset, n):
    k = np.arange(offset, offset + n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        if rule == "splitmix64":
            z = np.uint64(seed) + (k + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z ^= z >> np.uint64(31)
            return np.where(z >> np.uint64(63), 1, -1).astype(np.float32)
        if rule == "mulberry32_7919_13":
            s0 = ((int(seed) & 0xFFFFFFFF) * 7919 + 13) & 0xFFFFFFFF
            s0 = s0 or 0x9E3779B9
            t = np.uint32(s0) + (k + np.uint64(1)).astype(np.uint32) * np.uint32(0x6D2B79F5)
            t = (t ^ (t >> np.uint32(15))) * (t | np.uint32(1))
            t ^= t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61))
            t ^= t >> np.uint32(14)
            return np.where(t < np.uint32(0x80000000), -1, 1).astype(np.float32)
    raise FormatError("unknown sign rule")


def signs_of(rot, n):
    return sign_stream(rot.get("rule", "splitmix64"), rot["sign_seed"], rot["sign_offset"], n)


def fwht512(x):
    """Normalised Sylvester transform without constructing a dense 512-square matrix."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % 512:
        raise FormatError("Hadamard input length must be a multiple of 512")
    out = x.copy().reshape(-1, 512)
    for width in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        blocks = out.reshape(-1, 2, width)
        left, right = blocks[:, 0].copy(), blocks[:, 1].copy()
        blocks[:, 0], blocks[:, 1] = left + right, left - right
    out *= np.float32(1 / np.sqrt(512))
    return out.reshape(x.shape)


def value_table(c, t):
    return c.array(t["lut_off"], "<f2", 8192).astype(np.float32).reshape(4096, 2)


def rotated_rows(c, t, r0, r1):
    """Decode only the requested rows, keeping temporary storage bounded."""
    rows, seqs, bits = t["physical_rows"], t["cols"] // 256, t["bits"]
    if not 0 <= r0 <= r1 <= rows:
        raise FormatError("invalid row range")
    words_per_seq = 4 * bits
    raw = c.array(t["stream_off"], "<u4", rows * seqs * words_per_seq)
    if t["layout"] == "row":
        units = raw.reshape(rows, seqs, words_per_seq)[r0:r1]
    else:
        q = np.arange(seqs)[None, :] * rows + np.arange(r0, r1)[:, None]
        w = np.arange(words_per_seq)
        index = (((q[..., None] // 32) * bits + w // 4) * 32 + q[..., None] % 32) * 4 + w % 4
        units = raw[index]
    # The extra zero word allows the final state to straddle a 32-bit boundary.
    words = np.pad(units.astype(np.uint64), ((0, 0), (0, 0), (0, 1)))
    state = np.empty((r1 - r0, seqs, 128), dtype=np.uint16)
    for j in range(128):
        end = bits * (j + 1)
        start = end - 12
        if start < 0:
            state[..., j] = (words[..., 0] & np.uint64((1 << end) - 1)) << np.uint64(-start)
        else:
            wi, shift = divmod(start, 32)
            state[..., j] = ((words[..., wi] | (words[..., wi + 1] << np.uint64(32)))
                             >> np.uint64(shift)) & np.uint64(4095)
    scales = c.array(t["scale_off"], "<f2", rows)[r0:r1].astype(np.float32)
    return value_table(c, t)[state].reshape(r1 - r0, t["cols"]) * scales[:, None]


def vector(c, v):
    raw = c.array(v["off"], "<u2", v["n"])
    if v.get("dtype", "f16") == "bf16":
        return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if v.get("dtype", "f16") == "f16":
        return raw.view("<f2").astype(np.float32)
    raise FormatError("%s: unsupported vector dtype" % v["name"])


class NumpyOps:
    """Slow reference operations on the exact stored weights, with bounded row decoding.

    This does not simulate CUDA's 8-bit activation/value-table rounding.
    """
    def __init__(self, container, block_rows=128):
        self.container = container
        self.block_rows = int(block_rows)
        if self.block_rows < 1:
            raise ValueError("block_rows must be positive")
        self.tensors = {name: checked(t) for name, t in container.tensors.items()}
        self._signs = {}
        self._vectors = {}
        for t in self.tensors.values():
            container.array(t["stream_off"], "u1", t["stream_bytes"])
            container.array(t["lut_off"], "<f2", 8192)
            container.array(t["scale_off"], "<f2", t["physical_rows"])

    def _sign(self, name, side):
        key = (name, side)
        if key not in self._signs:
            t = self.tensors[name]
            self._signs[key] = signs_of(t[side], t["cols"] if side == "rot_in" else t["physical_rows"])
        return self._signs[key]

    def vector(self, name):
        if name not in self._vectors:
            self._vectors[name] = vector(self.container, self.container.vectors[name])
        return self._vectors[name]

    def linear(self, name, x):
        t = self.tensors[name]
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (t["cols"],):
            raise ValueError("%s: input shape mismatch" % name)
        rotated = fwht512(x * self._sign(name, "rot_in"))
        out = np.empty(t["physical_rows"], dtype=np.float32)
        for first in range(0, len(out), self.block_rows):
            last = min(len(out), first + self.block_rows)
            out[first:last] = rotated_rows(self.container, t, first, last) @ rotated
        if t["rot_out"] is not None:
            out = fwht512(out) * self._sign(name, "rot_out")
        return out[:t["rows"]]

    def embedding(self, token):
        c = self.container
        if not isinstance(token, (int, np.integer)) or not 0 <= token < c.model["vocab"]:
            raise ValueError("embedding token is outside the model vocabulary")
        if "embed_tokens" not in self.tensors:
            width = c.model["hidden"]
            v = c.vectors["embed_tokens"]
            part = dict(v, off=v["off"] + token * width * 2, n=width)
            return vector(c, part)
        out = np.zeros(c.model["hidden"], dtype=np.float32)
        for name in ("embed_tokens", "embed_tokens.res", "embed_tokens.res2"):
            if name in self.tensors:
                t = self.tensors[name]
                if t["rot_out"] is not None:
                    raise FormatError("embedding output rotations are unsupported")
                out += fwht512(rotated_rows(c, t, token, token + 1)[0]) * self._sign(name, "rot_in")
        return out
