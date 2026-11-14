#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Compare real Metal kernels with the independent public-format reader.

Run on Apple Silicon: python src/metal/check_metal.py build/bin/libyskn_metal.dylib
A missing/inaccessible Metal device is a failure, never a silently passing test.
"""
import ctypes as C
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import yskn_decode as reference


def main():
    lib = C.CDLL(str(Path(sys.argv[1]).resolve()))
    lib.yskn_metal_open.argtypes, lib.yskn_metal_open.restype = [C.c_char_p], C.c_void_p
    lib.yskn_metal_close.argtypes = [C.c_void_p]
    lib.yskn_metal_last_error.restype = C.c_char_p
    lib.yskn_metal_linear.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    lib.yskn_metal_row.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_int]
    lib.yskn_metal_embedding.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    fixture = ROOT / "tools/testdata/tiny.yskn"
    container = reference.Container(str(fixture))
    handle = lib.yskn_metal_open(str(fixture).encode())
    if not handle:
        raise RuntimeError(lib.yskn_metal_last_error().decode())
    try:
        rng = np.random.default_rng(260114)
        count = 0
        for name, raw in container.tensors.items():
            t = reference.checked(raw)
            matrix = reference.decode(container, t)
            for x in (rng.standard_normal(t["cols"]).astype(np.float32), np.ones(t["cols"], np.float32)):
                actual = np.zeros(t["rows"], np.float32)
                rc = lib.yskn_metal_linear(handle, name.encode(), x.ctypes.data, x.size, actual.ctypes.data, actual.size)
                if rc:
                    raise RuntimeError(lib.yskn_metal_last_error().decode())
                expected = matrix @ x
                np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=2e-4)
                print("PASS linear", name, "max error", float(np.max(np.abs(actual - expected))))
                count += 1
            if t["rot_out"] is None:
                for row in (0, t["rows"] - 1):
                    actual = np.empty(t["cols"], np.float32)
                    rc = lib.yskn_metal_row(handle, name.encode(), row, actual.ctypes.data, actual.size)
                    if rc:
                        raise RuntimeError(lib.yskn_metal_last_error().decode())
                    np.testing.assert_allclose(actual, matrix[row], atol=3e-6, rtol=2e-5)
                    count += 1
                    print("PASS row", name, row)
        x = np.zeros(512, np.float32)
        assert lib.yskn_metal_linear(handle, b"missing", x.ctypes.data, x.size, x.ctypes.data, x.size) == -1
        assert b"tensor not found" in lib.yskn_metal_last_error()
        first = next(iter(container.tensors.values()))
        assert lib.yskn_metal_linear(handle, first["name"].encode(), x.ctypes.data, 1, x.ctypes.data, 1) == -1
        count += 2
    finally:
        lib.yskn_metal_close(handle)
    # Native boundary: invalid files fail before touching GPU memory.
    with tempfile.TemporaryDirectory() as folder:
        invalid = Path(folder) / "bad.yskn"
        original = fixture.read_bytes()
        for data in (b"bad", original[:100], b"NOPE" + original[4:]):
            invalid.write_bytes(data)
            assert not lib.yskn_metal_open(str(invalid).encode())
            assert lib.yskn_metal_last_error()
            count += 1
        damaged = bytearray(original)
        header = json.loads(damaged[32:32 + struct.unpack_from("<Q", damaged, 8)[0]])
        header["tensors"][0]["stream_off"] = len(original) + 1
        encoded = json.dumps(header, separators=(",", ":")).encode()
        assert len(encoded) < 4096 - 32
        struct.pack_into("<Q", damaged, 8, len(encoded))
        damaged[32:32 + len(encoded)] = encoded
        invalid.write_bytes(damaged)
        assert not lib.yskn_metal_open(str(invalid).encode())
        assert b"past end" in lib.yskn_metal_last_error()
        count += 1
    # Embedding residuals and vector-only embeddings use the same native public ABI.
    with tempfile.TemporaryDirectory() as folder:
        fixture_out = Path(folder) / "embedding.yskn"
        original = fixture.read_bytes()
        original_header = json.loads(original[32:32 + struct.unpack_from("<Q", original, 8)[0]])
        def write_header(header):
            damaged = bytearray(original)
            encoded = json.dumps(header, separators=(",", ":")).encode()
            assert len(encoded) < 4096 - 32
            struct.pack_into("<Q", damaged, 8, len(encoded))
            damaged[32:32 + len(encoded)] = encoded
            fixture_out.write_bytes(damaged)
        emb = dict(container.tensors["tiny.embed"])
        head = dict(original_header)
        head["tensors"] = [dict(emb, name=name) for name in ("embed_tokens", "embed_tokens.res", "embed_tokens.res2")]
        write_header(head)
        h = lib.yskn_metal_open(str(fixture_out).encode())
        assert h, lib.yskn_metal_last_error().decode()
        try:
            actual = np.empty(emb["cols"], np.float32)
            assert lib.yskn_metal_embedding(h, 7, actual.ctypes.data, actual.size) == 0
            expected = reference.decode_rows(container, reference.checked(emb), 7, 8)[0] * 3
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-5)
            assert lib.yskn_metal_embedding(h, -1, actual.ctypes.data, actual.size) == -1
            count += 2
        finally:
            lib.yskn_metal_close(h)
        for vector_name in ("tiny.norm", "tiny.half"):
            v = container.vectors[vector_name]
            head = dict(original_header, tensors=[], vectors=[dict(v, name="embed_tokens")],
                        model=dict(original_header["model"], hidden=v["n"], vocab=1))
            write_header(head)
            h = lib.yskn_metal_open(str(fixture_out).encode())
            assert h, lib.yskn_metal_last_error().decode()
            try:
                actual = np.empty(v["n"], np.float32)
                assert lib.yskn_metal_embedding(h, 0, actual.ctypes.data, actual.size) == 0
                np.testing.assert_array_equal(actual, reference.vector(container, v))
                assert lib.yskn_metal_embedding(h, 1, actual.ctypes.data, actual.size) == -1
                count += 2
            finally:
                lib.yskn_metal_close(h)
    # Less common bit widths, padding, default signs, and unaligned sections.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "synthetic.yskn"
        rng = np.random.default_rng(42)
        for bits in (4, 5, 7):
            for layout in ("row", "warp"):
                stream = rng.integers(0, 2**32, 32 * 2 * 4 * bits, dtype=np.uint32).tobytes()
                lut = rng.normal(size=8192).astype("<f2").tobytes()
                scales = rng.uniform(0.01, 0.1, size=32).astype("<f2").tobytes()
                data = bytearray(4097)
                stream_off = len(data); data.extend(stream); data.extend(b"abc")
                lut_off = len(data); data.extend(lut); data.extend(b"x")
                scale_off = len(data); data.extend(scales)
                tensor = dict(name="test", role="head", rows=17, physical_rows=32, cols=512,
                              bits=bits, layout=layout, stream_off=stream_off, stream_bytes=len(stream),
                              lut_off=lut_off, scale_off=scale_off)
                header = dict(model=original_header["model"], tensors=[tensor], vectors=[])
                encoded = json.dumps(header, separators=(",", ":")).encode()
                struct.pack_into("<4sIQQQ", data, 0, b"YSKN", 1, len(encoded), 4097, len(data))
                data[32:32 + len(encoded)] = encoded
                path.write_bytes(data)
                c = reference.Container(str(path))
                expected_matrix = reference.decode(c, reference.checked(tensor))
                h = lib.yskn_metal_open(str(path).encode())
                assert h, lib.yskn_metal_last_error().decode()
                try:
                    x = rng.normal(size=512).astype(np.float32)
                    out = np.empty(17, np.float32)
                    assert lib.yskn_metal_linear(h, b"test", x.ctypes.data, x.size, out.ctypes.data, out.size) == 0
                    np.testing.assert_allclose(out, expected_matrix @ x, atol=3e-5, rtol=2e-4)
                    count += 1
                finally:
                    lib.yskn_metal_close(h)
    print("PASS:", count, "real Metal/reference and error-boundary checks")


if __name__ == "__main__":
    main()
