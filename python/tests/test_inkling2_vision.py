# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 Mini image input, the CPU side (yasakani.inkling2_vision): the tensor table, the config check, image reading
(still images only, EXIF, transparency), the grid and the shrink above the cap, the limits, the image key, the host cache, the
reference pixel values and the checks of the safetensors header.  No GPU and no model file.

Optional (else that test skips):
  YASAKANI_TEST_INK2_VISION_REF  a folder with images/t1..t4.png and golden/ (tN_pixel_values.npy and report.json made by the
                                 official image processor): the pixel values must equal the golden arrays exactly.
"""
import base64
import contextlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import inkling2_vision as V

try:
    from PIL import Image
except ImportError:                                  # pragma: no cover
    Image = None

GOOD_VISION_CONFIG = {"model_type": "inkling2_vision", "text_hidden_size": 2304, "temporal_patch_size": 2, "native_patch_size": 20,
                      "adapter_merge_size": 2, "projection_hidden_dim": 4096, "projector_bias": False, "hidden_size": 1024,
                      "num_hidden_layers": 23, "num_attention_heads": 16, "mlp_ratio": 4.0, "posemb_grid_size": 32,
                      "layer_norm_eps": 1e-05}


def config(**edit):
    vc = dict(GOOD_VISION_CONFIG, **edit)
    return {"model_type": "inkling2_loop", "text_config": {"hidden_size": 2304}, "vision_config": vc}


def png(image, fmt="PNG", **kw):
    b = io.BytesIO()
    image.save(b, fmt, **kw)
    return b.getvalue()


class TableAndConfig(unittest.TestCase):
    def test_table(self):
        self.assertEqual(len(V.VISION_TENSORS), 422)
        self.assertEqual(sum(V.tensor_nbytes(n) for n in V.VISION_TENSORS), V.VISION_BYTES)
        self.assertEqual(V.VISION_TENSORS["model.visual.backbone.embeddings.patch_embedding.weight"], (1024, 1200))
        self.assertEqual(V.VISION_TENSORS["model.visual.proj_out.weight"], (2304, 4096))
        self.assertEqual(V.VISION_TENSORS["model.visual.backbone.layers.22.mlp.c_fc.weight"], (4096, 1024))

    def test_config_ok(self):
        self.assertEqual(V.check_vision_config(config())["hidden_size"], 1024)
        V.check_vision_config(config(hidden_act="gelu", num_channels=3, rope_theta=10000))

    def test_config_refused_names_key(self):
        for key, value in (("num_hidden_layers", 24), ("hidden_size", 1152), ("projector_bias", True), ("projector_bias", 0),
                           ("layer_norm_eps", 1e-6), ("model_type", "siglip"), ("hidden_act", "relu"), ("rope_theta", 5e5)):
            with self.subTest(key=key, value=value):
                with self.assertRaises(V.VisionError) as cm:
                    V.check_vision_config(config(**{key: value}))
                self.assertIn(key, str(cm.exception))
        c = config()
        del c["vision_config"]["mlp_ratio"]
        with self.assertRaisesRegex(V.VisionError, "mlp_ratio is missing"):
            V.check_vision_config(c)
        c = config()
        c["text_config"]["hidden_size"] = 4096
        with self.assertRaisesRegex(V.VisionError, "text_config.hidden_size"):
            V.check_vision_config(c)
        with self.assertRaisesRegex(V.VisionError, "no image tower"):
            V.check_vision_config({"model_type": "inkling2_loop"})

    def test_header_config_names_the_missing_key(self):
        """The header of an image-tower file: its vision_config is checked as one, so a file without model_type or hidden_size
        is refused with the key's name, not as a folder that has no tower."""
        head = {"kind": V.VISION_KIND, "vision_config": dict(config()["vision_config"]), "text_hidden_size": V.TEXT_HIDDEN}
        V.check_vision_header(head)
        for key in ("model_type", "hidden_size", "mlp_ratio"):
            with self.subTest(key=key):
                h = dict(head, vision_config={k: v for k, v in head["vision_config"].items() if k != key})
                with self.assertRaisesRegex(V.VisionError, r"\(header\): vision_config\.%s is missing" % key):
                    V.check_vision_header(h)
        with self.assertRaisesRegex(V.VisionError, r"\(header\): vision_config\.model_type is missing"):
            V.check_vision_header(dict(head, vision_config=None))

    def test_tensor_check(self):
        V.check_tensor("model.visual.proj_in.weight", "BF16", [4096, 4096], 2 * 4096 * 4096)
        with self.assertRaisesRegex(V.VisionError, "unknown"):
            V.check_tensor("model.visual.extra", "BF16", [1], 2)
        with self.assertRaisesRegex(V.VisionError, "dtype F32"):
            V.check_tensor("model.visual.proj_in.weight", "F32", [4096, 4096], 4 * 4096 * 4096)
        with self.assertRaisesRegex(V.VisionError, "shape"):
            V.check_tensor("model.visual.proj_in.weight", "BF16", [4096, 4095], 2 * 4096 * 4095)
        with self.assertRaisesRegex(V.VisionError, "bytes"):
            V.check_tensor("model.visual.proj_in.weight", "BF16", [4096, 4096], 10)


class GridAndCap(unittest.TestCase):
    def test_grid(self):
        for (w, h), want in (((640, 480), (12, 16, 192)), ((1280, 720), (18, 32, 576)), ((333, 517), (13, 9, 117)),
                             ((40, 40), (1, 1, 1)), ((1920, 1080), (27, 48, 1296)), ((4096, 4096), (103, 103, 10609)),
                             ((41, 1), (1, 2, 2))):
            self.assertEqual(V.grid(w, h), want)
        with self.assertRaises(V.VisionError):
            V.grid(0, 10)

    def test_fit(self):
        self.assertEqual(V.fit_to_cap(1280, 720), (1280, 720))
        self.assertEqual(V.fit_to_cap(1024, 1024), (1024, 1024))
        for (w, h), cap in (((4032, 3024), 1024), ((1920, 1080), 1024), ((8000, 40), 64), ((4096, 4096), 4096), ((5000, 3), 64)):
            with self.subTest(w=w, h=h, cap=cap):
                nw, nh = V.fit_to_cap(w, h, cap)
                self.assertLessEqual(V.grid(nw, nh)[2], cap)
                self.assertLess(abs(nw / nh - w / h) / (w / h), 0.02 + 1.0 / min(nw, nh))
                self.assertGreater(V.grid(nw, nh)[2], cap * 0.85)       # not shrunk more than needed
        for bad in (63, 4097, 1024.0, True, None):
            with self.assertRaises(V.VisionError):
                V.fit_to_cap(100, 100, bad)


@unittest.skipIf(Image is None, "Pillow is not installed")
class Reading(unittest.TestCase):
    def test_rgb_unchanged(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, (37, 81, 3), dtype=np.uint8)
        im = V.prepare_image(png(Image.fromarray(a)))
        self.assertTrue(np.array_equal(im.rgb, a))
        self.assertEqual((im.rows, im.cols, im.tokens, im.shrunk), (1, 3, 3, False))
        self.assertEqual(im.key, V.prepare_image(png(Image.fromarray(a))).key)
        self.assertNotEqual(im.key, V.image_key(a[:, :80]))

    def test_rgba_checkerboard(self):
        a = np.zeros((16, 16, 4), dtype=np.uint8)
        a[..., 0] = 255
        a[..., 3] = 0
        a[0, 0, 3] = 255
        a[0, 1, 3] = 128
        rgb = V.prepare_image(png(Image.fromarray(a, "RGBA"))).rgb
        self.assertEqual(rgb[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(rgb[0, 1].tolist(), [(255 * 128 + 204 * 127 + 127) // 255, (204 * 127 + 127) // 255,
                                              (204 * 127 + 127) // 255])
        self.assertEqual(rgb[0, 2].tolist(), [204, 204, 204])
        self.assertEqual(rgb[0, 8].tolist(), [153, 153, 153])
        self.assertEqual(rgb[8, 8].tolist(), [204, 204, 204])
        la = Image.fromarray(np.full((8, 8, 2), [10, 0], dtype=np.uint8), "LA")
        self.assertEqual(V.prepare_image(png(la)).rgb[0, 0].tolist(), [204, 204, 204])
        p = Image.new("P", (8, 8), 0)
        p.putpalette([1, 2, 3] + [0] * 765)
        self.assertEqual(V.prepare_image(png(p, transparency=0)).rgb[0, 0].tolist(), [204, 204, 204])
        self.assertEqual(V.prepare_image(png(p)).rgb[0, 0].tolist(), [1, 2, 3])

    def test_exif_orientation(self):
        im = Image.new("RGB", (80, 40), (200, 10, 10))
        exif = Image.Exif()
        exif[0x0112] = 6                                     # rotate 90 degrees clockwise to display
        out = V.prepare_image(png(im, "JPEG", exif=exif.tobytes()))
        self.assertEqual((out.width, out.height), (40, 80))

    def test_refusals(self):
        frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 255, 0))]
        b = io.BytesIO()
        frames[0].save(b, "GIF", save_all=True, append_images=frames[1:])
        with self.assertRaisesRegex(V.VisionError, "only still images"):
            V.prepare_image(b.getvalue())
        with self.assertRaisesRegex(V.VisionError, "not a still image format"):
            V.prepare_image(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        with self.assertRaisesRegex(V.VisionError, "empty"):
            V.prepare_image(b"")
        with self.assertRaisesRegex(V.VisionError, "aspect"):
            V.prepare_image(png(Image.new("L", (402, 2))))
        with self.assertRaisesRegex(V.VisionError, "40 million"):
            V.prepare_image(png(Image.new("1", (8000, 5001))))
        with self.assertRaisesRegex(V.VisionError, "24 MiB"):
            V.decode_image(b"\x00" * (V.MAX_IMAGE_BYTES + 1))
        with self.assertRaisesRegex(V.VisionError, "cannot be decoded"):
            V.prepare_image(png(Image.new("RGB", (64, 64)))[:60])

    def test_only_raster_formats(self):
        # EPS would run Ghostscript on the bytes and WMF is drawn by the operating system: both must be refused unread
        from PIL import EpsImagePlugin
        eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 100 100\n%%EndComments\nshowpage\n%%EOF\n"
        hdr = struct.pack("<LHhhhhHL", 0x9AC6CDD7, 0, 0, 0, 200, 200, 1440, 0)
        cs = 0
        for i in range(0, 20, 2):
            cs ^= struct.unpack("<H", hdr[i:i + 2])[0]
        wmf = hdr + struct.pack("<H", cs) + struct.pack("<HHHLHLH", 1, 9, 0x300, 12, 0, 3, 0) + struct.pack("<LH", 3, 0)
        called = []
        orig = EpsImagePlugin.Ghostscript
        EpsImagePlugin.Ghostscript = lambda *a, **k: called.append(1) or orig(*a, **k)
        try:
            for name, data in (("eps", eps), ("wmf", wmf)):
                with self.subTest(name=name), self.assertRaisesRegex(V.VisionError, r"\(PNG, JPEG, WebP, GIF, BMP or TIFF; "):
                    V.decode_image(data)
        finally:
            EpsImagePlugin.Ghostscript = orig
        self.assertEqual(called, [])
        a = np.full((8, 8, 3), 77, dtype=np.uint8)
        for fmt in ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"):
            with self.subTest(fmt=fmt):
                b = io.BytesIO()
                kw = {"quality": 100} if fmt in ("JPEG", "WEBP") else {}
                Image.fromarray(a).save(b, fmt, **kw)
                self.assertEqual(V.decode_image(b.getvalue()).shape, (8, 8, 3))

    def test_one_format_list_and_a_pillow_without_a_reader(self):
        # the list is the one of yasakani.vision, and a Pillow build without a reader (here WebP) still opens the others
        # (Image.open raises KeyError for a format name it does not know)
        from unittest import mock
        from yasakani import vision
        self.assertIs(V.STILL_FORMATS, vision.STILL_FORMATS)
        Image.init()
        without = {k: v for k, v in Image.OPEN.items() if k != "WEBP"}
        b = io.BytesIO()
        Image.new("RGB", (5, 3), (1, 2, 3)).save(b, "GIF")              # GIF comes after WEBP in the list
        with mock.patch.object(Image, "OPEN", without):
            self.assertEqual(V.decode_image(b.getvalue()).shape, (3, 5, 3))

    def test_shrink_photo(self):
        h, w = 3024, 4032
        y, x = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(2)
        a = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) % 256)], -1).astype(np.uint8)
        a = np.clip(a.astype(np.int16) + rng.integers(-8, 9, a.shape), 0, 255).astype(np.uint8)
        data = png(Image.fromarray(a), "JPEG", quality=90)
        im = V.prepare_image(data)
        self.assertTrue(im.shrunk)
        self.assertLessEqual(im.tokens, 1024)
        self.assertEqual(im.original_size, (4032, 3024))
        self.assertEqual(im.rgb.shape, (im.height, im.width, 3))
        self.assertIn("cap 1,024", im.describe(1024))
        self.assertEqual(V.prepare_image(data).key, im.key)
        small = V.prepare_image(data, max_image_tokens=64)
        self.assertLessEqual(small.tokens, 64)

    def test_no_resize_within_cap(self):
        a = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        im = V.prepare_image(png(Image.fromarray(a)))
        self.assertFalse(im.shrunk)
        self.assertEqual(im.tokens, 576)
        self.assertTrue(np.array_equal(im.rgb, a))

    def test_sources(self):
        data = png(Image.new("RGB", (4, 4)))
        uri = "data:image/png;base64," + base64.b64encode(data).decode()
        self.assertEqual(V.source_bytes(uri), data)
        for bad, msg in (("https://example.invalid/a.png", "remote"), ("data:text/plain;base64,AA==", "data:image"),
                         ("C:/x.png", "local files require"), ("data:image/png;base64,@@", "base64")):
            with self.assertRaisesRegex(V.VisionError, msg):
                V.source_bytes(bad)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            p.write_bytes(data)
            self.assertEqual(V.source_bytes(str(p), allow_local_files=d), data)
            with self.assertRaisesRegex(V.VisionError, "inside the folder"):
                V.source_bytes(str(p), allow_local_files=str(Path(d) / "sub"))


class PixelValuesAndLimits(unittest.TestCase):
    def test_pixel_values(self):
        a = np.arange(41 * 3 * 3, dtype=np.int64).reshape(3, 41, 3).astype(np.uint8)
        pv = V.pixel_values(a)
        self.assertEqual(pv.shape, (2, 40, 40, 3))
        self.assertEqual(pv.dtype, np.float32)
        self.assertEqual(float(pv[0, 0, 0, 0]), float(np.float32(-1.0)))
        self.assertEqual(float(pv[0, 39, 0, 0]), float((np.float32(-1) - np.float32(127.5)) / np.float32(127.5)))
        self.assertAlmostEqual(float(pv[1, 0, 1, 0]), -1.0078431, places=6)
        self.assertEqual(float(pv[1, 0, 0, 2]), float((np.float32(a[0, 40, 2]) - np.float32(127.5)) / np.float32(127.5)))

    def test_native_patches(self):
        pv = np.random.default_rng(4).standard_normal((3, 40, 40, 3)).astype(np.float32)
        rows = V.native_patches(pv)
        self.assertEqual(rows.shape, (12, 1200))
        self.assertTrue(np.array_equal(rows[4 * 2 + 2 * 1 + 1], pv[2, 20:, 20:].reshape(-1)))
        self.assertTrue(np.array_equal(rows[1], pv[0, :20, 20:].reshape(-1)))

    def test_request_limits(self):
        def im(n):
            return V.PreparedImage(rgb=np.zeros((1, 1, 3), np.uint8), width=1, height=1, rows=1, cols=n, tokens=n, key="k",
                                   original_size=(1, 1), shrunk=False)
        self.assertEqual(V.check_request([im(1000)] * 8), 8000)
        with self.assertRaisesRegex(V.VisionError, "per request is 8192"):
            V.check_request([im(1000)] * 9)
        with self.assertRaisesRegex(V.VisionError, "at most 16 images"):
            V.check_request([im(1)] * 17)
        with self.assertRaisesRegex(V.VisionError, "limit per image"):
            V.check_request([im(1025)])
        self.assertEqual(V.check_request([im(3000)] * 3, max_image_tokens=4096, max_request_tokens=9000), 9000)
        with self.assertRaises(V.VisionError):
            V.request_token_limit(1024, 100)

    def test_cache(self):
        c = V.FeatureCache(max_bytes=3 * 10 * 2304 * 2)
        f = np.zeros((10, 2304), np.uint16)
        for k in "abc":
            self.assertTrue(c.put(k, f))
        self.assertIsNotNone(c.get("a"))                     # a is now the most recent
        c.put("d", f)
        self.assertIsNone(c.get("b"))
        self.assertIsNotNone(c.get("a"))
        self.assertIsNone(c.get("a", tokens=11))
        self.assertFalse(c.put("big", np.zeros((40, 2304), np.uint16)))
        self.assertEqual(c.bytes, 3 * f.nbytes)
        with self.assertRaises(V.VisionError):
            c.put("x", np.zeros((10, 2304), np.float32))


class Weights(unittest.TestCase):
    def header_file(self, d, edit=None, drop=None, extra=None):
        hdr, off = {}, 0
        for n, s in sorted(V.VISION_TENSORS.items()):
            nb = 2 * int(np.prod(s))
            hdr[n] = {"dtype": "BF16", "shape": list(s), "data_offsets": [off, off + nb]}
            off += nb
        hdr["model.language_model.x"] = {"dtype": "BF16", "shape": [1], "data_offsets": [off, off + 2]}
        if edit:
            hdr[edit[0]]["shape"] = edit[1]
        if drop:
            del hdr[drop]
        if extra:
            hdr[extra] = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
        js = json.dumps(hdr).encode()
        p = Path(d) / "model.safetensors"
        p.write_bytes(struct.pack("<Q", len(js)) + js)          # header only: the data is not there
        return p

    def test_safetensors_header_checks(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(V.VisionError, "not inside the file"):
                V.safetensors_vision_range(self.header_file(d))
            with self.assertRaisesRegex(V.VisionError, r"shape \[1024, 1201\], expected \[1024, 1200\]"):
                V.safetensors_vision_range(self.header_file(d, edit=("model.visual.backbone.embeddings.patch_embedding.weight",
                                                                     [1024, 1201])))
            with self.assertRaisesRegex(V.VisionError, "1 image-tower tensors are missing"):
                V.safetensors_vision_range(self.header_file(d, drop="model.visual.proj_in.weight"))
            with self.assertRaisesRegex(V.VisionError, "unknown image-tower tensor"):
                V.safetensors_vision_range(self.header_file(d, extra="model.visual.backbone.extra"))
            p = Path(d) / "bad.safetensors"
            p.write_bytes(struct.pack("<Q", 10 ** 9))
            with self.assertRaisesRegex(V.VisionError, "header length"):
                V.safetensors_vision_range(p)

    def test_folder_without_tower(self):
        # the line every chat / serve of a folder without the tower prints: no download that does not exist yet, and the
        # README section it names says how to add the tower to a folder converted before
        self.assertNotIn("download", V.NO_TOWER)
        readme = Path(__file__).resolve().parents[1] / "README.md"
        if readme.is_file():
            text = readme.read_text(encoding="utf-8")
            self.assertIn("**Image input**", text)
            self.assertIn("tools/convert/inkling2_vision_convert.py --src", text)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "yasakani.json").write_text(json.dumps({"architecture": "inkling2", "weights": {
                "file": "w.yskn", "format": "yskn", "version": 2}}), encoding="utf-8")
            self.assertIsNone(V.vision_weights(d))
            with self.assertRaisesRegex(V.VisionError, "no image tower"):
                list(V.iter_vision_tensors(d))
            (Path(d) / "yasakani.json").write_text(json.dumps({"architecture": "inkling2", "weights": {"file": "w.yskn",
                "format": "yskn"}, "vision": {"file": "../v.yskn", "format": "yskn", "version": 2}}), encoding="utf-8")
            with self.assertRaisesRegex(V.VisionError, "not a file inside"):
                V.vision_weights(d)


CONVERT = Path(__file__).resolve().parents[2] / "tools" / "convert" / "inkling2_vision_convert.py"


@unittest.skipUnless(CONVERT.is_file(), "tools/convert/inkling2_vision_convert.py not found")
class ConverterFailures(unittest.TestCase):
    """A source that changes or ends while it is read leaves no half-written file in the model folder."""

    def test_no_tmp_left_behind(self):
        import importlib.util
        from types import SimpleNamespace
        from unittest import mock
        spec = importlib.util.spec_from_file_location("ink2_vision_convert_test", CONVERT)
        C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C)

        class Writer:                                        # stands in for the container writer: plain bytes
            def __init__(self, path, **kw):
                self.f = open(path, "wb")

            def put(self, name, role, a):
                self.f.write(a.tobytes())

            def close(self):
                self.f.close()

        fake_v = SimpleNamespace(VISION_FILE="v.yskn", VISION_TENSORS={"a": (2,), "b": (2,)}, VISION_KIND="k", TEXT_HIDDEN=8)
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "w.safetensors", Path(d) / "out"
            dst.mkdir()
            src.write_bytes(bytes(range(8)))
            entries = {"a": (0, 4), "b": (4, 8)}
            p = dict(weights=src.name, vision_config={}, revision="r",
                     range=SimpleNamespace(path=src, begin=0, end=8, entries=entries))
            with mock.patch.object(C, "modules", return_value=(fake_v, SimpleNamespace(Writer=Writer))), \
                    contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(C, "range_sha256", return_value="0" * 64):     # the source changed while read
                    with self.assertRaisesRegex(SystemExit, "changed while it was read"):
                        C.write(src, dst, p)
                self.assertEqual(sorted(x.name for x in dst.iterdir()), [])
                p["range"] = SimpleNamespace(path=src, begin=0, end=12, entries={"a": (0, 4), "b": (8, 12)})
                with mock.patch.object(C, "range_sha256", return_value="0" * 64):     # ends inside a tensor
                    with self.assertRaisesRegex(SystemExit, "truncated"):
                        C.write(src, dst, p)
                self.assertEqual(sorted(x.name for x in dst.iterdir()), [])
                p["range"] = SimpleNamespace(path=src, begin=0, end=8, entries=entries)
                r = C.write(src, dst, p)                                              # and a good run writes the file
                self.assertEqual((sorted(x.name for x in dst.iterdir()), r["size"]), (["v.yskn"], 8))


@unittest.skipIf(Image is None, "Pillow is not installed")
class Golden(unittest.TestCase):
    """The pixel values of the official image processor for four test images (exact equality)."""

    def test_golden(self):
        ref = os.environ.get("YASAKANI_TEST_INK2_VISION_REF")
        if not ref or not (Path(ref) / "golden" / "report.json").is_file():
            self.skipTest("YASAKANI_TEST_INK2_VISION_REF is not set")
        report = json.loads((Path(ref) / "golden" / "report.json").read_text(encoding="utf-8"))["images"]
        for t in ("t1", "t2", "t3", "t4"):
            with self.subTest(image=t):
                im = V.prepare_image(Path(ref) / "images" / ("%s.png" % t))
                g = np.load(Path(ref) / "golden" / ("%s_pixel_values.npy" % t))
                pv = V.pixel_values(im.rgb)
                self.assertEqual(pv.shape, g.shape)
                self.assertEqual(float(np.abs(pv - g).max()), 0.0)
                r = report[t]
                self.assertEqual((im.rows, im.cols, im.tokens), (r["grid_rows"], r["grid_cols"], r["image_tokens"]))


if __name__ == "__main__":
    unittest.main()
