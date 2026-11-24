# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Still-image input of Inkling-2 Mini: the CPU side.

The image tower of Inkling-2 Mini (a 23-block vision transformer and a 2 x 2 merging projector, 422 bfloat16 tensors named
``model.visual.*``) runs on the GPU inside the engine library.  This module does everything before that and around it:

* the constants of the tower and the table of its 422 tensors with their shapes (``VISION_TENSORS``), and the check of the
  ``vision_config`` of config.json against them (``check_vision_config``);
* reading an image (``prepare_image``): decode with Pillow, refuse anything that is not one still image, apply the EXIF
  orientation, composite transparent images on the model's checkerboard, and shrink an image only when its token count is above
  the per-image cap;
* the grid of an image (``grid``: one LM token per 40 x 40 pixel tile, no resize) and the limits of one request
  (``check_request``);
* the image key (SHA-256 of the RGB bytes the encoder sees and their size) and a small LRU cache of image vectors in host
  memory (``FeatureCache``);
* the reader of the tower's weights for both kinds of model folder (``vision_weights``, ``iter_vision_tensors``): the separate
  file ``inkling_2_mini_vision.yskn`` of a converted folder, or the ``model.visual.*`` range of the published safetensors file;
* ``pixel_values``: the normalised, padded tiles of an image in NumPy.  The engine computes the same values on the GPU; this
  function exists so that the preprocessing can be tested without it.

The preprocessing follows the official image processor of the model: no resize, padding at the bottom and the right to multiples
of 40 pixels with the value -1 on the 0..255 scale, then ``v = (p - 127.5) / 127.5`` per channel (R, G, B).  The only step that
is not part of the official path is the shrink above the per-image cap (Pillow LANCZOS, aspect ratio kept): the official code
has no cap at all.

numpy and (for images) Pillow.  Nothing here touches the network or writes files.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import struct
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlsplit

import numpy as np

# The Pillow readers an image may use (STILL_FORMATS: no EPS, WMF, PDF, ...), and still_formats(Image): those this Pillow has.
# One list for both image paths; .vision imports only the standard library and numpy, and nothing there imports this module.
from .vision import STILL_FORMATS, still_formats

# ---- the tower ------------------------------------------------------------------------------------------------------------------------
VISION_FILE = "inkling_2_mini_vision.yskn"     # the separate image-tower file of a converted model folder
VISION_KIND = "inkling2_vision"                # the "kind" key of that file's header
MANIFEST = "yasakani.json"
PREFIX = "model.visual."
IMAGE_TOKEN_ID = 200054                        # the placeholder id of one image token in the prompt
TILE = 40                                      # pixels per side of the tile behind one LM image token (2 x 2 native patches)
NATIVE_PATCH = 20
PAD_VALUE = -1                                 # pad value on the 0..255 scale (official processor)
HIDDEN = 1024
LAYERS = 23
HEADS = 16
HEAD_DIM = HIDDEN // HEADS
INTER = 4096
PROJ_HIDDEN = 4096
TEXT_HIDDEN = 2304                             # values per image token handed to the language model
POS_GRID = 32

# vision_config of config.json: every value the engine is built for.  A different value is refused, naming the key.
VISION_CONFIG = (
    ("model_type", "inkling2_vision"),
    ("hidden_size", HIDDEN),
    ("num_hidden_layers", LAYERS),
    ("num_attention_heads", HEADS),
    ("mlp_ratio", 4.0),
    ("native_patch_size", NATIVE_PATCH),
    ("adapter_merge_size", 2),
    ("temporal_patch_size", 2),
    ("posemb_grid_size", POS_GRID),
    ("projection_hidden_dim", PROJ_HIDDEN),
    ("projector_bias", False),
    ("layer_norm_eps", 1e-5),
    ("text_hidden_size", TEXT_HIDDEN),
)
# keys that the official configuration class fills with a default when config.json does not have them
VISION_CONFIG_OPTIONAL = (("hidden_act", "gelu"), ("num_channels", 3), ("rope_theta", 10000.0))


def _vision_tensor_table() -> Dict[str, Tuple[int, ...]]:
    t: Dict[str, Tuple[int, ...]] = {}
    b = PREFIX + "backbone."
    t[b + "embeddings.class_embedding"] = (HIDDEN,)
    t[b + "embeddings.cls_position_embedding"] = (HIDDEN,)
    t[b + "embeddings.patch_embedding.weight"] = (HIDDEN, 3 * NATIVE_PATCH * NATIVE_PATCH)
    t[b + "embeddings.position_embedding"] = (POS_GRID * POS_GRID, HIDDEN)
    for i in range(LAYERS):
        p = b + "layers.%d." % i
        for n in ("ln_1.bias", "ln_1.weight", "ln_2.bias", "ln_2.weight", "ls_1.gamma", "ls_2.gamma", "mlp.c_proj.bias"):
            t[p + n] = (HIDDEN,)
        t[p + "mlp.c_fc.bias"] = (INTER,)
        t[p + "mlp.c_fc.weight"] = (INTER, HIDDEN)
        t[p + "mlp.c_proj.weight"] = (HIDDEN, INTER)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            t[p + "self_attn.%s.bias" % proj] = (HIDDEN,)
            t[p + "self_attn.%s.weight" % proj] = (HIDDEN, HIDDEN)
    t[b + "ln_pre.bias"] = (HIDDEN,)
    t[b + "ln_pre.weight"] = (HIDDEN,)
    t[PREFIX + "proj_in.weight"] = (PROJ_HIDDEN, 4 * HIDDEN)
    t[PREFIX + "proj_out.weight"] = (TEXT_HIDDEN, PROJ_HIDDEN)
    return t


VISION_TENSORS: Dict[str, Tuple[int, ...]] = _vision_tensor_table()   # name -> shape; every tensor is bfloat16
VISION_TENSOR_COUNT = 422
VISION_BYTES = 636_512_256                                           # sum of the 422 tensors (bfloat16)
assert len(VISION_TENSORS) == VISION_TENSOR_COUNT


def tensor_nbytes(name: str) -> int:
    return 2 * math.prod(VISION_TENSORS[name])


# ---- limits ---------------------------------------------------------------------------------------------------------------------------
DEFAULT_IMAGE_TOKENS = 1024          # per image (--vision-max-image-tokens); an image above it is shrunk, never one within it
IMAGE_TOKENS_RANGE = (64, 4096)      # values the per-image cap accepts (the engine reserves its scratch for the cap at load)
REQUEST_IMAGE_TOKENS_FACTOR = 8      # per request: 8 x the per-image cap by default (8,192)
MAX_IMAGES = 16                      # per request
MAX_IMAGE_BYTES = 24 * 1024 * 1024   # per image file
MAX_PIXELS = 40_000_000              # per image, before any shrink
MAX_ASPECT = 200                     # longer side / shorter side
MEMORY_CACHE_BYTES = 512 * 1024 * 1024
LOCAL_FILES_HINT = "--vision-allow-local-files"
STILL_ONLY = "only still images are supported (animated and multi-frame files, and video, are not)"
NO_TOWER = "this folder has no image tower (python/README.md, Image input, says how to add one)"


class VisionError(ValueError):
    """A refused input: a bad image, a limit, or a model folder or file that does not fit the image tower."""


def _int_value(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _same(value, want) -> bool:
    if isinstance(want, bool):
        return value is want
    if isinstance(want, str):
        return value == want
    if isinstance(want, int):
        return _int_value(value) and value == want
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == want


def check_vision_config(config: dict, where: str = "config.json", alone: bool = False) -> dict:
    """``config``: the whole config.json (``vision_config`` and ``text_config``) or a ``vision_config`` alone.  Returns the
    checked vision_config; raises VisionError naming the first key that differs and the value this engine needs.  ``alone``:
    the object is a vision_config whatever keys it holds (the header of an image-tower file), so a missing key is named."""
    if not isinstance(config, dict):
        raise VisionError("%s is not a JSON object" % where)
    if alone:
        vc, what = config, "vision_config."
    elif "vision_config" in config:
        vc = config["vision_config"]
        what = "vision_config."
        if not isinstance(vc, dict):
            raise VisionError("%s: vision_config is not an object" % where)
        text = config.get("text_config")
        if isinstance(text, dict) and "hidden_size" in text and not _same(text["hidden_size"], TEXT_HIDDEN):
            raise VisionError("%s: text_config.hidden_size is %r; the image tower gives %d values per token"
                              % (where, text["hidden_size"], TEXT_HIDDEN))
    elif "model_type" in config and "hidden_size" in config:
        vc, what = config, "vision_config."
    else:
        raise VisionError("%s has no vision_config: %s" % (where, NO_TOWER))
    for key, want in VISION_CONFIG:
        if key not in vc:
            raise VisionError("%s: %s%s is missing (expected %r)" % (where, what, key, want))
        if not _same(vc[key], want):
            raise VisionError("%s: %s%s is %r; this engine supports only %r (Inkling-2 Mini)" % (where, what, key, vc[key], want))
    for key, want in VISION_CONFIG_OPTIONAL:
        if key in vc and not _same(vc[key], want):
            raise VisionError("%s: %s%s is %r; this engine supports only %r" % (where, what, key, vc[key], want))
    rope = vc.get("rope_parameters") or vc.get("rope_scaling")
    if isinstance(rope, dict):
        if "rope_theta" in rope and not _same(rope["rope_theta"], 10000.0):
            raise VisionError("%s: %srope_parameters.rope_theta is %r; this engine supports only 10000" % (where, what,
                                                                                                         rope["rope_theta"]))
        if rope.get("rope_type") not in (None, "default", "axial"):
            raise VisionError("%s: %srope_parameters.rope_type is %r; this engine supports only the axial 2-D rotary"
                              % (where, what, rope.get("rope_type")))
    return vc


def check_tensor(name: str, dtype: str, shape: Sequence[int], nbytes: int) -> None:
    """One tensor of the image tower: name in the table, dtype bfloat16, shape and byte count equal.  VisionError otherwise."""
    want = VISION_TENSORS.get(name)
    if want is None:
        raise VisionError("unknown image-tower tensor %s (not one of the %d tensors of Inkling-2 Mini)" % (name, VISION_TENSOR_COUNT))
    if str(dtype).upper() not in ("BF16", "BFLOAT16"):
        raise VisionError("%s: dtype %s, expected BF16" % (name, dtype))
    if tuple(int(s) for s in shape) != want:
        raise VisionError("%s: shape %s, expected %s" % (name, list(shape), list(want)))
    if int(nbytes) != 2 * math.prod(want):
        raise VisionError("%s: %d bytes, expected %d" % (name, nbytes, 2 * math.prod(want)))


# ---- grid, cap, shrink ----------------------------------------------------------------------------------------------------------------
def grid(width: int, height: int) -> Tuple[int, int, int]:
    """(rows, cols, tokens) of an image of width x height pixels: rows = ceil(H / 40), cols = ceil(W / 40)."""
    if not (_int_value(width) and _int_value(height)) or width < 1 or height < 1:
        raise VisionError("invalid image size %r x %r" % (width, height))
    rows, cols = -(-height // TILE), -(-width // TILE)
    return rows, cols, rows * cols


def check_image_cap(max_image_tokens) -> int:
    lo, hi = IMAGE_TOKENS_RANGE
    if not _int_value(max_image_tokens) or not lo <= max_image_tokens <= hi:
        raise VisionError("the image token limit per image must be an integer from %d to %d, not %r" % (lo, hi, max_image_tokens))
    return max_image_tokens


def fit_to_cap(width: int, height: int, max_image_tokens: int = DEFAULT_IMAGE_TOKENS) -> Tuple[int, int]:
    """The size an image is given to the encoder.  An image whose token count is within the cap keeps its size.  Above it: the
    largest size with the same aspect ratio (each side rounded to whole pixels) whose token count is at most the cap."""
    cap = check_image_cap(max_image_tokens)
    if grid(width, height)[2] <= cap:
        return width, height

    def size(s: float) -> Tuple[int, int]:
        return max(1, min(width, int(round(width * s)))), max(1, min(height, int(round(height * s))))

    lo, hi = 0.0, math.sqrt(cap * TILE * TILE / (width * height)) * 1.5     # tokens(size(lo)) <= cap
    hi = min(hi, 1.0)
    for _ in range(60):                                                     # bisection on the scale factor
        mid = (lo + hi) / 2
        w, h = size(mid)
        if grid(w, h)[2] <= cap:
            lo = mid
        else:
            hi = mid
    w, h = size(lo)
    if grid(w, h)[2] > cap:                                                 # not reached for caps >= 64; kept as a guard
        raise VisionError("cannot shrink a %d x %d image to %d tokens" % (width, height, cap))
    return w, h


# ---- reading an image ----------------------------------------------------------------------------------------------------------------
@dataclass
class PreparedImage:
    rgb: np.ndarray            # uint8 [height, width, 3], C-contiguous: exactly what the encoder reads
    width: int
    height: int
    rows: int
    cols: int
    tokens: int
    key: str                   # image_key(rgb)
    original_size: Tuple[int, int]    # (width, height) after the EXIF orientation, before any shrink
    shrunk: bool

    def describe(self, cap: int) -> str:
        """The note printed when an image was shrunk."""
        return "image %dx%d shrunk to %dx%d (%d tokens; cap %s)" % (self.original_size + (self.width, self.height, self.tokens,
                                                                                         format(cap, ",")))


def image_key(rgb: np.ndarray) -> str:
    """SHA-256 of the image the encoder sees: its width and height (two little-endian uint32) and then its RGB bytes."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise VisionError("an image must be uint8 [height, width, 3]")
    h = hashlib.sha256(struct.pack("<II", a.shape[1], a.shape[0]))
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def source_bytes(source: str, *, allow_local_files: Union[bool, str, os.PathLike] = False) -> bytes:
    """The file bytes of an image named in a request: a base64 ``data:image/...`` URI, or (only with the server-side opt-in:
    True, or the folder the files must be in) a local path or file:// URL.  Remote URLs are never fetched."""
    if not isinstance(source, str) or not source:
        raise VisionError("an image source must be a nonempty string")
    if source.startswith("data:"):
        header, sep, data = source.partition(",")
        if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise VisionError("images must use a base64 data:image URI")
        if len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise VisionError("image exceeds 24 MiB")
        try:
            return base64.b64decode(data, validate=True)
        except ValueError:
            raise VisionError("the data:image URI is not valid base64") from None
    url = urlsplit(source)
    drive = len(url.scheme) == 1 and source[1:2] == ":"                       # C:\... is a local path, not a URL
    if url.scheme and url.scheme != "file" and not drive:
        raise VisionError("remote image URLs are not supported; send the image as a data:image/...;base64 URI")
    if not allow_local_files:
        raise VisionError("only data:image URIs are enabled; local files require %s" % LOCAL_FILES_HINT)
    if url.scheme == "file":
        if url.netloc not in ("", "localhost") or url.query or url.fragment:
            raise VisionError("remote file hosts, queries and fragments are not supported")
        path = unquote(url.path)
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
    else:
        path = source
    root = None if allow_local_files is True else allow_local_files
    return _read_file(path, root)


def _read_file(path, root=None) -> bytes:
    if str(path).replace("\\", "/").startswith("//"):
        raise VisionError("network shares and device paths are not supported")
    try:
        p = Path(path).resolve()
        r = Path(root).resolve() if root is not None else None
    except OSError:
        raise VisionError("invalid local file path") from None
    if str(p).replace("\\", "/").startswith("//"):
        raise VisionError("network shares are not supported")
    if r is not None and not p.is_relative_to(r):
        raise VisionError("local image files must be inside the folder given with %s" % LOCAL_FILES_HINT)
    try:
        with p.open("rb") as f:
            raw = f.read(MAX_IMAGE_BYTES + 1)
    except OSError:
        raise VisionError("cannot read the image file") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise VisionError("image exceeds 24 MiB")
    return raw


def _checkerboard_composite(rgba: np.ndarray) -> np.ndarray:
    """RGBA uint8 -> RGB uint8 on the official 8-pixel checkerboard (204 / 153), integer rounding as the official processor:
    rgb = (rgb * a + bg * (255 - a) + 127) // 255."""
    h, w, _ = rgba.shape
    yy = (np.arange(h) // 8)[:, None]
    xx = (np.arange(w) // 8)[None, :]
    bg = np.where((yy + xx) % 2 == 0, 204, 153).astype(np.int32)[:, :, None]
    a = rgba[:, :, 3:4].astype(np.int32)
    return ((rgba[:, :, :3].astype(np.int32) * a + bg * (255 - a) + 127) // 255).astype(np.uint8)


def decode_image(data: bytes) -> np.ndarray:
    """Image file bytes -> uint8 RGB [height, width, 3] at the file's own size (EXIF orientation applied, transparency
    composited).  VisionError for anything that is not one decodable still image within the size limits."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise VisionError("image data must be bytes")
    if len(data) > MAX_IMAGE_BYTES:
        raise VisionError("image exceeds 24 MiB")
    if len(data) == 0:
        raise VisionError("the image data is empty")
    try:
        from PIL import Image, ImageOps
    except ImportError as ex:
        raise ImportError("image input needs Pillow: pip install Pillow (into the Python that runs yasakani)") from ex
    try:
        with warnings.catch_warnings():                      # Pillow warns above 89 million pixels: refused below anyway
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(bytes(data)), formats=still_formats(Image))   # raster formats only: never EPS or WMF
        with image:
            if image.width < 1 or image.height < 1:
                raise VisionError("the image has no pixels")
            if image.width * image.height > MAX_PIXELS:
                raise VisionError("image exceeds 40 million pixels")
            if max(image.size) > MAX_ASPECT * min(image.size):
                raise VisionError("image aspect ratio exceeds %d:1" % MAX_ASPECT)
            if getattr(image, "n_frames", 1) != 1 or getattr(image, "is_animated", False):
                raise VisionError(STILL_ONLY)
            image.load()
            image = ImageOps.exif_transpose(image)
            transparent = image.has_transparency_data if hasattr(image, "has_transparency_data") else (
                image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in image.info)
            if transparent:
                return _checkerboard_composite(np.asarray(image.convert("RGBA"), dtype=np.uint8))
            return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    except VisionError:
        raise
    except Image.DecompressionBombError:                     # Pillow's own size limit, checked inside Image.open
        raise VisionError("image exceeds 40 million pixels") from None
    except Image.UnidentifiedImageError:
        raise VisionError("the data is not a still image format yasakani reads (PNG, JPEG, WebP, GIF, BMP or TIFF; video is not "
                          "supported)") from None
    except MemoryError:
        raise VisionError("the image is too large to decode") from None
    except Exception as ex:                                  # noqa: BLE001 - truncated or damaged image data
        raise VisionError("the image cannot be decoded (%s)" % (str(ex)[:120] or type(ex).__name__)) from None


def prepare_rgb(rgb: np.ndarray, max_image_tokens: int = DEFAULT_IMAGE_TOKENS) -> PreparedImage:
    """An RGB array (uint8 [height, width, 3]) -> the image the encoder reads: unchanged within the cap, else shrunk with
    Pillow LANCZOS (aspect ratio kept) to at most ``max_image_tokens`` tokens."""
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise VisionError("an image must be uint8 [height, width, 3]")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    w, h = fit_to_cap(width, height, max_image_tokens)
    shrunk = (w, h) != (width, height)
    if shrunk:
        from PIL import Image
        rgb = np.ascontiguousarray(np.asarray(Image.fromarray(rgb, "RGB").resize((w, h), Image.Resampling.LANCZOS),
                                              dtype=np.uint8))
    rows, cols, n = grid(w, h)
    return PreparedImage(rgb=rgb, width=w, height=h, rows=rows, cols=cols, tokens=n, key=image_key(rgb),
                         original_size=(width, height), shrunk=shrunk)


def prepare_image(source: Union[bytes, bytearray, memoryview, str, os.PathLike],
                  max_image_tokens: int = DEFAULT_IMAGE_TOKENS) -> PreparedImage:
    """Image file bytes, or the path of an image file -> PreparedImage (see decode_image and prepare_rgb)."""
    check_image_cap(max_image_tokens)
    data = source if isinstance(source, (bytes, bytearray, memoryview)) else _read_file(source)
    return prepare_rgb(decode_image(data), max_image_tokens)


def request_token_limit(max_image_tokens: int = DEFAULT_IMAGE_TOKENS, max_request_tokens: Optional[int] = None) -> int:
    cap = check_image_cap(max_image_tokens)
    if max_request_tokens is None:
        return REQUEST_IMAGE_TOKENS_FACTOR * cap
    if not _int_value(max_request_tokens) or not cap <= max_request_tokens <= 65536:
        raise VisionError("the image token limit per request must be an integer from %d (the per-image limit) to 65536, not %r"
                          % (cap, max_request_tokens))
    return max_request_tokens


def check_request(images: Sequence[PreparedImage], max_image_tokens: int = DEFAULT_IMAGE_TOKENS,
                  max_request_tokens: Optional[int] = None, max_images: int = MAX_IMAGES) -> int:
    """The limits of one request: number of images, tokens per image, tokens of all images.  Returns the total image tokens."""
    limit = request_token_limit(max_image_tokens, max_request_tokens)
    if len(images) > max_images:
        raise VisionError("at most %d images are supported per request (got %d)" % (max_images, len(images)))
    total = 0
    for k, im in enumerate(images):
        if im.tokens > max_image_tokens:
            raise VisionError("image %d has %d image tokens; the limit per image is %d" % (k + 1, im.tokens, max_image_tokens))
        total += im.tokens
    if total > limit:
        raise VisionError("the images of this request take %d image tokens; the limit per request is %d" % (total, limit))
    return total


# ---- reference preprocessing (NumPy) -------------------------------------------------------------------------------------------------
def pixel_values(rgb: np.ndarray) -> np.ndarray:
    """float32 [rows*cols, 40, 40, 3]: the image padded at the bottom and the right to whole tiles with -1 (0..255 scale),
    normalised v = (p - 127.5) / 127.5, tiles in raster order (t = r * cols + c), pixels (y, x, channel) inside a tile."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise VisionError("an image must be uint8 [height, width, 3]")
    h, w = rgb.shape[:2]
    rows, cols, n = grid(int(w), int(h))
    pad = np.full((rows * TILE, cols * TILE, 3), PAD_VALUE, dtype=np.float32)
    pad[:h, :w] = rgb
    v = (pad - np.float32(127.5)) / np.float32(127.5)
    return np.ascontiguousarray(v.reshape(rows, TILE, cols, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(n, TILE, TILE, 3))


def native_patches(pv: np.ndarray) -> np.ndarray:
    """pixel_values [N, 40, 40, 3] -> the encoder's patch rows [4N, 1200]: for token t its 2 x 2 native 20 x 20 patches in the
    order (top-left, top-right, bottom-left, bottom-right), i.e. sequence index 1 + 4t + 2*my + mx; values in (y, x, channel)
    order.  The class token (sequence index 0) is not included."""
    pv = np.asarray(pv)
    if pv.ndim != 4 or pv.shape[1:] != (TILE, TILE, 3):
        raise VisionError("pixel_values must be [N, 40, 40, 3]")
    n = pv.shape[0]
    p = pv.reshape(n, 2, NATIVE_PATCH, 2, NATIVE_PATCH, 3).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(p.reshape(4 * n, 3 * NATIVE_PATCH * NATIVE_PATCH))


# ---- host cache of image vectors ------------------------------------------------------------------------------------------------------
class FeatureCache:
    """LRU cache: image key -> bfloat16 image vectors (uint16 [tokens, 2304]) in host memory, at most ``max_bytes`` in all.
    Thread-safe; it stores its own copy of what it is given."""

    def __init__(self, max_bytes: int = MEMORY_CACHE_BYTES):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self._items: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: str, tokens: Optional[int] = None) -> Optional[np.ndarray]:
        with self._lock:
            a = self._items.get(key)
            if a is None or (tokens is not None and a.shape[0] != tokens):
                return None
            self._items.move_to_end(key)
            return a

    def put(self, key: str, features: np.ndarray) -> bool:
        a = np.asarray(features)
        if a.dtype != np.uint16 or a.ndim != 2 or a.shape[1] != TEXT_HIDDEN or a.shape[0] < 1:
            raise VisionError("image vectors must be bfloat16 bits (uint16) [tokens, %d]" % TEXT_HIDDEN)
        if a.nbytes > self.max_bytes:
            return False
        a = np.array(a, copy=True)
        a.setflags(write=False)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.bytes -= old.nbytes
            while self._items and self.bytes + a.nbytes > self.max_bytes:
                _, gone = self._items.popitem(last=False)
                self.bytes -= gone.nbytes
            self._items[key] = a
            self.bytes += a.nbytes
        return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.bytes = 0


# ---- the tower's weights --------------------------------------------------------------------------------------------------------------
@dataclass
class SafetensorsRange:
    """The ``model.visual.*`` tensors of a safetensors file, checked: {name: (absolute begin, absolute end)}, and the one
    contiguous byte range [begin, end) they fill."""
    path: Path
    entries: Dict[str, Tuple[int, int]]
    begin: int
    end: int
    metadata: dict


MAX_HEADER_BYTES = 256 * 1024 * 1024


def read_safetensors_header(path) -> Tuple[dict, int, int]:
    """(header without __metadata__, the absolute offset of the data, the file size) of a safetensors file: reads only the
    8-byte length and the JSON header."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(8)
            if len(head) < 8:
                raise VisionError("%s is not a safetensors file (%d bytes)" % (path.name, size))
            (n,) = struct.unpack("<Q", head)
            if n > size - 8 or n > MAX_HEADER_BYTES:
                raise VisionError("%s: the header length %d does not fit the file (%d bytes)" % (path.name, n, size))
            raw = f.read(n)
    except OSError as ex:
        raise VisionError("cannot read %s (%s)" % (path.name, ex.strerror or type(ex).__name__)) from None
    try:
        header = json.loads(raw.decode("utf-8"))
    except ValueError as ex:
        raise VisionError("%s: the header is not JSON (%s)" % (path.name, ex)) from None
    if not isinstance(header, dict):
        raise VisionError("%s: the header is not a JSON object" % path.name)
    return header, 8 + n, size


def safetensors_vision_range(path) -> SafetensorsRange:
    """Check the ``model.visual.*`` entries of a safetensors header against the table (name set, dtype, shape, byte count,
    offsets inside the file) and that they form one contiguous byte range.  Reads only the header."""
    path = Path(path)
    header, base, size = read_safetensors_header(path)
    meta = header.pop("__metadata__", {})
    room = size - base
    entries: Dict[str, Tuple[int, int]] = {}
    for name, e in header.items():
        if not name.startswith(PREFIX):
            continue
        if not isinstance(e, dict):
            raise VisionError("%s: the header entry of %s is not an object" % (path.name, name))
        shape, off = e.get("shape"), e.get("data_offsets")
        if not isinstance(shape, list) or not all(_int_value(s) and s >= 0 for s in shape):
            raise VisionError("%s: the shape of %s is not a list of sizes: %r" % (path.name, name, shape))
        if not isinstance(off, list) or len(off) != 2 or not all(_int_value(x) for x in off):
            raise VisionError("%s: the data_offsets of %s are not two integers: %r" % (path.name, name, off))
        check_tensor(name, str(e.get("dtype")), shape, off[1] - off[0])   # the table first: the clearest message
        entries[name] = (off[0], off[1])
    if not entries:
        raise VisionError("%s has no model.visual.* tensors: %s" % (path.name, NO_TOWER))
    missing = [n for n in VISION_TENSORS if n not in entries]
    if missing:
        raise VisionError("%s: %d image-tower tensors are missing, first: %s" % (path.name, len(missing), missing[0]))
    for name, (b, en) in entries.items():
        if b < 0 or en < b or en > room:
            raise VisionError("%s: the data of %s (bytes %d to %d) is not inside the file: truncated or damaged" % (path.name, name,
                                                                                                                   b, en))
        entries[name] = (base + b, base + en)
    spans = sorted(entries.values())
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        if b0 != a1:
            raise VisionError("%s: the model.visual.* tensors are not one contiguous byte range (gap or overlap at byte %d)"
                              % (path.name, a1))
    begin, end = spans[0][0], spans[-1][1]
    if end - begin != VISION_BYTES:
        raise VisionError("%s: the image tower has %d bytes, expected %d" % (path.name, end - begin, VISION_BYTES))
    return SafetensorsRange(path=path, entries=entries, begin=begin, end=end, metadata=meta if isinstance(meta, dict) else {})


def check_vision_header(header: dict, where: str = VISION_FILE) -> None:
    """The extra keys of an image-tower .yskn file: kind, vision_config, text_hidden_size."""
    if header.get("kind") != VISION_KIND:
        raise VisionError("%s: kind %r, expected %r" % (where, header.get("kind"), VISION_KIND))
    check_vision_config(header.get("vision_config") or {}, where + " (header)", alone=True)
    if not _same(header.get("text_hidden_size"), TEXT_HIDDEN):
        raise VisionError("%s: text_hidden_size %r, expected %d" % (where, header.get("text_hidden_size"), TEXT_HIDDEN))


@dataclass
class VisionWeights:
    """Where a model folder keeps its image tower."""
    folder: Path
    kind: str                  # "yskn" (inkling_2_mini_vision.yskn) or "hf-safetensors" (the published weights file)
    path: Path
    entry: dict                # the manifest entry ({} for hf-safetensors)


def _inside(folder: Path, name) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or Path(name).drive or ".." in Path(name).parts:
        raise VisionError("%s: %r is not a file inside the model folder" % (MANIFEST, name))
    return folder / name


def vision_weights(folder) -> Optional[VisionWeights]:
    """The image tower of a model folder, or None when it has none.  Converted folders list the separate file in the
    ``vision`` entry of yasakani.json; folders whose weights are the published safetensors file read ``model.visual.*``
    from it.  Also checks the vision_config of the folder's config.json (VisionError when it differs)."""
    folder = Path(folder)
    try:
        meta = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise VisionError("cannot read %s in the model folder (%s)" % (MANIFEST, ex)) from None
    if not isinstance(meta, dict):
        raise VisionError("%s is not a JSON object" % MANIFEST)
    v = meta.get("vision")
    w = meta.get("weights") or {}
    if isinstance(v, dict) and v.get("file"):
        if str(v.get("format") or "yskn") != "yskn" or int(v.get("version") or 2) != 2:
            raise VisionError("vision file format %r version %r is not supported by this version of the package"
                              % (v.get("format"), v.get("version")))
        p = _inside(folder, v["file"])
        if not p.is_file():
            raise VisionError("the image-tower file %s is missing: %s" % (v["file"], NO_TOWER))
        kind, entry = "yskn", dict(v)
    elif isinstance(w, dict) and str(w.get("format") or "hf-safetensors") == "hf-safetensors" and w.get("file"):
        p = _inside(folder, w["file"])
        kind, entry = "hf-safetensors", {}
        if not p.is_file():
            return None
    else:
        return None
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise VisionError("cannot read config.json in the model folder (%s)" % ex) from None
    if kind == "hf-safetensors" and not isinstance(config.get("vision_config"), dict):
        return None
    check_vision_config(config)
    return VisionWeights(folder=folder, kind=kind, path=p, entry=entry)


def iter_vision_tensors(folder_or_weights, check_crc: bool = False) -> Iterator[Tuple[str, memoryview]]:
    """(name, bfloat16 bytes) of each of the 422 tensors, for the engine's ``ink2_vision_set_tensor``.  The bytes are views of
    a read-only memory map (nothing is copied).  Every name, dtype, shape and byte count is checked before the first tensor is
    yielded; ``check_crc`` also checks the CRC-32 of each part of a .yskn file first."""
    vw = folder_or_weights if isinstance(folder_or_weights, VisionWeights) else vision_weights(folder_or_weights)
    if vw is None:
        raise VisionError(NO_TOWER)
    if vw.kind == "yskn":
        from .ink2_yskn import FormatError, YsknFile
        try:
            yf = YsknFile(vw.path)
        except (FormatError, OSError) as ex:
            raise VisionError("%s: %s" % (vw.path.name, ex)) from None
        try:
            check_vision_header(yf.header, vw.path.name)
            names = list(yf.tensors)
            if len(names) != VISION_TENSOR_COUNT or set(names) != set(VISION_TENSORS):
                extra = sorted(set(names) - set(VISION_TENSORS))
                missing = sorted(set(VISION_TENSORS) - set(names))
                raise VisionError("%s: %s" % (vw.path.name, ("unknown tensor %s" % extra[0]) if extra
                                              else "%d tensors are missing, first: %s" % (len(missing), missing[0])))
            for n in names:
                e = yf.tensors[n]
                if e.get("encoding") != "bf16":
                    raise VisionError("%s: %s is stored as %s, expected bf16" % (vw.path.name, n, e.get("encoding")))
                p = yf.part_entry(n, "data")
                check_tensor(n, "BF16", e["shape"], p["nbytes"])
            if check_crc:
                bad = yf.check()
                if bad:
                    raise VisionError("%s: the data of %s is damaged (CRC-32 differs); convert again" % (vw.path.name, bad[0][0]))
            for n in names:
                yield n, memoryview(yf.part(n, "data")).cast("B")
        finally:
            yf.close()
    else:
        r = safetensors_vision_range(vw.path)
        mm = np.memmap(vw.path, dtype=np.uint8, mode="r", offset=r.begin, shape=(r.end - r.begin,))
        try:
            for n, (b, e) in sorted(r.entries.items(), key=lambda kv: kv[1]):
                yield n, memoryview(mm[b - r.begin:e - r.begin]).cast("B")
        finally:
            del mm
