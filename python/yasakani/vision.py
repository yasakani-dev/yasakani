# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Optional still-image input. No Pillow or native vision library is imported until an image is requested."""
from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

import numpy as np

MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_IMAGES = 16


class PreparedVisual(tuple):
    """Preserve the (ids, in_reasoning) interface; attach visual data to this preparation, not shared runner state."""
    def __new__(cls, ids, in_reasoning, visual):
        value = super().__new__(cls, (ids, in_reasoning))
        value.visual = visual
        return value


def split_images(messages):
    sources, out = [], []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
            out.append(message)
            continue
        if message.get("role") != "user":
            from .chat import RequestError
            raise RequestError("images are supported only in user messages")
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                value = part.get("image_url")
                if not isinstance(value, dict) or not isinstance(value.get("url"), str) or not value["url"]:
                    from .chat import RequestError
                    raise RequestError("image_url needs a nonempty url string")
                sources.append(value["url"])
                parts.append({"type": "text", "text": MARKER})
            else:
                parts.append(part)
        out.append(dict(message, content=parts))
    if len(sources) > MAX_IMAGES:
        from .chat import RequestError
        raise RequestError("at most %d images are supported per request" % MAX_IMAGES)
    return (out if sources else messages), sources


def read_image(source, *, allow_local_files=False):
    """Read a base64 data URI. Local files require an explicit server-side opt-in; never fetch a remote URL."""
    if source.startswith("data:"):
        header, sep, data = source.partition(",")
        if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("images must use a base64 data:image URI")
        if len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("image exceeds 24 MiB")
        raw = base64.b64decode(data, validate=True)
    else:
        if not allow_local_files:
            raise ValueError("only data:image URIs are enabled; local files require --vision-allow-local-files")
        url = urlsplit(source)
        if url.scheme == "file":
            if url.netloc not in ("", "localhost") or url.query or url.fragment:
                raise ValueError("remote file hosts, queries and fragments are not supported")
            path = unquote(url.path)
            if len(path) > 2 and path[0] == "/" and path[2] == ":":
                path = path[1:]
        elif url.scheme and not (len(url.scheme) == 1 and source[1:2] == ":"):
            raise ValueError("remote image URLs are not supported; use a data:image URI")
        else:
            path = source
        if path.replace("\\", "/").startswith("//"):
            raise ValueError("network shares and device paths are not supported")
        path = Path(path).resolve()
        if str(path).replace("\\", "/").startswith("//"):
            raise ValueError("network shares are not supported")
        with path.open("rb") as stream:
            raw = stream.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 24 MiB")
    try:
        from PIL import Image, ImageOps
    except ImportError as ex:
        raise ImportError("image input needs Pillow (install yasakani[vision])") from ex
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("image exceeds 40 million pixels")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("only still images are supported")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def validate_features(item):
    grid = np.asarray(item["grid"])
    if grid.shape != (2,) or grid.dtype.kind not in "iu" or np.any(grid <= 0):
        raise ValueError("invalid image grid")
    h, w = map(int, grid)
    n = h * w
    features = item["features"]
    if features.dtype != np.uint16 or features.shape != (n, 5120):
        raise ValueError("invalid bfloat16 image feature shape")
    # Reject infinities/NaNs in cached bfloat16 data too.
    if np.any((features & 0x7f80) == 0x7f80):
        raise ValueError("non-finite image features")
    positions = np.stack([np.zeros(n, dtype=np.int32), np.arange(n, dtype=np.int32) // w,
                          np.arange(n, dtype=np.int32) % w])
    if item["positions"].dtype != np.int32 or not np.array_equal(positions, item["positions"]):
        raise ValueError("image positions differ from mtmd")
    return h, w, positions


def build_visual_prompt(prompt, images, image_id, *, max_tokens=None):
    """Expand image placeholders and calculate Qwen's [time, height, width] rotary positions."""
    if not images or sum(int(t) == image_id for t in prompt) != len(images):
        raise ValueError("image placeholder count does not match supplied images")
    grids = [validate_features(item) for item in images]
    size = len(prompt) - len(images) + sum(h * w for h, w, _ in grids)
    if max_tokens is not None and size >= max_tokens:
        from .chat import ContextLengthError
        raise ContextLengthError("image input exceeds the model's maximum context length (%d expanded tokens)" % size)
    ids, positions, slots, features = [], [], [], []
    offset, index = 0, 0
    for token in prompt:
        if int(token) != image_id:
            ids.append(int(token))
            positions.append([offset] * 3)
            offset += 1
        else:
            h, w, relative = grids[index]
            n = h * w
            slots.extend(range(len(ids), len(ids) + n))
            ids.extend([image_id] * n)
            positions.extend((relative + offset).T.tolist())
            features.append(images[index]["features"])
            offset += max(h, w)
            index += 1
    return np.asarray(ids, np.int32), dict(features=np.ascontiguousarray(np.concatenate(features)),
        slots=np.asarray(slots, np.int32), positions=np.ascontiguousarray(np.asarray(positions, np.int32).T),
        delta=offset - len(ids))


class VisionProcessor:
    """Lazy CPU encoder with a bounded memory cache and optional disk cache. Calls are serialized."""
    def __init__(self, *, mmproj, llama_dir, model, cache=None, threads=8, allow_local_files=False):
        self.mmproj, self.llama_dir, self.model = map(Path, (mmproj, llama_dir, model))
        self.cache = Path(cache) if cache else None
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError("vision threads must be positive")
        self.allow_local_files = allow_local_files
        self._encoder = None
        self._memory = OrderedDict()
        self._memory_bytes = 0
        self._identity = None
        self._lock = threading.RLock()

    def _fingerprint(self):
        # Replacing a projector/model/backend invalidates cached features, including after a server restart.
        paths = [self.mmproj, self.model, self.llama_dir / "llama.dll", self.llama_dir / "mtmd.dll"]
        paths += sorted(self.llama_dir.glob("ggml*.dll"))
        files = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        return json.dumps(["b10488-cpu-bf16-rne-v1", self.threads, files]).encode()

    def _image_features(self, image):
        identity = self._fingerprint()
        if identity != self._identity:
            self.close()
            self._identity = identity
        key = hashlib.sha256(identity + str(image.size).encode() + image.tobytes()).hexdigest()
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        path = self.cache / (key + ".npz") if self.cache else None
        item = None
        if path and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as stored:
                    item = {k: stored[k].copy() for k in ("features", "positions", "grid")}
                validate_features(item)
            except (ValueError, OSError, KeyError, EOFError):
                item = None
        if item is None:
            if self._encoder is None:
                from ._mtmd import MtmdEncoder
                self._encoder = MtmdEncoder(self.llama_dir, self.mmproj, self.model, threads=self.threads)
            item = self._encoder.encode(image)
            validate_features(item)
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with path.open("xb") as stream:
                        np.savez(stream, **item)
                except FileExistsError:
                    pass
        size = sum(v.nbytes for v in item.values())
        while self._memory and self._memory_bytes + size > 64 * 1024 * 1024:
            _, old = self._memory.popitem(last=False)
            self._memory_bytes -= sum(v.nbytes for v in old.values())
        if size <= 64 * 1024 * 1024:
            self._memory[key] = item
            self._memory_bytes += size
        return item

    def prepare(self, prompt, sources, fmt, max_tokens):
        image_id = fmt.token_id("<|image_pad|>")
        expected = [fmt.token_id(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")]
        if fmt.encode(MARKER).tolist() != expected:
            raise ValueError("tokenizer does not encode image markers as single tokens")
        with self._lock:
            images = [self._image_features(read_image(s, allow_local_files=self.allow_local_files)) for s in sources]
            return build_visual_prompt(prompt, images, image_id, max_tokens=max_tokens)

    def close(self):
        with self._lock:
            if self._encoder is not None:
                self._encoder.close()
                self._encoder = None
            self._memory.clear()
            self._memory_bytes = 0
