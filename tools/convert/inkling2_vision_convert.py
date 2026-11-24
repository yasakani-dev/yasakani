#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""inkling2_vision_convert.py - add the image tower of Inkling-2 Mini to a converted YASAKANI model folder.

  python inkling2_vision_convert.py --src <downloaded folder> --dst <converted model folder> [--revision SHA]

The published weights file holds the image tower as 422 bfloat16 tensors named ``model.visual.*`` (636,512,256 bytes, one
contiguous byte range of the file).  A converted folder (``--scheme quality`` of inkling2_convert.py, weights format ``yskn``)
does not have them in its weights file.  This tool writes them, unchanged, into a separate file of the folder:

* ``inkling_2_mini_vision.yskn``: the YASAKANI container, version 2 (docs/FORMAT.md, section 10), 422 tensors, encoding
  ``bf16``, the bytes of the published file copied as they are (no conversion), each part aligned to 4,096 bytes with its
  CRC-32.  The header carries ``"kind": "inkling2_vision"``, the ``vision_config`` of config.json, ``"text_hidden_size"`` and
  the source (file name, byte range, revision, SHA-256 of the range).
* ``yasakani.json`` gets the entry ``"vision": {"file", "format", "version", "sha256", "size"}``, and the file is listed in its
  ``sha256`` and ``sizes`` maps so that ``yasakani verify`` checks it.  NOTICE gets one bullet about the file,
  placed before the acceptable-use bullet (or at the end when there is none).

Before anything is written, the tool checks the vision_config of both config.json files against the values the engine is built
for, and every one of the 422 tensor entries of the safetensors header (name, dtype, shape, byte count, offsets); a difference is
refused with a message that names the key or the tensor.  It reads only the header and the byte range of the image tower (never
the whole 19.5 GB file).  It does not touch the text weights.  Running it again gives the same file, byte for byte.

A folder whose weights are the published safetensors file (``--scheme published``) needs no extra file: the engine reads
``model.visual.*`` from that file.  The tool says so and changes nothing.

Nothing is downloaded; nothing in the source folder is executed or changed.  numpy only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEIGHT_NAMES = ("model-00001-of-00001.safetensors", "model.safetensors")
CHUNK = 1 << 24
# The bullet added to NOTICE, wrapped like the other bullets of that file (about 120 characters, two-space continuation).
NOTICE_LINE = ("* %s holds the image tower of the model: the published bfloat16 tensors model.visual.* of\n"
               "  %s, unchanged, in the YASAKANI container format.")
# The one-line form an earlier version of this tool wrote; replaced by NOTICE_LINE so that no second bullet is added.
NOTICE_LINE_V1 = ("* %s holds the image tower of the model: the published bfloat16 tensors model.visual.* of %s,"
                  " unchanged, in the YASAKANI container format (docs/FORMAT.md, section 10).")
# The bullet about the image tower goes before this one, so the acceptable-use bullet stays the last one and covers it.
NOTICE_POLICY = "* The acceptable-use policy above applies"
NOTICE_OLD = "The vision tower is not included."
NOTICE_NEW = "The vision tower is not in this weights file."


def modules():
    """yasakani.inkling2_vision and yasakani.ink2_yskn from the python/ folder of this repository."""
    root = str(HERE.parent.parent / "python")
    if root not in sys.path:
        sys.path.insert(0, root)
    from yasakani import ink2_yskn, inkling2_vision
    return inkling2_vision, ink2_yskn


def say(*a) -> None:
    print(*a, flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path, what: str) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise SystemExit("cannot read %s (%s)" % (what, ex))
    if not isinstance(d, dict):
        raise SystemExit("%s is not a JSON object" % what)
    return d


def write_text_replace(path: Path, text: str) -> None:
    """Write a small text file through a temporary file in the same folder and a rename."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def source_revision(src: Path, weights: str, given: str = None) -> str:
    """The revision of the downloaded folder: --revision, the download metadata of huggingface_hub, or the source entry of a
    yasakani.json in the folder (None when nothing names it)."""
    meta = src / ".cache" / "huggingface" / "download" / (weights + ".metadata")
    found = None
    try:
        lines = meta.read_text(encoding="utf-8").splitlines()
        found = lines[0].strip() if lines and lines[0].strip() else None
    except OSError:
        pass
    if found is None and (src / "yasakani.json").is_file():
        m = read_json(src / "yasakani.json", "yasakani.json of the source")
        rev = (m.get("source") or {}).get("revision") if isinstance(m.get("source"), dict) else None
        found = rev if isinstance(rev, str) and rev else None
    if given and found and given != found:
        raise SystemExit("the source folder names revision %s, not %s" % (found, given))
    return given or found


def plan_source(src: Path):
    """The checks of the source alone (config.json and the safetensors header): (vision_config, weights file name, range)."""
    V, _ = modules()
    if not src.is_dir():
        raise SystemExit("source folder not found: %s" % src)
    src_config = read_json(src / "config.json", "config.json of the source")
    if src_config.get("model_type") != "inkling2_loop":
        raise SystemExit("not an Inkling-2 folder: config.json of the source has model_type %r" % src_config.get("model_type"))
    weights = next((n for n in WEIGHT_NAMES if (src / n).is_file()), None)
    if weights is None:
        raise SystemExit("no weights file (%s) in %s" % (" or ".join(WEIGHT_NAMES), src))
    try:
        vc = V.check_vision_config(src_config, "config.json of the source")
        rng = V.safetensors_vision_range(src / weights)
    except V.VisionError as ex:
        raise SystemExit("refused: %s" % ex)
    return vc, weights, rng


def plan(src: Path, dst: Path, revision: str = None):
    """Every check, before anything is written.  Returns what write() needs, or None when the folder needs no file."""
    V, _ = modules()
    if not dst.is_dir():
        raise SystemExit("model folder not found: %s" % dst)
    manifest = read_json(dst / V.MANIFEST, "%s of the model folder" % V.MANIFEST)
    if manifest.get("architecture") != "inkling2":
        raise SystemExit("%s does not describe an Inkling-2 model folder" % V.MANIFEST)
    w = manifest.get("weights") or {}
    if not isinstance(w, dict) or not w.get("file"):
        raise SystemExit("%s has no weights entry" % V.MANIFEST)
    vc, weights, rng = plan_source(src)
    src_config = read_json(src / "config.json", "config.json of the source")
    dst_config = read_json(dst / "config.json", "config.json of the model folder")
    try:
        V.check_vision_config(dst_config, "config.json of the model folder")
    except V.VisionError as ex:
        raise SystemExit("refused: %s" % ex)
    if dst_config.get("vision_config") != src_config.get("vision_config") or \
            (dst_config.get("text_config") or {}).get("hidden_size") != (src_config.get("text_config") or {}).get("hidden_size"):
        raise SystemExit("refused: the vision_config of the source and of the model folder differ (not the same model)")
    if str(w.get("format") or "hf-safetensors") == "hf-safetensors":
        say("%s: the weights are the published safetensors file; the engine reads the image tower from it (nothing to write)"
            % dst)
        return None
    rev = source_revision(src, weights, revision)
    man_rev = (manifest.get("source") or {}).get("revision") if isinstance(manifest.get("source"), dict) else None
    if rev and man_rev and rev != man_rev:
        raise SystemExit("refused: the source is revision %s, the model folder was converted from revision %s" % (rev, man_rev))
    rev = rev or man_rev
    if not rev:
        raise SystemExit("give the source revision with --revision (no download metadata found)")
    if w.get("file") == V.VISION_FILE:
        raise SystemExit("refused: the weights file of the folder is named %s" % V.VISION_FILE)
    return dict(manifest=manifest, vision_config=vc, weights=weights, range=rng, revision=rev)


def range_sha256(path: Path, begin: int, end: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(begin)
        left = end - begin
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise SystemExit("%s ends inside the image tower (truncated)" % path.name)
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def write(src: Path, dst: Path, p: dict) -> dict:
    V, Y = modules()
    rng = p["range"]
    order = sorted(rng.entries.items(), key=lambda kv: kv[1])            # file order of the published checkpoint
    say("source   %s: %d image-tower tensors, bytes %d to %d (%d bytes)" % (p["weights"], len(order), rng.begin, rng.end,
                                                                          rng.end - rng.begin))
    src_sha = range_sha256(rng.path, rng.begin, rng.end)
    tensors = [dict(name=n, encoding="bf16", shape=list(V.VISION_TENSORS[n]), role="vision") for n, _ in order]
    model = {"arch": V.VISION_KIND, "model_id": "inkling-2-mini-vision", "converter": "tools/convert/inkling2_vision_convert.py"}
    extra = {"kind": V.VISION_KIND, "vision_config": p["vision_config"], "text_hidden_size": V.TEXT_HIDDEN,
             "source": {"file": p["weights"], "range": [rng.begin, rng.end], "revision": p["revision"], "sha256": src_sha}}
    out = dst / V.VISION_FILE
    tmp = dst / (V.VISION_FILE + ".tmp")
    w = Y.Writer(tmp, model=model, tensors=tensors, extra=extra)
    h = hashlib.sha256()
    done = False
    try:
        with open(rng.path, "rb") as f:
            for name, (b, e) in order:
                f.seek(b)
                data = f.read(e - b)
                if len(data) != e - b:
                    raise SystemExit("%s ends inside %s (truncated); nothing was replaced" % (p["weights"], name))
                h.update(data)
                w.put(name, "data", np.frombuffer(data, dtype="<u2"))
        w.close()
        if h.hexdigest() != src_sha:
            raise SystemExit("%s changed while it was read; nothing was replaced" % p["weights"])
        os.replace(tmp, out)
        done = True
    finally:
        if not w.f.closed:
            w.f.close()
        if not done:                                         # a half-written file is never left in the folder
            try:
                tmp.unlink()
            except OSError:
                pass
    size = out.stat().st_size
    sha = sha256_of(out)
    say("wrote    %s: %d tensors, %d bytes, SHA-256 %s" % (V.VISION_FILE, len(order), size, sha))
    return dict(file=V.VISION_FILE, size=size, sha256=sha, source_sha256=src_sha)


def place_notice_bullet(text: str, bullet: str, known: tuple) -> str:
    """NOTICE text with exactly one ``bullet``, placed before the acceptable-use bullet so that bullet stays the last one.

    Any form of the bullet in ``known`` (this form or an older one, anywhere in the text) is taken out first, so a NOTICE
    made by an earlier version of this tool comes out the same as a fresh one.  Without an acceptable-use bullet the
    bullet goes at the end.
    """
    for old in known:
        while old + "\n" in text:
            text = text.replace(old + "\n", "", 1)
        if text.endswith(old):
            text = text[:-len(old)]
    lines = text.rstrip("\n").split("\n")
    at = next((i for i, s in enumerate(lines) if s.startswith(NOTICE_POLICY)), None)
    if at is None:
        return "\n".join(lines) + "\n" + bullet + "\n"
    return "\n".join(lines[:at] + bullet.split("\n") + lines[at:]) + "\n"


def update_folder(dst: Path, p: dict, r: dict) -> dict:
    """NOTICE line and the manifest entry (vision, sha256, sizes)."""
    V, _ = modules()
    manifest = p["manifest"]
    sums = manifest.get("sha256") if isinstance(manifest.get("sha256"), dict) else {}
    sizes = manifest.get("sizes") if isinstance(manifest.get("sizes"), dict) else {}
    notice = dst / "NOTICE"
    if notice.is_file():
        text = notice.read_text(encoding="utf-8")
        line = NOTICE_LINE % (V.VISION_FILE, p["weights"])
        new = place_notice_bullet(text.replace(NOTICE_OLD, NOTICE_NEW), line,
                                  (line, NOTICE_LINE_V1 % (V.VISION_FILE, p["weights"])))
        if new != text:
            write_text_replace(notice, new)
        if "NOTICE" in sums:
            sums["NOTICE"], sizes["NOTICE"] = sha256_of(notice), notice.stat().st_size
    manifest["vision"] = {"file": r["file"], "format": "yskn", "version": 2, "sha256": r["sha256"], "size": r["size"],
                          "note": "the published image-tower tensors (model.visual.*), unchanged"}
    sums[r["file"]], sizes[r["file"]] = r["sha256"], r["size"]
    manifest["sha256"], manifest["sizes"] = sums, sizes
    write_text_replace(dst / V.MANIFEST, json.dumps(manifest, indent=1) + "\n")
    say("wrote    %s: vision entry added" % V.MANIFEST)
    return manifest


def convert_vision(src: Path, dst: Path, revision: str = None) -> dict:
    p = plan(Path(src), Path(dst), revision)
    if p is None:
        return {}
    r = write(Path(src), Path(dst), p)
    update_folder(Path(dst), p, r)
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the downloaded folder (config.json and the published .safetensors weights)")
    ap.add_argument("--dst", required=True, help="the converted model folder (yasakani.json, weights format yskn)")
    ap.add_argument("--revision", help="source revision (default: from the download metadata, or the folder's manifest)")
    a = ap.parse_args(argv)
    convert_vision(Path(a.src), Path(a.dst), a.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
