#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained, ad-hoc signed Apple Silicon app and checksum ZIP.

Install requirements-build.txt into a virtual environment first. No downloads,
model files, certificates, notarization or publication are performed here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "macos-dist")
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("build on an Apple Silicon Mac")
    if sys.version_info[:2] != (3, 14):
        parser.error("the pinned build uses CPython 3.14")
    from yasakani import __version__
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app_out = output / "YASAKANI.app"
    if app_out.exists():
        parser.error("output already contains YASAKANI.app; use a fresh --output folder")
    build_root = ROOT / "build"
    build_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="macos-package-", dir=build_root) as temp:
        work = Path(temp)
        env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="14.0", PYINSTALLER_CONFIG_DIR=str(work / "pyinstaller-cache"))
        native = work / "native"
        run("cmake", "-S", ROOT, "-B", native, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0",
            "-DCMAKE_OSX_ARCHITECTURES=arm64", env=env)
        run("cmake", "--build", native, "--config", "Release", "--parallel", "4", env=env)
        run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "yasakani",
            "--target-architecture", "arm64", "--distpath", work / "frozen", "--workpath", work / "freeze-work",
            "--specpath", work, "--paths", ROOT / "python", "--collect-all", "tokenizers",
            "--add-binary", str(native / "bin" / "libyskn_metal.dylib") + ":yasakani/bin",
            ROOT / "macos" / "runtime_entry.py", env=env)
        app = work / "YASAKANI.app"
        contents = app / "Contents"
        resources = contents / "Resources"
        binaries = contents / "MacOS"
        binaries.mkdir(parents=True)
        resources.mkdir()
        run("xcrun", "swiftc", "-O", "-target", "arm64-apple-macos14.0", "-module-cache-path", work / "swift-cache",
            ROOT / "macos" / "Launcher.swift", "-o", binaries / "YASAKANI", env=env)
        with (ROOT / "macos" / "Info.plist").open("rb") as stream:
            info = plistlib.load(stream)
        info.update(CFBundleShortVersionString=__version__, CFBundleVersion=__version__, LSMinimumSystemVersion="14.0")
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(info, stream)
        shutil.copytree(work / "frozen" / "yasakani", resources / "runtime", symlinks=True)
        shutil.copy2(ROOT / "quickstart" / "chat.html", resources / "chat.html")
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(ROOT / name, resources / name)
        shutil.copy2(ROOT / "docs" / "MACOS.md", resources / "MACOS.md")
        # Preserve the license texts distributed by every bundled Python dependency.
        licenses = resources / "licenses"
        licenses.mkdir()
        versions = {}
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            versions[name] = dist.version
            for entry in dist.files or []:
                if any(word in entry.name.lower() for word in ("license", "copying", "notice")):
                    source = Path(dist.locate_file(entry))
                    if source.is_file():
                        destination = licenses / name / str(entry)
                        # Distribution records must not be able to escape the license directory.
                        if ".." not in entry.parts and not entry.is_absolute():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, destination)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        manifest = dict(version=__version__, architecture="arm64", minimum_macos="14.0", python=platform.python_version(),
                        source_revision=revision, source_dirty=dirty, packages=versions, signing="ad-hoc; not notarized")
        (resources / "build-info.json").write_text(json.dumps(manifest, indent=2) + "\n")
        run("codesign", "--force", "--deep", "--sign", "-", app)
        run("codesign", "--verify", "--deep", "--strict", app)
        run(resources / "runtime" / "yasakani", "--version")
        run(resources / "runtime" / "yasakani", "serve", "--help")
        shutil.copytree(app, app_out, symlinks=True)
    archive = output / ("YASAKANI-%s-macos-arm64.zip" % __version__)
    run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", app_out, archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / (archive.name + ".sha256")).write_text(digest + "  " + archive.name + "\n")
    print(archive)


if __name__ == "__main__":
    main()
