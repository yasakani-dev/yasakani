# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""YASAKANI: a fast local inference engine for Qwen3.8-27B on one NVIDIA RTX GPU (tested and tuned on the RTX 5090 under Windows)."""

__version__ = "0.1.7.dev1"
BUILD_NOTE = "test build"            # 0.1.7.dev1 is a test build: wide copy passes on RTX 30 / 40 and RTX 50 other than the 5090

from ._native import NativeLib, YasakaniError, find_library      # noqa: F401
from .engine import Engine, GenResult                            # noqa: F401
from .runtime import ModelPack, RuntimeProfile                   # noqa: F401

__all__ = ["__version__", "BUILD_NOTE", "Engine", "GenResult", "ModelPack", "RuntimeProfile", "NativeLib", "YasakaniError", "find_library"]
