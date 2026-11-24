# -*- coding: utf-8 -*-
"""YASAKANI quickstart (Windows + one NVIDIA RTX GPU of the RTX 30, 40 or 50 series with 16 GB or more; tested on the RTX 5090).

Double-click START.cmd.  This script then
  1. checks this computer (Windows, Python, GPU, NVIDIA driver, free disk space) before anything is downloaded,
  2. lets you pick one of the four models: the three Qwen3.8-27B models and, since 0.3.2, Inkling-2 Mini (experimental; text, and
     images since 0.3.5) (on a GPU with less than 24 GB only the fastest model fits),
  3. downloads the engine zip from the release page and checks its SHA-256,
  4. makes a private Python environment in this folder and installs the package,
  5. downloads the model folder from Hugging Face,
  6. runs `yasakani info`, `yasakani verify` and `yasakani selfcheck`,
  7. and offers: open the page in your browser (chat, speed test, API address, model switch), measure the speed, compare
     look-ahead on and off, chat in the terminal, or start the local server.

From the second run on (the chosen model passed the checks of step 6 in an earlier run) step 7 opens the browser page directly,
without the model question and the menu.  START.cmd --menu shows the menu again.

The browser page (chat.html) is served by this script on 127.0.0.1 only.  It opens as soon as `yasakani serve` is started and shows
the progress while the model loads (GET /webchat/status).  The script passes the requests of that page on to `yasakani serve`, which
runs on this computer too.  Nothing is sent anywhere else.  The quit button and the model switch of the page are POSTs to this
script (/webchat/shutdown, /webchat/switch_model) that need this address in Host and Origin and a one-time token; the script then
stops `yasakani serve` and ends, or starts it again with the other downloaded model on the same address.

Updates: at most once a day, when the menu is shown or the browser page is opened directly, the script asks GitHub's API for the
latest release (one HTTPS request without anything about this computer; it says so first, and --no-update-check turns it off).
The page's "check for updates" asks the same once, at once, only when it is pressed (also with --no-update-check).
Nothing is updated without a yes (at the menu: the question; in the page: the update button).  An update downloads the start files
of that release, checks them against its SHA256SUMS.txt, keeps the current ones in backup/<version>/, puts the new ones in place and
runs the new quickstart.py, which installs its own engine as usual.  --action rollback goes back.

Everything is kept inside the folder this file is in.  Nothing is installed system-wide.
Run it again any time: finished steps are skipped.

Apache License 2.0 - part of https://github.com/yasakani-dev/yasakani
"""
import argparse
import hashlib
import hmac
import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QS_VERSION = "0.3.5"
ENGINE_ZIP = "yasakani-0.3.5-win-x64.zip"
ENGINE_URL = "https://github.com/yasakani-dev/yasakani/releases/download/v0.3.5/" + ENGINE_ZIP
ENGINE_SHA256 = "831576fe908c0dc52494aae6534d2a45e27f310b3ceaba9bbc0b1e65e4a8a7fc"
# Does the engine zip of ENGINE_SHA256 read images (the image part of Inkling-2 Mini: capability CAP_VISION of its library)?  The
# zips of 0.3.5 and later do; the 0.3.4 zip did not.  The launcher always installs that zip, so this is what yasakani/inkling2.py
# library_has_vision() answers for the server it starts; the context note counts the image part only when it is True.  It changes
# together with ENGINE_ZIP / ENGINE_SHA256 and must be True while INK2_IMAGE is on.
ENGINE_IMAGE = True
REPO_URL = "https://github.com/yasakani-dev/yasakani"
SUPPORT_URL = "https://camp-fire.jp/projects/992233/view"
ENGINE_PORT, WEB_PORT = 8765, 8766      # `yasakani serve` and the browser chat; the next free ports are used if these are taken
MAX_BODY = 16 << 20

# ---- updates.  The check is one GET of the API below (5 s, no identifiers); the files come only from this repository's release
# download URLs; the start files are the members of the quickstart zip (release.py: LICENSE + quickstart/*).
UPDATE_API = "https://api.github.com/repos/yasakani-dev/yasakani/releases/latest"
UPDATE_DOWNLOAD = REPO_URL + "/releases/download/"
UPDATE_UA = "yasakani-quickstart"
START_FILES = ("LICENSE", "README.txt", "START.cmd", "chat.html", "lookahead_check.py", "quickstart.py")
MAX_START_FILE, MAX_START_ZIP, MAX_SUMS, MAX_API = 4 << 20, 16 << 20, 64 << 10, 1 << 20
CHECK_EVERY = 24 * 3600
CHECK_NOW_WAIT = 12.0               # the page's "check for updates": how long its request waits for the check (fetch_latest: 5 s)
UPDATE_RC = 75                      # run_web: the page asked for the update; the server is stopped
SWITCH_RC = 76                      # run_web: the page picked another downloaded model (a.web_switch); the server is stopped
QUIT_RC = 77                        # run_web: the quit button of the page; the server is stopped and the launcher ends
CONTROL_MAX_BODY = 4096             # the body of a control request of the page (/webchat/shutdown, /webchat/switch_model)
DEFAULT_MODEL_ID = "qwen3.8-27b"    # the model's id in `yasakani serve` (MODELS[...]["model_id"] for another); once it is ready the
                                    # page gets the engine's own answer
MODEL_ID_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}")    # the engine's id goes to the page only when it is a plain name
GPU_CACHE_SECONDS, GPU_QUERY_TIMEOUT = 2.0, 5        # the GPU line of the page: nvidia-smi at most this often, and its time limit
GPU_SETTLE_SECONDS, GPU_SETTLE_MIB = 15.0, 1024      # a model switch waits this long for the GPU memory to come back this near
FAIL_GRACE = 5.0                    # seconds the page's server waits for a page to ask after `yasakani serve` ended by itself
END_GRACE = 2.0                     # seconds the page's server stays after Enter or the end of the server, so the page can say so
REOPEN_SECONDS = 10.0               # a model switch: how long the page's port is tried again
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'")
# The files of the published Inkling-2 Mini folder of 0.3.2 (revision 8de37e99..., name: bytes).  The download size of the model
# question and of the download message is the sum of this list with INK2_IMAGE's files added or in their place.
INK2_FOLDER_SIZES = {
    "inkling_2_mini_quality.yskn": 15052423172, "mtp_d4.yskn": 209688040, "tokenizer.json": 26178678,
    "SOURCE_MODEL_CARD.md": 54831, "tokenizer_config.json": 13235, "LICENSE": 11358, "chat_template.jinja": 6868,
    "README.md": 6865, "selfcheck.json": 5040, "yasakani.json": 2751, "config.json": 2722, "NOTICE": 1959,
    "special_tokens_map.json": 529, "generation_config.json": 98,
}

# ---- image input of Inkling-2 Mini: ONE more file in its folder (INK2_IMAGE_FILE, listed in yasakani.json under "vision").
# THE SWITCH.  None: the launcher is that of 0.3.4 (text only; images are never mentioned).  On (since 0.3.5): the commit of the
# repository that adds the image file, and name: (bytes, sha256) of every file of that commit that is new or differs from the pinned
# folder above (the image file, yasakani.json with its "vision" entry, NOTICE and README.md).  A new download gets the folder of that
# commit (the image file included), and a folder of 0.3.2 to 0.3.4 is offered only these files (about 0.64 GB; the text weights are
# kept and not read again).  START.cmd --image-file asks again after a no.  It needs an engine zip with image input (ENGINE_IMAGE).
INK2_IMAGE = {"revision": "e186e163e6d140f767a0fd1f1d31ed93e1a3e1a8",
              "files": {"inkling_2_mini_vision.yskn": (637136896, "858a5a5cacf28c1fdae0dcee23350565ac3eba77d86bc02057ce6c3bb249eabb"),
                        "yasakani.json": (3162, "e201cb7f0c1247565b07dd583ca335ded8352ef09b335344fb1481570fbe28fb"),
                        "NOTICE": (2166, "26efb9e1322dcbd3000266093f05b8718778a0c37a7da33366e0700594e282d9"),
                        "README.md": (7581, "7c3b12a4a71eb8d35e1c0bae002289be3c4d1bab44d53fa134732e2ba100326c")}}
INK2_IMAGE_FILE = "inkling_2_mini_vision.yskn"


def ink2_download_bytes(image=None):
    """Bytes of a new download of the Inkling-2 Mini folder: INK2_FOLDER_SIZES, with the files of ``image`` (INK2_IMAGE) added or
    in place of the old ones."""
    sizes = dict(INK2_FOLDER_SIZES)
    for name, (size, _sha) in ((image or {}).get("files") or {}).items():
        sizes[name] = size
    return sum(sizes.values())


MODELS = {
    "quality": {"repo": "yasakani-dev/Qwen3.8-27B-YASAKANI-quality", "need_gb": 16},
    "speed": {"repo": "yasakani-dev/Qwen3.8-27B-YASAKANI-speed", "need_gb": 14},
    "max": {"repo": "yasakani-dev/Qwen3.8-27B-YASAKANI-max-quality", "need_gb": 20},
    # 0.3.2: the converted Inkling-2 Mini (experimental, one request at a time; image input since 0.3.5), at a pinned revision of
    # its repository.  Its files are 15.93 GB with the image file (shown as 15.9; 15.29 without); need_gb is that rounded up plus
    # 1 GB, as for the others.
    "inkling2": {"repo": "yasakani-dev/Inkling-2-Mini-YASAKANI-quality",
                 "revision": (INK2_IMAGE or {}).get("revision") or "8de37e9980397da63b050fe008493a35f1c483e9",
                 "need_gb": 17, "get_gb": "%.1f" % (ink2_download_bytes(INK2_IMAGE) / 1e9), "model_id": "inkling-2-mini-quality"},
}
MODEL_ORDER = ("quality", "speed", "max", "inkling2")   # the order of the model question (1, 2, 3, 4)
MODEL_KEYS = {str(i + 1): name for i, name in enumerate(MODEL_ORDER)}     # the answers of the model question
TEST_MODEL_SRC = "YASAKANI_QS_TEST_MODEL_SRC_"      # + the model's name in capitals: tests only, a folder instead of the download
ACTIONS = ("auto", "menu", "bench", "compare", "web", "chat", "serve", "none", "update", "rollback")
MENU = {"1": "web", "2": "bench", "3": "compare", "4": "chat", "5": "serve", "7": "update"}   # "6" and anything else: quit
MENU_DEFAULT = "1"                  # Enter alone opens the page in the browser

# ---- GPUs (0.1.3).  The engine is built for compute capability 8.6, 8.9 and 12.x (RTX 30, 40 and 50 series) and tested on the
# RTX 5090 only.  The context rule is the one of yasakani/gpu.py, which the package applies each time it opens the model; it is
# repeated here only to tell the context before anything is downloaded (python/tests/test_gpu.py checks that the two agree).
GPU_FIELDS = ("index", "uuid", "name", "compute_cap", "memory.total", "memory.used", "driver_version")
MIN_TOTAL_MIB = 15000                 # "16 GB": the smallest cards the models fit on (fastest model only)
QUALITY_MIN_TOTAL_MIB = 23000         # "24 GB": the quality and the max-quality model need this much
FULL_CONTEXT_MIN_TOTAL_MIB = 30000    # cards with this much memory keep the full context (the RTX 5090)
KV_BYTES_PER_TOKEN, CONTEXT_STEP, MIN_CONTEXT, HEAD_OFF_BELOW, MARGIN_MIB, FULL_CONTEXT = 22400, 16384, 16384, 65536, 512, 262144
MODEL_BYTES = {   # weights, draft model, its output head (the published files)
    "quality": (12365242368, 1164969472, 1411010584),
    "speed": (10311344128, 1164969472, 1411010584),
    "max": (15723962368, 1164969472, 1411010584),
}
OVERHEAD_MIB = {"quality": 2300, "speed": 2100, "max": 2300}
# Inkling-2 Mini (0.3.2): the rule of yasakani/gpu.py ink2_context() with the files of the published folder (weights, drafter), the
# drafter loaded (the default) and 512 rollback positions, repeated here only to tell the context before anything is downloaded
# (python/tests/test_gpu.py checks that the two agree).
INK2_FILES_BYTES = (INK2_FOLDER_SIZES["inkling_2_mini_quality.yskn"], INK2_FOLDER_SIZES["mtp_d4.yskn"])
INK2_BYTES_PER_POSITION = 65536 + 12288 + 10240       # config.json's cache + the engine's rest + the drafter's, per position
INK2_ROLLBACK_MIB, INK2_OVERHEAD_MIB = (512 - 64) * 1.8, 1800
# The fast path (0.3.3; since 0.3.4 --fast auto turns it on for compute capability 8.6, 8.9 and 12.0, the RTX 30, 40 and 50 series): an
# 8-bit cache (one byte per value and one scale byte per 32 values) and no score buffer per position, and INK2_FAST_OVERHEAD_MIB more
# (gpu.py ink2_bytes_per_position).
INK2_FAST_BYTES_PER_POSITION = 65536 // 4 + 65536 // 4 // 32 + 10240
INK2_FAST_OVERHEAD_MIB = 500
INK2_FAST_ENV = "YASAKANI_INK2_FAST"
INK2_DEFAULT_CONTEXT, INK2_CONTEXT_STEP, INK2_MIN_CONTEXT = 32768, 4096, 8192
# The image part of Inkling-2 Mini, when its folder has INK2_IMAGE_FILE: yasakani/gpu.py ink2_vision_mib() with the defaults of
# chat and serve (--vision auto, 1,024 image tokens per image, 8 x that per request; yasakani/inkling2.py vision_plan_mib()),
# repeated here only for the context note (tests/test_quickstart_ink2_vision.py checks that the two agree).
INK2_VISION_WEIGHTS_BYTES, INK2_VISION_SCRATCH_BYTES, INK2_VISION_SCRATCH_BYTES_PER_TOKEN = 636512256, 20480, 86720
INK2_VISION_FEATURE_BYTES_PER_ROW, INK2_VISION_IMAGE_TOKENS, INK2_VISION_REQUEST_FACTOR = 2304 * 2, 1024, 8
INK2_VISION_ENV = "YASAKANI_INK2_VISION"

# ---- the NVIDIA driver (0.1.4).  yskn.dll carries the CUDA 13.2 runtime (linked in, there is no separate runtime DLL).  Under CUDA minor
# version compatibility a program built with any CUDA 13.x toolkit runs on a driver of the R580 branch or newer, which reports CUDA 13.0
# or newer (NVIDIA, CUDA Toolkit release notes, "Minimum Required Driver Version for CUDA Minor Version Compatibility": 13.x >= 580;
# https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).  With an older driver the engine stops at its first
# CUDA call, which comes after the engine and the model have been downloaded, so the quickstart checks the driver in step 1.
MIN_DRIVER = 580                  # the first number of nvidia-smi's driver_version: 580.88 or 616.92 (Windows), 580.65.06 (Linux)
MIN_CUDA_DRIVER_API = 13000       # cuDriverGetVersion() of such a driver: 13000 = CUDA 13.0 (driver 616.92 gives 13040 = CUDA 13.4)

T = {
    "banner": {
        "ja": "YASAKANI クイックスタート {v}  -  Qwen3.8-27B を NVIDIA の RTX 1 枚で動かします（確かめたのは RTX 5090）",
        "en": "YASAKANI quickstart {v}  -  runs Qwen3.8-27B on one NVIDIA RTX card (tested on the RTX 5090)",
    },
    "support": {
        "ja": "開発を応援してください: CAMPFIRE のクラウドファンディング（2026 年 12 月 16 日まで）\n  {u}",
        "en": "Like it? A star on GitHub helps: {r}\n  (The crowdfunding at {u} is open to residents of Japan.)",
    },
    "not_windows": {
        "ja": "このソフトは Windows 11 専用です。",
        "en": "This software runs on Windows 11 only.",
    },
    "old_python": {
        "ja": "Python 3.10 以上が必要です（いまは {v}）。",
        "en": "Python 3.10 or newer is needed (this is {v}).",
    },
    "bad_path": {
        "ja": "このフォルダーの場所に、日本語などの英数字以外の文字が入っています:\n  {p}\n"
              "C:\\yasakani のような英数字だけの場所にフォルダーごと移して、もう一度 START.cmd をダブルクリックしてください。",
        "en": "The path of this folder contains characters outside plain ASCII:\n  {p}\n"
              "Move the folder to a plain path such as C:\\yasakani and double-click START.cmd again.",
    },
    "step": {"ja": "[{i}/{n}] {t}", "en": "[{i}/{n}] {t}"},
    "s_check": {"ja": "このパソコンの確認", "en": "Checking this computer"},
    "s_model": {"ja": "モデルを選ぶ", "en": "Choosing the model"},
    "s_engine": {"ja": "エンジンのダウンロードと確認", "en": "Downloading and checking the engine"},
    "s_venv": {"ja": "Python の環境を作って、パッケージを入れる", "en": "Making the Python environment and installing the package"},
    "s_dl": {"ja": "モデルのダウンロード", "en": "Downloading the model"},
    "s_verify": {"ja": "点検（info・verify・selfcheck）", "en": "Checks (info, verify, selfcheck)"},
    "s_menu": {"ja": "準備ができました", "en": "Ready"},
    "no_smi": {
        "ja": "nvidia-smi が見つかりません。NVIDIA の GPU とドライバーが要ります。ドライバーを入れてから、もう一度試してください。\n"
              "AMD・Intel の GPU と Mac には対応していません。",
        "en": "nvidia-smi was not found. YASAKANI needs an NVIDIA GPU and its driver; install the driver and try again.\n"
              "AMD and Intel GPUs and Macs are not supported.",
    },
    "no_gpu": {
        "ja": "NVIDIA の GPU が見つかりません。YASAKANI は NVIDIA の RTX 30・40・50 シリーズ（GPU のメモリ 16 GB 以上）向けに作ってあります（確かめたのは RTX 5090 だけ）。\n"
              "AMD・Intel の GPU と Mac には対応していません。",
        "en": "No NVIDIA GPU was found. YASAKANI is built for NVIDIA RTX 30, 40 and 50 series cards with 16 GB or more (tested on the RTX 5090 only).\n"
              "AMD and Intel GPUs and Macs are not supported.",
    },
    "gpu_is": {
        "ja": "GPU: {g}（計算能力 {c}、メモリ {t} MiB）",
        "en": "GPU: {g} (compute capability {c}, {t} MiB)",
    },
    "gpu_pick": {"ja": "GPU が {n} 枚あります。使うのは {g} です。", "en": "There are {n} GPUs. Using {g}."},
    "gpu_tested": {"ja": "動作を確かめてある GPU です。", "en": "This is the GPU YASAKANI is tested on."},
    "gpu_other": {
        "ja": "この GPU は動くように作ってありますが、最適化も動作の確認もしていません（確かめたのは RTX 5090 だけです）。\n"
              "速さは RTX 5090 より遅くなる見込みで、動作も保証しません。",
        "en": "YASAKANI is made to run on this GPU, but it has not been optimized or tested on it (only the RTX 5090 has been tested).\n"
              "It is expected to be slower than on the RTX 5090, and its operation is not guaranteed.",
    },
    "gpu_small": {
        "ja": "この GPU のメモリ（{t} MiB）では足りません。YASAKANI には GPU のメモリが 16 GB 以上要ります。",
        "en": "This GPU's memory ({t} MiB) is too small. YASAKANI needs a GPU with 16 GB or more.",
    },
    "gpu_old": {
        "ja": "この GPU（計算能力 {c}）では動きません。動くように作ってあるのは RTX 30・40・50 シリーズ（計算能力 8.6 以上）で、メモリが 16 GB 以上のカードです。",
        "en": "YASAKANI does not run on this GPU (compute capability {c}). It is built for RTX 30, 40 and 50 series cards "
              "(compute capability 8.6 or higher) with 16 GB or more.",
    },
    "gpu_nocode": {
        "ja": "この GPU（計算能力 {c}）用のコードは、この版のエンジンに入っていません。動くのは RTX 30・40・50 シリーズ"
              "（計算能力 8.6・8.9・12.x）です。",
        "en": "The engine of this version has no code for this GPU (compute capability {c}). It runs on RTX 30, 40 and 50 series "
              "cards (compute capability 8.6, 8.9 and 12.x).",
    },
    "driver_ok": {
        "ja": "ドライバー: {d}（580 以上なので使えます）",
        "en": "Driver: {d} (580 or newer: fine)",
    },
    "driver_old": {
        "ja": "NVIDIA のドライバーが古いので止めました（今は {d}）。エンジンは CUDA 13 で作ってあり、ドライバー 580 以上が要ります。\n"
              "新しいドライバーを入れてパソコンを再起動してから、もう一度 START.cmd をダブルクリックしてください。\n"
              "  https://www.nvidia.com/ja-jp/drivers/\n"
              "（エンジンとモデルのダウンロードの前に確かめています）",
        "en": "Stopped: the NVIDIA driver is too old ({d}). The engine is built with CUDA 13 and needs driver 580 or newer.\n"
              "Install a current driver, restart the PC and double-click START.cmd again.\n"
              "  https://www.nvidia.com/en-us/drivers/\n"
              "(This is checked before the engine and the model are downloaded.)",
    },
    "driver_unknown": {
        "ja": "ドライバーの版を読み取れませんでした。エンジンにはドライバー 580 以上が要ります。このまま続けます。\n"
              "後の点検で止まるときは、新しいドライバーを入れてください: https://www.nvidia.com/ja-jp/drivers/",
        "en": "The driver version could not be read. The engine needs driver 580 or newer. Going on.\n"
              "If the checks fail later, install a current driver: https://www.nvidia.com/en-us/drivers/",
    },
    "driver_skip": {
        "ja": "ドライバーが 580 より古いようですが（{d}）、--skip-driver-check の指定があるので続けます。エンジンは恐らく起動できません。",
        "en": "The driver looks older than 580 ({d}), but --skip-driver-check was given, so going on. The engine will most likely not start.",
    },
    "speed_only": {
        "ja": "GPU のメモリが 24 GB 未満なので、使えるのは最速モデルだけです（品質モデル・最高品質モデル・Inkling-2 Mini は入りません）。",
        "en": "The GPU has less than 24 GB, so only the fastest model can be used (the quality and max-quality models and Inkling-2 Mini "
              "do not fit).",
    },
    "quality_no": {
        "ja": "品質モデル・最高品質モデル・Inkling-2 Mini には GPU のメモリが 24 GB 以上要ります。--model speed で最速モデルを選んでください。",
        "en": "The quality and max-quality models and Inkling-2 Mini need a GPU with 24 GB or more. Choose the fastest model with "
              "--model speed.",
    },
    "ctx_ink": {
        "ja": "Inkling-2 Mini の文脈の長さは、起動のたびに GPU の空きメモリから決めます（いまの空きで {n} トークン。最大 32,768）。",
        "en": "The context length of Inkling-2 Mini is chosen from the free GPU memory each time it starts ({n} tokens with what is "
              "free now; at most 32,768).",
    },
    "ctx_auto": {
        "ja": "文脈の長さは、起動のたびに GPU の空きメモリから決めます（いまの空きで {n} トークン。最大 262,144）。",
        "en": "The context length is chosen from the free GPU memory each time the model starts ({n} tokens with what is free now; "
              "at most 262,144).",
    },
    "ctx_head": {
        "ja": "文脈の場所を空けるため、補助モデルの出口の表（bdraft2_k8.shead）は読み込みません（1 回の計算が約 3% 長くなります）。",
        "en": "To make room for the context, the output head of the draft model (bdraft2_k8.shead) is not loaded "
              "(each pass takes about 3% longer).",
    },
    "ctx_short": {
        "ja": "いまの空きメモリでは足りません（GPU のメモリを {u} MiB 使用中）。GPU を使うほかのプログラムを閉じてから、もう一度試してください。",
        "en": "There is not enough free GPU memory now ({u} MiB in use). Close other programs that use the GPU and try again.",
    },
    "disk": {
        "ja": "空き容量: {f:.0f} GB（必要: 約 {n} GB）",
        "en": "Free disk space: {f:.0f} GB (needed: about {n} GB)",
    },
    "disk_short": {
        "ja": "空き容量が足りません。約 {n} GB を空けてから、もう一度試してください。",
        "en": "Not enough free disk space. Free about {n} GB and try again.",
    },
    "pick": {
        "ja": "どのモデルを使いますか？\n"
              "  1) 品質モデル〈おすすめ〉  元のモデルに近い。ファイル 12.4 GB\n"
              "  2) 最速モデル            いちばん速い。ファイル 10.3 GB\n"
              "  3) 最高品質モデル        元のモデルにいちばん近い。ファイル 15.7 GB。生成は品質モデルの約 0.8〜0.9 倍の速さ\n"
              "  4) Inkling-2 Mini〈試験的〉 " + ("文章と画像" if INK2_IMAGE else "文章だけ") + "。依頼は 1 件ずつ。GPU 24 GB 以上。"
              "ダウンロード約 " + MODELS["inkling2"]["get_gb"] + " GB\n"
              "番号を入れて Enter [{d}]: ",
        "en": "Which model?\n"
              "  1) quality model (recommended)  close to the original. 12.4 GB file\n"
              "  2) fastest model                the fastest. 10.3 GB file\n"
              "  3) max-quality model            the closest to the original. 15.7 GB file; writes at about 0.8 to 0.9 times\n"
              "                                  the speed of the quality model\n"
              "  4) Inkling-2 Mini (experimental) " + ("text and images" if INK2_IMAGE else "text only")
              + ", one request at a time, a GPU with 24 GB or more;\n"
              "                                  about " + MODELS["inkling2"]["get_gb"] + " GB download\n"
              "Type the number and press Enter [{d}]: ",
    },
    "picked": {"ja": "モデル: {m}", "en": "Model: {m}"},
    # image input of Inkling-2 Mini: said only while INK2_IMAGE is on
    "img_ask": {
        "ja": "Inkling-2 Mini で画像を読むには、ファイルがもう 1 つ要ります（約 {g} GB）。いまダウンロードしますか？ [y/N]: ",
        "en": "Image input for Inkling-2 Mini needs one more file (about {g} GB). Download it now? [y/N]: ",
    },
    "img_later": {
        "ja": "Inkling-2 Mini は文章だけで使います。画像も使う時は START.cmd --image-file（約 {g} GB）。",
        "en": "Inkling-2 Mini runs with text only. For image input: START.cmd --image-file (about {g} GB).",
    },
    "img_get": {"ja": "画像用のファイルをダウンロードします（約 {g} GB）。", "en": "Downloading the file for image input (about {g} GB)."},
    "img_done": {"ja": "画像用のファイルを確かめて入れました。画像も使えます。",
                 "en": "The file for image input was checked and put in place. Images work now."},
    "img_fail": {
        "ja": "画像用のファイルをダウンロードできませんでした（{e}）。文章だけで続けます。",
        "en": "The file for image input could not be downloaded ({e}). Going on with text only.",
    },
    "img_bad": {
        "ja": "画像用のファイルが公開の値と違うので使いません（{e}）。文章だけで続けます。",
        "en": "The file for image input differs from the published values and was not used ({e}). Going on with text only.",
    },
    "img_disk": {
        "ja": "空き容量が足りないので、画像用のファイル（約 {g} GB）は入れません。文章だけで続けます。",
        "en": "Not enough free disk space for the file for image input (about {g} GB). Going on with text only.",
    },
    "m_quality": {"ja": "品質モデル", "en": "quality model"},
    "m_speed": {"ja": "最速モデル", "en": "fastest model"},
    "m_max": {"ja": "最高品質モデル", "en": "max-quality model"},
    "m_inkling2": {"ja": "Inkling-2 Mini（試験的）", "en": "Inkling-2 Mini model (experimental)"},
    "checks_done": {"ja": "点検は済んでいます（飛ばします）。", "en": "The checks were done before (skipped)."},
    "checks_skipped": {"ja": "--skip-checks なので点検はしません（出力が正しいかは確かめていません）。",
                       "en": "The checks are not run (--skip-checks): the output has not been checked."},
    "engine_have": {"ja": "エンジンは準備済みです。", "en": "The engine is already in place."},
    "engine_local": {"ja": "このフォルダーにある {z} を使います。", "en": "Using {z} from this folder."},
    "engine_get": {"ja": "ダウンロード中: {u}", "en": "Downloading: {u}"},
    "sha_ok": {"ja": "SHA-256 は公開ページの値と同じです。", "en": "The SHA-256 matches the published one."},
    "engine_wait": {"ja": "前のサーバーが終わるのを待っています（数秒かかることがあります）。",
                    "en": "Waiting for the server that ran before to end (this can take a few seconds)."},
    "engine_busy": {
        "ja": "エンジンのフォルダーを置き換えられませんでした。使用中のファイルがあります: {p}\n"
              "ほかの YASAKANI の画面を閉じて、もう一度 START.cmd を開いてください。",
        "en": "The engine folder could not be replaced: a file in it is in use: {p}\n"
              "Close the other YASAKANI windows and run START.cmd again."},
    "sha_bad": {
        "ja": "SHA-256 が違います（ダウンロードが壊れたか、別のファイルです）。使わずに止めました。もう一度試してください。\n  期待: {e}\n  実際: {a}",
        "en": "The SHA-256 differs (broken download or a different file). It was not used; please try again.\n  expected: {e}\n  got:      {a}",
    },
    "dl_fail": {
        "ja": "ダウンロードできませんでした: {e}\nインターネットの接続を確かめて、もう一度 START.cmd をダブルクリックしてください（続きから始まります）。",
        "en": "The download failed: {e}\nCheck the internet connection and double-click START.cmd again (it continues where it stopped).",
    },
    "venv_have": {"ja": "Python の環境は準備済みです。", "en": "The Python environment is already in place."},
    "pip_fail": {
        "ja": "パッケージを入れられませんでした。上のメッセージを確かめてください。",
        "en": "Installing the package failed. See the messages above.",
    },
    "model_have": {"ja": "モデルは準備済みです: {p}", "en": "The model is already in place: {p}"},
    "model_get": {
        "ja": "Hugging Face から {r} をダウンロードします（約 {g} GB。回線によっては時間がかかります）。",
        "en": "Downloading {r} from Hugging Face (about {g} GB; this can take a while).",
    },
    "model_test_src": {
        "ja": "試験用: モデルのファイルを {p} から入れます（{v}。Hugging Face には問い合わせません）。",
        "en": "Test only: the model files come from {p} ({v}; Hugging Face is not asked).",
    },
    "retry_dl": {"ja": "もう一度、別の方法で試します。", "en": "Trying once more with another transfer method."},
    "run": {"ja": "> yasakani {c}", "en": "> yasakani {c}"},
    "check_fail": {
        "ja": "「yasakani {c}」が失敗しました（終了コード {rc}）。上のメッセージを確かめてください。\n"
              "よくある原因: ほかのプログラムが GPU のメモリを使っている、ドライバーが古い、Visual C++ ランタイムが無い。\n"
              "FAQ: {r}/blob/main/docs/FAQ.ja.md",
        "en": "`yasakani {c}` failed (exit code {rc}). See the messages above.\n"
              "Common causes: another program is using the GPU memory, an old driver, or the Visual C++ runtime is missing.\n"
              "FAQ: {r}/blob/main/docs/FAQ.md",
    },
    "checks_ok": {
        "ja": "点検はすべて合格です。このパソコンは、公開されている基準と同じ出力を再現しました。",
        "en": "All checks passed. This computer reproduces the published reference output.",
    },
    "checks_other": {
        "ja": "info と verify は合格です。selfcheck の出力は RTX 5090 で作った基準と途中から違いましたが、基準の文章を読ませたときの予測は"
              "\nRTX 5090 と許容範囲内で一致し、この GPU 自身が生成した文章を読み直した結果も一致しました。"
              "\nこの GPU 向けの最適化と動作の確認はしていません。おかしな出力が続くときは、yasakani info の結果を添えて"
              "\nGitHub の Issues で知らせてください。このまま続けます。",
        "en": "info and verify passed. The output of selfcheck differs part-way from the reference, which was made on an RTX 5090, but when\n"
              "this GPU reads the reference text its predictions agree with the RTX 5090's within the tolerances, and it also agrees with\n"
              "itself when it reads its own output again. YASAKANI has not been optimized or tested on this GPU. If the output looks wrong,\n"
              "please report it on GitHub Issues with the output of yasakani info. Going on.",
    },
    "selfcheck_fail_other": {
        "ja": "selfcheck が不合格でした（終了コード {rc}）。この GPU の予測が、RTX 5090 の基準と許容範囲を超えて違うか、この GPU 自身が"
              "\n生成した文章と合いません（上の表示）。verify は合格なので、この GPU ではエンジンが正しく計算できていない恐れがあります。"
              "\nyasakani info と selfcheck の表示を添えて GitHub の Issues で知らせてください: {r}/issues"
              "\nそれでも使うときは、コマンドプロンプトで START.cmd --skip-checks と打つと点検を飛ばして進みます（出力が正しい保証はありません）。",
        "en": "selfcheck failed (exit code {rc}). This GPU's predictions differ from the RTX 5090's reference by more than the tolerances, or\n"
              "they do not agree with what this GPU generated itself (see above). verify passed, so the engine may not compute correctly\n"
              "on this GPU. Please report it on GitHub Issues with the output of yasakani info and of selfcheck: {r}/issues\n"
              "To use it anyway, type START.cmd --skip-checks in a command prompt: the checks are skipped (the output may be wrong).",
    },
    "menu": {
        "ja": "\n何をしますか？\n"
              "  1) ブラウザーで開く〈おすすめ〉 （チャット・速さの測定・モデルの切り替え）\n"
              "  2) 速さを測る                   （yasakani bench）\n"
              "  3) 先読みあり・なしを比べる     （技術の解説の数字を、このパソコンで確認します。1 分ほど）\n"
              "  4) この画面でチャット           （yasakani chat。終わる時は /exit）\n"
              "  5) サーバーを起動               （OpenAI 互換。http://127.0.0.1:8765/v1 。止める時は Ctrl+C）\n"
              "  6) 終わる\n"
              "番号を入れて Enter [{d}]: ",
        "en": "\nWhat next?\n"
              "  1) open in the browser (recommended)  (chat, speed test, model switch)\n"
              "  2) measure the speed                  (yasakani bench)\n"
              "  3) compare look-ahead on and off      (checks the numbers of the technical notes on this PC; about a minute)\n"
              "  4) chat in this window                (yasakani chat; type /exit to leave)\n"
              "  5) start the server                   (OpenAI-compatible, http://127.0.0.1:8765/v1 ; Ctrl+C stops it)\n"
              "  6) quit\n"
              "Type the number and press Enter [{d}]: ",
    },
    "direct": {
        "ja": "{m}で、ブラウザーの画面を開きます（このモデルの点検は済んでいます）。\n"
              "メニューを出す時は、このフォルダーで開いたコマンドプロンプトで START.cmd --menu と打ってください。",
        "en": "Opening the page in your browser with the {m} (this model passed its checks before).\n"
              "For the menu, type START.cmd --menu in a command prompt opened in this folder.",
    },
    "web_boot": {
        "ja": "サーバーを起動します（モデルの読み込みに 1 分ほどかかることがあります）。",
        "en": "Starting the server (loading the model can take about a minute).",
    },
    "web_ready": {
        "ja": "\nブラウザーの画面を開きます: {u}\n"
              "（開かない時は、上のアドレスをブラウザーに入れてください）\n"
              "モデルの読み込みが終わると、そのままチャットできます。\n"
              "終わる時は、ブラウザーの終了ボタンか、この画面で Enter を押してください。",
        "en": "\nOpening the page in your browser: {u}\n"
              "(If no browser window opens, type the address above into your browser.)\n"
              "The chat works as soon as the model is loaded.\n"
              "To stop, use the quit button of the page or press Enter in this window.",
    },
    "web_ready_manual": {
        "ja": "\nブラウザーで次のアドレスを開いてください: {u}\n"
              "モデルの読み込みが終わると、そのままチャットできます。\n"
              "終わる時は、ブラウザーの終了ボタンか、この画面で Enter を押してください。",
        "en": "\nOpen this address in your browser: {u}\n"
              "The chat works as soon as the model is loaded.\n"
              "To stop, use the quit button of the page or press Enter in this window.",
    },
    "web_again": {
        "ja": "\nブラウザーの画面はそのまま使えます: {u}\n"
              "終わる時は、ブラウザーの終了ボタンか、この画面で Enter を押してください。",
        "en": "\nThe page in your browser goes on at the same address: {u}\n"
              "To stop, use the quit button of the page or press Enter in this window.",
    },
    "web_moved": {
        "ja": "前のアドレスを使えないので、新しいアドレスで開きます。前のタブは閉じてください。",
        "en": "The address used before is not free, so the page opens at a new address. Please close the old tab.",
    },
    "web_loaded": {
        "ja": "モデルの読み込みが終わりました（{s} 秒）。ブラウザーでチャットできます。",
        "en": "The model is loaded ({s} s). The chat in the browser is ready.",
    },
    "web_quit": {
        "ja": "ブラウザーの終了ボタンが押されました。サーバーを止めます。",
        "en": "The quit button of the page was pressed. Stopping the server.",
    },
    "web_switch": {
        "ja": "ブラウザーの画面で{m}が選ばれました。サーバーを止めて、モデルを読み込み直します。",
        "en": "The page picked the {m}. Stopping the server and loading that model.",
    },
    "switch_back": {
        "ja": "{m}は点検に合格しなかったので、{o}に戻します。",
        "en": "The {m} did not pass the checks, so going back to the {o}.",
    },
    "cmp_qwen_only": {
        "ja": "先読みあり・なしの比べは、Qwen3.8 のモデルだけで使えます。",
        "en": "The look-ahead comparison works with the Qwen3.8 models only.",
    },
    "cmp_nofile": {
        "ja": "lookahead_check.py が見つかりません: {p}\nzip をもう一度展開してください。",
        "en": "lookahead_check.py was not found: {p}\nUnpack the zip again.",
    },
    "web_fail": {
        "ja": "サーバーを起動できませんでした。上のメッセージを確かめてください。",
        "en": "The server did not start. See the messages above.",
    },
    "web_died": {
        "ja": "サーバーが途中で止まりました。上のメッセージを確認してください。",
        "en": "The server stopped while it was running. See the messages above.",
    },
    "web_nopage": {
        "ja": "chat.html が見つかりません: {p}\nzip をもう一度展開してください。",
        "en": "chat.html was not found: {p}\nUnpack the zip again.",
    },
    "web_end": {"ja": "チャットを終わりました。", "en": "The chat was stopped."},
    "load_note": {
        "ja": "（モデルを読み込みます。初回は 1 分ほどかかることがあります）",
        "en": "(loading the model; the first time this can take about a minute)",
    },
    "again": {
        "ja": "次からも START.cmd をダブルクリックするだけです（済んだ手順は飛ばします）。\n"
              "自分でコマンドを打つときの設定は {f} に書きました。",
        "en": "Next time just double-click START.cmd again (finished steps are skipped).\n"
              "The settings for typing commands yourself are in {f}.",
    },
    "again_direct": {
        "ja": "次からは START.cmd をダブルクリックすると、ブラウザーの画面が直接開きます。\n"
              "メニューを出す時は、このフォルダーで開いたコマンドプロンプトで START.cmd --menu と打ってください。\n"
              "自分でコマンドを打つ時の設定は {f} に書きました。",
        "en": "Next time, double-clicking START.cmd opens the page in your browser directly.\n"
              "For the menu, type START.cmd --menu in a command prompt opened in this folder.\n"
              "The settings for typing commands yourself are in {f}.",
    },
    "bye": {"ja": "終了します。", "en": "Bye."},
    "stopped": {"ja": "中止しました。", "en": "Stopped."},
    "menu_update": {
        "ja": "  7) 新しい版を確認する           （github.com に問い合わせます）\n",
        "en": "  7) check for a new version            (asks github.com)\n",
    },
    "upd_line": {
        "ja": "github.com に新しい版を問い合わせます（1 回だけ。このパソコンの情報は送りません。止めるときは --no-update-check）",
        "en": "Asking github.com for a new version (one request; nothing about this PC is sent; --no-update-check turns this off)",
    },
    "upd_latest": {"ja": "最新の版です（{v}）。", "en": "This is the latest version ({v})."},
    "upd_fail": {
        "ja": "新しい版を確かめられませんでした（{e}）。このまま続けます。",
        "en": "Could not check for a new version ({e}). Going on.",
    },
    "upd_found": {
        "ja": "新しい版があります: {o} → {n}\n  {u}",
        "en": "A new version is available: {o} -> {n}\n  {u}",
    },
    "upd_ask": {
        "ja": "いま更新しますか？（モデルはそのまま使います） [y/N]: ",
        "en": "Update now? (The models are kept.) [y/N]: ",
    },
    "upd_later": {
        "ja": "更新しません。あとで更新するときは、メニューの 7 か START.cmd --action update です。",
        "en": "Not updating. To update later, use 7 in the menu or START.cmd --action update.",
    },
    "upd_page": {
        "ja": "新しい版があります: {o} → {n}\n"
              "更新する時は、ブラウザーの画面の更新ボタンを押すか、START.cmd --action update を使ってください。",
        "en": "A new version is available: {o} -> {n}\n"
              "To update, press the update button of the page or use START.cmd --action update.",
    },
    "upd_get": {"ja": "ダウンロード中: {u}", "en": "Downloading: {u}"},
    "upd_sha_ok": {"ja": "SHA-256 は、その版の SHA256SUMS.txt の値と同じです。", "en": "The SHA-256 matches the release's SHA256SUMS.txt."},
    "upd_bad": {
        "ja": "更新をやめました: {e}\n今の版はそのままです。",
        "en": "The update was stopped: {e}\nThe current version is unchanged.",
    },
    "upd_backup": {"ja": "今の起動ファイルを控えました: {p}", "en": "The current start files were copied to {p}"},
    "upd_done": {
        "ja": "起動ファイルを {n} に入れ替えました。新しい版を起動します（エンジンとパッケージは新しい版が入れ直します）。\n"
              "元に戻すときは: START.cmd --action rollback",
        "en": "The start files are now those of {n}. Starting the new version (it installs its own engine and package).\n"
              "To go back: START.cmd --action rollback",
    },
    "upd_web": {
        "ja": "チャットの画面から更新が頼まれました。サーバーを止めて更新します。",
        "en": "The chat page asked for the update. Stopping the server and updating.",
    },
    "upd_off": {
        "ja": "新しい版の自動の確認を止めました（--update-check で戻せます）。",
        "en": "The automatic check for a new version is off now (--update-check turns it on again).",
    },
    "upd_on": {"ja": "新しい版の自動の確認を使います（1 日に 1 回まで）。", "en": "The automatic check for a new version is on (at most once a day)."},
    "upd_dev": {
        "ja": "開発用: 更新の問い合わせ先を {u} にしています（GitHub ではありません）。",
        "en": "Development only: updates are looked up at {u} (not GitHub).",
    },
    "rb_none": {"ja": "戻せる控えがありません。", "en": "There is no backup to go back to."},
    "rb_ask": {
        "ja": "起動ファイルを {o} に戻しますか？ [y/N]: ",
        "en": "Put the start files of {o} back? [y/N]: ",
    },
    "rb_done_model": {
        "ja": "起動ファイルを {o} に戻しました。{o} は{m}を使えません。START.cmd を起動して、モデルを選び直してください。",
        "en": "The start files of {o} are back. Version {o} cannot use the {m}. Start START.cmd again and choose a model.",
    },
    "rb_done": {
        "ja": "起動ファイルを {o} に戻しました。その版を起動します（エンジンとパッケージもその版のものに入れ直します）。",
        "en": "The start files of {o} are back. Starting that version (it installs its own engine and package again).",
    },
}


class Ui:
    def __init__(self, lang):
        self.lang = lang

    def t(self, key, **kw):
        s = T[key].get(self.lang) or T[key]["en"]
        return s.format(**kw) if kw else s

    def say(self, key, **kw):
        print(self.t(key, **kw), flush=True)

    def step(self, i, n, key):
        print("\n" + "=" * 72, flush=True)
        print(self.t("step", i=i, n=n, t=self.t(key)), flush=True)
        print("=" * 72, flush=True)


def ui_language():
    try:
        import ctypes
        lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        return "ja" if (lid & 0x3FF) == 0x11 else "en"
    except Exception:
        return "en"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "yasakani-quickstart/" + QS_VERSION})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while True:
            b = r.read(1 << 18)
            if not b:
                break
            f.write(b)
            done += len(b)
            pct = int(done * 100 / total) if total else -1
            if pct != shown and (pct % 5 == 0 or pct < 0):
                shown = pct
                if total:
                    print("  %3d%%  %.1f / %.1f MB" % (pct, done / 1e6, total / 1e6), flush=True)
    os.replace(tmp, dst)


def safe_extract(zip_path, dst):
    root = os.path.abspath(dst)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            p = os.path.abspath(os.path.join(root, m.filename))
            if not (p == root or p.startswith(root + os.sep)):
                raise RuntimeError("unsafe path in zip: " + m.filename)
        z.extractall(root)


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_STATE_LOCK = threading.Lock()          # the page's "check for updates" saves from a thread of its own


def save_state(path, st):
    """Writes the state file through a temporary file.  One writer at a time (two threads sharing the temporary file could make
    os.replace fail on Windows), and a short wait when another program has the file open at that moment (Windows refuses to
    replace an open file: a reader, a virus scanner)."""
    with _STATE_LOCK:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(st), f, indent=1, sort_keys=True)
        for left in range(39, -1, -1):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if not left:
                    raise
                time.sleep(0.05)


def ask(prompt, default="", eof=None):
    """One typed line; Enter alone is ``default``.  At the end of the input (no keyboard: the input is closed, or a file that has
    ended) it is ``eof`` when that is given, else ``default``."""
    try:
        s = input(prompt).strip()
    except EOFError:
        return default if eof is None else eof
    return s or default


def menu_prompt(ui):
    m = ui.t("menu", d=MENU_DEFAULT)
    i = m.rfind("\n")                                   # the update item goes after the last item, above the prompt line
    return m[:i + 1] + ui.t("menu_update") + m[i + 1:]


def menu_choice(ui):
    """Shows the menu.  -> "web", "bench", "compare", "chat", "serve", "update", or None for quit.  Enter alone is the first item;
    the end of the input (no keyboard) is quit: otherwise the page and the server would be started again and again."""
    return MENU.get(ask(menu_prompt(ui), MENU_DEFAULT, eof="6"))


def flush_keys():
    """Drops the keys that wait in the console.  Before the page's wait starts: a key typed during the steps before must not end the
    page at once.  Before the menu comes back after the page: Enter alone is the first item of the menu, so a second Enter left over
    would open the page again at once.  It is not called where this script ends, so keys typed for the next program stay."""
    try:
        import msvcrt
    except ImportError:
        return
    if sys.stdin is not None and sys.stdin.isatty():
        while msvcrt.kbhit():
            msvcrt.getwch()


def direct_start(action, model, checked, model_dir=None):
    """--action auto: True when the launcher opens the page in the browser at once, without the model question and the menu.  That
    is when the chosen model passed the checks on this computer in an earlier run (``checked`` is models_ok of the saved state, read
    before this run changes it) and no --model-dir is given.  Otherwise auto behaves as the menu: the first run, a model that was
    never checked, a model folder of the user's own."""
    return action == "auto" and not model_dir and model in MODELS and bool((checked or {}).get(model))


DOWNLOAD_CODE = ("import sys\nfrom huggingface_hub import snapshot_download\n"
                 "kw = {\"revision\": sys.argv[3]} if len(sys.argv) > 3 else {}\n"
                 "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], **kw)\n")


def download_command(vpy, model, mdir):
    """The command that downloads ``model`` from Hugging Face into ``mdir``, with the Python of the private environment.  A model
    with a pinned revision (Inkling-2 Mini) gets it as a third argument; the Qwen models download the main branch, as before."""
    m = MODELS[model]
    return [vpy, "-c", DOWNLOAD_CODE, m["repo"], mdir] + ([m["revision"]] if m.get("revision") else [])


def copy_model_files(src, dst):
    """Tests only (TEST_MODEL_SRC): the regular files at the top of the folder ``src`` -> ``dst``, as a hard link where the drive
    allows it, else as a copy.  -> the number of files; raises OSError."""
    os.makedirs(dst, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name.startswith(".") or not os.path.isfile(s):
            continue
        if os.path.lexists(d):
            os.remove(d)
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
        n += 1
    if not n:
        raise OSError("no files in " + src)
    return n


def fetch_model(ui, vpy, env, model, mdir, call=subprocess.call, environ=None):
    """Step 5: downloads ``model`` into ``mdir`` (once more without Xet when the first try fails).  -> 0, or the exit code of the
    download.  Tests only: when the environment has TEST_MODEL_SRC + the model's name in capitals, the files of that folder are
    linked or copied instead and Hugging Face is not asked."""
    var = TEST_MODEL_SRC + model.upper()
    src = (os.environ if environ is None else environ).get(var)
    if src:
        ui.say("model_test_src", p=src, v=var)
        try:
            copy_model_files(src, mdir)
        except OSError as e:
            print("%s: %s" % (type(e).__name__, e), flush=True)
            return 1
        return 0
    m = MODELS[model]
    ui.say("model_get", r=m["repo"], g=m.get("get_gb") or m["need_gb"] - 1)
    os.makedirs(mdir, exist_ok=True)
    cmd = download_command(vpy, model, mdir)
    rc = call(cmd, env=env)
    if rc != 0:
        ui.say("retry_dl")
        env2 = dict(env)
        env2["HF_HUB_DISABLE_XET"] = "1"
        rc = call(cmd, env=env2)
    return rc


def is_inkling2_dir(mdir):
    """True when the model folder is an Inkling-2 folder of the package ("architecture": "inkling2" in its yasakani.json)."""
    try:
        with open(os.path.join(mdir, "yasakani.json"), encoding="utf-8") as f:
            return json.load(f).get("architecture") == "inkling2"
    except (OSError, ValueError, AttributeError):
        return False


def read_manifest(mdir):
    """yasakani.json of ``mdir`` as a dict ({} when it is missing or not a JSON object)."""
    try:
        with open(os.path.join(mdir, "yasakani.json"), encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def ink2_has_image(mdir):
    """True when the folder's yasakani.json names an image file ("vision") and that file is there with its size."""
    v = read_manifest(mdir).get("vision")
    if not isinstance(v, dict) or not isinstance(v.get("file"), str) or os.path.basename(v["file"]) != v["file"]:
        return False
    try:
        return os.path.getsize(os.path.join(mdir, v["file"])) == v.get("size")
    except OSError:
        return False


IMAGE_CODE = ("import sys\nfrom huggingface_hub import hf_hub_download\n"
              "for name in sys.argv[4:]:\n"
              "    hf_hub_download(repo_id=sys.argv[1], filename=name, revision=sys.argv[3], local_dir=sys.argv[2])\n")


def image_files_command(vpy, image, ddir):
    """The command that downloads the files of ``image`` (INK2_IMAGE) at its revision into ``ddir``."""
    return [vpy, "-c", IMAGE_CODE, MODELS["inkling2"]["repo"], ddir, image["revision"]] + sorted(image["files"])


def check_image_files(ddir, image, mdir):
    """The files of ``image`` in ``ddir``: each has the pinned size and SHA-256, the new yasakani.json names the image file with
    its pinned values, and its text weights and drafter are those of the folder ``mdir`` (compared in the two yasakani.json files;
    the weights are not read).  -> None, or what is wrong."""
    for name, (size, sha) in sorted(image["files"].items()):
        p = os.path.join(ddir, name)
        if not os.path.isfile(p):
            return "%s: missing" % name
        if os.path.getsize(p) != size:
            return "%s: %d bytes, expected %d" % (name, os.path.getsize(p), size)
        got = sha256_of(p)
        if got != sha:
            return "%s: SHA-256 %s, expected %s" % (name, got, sha)
    new, old = read_manifest(ddir), read_manifest(mdir)
    v, (size, sha) = new.get("vision") or {}, image["files"][INK2_IMAGE_FILE]
    if not isinstance(v, dict) or (v.get("file"), v.get("size"), v.get("sha256")) != (INK2_IMAGE_FILE, size, sha):
        return "yasakani.json: the image file is not listed as expected"
    for key in ("weights", "drafter"):
        a, b = new.get(key) or {}, old.get(key) or {}
        if not isinstance(a, dict) or not isinstance(b, dict) or not a.get("sha256") \
                or (a.get("file"), a.get("size"), a.get("sha256")) != (b.get("file"), b.get("size"), b.get("sha256")):
            return "yasakani.json: the %s differ from this folder's" % key
    return None


def fetch_image_files(ui, vpy, env, image, ddir, call=subprocess.call, environ=None):
    """Downloads the files of ``image`` into the empty folder ``ddir`` (once more without Xet when the first try fails).  -> 0, or
    the exit code.  Tests only: TEST_MODEL_SRC + "INKLING2" names a folder the files are copied from (Hugging Face is not asked)."""
    var = TEST_MODEL_SRC + "INKLING2"
    src = (os.environ if environ is None else environ).get(var)
    os.makedirs(ddir, exist_ok=True)
    if src:
        ui.say("model_test_src", p=src, v=var)
        try:
            for name in sorted(image["files"]):
                shutil.copyfile(os.path.join(src, name), os.path.join(ddir, name))
        except OSError as e:
            print("%s: %s" % (type(e).__name__, e), flush=True)
            return 1
        return 0
    cmd = image_files_command(vpy, image, ddir)
    rc = call(cmd, env=env)
    if rc != 0:
        ui.say("retry_dl")
        env2 = dict(env)
        env2["HF_HUB_DISABLE_XET"] = "1"
        rc = call(cmd, env=env2)
    return rc


def install_image_files(ddir, image, mdir):
    """Moves the checked files of ``image`` from ``ddir`` into ``mdir``: yasakani.json last, so that the folder names the image
    file only once it is there.  The download notes of Hugging Face for the replaced files are removed (a later download of the
    folder then fetches only those again)."""
    names = sorted(image["files"], key=lambda n: n == "yasakani.json")
    for name in names:
        os.replace(os.path.join(ddir, name), os.path.join(mdir, name))
        note = os.path.join(mdir, ".cache", "huggingface", "download", name + ".metadata")
        if os.path.isfile(note):
            os.remove(note)


def offer_image_part(ui, st, save, vpy, env, home, mdir, ask_now, forced, image=None, call=subprocess.call, environ=None):
    """Step 5 for an Inkling-2 Mini folder without the image file while the switch is on: asks once whether to get it (``forced``:
    START.cmd --image-file, no question).  A no is kept in the state: later starts say one line.  When the question cannot be
    asked (``ask_now`` False, or no keyboard) the line is said and nothing is kept.  Every failure leaves the folder as it was
    (text only).  -> True when the image file was put in place."""
    image = INK2_IMAGE if image is None else image
    if not image or ink2_has_image(mdir) or not is_inkling2_dir(mdir):
        return False
    size = image["files"][INK2_IMAGE_FILE][0]
    gb = "%.2f" % (sum(s for s, _ in image["files"].values()) / 1e9)
    if not forced:
        if not ask_now or st.get("ink2_image_no") == image["revision"]:
            ui.say("img_later", g=gb)
            return False
        ans = ask(ui.t("img_ask", g=gb), "n", eof="")
        if not ans:
            ui.say("img_later", g=gb)
            return False
        if ans.lower() not in ("y", "yes"):
            st["ink2_image_no"] = image["revision"]
            save()
            ui.say("img_later", g=gb)
            return False
    st.pop("ink2_image_no", None)
    save()
    if shutil.disk_usage(home).free < size + (1 << 30):
        ui.say("img_disk", g=gb)
        return False
    ddir = os.path.join(home, "downloads", "inkling2_image")
    if os.path.isdir(ddir):
        shutil.rmtree(ddir, ignore_errors=True)
    try:
        ui.say("img_get", g=gb)
        rc = fetch_image_files(ui, vpy, env, image, ddir, call=call, environ=environ)
        if rc != 0:
            ui.say("img_fail", e="exit code %d" % rc)
            return False
        bad = check_image_files(ddir, image, mdir)
        if bad:
            ui.say("img_bad", e=bad)
            return False
        install_image_files(ddir, image, mdir)
    except OSError as e:
        ui.say("img_fail", e="%s: %s" % (type(e).__name__, e))
        return False
    finally:
        shutil.rmtree(ddir, ignore_errors=True)
    ui.say("img_done")
    return True


def installed_models(home, downloaded, quality_fits, isfile=os.path.isfile):
    """The models the page can switch between: those the quickstart downloaded into this home (models_downloaded of the state, and
    the folder is still there) that fit this GPU, in the order of the model question."""
    return [name for name in MODEL_ORDER
            if (downloaded or {}).get(name) and (quality_fits or name == "speed")
            and isfile(os.path.join(home, "models", name, "yasakani.json"))]


def parse_gpus(text, fields=GPU_FIELDS):
    """Lines of ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` -> dicts (index, uuid, name, cc, total, used,
    driver).  Values nvidia-smi could not give are None; lines without one value per field are skipped."""
    def val(s):
        s = s.strip()
        return None if (not s or s.startswith("[") or s.upper() == "N/A") else s

    def num(s):
        try:
            return int(round(float(s))) if s is not None else None
        except ValueError:
            return None
    out = []
    for n, line in enumerate(ln for ln in text.splitlines() if ln.strip()):
        parts = line.split(",")
        if len(parts) != len(fields):
            continue
        v = {k: val(p) for k, p in zip(fields, parts)}
        cc = None
        if v.get("compute_cap"):
            try:
                major, minor = v["compute_cap"].split(".")
                cc = (int(major), int(minor))
            except ValueError:
                cc = None
        index = num(v.get("index")) if "index" in v else n
        if not v.get("name") or index is None:
            continue
        out.append(dict(index=index, uuid=v.get("uuid"), name=v["name"], cc=cc, total=num(v.get("memory.total")),
                        used=num(v.get("memory.used")), driver=v.get("driver_version")))
    return out


def query_gpus(run=subprocess.run):
    """Every GPU nvidia-smi lists ([] for none).  Raises OSError when nvidia-smi cannot be run.  Drivers that do not know
    compute_cap are asked without it."""
    fallback = tuple(f for f in GPU_FIELDS if f != "compute_cap")
    for fields in (GPU_FIELDS, fallback):
        out = run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            return parse_gpus(out.stdout or "", fields)
    return []


def gpu_verdict(g):
    """"tested" (the RTX 5090), "other" (runs, not optimized or tested), "small" (under 16 GB), "old" (compute capability below 8.6)
    or "nocode" (9.x to 11.x, 13 and newer: the engine has no code for it)."""
    cc = g.get("cc")
    if cc is not None and cc < (8, 6):
        return "old"
    if cc is not None and not (cc < (9, 0) or cc[0] == 12):
        return "nocode"
    if re.search(r"(^|\s)RTX 5090$", g["name"].strip()) and cc in (None, (12, 0)):
        return "tested"
    if g.get("total") is not None and g["total"] < MIN_TOTAL_MIB:
        return "small"
    return "other"


def pick_gpu(gpus, environ=None):
    """The GPU to use: the one CUDA_VISIBLE_DEVICES names first, else the best usable one (the RTX 5090 first, then the one with the
    most memory).  -> (gpu, verdict, pin): ``pin`` is the GPU's UUID to put in CUDA_VISIBLE_DEVICES when the PC has more than one
    GPU, so that the engine runs on the GPU checked here (None: nothing to set)."""
    env = os.environ if environ is None else environ
    rank = {"tested": 0, "other": 1, "small": 2, "nocode": 3, "old": 4}
    if "CUDA_VISIBLE_DEVICES" in env:
        first = (env.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
        pci = env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID").strip().upper() == "PCI_BUS_ID"   # the package sets it when not set (0.1.5)
        named = [g for g in gpus if (first.upper().startswith("GPU-") and (g.get("uuid") or "").lower().startswith(first.lower()))
                 or (first.isdigit() and (len(gpus) == 1 or pci) and g["index"] == int(first))]
        if len(named) == 1:
            return named[0], gpu_verdict(named[0]), None
        worst = max(gpus, key=lambda g: (rank[gpu_verdict(g)], -(g.get("total") or 0)))   # not known which: the least able one
        return worst, gpu_verdict(worst), None
    best = min(gpus, key=lambda g: (rank[gpu_verdict(g)], -(g.get("total") or 0), -((g.get("total") or 0) - (g.get("used") or 0))))
    pin = best.get("uuid") if len(gpus) > 1 else None
    return best, gpu_verdict(best), pin


def estimate_context(free_mib, model):
    """The context the package will choose (yasakani/gpu.py, choose()): (tokens, head_off); tokens 0 = it does not fit."""
    w, d, h = MODEL_BYTES[model]
    mib = float(1 << 20)

    def ctx(files):
        room = free_mib - files / mib - OVERHEAD_MIB[model] - MARGIN_MIB
        n = max(0, int(room * mib // KV_BYTES_PER_TOKEN))
        return FULL_CONTEXT if n >= FULL_CONTEXT else (n // CONTEXT_STEP) * CONTEXT_STEP

    with_head = ctx(w + d + h)
    if with_head >= HEAD_OFF_BELOW:
        return with_head, False
    without = ctx(w + d)
    return (without, True) if without > with_head else (with_head, False)


# A copy of FAST_AUTO_CAPS of the package (python/yasakani/inkling2.py): auto turns the fast path on for RTX 30 (8.6), RTX 40 (8.9)
# and RTX 50 (12.0) cards (0.3.4; 12.0 only in 0.3.3).  Keep the two equal (tests/test_quickstart_checks_ctx.py compares them).
INK2_FAST_AUTO_CAPS = ((8, 6), (8, 9), (12, 0))


def ink2_fast_expected(cc, environ=None):
    """Does the package expect the fast path for a GPU of compute capability ``cc`` ((major, minor) or None)?  As inkling2.py
    fast_expected(): YASAKANI_INK2_FAST on / off, else auto = on for INK2_FAST_AUTO_CAPS (unknown: off)."""
    env = os.environ if environ is None else environ
    mode = (env.get(INK2_FAST_ENV) or "auto").strip().lower()
    return mode == "on" or (mode == "auto" and tuple(cc or ()) in INK2_FAST_AUTO_CAPS)


def ink2_vision_mib():
    """GPU memory (MiB) of the image part with the defaults of chat and serve (yasakani/gpu.py ink2_vision_mib)."""
    n = INK2_VISION_WEIGHTS_BYTES + INK2_VISION_SCRATCH_BYTES + INK2_VISION_SCRATCH_BYTES_PER_TOKEN * INK2_VISION_IMAGE_TOKENS
    n += INK2_VISION_FEATURE_BYTES_PER_ROW * INK2_VISION_REQUEST_FACTOR * INK2_VISION_IMAGE_TOKENS
    return n / float(1 << 20)


def ink2_vision_planned(mdir, image=None, environ=None, engine=None):
    """Will chat and serve load the image part of the Inkling-2 Mini folder ``mdir``?  When the folder has it (ink2_has_image), or
    when the folder is not there yet and the switch ``image`` (INK2_IMAGE) is on (a new download brings it); never when
    YASAKANI_INK2_VISION is off, and with auto (the default) only when the engine library reads images (``engine``; None =
    ENGINE_IMAGE, the engine zip the launcher installs), as yasakani/inkling2.py vision_plan_mib()."""
    env = os.environ if environ is None else environ
    mode = (env.get(INK2_VISION_ENV) or "auto").strip().lower()
    if mode == "off" or (mode != "on" and not (ENGINE_IMAGE if engine is None else engine)):
        return False
    return ink2_has_image(mdir) or bool(image and not os.path.isfile(os.path.join(mdir, "yasakani.json")))


def estimate_ink2_context(free_mib, fast=False, vision=False):
    """The context the package will choose for the Inkling-2 Mini folder (yasakani/gpu.py, ink2_context()); 0 = it does not fit.
    ``fast``: planned for the fast path (chat and serve on an RTX 30, 40 or 50 card); without it the standard path, which
    `yasakani selfcheck` plans with since 0.3.3.  ``vision``: the image part is loaded too (ink2_vision_planned)."""
    over = INK2_OVERHEAD_MIB + (INK2_FAST_OVERHEAD_MIB if fast else 0)
    room = free_mib - sum(INK2_FILES_BYTES) / float(1 << 20) - over - INK2_ROLLBACK_MIB - MARGIN_MIB
    if vision:
        room -= ink2_vision_mib()
    n = max(0, int(room * (1 << 20) // (INK2_FAST_BYTES_PER_POSITION if fast else INK2_BYTES_PER_POSITION)))
    if n >= INK2_DEFAULT_CONTEXT:
        return INK2_DEFAULT_CONTEXT
    n = (n // INK2_CONTEXT_STEP) * INK2_CONTEXT_STEP
    return n if n >= INK2_MIN_CONTEXT else 0


def driver_major(text):
    """616 from nvidia-smi's driver_version "616.92" (580 from "580.65.06"); None when there is no usable value."""
    m = re.fullmatch(r"\s*(\d+)(?:\.\d+)*\s*", text or "")
    return int(m.group(1)) if m else None


def cuda_driver_version(load=None):
    """The newest CUDA version the installed driver supports, from cuDriverGetVersion() of nvcuda.dll (13040 = CUDA 13.4); None when it
    cannot be read.  This only asks the driver library for a number: nothing is started on the GPU.  ``load`` (for the tests) returns
    the library object."""
    try:
        import ctypes
        if load is None:
            lib = ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32", "nvcuda.dll"))
        else:
            lib = load()
        v = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(v)) != 0 or v.value <= 0:
            return None
        return int(v.value)
    except Exception:  # noqa: BLE001 - no driver library, not Windows, ...: not known
        return None


def driver_label(driver, cuda, lang="en"):
    """What the messages show: the driver version and the newest CUDA version the driver supports (not the engine's), e.g.
    "616.92, supports up to CUDA 13.4" / "616.92、CUDA 13.4 まで対応", "616.92", "supports up to CUDA 12.8" or "?"."""
    parts = [driver] if driver else []
    if cuda:
        v = "%d.%d" % (cuda // 1000, (cuda % 1000) // 10)
        parts.append(("CUDA %s まで対応" if lang == "ja" else "supports up to CUDA %s") % v)
    return ("、" if lang == "ja" else ", ").join(parts) or "?"


def driver_verdict(driver, cuda=None):
    """"ok", "old" (older than the R580 branch, or a driver for CUDA 12 or older: the engine cannot start on it) or "unknown" (neither
    nvidia-smi's driver_version nor cuDriverGetVersion() gave a number).  Each number that is known must be new enough."""
    checks = []
    major = driver_major(driver)
    if major is not None:
        checks.append(major >= MIN_DRIVER)
    if cuda:
        checks.append(cuda >= MIN_CUDA_DRIVER_API)
    if not checks:
        return "unknown"
    return "ok" if all(checks) else "old"


# ------------------------------------------------------------------------------------------------------------------------ updates
class UpdateError(Exception):
    pass


class UpdateSource:
    """Where updates come from: GitHub's API for the latest release and this repository's download URLs.  ``dev_base`` (development
    only, --update-dev-base) points both at a server on this computer, e.g. http://127.0.0.1:8900, for testing an update end to end."""

    def __init__(self, dev_base=None):
        self.dev = bool(dev_base)
        if dev_base:
            m = re.fullmatch(r"http://(127\.0\.0\.1|localhost|\[::1\]):(\d{1,5})/?", dev_base)
            if not m:
                raise UpdateError("--update-dev-base must be http://127.0.0.1:<port> (a server on this computer)")
            base = "http://%s:%s/" % (m.group(1), m.group(2))
            self.api = base + "repos/yasakani-dev/yasakani/releases/latest"
            self.download = base + "yasakani-dev/yasakani/releases/download/"
            self.base = base.rstrip("/")
        else:
            self.api, self.download, self.base = UPDATE_API, UPDATE_DOWNLOAD, None


def parse_version(s):
    """(0, 1, 10) from "0.1.10" or "v0.1.10"; None for anything else (pre-release tags such as v0.1.7-test1 included)."""
    m = re.fullmatch(r"v?(\d{1,4})\.(\d{1,4})\.(\d{1,4})", str(s or "").strip())
    return tuple(int(x) for x in m.groups()) if m else None


def is_newer(new, old):
    a, b = parse_version(new), parse_version(old)
    return a is not None and b is not None and a > b


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Release downloads redirect to GitHub's file host; follow https redirects only (http only to this computer in development)."""

    def __init__(self, dev):
        self.dev = dev

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not (newurl.startswith("https://") or (self.dev and re.match(r"http://(127\.0\.0\.1|localhost|\[::1\]):\d+/", newurl))):
            raise UpdateError("refused a redirect to " + newurl.split("?", 1)[0])
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_bytes(url, limit, timeout, dev=False):
    """GET ``url`` with a generic User-Agent and nothing else about this computer; at most ``limit`` bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": UPDATE_UA, "Accept": "application/vnd.github+json, */*"})
    opener = urllib.request.build_opener(_Redirects(dev))
    with opener.open(req, timeout=timeout) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise UpdateError("the answer is larger than %d bytes" % limit)
    return data


def fetch_latest(src, timeout=5):
    """The latest release: {"version", "tag", "page", "assets"}; raises on any problem.  Drafts and pre-releases are refused."""
    try:
        j = json.loads(fetch_bytes(src.api, MAX_API, timeout, src.dev).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise UpdateError("the answer of the release API is not JSON")
    if not isinstance(j, dict):
        raise UpdateError("the answer of the release API is not a release")
    tag = j.get("tag_name")
    if j.get("draft") or j.get("prerelease"):
        raise UpdateError("the latest release is a pre-release")
    v = parse_version(tag)
    if not (isinstance(tag, str) and tag.startswith("v") and v):
        raise UpdateError("the latest release has an unexpected tag")
    assets = {}
    for x in j.get("assets") or []:
        if isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("browser_download_url"), str):
            assets[x["name"]] = x["browser_download_url"]
    version = "%d.%d.%d" % v
    return {"version": version, "tag": tag, "page": REPO_URL + "/releases/tag/" + tag, "assets": assets}


def release_url(src, rel, name):
    """The download URL of ``name`` in that release.  Only URLs under this repository's download path of that tag are used."""
    base = src.download + rel["tag"] + "/"
    url = (rel.get("assets") or {}).get(name) or base + name
    if url != base + name:
        raise UpdateError("refused a download URL outside %s: %s" % (base, url))
    return url


def parse_sums(text):
    out = {}
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1).lower()
    return out


def check_start_zip(data, version):
    """The members of a quickstart zip -> {name: bytes}.  Refused: anything but plain files at the top whose names are START_FILES
    (no folders, no paths), duplicates, encrypted or too large members, a zip without quickstart.py and START.cmd, or a quickstart.py
    of another version."""
    if len(data) > MAX_START_ZIP:
        raise UpdateError("the zip is too large")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UpdateError("not a zip file")
    out = {}
    with z:
        infos = z.infolist()
        if len(infos) > len(START_FILES):
            raise UpdateError("the zip has too many members")
        for m in infos:
            if m.filename not in START_FILES or m.filename in out:
                raise UpdateError("unexpected member in the zip: %r" % m.filename)
            if m.is_dir() or m.flag_bits & 0x1 or m.file_size > MAX_START_FILE:
                raise UpdateError("member not allowed: %r" % m.filename)
            with z.open(m) as f:
                b = f.read(MAX_START_FILE + 1)
            if len(b) > MAX_START_FILE or len(b) != m.file_size:
                raise UpdateError("member not allowed: %r" % m.filename)
            out[m.filename] = b
    for need in ("quickstart.py", "START.cmd"):
        if need not in out:
            raise UpdateError("the zip has no " + need)
    if ('QS_VERSION = "%s"' % version).encode("ascii") not in out["quickstart.py"]:
        raise UpdateError("quickstart.py in the zip is not version " + version)
    return out


def download_update(ui, src, rel):
    """Downloads and checks the start files of ``rel`` -> {name: bytes}.  Nothing on disk changes."""
    zname = "yasakani-quickstart-%s.zip" % rel["version"]
    zurl, surl = release_url(src, rel, zname), release_url(src, rel, "SHA256SUMS.txt")
    ui.say("upd_get", u=surl)
    sums = parse_sums(fetch_bytes(surl, MAX_SUMS, 30, src.dev).decode("ascii", "replace"))
    if zname not in sums:
        raise UpdateError("SHA256SUMS.txt of %s does not list %s" % (rel["tag"], zname))
    ui.say("upd_get", u=zurl)
    data = fetch_bytes(zurl, MAX_START_ZIP, 60, src.dev)
    got = hashlib.sha256(data).hexdigest()
    if got != sums[zname]:
        raise UpdateError("the SHA-256 of %s differs from SHA256SUMS.txt (expected %s, got %s)" % (zname, sums[zname], got))
    ui.say("upd_sha_ok")
    return check_start_zip(data, rel["version"])


def _write_files(folder, files):
    """Writes every file as <name>.new first, then renames them over the old ones.  START.cmd may be the batch file that is running:
    START.cmd runs this script from inside one ( ... ) block, which cmd.exe has read completely before it runs it, so a new START.cmd
    does not change what the running one does after this script ends."""
    for n, b in files.items():
        with open(os.path.join(folder, n + ".new"), "wb") as f:
            f.write(b)
    for n in files:
        os.replace(os.path.join(folder, n + ".new"), os.path.join(folder, n))


def install_start_files(folder, files, old_version=None, before_replace=None):
    """Copies the current start files to backup/<old_version>/ and puts ``files`` in their place.  -> the backup folder.  If a file
    cannot be replaced, the files of the backup are put back.  ``before_replace`` is called once the backup is complete and before
    the first file is replaced (apply_update saves the pointer to the backup there, so --action rollback works after an
    interrupted update)."""
    old_version = old_version or QS_VERSION
    if not parse_version(old_version):
        raise UpdateError("bad version " + old_version)
    bdir = os.path.join(folder, "backup", old_version)
    tmp = bdir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    kept = []
    for n in START_FILES:
        p = os.path.join(folder, n)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(tmp, n))
            kept.append(n)
    with open(os.path.join(tmp, "backup.json"), "w", encoding="utf-8") as f:
        json.dump({"version": old_version, "files": kept, "made": int(time.time())}, f, indent=1)
    if os.path.isdir(bdir):
        shutil.rmtree(bdir)
    os.replace(tmp, bdir)
    if before_replace is not None:
        before_replace(bdir)
    try:
        _write_files(folder, files)
    except OSError as e:
        restore = {}
        for n in kept:
            with open(os.path.join(bdir, n), "rb") as f:
                restore[n] = f.read()
        _write_files(folder, restore)
        e.restored = True                               # the old start files are back in place
        raise
    return bdir


def read_backup(folder, version):
    """The files of backup/<version>/ -> {name: bytes}; raises when there is no usable backup."""
    if not parse_version(version):
        raise UpdateError("bad version")
    bdir = os.path.join(folder, "backup", version)
    try:
        with open(os.path.join(bdir, "backup.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        raise UpdateError("no backup of " + version)
    out = {}
    for n in man.get("files") or []:
        if n not in START_FILES:
            raise UpdateError("unexpected file in the backup: %r" % n)
        p = os.path.join(bdir, n)
        if os.path.getsize(p) > MAX_START_FILE:
            raise UpdateError("file too large in the backup: %r" % n)
        with open(p, "rb") as f:
            out[n] = f.read()
    if "quickstart.py" not in out or "START.cmd" not in out:
        raise UpdateError("the backup of %s is not complete" % version)
    return out


def check_disabled(a, st, environ=None):
    env = os.environ if environ is None else environ
    return bool(getattr(a, "no_update_check", False)) or env.get("YASAKANI_NO_UPDATE_CHECK", "") not in ("", "0") \
        or st.get("update_check") is False


def update_check(ui, st, save, src, force=False, now=None, disabled=False, result=None):
    """The check of the menu (at most once a day) or, with ``force``, of the menu item, --action update and the page's "check for
    updates".  -> the newer release or None.  Never raises: a failed check prints one line and goes on.  ``result`` (a dict), when
    given, gets "state" ("newer", "latest", "skipped" or "failed") and for a failure "message", for a check that asked."""
    result = {} if result is None else result
    now = time.time() if now is None else now
    if disabled and not force:
        return None
    last = st.get("update_checked") or 0
    if not force and 0 <= now - last < CHECK_EVERY:
        c = st.get("update_latest")
        if isinstance(c, dict) and is_newer(c.get("version"), QS_VERSION) and c.get("version") != st.get("update_skip") \
                and parse_version(c.get("tag")):
            return {"version": c["version"], "tag": c["tag"], "page": REPO_URL + "/releases/tag/" + c["tag"], "assets": {}}
        return None
    print(ui.t("upd_line"), flush=True)
    st["update_checked"] = now
    try:
        rel = fetch_latest(src)
    except Exception as e:  # noqa: BLE001 - offline, timeout, refused, rate limit, ...: never stop the launcher for it
        if hasattr(e, "close"):
            e.close()                                   # an HTTPError holds the answer
        save()
        result.update(state="failed", message=str(e) if isinstance(e, UpdateError) else type(e).__name__)
        ui.say("upd_fail", e=result["message"])
        return None
    st["update_latest"] = {"version": rel["version"], "tag": rel["tag"]}
    save()
    if not is_newer(rel["version"], QS_VERSION) or (not force and rel["version"] == st.get("update_skip")):
        result["state"] = "latest" if not is_newer(rel["version"], QS_VERSION) else "skipped"
        if force or not is_newer(rel["version"], QS_VERSION):
            ui.say("upd_latest", v=QS_VERSION)
        return None
    result["state"] = "newer"
    return rel


def run_quickstart(folder, args):
    """Runs the quickstart.py of ``folder`` (the new one after an update) with the system Python that runs this one."""
    print(flush=True)
    return subprocess.call([sys.executable, os.path.join(folder, "quickstart.py")] + list(args))


def apply_update(ui, folder, rel, src, yes, after_args, st, save):
    """Asks (unless ``yes``), downloads, checks, backs up, replaces, then runs the new quickstart.py.  -> its exit code, 9 when the
    update failed (nothing changed), None when the answer was no."""
    ui.say("upd_found", o=QS_VERSION, n=rel["version"], u=rel["page"])
    if not yes and ask(ui.t("upd_ask"), "n").lower() not in ("y", "yes"):
        ui.say("upd_later")
        return None
    had, before = "update_backup" in st, st.get("update_backup")

    def keep_pointer(bdir):
        # saved before the first start file is replaced: if the update stops halfway, --action rollback finds the backup
        st["update_backup"] = {"version": QS_VERSION, "from": rel["version"]}
        save()

    try:
        files = download_update(ui, src, rel)
        bdir = install_start_files(folder, files, before_replace=keep_pointer)
    except Exception as e:  # noqa: BLE001 - network, checksum, zip, disk: the current version stays
        if st.get("update_backup") != before and getattr(e, "restored", False):   # the old files are back: keep the earlier pointer
            if had:
                st["update_backup"] = before
            else:
                st.pop("update_backup", None)
            save()
        ui.say("upd_bad", e=str(e) if isinstance(e, UpdateError) else "%s: %s" % (type(e).__name__, e))
        return 9
    st.pop("update_skip", None)
    save()
    ui.say("upd_backup", p=bdir)
    ui.say("upd_done", n=rel["version"])
    return run_quickstart(folder, after_args)


def update_answered(a):
    """--yes answers the update question only together with --action update.  The daily check at the menu and the menu item
    always ask, also when the launcher runs with --yes (for example after an update, which passes --yes on)."""
    return bool(getattr(a, "yes", False)) and getattr(a, "action", None) == "update"


def apply_rollback(ui, folder, st, save, yes, after_args):
    b = st.get("update_backup") or {}
    try:
        files = read_backup(folder, b.get("version", ""))
    except UpdateError:
        ui.say("rb_none")
        return 1
    if not yes and ask(ui.t("rb_ask", o=b["version"]), "n").lower() not in ("y", "yes"):
        ui.say("stopped")
        return 1
    _write_files(folder, files)
    st.pop("update_backup", None)
    if b.get("from"):
        st["update_skip"] = b["from"]                 # the daily check does not offer that version again (the menu item still does)
    m = st.get("model")
    if isinstance(m, str) and ("\"m_%s\"" % m).encode("utf-8") not in files["quickstart.py"]:
        # the launcher that came back has no such model (it would stop on the name): forget the choice and let the user start it
        st.pop("model", None)
        save()
        ui.say("rb_done_model", o=b["version"], m=ui.t("m_" + m) if ("m_" + m) in T else m)
        return 0
    save()
    ui.say("rb_done", o=b["version"])
    return run_quickstart(folder, after_args)


# ---------------------------------------------------------------------------------------------------------------- browser chat
def port_is_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def pick_port(preferred, avoid=()):
    for p in [preferred] + list(range(preferred + 10, preferred + 60)):
        if p not in avoid and port_is_free(p):
            return p
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class LocalServer(ThreadingHTTPServer):
    """Listens on 127.0.0.1 only and never shares its port.  It remembers the connections it accepted: browsers keep theirs open,
    and a thread of a stopped server would go on answering the page on such a connection.  close_connections() ends them (after a
    model switch the next server listens on the same port, and the page must reach that one)."""
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, *args, **kw):
        self._conns, self._conns_lock, self._closing = set(), threading.Lock(), False
        super().__init__(*args, **kw)

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def get_request(self):
        request, addr = super().get_request()
        with self._conns_lock:
            self._conns.add(request)
        return request, addr

    def shutdown_request(self, request):
        with self._conns_lock:
            self._conns.discard(request)
        super().shutdown_request(request)

    def close_connections(self):
        """Ends every connection that is still open (call it after shutdown() and server_close()).  The thread of each one sees the
        end of its connection and finishes."""
        self._closing = True
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass                                    # the other side closed it first

    def handle_error(self, request, client_address):
        # a connection that broke (the browser went away, or close_connections() above) is not worth a traceback in the console
        if self._closing or isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class UpdateRequest:
    """The update button of the chat page.  It exists only when this launcher found a newer version.  The page gets the
    one-time ``token`` from /webchat/config (which only pages of this address can read) and sends it in the X-Yasakani-Token header
    of a POST to /webchat/update; the request must also name this address in Host and Origin.  The token works once.
    Made without ``rel`` it has no button until arm() (the page's "check for updates" found a newer version)."""

    HEADER = "X-Yasakani-Token"

    def __init__(self, rel=None):
        self.rel = rel
        self.token = hashlib.sha256(os.urandom(32)).hexdigest() if rel else None
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._closed = False

    def arm(self, rel):
        """The button of ``rel`` from now on, as if the launcher had known it when the page was opened.  -> False when the
        launcher is leaving the page (close() came first)."""
        with self._lock:
            if self._closed:
                return False
            if self.rel is None:
                self.rel, self.token = rel, hashlib.sha256(os.urandom(32)).hexdigest()
            return True

    def take(self, token):
        with self._lock:
            if not self.token or not token or not hmac.compare_digest(self.token.encode(), token.encode("utf-8", "replace")):
                return False
            self.token = None
            return True

    def close(self):
        """From here on the update button is not taken (the launcher leaves the page for another reason: the quit button, a model
        switch, Enter, the end of the server).  -> False when the token was used or closed before."""
        with self._lock:
            was_open, self.token, self._closed = self.token is not None, None, True
            return was_open


class ControlRequest:
    """The quit button and the model switch of the chat page, made like UpdateRequest.  The page gets the one-time ``token`` from
    /webchat/config and sends it in the X-YASAKANI-Control header of a POST to /webchat/shutdown or /webchat/switch_model; the
    request must also name this address in Host and Origin.  The token works once, for one of the two requests.  After that this
    server of the page takes no control request any more (the server that comes after a model switch has a token of its own)."""

    HEADER = "X-YASAKANI-Control"

    def __init__(self):
        self.token = hashlib.sha256(os.urandom(32)).hexdigest()
        self.event = threading.Event()                  # set when the answer to the accepted request has been written
        self.action, self.model = None, None            # "shutdown", or "switch" and the model's name, once a request was accepted
        self._lock = threading.Lock()

    def take(self, token, action, model=None):
        with self._lock:
            if not self.token or not token or not hmac.compare_digest(self.token.encode(), token.encode("utf-8", "replace")):
                return False
            self.token, self.action, self.model = None, action, model
            return True

    def valid(self, token):
        """The token of the page, without using it up (the page's "check for updates" changes nothing that the token guards)."""
        with self._lock:
            return bool(self.token and token and hmac.compare_digest(self.token.encode(), token.encode("utf-8", "replace")))

    def close(self):
        """From here on no control request is accepted (the launcher leaves the page for another reason, e.g. Enter).  -> False
        when the token was used or closed before."""
        with self._lock:
            was_open, self.token = self.token is not None, None
            return was_open


class UpdateCheckNow:
    """The "check for updates" control of the chat page: POST /webchat/check_update with the page's control token runs ``check``
    (update_check with force=True, as menu item 7: one request to github.com) in a thread of its own, one at a time; a second
    press while it runs gets that check's answer.  A newer version arms ``update`` (an UpdateRequest), so the page's update
    button and POST /webchat/update work as when the launcher knew of it at the start.  The answer is JSON: {"state": "newer",
    "version"}, {"state": "latest", "version"} or {"state": "failed", "message"}; "busy" when the check did not end in ``wait``
    seconds (it goes on and arms the button if it finds a newer version)."""

    def __init__(self, check, update, wait=CHECK_NOW_WAIT):
        self.check, self.update, self.wait = check, update, wait
        self._lock = threading.Lock()
        self._box, self._thread, self._closed = None, None, False

    def _work(self, box):
        result = {}
        try:
            rel = self.check(result)
        except Exception as e:  # noqa: BLE001 - update_check does not raise; this is only a guard for the page's answer
            rel, result = None, {"state": "failed", "message": type(e).__name__}
        if rel and self.update.arm(rel):
            answer = {"state": "newer", "version": rel["version"]}
        elif rel:
            answer = {"state": "failed", "message": "the launcher is leaving the page"}
        elif result.get("state") in ("latest", "skipped"):
            answer = {"state": "latest", "version": QS_VERSION}
        else:
            answer = {"state": "failed", "message": " ".join(str(result.get("message") or "no answer").split())[:160]}
        box["answer"] = answer
        box["done"].set()

    def run(self):
        with self._lock:
            if self._closed:
                return {"state": "failed", "message": "the launcher is leaving the page"}
            if self.update.rel is not None:              # known already: the page's button is there, nothing to ask
                return {"state": "newer", "version": self.update.rel["version"]}
            box = self._box
            if box is None or box["done"].is_set():
                box = self._box = {"done": threading.Event(), "answer": None}
                self._thread = threading.Thread(target=self._work, args=(box,), daemon=True)
                self._thread.start()
        if not box["done"].wait(self.wait):
            return {"state": "busy", "message": "the check is still under way"}
        return dict(box["answer"])

    def close(self, wait=10.0):
        """No more checks; a check under way may end first (it writes the state file)."""
        with self._lock:
            self._closed, t = True, self._thread
        if t is not None:
            t.join(wait)


class WebSession:
    """What the chat page is told about this run of `yasakani serve`: GET /webchat/status, and the part of /webchat/config that
    the engine reports.  watch() asks the engine's /health in a thread of its own until the model is ready, so that neither a
    request of the page nor the Enter key in the console ever waits for the engine."""

    def __init__(self, model, proc=None, engine_port=None, installed=(), on_ready=None, clock=time.monotonic):
        self.model, self.proc, self.engine_port = model, proc, engine_port
        self.installed = tuple(installed)               # the models the page can switch between (empty: no switching)
        self.on_ready, self._clock = on_ready, clock
        self.t0 = clock()                               # `yasakani serve` was started just before
        self.ready = False                              # /health said "ok" once
        self.model_id = (MODELS.get(model) or {}).get("model_id", DEFAULT_MODEL_ID)   # until the engine tells its own
        self.max_context = None
        self.leaving = None                             # "stopping" or "switching": the launcher itself ends this run
        self.phase = None                               # "checking": the checks of a model switch run (open_check_server)
        self.told_failed = threading.Event()            # a page was told "failed"
        self._thread = None

    def elapsed(self):
        return round(max(0.0, self._clock() - self.t0), 1)

    def state(self):
        if self.leaving:
            return self.leaving
        if self.proc is not None and self.proc.poll() is not None:
            return "failed"                             # the server ended and the launcher did not ask for it
        return "ready" if self.ready else "loading"

    def status(self):
        s = {"state": self.state(), "elapsed": self.elapsed(), "model": self.model}
        if self.phase:
            s["phase"] = self.phase                     # only while there is something to add to the state
        return s

    def leave(self, state):
        """The launcher stops `yasakani serve` now ("stopping") or changes the model ("switching").  The first call counts."""
        if self.leaving is None:
            self.leaving = state

    def probe(self):
        """Asks the engine once.  -> True when the model is ready; its context limit and its id are known then."""
        if self.ready:
            return True
        try:
            code, body, _ = get_json(self.engine_port, "/health")
        except (OSError, http.client.HTTPException):
            return False
        if code != 200 or not isinstance(body, dict) or body.get("status") != "ok":
            return False
        ctx = body.get("max_context_tokens")
        self.max_context = ctx if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0 else None
        try:
            code, body, _ = get_json(self.engine_port, "/v1/models")
            data = body.get("data") if code == 200 and isinstance(body, dict) else None
            first = data[0] if isinstance(data, list) and data else None
            # a model folder of the user's own can report any text; the page shows the id and puts it into the samples for other
            # programs (a command line among them), so only a plain name is passed on
            if isinstance(first, dict) and isinstance(first.get("id"), str) and MODEL_ID_OK.fullmatch(first["id"]):
                self.model_id = first["id"]
        except (OSError, http.client.HTTPException):
            pass                                        # the usual id stays
        self.ready = True
        return True

    def watch(self, poll=0.5):
        """Starts the thread that waits for the model (once).  It ends when the model is ready, when the server ended or when the
        launcher leaves."""
        if self._thread is not None or self.engine_port is None:
            return

        def run():
            while not self.leaving and (self.proc is None or self.proc.poll() is None):
                time.sleep(poll)                        # first too: a server that was just started does not listen yet
                if self.probe():
                    if self.on_ready is not None and not self.leaving:
                        self.on_ready(self.elapsed())
                    return

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()


def _smi_quick(cmd, **kw):
    kw["timeout"] = GPU_QUERY_TIMEOUT                   # a request of the page never waits a minute for nvidia-smi
    return subprocess.run(cmd, **kw)


class GpuWatch:
    """The GPU line of the chat page: name, driver and memory of the GPU the engine runs on (the one step 1 picked), read from
    nvidia-smi with the launcher's own parser, at most once in GPU_CACHE_SECONDS.  When nvidia-smi cannot be asked, the facts of
    step 1 stay and the memory in use is None: a request of the page never fails for it."""

    def __init__(self, gpu, query=None, ttl=GPU_CACHE_SECONDS, clock=time.monotonic):
        self.gpu = dict(gpu or {})
        self._query = query or (lambda: query_gpus(run=_smi_quick))
        self._ttl, self._clock = ttl, clock
        self._lock, self._at, self._value = threading.Lock(), None, None

    def _read(self):
        now = None
        try:
            for g in self._query():
                same = g.get("uuid") == self.gpu["uuid"] if self.gpu.get("uuid") else g.get("index") == self.gpu.get("index")
                if same:
                    now = g
                    break
        except Exception:  # noqa: BLE001 - nvidia-smi is gone, slow or prints something else: the memory in use is not known
            now = None
        g = now or self.gpu
        return {"name": g.get("name"), "driver": g.get("driver"), "total_mib": g.get("total"),
                "used_mib": now.get("used") if now else None}

    def get(self, fresh=False):
        with self._lock:                                # one nvidia-smi at a time; the other requests wait for its answer
            if fresh or self._at is None or self._clock() - self._at >= self._ttl:
                self._value = self._read()
                self._at = self._clock()
            return dict(self._value)


def make_web_handler(page, conf, engine_port, web_port, update=None, control=None, session=None, gpu=None, checknow=None):
    """The handler of the browser chat: the page, its settings, and three paths passed on to `yasakani serve`.

    Like `yasakani serve`, it answers only requests that name this address in ``Host`` and, if they carry an ``Origin``, come from
    this address - so a web page of another site that is open in the browser cannot talk to the model through it.  With ``update``
    (an UpdateRequest) it also answers POST /webchat/update; without it that path does not exist.  In the same way ``control`` (a
    ControlRequest) brings POST /webchat/shutdown and /webchat/switch_model, and ``session`` (a WebSession) GET /webchat/status.
    ``gpu`` (a GpuWatch) gives the GPU line of /webchat/config.  ``engine_port`` None means that no `yasakani serve` runs now (the
    checks of a model switch, open_check_server): api_base is null and the paths that are passed on answer 503.  ``checknow`` (an
    UpdateCheckNow, only together with ``control``) brings POST /webchat/check_update, the page's "check for updates"."""
    own = ("127.0.0.1:%d" % web_port, "localhost:%d" % web_port)
    origins = tuple("http://" + h for h in own)
    if control is None:
        checknow = None                                 # it takes the control token of the page
    if control is not None and session is None:
        session = WebSession(conf.get("model"))         # a control request is checked against the session's models
    # the address of `yasakani serve` for other programs on this computer
    api_base = "http://127.0.0.1:%d/v1" % engine_port if engine_port is not None else None

    def config_bytes():
        # built for every request: the tokens, the GPU memory and what the engine reported change while the page is open
        c = dict(conf, api_base=api_base, model_id=DEFAULT_MODEL_ID, max_context_tokens=None, models_installed=[],
                 control_token=control.token if control is not None else None,
                 gpu=gpu.get() if gpu is not None else {"name": None, "driver": None, "total_mib": None, "used_mib": None})
        rel = update.rel if update is not None else None
        if rel is not None:                             # known at the start, or found by the page's "check for updates"
            c["update"] = {"version": rel["version"], "page": rel["page"]}
            c["update_token"] = update.token            # null once it was used or the launcher is leaving the page
        if checknow is not None:
            c["update_check_now"] = True                # the page shows its "check for updates"
        if session is not None:
            c.update(model_id=session.model_id, max_context_tokens=session.max_context,
                     models_installed=[{"name": m, "current": m == session.model} for m in session.installed])
        return json.dumps(c).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 300
        server_version = "yasakani-quickstart/" + QS_VERSION
        sys_version = ""

        def log_message(self, *args):
            pass

        def _refused(self):
            if (self.headers.get("Host") or "").lower() not in own:
                return "host"
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() not in origins:
                return "origin"
            if (self.headers.get("Sec-Fetch-Site") or "none").lower() not in ("same-origin", "none"):
                return "site"
            return None

        def _send(self, code, ctype, body, extra=(), close=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")     # the page and its answers are never shown inside another page
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in extra:
                self.send_header(k, v)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, message, kind, close=False):
            self._send(code, "application/json; charset=utf-8",
                       json.dumps({"error": {"message": message, "type": kind}}).encode("utf-8"), close=close)

        def _relay(self, method, path, body=None):
            if engine_port is None:                     # no `yasakani serve` now (the checks of a model switch)
                return self._error(503, "the model is being changed", "unavailable", close=True)
            conn = http.client.HTTPConnection("127.0.0.1", engine_port, timeout=900)
            started = False
            try:
                headers = {"Accept": "*/*"}
                if body is not None:
                    headers["Content-Type"] = "application/json"
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                ctype = resp.getheader("Content-Type") or "application/octet-stream"
                if ctype.lower().startswith("text/event-stream"):
                    started = True
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = resp.read()
                    started = True
                    self._send(resp.status, ctype, data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True      # the browser went away; closing the connection below stops the generation
            except (OSError, http.client.HTTPException) as ex:
                self.close_connection = True
                if not started:
                    try:
                        self._error(502, "the YASAKANI server did not answer (%s)" % type(ex).__name__, "upstream", close=True)
                    except OSError:
                        pass
            finally:
                conn.close()

        def do_GET(self):
            why = self._refused()
            if why:
                return self._error(403, "forbidden: " + why, "forbidden", close=True)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", page, (("Content-Security-Policy", PAGE_CSP),))
            if path == "/webchat/config":
                return self._send(200, "application/json; charset=utf-8", config_bytes())
            if path == "/webchat/status" and session is not None:
                status = session.status()
                self._send(200, "application/json; charset=utf-8", json.dumps(status).encode("utf-8"))
                if status["state"] == "failed":
                    session.told_failed.set()           # run_web keeps this server until a page has heard it (or FAIL_GRACE)
                return None
            if path in ("/health", "/v1/models"):
                return self._relay("GET", path)
            if path == "/favicon.ico":
                return self._send(404, "text/plain; charset=utf-8", b"")
            return self._error(404, "not found", "not_found")

        def do_POST(self):
            why = self._refused()
            if why:
                return self._error(403, "forbidden: " + why, "forbidden", close=True)
            path = self.path.split("?", 1)[0]
            if update is not None and update.rel is not None and path == "/webchat/update":
                return self._update()
            if checknow is not None and path == "/webchat/check_update":
                return self._check_now()
            if control is not None and path in ("/webchat/shutdown", "/webchat/switch_model"):
                return self._control(path == "/webchat/switch_model")
            if path != "/v1/chat/completions":
                return self._error(404, "not found", "not_found", close=True)
            try:
                n = int(self.headers.get("Content-Length", "") or -1)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._error(413, "request body too large or length missing", "invalid_request_error", close=True)
            self._relay("POST", "/v1/chat/completions", self.rfile.read(n))

        def _update(self):
            if (self.headers.get("Origin") or "").lower() not in origins:      # browsers send Origin with every POST of fetch()
                return self._error(403, "forbidden: origin", "forbidden", close=True)
            try:
                n = int(self.headers.get("Content-Length", "") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > 4096:
                return self._error(413, "request body too large", "invalid_request_error", close=True)
            if n:
                self.rfile.read(n)
            if not update.take(self.headers.get(UpdateRequest.HEADER)):
                return self._error(403, "forbidden: token", "forbidden", close=True)
            self._send(202, "application/json; charset=utf-8",
                       json.dumps({"status": "updating", "version": update.rel["version"]}).encode("utf-8"), close=True)
            self.wfile.flush()
            update.event.set()

        def _check_now(self):
            # as _control: Origin, the size, then the token (not used up: nothing that the token guards changes here)
            if (self.headers.get("Origin") or "").lower() not in origins:
                return self._error(403, "forbidden: origin", "forbidden", close=True)
            try:
                n = int(self.headers.get("Content-Length", "") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > CONTROL_MAX_BODY:
                return self._error(413, "request body too large", "invalid_request_error", close=True)
            if n:
                self.rfile.read(n)
            if not control.valid(self.headers.get(ControlRequest.HEADER)):
                return self._error(403, "forbidden: token", "forbidden", close=True)
            self._send(200, "application/json; charset=utf-8", json.dumps(checknow.run()).encode("utf-8"))

        def _control(self, switch):
            # the order: Origin, the size, the request itself (a refused request leaves the token valid), then the token
            if (self.headers.get("Origin") or "").lower() not in origins:      # as /webchat/update: Origin is required here
                return self._error(403, "forbidden: origin", "forbidden", close=True)
            try:
                n = int(self.headers.get("Content-Length", "") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > CONTROL_MAX_BODY:
                return self._error(413, "request body too large", "invalid_request_error", close=True)
            body = self.rfile.read(n) if n else b""
            name = None
            if switch:
                try:
                    j = json.loads(body.decode("utf-8"))
                except (ValueError, RecursionError):
                    j = None
                name = j.get("model") if isinstance(j, dict) else None
                if not isinstance(name, str) or name not in MODELS:
                    return self._error(400, "the body must be JSON with \"model\": one of " + ", ".join(MODEL_ORDER),
                                       "invalid_request_error", close=True)
                if name not in session.installed:
                    return self._error(409, "that model is not downloaded on this computer", "conflict", close=True)
                if name == session.model:
                    return self._error(409, "that model is in use already", "conflict", close=True)
            if not control.take(self.headers.get(ControlRequest.HEADER), "switch" if switch else "shutdown", name):
                return self._error(403, "forbidden: token", "forbidden", close=True)
            session.leave("switching" if switch else "stopping")       # /webchat/status says so from now on
            answer = {"status": "switching", "model": name} if switch else {"status": "stopping"}
            try:
                self._send(202, "application/json; charset=utf-8", json.dumps(answer).encode("utf-8"), close=True)
                self.wfile.flush()
            finally:
                control.event.set()                     # also when the page went away before it read the answer

    return Handler


def get_json(port, path, headers=None, timeout=5):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path, headers=headers or {})
        r = conn.getresponse()
        data = r.read()
        try:
            return r.status, json.loads(data or b"{}"), r
        except ValueError:
            return r.status, {}, r
    finally:
        conn.close()


def wait_ready(port, proc, seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            return False
        try:
            code, body, _ = get_json(port, "/health")
            if code == 200 and body.get("status") == "ok":
                return True
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(1.0)
    return False


def web_selftest(web_port):
    """Checks the browser chat once, through the same address the browser uses.  Prints one line per check."""
    results = []

    def line(name, good, extra=""):
        results.append(bool(good))
        print("  %-46s %s%s" % (name, "PASS" if good else "FAIL", ("   " + extra) if extra else ""), flush=True)

    def raw(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", web_port, timeout=600)
        try:
            h = dict(headers or {})
            if body is not None:
                h.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.getheader("Content-Security-Policy") or "", r.read()
        finally:
            conn.close()

    print("\nbrowser chat self-test (http://127.0.0.1:%d/)" % web_port, flush=True)
    try:
        code, ctype, csp, data = raw("GET", "/")
        line("the page is served", code == 200 and ctype.startswith("text/html") and b"YASAKANI" in data and "default-src 'none'" in csp,
             "%d bytes" % len(data))
        code, ctype, _, data = raw("GET", "/webchat/config")
        conf = json.loads(data) if code == 200 else {}
        line("the settings of the page", code == 200 and conf.get("version") == QS_VERSION)
        line("the address for other programs",
             re.fullmatch(r"http://127\.0\.0\.1:\d{1,5}/v1", str(conf.get("api_base"))) is not None
             and isinstance(conf.get("model_id"), str) and isinstance(conf.get("gpu"), dict)
             and isinstance(conf.get("models_installed"), list) and "control_token" in conf, str(conf.get("api_base")))
        code, ctype, _, data = raw("GET", "/health")
        h = json.loads(data) if code == 200 else {}
        line("the model is ready", h.get("status") == "ok", "context %s tokens" % h.get("max_context_tokens"))
        code, _, _, data = raw("GET", "/webchat/status")
        s = json.loads(data) if code == 200 else {}
        line("the page is told so", s.get("state") == "ready", "%s after %s s" % (s.get("state"), s.get("elapsed")))
        code, _, _, _ = raw("GET", "/health", headers={"Origin": "http://example.com"})
        line("a page of another site is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("GET", "/health", headers={"Host": "example.com"})
        line("another host name is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("POST", "/v1/chat/completions", body=b"{}", headers={"Sec-Fetch-Site": "cross-site"})
        line("a cross-site request is refused", code == 403, "HTTP %d" % code)
        # the two control paths, without anything that could stop this server: no token, and a request of another site
        code, _, _, _ = raw("POST", "/webchat/shutdown", body=b"", headers={"Origin": "http://127.0.0.1:%d" % web_port})
        line("quitting needs the token of the page", code in (403, 404), "HTTP %d" % code)
        code, _, _, _ = raw("POST", "/webchat/switch_model", body=b'{"model": "speed"}',
                            headers={"Origin": "http://example.com", ControlRequest.HEADER: "0" * 64})
        line("a model switch from another site is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("GET", "/yasakani_env.cmd")
        line("other files are not served", code == 404, "HTTP %d" % code)
        body = json.dumps({"messages": [{"role": "user", "content": "Write the numbers from 1 to 120, separated by spaces."}],
                           "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 480,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", web_port, timeout=600)
        try:
            conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            n, text, usage, t_first, t_last, buf = 0, [], None, 0.0, 0.0, b""
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    if not ev.startswith(b"data:"):
                        continue
                    d = ev[5:].strip()
                    if d == b"[DONE]":
                        continue
                    o = json.loads(d)
                    if o.get("usage"):
                        usage = o["usage"]
                    for c in o.get("choices") or []:
                        piece = (c.get("delta") or {}).get("content")
                        if piece:
                            n += 1
                            text.append(piece)
                            t_last = time.perf_counter()
                            t_first = t_first or t_last
            secs = t_last - t_first
            got = usage["completion_tokens"] if usage else n
            rate = (got - 1) / secs if got > 16 and secs > 0.1 else 0.0
            answer = "".join(text)
            line("an answer arrives as a stream", r.status == 200 and n > 16 and " 20 " in answer,
                 "%d tokens, %.0f tokens/s" % (got, rate))
        finally:
            conn.close()
    except Exception as ex:  # noqa: BLE001 - any failure is a failed self-test
        line("self-test ran to the end", False, "%s: %s" % (type(ex).__name__, ex))
    good = all(results)
    print("browser chat self-test: %s" % ("PASS" if good else "FAIL"), flush=True)
    return good


def wait_for_stop(proc, update, poll=0.2, control=None, session=None):
    """Waits until Enter is pressed in this window (-> "enter"), the server ends ("ended"), the page asked for the update
    ("update"), or the page's quit button or model switch was accepted (``control``: "shutdown" or "switch").  Enter is read
    with msvcrt, not input(): a thread left waiting in input() would take the first line typed into the next quickstart.py.
    Nothing in this loop waits for the engine: ``session`` asks it in a thread of its own and tells the page whether the model is
    loading, ready or failed, so Enter is seen within ``poll`` seconds also while the model loads."""
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    tty = sys.stdin is not None and sys.stdin.isatty()
    if session is not None:
        session.watch()
    while True:
        if update is not None and update.event.is_set():
            return "update"
        if control is not None and control.event.is_set():
            return control.action
        if proc.poll() is not None:
            return "ended"
        if tty and msvcrt is not None:
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    return "enter"
        time.sleep(poll)


def remove_tree(path, seconds=60.0, say=None, sleep=time.sleep, clock=time.monotonic):
    """Removes the folder ``path``.  -> True when it is gone, False when a file in it stayed in use for ``seconds``.
    Right after an update the server of the version before can still be ending (the launcher ended the process it started, and
    the real interpreter and its GPU memory go a moment later).  Until then Windows refuses to delete the engine library, so the
    removal is tried again twice a second.  ``say`` is called once, before the first wait."""
    end, told = clock() + seconds, False
    while True:
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if clock() >= end:
                return False
            if not told and say is not None:
                say()
            told = True
            sleep(0.5)


def wait_released(path, seconds=15.0, sleep=time.sleep, clock=time.monotonic):
    """-> True when no process has the file ``path`` loaded any more (it can be opened for writing), False after ``seconds``.
    A missing file and a file that is write-protected anyway count as free."""
    end = clock() + seconds
    while True:
        try:
            if not os.access(path, os.W_OK):
                return True
            with open(path, "r+b"):
                return True
        except FileNotFoundError:
            return True
        except OSError:
            if clock() >= end:
                return False
            sleep(0.2)


def stop_process(proc, seconds=30, lib=None):
    """Stops `yasakani serve`: terminate, wait up to ``seconds``, then kill.  (python.exe of the private environment starts the real
    interpreter as its child inside a job object that ends with it, so ending the one we started leaves nothing behind.)
    The real interpreter can take a moment longer to go, and until then Windows keeps the engine library ``lib`` locked.  With
    ``lib`` given this waits (see wait_released) until it is free, so that whatever comes next - an update, a rollback, also a
    launcher of a version before this one - can replace the engine folder."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    if lib:
        wait_released(lib)


def wait_gpu_free(read_used, before, seconds=GPU_SETTLE_SECONDS, margin=GPU_SETTLE_MIB, sleep=time.sleep, clock=time.monotonic):
    """After `yasakani serve` was stopped for a model switch: waits until the GPU memory in use is back near ``before`` (the MiB
    in use before that server loaded its model), at most ``seconds``.  The next server chooses its context from the memory that is
    free when it starts, and the driver can take a moment to give the memory of the stopped one back.  -> True when the memory is
    back; False when the time is over or the memory cannot be read (the caller goes on either way)."""
    if before is None:
        return False
    end = clock() + seconds
    while True:
        used = read_used()
        if used is None:
            return False
        if used <= before + margin:
            return True
        if clock() >= end:
            return False
        sleep(0.5)


def open_page_server(make_handler, port, seconds=0.0, sleep=time.sleep, clock=time.monotonic):
    """A LocalServer on exactly ``port`` (``make_handler(port)`` gives its handler); None when the port cannot be had.  With
    ``seconds`` it tries again for that long: after a model switch the server of the model before has just closed on this port."""
    end = clock() + seconds
    while True:
        try:
            return LocalServer(("127.0.0.1", port), make_handler(port))
        except OSError:
            if clock() >= end:
                return None
            sleep(0.2)


def close_page_server(httpd):
    """Closes a LocalServer that serves in a thread, and the connections it accepted."""
    httpd.shutdown()
    httpd.server_close()
    httpd.close_connections()


def read_page():
    """chat.html next to this file.  -> (its bytes, its path); the bytes are None when the file is not there."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")
    if not os.path.isfile(path):
        return None, path
    with open(path, "rb") as f:
        return f.read(), path


def page_settings(ui, name):
    """The part of /webchat/config that does not change while the page is open."""
    return {"lang": ui.lang, "model": name, "version": QS_VERSION, "repo": REPO_URL, "support": SUPPORT_URL}


def open_check_server(ui, port, model, installed=(), gpu=None):
    """A model switch to a model that never passed the checks runs them first.  No `yasakani serve` runs during that time, and a
    page that waits at its address could not know why nothing answers.  This minimal server takes the page's port for that time:
    GET /webchat/status answers state "switching" with "phase": "checking", the seconds since the checks began and the new model's
    name.  It has no engine behind it (the paths that are passed on answer 503) and takes no button of the page (control_token is
    null, the control paths do not exist).  -> the server, or None when the page file or the port cannot be had (the console shows
    the checks either way).  The caller closes it with close_page_server() right before the browser chat starts on that port."""
    page = read_page()[0]
    if page is None or not port:
        return None
    session = WebSession(model, installed=installed)
    session.leave("switching")
    session.phase = "checking"
    httpd = open_page_server(lambda p: make_web_handler(page, page_settings(ui, model), None, p, None, None, session, gpu),
                             port, REOPEN_SECONDS)
    if httpd is not None:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def stay_for_the_page(session, since, clock=time.monotonic, sleep=time.sleep):
    """After Enter in the console, or after `yasakani serve` ended by itself: the page's server goes on answering /webchat/status
    ("stopping" or "failed") until END_GRACE seconds after ``since``, so that an open page can show the end instead of 'cannot be
    reached'.  After a server that ended by itself it first waits until a page has asked, at most FAIL_GRACE seconds.
    `yasakani serve` is stopped before this is called, so nothing is left behind when Ctrl+C ends the wait."""
    try:
        if session.state() == "failed":
            session.told_failed.wait(FAIL_GRACE)
        rest = END_GRACE - (clock() - since)
        if rest > 0:
            sleep(rest)
    except KeyboardInterrupt:
        pass


def run_web(ui, a, vpy, env, mdir, dll, model, update_rel=None, gpu=None, installed=(), reopen=None, check_now=None):
    """The page in the browser.  Starts `yasakani serve`, opens the page at once (it shows the loading and goes on when the model
    is ready), then waits for Enter, a button of the page or the end of the server.  -> 0, 7 (the server did not start, or it
    ended by itself with an error), 8 (--web-test failed), UPDATE_RC, QUIT_RC, or SWITCH_RC with the model's name in a.web_switch.
    ``gpu`` is a GpuWatch and ``installed`` the models the page can switch between.  ``reopen`` is the port of a page that waits
    after a model switch: the page's server comes back on that port and no browser window is opened.  On every way out
    `yasakani serve` is stopped and the page's server is closed; the tokens of the page's buttons end with the wait (if the update
    button and a control request were both taken within one look of the wait, the update counts).  ``check_now`` (a function of a
    result dict -> the newer release or None: update_check with force=True) is the page's "check for updates"; the release that
    the page's update button stands for, known at the start or found by it, is left in a.web_update."""
    a.web_update = None
    page, page_path = read_page()
    if page is None:
        ui.say("web_nopage", p=page_path)
        return 7
    wanted = reopen or getattr(a, "web_port", None) or WEB_PORT
    eport = pick_port(ENGINE_PORT, avoid=(wanted,))
    control = None if a.web_test else ControlRequest()  # --web-test only looks: no quit button, no model switch
    update = UpdateRequest(update_rel) if update_rel or (check_now and control) else None
    checknow = UpdateCheckNow(check_now, update) if check_now and control else None
    before = gpu.get(fresh=True)["used_mib"] if gpu is not None else None     # the GPU memory in use before this model is loaded
    ui.say("web_boot")
    proc = subprocess.Popen([vpy, "-m", "yasakani", "serve", "--model-dir", mdir, "--dll", dll, "--port", str(eport)], env=env)
    name = "custom" if a.model_dir else model
    session = WebSession(name, proc, eport, () if a.model_dir else installed,
                         on_ready=lambda seconds: ui.say("web_loaded", s="%.0f" % seconds))
    conf = page_settings(ui, name)

    def handler(port):
        return make_web_handler(page, conf, eport, port, update, control, session, gpu, checknow)

    httpd, why = None, None
    try:
        if reopen:
            httpd = open_page_server(handler, reopen, REOPEN_SECONDS)
            if httpd is None:
                ui.say("web_moved")
        if httpd is None:
            wport = pick_port(wanted, avoid=(eport,))
            httpd = LocalServer(("127.0.0.1", wport), handler(wport))
        wport = httpd.server_address[1]
        a.web_port_used = wport
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d/" % wport
        if a.web_test:
            if not wait_ready(eport, proc):
                ui.say("web_fail")
                return 7
            session.probe()                             # /webchat/status says "ready" for the self-test
            return 0 if web_selftest(wport) else 8
        kept = wport == reopen                          # the page that asked for the switch is open at this address
        ui.say("web_again" if kept else "web_ready_manual" if a.no_browser else "web_ready", u=url)
        if not kept and not a.no_browser:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - the address is printed above
                pass
        flush_keys()                                    # a key typed during the steps before must not end the page at once
        try:
            why = wait_for_stop(proc, update, control=control, session=session)
        except KeyboardInterrupt:
            why = "enter"
        # Whatever ended the wait: the page's buttons are not taken any more.  A button that was taken in the same moment (its
        # token is used and the page has its 202) counts.
        # (an update button that was never armed has no token either: not pressed)
        pressed = update is not None and not update.close() and update.rel is not None
        asked = not control.close()
        if pressed:
            why = "update"
        elif asked and control.action and why in ("enter", "ended"):
            why = control.action
        if why == "update":
            ui.say("upd_web")
            return UPDATE_RC
        if why == "shutdown":
            ui.say("web_quit")
            return QUIT_RC
        if why == "switch":
            a.web_switch = control.model
            ui.say("web_switch", m=ui.t("m_" + control.model))
            return SWITCH_RC
        if why == "ended" and not session.ready:        # it never loaded the model: the failure of `yasakani serve` is above
            ui.say("web_fail")
            return 7
        if why == "ended" and proc.poll():              # it ended by itself with an error after it had been ready.  Not 0, so
            ui.say("web_died")                          # that START.cmd keeps the window with the server's messages open
            return 7
        return 0
    finally:
        # `yasakani serve` is stopped first, so nothing is left behind whatever happens after that.  The page's server is closed
        # last, so that the page hears what happens ("stopping", "switching", "failed").
        if control is not None:
            control.close()
        if update is not None:
            update.close()
        if checknow is not None:
            checknow.close()
        if update is not None and update.rel is not None:
            a.web_update = update.rel
        t_end = time.monotonic()
        if proc.poll() is None:
            session.leave("stopping")
            stop_process(proc, lib=dll)
            if why == "switch":
                wait_gpu_free(lambda: gpu.get(fresh=True)["used_mib"] if gpu is not None else None, before)
        try:
            if why != "switch":
                ui.say("web_end")
            if httpd is not None and why in ("enter", "ended"):
                stay_for_the_page(session, t_end)       # an open page gets the time to ask and to show the end
        finally:
            if httpd is not None:
                close_page_server(httpd)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="YASAKANI quickstart")
    ap.add_argument("--model", choices=sorted(MODELS), help="skip the question: quality, speed, max or inkling2 (Inkling-2 Mini, "
                                                           "experimental)")
    ap.add_argument("--model-dir", help="use this model folder instead of downloading one")
    ap.add_argument("--engine-zip", help="use this engine zip instead of downloading it (its SHA-256 is still checked)")
    ap.add_argument("--home", help="where to keep everything (default: the folder of this file)")
    ap.add_argument("--action", choices=ACTIONS, default="auto",
                    help="what to do after the checks.  auto (default): open the page in the browser at once when the chosen "
                         "model passed the checks in an earlier run, else show the menu; menu = always show the menu; "
                         "web = the page in the browser (chat, speed test, API address, model switch, quit button); "
                         "compare = look-ahead on and off; update = look for a new version now and install it after a yes "
                         "(with --yes without asking); rollback = put back the start files of the version before the last update")
    ap.add_argument("--menu", action="store_true",
                    help="show the menu instead of opening the page in the browser at once (the same as --action menu; "
                         "an --action given as well wins)")
    ap.add_argument("--no-update-check", action="store_true",
                    help="turn the automatic check for a new version off (kept in quickstart_state.json)")
    ap.add_argument("--update-check", action="store_true", help="turn the automatic check for a new version on again")
    ap.add_argument("--update-dev-base", metavar="URL",
                    help="development only: look for updates at http://127.0.0.1:<port> (a test server on this computer) "
                         "instead of GitHub")
    ap.add_argument("--web-port", type=int,
                    help="with the page in the browser: the port to try first (an update keeps the page's port)")
    ap.add_argument("--no-browser", action="store_true",
                    help="with the page in the browser: do not open a browser window (the address is printed)")
    ap.add_argument("--web-test", action="store_true",
                    help="start the page in the browser, wait for the model, check the page once and stop")
    ap.add_argument("--lang", choices=("ja", "en"), help="language of the messages (default: the Windows display language)")
    ap.add_argument("--yes", action="store_true", help="answer yes to questions (the update question only with --action update)")
    ap.add_argument("--skip-checks", action="store_true", help="do not run info / verify / selfcheck")
    ap.add_argument("--skip-driver-check", action="store_true",
                    help="do not stop when the NVIDIA driver is older than 580 (the engine will most likely not start)")
    if INK2_IMAGE:                                      # only while the switch is on: the help of 0.3.4 stays as it was
        ap.add_argument("--image-file", action="store_true",
                        help="Inkling-2 Mini: download the one more file for image input now (also after a no to the question)")
    a = ap.parse_args(argv)
    a.image_file = bool(getattr(a, "image_file", False))
    if a.web_test:
        a.action = "web"
    elif a.menu and a.action == "auto":
        a.action = "menu"
    return a


def main(argv=None):
    a = parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ui = Ui(a.lang or ui_language())
    print()
    ui.say("banner", v=QS_VERSION)
    print(REPO_URL)

    if sys.platform != "win32":
        ui.say("not_windows")
        return 2
    if sys.version_info < (3, 10):
        ui.say("old_python", v=sys.version.split()[0])
        return 2
    home = os.path.abspath(a.home or os.path.dirname(os.path.abspath(__file__)))
    try:
        home.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=home)
        return 2
    os.makedirs(home, exist_ok=True)
    state_path = os.path.join(home, "quickstart_state.json")
    st = load_state(state_path)
    n = 7

    def save():
        save_state(state_path, st)

    # updates: the folder of the start files, where they come from, and the arguments the next quickstart.py gets
    qs_dir = os.path.dirname(os.path.abspath(__file__))
    st["quickstart_dir"] = qs_dir                       # `yasakani update` finds the launcher through this
    if a.no_update_check or a.update_check:
        st["update_check"] = not a.no_update_check
        ui.say("upd_off" if a.no_update_check else "upd_on")
    save()
    try:
        src = UpdateSource(a.update_dev_base)
    except UpdateError as e:
        print(e)
        return 2
    if src.dev:
        ui.say("upd_dev", u=src.base)

    def after_args(action, web_port=None):
        out = ["--action", action, "--lang", ui.lang]
        for flag, val in (("--home", a.home), ("--model-dir", a.model_dir), ("--update-dev-base", a.update_dev_base)):
            if val:
                out += [flag, val]
        for flag, on in (("--skip-checks", a.skip_checks), ("--skip-driver-check", a.skip_driver_check), ("--yes", a.yes),
                         ("--no-browser", a.no_browser or action == "web")):
            if on:
                out.append(flag)
        if web_port:
            out += ["--web-port", str(web_port)]
        return out

    if a.action == "update":                            # before the checks of this computer: they are the new version's job
        rel = update_check(ui, st, save, src, force=True)
        if not rel:
            return 0
        rc = apply_update(ui, qs_dir, rel, src, update_answered(a), after_args("none"), st, save)
        return 0 if rc is None else rc
    if a.action == "rollback":
        return apply_rollback(ui, qs_dir, st, save, a.yes, after_args("none"))

    # 1. this computer
    ui.step(1, n, "s_check")
    print("Python %s" % sys.version.split()[0])
    try:
        gpus = query_gpus()
    except (OSError, subprocess.SubprocessError):
        ui.say("no_smi")
        return 2
    if not gpus:
        ui.say("no_gpu")
        return 2
    for g in gpus:
        ui.say("gpu_is", g=g["name"], c="%d.%d" % g["cc"] if g["cc"] else "?", t=g["total"] if g["total"] is not None else "?")
    gpu, verdict, pin = pick_gpu(gpus)
    if len(gpus) > 1:
        ui.say("gpu_pick", n=len(gpus), g=gpu["name"])
    if verdict == "old":
        ui.say("gpu_old", c="%d.%d" % gpu["cc"])
        return 2
    if verdict == "nocode":
        ui.say("gpu_nocode", c="%d.%d" % gpu["cc"])
        return 2
    if verdict == "small":
        ui.say("gpu_small", t=gpu["total"])
        return 2
    ui.say("gpu_tested" if verdict == "tested" else "gpu_other")
    cuda = cuda_driver_version()
    dv, dlabel = driver_verdict(gpu["driver"], cuda), driver_label(gpu["driver"], cuda, ui.lang)
    if dv == "old":                                     # before anything is downloaded: the engine cannot start on this driver
        if not a.skip_driver_check:
            ui.say("driver_old", d=dlabel)
            return 2
        ui.say("driver_skip", d=dlabel)
    elif dv == "unknown":
        ui.say("driver_unknown")
    else:
        ui.say("driver_ok", d=dlabel)
    quality_fits = gpu["total"] is None or gpu["total"] >= QUALITY_MIN_TOTAL_MIB
    gpu_watch = GpuWatch(gpu)                           # the page shows this GPU's memory while it is open

    # 2. model
    ui.step(2, n, "s_model")
    ok = st.setdefault("models_ok", {})
    model = a.model
    if not quality_fits and not a.model_dir:
        if model in MODELS and model != "speed":       # the quality, max-quality and Inkling-2 models need 24 GB
            ui.say("quality_no")
            return 2
        model = "speed"
        ui.say("speed_only")
    # --action auto: straight to the page in the browser when this model passed the checks in an earlier run, else as the menu
    direct = direct_start(a.action, model or st.get("model"), ok, a.model_dir)
    if a.action == "auto":
        a.action = "web" if direct else "menu"
    if not model:
        prev = st.get("model") or "quality"
        if a.yes or a.model_dir or a.action != "menu":
            model = prev
        else:
            d = {name: key for key, name in MODEL_KEYS.items()}.get(prev, "1")
            model = MODEL_KEYS.get(ask(ui.t("pick", d=d), d), "quality")

    def context_note(used):
        """On cards that do not keep the full context: the context the package will choose with ``used`` MiB in use.  Inkling-2
        Mini on every card, when its context would be shorter than the folder's 32,768 tokens or it does not fit."""
        if gpu["total"] is None or used is None or a.model_dir:
            return
        if model == "inkling2":                 # the context of chat and serve: the fast path on RTX 30 / 40 / 50 cards (0.3.4)
            n_ctx = estimate_ink2_context(gpu["total"] - used, fast=ink2_fast_expected(gpu.get("cc")),
                                          vision=ink2_vision_planned(os.path.join(home, "models", model), INK2_IMAGE))
            if n_ctx <= 0:
                ui.say("ctx_short", u=used)
            elif n_ctx < INK2_DEFAULT_CONTEXT:
                ui.say("ctx_ink", n="{:,}".format(n_ctx))
            return
        if gpu["total"] >= FULL_CONTEXT_MIN_TOTAL_MIB:
            return
        n_ctx, head_off = estimate_context(gpu["total"] - used, model)
        if n_ctx < MIN_CONTEXT:
            ui.say("ctx_short", u=used)
        else:
            ui.say("ctx_auto", n="{:,}".format(n_ctx))
            if head_off:
                ui.say("ctx_head")

    ui.say("picked", m=ui.t("m_" + model))
    context_note(gpu["used"])
    mdir = os.path.abspath(a.model_dir) if a.model_dir else os.path.join(home, "models", model)
    try:
        mdir.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=mdir)
        return 2
    have_model = os.path.isfile(os.path.join(mdir, "yasakani.json"))
    need = 2 if (a.model_dir or have_model) else MODELS[model]["need_gb"]
    free = shutil.disk_usage(home).free / 1e9
    ui.say("disk", f=free, n=need)
    if free < need:
        ui.say("disk_short", n=need)
        return 2
    st["model"] = model
    save_state(state_path, st)

    # 3. engine
    ui.step(3, n, "s_engine")
    edir = os.path.join(home, "engine")
    dll = os.path.join(edir, "yskn.dll")
    if st.get("engine_sha") == ENGINE_SHA256 and os.path.isfile(dll) and os.path.isfile(os.path.join(edir, "pyproject.toml")):
        ui.say("engine_have")
    else:
        local = a.engine_zip or os.path.join(home, ENGINE_ZIP)
        if os.path.isfile(local):
            ui.say("engine_local", z=os.path.basename(local))
            zpath, fetched = local, False
        else:
            os.makedirs(os.path.join(home, "downloads"), exist_ok=True)
            zpath, fetched = os.path.join(home, "downloads", ENGINE_ZIP), True
            ui.say("engine_get", u=ENGINE_URL)
            try:
                download(ENGINE_URL, zpath)
            except Exception as e:  # network errors of any kind
                ui.say("dl_fail", e="%s: %s" % (type(e).__name__, e))
                return 3
        got = sha256_of(zpath)
        if got != ENGINE_SHA256:
            if fetched:
                os.remove(zpath)
            ui.say("sha_bad", e=ENGINE_SHA256, a=got)
            return 3
        ui.say("sha_ok")
        if os.path.isdir(edir) and not remove_tree(edir, say=lambda: ui.say("engine_wait")):
            ui.say("engine_busy", p=edir)        # the state still names the engine before: the next start tries again
            return 3
        safe_extract(zpath, edir)
        st["engine_sha"], st["venv_ok"] = ENGINE_SHA256, False
        save_state(state_path, st)

    # 4. environment
    ui.step(4, n, "s_venv")
    vdir = os.path.join(home, "venv")
    vpy = os.path.join(vdir, "Scripts", "python.exe")
    if st.get("venv_ok") and os.path.isfile(vpy):
        ui.say("venv_have")
    else:
        if not os.path.isfile(vpy):
            subprocess.check_call([sys.executable, "-m", "venv", vdir])
        rc = subprocess.call([vpy, "-m", "pip", "install", "--disable-pip-version-check", edir, "huggingface_hub"])
        if rc != 0:
            ui.say("pip_fail")
            return 4
        st["venv_ok"] = True
        save_state(state_path, st)

    env = dict(os.environ)
    env["YASAKANI_DLL"] = dll
    env["YASAKANI_MODEL_DIR"] = mdir
    if pin:
        env["CUDA_VISIBLE_DEVICES"] = pin            # more than one GPU: the engine runs on the GPU checked in step 1

    # 5. model files
    ui.step(5, n, "s_dl")
    got = st.setdefault("models_downloaded", {})
    if a.model_dir or (got.get(model) and have_model):
        ui.say("model_have", p=mdir)
        if model == "inkling2" and INK2_IMAGE and not a.model_dir:
            # a folder of 0.3.2 to 0.3.4: the image file is offered once (not when the page's update started this run: the
            # page waits for the server and nobody looks at this window)
            offer_image_part(ui, st, save, vpy, env, home, mdir, ask_now=not (a.action == "web" and a.web_port),
                             forced=a.image_file)
    else:
        rc = fetch_model(ui, vpy, env, model, mdir)
        if rc != 0:
            ui.say("dl_fail", e="exit code %d" % rc)
            return 5
        got[model] = True
        save_state(state_path, st)

    def yk(cmd, *extra):
        args = [vpy, "-m", "yasakani", cmd, "--model-dir", mdir]
        if cmd != "verify":
            args += ["--dll", dll]
        args += list(extra)
        print()
        ui.say("run", c=cmd)
        return subprocess.call(args, env=env)

    def checks_done():
        return bool(a.skip_checks or (ok.get(model) and not a.model_dir))

    def run_checks():
        """info, verify and selfcheck for the model in use, unless it passed them before.  -> 0, or the exit code to end with."""
        if checks_done():
            ui.say("checks_skipped" if a.skip_checks else "checks_done")     # #18: --skip-checks never ran them
            return 0
        differs = False
        for cmd in ("info", "verify", "selfcheck"):
            if cmd == "selfcheck":
                ui.say("load_note")
            rc = yk(cmd)
            if rc == 3 and cmd == "selfcheck" and verdict != "tested":
                differs = True
            elif rc == 1 and cmd == "selfcheck" and verdict != "tested":   # 0.1.4: the predictions or the self-consistency failed
                ui.say("selfcheck_fail_other", rc=rc, r=REPO_URL)
                return 6
            elif rc != 0:
                ui.say("check_fail", c=cmd, rc=rc, r=REPO_URL)
                return 6
        print()
        ui.say("checks_other" if differs else "checks_ok")
        if not a.model_dir:
            ok[model] = True
            save_state(state_path, st)
        return 0

    envfile = os.path.join(home, "yasakani_env.cmd")

    def write_env():
        with open(envfile, "w", encoding="ascii", newline="\r\n") as f:
            f.write("@echo off\nrem Run this in a command prompt, then type: yasakani chat   (or serve, bench, info ...)\n")
            f.write('set "YASAKANI_DLL=%s"\nset "YASAKANI_MODEL_DIR=%s"\nset "PATH=%s;%%PATH%%"\n'
                    % (dll, mdir, os.path.join(vdir, "Scripts")))
            if pin:
                f.write('set "CUDA_VISIBLE_DEVICES=%s"\n' % pin)

    # 6. checks
    ui.step(6, n, "s_verify")
    rc = run_checks()
    if rc:
        return rc
    write_env()

    # 7. use it
    ui.step(7, n, "s_menu")
    print(ui.t("support", u=SUPPORT_URL, r=REPO_URL))
    action = a.action
    pending = None                                      # a newer release this launcher knows of (offered once, then in the chat page)

    def check_now(result):                              # the page's "check for updates": as menu item 7, also with --no-update-check
        rel = update_check(ui, st, save, src, force=True, result=result)
        if rel:
            ui.say("upd_page", o=QS_VERSION, n=rel["version"])
        return rel

    if action == "menu" or direct:
        print()
        pending = update_check(ui, st, save, src, disabled=check_disabled(a, st))
        if pending and action == "menu":
            rc = apply_update(ui, qs_dir, pending, src, update_answered(a), after_args("menu"), st, save)
            if rc is not None:
                return rc
        elif pending:                                   # the direct start asks nothing here: the page shows its update button
            ui.say("upd_page", o=QS_VERSION, n=pending["version"])
    if direct:
        ui.say("direct", m=ui.t("m_" + model))
    while True:
        if action == "menu":
            cmd = menu_choice(ui)
        else:
            cmd = None if action == "none" else action
        if not cmd:
            break
        if cmd == "update":
            rel = update_check(ui, st, save, src, force=True)
            if rel:
                rc = apply_update(ui, qs_dir, rel, src, update_answered(a), after_args("menu"), st, save)
                if rc is not None:
                    return rc
            continue
        if cmd == "web":
            reopen = None
            while True:                                 # once more for every model the page switches to
                rc = run_web(ui, a, vpy, env, mdir, dll, model, pending, gpu_watch,
                             [] if a.model_dir else installed_models(home, got, quality_fits), reopen, check_now)
                pending = getattr(a, "web_update", None) or pending     # also a release the page's "check for updates" found
                if rc != SWITCH_RC:
                    break
                # The page waits at its address.  What the steps above did for the first model, for this one: run the checks if
                # it never passed them, then remember it and write the settings file.  Then `yasakani serve` again, on the page's
                # port.  While the checks run, a minimal server tells the page so at its address (open_check_server).
                back, reopen = (model, mdir), a.web_port_used
                model = a.web_switch
                mdir = os.path.join(home, "models", model)
                env["YASAKANI_MODEL_DIR"] = mdir
                print()
                ui.say("picked", m=ui.t("m_" + model))
                context_note(gpu_watch.get(fresh=True)["used_mib"])
                told = None if checks_done() else open_check_server(ui, reopen, model, installed_models(home, got, quality_fits),
                                                                    gpu_watch)
                try:
                    rc = run_checks()
                finally:
                    if told is not None:
                        close_page_server(told)
                if rc:
                    # The state file and the settings file name a model only after it passed the checks, so they still name the
                    # model before.  The page goes on with that one.
                    print()
                    ui.say("switch_back", m=ui.t("m_" + model), o=ui.t("m_" + back[0]))
                    model, mdir = back
                    env["YASAKANI_MODEL_DIR"] = mdir
                    continue
                st["model"] = model
                save_state(state_path, st)
                write_env()
            if rc == QUIT_RC:                           # the quit button of the page ends the launcher, also from the menu
                break
            if rc == UPDATE_RC:                         # the button of the page: that is the yes
                rc = apply_update(ui, qs_dir, pending, src, True, after_args("web", a.web_port_used), st, save)
                if rc != 9:
                    return 0 if rc is None else rc
                if action != "menu":
                    return 9                            # the update failed and nothing changed: not 0, so that START.cmd keeps the window with the reason open
                pending, rc = None, 0                   # the update failed and nothing changed: back to the menu
            if rc and action != "menu":
                return rc
            if action == "menu":
                flush_keys()                            # an Enter typed twice must not open the page again from the menu
        elif cmd == "compare":
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lookahead_check.py")
            if is_inkling2_dir(mdir):                   # the script measures the look-ahead of the Qwen3.8 engine
                ui.say("cmp_qwen_only")
                rc = 0
            elif not os.path.isfile(script):
                ui.say("cmp_nofile", p=script)
                rc = 7
            else:
                print()
                rc = subprocess.call([vpy, script, "--model-dir", mdir, "--dll", dll, "--lang", ui.lang], env=env)
            if rc and action != "menu":
                return rc
        else:
            ui.say("load_note")
            yk(cmd)
        if action != "menu":
            break
    print()
    # what the next double-click does: the page at once when the model of the state file has passed the checks, else the steps
    ui.say("again_direct" if direct_start("auto", st.get("model"), ok) else "again", f=envfile)
    print(ui.t("support", u=SUPPORT_URL, r=REPO_URL))
    ui.say("bye")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
