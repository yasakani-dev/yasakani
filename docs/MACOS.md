# Experimental macOS version

This port runs the original `.yskn` weights on **Apple Silicon, macOS 14 or newer**.
Compressed matrix-vector products run on Metal. Attention, recurrent state and
small operations run through NumPy on the CPU. The prebuilt app needs no CUDA,
separate Python installation, Homebrew or compiler.

This is a new sequential backend, not a reproduction of the RTX 5090 engine.
It uses the stored floating-point value tables, float32 activations/state/KV,
and no speculative decoder. The CUDA speed, quality and bit-identical selfcheck
results do **not** describe this backend. Intel Macs and image input are not
supported. The weights stay compressed; they are not expanded into a 108 GB
float32 model. Start with a 4,096-token context and allow enough free unified
memory for the 10–13 GB model, recurrent state, KV cache and macOS.

## Prebuilt app and launcher

1. Download the `YASAKANI-macos-arm64` artifact from a successful **macOS prebuilt**
   Actions run. It contains `YASAKANI-<version>-macos-arm64.zip` and its `.sha256`.
   Download artifacts only from a commit you intend to run.
2. Check it with `shasum -a 256 -c YASAKANI-<version>-macos-arm64.zip.sha256`, then
   unzip it and move `YASAKANI.app` to a location of your choice.
3. Download either published YASAKANI model folder, including `yasakani.json`,
   its main `.yskn`, `tokenizer.json`, `tokenizer_config.json` and chat template.
   Use the quality/speed model links in the main README. The drafter and KV
   profile are not needed by Metal; downloading the complete folder also works.
   The app never downloads model weights automatically.
4. Open the app, choose that model folder, then click **Start**. When the model
   is ready, use **Open Chat** to open the included browser chat.
5. **Stop** shuts down the server. Quitting the launcher also stops its child.

The app is **ad-hoc signed, not Developer ID signed or notarized**. A downloaded
build may need approval in macOS System Settings → Privacy & Security → Open
Anyway. Only approve a build whose source/checksum you have reviewed. No global
Gatekeeper settings need to be changed.

The server listens on `127.0.0.1` and uses the existing same-origin checks. The
launcher does not open a LAN port. Logs stay in the launcher window. Model paths
are passed as process arguments, including spaces and non-ASCII characters.

## Command line

The bundled CLI is also usable directly:

```sh
"YASAKANI.app/Contents/Resources/runtime/yasakani" serve \
  --backend metal --model-dir /path/to/model --max-ctx 4096
```

Use `chat` or `bench` with the same engine options. The OpenAI-compatible API is
at `http://127.0.0.1:8765/v1`. Metal is selected automatically on macOS unless an
explicit `--dll` is given; `--backend cuda` retains the existing engine path.
To select a nonstandard Metal library use `--backend metal --dll PATH` or
`YASAKANI_METAL_LIBRARY`. The default context for Metal is 4,096 tokens.

Sampling is seeded and repeatable within this backend. Requests start with a
fresh recurrent state (no prefix cache). Cancellation is checked at token
boundaries. Published CUDA `selfcheck.json` files are not Metal reference files;
use a separately generated and reviewed `selfcheck-metal.json` (or `--reference`).
References written by Metal are labeled with their backend and CUDA references
are refused.

## Build from source

Install Xcode Command Line Tools, CMake and **CPython 3.14** on Apple Silicon:

```sh
python3.14 -m venv build/venv
build/venv/bin/python -m pip install -r macos/requirements-build.txt ./python
build/venv/bin/python macos/build.py
```

The result is `build/macos-dist/YASAKANI.app`, a ZIP and its SHA-256 file.
The builder refuses to overwrite an existing app: select a fresh folder with
`--output`. It bundles Python, NumPy, the tokenizer, templates, licenses, the
Metal library and the native AppKit launcher. `Contents/Resources/build-info.json`
records the source revision, dirty state and dependency versions. No model,
signing credential or API key is included. The workflow uploads artifacts only;
it does not publish a GitHub Release or perform notarization.

For library-only development:

```sh
cmake -S . -B build -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build
cd python
../build/venv/bin/python -m unittest discover -v
cd ..
build/venv/bin/python src/metal/check_metal.py build/bin/libyskn_metal.dylib
```

The last command requires a real accessible Metal device and fails if it is
unavailable. It compares GPU operations against `tools/yskn_decode.py`; the
Python tests separately check format parsing, attention, recurrent state,
sampling and HTTP behavior. The existing CUDA stub tests are API tests, not
evidence of real-model Metal generation. GitHub-hosted macOS VMs may not expose
a Metal GPU, so the workflow builds the library and runs CPU numerical tests;
run the GPU check and real-model generation on physical hardware before release.

To exercise the packaged CLI, chat page, real model and repeated greedy requests:

```sh
build/venv/bin/python macos/smoke.py --app build/macos-dist/YASAKANI.app --model-dir /path/to/model
```

The smoke test starts its own loopback server and stops it afterwards. It checks
nonempty, repeatable generation, not broad language quality or CUDA parity.

## Validation of the initial port (2026-11-14)

Physical machine: Apple M2 Max, 32 GiB unified memory, macOS 26.6.2.
Model: `yasakani-dev/Qwen3.8-27B-YASAKANI-speed`, revision
`cd78f90d205677f8b3d2ce95eae82a64e278b910`; every listed model checksum matched.

* Python suite with the real tokenizer: 177 tests, 144 passed, 33 skipped
  because the optional Windows image-input dependency Pillow was not installed.
* Real Metal versus the independent format decoder: 30 fixture/error checks
  passed. Allocation-failure/retry fault injection also passed.
* Released-model operations: three embeddings and three representative matrices
  matched the independent decoder; maximum absolute matrix error was `6.68e-6`.
* Packaged runtime over HTTP: the 23-token prompt “What is 2 + 2? Answer briefly.”
  returned “4” on two consecutive greedy requests (2 output tokens including
  the stop token). Total request times were 52.575 s and 71.080 s. These include
  prefill and are **not** decode-throughput measurements. This initial port is slow.
* The actual app launched the model, reached Ready, and opened the bundled chat
  in Chrome. Launcher error/stop/quit behavior was also exercised with a test
  server, including a child ignoring SIGTERM.

Only the fastest model was exercised end to end. The quality model, other Macs,
macOS 14 itself, long contexts, broader generation quality and CUDA parity remain
unverified. Ad-hoc signature verification passed; Developer ID signing and
notarization were not performed.

## 日本語

Apple Silicon / macOS 14以降向けの実験版です。公開の`.yskn`モデルを直接読み、
圧縮行列をMetalで計算します。CUDA版の投機デコードや量子化KVキャッシュは使わず、
速度・品質・生成結果の完全一致は保証しません。Intel Macと画像入力は未対応です。

プリビルドZIPを展開してアプリを開き、モデルフォルダを選択して起動します。
Pythonの別途導入は不要です。モデル自体は同梱していません。
署名はアドホック署名で、公証は未実施です。必要ならmacOSの「プライバシーと
セキュリティ」から対象アプリのみ「このまま開く」で許可してください。
