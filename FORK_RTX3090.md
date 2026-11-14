# RTX 3090 (sm_86) fork — unofficial

This is an **unofficial** fork of [YASAKANI](https://github.com/yasakani-dev/yasakani) that adds an experimental build
for the NVIDIA RTX 3090 (compute capability 8.6, sm_86). It is not affiliated with or endorsed by the YASAKANI authors.
The upstream code is licensed under Apache-2.0; see `LICENSE` and `NOTICE` (unchanged).
Based on upstream commit `91a5178`. Only source is published here: no prebuilt `yskn.dll` and no model files.

Problems with the 3090 build belong to this repository, not to the upstream project.

## Status: runs on one RTX 3090, lightly tested, not tuned

Checked on 2026-11-14 on one PC (RTX 3090 24 GB, driver 621.01, Windows 11, CUDA Toolkit 13.5, Visual Studio 18 Community
with MSVC 19.52, CMake 4.4.4, Ninja 1.13.2, Python 3.11.9) with the fastest model
(`yasakani-dev/Qwen3.8-27B-YASAKANI-speed`, revision `cd78f90d205677f8b3d2ce95eae82a64e278b910`):

| Check | Result |
|---|---|
| `yasakani verify` | all 12 files match |
| `yasakani selfcheck --max-ctx 32768` | 3 of 3 pass: the token ids are the same as the reference made on an RTX 5090 (3, 160 and 32 tokens) |
| `quickstart/lookahead_check.py` (context 262,144, greedy, 768 tokens) | the text is the same with and without look-ahead in all 4 tasks |
| `yasakani serve --max-ctx 32768` with `bench/20261108/scripts/custom_bench.py` (the three tasks of `bench/20261103/prompts/cases.json`, greedy, one run) | `edit_code` (4,769 prompt tokens, 4,708 written) and `edit_jsonl` are exact; `jsonl_orders_8k` wrote 8,192 tokens. All three answers are byte-identical to the RTX 5090 records in `bench/20261108/results/n12_ysk_speed_v8/own/`. Speed: 379 / 360 / 386 tok/s (5090: 1,222 / 1,156 / 1,260) |
| Sampling (temperature 1.0, top-p 0.95, top-k 20, seed 1234; 2 prompts, 256 tokens), look-ahead on and three kinds of off, 8 runs each | 64 of 64 runs gave the same tokens as the other runs of their prompt, across all four settings |
| The same request repeated (greedy and sampled, look-ahead on and off, 8 runs each) | 64 of 64 identical within their group |
| `bench/20261101/scripts/onoff_eq.py --quick` (30 runs) | all 15 greedy runs equal; 2 of 15 sampled runs differed from the other runs of their case from one position on (see "Run-to-run differences") |
| Server with its default settings (sampling, thinking on), no thinking, a second turn, a stream dropped by the client, a 30,052-token prompt | all answered normally; the answers read correctly |
| GPU memory | context 32,768: nvidia-smi showed 15,907 MiB in use with the model loaded, against about 1,000 MiB before (about 14.6 GiB for the engine); context 262,144: it loads, and the loader reports 3.8 of 25.8 GB free |

Speed, tokens per second, measured with `lookahead_check.py` at context 262,144 (the upstream figures for the RTX 5090
are from docs/TECHNOLOGY.md, table 1.3, measured at context 16,384):

| | no look-ahead | prose | thinking | code | CSV to JSON |
|---|---|---|---|---|---|
| RTX 3090, this build | 47 | 90 | 105 | 137 | 393 |
| RTX 5090, upstream | 126 | 279 | 298 | about 420 | 1,264 |

That is 0.31 to 0.37 times the RTX 5090. One pass takes 21.4 ms without look-ahead (7.9 ms on the 5090). Nothing was
tuned for this card: the worker counts, splits and grids are the upstream values for the 5090, and PDL is off.
Reading a 30,052-token prompt took about 34 s.

Not checked: the quality model, prompts longer than about 30,000 tokens, generations longer than 8,192 tokens, tool
calls, the terminal chat, other 8.x cards, accuracy against the original model.
The `120` build of this fork has not been run on an RTX 50 series card: only its device code was compared with upstream.
All of this is one PC, and most commands were run once.

### Run-to-run differences

Upstream documents that a repeated request now and then gives different tokens from one position on while other
programs use the GPU, and that the cause is not known (docs/MEASUREMENTS.md, "Checks on this release"). The same was
seen here: 2 of 158 runs in the checks above (both sampled, both while another program held about 2 GB of GPU memory).
The repeated-request checks right after showed none in 128 runs. This fork does not fix or explain it.

## Build and run

Do not use the upstream quickstart or the `yskn.dll` of the upstream release page: they are built for sm_120
(the RTX 50 series) and do not run on an RTX 3090. Where `python/README.md` and `BUILDING.md` say "the released
`yskn.dll`", "sm_120" or "Visual Studio 2022", read the steps below instead.

You need: Visual Studio with the C++ tools **and a Windows SDK**, the NVIDIA CUDA Toolkit, CMake 3.24 or newer, Ninja,
Python 3.10 or newer. Tested with the versions listed above; others are untried. Run everything from the
"x64 Native Tools Command Prompt for VS", in the folder that holds `CMakeLists.txt`:

```bat
cmake -S . -B build86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DYSKN_CUDA_ARCH=86
cmake --build build86
python -m pip install numpy tokenizers jinja2 huggingface_hub
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-speed --revision cd78f90d205677f8b3d2ce95eae82a64e278b910 --local-dir models\speed
set PYTHONPATH=%CD%\python
python -X utf8 -m yasakani verify    --model-dir models\speed
python -X utf8 -m yasakani selfcheck --model-dir models\speed --dll build86\yskn.dll --max-ctx 32768
python -X utf8 -m yasakani serve     --model-dir models\speed --dll build86\yskn.dll --max-ctx 32768
```

In PowerShell, set the path with `$env:PYTHONPATH = "$PWD\python"` instead of the `set` line.
`YSKN_CUDA_ARCH` defaults to `120` (RTX 50 series), which builds the upstream code as before.
The model is a download of about 13 GB. Only the fastest model at the revision above was checked.

Notes for a 24 GB card:

* Without `--max-ctx` the engine reserves the full 262,144-token context. On the test PC that left 3.8 GB free. If other
  programs use GPU memory, Windows moves part of it to shared memory and generation becomes several times slower,
  without an error; the warning at start only appears above 95% use. Use `--max-ctx 32768` (tested) unless you need
  more, and close other programs that use the GPU.
* This card is about a third as fast as the RTX 5090. Use the server with `"stream": true`, or lower the limit with
  `--default-max-tokens` / `--max-tokens-cap`: as upstream, a request without streaming keeps generating after the
  client has gone, and other requests wait for it.
* `.yskn` files are read by native code. Load only the official files from the Hugging Face repository above and
  check them with `yasakani verify` (see `SECURITY.md`).

To repeat the checks: `python -X utf8 quickstart\lookahead_check.py --model-dir models\speed --dll build86\yskn.dll`
(it has no `--max-ctx`, so it uses 262,144); with the server running,
`python bench\20261108\scripts\custom_bench.py --base http://127.0.0.1:8765/v1 --label rtx3090 --out-dir out --cases bench\20261103\prompts\cases.json --repeats 1`.

## What changed, and why

Only two PTX instructions in the source need sm_90 or newer, and both are compiled out below sm_90:

* `cp.async.bulk.prefetch` is a performance hint. Nothing depends on it.
* `griddepcontrol` is **not** just a hint: it is the synchronisation that programmatic dependent launch (PDL)
  relies on. It is removed only together with PDL itself, which is switched off at run time, so that kernels
  are launched plainly and stream order makes each one wait for the previous one.

| File | Change | Why |
|---|---|---|
| `CMakeLists.txt` | `YSKN_CUDA_ARCH` cache variable (default 120) sets `CMAKE_CUDA_ARCHITECTURES` and is passed to every source as a define. | Select the target; let host code know it. |
| `src/pdl.cuh` | `griddepcontrol.wait` / `.launch_dependents` only when `__CUDA_ARCH__ >= 900`; empty otherwise. | sm_90+ instructions (PDL). |
| `src/engine.cpp` | `sm90_features()`: false in a build whose target is below 90, otherwise read from the GPU once; `pdl_enabled()` follows it, so `launch_k` uses plain `<<<>>>` launches. | Without the attribute, stream order already makes each kernel wait for the previous one. |
| `src/q35_kernels.cu` | `cp.async.bulk.prefetch.L2.global` only when `__CUDA_ARCH__ >= 900`. | sm_90+ instruction (L2 prefetch, off by default). |
| `src/engine35.cpp` | Where `sm90_features()` is false, the L2 prefetch settings (`YSKN_L2PF`, `YSKN_L2PF2`, `set_opt` keys 25 and 27) are forced to 0. `load_bdraft` checks earlier CUDA errors before the SHEAD step. | No empty prefetch launches; the SHEAD fallback no longer hides earlier errors. |
| `src/loader.cpp` | `dev_copy` / `dev_zero` and the weight repack check CUDA return codes and throw. | Out-of-memory or launch failures during load used to pass silently (24 GB is tighter than 32 GB). |
| `src/c_api.cpp` | `yskn_open` throws if the GPU's compute capability is below the build target. | Clear error instead of "no kernel image" later. |
| `README.md`, `README.ja.md` | A note at the top that this is an unofficial fork. The rest is the upstream text, unchanged. | So that the fork is not taken for the official project. |
| `.github/FUNDING.yml` | Removed. | The sponsor button of an unofficial fork should not stand in for the upstream project; to support YASAKANI, see the upstream repository. |
| `.gitignore` | `/build*/` added. | The build folders of the steps above. |

With `YSKN_CUDA_ARCH=120` the sm_120 device code was checked to be the same as upstream: the PTX of all 23 `.cu` files
matches the upstream tree except for internal symbol ids that depend on the source path. Host-side changes only alter
behaviour when a CUDA call fails.

The `86` build keeps PDL and the L2 prefetch off on every GPU, also when it runs on a 9.0+ GPU, because
its device code has no `griddepcontrol`. It is meant for compute capability 8.x; on an RTX 50 series
card use the `120` build. Only `120` and `86` have been built; other values of `YSKN_CUDA_ARCH` are untried.

One case changes on any GPU: `yskn_open` with a device index that does not exist now fails, where the
upstream code silently used GPU 0.

Not changed: tuning values (workers, splits, grids), messages that mention sm_120, Python code, the quickstart.

## 日本語の要約

非公式のフォークです。YASAKANI の作者とは関係がありません。RTX 3090 (sm_86) 向けに `-DYSKN_CUDA_ARCH=86` で
ビルドできるようにしました。公開しているのはソースだけで、ビルド済みの DLL とモデルは含みません。

sm_90 以上の命令 2 種類 (PDL の griddepcontrol と L2 の一括先読み) を 8.6 では外し、実行時にも切っています。
加えて、ロード中の CUDA の失敗を黙って進めずに例外にしました。

RTX 3090 の実機 1 台・最速モデルで確かめたこと:

* selfcheck: 5090 で作られた基準と 3 本ともトークン列が一致
* 先読みあり/なしの一致: 貪欲法 4 課題、温度つきは 64 回すべて一致
* サーバー経由の長い入力 (約 4,800 トークン) と 8,192 トークンの生成: 5090 の記録とバイト単位で一致
* 既定の設定 (温度つき・思考あり) のサーバーの応答、30,052 トークンの入力

速度は調整前で、5090 の 0.31〜0.37 倍 (先読みなし 47 tok/s、先読みあり 90〜393 tok/s) です。
品質モデル・端末の chat・ツール呼び出しは未確認です。上流が記録している「たまに途中から出力が変わる」現象は
ここでも 158 回中 2 回見られました (原因は上流でも不明)。

使い方は上の "Build and run" のとおりです。上流の quickstart と配布の yskn.dll は RTX 5090 用なので使いません
(quickstart は 5090 以外では続けるか尋ね、続けると 5090 用の DLL を取得します)。Windows SDK を含む Visual Studio の
C++ ツール、CUDA Toolkit、CMake、Ninja が要り、「x64 Native Tools Command Prompt for VS」から実行します。
24 GB では `--max-ctx 32768` を付け、GPU を使う他のアプリを閉じてください。3090 版の不具合は上流ではなく
このリポジトリへ報告してください。
