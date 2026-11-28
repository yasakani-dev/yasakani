# Building yskn.dll from source

The release page has a prebuilt `yskn.dll`. You can also build it yourself. You do not need a GPU to build it, only to run it.
On Linux the same source builds `libyskn.so`, from source only: see [Linux](#linux-building-from-source).
日本語：Visual Studio 2022 Build Tools（C++）・CUDA Toolkit・CMake・Ninja を入れ、「x64 Native Tools Command Prompt for VS 2022」で下の 2 つのコマンドを実行します。Visual Studio 2026（MSVC 19.51）と CUDA Toolkit 13.2 でもビルドできたという利用者の方の報告があります（[#31](https://github.com/yasakani-dev/yasakani/issues/31)）。
Linux では CUDA Toolkit 13、GCC、CMake を入れ、下の「Linux (building from source)」の節の手順でソースからビルドします。寄稿者が確かめた方法で、公式の対応ではありません。ビルド済みの Linux 版はクラウドファンディングの目標の 1 つです。

## What the release was built with (2026-11-15)

| | |
|---|---|
| OS | Windows 11, x64 |
| C++ compiler | Microsoft Visual C++ 19.44.35226 (Visual Studio 2022 Build Tools, toolset 14.44.35207) |
| CUDA | NVIDIA CUDA Toolkit 13.2 (nvcc 13.2.78) |
| Build tools | CMake (3.24 or newer is required) and Ninja |
| Target | one DLL with machine code for compute capability 8.6 (`sm_86`, RTX 30 series), 8.9 (`sm_89`, RTX 40 series) and 12.0 (`sm_120`, RTX 50 series), plus PTX for `compute_120`. Tested on the RTX 5090 only |

The `yskn.dll` on the release page is the build of 0.4.0 (SHA-256 `9b4d867c75f8fd99c73adb4310e00770aeffe6f3fdd3c29858131f78ea6fadcc`,
70,103,040 bytes), compiled cleanly from the 0.4.0 source for all three GPU generations (`86-real;89-real;120`) with the tool versions of
the table above (MSVC 19.44.35226, CUDA 13.2.78, CMake + Ninja). It adds the multi-branch speculative look-ahead tree (ω mode) with exact
Tensor Core ancestor attention (`src/q35_atx.cu`, `src/q35_bdraft.cu`, `src/lfrow_kernels.cuh`, `src/engine35.cpp`, `src/engine35.hpp`,
`src/specx.cpp`, `src/specx.hpp`, `src/q35.hpp`), pattern edit copy (`src/edit_copy.hpp`), and deterministic copy-loop / width-invariance
controls (`src/c_api.cpp`, `src/lnord.cu`).
Earlier, the `yskn.dll` of 0.3.1 to 0.3.5 (SHA-256 `41a6bbd9bc1f865c79fb77274b84c26f7d167acb3a58097e7e5933bc723d3f5c`,
64,692,736 bytes) was the source of 0.2.0 plus two options that are off by default, `--suppress-tokens`
([#26](https://github.com/yasakani-dev/yasakani/pull/26)) and `--yarn2` ([#27](https://github.com/yasakani-dev/yasakani/pull/27)), in
`include/yskn_abi.h`, `src/args.hpp`, `src/c_api.cpp`, `src/engine35.cpp`, `src/engine35.hpp`, `src/misc_kernels.cu`, `src/pm_test.cu`,
`src/q35.hpp` and `src/q35_kernels.cu`. These files were compiled again in a copy of the build folder of 0.2.0 with the tool versions of
the table, and linked. The objects of `src/q35_atxpk.cu` and `src/q35_atxpk_ms.cu`, whose source did not change, are those of the 0.2.0
build: nvcc 13.2 gives some kernels of these files different machine code from one compilation to the next, so we kept the tested ones.
0.3.2 to 0.3.5 ship this same file.
The `yskn.dll` of 0.2.0 (SHA-256 `c995bb24c297ae823dfc92eaaf3357401083a9922abe2fdfeb56509c323fd052`, 64,498,688 bytes) was made in two
steps. The first step (SHA-256 `3e06622b…39e1`, 63,142,400 bytes) is the source of 0.1.7 plus the
new prompt reading and the edit copy ([docs/TECHNOLOGY.md](docs/TECHNOLOGY.md),
5.5 and chapter 8): GPU code for long passes (among others `src/q35_g6.cu`, `src/q35_g6_whole.cuh`, `src/kvq_fast.cuh`,
`src/kvq_vit2.cuh`, `src/q35_atxpk.cu`, `src/q35_atxpk4.cuh`, `src/q35_kernels.cu`, `src/misc_kernels.cu`) and host code
(`src/engine35.cpp`, `src/specx.cpp`, `src/c_api.cpp`, `src/edit_copy.hpp`, `src/prefill_mode.hpp`). Like 0.1.7 it has machine code for
the three GPU generations of the table above. It was built with the tool versions of the table, with the two commands below, by
compiling the changed files again in the build folder of the earlier build and linking (about four minutes). The second step added
the reader of 5-bit codes for the max-quality model (`src/lnord.cu`, `src/pfa.cu`, `src/q35_g6.cu`, `src/q35_flow.cu`,
`src/engine35.cpp`, `src/format.hpp`, `src/loader.cpp`), compiled the same way into a copy of the build folder of the first step. The
GPU code of the first step is in it unchanged, function by function (514 functions for each of the three GPU generations).
The 0.1.7 build (SHA-256 `2d64c3b42bb9e689449cdd7ccd7cd5f206f70615bcadb6c0640a16396386f468`) was
the source of 0.1.6 plus the experimental code for several requests at once (`serve --parallel N`): two new GPU files for passes of several
requests (`src/q35_atxpk_ms.cu`, `src/q35_sample_ms.cu`), kernels for several rows in other GPU files, and host code in `src/engine35.cpp`
and `src/c_api.cpp`. It was compiled with the two commands below into the build folder of the 0.1.5 build, one step at a time as the
code was added. The weight upload of [#25](https://github.com/yasakani-dev/yasakani/pull/25) (`src/devmem.cpp`, `src/devmem.hpp`,
`src/engine35.cpp`, `src/loader.cpp`) compiled four C++ files again and linked, and the 26 compiled GPU files stayed byte for byte those
of the build before it (`a0991ab6…7170`), giving `0d3d513b…45a6`. The last step added the wide copy passes per GPU and the
check on the card (`include/yskn_abi.h`, `src/c_api.cpp`, `src/engine35.cpp`, `src/engine35.hpp`, `src/gpu_caps.cu`, `src/gpu_caps.hpp`,
`src/specx.cpp`, `src/specx.hpp`): it compiled six C++ files and three GPU files again (`src/gpu_caps.cu`, and `src/pm_test.cu` and
`src/repack.cu`, which include `include/yskn_abi.h`) and linked. The other 23 compiled GPU files are byte for byte those of
`0d3d513b…45a6`, the GPU code of all three architectures is the same function by function (compared with `cuobjdump`), and three
functions are exported in addition (`yskn_build_info`, `yskn_time_cycle`, `yskn_wide_state`). The 0.1.6 build (`967de3ea…be8f`) was the
build of 0.1.5 (`918560d6…3f7a`) with the change of 0.1.6 in `src/c_api.cpp` (every entry point selects the engine's GPU,
[#20](https://github.com/yasakani-dev/yasakani/issues/20)) compiled into its build folder with the two commands below (the CUDA compiler was named
explicitly with `-DCMAKE_CUDA_COMPILER`): one C++ file was compiled again and linked, the exported functions are the same, and the GPU
code is that of 0.1.4, instruction for instruction (compared with `cuobjdump`). The 0.1.5 build was made the same way from the build of
0.1.4 (`bae77ae6…aa09`): five C++ files with the host-code changes of 0.1.5 (`src/engine35.cpp`, `src/format.hpp`, `src/specx.cpp`)
were compiled again. The 0.1.4 build was made the same way: its last change, one condition in `src/engine35.cpp`, was compiled
into the build folder of the build
before it. 0.1.4 adds GPU code: on compute capability 12.x, passes of 17 to 64 rows (the wide copy passes) have matrix
kernels of their own, which add the same integer sums as the 16-row kernels, so they give the same results bit for bit (`src/pfa.cu`,
`src/lnord.cu`, `src/q35_kernels.cu`; see docs/TECHNOLOGY.md, 5.4). Every other kernel compiles from the source of 0.1.3, which is the
build of 2026-11-07 (`0f5d973b…0bbb`, the engine of 0.1.0 to 0.1.2) plus two fixes (a missing synchronization between GPU threads, and a
scratch buffer per engine for the KV5 encoder). The host code adds the wider copy passes and the saved prompt states. In our checks on the
RTX 5090 the 0.1.4 build wrote the same tokens as decoding one token per pass, and passes of 17 to 64 rows gave the same predictions as
passes of 16 rows; the 0.1.5 build wrote the same tokens in the same number of passes as the 0.1.4 build (60 of 60 greedy generations
with each model). The 0.1.6 build wrote the same tokens in the same number of passes as the 0.1.5 build (60 of 60 greedy generations
with each model). In `yasakani bench` (its 4 prompts with each model, the median of 6 runs each) the speed of the 0.1.6 build
differed from that of the 0.1.5 build by -0.44 % to +0.26 %. The 0.1.4 speed rows in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) were
measured with the 0.1.4 build. We have not tried other compiler or CUDA
versions. A user built the 0.3.2 source with Visual Studio 2026 Community (MSVC 19.51.36263) and CUDA Toolkit 13.2 (nvcc 13.2.51), with
`-DCMAKE_CUDA_ARCHITECTURES=120`, in about three minutes and without errors
([#31](https://github.com/yasakani-dev/yasakani/issues/31)); we have not built with Visual Studio 2026 ourselves. Compute capability 12.0 needs a recent CUDA Toolkit.

## Steps

Open the **x64 Native Tools Command Prompt for VS 2022** (so that the compiler is on the path), go to the source folder (the one with
`CMakeLists.txt`) and run:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build\yskn.dll` (about 64 MB; the CUDA runtime is linked statically, so no CUDA Toolkit is needed to run it). The recorded build of
the release took about four minutes and printed many compiler warnings; they are known.

Pass the file to the tools with `--dll`:

```bat
yasakani info --model-dir D:\models\qwen38 --dll build\yskn.dll
```

`yasakani info` prints the SHA-256 of the DLL. A build made with a different compiler or CUDA version will not be byte-identical to the file on the
release page; that is expected. Check the model files with `yasakani verify`, and the whole installation with `yasakani selfcheck`.

Two engine settings are new in 0.2.0. `YSKN_Q35_PF_V2` chooses the prompt reading: 1 (the engine's default) is the reading of 0.2.0, 0
reads prompts and runs passes of 64 rows as 0.1.7 did. `YSKN_EDIT_COPY=1` turns the edit copy on; the engine's default is off. You do not
set them yourself: the Python package sets both from `--prefill` (or `YASAKANI_PREFILL`) and `--edit-copy` (or `YASAKANI_EDIT_COPY`), also
for a library you built and pass with `--dll` ([docs/TECHNOLOGY.md](docs/TECHNOLOGY.md), 5.5 and chapter 8).

## The Inkling-2 Mini library (`yskn_inkling2.dll`, since 0.3.1)

日本語：Inkling-2 Mini 用のエンジンは別の DLL（`yskn_inkling2.dll`）で、エンジンの zip の `yskn.dll` の隣に入っています。ソースは `src/inkling2/` です。「x64 Native Tools Command Prompt for VS 2022」で `src\inkling2\build.cmd cuda` を実行すると `build\inkling2\yskn_inkling2.dll` ができます。Qwen3.8 用の `yskn.dll` とはコードを共有しません。

The engine zip of 0.3.1 holds a second library next to `yskn.dll`: `yskn_inkling2.dll`, the engine for Inkling-2 Mini (text only in 0.3.1, still images since 0.3.5; one
request at a time; experimental). It has its own C interface (`src/inkling2/ink2_api.h`) and shares no code with `yskn.dll`.
The Python package loads it only for
a model folder whose `yasakani.json` says `"architecture": "inkling2"`, and finds it in this order: `--dll` when it names this library or a
folder that holds it, the environment variable `YASAKANI_INK2_DLL`, then `yskn_inkling2.dll` next to `yskn.dll` (`--dll`, `YASAKANI_DLL` or
the `bin` folder next to the package). The launcher passes the `yskn.dll` of its engine folder, so it needs no setting.

What is in `src/inkling2/`:

* `ink2_kernels.inc`: every GPU kernel, one source for two builds. `ink2_cuda.cu` compiles it as CUDA kernels; `ink2_emu.cpp` runs the
  same source on the CPU under a small thread-block emulator (used by the self-test and by the CPU build for the unit tests).
* `ink2_engine.cpp`: the order of the layer visits, the caches, the loading of the weights and the C interface (interface version 4 since 0.3.5,
  `ink2_api.h`). It checks the values of `config.json` it is given (head counts, sizes, experts and top-k, kernel and window sizes)
  before it allocates anything. `ink2_kargs.h`: the argument blocks of the kernels. `ink2_selftest.cpp`: every kernel on random
  inputs, the CPU emulation against the GPU build. `ink2_micro.cu`: launch and barrier measurements of the GPU.
* The dense (non-expert) matrices are multiplied either from bfloat16 (the published weights) or as they are stored in a converted
  `.yskn` file (4-bit NVFP4 and 8-bit MXFP8 rows, `--scheme quality`); `ink2_capabilities()` says so (bit 1).
* `ink2_drafter.inc`, `ink2_drafter_kernels.inc`, `ink2_drafter_kargs.h`, `ink2_drafter_api.h`: the model's drafter
  (`mtp.safetensors`) inside the same library, and the pass that checks its guesses (`ink2_capabilities()` bit 2). Its matrices
  can also be given as they are stored in a converted drafter file (`mtp_d4.yskn`, [docs/FORMAT.md](docs/FORMAT.md) section 10.6;
  bit 4, drafter interface version 2). The pass checks the guesses only up to the first one the drafter is unsure of (the
  confidence cut of `--draft auto`).
* The fast path (since 0.3.3; the package's `--fast`, see `python/yasakani/inkling2.py`): `ink2_kv_codec.inc` and
  `ink2_kv_mxfp8.inc` (an 8-bit key/value cache for the full-attention visits: E4M3 codes with one power-of-two scale per 32
  values, BF16 copies of the first 8 and the most recent positions; option `kv_mxfp8`), `ink2_kv_attention.inc` (attention over
  that cache), `ink2_splitk_attention.inc` (Split-K attention for single steps and short passes; with the option `pass_exact`,
  on by default, every row of a pass with logits gets the bits of a single step), `ink2_flash_attn.inc` and `ink2_flash_tc.cuh`
  (tiled Flash-GQA attention of prompt passes, FP32 and BF16 tensor cores), `ink2_packed_prefill.cuh`, `ink2_decode_experts.cuh`
  and `ink2_expert_tile.cuh` (products of prompt and verification passes), `ink2_fast.cu`, `ink2_fast.h`, `ink2_fast2.h`,
  `ink2_fast_engine.inc`, `ink2_fast2_engine.inc` and `ink2_fast2_gen.inc` (the fast kernels for RTX 30, 40 and 50),
  `ink2_nvfp4.cu` (expert products of prompt passes on the NVFP4 tensor cores of the RTX 50 series, built for `sm_120a`), and
  `ink2_drafter_sk.inc` / `ink2_drafter_sk_kargs.h` (Split-K attention of the drafter). All of it is off unless the package (or a
  caller of the load options and `ink2_set_option`) turns it on; with it off the library gives the results of 0.3.2 bit for bit.
* The image tower (since 0.3.5): `ink2_vision.cu` (the encoder: the patches of 40 x 40 pixels, 23 blocks and the projector, with the
  bfloat16 weights as published and FP32 sums), `ink2_vision.h` and `ink2_vision_api.h` (the exported `ink2_vision_*` calls and
  `ink2_feed_many_img`; `ink2_capabilities()` bit 32). `build.cmd cuda` builds it in; the host build has no image tower.
* `build.cmd`: the build. There is no CMake file for this library.

The released `yskn_inkling2.dll` of 0.3.5 has SHA-256 `84916af8672a428f5aa512c54161ee7594a19b6d0cf719623fe6c7f2f18d9771` (25,181,184 bytes): the
engine with the drafter (also from a converted drafter file, `mtp_d4.yskn`), with the products for packed dense matrices, with the
fast path and, new in 0.3.5, with the image tower, built from `src/inkling2/` of this release. 0.3.3 and 0.3.4 shipped the file
`282ddcc470e570207dae1362c188be57e6170518e03283475790a28c46a28c7f` (24,137,728 bytes), without the image tower. It was built with the tools of
the table above (Visual C++ 19.44, CUDA Toolkit 13.2) by `build.cmd cuda`. Like `yskn.dll` it has machine code for compute capability 8.6
(RTX 30 series), 8.9 (RTX 40 series) and 12.0 (RTX 50 series), plus PTX for `compute_120`, and machine code for `sm_120a` for the NVFP4
prompt products; the CUDA runtime is linked statically. It
also uses the OpenMP library of Visual C++ (`vcomp140.dll`, part of the Visual C++ 2015-2022 x64 runtime). Only the RTX 5090 has been
tested. At a context of 32,768 tokens the engine allocates about 23 GB of GPU memory for the published weights (24 GB with the drafter)
and about 19 GB for the weights of `--scheme quality` (19.5 GB with their 4-bit drafter), so cards with less than 24 GB cannot run it. The image tower takes about 0.75 GB more with the defaults (`--vision off` does not load it).
With the fast path (its 8-bit key/value cache) the weights of `--scheme quality` with their drafter took 17.9 GB at 32,768 tokens and
20.5 GB at 131,072 tokens (the engine's allocations, RTX 5090).

To build it, open the **x64 Native Tools Command Prompt for VS 2022** with the CUDA Toolkit 13 installed (the installer sets `CUDA_PATH`;
without it `nvcc` must be on the path), go to the source folder and run:

```bat
src\inkling2\build.cmd cuda
yasakani info --model-dir D:\models\inkling2-mini --dll build\inkling2\yskn_inkling2.dll
```

The build takes a few minutes and writes `build\inkling2\yskn_inkling2.dll` (about 25 MB). `set INK2_ARCHS=-gencode
arch=compute_120,code=sm_120` before it builds for the RTX 50 series only; `set INK2_NVFP4_TC=0` builds it without the NVFP4 prompt
products (the fast path then runs without them; since 0.3.4 `--fast auto` still turns it on for RTX 30, 40 and 50 cards). `src\inkling2\build.cmd host` builds
`build\inkling2\ink2_host.dll`, the kernels on the CPU only: it is not released and serves the unit tests of the package
(`YASAKANI_TEST_INK2_DLL`, with the tiny test checkpoint in `YASAKANI_TEST_INK2_SOURCE`; see `python/tests/test_inkling2.py`). The released
library is for Windows. Since 0.3.5 the same source also builds on Linux, from source only (see
[Inkling-2 Mini on Linux: `libyskn_inkling2.so`](#linux-inkling-2-library) below). Then check it
with `yasakani selfcheck` on the model folder: it compares the greedy tokens with those of the official test of the model.

## Linux (building from source)

The engine also builds on Linux, as `libyskn.so` (contributed in [#3](https://github.com/yasakani-dev/yasakani/pull/3)). This is building
from source, tested by contributors, not an official release: there is no prebuilt Linux release or launcher, the quickstart (`START.cmd`)
is for Windows, and the speed tables were measured with the Windows build only. An official Linux release (prebuilt files, a
launcher, checks on plain Linux, and support) is one of the crowdfunding goals.

### What it was tested with

| | WSL2 ([#10](https://github.com/yasakani-dev/yasakani/pull/10)) | Native Linux ([#8](https://github.com/yasakani-dev/yasakani/pull/8)) |
|---|---|---|
| OS | Ubuntu 24.04 under WSL2 on Windows 11 | Ubuntu 26.04.1 LTS, x86-64, kernel 7.0.0, glibc 2.43 |
| GPU and driver | RTX 5090; under WSL2 the driver is the Windows NVIDIA driver (tested with 621.24) | RTX 5090, NVIDIA driver 615.71.09 |
| C++ compiler | GCC 13.3 (the Ubuntu 24.04 default) | GCC 15.2.0 |
| CUDA | NVIDIA CUDA Toolkit 13.1, 13.3 (nvcc 13.3.73) and 13.5 (nvcc 13.5.37) | NVIDIA CUDA Toolkit 13.3 (nvcc 13.3.73) |
| Build tools | CMake 3.28 (3.24 or newer is required) with its default generator, Unix Makefiles | CMake 4.2.3, Ninja 1.13.2 |
| Python | 3.12 (3.10 or newer is required) | 3.13.13 (in a virtual environment made with uv 0.11.6) |
| CPU | | Intel Core i7-14700K (20 cores, 28 threads) |

The contributors built the 0.1.2 source (November 14, 2026). Under WSL2 each of these builds passed `yasakani selfcheck` with the fastest
model, and the CUDA 13.5 build with the quality model too. The CUDA 13.5 build followed the steps below in a new
`nvidia/cuda:13.5.0-devel-ubuntu24.04` container (which has the toolkit) with only `build-essential cmake git python3-venv` added. The
native Linux column is a contributor's working build; we have not tried native Linux ourselves. We built the 0.1.4 source with the steps
below under WSL2 (Ubuntu 24.04, CUDA 13.1, GCC 13.3, Windows driver 616.92), and `yasakani selfcheck` passed with the fastest model on our
RTX 5090 (November 15, 2026). The 0.2.0 source compiles and links with the same steps under WSL2 (Ubuntu 24.04, CUDA 13.1, code for RTX 30,
40 and 50); nothing was run with that build. Other distributions, compilers and CUDA versions have not been tried.

### Before you build

* An NVIDIA driver 580 or newer (the oldest for CUDA 13; see [docs/FAQ.md](docs/FAQ.md)). On native Linux this is NVIDIA's Linux driver.
  Under WSL2 it is the Windows driver, which is mapped into WSL: install the toolkit only, and no Linux driver (no `cuda-drivers` or
  `nvidia-open` package), as NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) says. `nvidia-smi` in
  the Linux terminal should list the GPU.
* The CUDA Toolkit 13 from NVIDIA's package repository for Ubuntu ([CUDA downloads](https://developer.nvidia.com/cuda-downloads): set up
  the repository, then `sudo apt install cuda-toolkit`, or `cuda-toolkit-13-3` and so on for a given version). Ubuntu's own
  `nvidia-cuda-toolkit` package is CUDA 12.0 on 24.04, too old for compute capability 12.0. Then the other tools:
  `sudo apt install build-essential cmake git python3-venv` (and `ninja-build` for `-G Ninja`).
* Under WSL2, keep the source and the model folder on the Linux file system (for example under `~`), not under `/mnt/c`; files there are
  slower to reach from Linux ([Microsoft's notes](https://learn.microsoft.com/en-us/windows/wsl/filesystems)).

### Steps

Get the source and build:

```sh
git clone https://github.com/yasakani-dev/yasakani
cd yasakani
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j
```

`-DCMAKE_CUDA_COMPILER` names the CUDA compiler, which NVIDIA's packages install in `/usr/local/cuda/bin`, a folder that is often not on
the path; without it CMake stops with "No CMAKE_CUDA_COMPILER could be found". With `nvcc` on the path it can be left out, and `-G Ninja`
works as well (the native build used both). The result is `build/bin/libyskn.so`. Like `yskn.dll` it has machine code for the RTX 30, 40
and 50 series ([GPUs](#gpus)); `-DCMAKE_CUDA_ARCHITECTURES=120` in the first command builds for one generation only, as on Windows. The
build uses the same CUDA flags as the Windows build (`-O3 --fmad=false`) and prints compiler warnings that do not stop it. With the 0.1.4
source it made a file of about 60 MB in about three minutes (`-j` on a 16-core CPU, under WSL2).

### Running it

Install the Python package in a virtual environment, download a model (here the quality model) and pass the library with `--dll`:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install ./python huggingface_hub
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-quality --local-dir ~/models/qwen38-quality
yasakani info      --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so
yasakani verify    --model-dir ~/models/qwen38-quality
yasakani selfcheck --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so
yasakani serve     --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so
```

After `pip install ./python` the package does not find `build/bin/libyskn.so` by itself (it looks next to the installed copy, not in the
source folder), so pass `--dll` as above, or set `export YASAKANI_DLL=$PWD/build/bin/libyskn.so` in the source folder. Run from the source
folder instead (`PYTHONPATH=python python -m yasakani selfcheck --model-dir ...`), the package finds it without either.

**The CUDA runtime.** Unlike `yskn.dll`, which has the CUDA runtime linked in statically, `libyskn.so` loads it at run time: it needs
`libcudart.so.13`. The build writes the toolkit's library folder into the file, and NVIDIA's packages also register their library folders with
the system's loader, so on the computer that built it, or on any computer with NVIDIA's CUDA 13 packages, nothing more is needed. Elsewhere,
loading the engine (`selfcheck`, `chat`, `serve`, `bench`; `info` and `verify` do not load it) fails with
`libcudart.so.13: cannot open shared object file`, at the end of a message written for Windows. Then install the CUDA runtime alone from
NVIDIA's repository, ideally the version the library was built with (for example `sudo apt install cuda-cudart-13-5`), or put a folder that holds
`libcudart.so.13` on `LD_LIBRARY_PATH`:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda-13.5/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

### Image input on Linux

The optional image input of the Qwen3.8 models ([python/README.md](python/README.md#image-input-optional)) uses the multimodal library of llama.cpp b10488. On
Windows the server takes the libraries of the b10488 release and compares them by SHA-256. That release has no matching Linux package to
compare with, so on Linux the same release is built from source, as shared libraries:

```sh
git clone --depth 1 --branch b10488 https://github.com/ggml-org/llama.cpp llama-b10488-src
git -C llama-b10488-src rev-parse --short=9 HEAD        # must print 9d77fa172
cmake -S llama-b10488-src -B llama-b10488-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
      -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build llama-b10488-build --target mtmd
mkdir llama-b10488
cp -a llama-b10488-build/bin/lib*.so* llama-b10488/
```

* `-DLLAMA_BUILD_TOOLS=ON` is needed because the `mtmd` target belongs to the tools; `--target mtmd` then builds only the library and
  what it depends on, not the programs.
* The `RPATH` options let the libraries find each other in whatever folder they are copied to, with no `LD_LIBRARY_PATH`.
* `-DGGML_CUDA=ON` is only needed for the projector on the GPU (below). A CUDA build was the one the contributor tested, in both
  modes; a build without it should be enough for the CPU mode, but that has not been tried. From the build files of llama.cpp: with
  these options the CUDA backend (`libggml-cuda.so`) is linked into `libggml.so`, not loaded on demand, so the CUDA runtime libraries
  (`libcudart`, `libcublas`) must be where the loader finds them. Without them the libraries do not load at all and every image
  request fails, also with the projector on the CPU.
* The tag alone does not pin the source: check the commit as above. The server checks it too (below).

Pass the folder as `--llama-dir`, with `--mmproj` and `--vision-model` as on Windows, and install the package with Pillow
(`pip install "./python[vision]"`):

```sh
yasakani serve --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so \
  --mmproj /gguf/mmproj-Qwen3.8-27B-BF16.gguf --llama-dir $PWD/llama-b10488 --vision-model /gguf/Qwen3.8-27B-Q4_K_M.gguf
```

On Linux there is no checksum to compare with. At start the server checks that `libllama.so`, `libmtmd.so`, `libggml.so` and
`libggml-base.so` are in the folder, and loads `libggml-base.so` to read the version and the commit it was built from: they must be
those of b10488 (ggml `0.20.2`, commit `9d77fa172`). A build of another release, of a changed ggml source (`-dirty`) or from a copy
without git (`unknown`) is refused, and a library with another structure layout when the encoder opens. This tells another release
apart, not every change: build b10488 unchanged. The server loads every `libggml-*.so` backend in the folder (and llama.cpp loads one
more library named by the environment variable `GGML_BACKEND_PATH`, on Windows too), so keep the folder writable only by you.

This part was contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8); its author ran it on native Ubuntu 26.04 with an
RTX 5090 (the table above). We have not run this image input on Linux ourselves, and it is not part of an official release.

By default the projector runs on the CPU, as on Windows. `YASAKANI_VISION_GPU=1` in the environment of the server moves it to the GPU,
with flash attention switched off. This is experimental and off by default: the image test was not repeated in this mode, and we have not
tried it. The measurements behind that choice, and its memory cost, are in [python/README.md](python/README.md#image-input-on-linux).
Mind the GPU memory: the projector and its work memory stay on the GPU next to the model, and the memory of an encode grows with the
square of the image tokens, so with the variable set `--vision-max-image-tokens` can be at most 2,048.

<a name="linux-inkling-2"></a>

### Inkling-2 Mini on Linux: the engine inside `libyskn.so` (`-DYSKN_WITH_INK2=ON`; #28, #30, #32)

日本語：Linux では、Inkling-2 Mini のエンジンを `libyskn.so` に入れて組み立てられます。@audreyt さんの寄稿です（[#28](https://github.com/yasakani-dev/yasakani/pull/28)・[#30](https://github.com/yasakani-dev/yasakani/pull/30)・[#32](https://github.com/yasakani-dev/yasakani/pull/32)）。CMake の `-DYSKN_WITH_INK2=ON` を付けます（標準は付けません）。この時は実行に cuBLAS（CUDA Toolkit の `libcublas.so`）が要ります。ダウンロードしたままのモデルのフォルダー（`yasakani.json` が無い物）を変換せずに読みます。0.3.5 で速くなり、先読み用のモデル（`mtp.safetensors`）も使います。キーと値は標準で bf16 で持ちます。`YSKN_INK2_KV=fp32` で 32 ビットになります。先読みは、標準では貪欲法の時だけです。環境変数で、サンプリングにも使うか読み込まないかを選べます（下）。画像も読めます。オプションと決まりは Windows と同じです（チャットの `/image` はありません）。画像には Pillow が要るので、パッケージは `pip install "./python[vision]"` で入れます。会話の次の番では、プレフィックスキャッシュで前の部分を読み直しません。ソースからの組み立てで、公式の配布物ではありません。配布する Windows の `yskn.dll` には入っていません。

[#28](https://github.com/yasakani-dev/yasakani/pull/28) (@audreyt) contributed a second engine for Inkling-2 Mini, inside `libyskn.so`;
[#30](https://github.com/yasakani-dev/yasakani/pull/30) and [#32](https://github.com/yasakani-dev/yasakani/pull/32), by the same author,
make it faster and let it read images (since 0.3.5). It is built only when you ask for it, from source; it is not part of an official
release, and the released Windows `yskn.dll` does not contain it:

* Add `-DYSKN_WITH_INK2=ON` to the first `cmake` command of the [steps](#steps-1) above. The option stops with an error on Windows.
* With it, `libyskn.so` also needs cuBLAS when it loads (`libcublas.so.13` and `libcublasLt.so.13` of the CUDA Toolkit), for every model,
  the Qwen3.8 models too. NVIDIA's `cuda-toolkit` package has them.
* It reads the folder of the published model as downloaded, with no `yasakani.json` in it (for example after
  `hf download thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini`). The package sends such a folder to this engine only
  when the library was built with the option. A library built without it gives the error of earlier versions for such a folder, with a
  hint. A folder made by `inkling2_convert.py` (with `"architecture": "inkling2"` in its `yasakani.json`) always takes the path of
  the Inkling-2 library above: `yskn_inkling2.dll` on Windows, and on Linux `libyskn_inkling2.so`, built from source
  ([below](#linux-inkling-2-library)).
* It runs the shapes of Inkling-2 Mini only. Before it loads anything it checks `config.json` and the type, shape and size of every
  text tensor in the weights file, and it stops with an error on anything else. It reads the published weights, not those of
  `--scheme quality`. One request at a time; `yasakani serve` takes tool calls. `--yarn2`, `--suppress-tokens`, `--parallel` and
  `selfcheck` are not available for this engine. Sampling uses the model's defaults (temperature 1.0, top-p 1.0, no top-k). It opens a
  context of 32,768 tokens unless `--max-ctx` is given, without looking at the free GPU memory.
* **Faster since 0.3.5 ([#30](https://github.com/yasakani-dev/yasakani/pull/30)):** vectorized kernels, prompt reading on the Tensor
  Cores in BF16, CUDA graphs, a screened output head, and the model's drafter (`mtp.safetensors`, loaded when the folder has it) with
  tree verification. We added three changes on top:
  * **Keys and values are stored in bf16 by default**, 16 bits per value like the engine of 0.3.4 (which used FP16).
    `YSKN_INK2_KV=fp32` in the environment stores them in 32 bits, as the pull request did by default; that takes more GPU memory
    (below).
  * **The drafter drafts for greedy decoding only by default** (temperature 0 or top-k 1). With sampling few drafted tokens are kept,
    so drafting costs more than it gives (numbers below). The drafter is still loaded and follows the text, so that a later greedy
    request can draft. `YSKN_INK2_MTP=1` makes it draft for sampling too, and `YSKN_INK2_MTP=0` does not load it.
  * A step of sampled generation with the drafter took seconds for each rejected draft token. We changed how it looks up the draft
    probabilities; it draws the same tokens without that delay.
* **Image input since 0.3.5 ([#32](https://github.com/yasakani-dev/yasakani/pull/32)).** The image tower of the published model runs on
  the GPU inside `libyskn.so` (`src/ink2_vision.*`) and reads its tensors from the downloaded weights file. The options and rules are
  those of the converted folder ([python/README.md](python/README.md#inkling-2-mini-image-input)): `--vision auto|on|off` (or
  `YASAKANI_INK2_VISION`), `--vision-max-image-tokens`, `--vision-max-request-image-tokens`, `--vision-allow-local-files`,
  `chat --image` (the chat of this engine has no `/image`), and in `serve` image parts in user and tool messages (`image_url`, `input_image` and `image`), with the
  same limits, the same shrink of large images and the same messages for what is refused. The package reads the image files with
  Pillow, so install it with `pip install "./python[vision]"` instead of `pip install ./python` ([Running it](#running-it)); without
  Pillow a request with an image fails with `image input needs Pillow`. What we added on top:
  * Before the image tower is used, `vision_config` in `config.json` and the type, shape and size of every image-tower tensor are
    checked against Inkling-2 Mini. A folder with another configuration, or with a damaged entry, opens with text only and one line
    that gives the reason (`--vision auto`); `--vision on` or `--image` stops with that reason.
  * A failed GPU allocation in the image tower frees what it took.
  * The export of the image entry is built only with `-DYSKN_WITH_INK2=ON`. The default build, and so the released Windows `yskn.dll`,
    keeps the entries and messages of 0.3.4.
* **GPU memory of the image tower.** Its weights are loaded at the first image. Unlike on Windows, this memory is not counted when the
  context is chosen (32,768 tokens unless `--max-ctx` is given). On a card with little free memory, leave room with a smaller
  `--max-ctx`, or use `--vision off`.
* **Image vectors.** The vectors of recent images stay in host memory (up to 512 MiB). `--vision-cache DIR` (from the contributor) also
  keeps them in DIR as `.npz` files, so that a later start need not encode the same image again. Only this engine takes it; for a
  converted folder it is refused.
* **Prefix cache ([#32](https://github.com/yasakani-dev/yasakani/pull/32)).** The engine keeps up to 6 snapshots of its state at the end
  of earlier prompts, matched on the tokens and on the images at their places, so that the next turn of a conversation reads only the
  new part. The snapshots are kept in pinned host memory: by our reading of the code up to about 1 GiB with the bf16 store, and about
  2 GiB with `YSKN_INK2_KV=fp32`. When that memory cannot be allocated, the engine skips the snapshot and goes on.
* **What we ran** (November 24, 2026, one RTX 5090 under WSL2, Ubuntu 24.04, CUDA 13.1, GCC 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120`, the
  published folder with `mtp.safetensors`, one run of each):
  * The 30 tokens that the official test of the model expects: 30 of 30.
  * `yasakani bench --max-ctx 8192 --n 128 --runs 2`, greedy: 203.0 tokens/s on average (21.9 with the engine of 0.3.4 in our earlier
    run).
  * Sampled with the model's defaults, a code prompt, 128 tokens, seeds 1, 2 and 3, a context of 4,096 tokens: 35.3, 43.0 and 44.8
    tokens/s by default; 21.9, 21.6 and 22.0 with the drafter drafting for sampling too; 56.4, 56.9 and 57.7 without the drafter. In our earlier run
    of 128 sampled tokens the engine of 0.3.4 gave 17.0 to 17.4 tokens/s.
  * GPU memory at the default context of 32,768 tokens (the peak of `bench --n 32` minus the 1,640 MiB in use before): 22,105 MiB. With
    `YSKN_INK2_KV=fp32` it was 23,593 MiB (measured with #30 alone, where the default also gave 22,105 MiB).
  * Images: four image prompts (a table, shapes and colours, the text in a picture) gave the same prompt tokens and the same first 64
    greedy tokens as our float32 reference that the Windows check uses. A 4032 × 3024 photo was shrunk to 999 image
    tokens, as on Windows. In the second turn of a conversation 144 of the 264 prompt tokens came from the prefix cache.
  * `yasakani serve`: all three kinds of image part were answered. A video part and an image in a system message got 400 with the
    messages of Windows. A copy of the folder with another `native_patch_size` in `config.json` opened with text only and the reason,
    and `chat --image` stopped with it.
  * Not checked: native Linux, GPUs other than the RTX 5090, `compute-sanitizer`, and the quality of the answers beyond these tests.
    The author of #28 ran that engine on an RTX 5090 and on a DGX Spark (GB10).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DYSKN_WITH_INK2=ON
cmake --build build -j
yasakani chat  --model-dir ~/models/Inkling-2-Mini --dll build/bin/libyskn.so
yasakani chat  --model-dir ~/models/Inkling-2-Mini --dll build/bin/libyskn.so --image picture.png
yasakani serve --model-dir ~/models/Inkling-2-Mini --dll build/bin/libyskn.so
```

<a name="linux-inkling-2-library"></a>

### Inkling-2 Mini on Linux: `libyskn_inkling2.so` (`build_linux.sh`, since 0.3.5)

日本語：0.3.5 から、Inkling-2 Mini 用のライブラリー（`src/inkling2/`）を Linux でもソースから組み立てられます（[#33](https://github.com/yasakani-dev/yasakani/pull/33)）。`bash src/inkling2/build_linux.sh` で `build/inkling2/libyskn_inkling2.so` ができ、`YASAKANI_INK2_DLL` でその場所を教えます。RTX 50 シリーズでは `INK2_NVFP4_TC=1` を付けます。CPU のエミュレーションとその自己試験は入っていません。寄稿者が確認した方法で、公式の対応ではありません。変換したフォルダー（`"architecture": "inkling2"`）は、このライブラリーが読みます。ダウンロードしたままのフォルダーは、上の `libyskn.so` のエンジンが読みます。画像には Pillow が要るので、パッケージは `pip install "./python[vision]"` で入れます。

[#33](https://github.com/yasakani-dev/yasakani/pull/33) (@Albrod) contributed a Linux build of the Inkling-2 library of
[the section above](#the-inkling-2-mini-library-yskn_inkling2dll-since-031). This is building from source, tested by contributors, not an
official release: there is no prebuilt `libyskn_inkling2.so`, and the released library is the Windows `yskn_inkling2.dll`.

* `src/inkling2/build_linux.sh` builds `build/inkling2/libyskn_inkling2.so` with nvcc and g++ (CUDA Toolkit 13, GCC). It uses the CUDA
  flags of `build.cmd cuda` (`-O3 -fmad=false`, `ink2_fast.cu` with the default `-fmad`), the same GPU generations (`sm_86`, `sm_89`,
  `sm_120` and PTX for `compute_120`; `INK2_ARCHS` chooses others) and the static CUDA runtime, and it includes the image tower
  (`ink2_vision.cu`). The library then depends only on the C and C++ runtime libraries of the system.
* `INK2_NVFP4_TC=1` also builds the NVFP4 prompt products for the RTX 50 series (`ink2_nvfp4.cu`, `sm_120a`). Unlike `build.cmd`, the
  script leaves them out by default; on an RTX 50 card set it, as in the commands below. Without them `--fast auto` still turns the fast
  path on, without the NVFP4 prompt products.
* The CPU emulation (`ink2_emu.cpp`, which uses Windows fibers) is not in the Linux build. `ink2_host_stub_linux.cpp` takes its place, so
  the `host` backend and the kernel self-test `ink2_selftest` stop with a message, and the unit tests of the package that need the CPU
  library (`YASAKANI_TEST_INK2_DLL`) cannot run on Linux. The `cuda` backend, which runs the model, does not use them.
* The package looks only for `yskn_inkling2.dll` by name. On Linux give the library with `YASAKANI_INK2_DLL` (or `--dll` naming the
  `.so` file itself).
* For images, install the package with Pillow: `pip install "./python[vision]"` instead of `pip install ./python`, as for
  [the engine above](#linux-inkling-2).
* Which engine reads which folder: a folder made by `inkling2_convert.py` (`"architecture": "inkling2"` in its `yasakani.json`, for
  example the published converted folder) goes to this library; the folder of the published model as downloaded (no `yasakani.json`) goes
  to the engine inside `libyskn.so` of [the section above](#linux-inkling-2). The two can be built side by side.
* What we ran (November 24, 2026, one RTX 5090 under WSL2, Ubuntu 24.04, CUDA 13.1, GCC 13.3; the converted folder with the image file,
  defaults, the build with `INK2_NVFP4_TC=1`): the build with and without `INK2_NVFP4_TC=1` without errors (about 25 MB, the 65 `ink2_*` entries of the
  Windows library); `yasakani selfcheck` passed (3/3); `yasakani bench` (256 tokens, the 4 built-in prompts, one run each, a 32,768-token
  context) gave 283.2 tokens/s on average, with at most 19,992 MiB of GPU memory in use in nvidia-smi (about 1,640 MiB of that before
  the start); and one request with an image (a table, "What is the total of Gamma in the table?") gave the right total.
* What its author ran (their numbers, an RTX 3090, Linux, CUDA 13.3, GCC 13.3, with the sources of 0.3.3 and 0.3.4 before the image
  tower): `selfcheck` 3/3 with `--fast off`/`on` and `--draft off`/`on`; with 0.3.4 and the defaults, `--fast auto` chose the fast path
  and `bench --draft auto` gave 126.9 tokens/s with a peak of 18.4 GB in nvidia-smi.

```sh
INK2_NVFP4_TC=1 bash src/inkling2/build_linux.sh      # RTX 50 series; without INK2_NVFP4_TC on RTX 30 and 40
export YASAKANI_INK2_DLL=$PWD/build/inkling2/libyskn_inkling2.so
yasakani selfcheck --model-dir ~/models/inkling2-mini
yasakani chat      --model-dir ~/models/inkling2-mini
```

## GPUs

By default the build makes machine code for three GPU generations in one DLL: `sm_86` (RTX 30 series), `sm_89` (RTX 40 series) and `sm_120`
(RTX 50 series). Two features need compute capability 9.0 or newer: programmatic dependent launch and the bulk L2 prefetch. The RTX 30 and
RTX 40 code is built without them, and the engine uses them only on a GPU that runs the RTX 50 code. When it opens a model, the engine checks
the GPU and stops with a message on GPUs older than compute capability 8.6 (RTX 20 series and earlier) and on GPUs the build has no code for.
AMD GPUs and Macs are not supported.

Only the RTX 5090 has been tested. The other cards run code compiled from the same source but not tuned for them. They are slower than
the RTX 5090, and we have not checked that the engine works on each of them.

To build for one generation only (a faster build and a smaller file), name it when you configure, in a new build folder:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
```

Use `120` for the RTX 50 series, `89` for the RTX 40 series or `86` for the RTX 30 series. Values below 86 are refused.

One process uses one GPU. On a PC with several GPUs, choose it with `--device N` (GPU N of `yasakani info` and nvidia-smi) or with the
`CUDA_VISIBLE_DEVICES` environment variable. With 0.1.5, `--device` could fail while the other GPUs were visible
([#20](https://github.com/yasakani-dev/yasakani/issues/20)); 0.1.6 has a fix for this, confirmed by the
reporter of #20 on a PC with two GPUs on November 18, 2026.

## Measurement builds: a plain 8-bit or an FP16 key-value cache

日本語：品質の表の「素の8bit」と「FP16」の行を測り直すための、測定専用のビルドです。ソースのフォルダーを写し、写しの `CMakeLists.txt` で `YSKN_KVX_MODE=3` を `2`（素の8bit）か `0`（FP16）に変え、別のビルドフォルダーで同じ 2 つのコマンドを実行します。配布はしていません。測り方は [tools/quality/README.md](tools/quality/README.md) にあります。

The released `yskn.dll` is built with `YSKN_KVX_MODE=3`: the keys and values are rotated and stored with KV5. The same source builds two
other key-value caches. They are there to repeat the rows of [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) that were measured with a plain
8-bit or an FP16 cache. They are not released.

| `YSKN_KVX_MODE` | Key-value cache | Option of `tools/quality/score_yasakani.py` |
|---|---|---|
| `3` (the release) | KV5 | `--kv default` |
| `2` | plain 8-bit: rotated like mode 3, one 32-bit float scale per row, no KV5 | `--kv int8` |
| `0` | FP16, not rotated | `--kv fp16` |

There is no CMake option for the mode. It is set in one line of `CMakeLists.txt`:

```cmake
target_compile_definitions(yskn PRIVATE YSKN_BUILD YSKN_KVX_MODE=3 NOMINMAX _CRT_SECURE_NO_WARNINGS)
```

Copy the source folder (`CMakeLists.txt`, `src/`, `include/` and `third_party/` are enough), so that the release configuration stays as it is.
In the copy, change `YSKN_KVX_MODE=3` in that line to `YSKN_KVX_MODE=2` or `YSKN_KVX_MODE=0`; nothing else changes. Then build the copy
like the release, in its own build folder:

```bat
cmake -S . -B build_fp16 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_fp16
```

Add `-DCMAKE_CUDA_ARCHITECTURES=120` to the first command for a faster build that runs on the RTX 50 series only.

* A library built in mode 2 or 0 does not read the KV5 table of the model folder (`kvq_u5.bin`, `YSKN_TB_PROFILE`) or `YSKN_TB_CAP`.
* Mode 2 stores 8-bit values only. Asked for a 16-bit cache, it stops with "Rotated INT8 runtime requires INT8 storage".
* Mode 0 keeps FP16 only with `YSKN_Q35_KV8=0` and `YSKN_Q35_KV_FP16=1`, which `--kv fp16` sets. With the defaults of the Python package
  (`YSKN_Q35_KV8=1`) it stores 8-bit values without the rotation, which is neither row of the table. With FP16 the attention also passes the
  queries and the softmax probabilities to the matrix units in FP16 (BF16 in the other modes).
* We have used these builds for the quality measurement only (16 tokens per pass, nothing generated), on the RTX 5090. Generation, the
  helper model, `yasakani selfcheck` and the speed have not been checked with them. Use the released `yskn.dll` for everything else.
* Up to 0.1.3, modes 2 and 0 also needed a three-line change in `src/engine35.cpp` (`#include "q35_atxpk.hpp"` moved out of the block for
  mode 3). The change is in the source since 0.1.4. The code that mode 3 compiles is the same as before it.
* Builds of the 0.1.4 source in modes 2 and 0 gave the four rows of the quality measurement to every digit as builds of the 0.1.2 source
  with that change (KL, top-1 agreement, and each of the 20 windows; on the RTX 5090, November 15, 2026).
* On Linux, make the same change in the copy and build it with the commands of [Linux](#linux-building-from-source). We have
  compiled modes 2 and 0 of the 0.1.4 source that way under WSL2 (for `sm_120`), but not run them.

## Layout

* `src/`: the engine (C++ and CUDA); `include/`: the C interface that the Python package binds to; `third_party/`: nlohmann/json (MIT).
* `src/inkling2/`: the engine for Inkling-2 Mini (`yskn_inkling2.dll`), with its own C interface and build script (see above).
* The Python package is built and installed separately (`pip install .` in its folder).
* `tools/convert/`: the program that made the released model files from the original weights (Qwen3.8-27B), and the converter for
  Inkling-2 Mini (`inkling2_convert.py`, `inkling2_quant.py`). See
  [tools/convert/README.md](tools/convert/README.md).
