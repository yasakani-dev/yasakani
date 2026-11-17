# Building yskn.dll from source

The release page has a prebuilt `yskn.dll`. You can also build it yourself. You do not need a GPU to build it, only to run it.
On Linux the same source builds `libyskn.so`, from source only: see [Linux](#linux-building-from-source).
日本語：Visual Studio 2022 Build Tools（C++）、CUDA Toolkit、CMake、Ninja を入れ、「x64 Native Tools Command Prompt for VS 2022」で下の 2 つのコマンドを実行します。
Linux では CUDA Toolkit 13、GCC、CMake を入れ、下の「Linux (building from source)」の節の手順でソースからビルドします。寄稿者が確かめた方法で、公式の対応ではありません。ビルド済みの Linux 版はクラウドファンディングの目標の 1 つです。

## What the release was built with (2026-11-15)

| | |
|---|---|
| OS | Windows 11, x64 |
| C++ compiler | Microsoft Visual C++ 19.44.35226 (Visual Studio 2022 Build Tools, toolset 14.44.35207) |
| CUDA | NVIDIA CUDA Toolkit 13.2 (nvcc 13.2.78) |
| Build tools | CMake (3.24 or newer is required) and Ninja |
| Target | one DLL with machine code for compute capability 8.6 (`sm_86`, RTX 30 series), 8.9 (`sm_89`, RTX 40 series) and 12.0 (`sm_120`, RTX 50 series), plus PTX for `compute_120`. Tested on the RTX 5090 only |

The `yskn.dll` of the test build v0.1.7-test1 is this build (SHA-256 `6bfb18a8da8f905bd22fdd300095d9d8629537a26777c4ba4b40772728c40c52`).
It was made the same way from the build of 0.1.6 (`967de3ea…be8f`): the files with the host-code changes of 0.1.7 and the files that
include `include/yskn_abi.h` were compiled again and linked. Its GPU code is that of 0.1.6 (compared with `cuobjdump`: the machine code
of all three architectures and the PTX are the same). The `yskn.dll` of 0.1.6 is the
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
measured with the 0.1.4 build. Other compiler or CUDA
versions have not been tried. Compute capability 12.0 needs a recent CUDA Toolkit.

## Steps

Open the **x64 Native Tools Command Prompt for VS 2022** (so that the compiler is on the path), go to the source folder (the one with
`CMakeLists.txt`) and run:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build\yskn.dll` (about 58 MB; the CUDA runtime is linked statically, so no CUDA Toolkit is needed to run it). The recorded build of
the release took about four minutes and printed many compiler warnings; they are known.

Pass the file to the tools with `--dll`:

```bat
yasakani info --model-dir D:\models\qwen38 --dll build\yskn.dll
```

`yasakani info` prints the SHA-256 of the DLL. A build made with a different compiler or CUDA version will not be byte-identical to the file on the
release page; that is expected. Check the model files with `yasakani verify`, and the whole installation with `yasakani selfcheck`.

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
RTX 5090 (November 15, 2026). Other distributions, compilers and CUDA versions have not been tried.

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

The optional image input ([python/README.md](python/README.md#image-input-optional)) uses the multimodal library of llama.cpp b10488. On
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
RTX 5090 (the table above). We have not run image input on Linux ourselves, and it is not part of an official release.

By default the projector runs on the CPU, as on Windows. `YASAKANI_VISION_GPU=1` in the environment of the server moves it to the GPU,
with flash attention switched off. This is experimental and off by default: the image test was not repeated in this mode, and we have not
tried it. The measurements behind that choice, and its memory cost, are in [python/README.md](python/README.md#image-input-on-linux).
Mind the GPU memory: the projector and its work memory stay on the GPU next to the model, and the memory of an encode grows with the
square of the image tokens, so with the variable set `--vision-max-image-tokens` can be at most 2,048.

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
([#20](https://github.com/yasakani-dev/yasakani/issues/20)); 0.1.6 has a fix for this, not yet tried on a PC with two GPUs.

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
* The Python package is built and installed separately (`pip install .` in its folder).
* `tools/convert/`: the program that made the released model files from the original weights (Qwen3.8-27B only). See
  [tools/convert/README.md](tools/convert/README.md).
