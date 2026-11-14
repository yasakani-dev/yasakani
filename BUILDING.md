# Building yskn.dll from source

The release page has a prebuilt `yskn.dll`. You can also build it yourself. You do not need a GPU to build it, only to run it.
On Linux the same source builds `libyskn.so`: see [Linux](#linux-building-from-source).
日本語：Visual Studio 2022 Build Tools（C++）、CUDA Toolkit、CMake、Ninja を入れ、「x64 Native Tools Command Prompt for VS 2022」で下の 2 つのコマンドを実行します。
Linux（WSL2・Ubuntu 24.04 で確認）では、CUDA Toolkit、GCC、CMake を入れ、下の「Linux (building from source)」の節の手順でソースからビルドします。

## What the release was built with (2026-11-07)

| | |
|---|---|
| OS | Windows 11, x64 |
| C++ compiler | Microsoft Visual C++ 19.44.35226 (Visual Studio 2022 Build Tools, toolset 14.44.35207) |
| CUDA | NVIDIA CUDA Toolkit 13.2 (nvcc 13.2.78) |
| Build tools | CMake (3.24 or newer is required) and Ninja |
| Target | compute capability 12.0 (`sm_120`), which the whole RTX 50 series shares. Tested on the RTX 5090 only |

The `yskn.dll` on the release page is this build (SHA-256 `0f5d973b791c8d66034d87bdfc7f2c018718dc0e851b9a6ee6dc1eb275150bbb`). It was built
from this source with the two commands below (the CUDA compiler was named explicitly with `-DCMAKE_CUDA_COMPILER`), and it is the build
the YASAKANI speed tables in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) were measured with. Other compiler or CUDA versions have not been
tried. Compute capability 12.0 needs a recent CUDA Toolkit.

## Steps

Open the **x64 Native Tools Command Prompt for VS 2022** (so that the compiler is on the path), go to the source folder (the one with
`CMakeLists.txt`) and run:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build\yskn.dll` (about 26 MB; the CUDA runtime is linked statically, so no CUDA Toolkit is needed to run it). The recorded build of
the release took about a minute and a half and printed many compiler warnings; they are known.

Pass the file to the tools with `--dll`:

```bat
yasakani info --model-dir D:\models\qwen38 --dll build\yskn.dll
```

`yasakani info` prints the SHA-256 of the DLL. A build made with a different compiler or CUDA version will not be byte-identical to the file on the
release page; that is expected. Check the model files with `yasakani verify`, and the whole installation with `yasakani selfcheck`.

## Linux (building from source)

The engine also builds on Linux, as `libyskn.so` (contributed in [#3](https://github.com/yasakani-dev/yasakani/pull/3)). This is building from
source only: there is no prebuilt Linux release or launcher, and the quickstart (`START.cmd`) and the optional image input are for Windows.
A prebuilt, officially supported Linux release is one of the project's funding goals.

### What it was tested with (2026-11-14)

| | |
|---|---|
| OS | Ubuntu 24.04 under WSL2 on Windows 11. Native Linux (outside WSL2) has not been tested yet |
| GPU and driver | RTX 5090. Under WSL2 the driver is the Windows NVIDIA driver (tested with 621.24) |
| C++ compiler | GCC 13.3 (the Ubuntu 24.04 default) |
| CUDA | NVIDIA CUDA Toolkit 13.1, 13.3 (nvcc 13.3.73) and 13.5 (nvcc 13.5.37) |
| Build tools | CMake 3.28 (3.24 or newer is required) with its default generator, Unix Makefiles |
| Python | 3.12 (3.10 or newer is required) |

Each of these builds passed `yasakani selfcheck` with the fastest model, and the CUDA 13.5 build with the quality model too. The CUDA 13.5
build followed this section as written, in a new `nvidia/cuda:13.5.0-devel-ubuntu24.04` container (which has the toolkit) with only
`build-essential cmake git python3-venv` added. Other distributions, compilers and CUDA versions have not been tried.

### Before you build

* Install the CUDA Toolkit from NVIDIA's package repository for Ubuntu ([CUDA downloads](https://developer.nvidia.com/cuda-downloads): set up
  the repository, then `sudo apt install cuda-toolkit`, or `cuda-toolkit-13-3` and so on for a given version). Ubuntu's own
  `nvidia-cuda-toolkit` package is CUDA 12.0 on 24.04, too old for compute capability 12.0. Then the other tools:
  `sudo apt install build-essential cmake git python3-venv`.
* Under WSL2, install the toolkit only. The Windows driver is mapped into WSL, and NVIDIA's
  [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) says not to install a Linux driver there (no `cuda-drivers`
  or `nvidia-open` package). `nvidia-smi` in the WSL terminal should list the RTX 5090.
* Keep the source and the model folder on the Linux file system (for example under `~`), not under `/mnt/c`; files there are slower
  to reach from Linux ([Microsoft's notes](https://learn.microsoft.com/en-us/windows/wsl/filesystems)).

### Steps

Get the source and build:

```sh
git clone https://github.com/yasakani-dev/yasakani
cd yasakani
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j
```

`-DCMAKE_CUDA_COMPILER` names the CUDA compiler, which NVIDIA's packages install in `/usr/local/cuda/bin`, a folder that is often not on the
path. The result is `build/bin/libyskn.so` (about 26 MB). The build uses the same CUDA flags as the Windows build (`-O3 --fmad=false`) and
prints some compiler warnings that do not stop it. With `-j` (all compile jobs at once) it took a minute and a quarter on a 16-core CPU and
used at most about 4 GB of memory.

### Running it

Install the Python package in a virtual environment, download a model (here the quality model; `hf` comes with the package's dependencies)
and pass the library with `--dll`:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install ./python
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-quality --local-dir ~/models/qwen38-quality
yasakani info      --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so
yasakani verify    --model-dir ~/models/qwen38-quality
yasakani selfcheck --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so
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

## Layout

* `src/`: the engine (C++ and CUDA); `include/`: the C interface that the Python package binds to; `third_party/`: nlohmann/json (MIT).
* The Python package is built and installed separately (`pip install .` in its folder).
* `tools/convert/`: the program that made the released model files from the original weights (Qwen3.8-27B only). See
  [tools/convert/README.md](tools/convert/README.md).
