# Building yskn.dll from source

The release page has a prebuilt `yskn.dll`. You can also build it yourself. You do not need a GPU to build it, only to run it.
日本語：Visual Studio 2022 Build Tools（C++）、CUDA Toolkit、CMake、Ninja を入れ、「x64 Native Tools Command Prompt for VS 2022」で下の 2 つのコマンドを実行します。

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

## Linux

There is no prebuilt Linux release: the engine is built from source, as `libyskn.so`. This is a community-tested path (contributed in
[#3](https://github.com/yasakani-dev/yasakani/pull/3)); the speed tables were measured with the Windows build only.

What a working build was made with (2026-11-14, a contributor's machine):

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS, x86-64, kernel 7.0.0, glibc 2.43 (native Linux, not WSL) |
| C / C++ compiler | GCC 15.2.0 |
| CUDA | NVIDIA CUDA Toolkit 13.3 (nvcc 13.3.73) |
| Build tools | CMake 4.2.3, Ninja 1.13.2 |
| Python | 3.13.13 for the package (in a virtual environment made with uv 0.11.6) |
| GPU / driver | GeForce RTX 5090, NVIDIA driver 615.71.09 |
| CPU | Intel Core i7-14700K (20 cores, 28 threads) |

You need GCC (or another C++17 compiler), the CUDA Toolkit with `nvcc` on the path, CMake 3.24 or newer and Ninja. Compute capability
12.0 needs a recent CUDA Toolkit. In the source folder run:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build/bin/libyskn.so` (about 26 MB). Unlike `yskn.dll`, it loads the CUDA runtime (`libcudart.so`) as a shared
library, so the machine that runs it needs that library too (it comes with the CUDA Toolkit). Install the Python package and point it at
the library with `--dll` or with the environment variable `YASAKANI_DLL`:

```sh
python3 -m venv venv
venv/bin/pip install ./python
export YASAKANI_DLL=$PWD/build/bin/libyskn.so
venv/bin/yasakani info --model-dir /models/qwen38-speed
venv/bin/yasakani verify --model-dir /models/qwen38-speed
venv/bin/yasakani selfcheck --model-dir /models/qwen38-speed
venv/bin/yasakani serve --model-dir /models/qwen38-speed
```

When the package is run from the source tree, `build/bin/libyskn.so` is also found without `--dll`.

### Image input on Linux

The optional image input ([python/README.md](python/README.md#image-input-optional)) uses the multimodal library of llama.cpp b10488. On
Windows the server takes the libraries of the b10488 release and compares them by SHA-256. That release has no matching Linux package to
compare with, so on Linux the same release is built from source, as shared libraries:

```sh
git clone --depth 1 --branch b10488 https://github.com/ggml-org/llama.cpp llama-b10488-src
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
* `-DGGML_CUDA=ON` is only needed for the projector on the GPU (below). A CUDA build was the one tested, in both modes; a build
  without it should be enough for the CPU mode, but that has not been tried. The CUDA backend (`libggml-cuda.so`) needs the CUDA
  runtime libraries (`libcudart`, `libcublas`) where the loader finds them; a backend that cannot be loaded is skipped without an
  error, and the projector then runs on the CPU.

Pass the folder as `--llama-dir`, with `--mmproj` and `--vision-model` as on Windows, and install the package with Pillow
(`pip install "./python[vision]"`):

```sh
venv/bin/yasakani serve --model-dir /models/qwen38-quality \
  --mmproj /gguf/mmproj-Qwen3.8-27B-BF16.gguf --llama-dir $PWD/llama-b10488 --vision-model /gguf/Qwen3.8-27B-Q4_K_M.gguf
```

On Linux the server checks only that `libllama.so`, `libmtmd.so`, `libggml.so` and `libggml-base.so` are in the folder; there is no
checksum, so it is up to you to build exactly b10488. A library with another structure layout is still refused when the encoder opens.

By default the projector runs on the CPU, as on Windows. `YASAKANI_VISION_GPU=1` in the environment of the server moves it to the GPU,
with flash attention switched off; the measurements behind that choice, and its memory cost, are in
[python/README.md](python/README.md#image-input-on-linux). Mind the GPU memory: the projector's work memory comes on top of the model.

## Layout

* `src/`: the engine (C++ and CUDA); `include/`: the C interface that the Python package binds to; `third_party/`: nlohmann/json (MIT).
* The Python package is built and installed separately (`pip install .` in its folder).
* `tools/convert/`: the program that made the released model files from the original weights (Qwen3.8-27B only). See
  [tools/convert/README.md](tools/convert/README.md).
