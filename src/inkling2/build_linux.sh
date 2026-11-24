#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
# Builds libyskn_inkling2.so (Linux, CUDA backend only) with the flags of build.cmd.
#   INK2_ARCHS     nvcc -gencode list (default: sm_86, sm_89, sm_120 + PTX)
#   INK2_NVFP4_TC  1 = also build the RTX 50-only NVFP4 tensor-core prompt products
#                  (ink2_nvfp4.cu, sm_120a), as build.cmd does by default
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../build/inkling2
mkdir -p "$OUT/obj"
rm -f "$OUT"/obj/*.o

ARCHS=${INK2_ARCHS:-"-gencode arch=compute_86,code=sm_86
  -gencode arch=compute_89,code=sm_89
  -gencode arch=compute_120,code=[sm_120,compute_120]"}
CX="-std=c++17 -O2 -fPIC -fvisibility=hidden -ffp-contract=off"  # engine defines INK2_BUILD
# (build.cmd also passes /openmp and /arch:AVX2; they serve ink2_emu.cpp, which this build leaves out)
NVX="-std=c++17 -Xcompiler -fPIC,-fvisibility=hidden"
CUDA_INC="${CUDA_HOME:-/usr/local/cuda}/include"

TC=""
if [ "${INK2_NVFP4_TC:-0}" = 1 ]; then
  nvcc -c -O3 -gencode arch=compute_120a,code=sm_120a -fmad=false $NVX \
       ink2_nvfp4.cu -o "$OUT/obj/ink2_nvfp4.o"
  TC="-DINK2_WITH_NVFP4_TC"
fi

nvcc -c -O3 $ARCHS -fmad=false $NVX $TC ink2_cuda.cu  -o "$OUT/obj/ink2_cuda.o"
nvcc -c -O3 $ARCHS -fmad=false $NVX     ink2_micro.cu -o "$OUT/obj/ink2_micro.o"
# the image encoder (ink2_vision.cu, the image tower); the engine reaches it with INK2_WITH_VISION, as build.cmd
nvcc -c -O3 $ARCHS -fmad=false $NVX $TC ink2_vision.cu -o "$OUT/obj/ink2_vision.o"
nvcc -c -O3 $ARCHS $NVX ink2_fast.cu -o "$OUT/obj/ink2_fast.o"  # default -fmad, as build.cmd

g++ $CX -DINK2_WITH_CUDA -DINK2_WITH_VISION -I"$CUDA_INC" -c ink2_engine.cpp -o "$OUT/obj/ink2_engine.o"
g++ $CX -c ink2_selftest.cpp        -o "$OUT/obj/ink2_selftest.o"
g++ $CX -c ink2_host_stub_linux.cpp -o "$OUT/obj/ink2_host_stub_linux.o"

nvcc -shared $ARCHS -cudart static -o "$OUT/libyskn_inkling2.so" "$OUT"/obj/*.o \
     -lpthread -ldl -lrt
echo "built $OUT/libyskn_inkling2.so"
