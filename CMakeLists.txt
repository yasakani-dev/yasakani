cmake_minimum_required(VERSION 3.24)
project(yasakani_engine LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# sm_86: the target is a cache variable. 120 (the RTX 50 series) is the default; 86 (RTX 3090, Ampere) is experimental.
# cmake -DYSKN_CUDA_ARCH=86 ...
set(YSKN_CUDA_ARCH "120" CACHE STRING "CUDA compute capability to build for: 120 (default) or 86 (RTX 3090, experimental)")
set(CMAKE_CUDA_ARCHITECTURES ${YSKN_CUDA_ARCH})
find_package(CUDAToolkit REQUIRED)

add_library(yskn SHARED
  src/attention_gqa.cu
  src/attention_kv4q.cu
  src/c_api.cpp
  src/devmem.cpp
  src/draft_kernels.cu
  src/engine.cpp
  src/engine35.cpp
  src/gemm_compressed.cu
  src/gemv_sigma_int.cu
  src/kvx_probe.cu
  src/lnord.cu
  src/loader.cpp
  src/misc_kernels.cu
  src/pfa.cu
  src/pm_test.cu
  src/post_kernels.cu
  src/q35_atx.cu
  src/q35_atxpk.cu
  src/q35_bdraft.cu
  src/q35_chk.cu
  src/q35_fa.cu
  src/q35_flow.cu
  src/q35_g6.cu
  src/q35_kernels.cu
  src/q35_kz.cu
  src/q35_prefill.cu
  src/q35_sample.cu
  src/repack.cu
  src/specx.cpp
  src/test_hooks.cpp
)
target_include_directories(yskn PRIVATE src include third_party)
target_compile_definitions(yskn PRIVATE YSKN_BUILD YSKN_KVX_MODE=3 NOMINMAX _CRT_SECURE_NO_WARNINGS YSKN_CUDA_ARCH=${YSKN_CUDA_ARCH})
if(MSVC)
  target_compile_options(yskn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj;/O2;/d1trimfile:${CMAKE_SOURCE_DIR}/>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8;-Xcompiler=/bigobj;-Xcompiler=/d1trimfile:${CMAKE_SOURCE_DIR}/;-O3;--fmad=false>)
endif()
target_link_libraries(yskn PRIVATE CUDA::cudart)
