cmake_minimum_required(VERSION 3.24)
# GPU code for RTX 30 (sm_86), RTX 40 (sm_89) and RTX 50 (sm_120) in one library. PTX is kept for compute_120 only, so that
# newer 12.x GPUs can compile it at load time. To build one architecture, pass e.g. -DCMAKE_CUDA_ARCHITECTURES=120 (or 86, 89).
set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real;120" CACHE STRING "CUDA architectures to build (default: RTX 30, 40 and 50 series)")
project(yasakani_engine LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 86)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES contains ${arch}: the engine needs compute capability 8.6 or newer (86, 89, 120).")
  endif()
endforeach()
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
  src/gpu_caps.cu
  src/kvx_probe.cu
  src/lnord.cu
  src/loader.cpp
  src/misc_kernels.cu
  src/pfa.cu
  src/pm_test.cu
  src/post_kernels.cu
  src/q35_atx.cu
  src/q35_atxpk.cu
  src/q35_atxpk_ms.cu
  src/q35_bdraft.cu
  src/q35_chk.cu
  src/q35_fa.cu
  src/q35_flow.cu
  src/q35_g6.cu
  src/q35_kernels.cu
  src/q35_kz.cu
  src/q35_prefill.cu
  src/q35_sample.cu
  src/q35_sample_ms.cu
  src/repack.cu
  src/specx.cpp
  src/test_hooks.cpp
)
target_include_directories(yskn PRIVATE src include third_party)
target_compile_definitions(yskn PRIVATE YSKN_BUILD YSKN_KVX_MODE=3 NOMINMAX _CRT_SECURE_NO_WARNINGS)
if(MSVC)
  target_compile_options(yskn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj;/O2;/d1trimfile:${CMAKE_SOURCE_DIR}/>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8;-Xcompiler=/bigobj;-Xcompiler=/d1trimfile:${CMAKE_SOURCE_DIR}/;-O3;--fmad=false>)
else()
  # Same CUDA code generation as the MSVC build: --fmad=false keeps the output identical to the reference.
  target_compile_options(yskn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>)
  # build/bin/libyskn.so is one of the places the Python package looks, so --dll is not needed.
  set_target_properties(yskn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
target_link_libraries(yskn PRIVATE CUDA::cudart)
# The Inkling-2 engine contributed in #28 (src/engine_ink2.cpp, src/ink2_kernels.cu): Linux only, off by default. With it the library
# needs cuBLAS (libcublas.so of the CUDA Toolkit) at run time, for every model. The released Windows yskn.dll is built without it.
option(YSKN_WITH_INK2 "Build the Inkling-2 engine of #28 into the library (Linux; needs cuBLAS at run time)" OFF)
if(YSKN_WITH_INK2)
  if(WIN32)
    message(FATAL_ERROR "YSKN_WITH_INK2 builds on Linux only (src/safetensors.hpp uses POSIX file calls). On Windows the Inkling-2 "
                        "engine is yskn_inkling2.dll (BUILDING.md).")
  endif()
  target_sources(yskn PRIVATE src/engine_ink2.cpp src/ink2_kernels.cu src/ink2_vision.cu)
  target_compile_definitions(yskn PRIVATE YSKN_WITH_INK2=1)
  target_link_libraries(yskn PRIVATE CUDA::cublas)
endif()
