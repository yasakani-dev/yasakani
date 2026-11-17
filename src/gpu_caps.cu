// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Which GPUs the engine runs on, and whether the optional sm_90+ features may be used.
//
// The release DLL carries machine code for sm_86 (RTX 30), sm_89 (RTX 40) and sm_120 (RTX 50). Programmatic dependent
// launch (griddepcontrol) and the bulk L2 prefetch (cp.async.bulk.prefetch) exist only from compute capability 9.0, so the
// sm_86 / sm_89 code is built without them (pdl.cuh, q35_kernels.cu) and the host must not ask for them there. The switch
// follows the code the GPU actually runs (the PTX version of a probe kernel), not the GPU's compute capability alone: code
// built for an older architecture and JIT-compiled for a newer GPU has no griddepcontrol either.
#include "gpu_caps.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <stdexcept>
#include <string>

namespace yskn {
namespace {
__global__ void gpu_caps_probe_kernel() {}   // never launched; only its attributes are read

// -1: no GPU opened yet; 1: every GPU opened in this process runs sm_90+ code; 0: at least one does not.
std::atomic<int> g_sm90{-1};

std::string cuda_error(cudaError_t e) { return std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e); }
}  // namespace

void select_gpu(int device) {
  cudaError_t e = cudaSetDevice(device);
  if (e != cudaSuccess) throw std::runtime_error("CUDA device " + std::to_string(device) + " cannot be used (" + cuda_error(e) + ")");
  cudaDeviceProp p{};
  e = cudaGetDeviceProperties(&p, device);
  if (e != cudaSuccess) throw std::runtime_error("CUDA device " + std::to_string(device) + " cannot be used (" + cuda_error(e) + ")");
  const std::string gpu = std::string(p.name) + " (compute capability " + std::to_string(p.major) + "." + std::to_string(p.minor) + ")";
  if (p.major * 10 + p.minor < 86)
    throw std::runtime_error("this GPU is not supported: " + gpu + ". YASAKANI needs an NVIDIA GeForce RTX 30, 40 or 50 series GPU "
                             "(compute capability 8.6 or newer).");
  cudaFuncAttributes fa{};
  e = cudaFuncGetAttributes(&fa, gpu_caps_probe_kernel);
  if (e != cudaSuccess) {
    (void)cudaGetLastError();   // clear the (non-sticky) error
    throw std::runtime_error("this build of the engine has no GPU code for " + gpu + " (" + cuda_error(e) + "). The release build runs on "
                             "compute capability 8.6, 8.9 and 12.x (RTX 30, 40 and 50 series).");
  }
  const int sm90 = fa.ptxVersion >= 90 ? 1 : 0;
  int cur = g_sm90.load();
  while (!g_sm90.compare_exchange_weak(cur, cur == 0 ? 0 : sm90)) {}
}

bool gpu_sm90_features() { return g_sm90.load(std::memory_order_relaxed) == 1; }

void gpu_code_arch(int* ptx, int* sass) {
  cudaFuncAttributes fa{};
  const cudaError_t e = cudaFuncGetAttributes(&fa, gpu_caps_probe_kernel);
  if (e != cudaSuccess) { (void)cudaGetLastError(); throw std::runtime_error("gpu_code_arch: " + cuda_error(e)); }
  if (ptx) *ptx = fa.ptxVersion;
  if (sass) *sass = fa.binaryVersion;
}
}  // namespace yskn
