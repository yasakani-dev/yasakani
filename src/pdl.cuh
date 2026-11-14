// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
// Modified in the unofficial RTX 3090 (sm_86) fork: griddepcontrol only for sm_90+ (see FORK_RTX3090.md).





#include <cuda_runtime.h>
#include <utility>
namespace yskn {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
#else
__device__ __forceinline__ void pdl_wait() {}
__device__ __forceinline__ void pdl_trigger() {}
#endif
bool pdl_enabled();
template <typename... KArgs, typename... Args>
inline void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  if (!pdl_enabled()) { kern<<<grid, block, smem, st>>>(args...); return; }
  cudaLaunchConfig_t cfg{}; cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}
} 
