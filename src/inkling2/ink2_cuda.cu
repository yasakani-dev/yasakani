// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// CUDA backend: the kernels of ink2_kernels.inc as CUDA kernels.
// Build with -fmad=false: plain a * b + c stays unfused, KFMA is the fused form.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <string>

#include "ink2_kargs.h"
#include "ink2_fast.h"  // fast kernels

// Programmatic dependent launch (fast kernels, sm_90 and later): every kernel first waits for the kernel before it
// in the stream (griddepcontrol.wait: a no-op when it was not launched as a dependent), then lets the next one be
// scheduled (griddepcontrol.launch_dependents). The arithmetic of the kernels is unchanged. The backend launches with
// the attribute only when the engine asks for it (fast path, option "pdl").
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define KPDL()                                             \
  do {                                                     \
    asm volatile("griddepcontrol.wait;" ::: "memory");     \
    asm volatile("griddepcontrol.launch_dependents;" :::); \
  } while (0)
#else
#define KPDL() \
  do {         \
  } while (0)
#endif
#define KERNEL(name, A)                                  \
  __device__ __forceinline__ void kb_##name(const A& a); \
  __global__ void kk_##name(A a) {                       \
    KPDL();                                              \
    kb_##name(a);                                        \
  }                                                      \
  __device__ __forceinline__ void kb_##name(const A& a)
// the same with a bound on the block size and the blocks per SM it must fit (registers); the same kb_/PDL split
#define KERNELB(name, A, nthr, nblk)                             \
  __device__ __forceinline__ void kb_##name(const A& a);         \
  __global__ void __launch_bounds__(nthr, nblk) kk_##name(A a) { \
    KPDL();                                                      \
    kb_##name(a);                                                \
  }                                                              \
  __device__ __forceinline__ void kb_##name(const A& a)
#define KINL __device__ __forceinline__
#define KCTXP
#define KCTXA
#define KTID ((int)threadIdx.x)
#define KBID ((int)blockIdx.x)
#define KNT ((int)blockDim.x)
#define KSYNC() __syncthreads()
#define KSHFL_DOWN(v, d) __shfl_down_sync(0xffffffffu, (v), (d))
#define KSHFL(v, src) __shfl_sync(0xffffffffu, (v), (src))
#define KSHARED(T, name, n) __shared__ T name[n]
#define KLOAD64(p) (*(const unsigned long long*)(p))
#define KLOAD32(p) (*(const unsigned*)(p))
#define KF4 float4
#define KLOADF4(p) (*(const float4*)(p))
#define KF4ZERO make_float4(0.0f, 0.0f, 0.0f, 0.0f)
#define KFMA(a, b, c) fmaf((a), (b), (c))
#define KBITS2F(u) __uint_as_float(u)
#define KUNROLL _Pragma("unroll")
#define KNOUNROLL _Pragma("unroll 1")

#include "ink2_kernels.inc"
#include "ink2_packed_prefill.cuh"
#include "ink2_decode_experts.cuh"
#include "ink2_flash_tc.cuh"

#ifdef INK2_WITH_NVFP4_TC
bool ink2_nvfp4_available();
cudaError_t ink2_nvfp4_experts(const ExpMArgs& a, bool gate_up, cudaStream_t stream);
#endif
#include "ink2_fast2_gen.inc"  // fast kernels, second set

namespace {

struct CudaBackend : Backend {
  size_t used = 0;
  bool failed = false;
  int multiprocessors = 0;
  size_t draft_l2_bytes = 0;
  bool set_draft_l2(int mib) override {
    if (!mib) {
      draft_l2_bytes = 0;
      return true;
    }
    int dev = 0, window = 0;
    size_t reserved = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&window, cudaDevAttrMaxAccessPolicyWindowSize, dev) != cudaSuccess ||
        cudaDeviceGetLimit(&reserved, cudaLimitPersistingL2CacheSize) != cudaSuccess ||
        window <= 0 || !reserved) return false;
    size_t bytes = static_cast<size_t>(mib) << 20;
    if (reserved < bytes) bytes = reserved;
    if (static_cast<size_t>(window) < bytes) bytes = static_cast<size_t>(window);
    draft_l2_bytes = bytes;
    return true;
  }
  bool packed_prompt = false;
  bool set_packed_prompt(bool enabled) override {
    if (enabled && !multiprocessors) {
      int dev = 0;
      if (cudaGetDevice(&dev) != cudaSuccess ||
          cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return false;
    }
    packed_prompt = enabled;
    return true;
  }
  std::string err;
  cudaStream_t stream = nullptr;  // every kernel runs here; copies use the default stream, which waits for it
  void* native_stream() override { return stream; }  // the image tower (ink2_vision.cu) queues its work here too

  void fail(const char* what, cudaError_t e) {
    if (failed) return;
    failed = true;
    char msg[300];
    snprintf(msg, sizeof msg, "cuda backend: %s: %s", what, cudaGetErrorString(e));
    err = msg;
  }
  const char* name() override { return "cuda"; }
  int sm_count() override {
    int device = 0, n = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
      cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }
  void* alloc(size_t bytes) override {
    size_t n = (bytes + 255) & ~(size_t)255;
    if (n == 0) n = 256;
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, n);
    if (e != cudaSuccess) {
      fail("cudaMalloc", e);
      return nullptr;
    }
    e = cudaMemset(p, 0, n);
    if (e != cudaSuccess) fail("cudaMemset", e);
    used += n;
    return p;
  }
  void release(void* p) override { cudaFree(p); }
  void upload(void* dst, const void* src, size_t bytes) override {
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) fail("cudaMemcpy to device", e);
  }
  void download(void* dst, const void* src, size_t bytes) override {
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) fail("cudaMemcpy to host", e);
  }
  void zero(void* dst, size_t bytes) override {
    cudaError_t e = cudaMemset(dst, 0, bytes);
    if (e != cudaSuccess) fail("cudaMemset", e);
  }
  void sync() override {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fail("cudaDeviceSynchronize", e);
  }
  void run(int kid, const void* args, int grid, int block) override {
    if (failed) return;
    if (kid == KID_dfold && draft_l2_bytes) {
      const auto& a = *static_cast<const DFoldArgs*>(args);
      const void* weights = a.fp ? static_cast<const void*>(a.fp) : a.f16 ? static_cast<const void*>(a.f16) : a.f32;
      const int bits = a.fp ? (a.fpfmt == INK2_DFMT_Q3 ? 3 : a.fpfmt == INK2_DFMT_NVFP4 ? 4 : 8) : a.f16 ? 16 : 32;
      const size_t bytes = static_cast<size_t>(a.V) * a.rank * bits / 8;
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeAccessPolicyWindow;
      auto& policy = attr.val.accessPolicyWindow;
      policy.base_ptr = const_cast<void*>(weights);
      policy.num_bytes = bytes < draft_l2_bytes ? bytes : draft_l2_bytes;
      policy.hitRatio = 1.0f;
      policy.hitProp = cudaAccessPropertyPersisting;
      policy.missProp = cudaAccessPropertyNormal;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(grid);
      cfg.blockDim = dim3(block);
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      cudaError_t error = cudaLaunchKernelEx(&cfg, kk_dfold, a);
      if (error != cudaSuccess) fail("drafter cache hint", error);
      return;
    }
    if (packed_prompt && (kid == KID_pgemvcm || kid == KID_pgemvcm8 || kid == KID_pgemvcm4 || kid == KID_pgemvcm2)) {
      auto a = *static_cast<const GemvArgs*>(args);
      if (packed_prompt_shape(a, block)) {
        int groups = a.nrhs / prompt_in;
        a.blocks = a.total_rows / prompt_out;
        if (a.blocks > 0) {
          int tiles = (2 * multiprocessors + groups - 1) / groups;
          if (tiles > a.blocks) tiles = a.blocks;
          prompt_packed<<<tiles * groups, block, 0, stream>>>(a);
          cudaError_t error = cudaGetLastError();
          if (error != cudaSuccess) fail("packed prompt product", error);
          return;
        }
      }
    }
    if (kid == KID_exp2m || kid == KID_exp2s) {
      const auto& a = *static_cast<const ExpMArgs*>(args);
      if (a.register_down && a.fmt == INK2_FMT_NVFP4 && a.cols == 1152 && a.rows % 16 == 0) {
        decode_expert_down<<<dim3(a.rows / 16, a.expert_slots), 256, 0, stream>>>(a);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) fail("short expert down projection", err);
        return;
      }
    }
#ifdef INK2_WITH_NVFP4_TC
    if (kid == KID_exp13cm || kid == KID_exp2m) {
      const auto& a = *static_cast<const ExpMArgs*>(args);
      if (a.nvfp4_planes && a.fmt == INK2_FMT_NVFP4 && a.cols % 64 == 0 && a.rows % 8 == 0) {
        cudaError_t err = ink2_nvfp4_experts(a, kid == KID_exp13cm, stream);
        if (err != cudaSuccess) fail("NVFP4 expert product", err);
        return;
      }
    }
#endif
    if (kid == KID_attn_flash && static_cast<const AttnArgs*>(args)->flash_tc) {
      if (!ink2_flash_tc_launch(*static_cast<const AttnArgs*>(args), grid, stream)) fail("flash attention", cudaErrorInvalidValue);
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess) fail("flash attention (tensor cores)", error);
      return;
    }
    if (pdl && f2on && ink2_f2_run(kid, args, grid, block, stream, pdl, f2)) {  // fast kernels, second set
      cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) fail("kernel launch (second set)", e);
      return;
    }
    switch (kid) {
#define X(name, A)                                   \
  case KID_##name:                                   \
    launch(kk_##name, *(const A*)args, grid, block); \
    break;
      INK2_KERNEL_LIST(X)
#undef X
      default:
        break;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fail("kernel launch", e);
  }
  int max_block(int kid) override {
    static int cache[KID_COUNT] = {};
    if (kid < 0 || kid >= KID_COUNT) return 0;
    if (cache[kid] == 0) {
      cudaFuncAttributes fa{};
      cudaError_t e = cudaErrorInvalidValue;
      switch (kid) {
#define X(name, A)                                   \
  case KID_##name:                                   \
    e = cudaFuncGetAttributes(&fa, kk_##name);       \
    break;
        INK2_KERNEL_LIST(X)
#undef X
        default:
          break;
      }
      cache[kid] = e == cudaSuccess ? fa.maxThreadsPerBlock : 256;
    }
    return cache[kid];
  }
  size_t bytes_allocated() override { return used; }
  bool device_memory(long long* free_bytes, long long* total_bytes) override {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
    *free_bytes = (long long)f;
    *total_bytes = (long long)t;
    return true;
  }
  const char* error() override { return failed ? err.c_str() : nullptr; }

  bool graph_supported() override { return true; }
#ifdef INK2_WITH_NVFP4_TC
  bool supports_nvfp4_prefill() override { return ink2_nvfp4_available(); }
#endif
  void graph_begin() override {
    cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) fail("cudaStreamBeginCapture", e);
  }
  void* graph_end() override {
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(stream, &graph);
    if (e != cudaSuccess) {
      fail("cudaStreamEndCapture", e);
      return nullptr;
    }
    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) {
      fail("cudaGraphInstantiate", e);
      return nullptr;
    }
    return exec;
  }
  void graph_launch(void* g) override {
    cudaError_t e = cudaGraphLaunch((cudaGraphExec_t)g, stream);
    if (e != cudaSuccess) fail("cudaGraphLaunch", e);
  }
  void graph_free(void* g) override {
    if (g) cudaGraphExecDestroy((cudaGraphExec_t)g);
  }
  // fast kernels (ink2_fast.cu)
  bool pdl = false;  // launch with programmatic stream serialization (sm_90 and later)
  template <class A>
  void launch(void (*k)(A), const A& a, int grid, int block) {
    if (!pdl) {
      k<<<grid, block, 0, stream>>>(a);
      return;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3((unsigned)block);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, k, a);
  }
  // Only when the code that runs was compiled for sm_90 or later (it then has the griddepcontrol prologue): a library
  // with PTX for an older target compiled at load time on a new card must not be launched as dependents.
  bool set_pdl(bool on) override {
    int dev = 0, maj = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    cudaFuncAttributes fa{};
    bool code90 = cudaFuncGetAttributes(&fa, kk_norm) == cudaSuccess && fa.ptxVersion >= 90 && ink2_fast_ptx_version() >= 90;
    cudaGetLastError();
    pdl = on && maj >= 9 && code90;
    return pdl;
  }
  int fast_cc = -1;
  int fast_level() override {
    if (fast_cc < 0) fast_cc = ink2_fast_probe();
    return fast_cc;
  }
  void run_fast(int fid, const void* args, int grid, int block, int smem) override {
    if (failed) return;
    int rc = ink2_fast_run(fid, args, grid, block, smem, stream, pdl);
    if (rc) fail("fast kernel launch", (cudaError_t)rc);
  }
  int fast_smem() override { return fast_level() ? ink2_fast_smem_max() : 0; }
  // fast kernels, second set (ink2_fast2_gen.inc)
  F2Opts f2 = {};
  bool f2on = false;
  void set_f2(const F2Opts* o) override {
    f2 = o ? *o : F2Opts{};
    f2on = o != nullptr;
    ink2_fast_set_f2(o);
  }
  int run_f2(int fid, const void* args, int grid, int block) override {
    if (failed) return 0;
    int rc = ink2_f2_run_own(fid, args, grid, block, stream, pdl, f2);
    if (rc) fail("kernel launch (second set)", (cudaError_t)rc);
    return rc;
  }
  int f2_gemvm_block() override { return ink2_f2_gemvm_max_block(); }
};

}  // namespace

Backend* ink2_make_cuda_backend(char* errbuf, int err_cap) {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count < 1) {
    snprintf(errbuf, err_cap, "cuda backend: no device (%s)", cudaGetErrorString(e));
    return nullptr;
  }
  e = cudaSetDevice(0);
  if (e == cudaSuccess) e = cudaFree(0);  // creates the context
  if (e != cudaSuccess) {
    snprintf(errbuf, err_cap, "cuda backend: cannot start the device (%s)", cudaGetErrorString(e));
    return nullptr;
  }
  CudaBackend* b = new CudaBackend();
  e = cudaStreamCreate(&b->stream);
  if (e != cudaSuccess) {
    snprintf(errbuf, err_cap, "cuda backend: cudaStreamCreate (%s)", cudaGetErrorString(e));
    delete b;
    return nullptr;
  }
  return b;
}

int ink2_cuda_nvfp4_prompt() {
#ifdef INK2_WITH_NVFP4_TC
  return ink2_nvfp4_available() ? 1 : 0;
#else
  return 0;
#endif
}
