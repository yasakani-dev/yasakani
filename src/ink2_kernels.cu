// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "ink2_kernels.cuh"

#include <cuda_fp8.h>
#if defined(__has_include)
#if __has_include(<cuda_fp4.h>)
#define YSKN_HAS_CUDA_FP4_H 1
#include <cuda_fp4.h>
#endif
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#define YSKN_HAS_CUDA_FP4_H 1
#include <cuda_fp4.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace yskn::ink2 {
namespace {

__device__ __forceinline__ float ink2_e4m3(uint8_t b) {
  const uint16_t h = uint16_t(((uint32_t(b) & 0x80u) << 8) | ((uint32_t(b) & 0x7Fu) << 7));
  return __half2float(__ushort_as_half(h)) * 256.0f;
}

__device__ __forceinline__ float2 ink2_e4m3x2(uint16_t u) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&r));
#else
  return make_float2(ink2_e4m3(uint8_t(u & 0xFFu)), ink2_e4m3(uint8_t(u >> 8)));
#endif
}

__device__ __forceinline__ float ink2_ue8m0(uint8_t b) {
  const uint32_t bits = (b == 0u) ? 0x00400000u : (uint32_t(b) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ float e2m1_bit(uint32_t nib) {
  const uint32_t n = nib & 7u;
  const uint32_t bits = (n < 2u) ? (n ? 0x3f000000u : 0u) : ((n << 22) + 0x3f000000u);
  return __uint_as_float(bits | ((nib & 8u) << 28));
}

__device__ __forceinline__ void sf4_decode16_h2(uint32_t w0, uint32_t w1, __half2 (&w)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080) && \
    defined(YSKN_HAS_CUDA_FP4_H)
  auto c2 = [](uint8_t b) -> __half2 {
    const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(b), __NV_E2M1);
    return *reinterpret_cast<const __half2*>(&r);
  };
#else
  auto c2 = [](uint8_t b) -> __half2 {
    return __floats2half2_rn(e2m1_bit(b & 0xFu), e2m1_bit(b >> 4));
  };
#endif
  w[0] = c2(uint8_t(w0 & 0xFFu));
  w[1] = c2(uint8_t((w0 >> 8) & 0xFFu));
  w[2] = c2(uint8_t((w0 >> 16) & 0xFFu));
  w[3] = c2(uint8_t(w0 >> 24));
  w[4] = c2(uint8_t(w1 & 0xFFu));
  w[5] = c2(uint8_t((w1 >> 8) & 0xFFu));
  w[6] = c2(uint8_t((w1 >> 16) & 0xFFu));
  w[7] = c2(uint8_t(w1 >> 24));
}

__device__ __forceinline__ void mx8_decode16_h2(const uint4& wv, __half2 (&w)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  auto c2 = [](uint16_t u) -> __half2 {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u), __NV_E4M3);
    return *reinterpret_cast<const __half2*>(&r);
  };
#else
  auto c2 = [](uint16_t u) -> __half2 {
    return __floats2half2_rn(ink2_e4m3(uint8_t(u & 0xFFu)), ink2_e4m3(uint8_t(u >> 8)));
  };
#endif
  w[0] = c2(uint16_t(wv.x & 0xFFFFu));
  w[1] = c2(uint16_t(wv.x >> 16));
  w[2] = c2(uint16_t(wv.y & 0xFFFFu));
  w[3] = c2(uint16_t(wv.y >> 16));
  w[4] = c2(uint16_t(wv.z & 0xFFFFu));
  w[5] = c2(uint16_t(wv.z >> 16));
  w[6] = c2(uint16_t(wv.w & 0xFFFFu));
  w[7] = c2(uint16_t(wv.w >> 16));
}

__device__ __forceinline__ float dot16_h2_2acc(const __half2 (&w)[8], const __half2* xh2) {
  const uint4 v0 = *reinterpret_cast<const uint4*>(xh2);
  const uint4 v1 = *reinterpret_cast<const uint4*>(xh2 + 4);
  const __half2* x0 = reinterpret_cast<const __half2*>(&v0);
  const __half2* x1 = reinterpret_cast<const __half2*>(&v1);
  __half2 a0 = __hmul2(w[0], x0[0]);
  __half2 a1 = __hmul2(w[1], x0[1]);
  a0 = __hfma2(w[2], x0[2], a0);
  a1 = __hfma2(w[3], x0[3], a1);
  a0 = __hfma2(w[4], x1[0], a0);
  a1 = __hfma2(w[5], x1[1], a1);
  a0 = __hfma2(w[6], x1[2], a0);
  a1 = __hfma2(w[7], x1[3], a1);
  const float2 f = __half22float2(__hadd2(a0, a1));
  return f.x + f.y;
}

__device__ __forceinline__ float situ_f32(float g, float u) {
  return (4.0f * tanhf(g * 0.25f)) * (1.0f / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
}

__device__ __forceinline__ float ink2_bf16_lo(uint32_t u) {
  return __uint_as_float(u << 16);
}

__device__ __forceinline__ float ink2_bf16_hi(uint32_t u) {
  return __uint_as_float(u & 0xFFFF0000u);
}

__device__ __forceinline__ uint32_t cvt_smem_ptr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* ptr) {
  const uint32_t addr = cvt_smem_ptr(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* ptr) {
  const uint32_t addr = cvt_smem_ptr(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4],
                                                   const uint32_t (&b)[2], const float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                 "r"(b[0]), "r"(b[1]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ uint32_t pack_bf2_f32(float a, float b) {
  const __nv_bfloat162 v = __float22bfloat162_rn(make_float2(a, b));
  uint32_t u;
  memcpy(&u, &v, sizeof(u));
  return u;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* sh) {
  const uint32_t tid = threadIdx.x;
  const uint32_t sg = tid >> 5;
  const uint32_t lane = tid & 31;
  const uint32_t nsg = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[sg] = v;
  __syncthreads();
  if (sg == 0) {
    float s = lane < nsg ? sh[lane] : 0.0f;
    s = warp_sum(s);
    if (lane == 0) sh[0] = s;
  }
  __syncthreads();
  const float res = sh[0];
  __syncthreads();
  return res;
}

__device__ __forceinline__ float block_sum_6w(float v, float* sh) {
  const uint32_t tid = threadIdx.x;
  const uint32_t sg = tid >> 5;
  const uint32_t lane = tid & 31;
  v = warp_sum(v);
  if (lane == 0) sh[sg] = v;
  __syncthreads();
  if (sg == 0) {
    float s = lane < 6 ? sh[lane] : 0.0f;
    for (int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) sh[0] = s;
  }
  __syncthreads();
  const float res = sh[0];
  __syncthreads();
  return res;
}

struct Ptrs6 {
  const float* p[6];
};

struct Ptrs3 {
  const float* p[3];
};

struct VoidPtrs3 {
  const void* p[3];
};

struct FltMutPtrs3 {
  float* p[3];
};

// ---- 1. Embed + RMSNorm ----
__global__ void k_embed_norm(const int32_t* tokens, int tok_off, const __nv_bfloat16* embed,
                             const __nv_bfloat16* norm_w, float* out, int H, float eps,
                             const __nv_bfloat16* vis_feats, const int32_t* vis_idx, int32_t img_tok_id) {
  __shared__ float sh[32];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int ntg = blockDim.x;
  const int id = tokens[tok_off + t];
  float* dst = out + size_t(t) * size_t(H);
  if (vis_feats && vis_idx && id == img_tok_id) {
    const int vrow = vis_idx[t];
    if (vrow >= 0) {
      const __nv_bfloat16* vptr = vis_feats + size_t(vrow) * size_t(H);
      if (H == 2304 && ntg == 192) {
        const uint2* r2 = reinterpret_cast<const uint2*>(vptr);
        const uint2 u0 = __ldg(r2 + tid), u1 = __ldg(r2 + tid + 192), u2 = __ldg(r2 + tid + 384);
        float4* d4 = reinterpret_cast<float4*>(dst);
        d4[tid] = make_float4(ink2_bf16_lo(u0.x), ink2_bf16_hi(u0.x), ink2_bf16_lo(u0.y), ink2_bf16_hi(u0.y));
        d4[tid + 192] = make_float4(ink2_bf16_lo(u1.x), ink2_bf16_hi(u1.x), ink2_bf16_lo(u1.y), ink2_bf16_hi(u1.y));
        d4[tid + 384] = make_float4(ink2_bf16_lo(u2.x), ink2_bf16_hi(u2.x), ink2_bf16_lo(u2.y), ink2_bf16_hi(u2.y));
        return;
      }
      for (int c = tid; c < H; c += ntg) dst[c] = __bfloat162float(vptr[c]);
      return;
    }
  }
  const __nv_bfloat16* row = embed + size_t(id) * size_t(H);
  if (H == 2304 && ntg == 192) {
    const uint2* r2 = reinterpret_cast<const uint2*>(row);
    const uint2* w2 = reinterpret_cast<const uint2*>(norm_w);
    const uint2 u0 = __ldg(r2 + tid), u1 = __ldg(r2 + tid + 192), u2 = __ldg(r2 + tid + 384);
    const float4 v0 = make_float4(ink2_bf16_lo(u0.x), ink2_bf16_hi(u0.x), ink2_bf16_lo(u0.y), ink2_bf16_hi(u0.y));
    const float4 v1 = make_float4(ink2_bf16_lo(u1.x), ink2_bf16_hi(u1.x), ink2_bf16_lo(u1.y), ink2_bf16_hi(u1.y));
    const float4 v2 = make_float4(ink2_bf16_lo(u2.x), ink2_bf16_hi(u2.x), ink2_bf16_lo(u2.y), ink2_bf16_hi(u2.y));
    const float ss = block_sum_6w(
        (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w) +
        (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w) +
        (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w), sh);
    const float inv = rsqrtf(ss / 2304.0f + eps);
    const uint2 n0 = __ldg(w2 + tid), n1 = __ldg(w2 + tid + 192), n2 = __ldg(w2 + tid + 384);
    float4* d4 = reinterpret_cast<float4*>(dst);
    d4[tid] = make_float4(v0.x * inv * ink2_bf16_lo(n0.x), v0.y * inv * ink2_bf16_hi(n0.x),
                          v0.z * inv * ink2_bf16_lo(n0.y), v0.w * inv * ink2_bf16_hi(n0.y));
    d4[tid + 192] = make_float4(v1.x * inv * ink2_bf16_lo(n1.x), v1.y * inv * ink2_bf16_hi(n1.x),
                                v1.z * inv * ink2_bf16_lo(n1.y), v1.w * inv * ink2_bf16_hi(n1.y));
    d4[tid + 384] = make_float4(v2.x * inv * ink2_bf16_lo(n2.x), v2.y * inv * ink2_bf16_hi(n2.x),
                                v2.z * inv * ink2_bf16_lo(n2.y), v2.w * inv * ink2_bf16_hi(n2.y));
    return;
  }
  float xs[24];
  float ss = 0.0f;
  int k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    const float v = __bfloat162float(row[c]);
    xs[k] = v;
    ss += v * v;
  }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / float(H) + eps);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) dst[c] = xs[k] * inv * __bfloat162float(norm_w[c]);
}

// ---- 2. Generic RMSNorm (zero-stack 192-thread fast path for H==2304 + x3 batch) ----
__device__ __forceinline__ void rmsnorm_192(const float* src, const void* w, bool w_f32, float* dst, float eps, float* sh) {
  const int tid = threadIdx.x;
  const float4* s4 = reinterpret_cast<const float4*>(src);
  float4* d4 = reinterpret_cast<float4*>(dst);
  const float4 v0 = s4[tid], v1 = s4[tid + 192], v2 = s4[tid + 384];
  const float ss = block_sum_6w(
      (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w) +
      (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w) +
      (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w), sh);
  const float inv = rsqrtf(ss / 2304.0f + eps);
  float4 y0 = make_float4(v0.x * inv, v0.y * inv, v0.z * inv, v0.w * inv);
  float4 y1 = make_float4(v1.x * inv, v1.y * inv, v1.z * inv, v1.w * inv);
  float4 y2 = make_float4(v2.x * inv, v2.y * inv, v2.z * inv, v2.w * inv);
  if (w) {
    if (w_f32) {
      const float4* wf = reinterpret_cast<const float4*>(w);
      const float4 w0 = wf[tid], w1 = wf[tid + 192], w2 = wf[tid + 384];
      y0.x *= w0.x; y0.y *= w0.y; y0.z *= w0.z; y0.w *= w0.w;
      y1.x *= w1.x; y1.y *= w1.y; y1.z *= w1.z; y1.w *= w1.w;
      y2.x *= w2.x; y2.y *= w2.y; y2.z *= w2.z; y2.w *= w2.w;
    } else {
      const uint2* wb = reinterpret_cast<const uint2*>(w);
      const uint2 u0 = wb[tid], u1 = wb[tid + 192], u2 = wb[tid + 384];
      y0.x *= ink2_bf16_lo(u0.x); y0.y *= ink2_bf16_hi(u0.x); y0.z *= ink2_bf16_lo(u0.y); y0.w *= ink2_bf16_hi(u0.y);
      y1.x *= ink2_bf16_lo(u1.x); y1.y *= ink2_bf16_hi(u1.x); y1.z *= ink2_bf16_lo(u1.y); y1.w *= ink2_bf16_hi(u1.y);
      y2.x *= ink2_bf16_lo(u2.x); y2.y *= ink2_bf16_hi(u2.x); y2.z *= ink2_bf16_lo(u2.y); y2.w *= ink2_bf16_hi(u2.y);
    }
  }
  d4[tid] = y0; d4[tid + 192] = y1; d4[tid + 384] = y2;
}

__global__ void k_rmsnorm(const float* x, const void* w, bool w_f32, float* out, int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x, tid = threadIdx.x, ntg = blockDim.x;
  const float* src = x + size_t(t) * size_t(H);
  float* dst = out + size_t(t) * size_t(H);
  if (H == 2304 && ntg == 192) {
    rmsnorm_192(src, w, w_f32, dst, eps, sh);
    return;
  }
  float xs[24];
  float ss = 0.0f;
  int k = 0;
  for (int c = tid; c < H; c += ntg, ++k) { const float v = src[c]; xs[k] = v; ss += v * v; }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / float(H) + eps);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    float y = xs[k] * inv;
    if (w) y *= w_f32 ? static_cast<const float*>(w)[c] : __bfloat162float(static_cast<const __nv_bfloat16*>(w)[c]);
    dst[c] = y;
  }
}

__global__ void k_rmsnorm_x3(Ptrs3 xs_in, VoidPtrs3 ws, bool w_f32, FltMutPtrs3 outs, int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, ntg = blockDim.x;
  const float* src = xs_in.p[s] + size_t(t) * size_t(H);
  float* dst = outs.p[s] + size_t(t) * size_t(H);
  const void* w = ws.p[s];
  if (H == 2304 && ntg == 192) {
    rmsnorm_192(src, w, w_f32, dst, eps, sh);
    return;
  }
  float xv[24];
  float ss = 0.0f;
  int k = 0;
  for (int c = tid; c < H; c += ntg, ++k) { const float v = src[c]; xv[k] = v; ss += v * v; }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / float(H) + eps);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    float y = xv[k] * inv;
    if (w) y *= w_f32 ? static_cast<const float*>(w)[c] : __bfloat162float(static_cast<const __nv_bfloat16*>(w)[c]);
    dst[c] = y;
  }
}

// ---- 3. Block Attention Residual (zero-stack 192-thread fast path + optional fused post_norm_w) ----
__device__ __forceinline__ float4 load_q4(const void* queries, bool q_f32, int q, int H, int c4) {
  const size_t idx = size_t(q) * size_t(H) + size_t(c4) * 4;
  if (q_f32) return *reinterpret_cast<const float4*>(static_cast<const float*>(queries) + idx);
  const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(queries) + idx);
  return make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
}

__global__ void k_attn_res_q1_h2304(Ptrs6 st, int M, uint32_t norm_mask, const void* queries, bool q_f32,
                                    float* out, int T, float eps, float scale, const void* post_norm_w, bool post_w_f32) {
  __shared__ float sh[32], sh_d[6][6], tw[6];
  const int t = blockIdx.x, q_idx = blockIdx.y, tid = threadIdx.x, sg = tid >> 5, lane = tid & 31;
  const float4 q0 = load_q4(queries, q_f32, q_idx, 2304, tid);
  const float4 q1 = load_q4(queries, q_f32, q_idx, 2304, tid + 192);
  const float4 q2 = load_q4(queries, q_f32, q_idx, 2304, tid + 384);

  float tinv0 = 1.f, tinv1 = 1.f, tinv2 = 1.f, tinv3 = 1.f, tinv4 = 1.f, tinv5 = 1.f;
  #define DO_M(M_IDX, VAR_TINV) \
    if (M_IDX < M) { \
      const float4* sr = reinterpret_cast<const float4*>(st.p[M_IDX] + size_t(t) * 2304); \
      const float4 v0 = sr[tid], v1 = sr[tid + 192], v2 = sr[tid + 384]; \
      float dot = (v0.x*q0.x + v0.y*q0.y + v0.z*q0.z + v0.w*q0.w) + \
                  (v1.x*q1.x + v1.y*q1.y + v1.z*q1.z + v1.w*q1.w) + \
                  (v2.x*q2.x + v2.y*q2.y + v2.z*q2.z + v2.w*q2.w); \
      float inv = 1.0f; \
      if ((norm_mask & (1u << M_IDX)) == 0u) { \
        const float ss = block_sum_6w( \
            (v0.x*v0.x + v0.y*v0.y + v0.z*v0.z + v0.w*v0.w) + \
            (v1.x*v1.x + v1.y*v1.y + v1.z*v1.z + v1.w*v1.w) + \
            (v2.x*v2.x + v2.y*v2.y + v2.z*v2.z + v2.w*v2.w), sh); \
        inv = rsqrtf(ss / 2304.0f + eps); \
        dot *= inv; \
      } \
      VAR_TINV = inv; \
      dot = warp_sum(dot); \
      if (lane == 0) sh_d[M_IDX][sg] = dot; \
    }
  DO_M(0, tinv0); DO_M(1, tinv1); DO_M(2, tinv2); DO_M(3, tinv3); DO_M(4, tinv4); DO_M(5, tinv5);
  #undef DO_M
  __syncthreads();
  if (tid == 0) {
    const float tsc0 = (sh_d[0][0]+sh_d[0][1]+sh_d[0][2]+sh_d[0][3]+sh_d[0][4]+sh_d[0][5]) * scale;
    const float tsc1 = (1<M)? (sh_d[1][0]+sh_d[1][1]+sh_d[1][2]+sh_d[1][3]+sh_d[1][4]+sh_d[1][5])*scale : -INFINITY;
    const float tsc2 = (2<M)? (sh_d[2][0]+sh_d[2][1]+sh_d[2][2]+sh_d[2][3]+sh_d[2][4]+sh_d[2][5])*scale : -INFINITY;
    const float tsc3 = (3<M)? (sh_d[3][0]+sh_d[3][1]+sh_d[3][2]+sh_d[3][3]+sh_d[3][4]+sh_d[3][5])*scale : -INFINITY;
    const float tsc4 = (4<M)? (sh_d[4][0]+sh_d[4][1]+sh_d[4][2]+sh_d[4][3]+sh_d[4][4]+sh_d[4][5])*scale : -INFINITY;
    const float tsc5 = (5<M)? (sh_d[5][0]+sh_d[5][1]+sh_d[5][2]+sh_d[5][3]+sh_d[5][4]+sh_d[5][5])*scale : -INFINITY;
    const float mx = fmaxf(fmaxf(fmaxf(tsc0, tsc1), fmaxf(tsc2, tsc3)), fmaxf(tsc4, tsc5));
    const float e0 = expf(tsc0 - mx);
    const float e1 = (1<M)? expf(tsc1 - mx) : 0.f;
    const float e2 = (2<M)? expf(tsc2 - mx) : 0.f;
    const float e3 = (3<M)? expf(tsc3 - mx) : 0.f;
    const float e4 = (4<M)? expf(tsc4 - mx) : 0.f;
    const float e5 = (5<M)? expf(tsc5 - mx) : 0.f;
    const float isum = 1.0f / (e0 + e1 + e2 + e3 + e4 + e5);
    tw[0] = e0 * isum * tinv0; tw[1] = e1 * isum * tinv1; tw[2] = e2 * isum * tinv2;
    tw[3] = e3 * isum * tinv3; tw[4] = e4 * isum * tinv4; tw[5] = e5 * isum * tinv5;
  }
  __syncthreads();
  float4 a0 = make_float4(0,0,0,0), a1 = make_float4(0,0,0,0), a2 = make_float4(0,0,0,0);
  #define ACC_M(M_IDX) \
    if (M_IDX < M) { \
      const float w = tw[M_IDX]; \
      const float4* sr = reinterpret_cast<const float4*>(st.p[M_IDX] + size_t(t) * 2304); \
      const float4 v0 = sr[tid], v1 = sr[tid + 192], v2 = sr[tid + 384]; \
      a0.x += w*v0.x; a0.y += w*v0.y; a0.z += w*v0.z; a0.w += w*v0.w; \
      a1.x += w*v1.x; a1.y += w*v1.y; a1.z += w*v1.z; a1.w += w*v1.w; \
      a2.x += w*v2.x; a2.y += w*v2.y; a2.z += w*v2.z; a2.w += w*v2.w; \
    }
  ACC_M(0); ACC_M(1); ACC_M(2); ACC_M(3); ACC_M(4); ACC_M(5);
  #undef ACC_M
  if (post_norm_w) {
    const float ss = block_sum_6w(
        (a0.x*a0.x + a0.y*a0.y + a0.z*a0.z + a0.w*a0.w) +
        (a1.x*a1.x + a1.y*a1.y + a1.z*a1.z + a1.w*a1.w) +
        (a2.x*a2.x + a2.y*a2.y + a2.z*a2.z + a2.w*a2.w), sh);
    const float inv = rsqrtf(ss / 2304.0f + eps);
    a0.x *= inv; a0.y *= inv; a0.z *= inv; a0.w *= inv;
    a1.x *= inv; a1.y *= inv; a1.z *= inv; a1.w *= inv;
    a2.x *= inv; a2.y *= inv; a2.z *= inv; a2.w *= inv;
    if (post_w_f32) {
      const float4* wf = reinterpret_cast<const float4*>(post_norm_w);
      const float4 w0 = wf[tid], w1 = wf[tid + 192], w2 = wf[tid + 384];
      a0.x *= w0.x; a0.y *= w0.y; a0.z *= w0.z; a0.w *= w0.w;
      a1.x *= w1.x; a1.y *= w1.y; a1.z *= w1.z; a1.w *= w1.w;
      a2.x *= w2.x; a2.y *= w2.y; a2.z *= w2.z; a2.w *= w2.w;
    } else {
      const uint2* wb = reinterpret_cast<const uint2*>(post_norm_w);
      const uint2 u0 = wb[tid], u1 = wb[tid + 192], u2 = wb[tid + 384];
      a0.x *= ink2_bf16_lo(u0.x); a0.y *= ink2_bf16_hi(u0.x); a0.z *= ink2_bf16_lo(u0.y); a0.w *= ink2_bf16_hi(u0.y);
      a1.x *= ink2_bf16_lo(u1.x); a1.y *= ink2_bf16_hi(u1.x); a1.z *= ink2_bf16_lo(u1.y); a1.w *= ink2_bf16_hi(u1.y);
      a2.x *= ink2_bf16_lo(u2.x); a2.y *= ink2_bf16_hi(u2.x); a2.z *= ink2_bf16_lo(u2.y); a2.w *= ink2_bf16_hi(u2.y);
    }
  }
  float4* dst4 = reinterpret_cast<float4*>(out + (size_t(q_idx) * size_t(T) + size_t(t)) * 2304);
  dst4[tid] = a0; dst4[tid + 192] = a1; dst4[tid + 384] = a2;
}

__global__ void k_attn_res(Ptrs6 st, int M, uint32_t norm_mask, const void* queries, bool q_f32, int Q, float* out,
                           int T, int H, float eps, float scale, const void* post_norm_w, bool post_w_f32) {
  __shared__ float sh[32], sh_d[6][4][8], tw[6][4];
  const int t = blockIdx.x, tid = threadIdx.x, ntg = blockDim.x;
  const int sg = tid >> 5, lane = tid & 31;
  if (H == 2304 && ntg == 192) {
    float4 qv[4][3];
    for (int q = 0; q < Q; ++q) {
      qv[q][0] = load_q4(queries, q_f32, q, 2304, tid);
      qv[q][1] = load_q4(queries, q_f32, q, 2304, tid + 192);
      qv[q][2] = load_q4(queries, q_f32, q, 2304, tid + 384);
    }
    float tinv_loc[6];
    for (int m = 0; m < M; ++m) {
      const float4* sr = reinterpret_cast<const float4*>(st.p[m] + size_t(t) * 2304);
      const float4 v0 = sr[tid], v1 = sr[tid + 192], v2 = sr[tid + 384];
      float inv = 1.0f;
      if ((norm_mask & (1u << m)) == 0u) {
        const float ss = block_sum_6w(
            (v0.x*v0.x + v0.y*v0.y + v0.z*v0.z + v0.w*v0.w) +
            (v1.x*v1.x + v1.y*v1.y + v1.z*v1.z + v1.w*v1.w) +
            (v2.x*v2.x + v2.y*v2.y + v2.z*v2.z + v2.w*v2.w), sh);
        inv = rsqrtf(ss / 2304.0f + eps);
      }
      tinv_loc[m] = inv;
      for (int q = 0; q < Q; ++q) {
        float dot = ((v0.x*qv[q][0].x + v0.y*qv[q][0].y + v0.z*qv[q][0].z + v0.w*qv[q][0].w) +
                     (v1.x*qv[q][1].x + v1.y*qv[q][1].y + v1.z*qv[q][1].z + v1.w*qv[q][1].w) +
                     (v2.x*qv[q][2].x + v2.y*qv[q][2].y + v2.z*qv[q][2].z + v2.w*qv[q][2].w)) * inv;
        dot = warp_sum(dot);
        if (lane == 0) sh_d[m][q][sg] = dot;
      }
    }
    __syncthreads();
    if (tid < Q) {
      float tsc[6], mx = -INFINITY, exps[6], sum = 0.0f;
      for (int m = 0; m < M; ++m) {
        tsc[m] = (sh_d[m][tid][0]+sh_d[m][tid][1]+sh_d[m][tid][2]+sh_d[m][tid][3]+sh_d[m][tid][4]+sh_d[m][tid][5]) * scale;
        mx = fmaxf(mx, tsc[m]);
      }
      for (int m = 0; m < M; ++m) { exps[m] = expf(tsc[m] - mx); sum += exps[m]; }
      const float isum = 1.0f / sum;
      for (int m = 0; m < M; ++m) tw[m][tid] = exps[m] * isum * tinv_loc[m];
    }
    __syncthreads();
    float4 acc0[4], acc1[4], acc2[4];
    for (int q = 0; q < Q; ++q) {
      acc0[q] = make_float4(0,0,0,0); acc1[q] = make_float4(0,0,0,0); acc2[q] = make_float4(0,0,0,0);
    }
    for (int m = 0; m < M; ++m) {
      const float4* sr = reinterpret_cast<const float4*>(st.p[m] + size_t(t) * 2304);
      const float4 v0 = sr[tid], v1 = sr[tid + 192], v2 = sr[tid + 384];
      for (int q = 0; q < Q; ++q) {
        const float w = tw[m][q];
        acc0[q].x += w*v0.x; acc0[q].y += w*v0.y; acc0[q].z += w*v0.z; acc0[q].w += w*v0.w;
        acc1[q].x += w*v1.x; acc1[q].y += w*v1.y; acc1[q].z += w*v1.z; acc1[q].w += w*v1.w;
        acc2[q].x += w*v2.x; acc2[q].y += w*v2.y; acc2[q].z += w*v2.z; acc2[q].w += w*v2.w;
      }
    }
    for (int q = 0; q < Q; ++q) {
      float4* d4 = reinterpret_cast<float4*>(out + (size_t(q) * size_t(T) + size_t(t)) * 2304);
      d4[tid] = acc0[q]; d4[tid + 192] = acc1[q]; d4[tid + 384] = acc2[q];
    }
    return;
  }

  __shared__ float tinv[6], tsc[6][4];
  float qv[4][24];
  for (int q = 0; q < Q; ++q) {
    int k = 0;
    for (int c = tid; c < H; c += ntg, ++k) {
      const size_t idx = size_t(q) * size_t(H) + size_t(c);
      qv[q][k] = q_f32 ? static_cast<const float*>(queries)[idx]
                       : __bfloat162float(static_cast<const __nv_bfloat16*>(queries)[idx]);
    }
  }
  for (int m = 0; m < M; ++m) {
    const float* sr = st.p[m] + size_t(t) * size_t(H);
    const bool normed = (norm_mask & (1u << m)) != 0u;
    float sv[24], ss = 0.0f;
    int k = 0;
    for (int c = tid; c < H; c += ntg, ++k) { const float v = sr[c]; sv[k] = v; if (!normed) ss += v * v; }
    float inv = 1.0f;
    if (!normed) { ss = block_sum(ss, sh); inv = rsqrtf(ss / float(H) + eps); }
    if (tid == 0) tinv[m] = inv;
    for (int q = 0; q < Q; ++q) {
      float dot = 0.0f;
      for (int kk = 0; kk < k; ++kk) dot += (sv[kk] * inv) * qv[q][kk];
      dot = block_sum(dot, sh);
      if (tid == 0) tsc[m][q] = dot * scale;
    }
  }
  __syncthreads();
  if (tid < Q) {
    float mx = -INFINITY, exps[6], sum = 0.0f;
    for (int m = 0; m < M; ++m) mx = fmaxf(mx, tsc[m][tid]);
    for (int m = 0; m < M; ++m) { exps[m] = expf(tsc[m][tid] - mx); sum += exps[m]; }
    const float isum = 1.0f / sum;
    for (int m = 0; m < M; ++m) tw[m][tid] = exps[m] * isum * tinv[m];
  }
  __syncthreads();
  if (Q == 1 && post_norm_w != nullptr) {
    float acc_v[24], ss = 0.0f;
    int kk = 0;
    for (int c = tid; c < H; c += ntg, ++kk) {
      float a = 0.0f;
      for (int m = 0; m < M; ++m) a += tw[m][0] * st.p[m][size_t(t) * size_t(H) + size_t(c)];
      acc_v[kk] = a; ss += a * a;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / float(H) + eps);
    kk = 0;
    for (int c = tid; c < H; c += ntg, ++kk) {
      const float pw = post_w_f32 ? static_cast<const float*>(post_norm_w)[c]
                                  : __bfloat162float(static_cast<const __nv_bfloat16*>(post_norm_w)[c]);
      out[size_t(t) * size_t(H) + size_t(c)] = acc_v[kk] * inv * pw;
    }
    return;
  }
  for (int c = tid; c < H; c += ntg) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < M; ++m) {
      const float kv = st.p[m][size_t(t) * size_t(H) + size_t(c)];
      for (int q = 0; q < Q; ++q) acc[q] += tw[m][q] * kv;
    }
    for (int q = 0; q < Q; ++q) out[(size_t(q) * size_t(T) + size_t(t)) * size_t(H) + size_t(c)] = acc[q];
  }
}

// ---- 4. Cross-Pass MessageGate (Multi-CTA Vectorized / Batch-x3 / Tensor Core) ----
__global__ void k_msg_prep(const float* h, const float* m, float* nhm_f32, __nv_bfloat16* nhm_bf16,
                           int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x, tid = threadIdx.x, ntg = blockDim.x, H4 = H >> 2;
  const float4* h4 = reinterpret_cast<const float4*>(h + size_t(t) * size_t(H));
  const float4* m4 = reinterpret_cast<const float4*>(m + size_t(t) * size_t(H));
  float4 hs[6], ms[6];
  float ssh = 0.0f, ssm = 0.0f;
  int k = 0;
  for (int c4 = tid; c4 < H4; c4 += ntg, ++k) {
    const float4 hv = h4[c4], mv = m4[c4];
    hs[k] = hv; ms[k] = mv;
    ssh += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    ssm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
  }
  ssh = block_sum(ssh, sh);
  ssm = block_sum(ssm, sh);
  const float invh = rsqrtf(ssh / float(H) + eps);
  const float invm = rsqrtf(ssm / float(H) + eps);

  if (nhm_f32) {
    float4* dst_h = reinterpret_cast<float4*>(nhm_f32 + size_t(t) * size_t(2 * H));
    float4* dst_m = reinterpret_cast<float4*>(nhm_f32 + size_t(t) * size_t(2 * H) + size_t(H));
    k = 0;
    for (int c4 = tid; c4 < H4; c4 += ntg, ++k) {
      dst_h[c4] = make_float4(hs[k].x * invh, hs[k].y * invh, hs[k].z * invh, hs[k].w * invh);
      dst_m[c4] = make_float4(ms[k].x * invm, ms[k].y * invm, ms[k].z * invm, ms[k].w * invm);
    }
  }
  if (nhm_bf16) {
    __nv_bfloat162* dst_h = reinterpret_cast<__nv_bfloat162*>(nhm_bf16 + size_t(t) * size_t(2 * H));
    __nv_bfloat162* dst_m = reinterpret_cast<__nv_bfloat162*>(nhm_bf16 + size_t(t) * size_t(2 * H) + size_t(H));
    k = 0;
    for (int c4 = tid; c4 < H4; c4 += ntg, ++k) {
      dst_h[c4 * 2 + 0] = __float22bfloat162_rn(make_float2(hs[k].x * invh, hs[k].y * invh));
      dst_h[c4 * 2 + 1] = __float22bfloat162_rn(make_float2(hs[k].z * invh, hs[k].w * invh));
      dst_m[c4 * 2 + 0] = __float22bfloat162_rn(make_float2(ms[k].x * invm, ms[k].y * invm));
      dst_m[c4 * 2 + 1] = __float22bfloat162_rn(make_float2(ms[k].z * invm, ms[k].w * invm));
    }
  }
}

template <int MC>
__global__ void k_msg_down_tm(const float* __restrict__ nhm_f32, const __nv_bfloat16* __restrict__ down_w,
                              float* __restrict__ g_f32, int H, int R) {
  __shared__ float sh[MC][4];
  const int r = blockIdx.x, tid = threadIdx.x, ntg = blockDim.x;
  const int sg = tid >> 5, lane = tid & 31, H8 = H >> 3;
  const size_t row_off = size_t(r) * size_t(2 * H);
  const uint4* d0p = reinterpret_cast<const uint4*>(down_w + row_off);
  const uint4* d1p = reinterpret_cast<const uint4*>(down_w + row_off + size_t(H));
  float acc[MC];
  #pragma unroll
  for (int m = 0; m < MC; ++m) acc[m] = 0.0f;
  for (int c8 = tid; c8 < H8; c8 += ntg) {
    const uint4 u0 = __ldg(d0p + c8), u1 = __ldg(d1p + c8);
    const float w0=ink2_bf16_lo(u0.x), w1=ink2_bf16_hi(u0.x), w2=ink2_bf16_lo(u0.y), w3=ink2_bf16_hi(u0.y);
    const float w4=ink2_bf16_lo(u0.z), w5=ink2_bf16_hi(u0.z), w6=ink2_bf16_lo(u0.w), w7=ink2_bf16_hi(u0.w);
    const float v0=ink2_bf16_lo(u1.x), v1=ink2_bf16_hi(u1.x), v2=ink2_bf16_lo(u1.y), v3=ink2_bf16_hi(u1.y);
    const float v4=ink2_bf16_lo(u1.z), v5=ink2_bf16_hi(u1.z), v6=ink2_bf16_lo(u1.w), v7=ink2_bf16_hi(u1.w);
    #pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float4* nh = reinterpret_cast<const float4*>(nhm_f32 + size_t(m) * size_t(2 * H));
      const float4* nm = reinterpret_cast<const float4*>(nhm_f32 + size_t(m) * size_t(2 * H) + size_t(H));
      const float4 h0 = __ldg(nh + c8 * 2), h1 = __ldg(nh + c8 * 2 + 1);
      const float4 m0 = __ldg(nm + c8 * 2), m1 = __ldg(nm + c8 * 2 + 1);
      acc[m] += (h0.x*w0 + h0.y*w1 + h0.z*w2 + h0.w*w3 + h1.x*w4 + h1.y*w5 + h1.z*w6 + h1.w*w7) +
                (m0.x*v0 + m0.y*v1 + m0.z*v2 + m0.w*v3 + m1.x*v4 + m1.y*v5 + m1.z*v6 + m1.w*v7);
    }
  }
  #pragma unroll
  for (int m = 0; m < MC; ++m) {
    const float s = warp_sum(acc[m]);
    if (lane == 0) sh[m][sg] = s;
  }
  __syncthreads();
  if (sg == 0 && lane < MC) {
    g_f32[size_t(lane) * size_t(R) + size_t(r)] = sh[lane][0] + sh[lane][1] + sh[lane][2] + sh[lane][3];
  }
}

__global__ void k_msg_down(const float* nhm_f32, const __nv_bfloat16* down_w, float* g_f32, int T, int H, int R) {
  __shared__ float sh[32];
  const int r = blockIdx.x, tid = threadIdx.x, ntg = blockDim.x;
  const size_t row_off = size_t(r) * size_t(2 * H);
  if (T == 1 && (H & 7) == 0) {
    const int H8 = H >> 3;
    const float4* nh = reinterpret_cast<const float4*>(nhm_f32);
    const float4* nm = reinterpret_cast<const float4*>(nhm_f32 + size_t(H));
    const uint4* d0p = reinterpret_cast<const uint4*>(down_w + row_off);
    const uint4* d1p = reinterpret_cast<const uint4*>(down_w + row_off + size_t(H));
    float acc = 0.0f;
    for (int c8 = tid; c8 < H8; c8 += ntg) {
      const uint4 u0 = __ldg(d0p + c8), u1 = __ldg(d1p + c8);
      const float4 h0 = nh[c8 * 2 + 0], h1 = nh[c8 * 2 + 1];
      const float4 m0 = nm[c8 * 2 + 0], m1 = nm[c8 * 2 + 1];
      acc += h0.x * ink2_bf16_lo(u0.x) + h0.y * ink2_bf16_hi(u0.x) +
             h0.z * ink2_bf16_lo(u0.y) + h0.w * ink2_bf16_hi(u0.y) +
             h1.x * ink2_bf16_lo(u0.z) + h1.y * ink2_bf16_hi(u0.z) +
             h1.z * ink2_bf16_lo(u0.w) + h1.w * ink2_bf16_hi(u0.w);
      acc += m0.x * ink2_bf16_lo(u1.x) + m0.y * ink2_bf16_hi(u1.x) +
             m0.z * ink2_bf16_lo(u1.y) + m0.w * ink2_bf16_hi(u1.y) +
             m1.x * ink2_bf16_lo(u1.z) + m1.y * ink2_bf16_hi(u1.z) +
             m1.z * ink2_bf16_lo(u1.w) + m1.w * ink2_bf16_hi(u1.w);
    }
    acc = block_sum(acc, sh);
    if (tid == 0) g_f32[size_t(r)] = acc;
    return;
  }
  for (int t = 0; t < T; ++t) {
    const float4* nh = reinterpret_cast<const float4*>(nhm_f32 + size_t(t) * size_t(2 * H));
    const float4* nm = reinterpret_cast<const float4*>(nhm_f32 + size_t(t) * size_t(2 * H) + size_t(H));
    float acc = 0.0f;
    if ((H & 7) == 0) {
      const int H8 = H >> 3;
      const uint4* d0p = reinterpret_cast<const uint4*>(down_w + row_off);
      const uint4* d1p = reinterpret_cast<const uint4*>(down_w + row_off + size_t(H));
      for (int c8 = tid; c8 < H8; c8 += ntg) {
        const uint4 u0 = __ldg(d0p + c8), u1 = __ldg(d1p + c8);
        const float4 h0 = nh[c8 * 2 + 0], h1 = nh[c8 * 2 + 1];
        const float4 m0 = nm[c8 * 2 + 0], m1 = nm[c8 * 2 + 1];
        acc += h0.x * ink2_bf16_lo(u0.x) + h0.y * ink2_bf16_hi(u0.x) +
               h0.z * ink2_bf16_lo(u0.y) + h0.w * ink2_bf16_hi(u0.y) +
               h1.x * ink2_bf16_lo(u0.z) + h1.y * ink2_bf16_hi(u0.z) +
               h1.z * ink2_bf16_lo(u0.w) + h1.w * ink2_bf16_hi(u0.w);
        acc += m0.x * ink2_bf16_lo(u1.x) + m0.y * ink2_bf16_hi(u1.x) +
               m0.z * ink2_bf16_lo(u1.y) + m0.w * ink2_bf16_hi(u1.y) +
               m1.x * ink2_bf16_lo(u1.z) + m1.y * ink2_bf16_hi(u1.z) +
               m1.z * ink2_bf16_lo(u1.w) + m1.w * ink2_bf16_hi(u1.w);
      }
    } else {
      for (int c = tid; c < H; c += ntg) {
        const float hv = reinterpret_cast<const float*>(nh)[c];
        const float mv = reinterpret_cast<const float*>(nm)[c];
        acc += hv * __bfloat162float(down_w[row_off + c]) + mv * __bfloat162float(down_w[row_off + H + c]);
      }
    }
    acc = block_sum(acc, sh);
    if (tid == 0) g_f32[size_t(t) * size_t(R) + size_t(r)] = acc;
  }
}

__global__ void k_msg_up_blend(const float* h, const float* m, const float* g_f32, const __nv_bfloat16* forget_w,
                               const __nv_bfloat16* gain_w, float* out, int H, int R) {
  __shared__ float sg_sh[256];
  const int tid = threadIdx.x, t = blockIdx.y;
  for (int r = tid; r < R; r += blockDim.x) sg_sh[r] = g_f32[size_t(t) * size_t(R) + size_t(r)];
  __syncthreads();
  if (R == 64) {
    const int c_local = tid >> 3, sub = tid & 7, c = blockIdx.x * 16 + c_local;
    if (c >= H) return;
    const size_t off = size_t(c) * 64 + size_t(sub) * 8;
    const uint4 uf = __ldg(reinterpret_cast<const uint4*>(forget_w + off));
    const uint4 ug = __ldg(reinterpret_cast<const uint4*>(gain_w + off));
    const float4 g0 = *reinterpret_cast<const float4*>(sg_sh + size_t(sub) * 8 + 0);
    const float4 g1 = *reinterpret_cast<const float4*>(sg_sh + size_t(sub) * 8 + 4);
    float df = g0.x * ink2_bf16_lo(uf.x) + g0.y * ink2_bf16_hi(uf.x) +
               g0.z * ink2_bf16_lo(uf.y) + g0.w * ink2_bf16_hi(uf.y) +
               g1.x * ink2_bf16_lo(uf.z) + g1.y * ink2_bf16_hi(uf.z) +
               g1.z * ink2_bf16_lo(uf.w) + g1.w * ink2_bf16_hi(uf.w);
    float dg = g0.x * ink2_bf16_lo(ug.x) + g0.y * ink2_bf16_hi(ug.x) +
               g0.z * ink2_bf16_lo(ug.y) + g0.w * ink2_bf16_hi(ug.y) +
               g1.x * ink2_bf16_lo(ug.z) + g1.y * ink2_bf16_hi(ug.z) +
               g1.z * ink2_bf16_lo(ug.w) + g1.w * ink2_bf16_hi(ug.w);
    for (int o = 4; o > 0; o >>= 1) {
      df += __shfl_xor_sync(0xffffffffu, df, o);
      dg += __shfl_xor_sync(0xffffffffu, dg, o);
    }
    if (sub == 0) {
      const size_t ioff = size_t(t) * size_t(H) + size_t(c);
      out[ioff] = (1.0f - tanhf(df)) * h[ioff] + tanhf(dg) * m[ioff];
    }
    return;
  }
  const int c = blockIdx.x * 4 + (tid >> 5), lane = tid & 31;
  if (c >= H) return;
  const size_t off = size_t(c) * size_t(R);
  float df = 0.0f, dg = 0.0f;
  for (int r = lane; r < R; r += 32) {
    df += sg_sh[r] * __bfloat162float(forget_w[off + r]);
    dg += sg_sh[r] * __bfloat162float(gain_w[off + r]);
  }
  df = warp_sum(df); dg = warp_sum(dg);
  if (lane == 0) {
    const size_t ioff = size_t(t) * size_t(H) + size_t(c);
    out[ioff] = (1.0f - tanhf(df)) * h[ioff] + tanhf(dg) * m[ioff];
  }
}

struct MsgItem3 {
  const float* h[3];
  const float* m[3];
  const __nv_bfloat16* dw[3];
  const __nv_bfloat16* fw[3];
  const __nv_bfloat16* gw[3];
  float* out[3];
};

__global__ void k_msg_prep_x3(MsgItem3 items, float* nhm3_f32, int T, int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, ntg = blockDim.x, H4 = H >> 2;
  const float4* h4 = reinterpret_cast<const float4*>(items.h[s] + size_t(t) * size_t(H));
  const float4* m4 = reinterpret_cast<const float4*>(items.m[s] + size_t(t) * size_t(H));
  float4 hs[6], ms[6];
  float ssh = 0.0f, ssm = 0.0f;
  int k = 0;
  for (int c4 = tid; c4 < H4; c4 += ntg, ++k) {
    const float4 hv = h4[c4], mv = m4[c4];
    hs[k] = hv; ms[k] = mv;
    ssh += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    ssm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
  }
  ssh = block_sum(ssh, sh);
  ssm = block_sum(ssm, sh);
  const float invh = rsqrtf(ssh / float(H) + eps);
  const float invm = rsqrtf(ssm / float(H) + eps);
  float* base = nhm3_f32 + (size_t(s) * size_t(T) + size_t(t)) * size_t(2 * H);
  float4* dst_h = reinterpret_cast<float4*>(base);
  float4* dst_m = reinterpret_cast<float4*>(base + size_t(H));
  k = 0;
  for (int c4 = tid; c4 < H4; c4 += ntg, ++k) {
    dst_h[c4] = make_float4(hs[k].x * invh, hs[k].y * invh, hs[k].z * invh, hs[k].w * invh);
    dst_m[c4] = make_float4(ms[k].x * invm, ms[k].y * invm, ms[k].z * invm, ms[k].w * invm);
  }
}

template <int MC>
__global__ void k_msg_down_x3_tm(MsgItem3 items, const float* __restrict__ nhm3_f32, float* __restrict__ g3_f32,
                                 int H, int R) {
  __shared__ float sh[MC][4];
  const int r = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, ntg = blockDim.x;
  const int sg = tid >> 5, lane = tid & 31, H8 = H >> 3;
  const __nv_bfloat16* down_w = items.dw[s];
  const size_t row_off = size_t(r) * size_t(2 * H);
  const uint4* d0p = reinterpret_cast<const uint4*>(down_w + row_off);
  const uint4* d1p = reinterpret_cast<const uint4*>(down_w + row_off + size_t(H));
  float acc[MC];
  #pragma unroll
  for (int m = 0; m < MC; ++m) acc[m] = 0.0f;
  for (int c8 = tid; c8 < H8; c8 += ntg) {
    const uint4 u0 = __ldg(d0p + c8), u1 = __ldg(d1p + c8);
    const float w0=ink2_bf16_lo(u0.x), w1=ink2_bf16_hi(u0.x), w2=ink2_bf16_lo(u0.y), w3=ink2_bf16_hi(u0.y);
    const float w4=ink2_bf16_lo(u0.z), w5=ink2_bf16_hi(u0.z), w6=ink2_bf16_lo(u0.w), w7=ink2_bf16_hi(u0.w);
    const float v0=ink2_bf16_lo(u1.x), v1=ink2_bf16_hi(u1.x), v2=ink2_bf16_lo(u1.y), v3=ink2_bf16_hi(u1.y);
    const float v4=ink2_bf16_lo(u1.z), v5=ink2_bf16_hi(u1.z), v6=ink2_bf16_lo(u1.w), v7=ink2_bf16_hi(u1.w);
    #pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float* base = nhm3_f32 + (size_t(s) * size_t(MC) + size_t(m)) * size_t(2 * H);
      const float4* nh = reinterpret_cast<const float4*>(base);
      const float4* nm = reinterpret_cast<const float4*>(base + size_t(H));
      const float4 h0 = __ldg(nh + c8 * 2), h1 = __ldg(nh + c8 * 2 + 1);
      const float4 m0 = __ldg(nm + c8 * 2), m1 = __ldg(nm + c8 * 2 + 1);
      acc[m] += (h0.x*w0 + h0.y*w1 + h0.z*w2 + h0.w*w3 + h1.x*w4 + h1.y*w5 + h1.z*w6 + h1.w*w7) +
                (m0.x*v0 + m0.y*v1 + m0.z*v2 + m0.w*v3 + m1.x*v4 + m1.y*v5 + m1.z*v6 + m1.w*v7);
    }
  }
  #pragma unroll
  for (int m = 0; m < MC; ++m) {
    const float sv = warp_sum(acc[m]);
    if (lane == 0) sh[m][sg] = sv;
  }
  __syncthreads();
  if (sg == 0 && lane < MC) {
    g3_f32[(size_t(s) * size_t(MC) + size_t(lane)) * size_t(R) + size_t(r)] =
        sh[lane][0] + sh[lane][1] + sh[lane][2] + sh[lane][3];
  }
}

__global__ void k_msg_down_x3(MsgItem3 items, const float* nhm3_f32, float* g3_f32, int T, int H, int R) {
  __shared__ float sh[32];
  const int r = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, ntg = blockDim.x;
  const __nv_bfloat16* down_w = items.dw[s];
  const size_t row_off = size_t(r) * size_t(2 * H);
  const int H8 = H >> 3;
  const uint4* d0p = reinterpret_cast<const uint4*>(down_w + row_off);
  const uint4* d1p = reinterpret_cast<const uint4*>(down_w + row_off + size_t(H));
  for (int t = 0; t < T; ++t) {
    const float* base = nhm3_f32 + (size_t(s) * size_t(T) + size_t(t)) * size_t(2 * H);
    const float4* nh = reinterpret_cast<const float4*>(base);
    const float4* nm = reinterpret_cast<const float4*>(base + size_t(H));
    float acc = 0.0f;
    for (int c8 = tid; c8 < H8; c8 += ntg) {
      const uint4 u0 = __ldg(d0p + c8), u1 = __ldg(d1p + c8);
      const float4 h0 = nh[c8 * 2 + 0], h1 = nh[c8 * 2 + 1];
      const float4 m0 = nm[c8 * 2 + 0], m1 = nm[c8 * 2 + 1];
      acc += h0.x * ink2_bf16_lo(u0.x) + h0.y * ink2_bf16_hi(u0.x) +
             h0.z * ink2_bf16_lo(u0.y) + h0.w * ink2_bf16_hi(u0.y) +
             h1.x * ink2_bf16_lo(u0.z) + h1.y * ink2_bf16_hi(u0.z) +
             h1.z * ink2_bf16_lo(u0.w) + h1.w * ink2_bf16_hi(u0.w);
      acc += m0.x * ink2_bf16_lo(u1.x) + m0.y * ink2_bf16_hi(u1.x) +
             m0.z * ink2_bf16_lo(u1.y) + m0.w * ink2_bf16_hi(u1.y) +
             m1.x * ink2_bf16_lo(u1.z) + m1.y * ink2_bf16_hi(u1.z) +
             m1.z * ink2_bf16_lo(u1.w) + m1.w * ink2_bf16_hi(u1.w);
    }
    acc = block_sum(acc, sh);
    if (tid == 0) g3_f32[(size_t(s) * size_t(T) + size_t(t)) * size_t(R) + size_t(r)] = acc;
  }
}

__global__ void k_msg_up_blend_x3(MsgItem3 items, const float* g3_f32, int T, int H, int R) {
  __shared__ float sg_sh[256];
  const int tid = threadIdx.x, t = blockIdx.y, s = blockIdx.z;
  const float* g_row = g3_f32 + (size_t(s) * size_t(T) + size_t(t)) * size_t(R);
  for (int r = tid; r < R; r += blockDim.x) sg_sh[r] = g_row[r];
  __syncthreads();
  const float* h = items.h[s];
  const float* m = items.m[s];
  float* out = items.out[s];
  const __nv_bfloat16* forget_w = items.fw[s];
  const __nv_bfloat16* gain_w = items.gw[s];
  if (R == 64) {
    const int c_local = tid >> 3, sub = tid & 7, c = blockIdx.x * 16 + c_local;
    if (c >= H) return;
    const size_t off = size_t(c) * 64 + size_t(sub) * 8;
    const uint4 uf = __ldg(reinterpret_cast<const uint4*>(forget_w + off));
    const uint4 ug = __ldg(reinterpret_cast<const uint4*>(gain_w + off));
    const float4 g0 = *reinterpret_cast<const float4*>(sg_sh + size_t(sub) * 8 + 0);
    const float4 g1 = *reinterpret_cast<const float4*>(sg_sh + size_t(sub) * 8 + 4);
    float df = g0.x * ink2_bf16_lo(uf.x) + g0.y * ink2_bf16_hi(uf.x) +
               g0.z * ink2_bf16_lo(uf.y) + g0.w * ink2_bf16_hi(uf.y) +
               g1.x * ink2_bf16_lo(uf.z) + g1.y * ink2_bf16_hi(uf.z) +
               g1.z * ink2_bf16_lo(uf.w) + g1.w * ink2_bf16_hi(uf.w);
    float dg = g0.x * ink2_bf16_lo(ug.x) + g0.y * ink2_bf16_hi(ug.x) +
               g0.z * ink2_bf16_lo(ug.y) + g0.w * ink2_bf16_hi(ug.y) +
               g1.x * ink2_bf16_lo(ug.z) + g1.y * ink2_bf16_hi(ug.z) +
               g1.z * ink2_bf16_lo(ug.w) + g1.w * ink2_bf16_hi(ug.w);
    for (int o = 4; o > 0; o >>= 1) {
      df += __shfl_xor_sync(0xffffffffu, df, o);
      dg += __shfl_xor_sync(0xffffffffu, dg, o);
    }
    if (sub == 0) {
      const size_t ioff = size_t(t) * size_t(H) + size_t(c);
      out[ioff] = (1.0f - tanhf(df)) * h[ioff] + tanhf(dg) * m[ioff];
    }
    return;
  }
  const int c = blockIdx.x * 4 + (tid >> 5), lane = tid & 31;
  if (c >= H) return;
  const size_t off = size_t(c) * size_t(R);
  float df = 0.0f, dg = 0.0f;
  for (int r = lane; r < R; r += 32) {
    df += sg_sh[r] * __bfloat162float(forget_w[off + r]);
    dg += sg_sh[r] * __bfloat162float(gain_w[off + r]);
  }
  df = warp_sum(df); dg = warp_sum(dg);
  if (lane == 0) {
    const size_t ioff = size_t(t) * size_t(H) + size_t(c);
    out[ioff] = (1.0f - tanhf(df)) * h[ioff] + tanhf(dg) * m[ioff];
  }
}

__global__ void k_msg_blend(const float* h, const float* m, const float* df, const float* dg, float* out, int n) {
  const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  if (i4 < n4) {
    const float4 hv = reinterpret_cast<const float4*>(h)[i4];
    const float4 mv = reinterpret_cast<const float4*>(m)[i4];
    const float4 fv = reinterpret_cast<const float4*>(df)[i4];
    const float4 gv = reinterpret_cast<const float4*>(dg)[i4];
    reinterpret_cast<float4*>(out)[i4] = make_float4(
        (1.0f - tanhf(fv.x)) * hv.x + tanhf(gv.x) * mv.x,
        (1.0f - tanhf(fv.y)) * hv.y + tanhf(gv.y) * mv.y,
        (1.0f - tanhf(fv.z)) * hv.z + tanhf(gv.z) * mv.z,
        (1.0f - tanhf(fv.w)) * hv.w + tanhf(gv.w) * mv.w);
    return;
  }
  const int i = (n4 << 2) + (i4 - n4);
  if (i < n) out[i] = (1.0f - tanhf(df[i])) * h[i] + tanhf(dg[i]) * m[i];
}

// ---- 5. Linear Projections (SMEM-free __ldg MV for M=1..8, cuBLAS TC for M>32) ----
struct SegSpec6 {
  float* d0; float* d1; float* d2; float* d3; float* d4; float* d5;
  int s0; int s1; int s2; int s3; int s4; int s5;
  int r0; int r1; int r2; int r3; int r4; int r5;
  int n_segs;
  int total_rows;
};

__device__ __forceinline__ float* seg_ptr(const SegSpec6& g, int n, int& local_n, int& seg_rows) {
  if (g.n_segs > 5 && n >= g.s5) { local_n = n - g.s5; seg_rows = g.r5; return g.d5; }
  if (g.n_segs > 4 && n >= g.s4) { local_n = n - g.s4; seg_rows = g.r4; return g.d4; }
  if (g.n_segs > 3 && n >= g.s3) { local_n = n - g.s3; seg_rows = g.r3; return g.d3; }
  if (g.n_segs > 2 && n >= g.s2) { local_n = n - g.s2; seg_rows = g.r2; return g.d2; }
  if (g.n_segs > 1 && n >= g.s1) { local_n = n - g.s1; seg_rows = g.r1; return g.d1; }
  local_n = n; seg_rows = g.r0; return g.d0;
}

__global__ void k_bf16_mv_cta_row(const float* x, const __nv_bfloat16* w, float* y, int N, int K, bool add) {
  __shared__ float sh[32];
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x, K8 = K >> 3;
  const uint4* w16 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K));
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc = 0.0f;
  for (int k8 = tid; k8 < K8; k8 += 128) {
    const uint4 u = __ldg(w16 + k8);
    const float4 xv0 = x4[k8 * 2 + 0], xv1 = x4[k8 * 2 + 1];
    acc += xv0.x * ink2_bf16_lo(u.x) + xv0.y * ink2_bf16_hi(u.x) +
           xv0.z * ink2_bf16_lo(u.y) + xv0.w * ink2_bf16_hi(u.y) +
           xv1.x * ink2_bf16_lo(u.z) + xv1.y * ink2_bf16_hi(u.z) +
           xv1.z * ink2_bf16_lo(u.w) + xv1.w * ink2_bf16_hi(u.w);
  }
  acc = block_sum(acc, sh);
  if (tid == 0) y[n] = add ? (y[n] + acc) : acc;
}

__global__ void k_bf16_mv_m1_1row_4w(const float* x, const __nv_bfloat16* w, float* y, int N, int K, bool add) {
  __shared__ float sh[4];
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x, sg = tid >> 5, lane = tid & 31, K8 = K >> 3;
  const uint4* w0 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K));
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc = 0.0f;
  for (int k8 = tid; k8 < K8; k8 += 128) {
    const float4 xv0 = x4[k8 * 2 + 0], xv1 = x4[k8 * 2 + 1];
    const uint4 u0 = __ldg(w0 + k8);
    acc += xv0.x * ink2_bf16_lo(u0.x) + xv0.y * ink2_bf16_hi(u0.x) +
           xv0.z * ink2_bf16_lo(u0.y) + xv0.w * ink2_bf16_hi(u0.y) +
           xv1.x * ink2_bf16_lo(u0.z) + xv1.y * ink2_bf16_hi(u0.z) +
           xv1.z * ink2_bf16_lo(u0.w) + xv1.w * ink2_bf16_hi(u0.w);
  }
  acc = warp_sum(acc);
  if (lane == 0) sh[sg] = acc;
  __syncthreads();
  if (sg == 0 && lane < 4) {
    float v = sh[lane];
    v += __shfl_xor_sync(0xfu, v, 2);
    v += __shfl_xor_sync(0xfu, v, 1);
    if (lane == 0) y[n] = add ? (y[n] + v) : v;
  }
}

__global__ void k_bf16_mv_multi_m1_1row_4w(const float* x, const __nv_bfloat16* w, SegSpec6 seg, int K) {
  __shared__ float sh[4];
  const int n = blockIdx.x;
  if (n >= seg.total_rows) return;
  const int tid = threadIdx.x, sg = tid >> 5, lane = tid & 31, K8 = K >> 3;
  const uint4* w0 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K));
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc = 0.0f;
  for (int k8 = tid; k8 < K8; k8 += 128) {
    const float4 xv0 = x4[k8 * 2 + 0], xv1 = x4[k8 * 2 + 1];
    const uint4 u0 = __ldg(w0 + k8);
    acc += xv0.x * ink2_bf16_lo(u0.x) + xv0.y * ink2_bf16_hi(u0.x) +
           xv0.z * ink2_bf16_lo(u0.y) + xv0.w * ink2_bf16_hi(u0.y) +
           xv1.x * ink2_bf16_lo(u0.z) + xv1.y * ink2_bf16_hi(u0.z) +
           xv1.z * ink2_bf16_lo(u0.w) + xv1.w * ink2_bf16_hi(u0.w);
  }
  acc = warp_sum(acc);
  if (lane == 0) sh[sg] = acc;
  __syncthreads();
  if (sg == 0 && lane < 4) {
    float v = sh[lane];
    v += __shfl_xor_sync(0xfu, v, 2);
    v += __shfl_xor_sync(0xfu, v, 1);
    if (lane == 0) {
      int ln, sr;
      float* dp = seg_ptr(seg, n, ln, sr);
      dp[ln] = v;
    }
  }
}

__global__ void k_bf16_mv_m1_2row(const float* x, const __nv_bfloat16* w, float* y, int N, int K, bool add) {
  const int n0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * 2;
  if (n0 >= N) return;
  const bool has1 = (n0 + 1) < N;
  const int lane = threadIdx.x & 31, K8 = K >> 3;
  const uint4* w0 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n0) * size_t(K));
  const uint4* w1 = has1 ? reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n0 + 1) * size_t(K)) : nullptr;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc0 = 0.0f, acc1 = 0.0f;
  for (int k8 = lane; k8 < K8; k8 += 32) {
    const float4 xv0 = x4[k8 * 2 + 0], xv1 = x4[k8 * 2 + 1];
    const uint4 u0 = __ldg(w0 + k8);
    acc0 += xv0.x * ink2_bf16_lo(u0.x) + xv0.y * ink2_bf16_hi(u0.x) +
            xv0.z * ink2_bf16_lo(u0.y) + xv0.w * ink2_bf16_hi(u0.y) +
            xv1.x * ink2_bf16_lo(u0.z) + xv1.y * ink2_bf16_hi(u0.z) +
            xv1.z * ink2_bf16_lo(u0.w) + xv1.w * ink2_bf16_hi(u0.w);
    if (has1) {
      const uint4 u1 = __ldg(w1 + k8);
      acc1 += xv0.x * ink2_bf16_lo(u1.x) + xv0.y * ink2_bf16_hi(u1.x) +
              xv0.z * ink2_bf16_lo(u1.y) + xv0.w * ink2_bf16_hi(u1.y) +
              xv1.x * ink2_bf16_lo(u1.z) + xv1.y * ink2_bf16_hi(u1.z) +
              xv1.z * ink2_bf16_lo(u1.w) + xv1.w * ink2_bf16_hi(u1.w);
    }
  }
  acc0 = warp_sum(acc0);
  if (has1) acc1 = warp_sum(acc1);
  if (lane == 0) {
    y[n0] = add ? (y[n0] + acc0) : acc0;
    if (has1) y[n0 + 1] = add ? (y[n0 + 1] + acc1) : acc1;
  }
}

template <int MC>
__global__ void k_router_mw_tm(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ y,
                               int N, int K, bool add) {
  __shared__ float sh_acc[MC][8];
  const int n = blockIdx.x, tid = threadIdx.x, sg = tid >> 5, lane = tid & 31, K8 = K >> 3;
  const uint4* w16 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K));
  float acc[MC];
  #pragma unroll
  for (int m = 0; m < MC; ++m) acc[m] = 0.0f;
  for (int k8 = tid; k8 < K8; k8 += 256) {
    const uint4 u = __ldg(w16 + k8);
    const float4 wv0 = make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
    const float4 wv1 = make_float4(ink2_bf16_lo(u.z), ink2_bf16_hi(u.z), ink2_bf16_lo(u.w), ink2_bf16_hi(u.w));
    #pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float4* xp = reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(k8) * 8);
      const float4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
      acc[m] += (xv0.x * wv0.x + xv0.y * wv0.y + xv0.z * wv0.z + xv0.w * wv0.w) +
                (xv1.x * wv1.x + xv1.y * wv1.y + xv1.z * wv1.z + xv1.w * wv1.w);
    }
  }
  #pragma unroll
  for (int m = 0; m < MC; ++m) {
    const float s = warp_sum(acc[m]);
    if (lane == 0) sh_acc[m][sg] = s;
  }
  __syncthreads();
  if (sg == 0 && lane < MC) {
    float tot = 0.0f;
    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) tot += sh_acc[lane][wi];
    float* dst = y + size_t(lane) * size_t(N) + size_t(n);
    *dst = add ? (*dst + tot) : tot;
  }
}

template <int MC, int ROWS_PER_WARP, int NWARPS>
__global__ void k_bf16_mv_ldg(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ y,
                              int N, int K, bool add) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int base_n = (blockIdx.x * NWARPS + warp) * ROWS_PER_WARP;
  float acc[ROWS_PER_WARP][MC];
  #pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r)
    #pragma unroll
    for (int m = 0; m < MC; ++m) acc[r][m] = 0.0f;
  const int K8 = K >> 3;
  for (int k8 = lane; k8 < K8; k8 += 32) {
    uint4 uw[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      const int n = base_n + r;
      uw[r] = (n < N) ? __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K)) + k8)
                      : make_uint4(0, 0, 0, 0);
    }
    #pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float4* xp = reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(k8) * 8);
      const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
      #pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const uint4 u = uw[r];
        acc[r][m] += (x0.x*ink2_bf16_lo(u.x) + x0.y*ink2_bf16_hi(u.x) + x0.z*ink2_bf16_lo(u.y) + x0.w*ink2_bf16_hi(u.y)) +
                     (x1.x*ink2_bf16_lo(u.z) + x1.y*ink2_bf16_hi(u.z) + x1.z*ink2_bf16_lo(u.w) + x1.w*ink2_bf16_hi(u.w));
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    const int n = base_n + r;
    if (n < N) {
      #pragma unroll
      for (int m = 0; m < MC; ++m) {
        const float s = warp_sum(acc[r][m]);
        if (lane == 0) {
          float* dst = y + size_t(m) * size_t(N) + size_t(n);
          *dst = add ? (*dst + s) : s;
        }
      }
    }
  }
}

template <int MC, int ROWS_PER_WARP, int NWARPS>
__global__ void k_bf16_mv_multi_ldg(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w, SegSpec6 seg, int K) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int base_n = (blockIdx.x * NWARPS + warp) * ROWS_PER_WARP;
  float acc[ROWS_PER_WARP][MC];
  #pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r)
    #pragma unroll
    for (int m = 0; m < MC; ++m) acc[r][m] = 0.0f;
  const int K8 = K >> 3;
  for (int k8 = lane; k8 < K8; k8 += 32) {
    uint4 uw[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      const int n = base_n + r;
      uw[r] = (n < seg.total_rows)
                  ? __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K)) + k8)
                  : make_uint4(0, 0, 0, 0);
    }
    #pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float4* xp = reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(k8) * 8);
      const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
      #pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const uint4 u = uw[r];
        acc[r][m] += (x0.x*ink2_bf16_lo(u.x) + x0.y*ink2_bf16_hi(u.x) + x0.z*ink2_bf16_lo(u.y) + x0.w*ink2_bf16_hi(u.y)) +
                     (x1.x*ink2_bf16_lo(u.z) + x1.y*ink2_bf16_hi(u.z) + x1.z*ink2_bf16_lo(u.w) + x1.w*ink2_bf16_hi(u.w));
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    const int n = base_n + r;
    if (n < seg.total_rows) {
      int local_n, seg_rows;
      float* seg_dst = seg_ptr(seg, n, local_n, seg_rows);
      #pragma unroll
      for (int m = 0; m < MC; ++m) {
        const float val = warp_sum(acc[r][m]);
        if (lane == 0) seg_dst[size_t(m) * size_t(seg_rows) + size_t(local_n)] = val;
      }
    }
  }
}

template <int MC>
__global__ void __launch_bounds__(128) k_bf16_mv_smem(
    const float* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ y,
    int N, int K, bool add) {
  const int tid = threadIdx.x, sub = tid >> 4, lg = tid & 15;
  const int n = blockIdx.x * 8 + sub;
  const bool valid = (n < N);
  const uint4* w16 = valid
      ? reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K))
      : nullptr;
  __shared__ alignas(16) float s_x[MC][256];
  float acc[MC];
  #pragma unroll
  for (int m = 0; m < MC; ++m) acc[m] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += 256) {
    for (int i = tid; i < MC * 64; i += 128) {
      const int m = i >> 6, v = i & 63;
      *reinterpret_cast<float4*>(&s_x[m][v << 2]) =
          *reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(k0) + size_t(v << 2));
    }
    __syncthreads();
    if (valid) {
      #pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int k_tile = p * 128 + lg * 8;
        const uint4 u = __ldg(w16 + ((k0 + k_tile) >> 3));
        const float4 wv0 = make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
        const float4 wv1 = make_float4(ink2_bf16_lo(u.z), ink2_bf16_hi(u.z), ink2_bf16_lo(u.w), ink2_bf16_hi(u.w));
        #pragma unroll
        for (int m = 0; m < MC; ++m) {
          const float4 x0 = *reinterpret_cast<const float4*>(&s_x[m][k_tile]);
          const float4 x1 = *reinterpret_cast<const float4*>(&s_x[m][k_tile + 4]);
          acc[m] += (x0.x * wv0.x + x0.y * wv0.y + x0.z * wv0.z + x0.w * wv0.w) +
                    (x1.x * wv1.x + x1.y * wv1.y + x1.z * wv1.z + x1.w * wv1.w);
        }
      }
    }
    if (k0 + 256 < K) __syncthreads();
  }
  #pragma unroll
  for (int m = 0; m < MC; ++m) {
    float v = acc[m];
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (valid && lg == 0) {
      float* dst = y + size_t(m) * size_t(N) + size_t(n);
      *dst = add ? (*dst + v) : v;
    }
  }
}

template <int MC>
__global__ void __launch_bounds__(128) k_bf16_mv_smem_multi(
    const float* __restrict__ x, const __nv_bfloat16* __restrict__ w, SegSpec6 seg, int K) {
  const int tid = threadIdx.x, sub = tid >> 4, lg = tid & 15;
  const int n = blockIdx.x * 8 + sub;
  const bool valid = (n < seg.total_rows);
  const uint4* w16 = valid
      ? reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K))
      : nullptr;
  __shared__ alignas(16) float s_x[MC][256];
  float acc[MC];
  #pragma unroll
  for (int m = 0; m < MC; ++m) acc[m] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += 256) {
    for (int i = tid; i < MC * 64; i += 128) {
      const int m = i >> 6, v = i & 63;
      *reinterpret_cast<float4*>(&s_x[m][v << 2]) =
          *reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(k0) + size_t(v << 2));
    }
    __syncthreads();
    if (valid) {
      #pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int k_tile = p * 128 + lg * 8;
        const uint4 u = __ldg(w16 + ((k0 + k_tile) >> 3));
        const float4 wv0 = make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
        const float4 wv1 = make_float4(ink2_bf16_lo(u.z), ink2_bf16_hi(u.z), ink2_bf16_lo(u.w), ink2_bf16_hi(u.w));
        #pragma unroll
        for (int m = 0; m < MC; ++m) {
          const float4 x0 = *reinterpret_cast<const float4*>(&s_x[m][k_tile]);
          const float4 x1 = *reinterpret_cast<const float4*>(&s_x[m][k_tile + 4]);
          acc[m] += (x0.x * wv0.x + x0.y * wv0.y + x0.z * wv0.z + x0.w * wv0.w) +
                    (x1.x * wv1.x + x1.y * wv1.y + x1.z * wv1.z + x1.w * wv1.w);
        }
      }
    }
    if (k0 + 256 < K) __syncthreads();
  }
  int local_n = 0, seg_rows = 1;
  float* seg_dst = valid ? seg_ptr(seg, n, local_n, seg_rows) : nullptr;
  #pragma unroll
  for (int m = 0; m < MC; ++m) {
    float v = acc[m];
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (valid && lg == 0) seg_dst[size_t(m) * size_t(seg_rows) + size_t(local_n)] = v;
  }
}

__global__ void k_bf16_mv(const float* x, const void* w, bool w_f32, float* y,
                          int M, int N, int K, bool add) {
  const int n = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (n >= N) return;
  const int lane = threadIdx.x & 31;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff = size_t(n) * size_t(K);
  if (!w_f32 && ((K & 7) == 0)) {
    const uint4* w16 = reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(w) + woff);
    const int K8 = K >> 3;
    for (int k8 = lane; k8 < K8; k8 += 32) {
      const uint4 u = __ldg(w16 + k8);
      const float4 wv0 = make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
      const float4 wv1 = make_float4(ink2_bf16_lo(u.z), ink2_bf16_hi(u.z), ink2_bf16_lo(u.w), ink2_bf16_hi(u.w));
      for (int m = 0; m < M; ++m) {
        const float4* xp = reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(k8) * 8);
        const float4 xv0 = xp[0], xv1 = xp[1];
        acc[m] += (xv0.x * wv0.x + xv0.y * wv0.y + xv0.z * wv0.z + xv0.w * wv0.w) +
                  (xv1.x * wv1.x + xv1.y * wv1.y + xv1.z * wv1.z + xv1.w * wv1.w);
      }
    }
  } else {
    for (int k = lane * 4; k < K; k += 128) {
      float4 wv;
      if (w_f32) {
        wv = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + woff + k);
      } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(w) + woff + k);
        wv = make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
      }
      for (int m = 0; m < M; ++m) {
        const float4 xv = *reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + k);
        acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
      }
    }
  }
  for (int m = 0; m < M; ++m) {
    const float s = warp_sum(acc[m]);
    if (lane == 0) {
      float* dst = y + size_t(m) * size_t(N) + n;
      *dst = add ? (*dst + s) : s;
    }
  }
}

__global__ void k_bf16_mv_multi(const float* x, const __nv_bfloat16* w, SegSpec6 seg, int M, int K) {
  const int n = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (n >= seg.total_rows) return;
  const int lane = threadIdx.x & 31;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint4* w16 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(w) + size_t(n) * size_t(K));
  const int K8 = K >> 3;
  for (int k8 = lane; k8 < K8; k8 += 32) {
    const uint4 u = __ldg(w16 + k8);
    const float4 wv0 = make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
    const float4 wv1 = make_float4(ink2_bf16_lo(u.z), ink2_bf16_hi(u.z), ink2_bf16_lo(u.w), ink2_bf16_hi(u.w));
    for (int m = 0; m < M; ++m) {
      const float4* xp = reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(k8) * 8);
      const float4 xv0 = xp[0], xv1 = xp[1];
      acc[m] += (xv0.x * wv0.x + xv0.y * wv0.y + xv0.z * wv0.z + xv0.w * wv0.w) +
                (xv1.x * wv1.x + xv1.y * wv1.y + xv1.z * wv1.z + xv1.w * wv1.w);
    }
  }
  int local_n, seg_rows;
  float* seg_dst = seg_ptr(seg, n, local_n, seg_rows);
  for (int m = 0; m < M; ++m) {
    const float val = warp_sum(acc[m]);
    if (lane == 0) seg_dst[size_t(m) * size_t(seg_rows) + size_t(local_n)] = val;
  }
}

__global__ void k_split_multi(const float* tmp_y, SegSpec6 seg, int M) {
  if ((seg.total_rows & 3) == 0) {
    const size_t idx4 = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    const int tot4 = seg.total_rows >> 2;
    const size_t total4 = size_t(M) * size_t(tot4);
    if (idx4 >= total4) return;
    const int m = int(idx4 / size_t(tot4));
    const int n4 = int(idx4 % size_t(tot4));
    const int n = n4 << 2;
    int ln, sr;
    float* dp = seg_ptr(seg, n, ln, sr);
    const float4 v = reinterpret_cast<const float4*>(tmp_y)[idx4];
    *reinterpret_cast<float4*>(dp + size_t(m) * size_t(sr) + size_t(ln)) = v;
    return;
  }
  const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  const size_t total = size_t(M) * size_t(seg.total_rows);
  if (idx >= total) return;
  const int m = int(idx / size_t(seg.total_rows));
  const int n = int(idx % size_t(seg.total_rows));
  int ln, sr;
  float* dp = seg_ptr(seg, n, ln, sr);
  dp[size_t(m) * size_t(sr) + size_t(ln)] = tmp_y[idx];
}

__global__ void k_quantize_q8_32(const __nv_bfloat16* w, int8_t* q, __half* sc, int N, int K) {
  const int warp = int(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int blocks = K >> 5;
  const int total_warps = N * blocks;
  if (warp >= total_warps) return;
  const int lane = threadIdx.x & 31;
  const int row = warp / blocks;
  const int blk = warp - row * blocks;
  const size_t idx = size_t(row) * size_t(K) + size_t(blk) * 32 + size_t(lane);
  const float v = __bfloat162float(w[idx]);
  float amax = fabsf(v);
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
  const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  int qi = __float2int_rn(v * inv);
  if (qi > 127) qi = 127;
  if (qi < -127) qi = -127;
  q[idx] = int8_t(qi);
  if (lane == 0) sc[size_t(row) * size_t(blocks) + size_t(blk)] = __float2half(scale);
}

template <int MC, int ROWS_PER_WARP>
__global__ void k_q8_32_mv_tm(const float* __restrict__ x, const int8_t* __restrict__ w_q,
                              const __half* __restrict__ w_sc, float* __restrict__ y, int N, int K) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int base_n = (blockIdx.x * 4 + warp) * ROWS_PER_WARP;
  const int blocks = K >> 5, K16 = K >> 4;
  float acc[ROWS_PER_WARP][MC];
  #pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r)
    #pragma unroll
    for (int m = 0; m < MC; ++m) acc[r][m] = 0.0f;

  for (int idx = lane; idx < K16; idx += 32) {
    const int blk = idx >> 1;
    int4 wv[ROWS_PER_WARP];
    float scv[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      const int n = base_n + r;
      if (n < N) {
        wv[r] = __ldg(reinterpret_cast<const int4*>(w_q + size_t(n) * size_t(K)) + idx);
        scv[r] = __half2float(__ldg(w_sc + size_t(n) * size_t(blocks) + size_t(blk)));
      } else {
        wv[r] = make_int4(0, 0, 0, 0);
        scv[r] = 0.0f;
      }
    }
    #pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float4* xp = reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + size_t(idx) * 16);
      const float4 x0 = __ldg(xp + 0), x1 = __ldg(xp + 1), x2 = __ldg(xp + 2), x3 = __ldg(xp + 3);
      #pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int4 w = wv[r];
        const float d =
            (x0.x * float(int8_t(w.x)) + x0.y * float(int8_t(w.x >> 8)) +
             x0.z * float(int8_t(w.x >> 16)) + x0.w * float(int8_t(w.x >> 24))) +
            (x1.x * float(int8_t(w.y)) + x1.y * float(int8_t(w.y >> 8)) +
             x1.z * float(int8_t(w.y >> 16)) + x1.w * float(int8_t(w.y >> 24))) +
            (x2.x * float(int8_t(w.z)) + x2.y * float(int8_t(w.z >> 8)) +
             x2.z * float(int8_t(w.z >> 16)) + x2.w * float(int8_t(w.z >> 24))) +
            (x3.x * float(int8_t(w.w)) + x3.y * float(int8_t(w.w >> 8)) +
             x3.z * float(int8_t(w.w >> 16)) + x3.w * float(int8_t(w.w >> 24)));
        acc[r][m] += d * scv[r];
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    const int n = base_n + r;
    if (n < N) {
      #pragma unroll
      for (int m = 0; m < MC; ++m) {
        const float s = warp_sum(acc[r][m]);
        if (lane == 0) y[size_t(m) * size_t(N) + size_t(n)] = s;
      }
    }
  }
}

__global__ void k_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ---- 6. Short Convolutions (fused residual add, paired K/V, T-parallel prefill) ----
__global__ void k_sconv_static(const float* x, const void* w, bool w_f32, float* hist, float* out,
                               int T, int C, float scale, const float* res_add,
                               bool save_snap, float* snap, int snap_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  float h0 = hist[0 * size_t(C) + c], h1 = hist[1 * size_t(C) + c], h2 = hist[2 * size_t(C) + c];
  if (T == 1) {
    const float xt = x[c];
    const float y = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    out[c] = res_add ? (res_add[c] + y) : y;
    hist[0 * size_t(C) + c] = h1;
    hist[1 * size_t(C) + c] = h2;
    hist[2 * size_t(C) + c] = xt;
    if (save_snap) {
      snap[0 * size_t(C) + c] = h1;
      snap[1 * size_t(C) + c] = h2;
      snap[2 * size_t(C) + c] = xt;
    }
    return;
  }
  for (int t = 0; t < T; ++t) {
    const size_t idx = size_t(t) * size_t(C) + size_t(c);
    const float xt = x[idx];
    const float y = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    out[idx] = res_add ? (res_add[idx] + y) : y;
    h0 = h1; h1 = h2; h2 = xt;
    if (save_snap) {
      const size_t sb = size_t(t) * size_t(snap_stride) + size_t(c);
      snap[sb + 0 * size_t(C)] = h0;
      snap[sb + 1 * size_t(C)] = h1;
      snap[sb + 2 * size_t(C)] = h2;
    }
  }
  hist[0 * size_t(C) + c] = h0;
  hist[1 * size_t(C) + c] = h1;
  hist[2 * size_t(C) + c] = h2;
}

__global__ void k_moe_reduce8_sconv_static(
    const float* __restrict__ pair_out, const void* __restrict__ w, bool w_f32,
    float* __restrict__ hist, float* __restrict__ out,
    int T, int C, int K_top, float scale, const float* __restrict__ res_add,
    bool save_snap, float* __restrict__ snap, int snap_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  float h0 = hist[0 * size_t(C) + c], h1 = hist[1 * size_t(C) + c], h2 = hist[2 * size_t(C) + c];
  if (T == 1) {
    float xt = 0.0f;
    for (int k = 0; k < K_top; ++k) xt += pair_out[size_t(k) * size_t(C) + size_t(c)];
    const float y = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    out[c] = res_add ? (res_add[c] + y) : y;
    hist[0 * size_t(C) + c] = h1;
    hist[1 * size_t(C) + c] = h2;
    hist[2 * size_t(C) + c] = xt;
    if (save_snap) {
      snap[0 * size_t(C) + c] = h1;
      snap[1 * size_t(C) + c] = h2;
      snap[2 * size_t(C) + c] = xt;
    }
    return;
  }
  for (int t = 0; t < T; ++t) {
    const size_t idx = size_t(t) * size_t(C) + size_t(c);
    float xt = 0.0f;
    const size_t pbase = size_t(t) * size_t(K_top) * size_t(C) + size_t(c);
    for (int k = 0; k < K_top; ++k) xt += pair_out[pbase + size_t(k) * size_t(C)];
    const float y = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    out[idx] = res_add ? (res_add[idx] + y) : y;
    h0 = h1; h1 = h2; h2 = xt;
    if (save_snap) {
      const size_t sb = size_t(t) * size_t(snap_stride) + size_t(c);
      snap[sb + 0 * size_t(C)] = h0;
      snap[sb + 1 * size_t(C)] = h1;
      snap[sb + 2 * size_t(C)] = h2;
    }
  }
  hist[0 * size_t(C) + c] = h0;
  hist[1 * size_t(C) + c] = h1;
  hist[2 * size_t(C) + c] = h2;
}

__global__ void k_sconv_static_pf(const float* x_ro, const void* w, bool w_f32, const float* hist,
                                  float* out, int T, int C, float scale, const float* res_add) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  const int t0 = blockIdx.y * 64;
  const int t1 = min(T, t0 + 64);
  float h0 = (t0 >= 3) ? x_ro[size_t(t0 - 3) * size_t(C) + c] : hist[size_t(t0) * size_t(C) + c];
  float h1 = (t0 >= 2) ? x_ro[size_t(t0 - 2) * size_t(C) + c] : hist[size_t(t0 + 1) * size_t(C) + c];
  float h2 = (t0 >= 1) ? x_ro[size_t(t0 - 1) * size_t(C) + c] : hist[size_t(t0 + 2) * size_t(C) + c];
  for (int t = t0; t < t1; ++t) {
    const size_t idx = size_t(t) * size_t(C) + size_t(c);
    const float xt = x_ro[idx];
    const float y = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    out[idx] = res_add ? (res_add[idx] + y) : y;
    h0 = h1; h1 = h2; h2 = xt;
  }
}

__global__ void k_sconv_static_kv(const float* xk, const void* wk, bool wk_f32, float* hk,
                                  const float* xv, const void* wv, bool wv_f32, float* hv,
                                  int T, int C, float scale, bool save_snap, float* snap_k, float* snap_v,
                                  int snap_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const bool is_v = (blockIdx.y != 0);
  const float* x = is_v ? xv : xk;
  const void* w = is_v ? wv : wk;
  const bool w_f32 = is_v ? wv_f32 : wk_f32;
  float* hist = is_v ? hv : hk;
  float* snap = is_v ? snap_v : snap_k;
  float* out = const_cast<float*>(x);
  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  float h0 = hist[0 * size_t(C) + c], h1 = hist[1 * size_t(C) + c], h2 = hist[2 * size_t(C) + c];
  if (T == 1) {
    const float xt = x[c];
    out[c] = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    hist[0 * size_t(C) + c] = h1;
    hist[1 * size_t(C) + c] = h2;
    hist[2 * size_t(C) + c] = xt;
    if (save_snap) {
      snap[0 * size_t(C) + c] = h1;
      snap[1 * size_t(C) + c] = h2;
      snap[2 * size_t(C) + c] = xt;
    }
    return;
  }
  for (int t = 0; t < T; ++t) {
    const size_t idx = size_t(t) * size_t(C) + size_t(c);
    const float xt = x[idx];
    out[idx] = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    h0 = h1; h1 = h2; h2 = xt;
    if (save_snap) {
      const size_t sb = size_t(t) * size_t(snap_stride) + size_t(c);
      snap[sb + 0 * size_t(C)] = h0;
      snap[sb + 1 * size_t(C)] = h1;
      snap[sb + 2 * size_t(C)] = h2;
    }
  }
  hist[0 * size_t(C) + c] = h0;
  hist[1 * size_t(C) + c] = h1;
  hist[2 * size_t(C) + c] = h2;
}

__global__ void k_sconv_static_kv_pf(const float* xk_ro, const void* wk, bool wk_f32, const float* hk, float* ok,
                                     const float* xv_ro, const void* wv, bool wv_f32, const float* hv, float* ov,
                                     int T, int C, float scale) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const bool is_v = (blockIdx.z != 0);
  const float* x_ro = is_v ? xv_ro : xk_ro;
  const void* w = is_v ? wv : wk;
  const bool w_f32 = is_v ? wv_f32 : wk_f32;
  const float* hist = is_v ? hv : hk;
  float* out = is_v ? ov : ok;
  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  const int t0 = blockIdx.y * 64;
  const int t1 = min(T, t0 + 64);
  float h0 = (t0 >= 3) ? x_ro[size_t(t0 - 3) * size_t(C) + c] : hist[size_t(t0) * size_t(C) + c];
  float h1 = (t0 >= 2) ? x_ro[size_t(t0 - 2) * size_t(C) + c] : hist[size_t(t0 + 1) * size_t(C) + c];
  float h2 = (t0 >= 1) ? x_ro[size_t(t0 - 1) * size_t(C) + c] : hist[size_t(t0 + 2) * size_t(C) + c];
  for (int t = t0; t < t1; ++t) {
    const size_t idx = size_t(t) * size_t(C) + size_t(c);
    const float xt = x_ro[idx];
    out[idx] = ((w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt) + xt) * scale;
    h0 = h1; h1 = h2; h2 = xt;
  }
}

__global__ void k_sconv_dynamic(const float* x, const float* dyn, const void* static_w, bool sw_f32,
                                float* hist, float* out, int T, int C, int n_heads,
                                bool save_snap, float* snap, int snap_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int head_dim = C / n_heads, h = c / head_dim;
  float sw[4];
  for (int lag = 0; lag < 4; ++lag) {
    const size_t idx = size_t(lag) * size_t(C) + size_t(c);
    sw[lag] = sw_f32 ? static_cast<const float*>(static_w)[idx]
                     : __bfloat162float(static_cast<const __nv_bfloat16*>(static_w)[idx]);
  }
  float h0 = hist[0 * size_t(C) + c], h1 = hist[1 * size_t(C) + c], h2 = hist[2 * size_t(C) + c];
  for (int t = 0; t < T; ++t) {
    const float xt = x[size_t(t) * size_t(C) + c];
    const float4 d = *reinterpret_cast<const float4*>(dyn + (size_t(t) * size_t(n_heads) + size_t(h)) * 4);
    const float conv = (sw[0] + d.x) * xt + (sw[1] + d.y) * h2 + (sw[2] + d.z) * h1 + (sw[3] + d.w) * h0;
    out[size_t(t) * size_t(C) + c] = (conv + xt) * (1.0f / 3.0f);
    h0 = h1; h1 = h2; h2 = xt;
    if (save_snap) {
      const size_t sb = size_t(t) * size_t(snap_stride) + size_t(c);
      snap[sb + 0 * size_t(C)] = h0;
      snap[sb + 1 * size_t(C)] = h1;
      snap[sb + 2 * size_t(C)] = h2;
    }
  }
  hist[0 * size_t(C) + c] = h0;
  hist[1 * size_t(C) + c] = h1;
  hist[2 * size_t(C) + c] = h2;
}

__global__ void k_sconv_dynamic_pf(const float* x_ro, const float* dyn, const void* static_w, bool sw_f32,
                                   const float* hist, float* out, int T, int C, int n_heads) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int head_dim = C / n_heads, h = c / head_dim;
  float sw[4];
  for (int lag = 0; lag < 4; ++lag) {
    const size_t idx = size_t(lag) * size_t(C) + size_t(c);
    sw[lag] = sw_f32 ? static_cast<const float*>(static_w)[idx]
                     : __bfloat162float(static_cast<const __nv_bfloat16*>(static_w)[idx]);
  }
  const int t0 = blockIdx.y * 64;
  const int t1 = min(T, t0 + 64);
  float h0 = (t0 >= 3) ? x_ro[size_t(t0 - 3) * size_t(C) + c] : hist[size_t(t0) * size_t(C) + c];
  float h1 = (t0 >= 2) ? x_ro[size_t(t0 - 2) * size_t(C) + c] : hist[size_t(t0 + 1) * size_t(C) + c];
  float h2 = (t0 >= 1) ? x_ro[size_t(t0 - 1) * size_t(C) + c] : hist[size_t(t0 + 2) * size_t(C) + c];
  for (int t = t0; t < t1; ++t) {
    const size_t idx = size_t(t) * size_t(C) + size_t(c);
    const float xt = x_ro[idx];
    const float4 d = *reinterpret_cast<const float4*>(dyn + (size_t(t) * size_t(n_heads) + size_t(h)) * 4);
    const float conv = (sw[0] + d.x) * xt + (sw[1] + d.y) * h2 + (sw[2] + d.z) * h1 + (sw[3] + d.w) * h0;
    out[idx] = (conv + xt) * (1.0f / 3.0f);
    h0 = h1; h1 = h2; h2 = xt;
  }
}

// ---- 7. QKVR Prep (BF16 or FP32 KV cache) ----
template <bool KV_BF16>
__device__ __forceinline__ float4 kv_load4(const void* base, size_t off) {
  if constexpr (KV_BF16) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(base) + off));
    return make_float4(ink2_bf16_lo(u.x), ink2_bf16_hi(u.x), ink2_bf16_lo(u.y), ink2_bf16_hi(u.y));
  } else {
    return __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(base) + off));
  }
}

template <bool KV_BF16>
__device__ __forceinline__ void kv_store1(void* base, size_t off, float v) {
  if constexpr (KV_BF16) {
    static_cast<__nv_bfloat16*>(base)[off] = __float2bfloat16_rn(v);
  } else {
    static_cast<float*>(base)[off] = v;
  }
}

template <bool KV_BF16>
__global__ void k_qkvr_prep(float* q, const float* k, const float* v, float* r,
                            const void* q_scale, const void* r_scale, bool sc_f32,
                            void* kc, void* vc, int Hq, int Hkv, int D, int d_rel,
                            int pos0, const int32_t* d_pos_ptr, int kv_cap, bool is_swa,
                            float qk_clip, float r_clip, float v_eps, float log_floor, float log_alpha) {
  __shared__ float sh[32];
  const int kind_h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int pos = (d_pos_ptr ? *d_pos_ptr : pos0) + t;
  const int slot = is_swa ? (pos % kv_cap) : pos;
  const float tau = is_swa ? 1.0f : (1.0f + log_alpha * logf(fmaxf(1.0f, float(pos + 1) / log_floor)));

  if (kind_h < Hq) {
    const int h = kind_h;
    const size_t off = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(tid);
    const size_t soff = size_t(h) * size_t(D) + size_t(tid);
    const float sc = sc_f32 ? static_cast<const float*>(q_scale)[soff]
                            : __bfloat162float(static_cast<const __nv_bfloat16*>(q_scale)[soff]);
    const float val = q[off] * sc;
    const float ss = block_sum(val * val, sh);
    const float mult = fminf(1.0f, qk_clip / (sqrtf(ss) + 1e-6f)) * tau;
    q[off] = val * mult;
    return;
  }
  int rem = kind_h - Hq;
  if (rem < Hkv) {
    const int h = rem;
    const float val = k[(size_t(t) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid)];
    const float ss = block_sum(val * val, sh);
    const float mult = fminf(1.0f, qk_clip / (sqrtf(ss) + 1e-6f));
    kv_store1<KV_BF16>(kc, (size_t(slot) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid), val * mult);
    return;
  }
  rem -= Hkv;
  if (rem < Hkv) {
    const int h = rem;
    const float val = v[(size_t(t) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid)];
    const float ss = block_sum(val * val, sh);
    const float inv = rsqrtf(ss / float(D) + v_eps);
    kv_store1<KV_BF16>(vc, (size_t(slot) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid), val * inv);
    return;
  }
  const int h = rem - Hkv;
  const size_t off = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(d_rel) + size_t(tid);
  const size_t soff = size_t(h) * size_t(d_rel) + size_t(tid);
  float val = 0.0f;
  if (tid < d_rel) {
    const float sc = sc_f32 ? static_cast<const float*>(r_scale)[soff]
                            : __bfloat162float(static_cast<const __nv_bfloat16*>(r_scale)[soff]);
    val = r[off] * sc;
  }
  const float ss = block_sum(val * val, sh);
  if (tid < d_rel) {
    const float mult = fminf(1.0f, r_clip / (sqrtf(ss) + 1e-6f)) * tau;
    r[off] = val * mult;
  }
}

// ---- 8. Rel Bias (coalesced shared r + __ldg) ----
__global__ void k_rel_bias(const float* r, const void* rel_proj, bool proj_f32, float* rb,
                           int Hq, int d_rel, int extent) {
  __shared__ float sr[32];
  const int tid = threadIdx.x, h = blockIdx.y, t = blockIdx.z;
  const float* rv = r + (size_t(t) * size_t(Hq) + size_t(h)) * size_t(d_rel);
  if (tid < d_rel) sr[tid] = rv[tid];
  __syncthreads();
  const int col = blockIdx.x * blockDim.x + tid;
  if (col >= extent) return;
  float acc = 0.0f;
  for (int d = 0; d < d_rel; ++d) {
    const size_t pidx = size_t(d) * size_t(extent) + size_t(col);
    const float pv = proj_f32 ? __ldg(static_cast<const float*>(rel_proj) + pidx)
                              : ink2_bf16_lo(__ldg(reinterpret_cast<const uint16_t*>(rel_proj) + pidx));
    acc += sr[d] * pv;
  }
  rb[(size_t(t) * size_t(Hq) + size_t(h)) * size_t(extent) + size_t(col)] = acc;
}

// ---- 9. Attention (BF16/FP32 KV cache: T=1 decode, T=2..8 templatized verify, TC mma prefill) ----
template <bool KV_BF16>
__global__ void k_attn_decode(const float* q, const void* kc, const void* vc, const float* rb,
                              float* pm, float* pl, float* po, int pos, const int32_t* d_pos_ptr,
                              int kv_cap, int Hq, int Hkv, int D, int rel_extent, bool is_swa,
                              int swa, int chunk, int nchunks, float scale) {
  __shared__ float tgm[4], tgl[4], tgo[4][128];
  const int h = blockIdx.x, cid = blockIdx.y;
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cur_pos = d_pos_ptr ? *d_pos_ptr : pos;
  const int lk = is_swa ? min(cur_pos + 1, swa) : (cur_pos + 1);
  int eff_chunk = chunk;
  if (d_pos_ptr) {
    eff_chunk = max(32, (lk + nchunks - 1) / nchunks);
    eff_chunk = (eff_chunk + 3) / 4 * 4;
  }
  const int c0 = (cur_pos + 1 - lk) + cid * eff_chunk;
  const int c1 = min(cur_pos + 1, c0 + eff_chunk);
  if (c0 >= c1) {
    if (sg == 0 && lane == 0) {
      const size_t idx = size_t(h) * size_t(nchunks) + size_t(cid);
      pm[idx] = -INFINITY;
      pl[idx] = 0.0f;
    }
    return;
  }
  const int hkv = h / (Hq / Hkv);
  const int d0 = lane * 4;
  const float4 qv = *reinterpret_cast<const float4*>(q + size_t(h) * size_t(D) + size_t(d0));
  float mw = -INFINITY, lw = 0.0f;
  float4 ow = make_float4(0, 0, 0, 0);
  for (int kp = c0 + sg; kp < c1; kp += 4) {
    const int slot = is_swa ? (kp % kv_cap) : kp;
    const size_t koff = (size_t(slot) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
    const float4 kf = kv_load4<KV_BF16>(kc, koff);
    const float dot = qv.x * kf.x + qv.y * kf.y + qv.z * kf.z + qv.w * kf.w;
    float sc = warp_sum(dot) * scale;
    const int dist = cur_pos - kp;
    if (dist < rel_extent) sc += rb[size_t(h) * size_t(rel_extent) + size_t(dist)];
    const float4 vv = kv_load4<KV_BF16>(vc, koff);
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = fmaxf(mw, sc);
      const float e1 = expf(mw - nm), e2 = expf(sc - nm);
      ow = make_float4(ow.x * e1 + vv.x * e2, ow.y * e1 + vv.y * e2,
                       ow.z * e1 + vv.z * e2, ow.w * e1 + vv.w * e2);
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0) { tgm[sg] = mw; tgl[sg] = lw; }
  *reinterpret_cast<float4*>(&tgo[sg][d0]) = ow;
  __syncthreads();
  if (sg == 0) {
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = make_float4(0, 0, 0, 0);
    for (int s = 0; s < 4; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *reinterpret_cast<const float4*>(&tgo[s][d0]);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = fmaxf(mm, m2);
        const float e1 = expf(mm - nm), e2 = expf(m2 - nm);
        oo = make_float4(oo.x * e1 + o2.x * e2, oo.y * e1 + o2.y * e2,
                         oo.z * e1 + o2.z * e2, oo.w * e1 + o2.w * e2);
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const size_t idx = size_t(h) * size_t(nchunks) + size_t(cid);
    if (lane == 0) { pm[idx] = mm; pl[idx] = ll; }
    *reinterpret_cast<float4*>(po + idx * size_t(D) + size_t(d0)) = oo;
  }
}

__global__ void k_attn_decode_reduce(const float* pm, const float* pl, const float* po,
                                     const float* sinks, const float* gate, float* out,
                                     int D, int nchunks) {
  const int h = blockIdx.x, lane = threadIdx.x, d0 = lane * 4;
  float mm = sinks[h], ll = 1.0f;
  float4 oo = make_float4(0, 0, 0, 0);
  for (int cid = 0; cid < nchunks; ++cid) {
    const size_t idx = size_t(h) * size_t(nchunks) + size_t(cid);
    const float l2 = pl[idx];
    if (l2 == 0.0f) continue;
    const float m2 = pm[idx];
    const float4 o2 = *reinterpret_cast<const float4*>(po + idx * size_t(D) + size_t(d0));
    const float nm = fmaxf(mm, m2);
    const float e1 = expf(mm - nm), e2 = expf(m2 - nm);
    oo = make_float4(oo.x * e1 + o2.x * e2, oo.y * e1 + o2.y * e2,
                     oo.z * e1 + o2.z * e2, oo.w * e1 + o2.w * e2);
    ll = ll * e1 + l2 * e2;
    mm = nm;
  }
  const float4 g4 = *reinterpret_cast<const float4*>(gate + size_t(h) * size_t(D) + size_t(d0));
  const float invl = 1.0f / ll;
  *reinterpret_cast<float4*>(out + size_t(h) * size_t(D) + size_t(d0)) = make_float4(
      (oo.x * invl) / (1.0f + expf(-g4.x)), (oo.y * invl) / (1.0f + expf(-g4.y)),
      (oo.z * invl) / (1.0f + expf(-g4.z)), (oo.w * invl) / (1.0f + expf(-g4.w)));
}

template <int TT, bool KV_BF16>
__global__ void k_attn_verify_tm(const float* q, const void* kc, const void* vc, const float* rb,
                                 float* pm, float* pl, float* po, int pos0, const int32_t* d_pos_ptr,
                                 int kv_cap, int Hq, int Hkv, int D, int rel_extent, bool is_swa,
                                 int swa, int nchunks, float scale) {
  __shared__ float tgm[4], tgl[4], tgo[4][128];
  const int h = blockIdx.x, cid = blockIdx.y;
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cur_pos = d_pos_ptr ? *d_pos_ptr : pos0;
  const int k_min = (is_swa && cur_pos + 1 > swa) ? (cur_pos + 1 - swa) : 0;
  const int k_max = cur_pos + TT;
  const int lk = k_max - k_min;
  int eff_chunk = max(32, (lk + nchunks - 1) / nchunks);
  eff_chunk = (eff_chunk + 3) / 4 * 4;
  const int c0 = k_min + cid * eff_chunk;
  const int c1 = min(k_max, c0 + eff_chunk);
  if (c0 >= c1) {
    if (sg == 0 && lane == 0) {
      #pragma unroll
      for (int t = 0; t < TT; ++t) {
        const size_t idx = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(nchunks) + size_t(cid);
        pm[idx] = -INFINITY;
        pl[idx] = 0.0f;
      }
    }
    return;
  }

  const int hkv = h / (Hq / Hkv);
  const int d0 = lane * 4;
  float4 qv[TT], ow[TT];
  float mw[TT], lw[TT];
  #pragma unroll
  for (int t = 0; t < TT; ++t) {
    qv[t] = *reinterpret_cast<const float4*>(q + (size_t(t) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(d0));
    mw[t] = -INFINITY;
    lw[t] = 0.0f;
    ow[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  for (int kp = c0 + sg; kp < c1; kp += 4) {
    const int slot = is_swa ? (kp % kv_cap) : kp;
    const size_t koff = (size_t(slot) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
    const float4 kv = kv_load4<KV_BF16>(kc, koff);
    const float4 vv = kv_load4<KV_BF16>(vc, koff);
    #pragma unroll
    for (int t = 0; t < TT; ++t) {
      const int qpos = cur_pos + t;
      const int t_min = (is_swa && qpos + 1 > swa) ? (qpos + 1 - swa) : 0;
      if (kp < t_min || kp > qpos) continue;
      float sc = warp_sum(qv[t].x * kv.x + qv[t].y * kv.y + qv[t].z * kv.z + qv[t].w * kv.w) * scale;
      const int dist = qpos - kp;
      if (dist < rel_extent) sc += rb[(size_t(t) * size_t(Hq) + size_t(h)) * size_t(rel_extent) + size_t(dist)];
      if (lw[t] == 0.0f) {
        mw[t] = sc; lw[t] = 1.0f; ow[t] = vv;
      } else {
        const float nm = fmaxf(mw[t], sc);
        const float e1 = expf(mw[t] - nm), e2 = expf(sc - nm);
        ow[t] = make_float4(ow[t].x * e1 + vv.x * e2, ow[t].y * e1 + vv.y * e2,
                            ow[t].z * e1 + vv.z * e2, ow[t].w * e1 + vv.w * e2);
        lw[t] = lw[t] * e1 + e2;
        mw[t] = nm;
      }
    }
  }

  #pragma unroll
  for (int t = 0; t < TT; ++t) {
    if (lane == 0) { tgm[sg] = mw[t]; tgl[sg] = lw[t]; }
    *reinterpret_cast<float4*>(&tgo[sg][d0]) = ow[t];
    __syncthreads();
    if (sg == 0) {
      float mm = -INFINITY, ll = 0.0f;
      float4 oo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int s = 0; s < 4; ++s) {
        const float l2 = tgl[s];
        if (l2 == 0.0f) continue;
        const float m2 = tgm[s];
        const float4 o2 = *reinterpret_cast<const float4*>(&tgo[s][d0]);
        if (ll == 0.0f) { mm = m2; ll = l2; oo = o2; }
        else {
          const float nm = fmaxf(mm, m2);
          const float e1 = expf(mm - nm), e2 = expf(m2 - nm);
          oo = make_float4(oo.x * e1 + o2.x * e2, oo.y * e1 + o2.y * e2,
                           oo.z * e1 + o2.z * e2, oo.w * e1 + o2.w * e2);
          ll = ll * e1 + l2 * e2;
          mm = nm;
        }
      }
      const size_t out_idx = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(nchunks) + size_t(cid);
      if (lane == 0) { pm[out_idx] = mm; pl[out_idx] = ll; }
      *reinterpret_cast<float4*>(po + out_idx * size_t(D) + size_t(d0)) = oo;
    }
    __syncthreads();
  }
}

__global__ void k_attn_verify_reduce(const float* pm, const float* pl, const float* po,
                                     const float* sinks, const float* gate, float* out,
                                     int Hq, int D, int nchunks) {
  const int h = blockIdx.x, t = blockIdx.y, d0 = (threadIdx.x & 31) * 4;
  float mm = sinks[size_t(t) * size_t(Hq) + size_t(h)], ll = 1.0f;
  float4 oo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const size_t row_base = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(nchunks);
  for (int cid = 0; cid < nchunks; ++cid) {
    const size_t idx = row_base + size_t(cid);
    const float l2 = pl[idx];
    if (l2 == 0.0f) continue;
    const float m2 = pm[idx];
    const float4 o2 = *reinterpret_cast<const float4*>(po + idx * size_t(D) + size_t(d0));
    const float nm = fmaxf(mm, m2);
    const float e1 = expf(mm - nm), e2 = expf(m2 - nm);
    oo = make_float4(oo.x * e1 + o2.x * e2, oo.y * e1 + o2.y * e2,
                     oo.z * e1 + o2.z * e2, oo.w * e1 + o2.w * e2);
    ll = ll * e1 + l2 * e2;
    mm = nm;
  }
  const size_t qoff = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(d0);
  const float4 g4 = *reinterpret_cast<const float4*>(gate + qoff);
  const float invl = 1.0f / ll;
  *reinterpret_cast<float4*>(out + qoff) = make_float4(
      (oo.x * invl) / (1.0f + expf(-g4.x)), (oo.y * invl) / (1.0f + expf(-g4.y)),
      (oo.z * invl) / (1.0f + expf(-g4.z)), (oo.w * invl) / (1.0f + expf(-g4.w)));
}

template <bool KV_BF16>
__global__ void k_attn_prefill(const float* q, const void* kc, const void* vc, const float* rb,
                               const float* sinks, const float* gate, float* out, int pos0, int kv_cap,
                               int Hq, int Hkv, int D, int rel_extent, bool is_swa, int swa, float scale) {
  __shared__ float tgm[4], tgl[4], tgo[4][128];
  const int h = blockIdx.x, t = blockIdx.y;
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int hkv = h / (Hq / Hkv), qpos = pos0 + t, d0 = lane * 4;
  const size_t qoff = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(d0);
  const float4 qv = *reinterpret_cast<const float4*>(q + qoff);
  const float* rbh = rb + (size_t(t) * size_t(Hq) + size_t(h)) * size_t(rel_extent);
  const int lk = is_swa ? min(qpos + 1, swa) : (qpos + 1);
  const int c0 = qpos + 1 - lk;
  float mw = -INFINITY, lw = 0.0f;
  float4 ow = make_float4(0, 0, 0, 0);
  for (int kp = c0 + sg; kp <= qpos; kp += 4) {
    const int slot = is_swa ? (kp % kv_cap) : kp;
    const size_t koff = (size_t(slot) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
    const float4 kf = kv_load4<KV_BF16>(kc, koff);
    const float dot = qv.x * kf.x + qv.y * kf.y + qv.z * kf.z + qv.w * kf.w;
    float sc = warp_sum(dot) * scale;
    const int dist = qpos - kp;
    if (dist < rel_extent) sc += rbh[dist];
    const float4 vv = kv_load4<KV_BF16>(vc, koff);
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = fmaxf(mw, sc);
      const float e1 = expf(mw - nm), e2 = expf(sc - nm);
      ow = make_float4(ow.x * e1 + vv.x * e2, ow.y * e1 + vv.y * e2,
                       ow.z * e1 + vv.z * e2, ow.w * e1 + vv.w * e2);
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0) { tgm[sg] = mw; tgl[sg] = lw; }
  *reinterpret_cast<float4*>(&tgo[sg][d0]) = ow;
  __syncthreads();
  if (sg == 0) {
    float mm = sinks[size_t(t) * size_t(Hq) + size_t(h)], ll = 1.0f;
    float4 oo = make_float4(0, 0, 0, 0);
    for (int s = 0; s < 4; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *reinterpret_cast<const float4*>(&tgo[s][d0]);
      const float nm = fmaxf(mm, m2);
      const float e1 = expf(mm - nm), e2 = expf(m2 - nm);
      oo = make_float4(oo.x * e1 + o2.x * e2, oo.y * e1 + o2.y * e2,
                       oo.z * e1 + o2.z * e2, oo.w * e1 + o2.w * e2);
      ll = ll * e1 + l2 * e2;
      mm = nm;
    }
    const float4 g4 = *reinterpret_cast<const float4*>(gate + qoff);
    const float invl = 1.0f / ll;
    *reinterpret_cast<float4*>(out + qoff) = make_float4(
        (oo.x * invl) / (1.0f + expf(-g4.x)), (oo.y * invl) / (1.0f + expf(-g4.y)),
        (oo.z * invl) / (1.0f + expf(-g4.z)), (oo.w * invl) / (1.0f + expf(-g4.w)));
  }
}

// Blackwell Tensor Core BF16 FlashAttention prefill (BM_Q=16, BN_KV=32, GQA=3, D=128) reading BF16/FP32 KV cache
template <bool KV_BF16>
__global__ void k_attn_prefill_tc(const float* q, const void* kc, const void* vc, const float* rb,
                                  const float* sinks, const float* gate, float* out, int T, int pos0, int kv_cap,
                                  int Hq, int Hkv, int rel_extent, bool is_swa, int swa, float scale) {
  __shared__ __nv_bfloat16 sh_K[32][136];
  __shared__ __nv_bfloat16 sh_V[128][34];
  const int t0 = blockIdx.x * 16;
  const int hkv = blockIdx.y;
  const int tid = threadIdx.x;  // 96 threads = 3 warps (1 per GQA Q head)
  const int wg = tid >> 5, lane = tid & 31;
  const int h = hkv * 3 + wg;
  const int gid = lane >> 2, t4 = lane & 3;
  const int r0 = t0 + gid, r1 = t0 + gid + 8;

  uint32_t q_reg[8][4];
  #pragma unroll
  for (int kd = 0; kd < 8; ++kd) {
    const int k0 = kd * 16;
    if (r0 < T) {
      const float* qr0 = q + (size_t(r0) * size_t(Hq) + size_t(h)) * 128 + k0;
      q_reg[kd][0] = pack_bf2_f32(qr0[t4 * 2], qr0[t4 * 2 + 1]);
      q_reg[kd][2] = pack_bf2_f32(qr0[8 + t4 * 2], qr0[8 + t4 * 2 + 1]);
    } else {
      q_reg[kd][0] = 0u; q_reg[kd][2] = 0u;
    }
    if (r1 < T) {
      const float* qr1 = q + (size_t(r1) * size_t(Hq) + size_t(h)) * 128 + k0;
      q_reg[kd][1] = pack_bf2_f32(qr1[t4 * 2], qr1[t4 * 2 + 1]);
      q_reg[kd][3] = pack_bf2_f32(qr1[8 + t4 * 2], qr1[8 + t4 * 2 + 1]);
    } else {
      q_reg[kd][1] = 0u; q_reg[kd][3] = 0u;
    }
  }

  const int qpos0 = pos0 + t0;
  const int qpos_last = pos0 + min(T, t0 + 16) - 1;
  const int kp_start = is_swa ? max(0, qpos0 + 1 - swa) : 0;
  const int kp_end = qpos_last;
  const int qp0 = pos0 + r0, qp1 = pos0 + r1;

  float m0 = (r0 < T) ? sinks[size_t(r0) * size_t(Hq) + size_t(h)] : 0.0f;
  float m1 = (r1 < T) ? sinks[size_t(r1) * size_t(Hq) + size_t(h)] : 0.0f;
  float l0 = 1.0f, l1 = 1.0f;
  float O[16][4];
  #pragma unroll
  for (int nt = 0; nt < 16; ++nt) O[nt][0] = O[nt][1] = O[nt][2] = O[nt][3] = 0.0f;

  const float* rbp0 = (r0 < T) ? (rb + (size_t(r0) * size_t(Hq) + size_t(h)) * size_t(rel_extent)) : nullptr;
  const float* rbp1 = (r1 < T) ? (rb + (size_t(r1) * size_t(Hq) + size_t(h)) * size_t(rel_extent)) : nullptr;

  for (int kp_blk = kp_start; kp_blk <= kp_end; kp_blk += 32) {
    // Load 32x128 FP32 K and V across 96 threads (1024 float4s), packing into BF16 SMEM
    for (int idx = tid; idx < 1024; idx += 96) {
      const int kp_loc = idx >> 5;
      const int d4 = (idx & 31) * 4;
      const int kp = kp_blk + kp_loc;
      if (kp <= kp_end) {
        const int slot = is_swa ? (kp % kv_cap) : kp;
        const size_t koff = (size_t(slot) * size_t(Hkv) + size_t(hkv)) * 128 + size_t(d4);
        if constexpr (KV_BF16) {
          const uint2 kv_u = __ldg(reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(kc) + koff));
          uint32_t* sk = reinterpret_cast<uint32_t*>(&sh_K[kp_loc][d4]);
          sk[0] = kv_u.x;
          sk[1] = kv_u.y;
          const uint2 vv_u = __ldg(reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(vc) + koff));
          const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vv_u);
          sh_V[d4 + 0][kp_loc] = vb[0];
          sh_V[d4 + 1][kp_loc] = vb[1];
          sh_V[d4 + 2][kp_loc] = vb[2];
          sh_V[d4 + 3][kp_loc] = vb[3];
        } else {
          const float4 kv = *reinterpret_cast<const float4*>(static_cast<const float*>(kc) + koff);
          uint32_t* sk = reinterpret_cast<uint32_t*>(&sh_K[kp_loc][d4]);
          sk[0] = pack_bf2_f32(kv.x, kv.y);
          sk[1] = pack_bf2_f32(kv.z, kv.w);
          const float4 vv = *reinterpret_cast<const float4*>(static_cast<const float*>(vc) + koff);
          sh_V[d4 + 0][kp_loc] = __float2bfloat16_rn(vv.x);
          sh_V[d4 + 1][kp_loc] = __float2bfloat16_rn(vv.y);
          sh_V[d4 + 2][kp_loc] = __float2bfloat16_rn(vv.z);
          sh_V[d4 + 3][kp_loc] = __float2bfloat16_rn(vv.w);
        }
      } else {
        *reinterpret_cast<uint2*>(&sh_K[kp_loc][d4]) = make_uint2(0u, 0u);
        sh_V[d4 + 0][kp_loc] = __float2bfloat16_rn(0.0f);
        sh_V[d4 + 1][kp_loc] = __float2bfloat16_rn(0.0f);
        sh_V[d4 + 2][kp_loc] = __float2bfloat16_rn(0.0f);
        sh_V[d4 + 3][kp_loc] = __float2bfloat16_rn(0.0f);
      }
    }
    __syncthreads();

    float S[4][4];
    #pragma unroll
    for (int kt = 0; kt < 4; ++kt) S[kt][0] = S[kt][1] = S[kt][2] = S[kt][3] = 0.0f;
    #pragma unroll
    for (int kd = 0; kd < 8; ++kd) {
      const int k0 = kd * 16;
      #pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        uint32_t b_frag[2];
        const int br = kt * 8 + (lane & 7);
        const int bc = k0 + ((lane & 8) ? 8 : 0);
        ldmatrix_x2(b_frag, &sh_K[br][bc]);
        mma_m16n8k16_bf16(S[kt], q_reg[kd], b_frag, S[kt]);
      }
    }

    #pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const int c0 = kp_blk + kt * 8 + t4 * 2;
      const int c1 = c0 + 1;
      if (r0 < T && c0 <= qp0 && (!is_swa || (qp0 - c0) < swa)) {
        const int d = qp0 - c0;
        S[kt][0] = S[kt][0] * scale + (d < rel_extent ? rbp0[d] : 0.0f);
      } else {
        S[kt][0] = -INFINITY;
      }
      if (r0 < T && c1 <= qp0 && (!is_swa || (qp0 - c1) < swa)) {
        const int d = qp0 - c1;
        S[kt][1] = S[kt][1] * scale + (d < rel_extent ? rbp0[d] : 0.0f);
      } else {
        S[kt][1] = -INFINITY;
      }
      if (r1 < T && c0 <= qp1 && (!is_swa || (qp1 - c0) < swa)) {
        const int d = qp1 - c0;
        S[kt][2] = S[kt][2] * scale + (d < rel_extent ? rbp1[d] : 0.0f);
      } else {
        S[kt][2] = -INFINITY;
      }
      if (r1 < T && c1 <= qp1 && (!is_swa || (qp1 - c1) < swa)) {
        const int d = qp1 - c1;
        S[kt][3] = S[kt][3] * scale + (d < rel_extent ? rbp1[d] : 0.0f);
      } else {
        S[kt][3] = -INFINITY;
      }
    }

    float rm0 = fmaxf(fmaxf(S[0][0], S[0][1]), fmaxf(S[1][0], S[1][1]));
    rm0 = fmaxf(rm0, fmaxf(fmaxf(S[2][0], S[2][1]), fmaxf(S[3][0], S[3][1])));
    rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 1));
    rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 2));

    float rm1 = fmaxf(fmaxf(S[0][2], S[0][3]), fmaxf(S[1][2], S[1][3]));
    rm1 = fmaxf(rm1, fmaxf(fmaxf(S[2][2], S[2][3]), fmaxf(S[3][2], S[3][3])));
    rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 1));
    rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 2));

    const float nm0 = fmaxf(m0, rm0), nm1 = fmaxf(m1, rm1);
    const float eo0 = expf(m0 - nm0), eo1 = expf(m1 - nm1);

    float lp0 = 0.0f, lp1 = 0.0f;
    #pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      S[kt][0] = expf(S[kt][0] - nm0);
      S[kt][1] = expf(S[kt][1] - nm0);
      S[kt][2] = expf(S[kt][2] - nm1);
      S[kt][3] = expf(S[kt][3] - nm1);
      lp0 += S[kt][0] + S[kt][1];
      lp1 += S[kt][2] + S[kt][3];
    }
    lp0 += __shfl_xor_sync(0xffffffffu, lp0, 1);
    lp0 += __shfl_xor_sync(0xffffffffu, lp0, 2);
    lp1 += __shfl_xor_sync(0xffffffffu, lp1, 1);
    lp1 += __shfl_xor_sync(0xffffffffu, lp1, 2);

    l0 = l0 * eo0 + lp0;
    l1 = l1 * eo1 + lp1;
    m0 = nm0; m1 = nm1;

    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      O[nt][0] *= eo0; O[nt][1] *= eo0;
      O[nt][2] *= eo1; O[nt][3] *= eo1;
    }

    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t a_frag[4];
      const int k0_p = ks * 2;
      a_frag[0] = pack_bf2_f32(S[k0_p][0], S[k0_p][1]);
      a_frag[1] = pack_bf2_f32(S[k0_p][2], S[k0_p][3]);
      a_frag[2] = pack_bf2_f32(S[k0_p + 1][0], S[k0_p + 1][1]);
      a_frag[3] = pack_bf2_f32(S[k0_p + 1][2], S[k0_p + 1][3]);
      const int kp_base = ks * 16;
      #pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        uint32_t b_frag[2];
        const int d_row = nt * 8 + gid;
        memcpy(&b_frag[0], &sh_V[d_row][kp_base + t4 * 2], 4);
        memcpy(&b_frag[1], &sh_V[d_row][kp_base + 8 + t4 * 2], 4);
        mma_m16n8k16_bf16(O[nt], a_frag, b_frag, O[nt]);
      }
    }
    __syncthreads();
  }

  const float invl0 = 1.0f / l0, invl1 = 1.0f / l1;
  if (r0 < T) {
    const size_t base0 = (size_t(r0) * size_t(Hq) + size_t(h)) * 128 + size_t(t4 * 2);
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const float g0 = gate[base0 + nt * 8 + 0], g1 = gate[base0 + nt * 8 + 1];
      out[base0 + nt * 8 + 0] = (O[nt][0] * invl0) / (1.0f + expf(-g0));
      out[base0 + nt * 8 + 1] = (O[nt][1] * invl0) / (1.0f + expf(-g1));
    }
  }
  if (r1 < T) {
    const size_t base1 = (size_t(r1) * size_t(Hq) + size_t(h)) * 128 + size_t(t4 * 2);
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const float g0 = gate[base1 + nt * 8 + 0], g1 = gate[base1 + nt * 8 + 1];
      out[base1 + nt * 8 + 0] = (O[nt][2] * invl1) / (1.0f + expf(-g0));
      out[base1 + nt * 8 + 1] = (O[nt][3] * invl1) / (1.0f + expf(-g1));
    }
  }
}

// ---- 10. MoE Router & Experts ----
__global__ void k_moe_route(const float* logits, const void* bias, bool bias_f32,
                            uint32_t* eid, float* ew, int E, int K, float route_scale) {
  __shared__ float ss[128], sh_v[4], sh_scale;
  __shared__ uint32_t sh_i[4], sh_win;
  const int t = blockIdx.x, tid = threadIdx.x, sg = tid >> 5, lane = tid & 31;
  const float l = logits[size_t(t) * size_t(E) + size_t(tid)];
  const float s = 1.0f / (1.0f + expf(-l));
  const float b = bias_f32 ? static_cast<const float*>(bias)[tid]
                           : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[tid]);
  ss[tid] = s;
  float v = s + b;
  __syncthreads();
  uint32_t chosen_reg = 0;
  float wsum = 0.0f;
  for (int slot = 0; slot < K; ++slot) {
    float bv = v;
    uint32_t bi = uint32_t(tid);
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { sh_v[sg] = bv; sh_i[sg] = bi; }
    __syncthreads();
    if (sg == 0) {
      float gv = lane < 4 ? sh_v[lane] : -INFINITY;
      uint32_t gi = lane < 4 ? sh_i[lane] : 0xffffffffu;
      for (int o = 2; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xfu, gv, o);
        const uint32_t oi = __shfl_xor_sync(0xfu, gi, o);
        if (ov > gv || (ov == gv && oi < gi)) { gv = ov; gi = oi; }
      }
      if (lane == 0) sh_win = gi;
    }
    __syncthreads();
    const uint32_t win = sh_win;
    if (tid == int(win)) v = -INFINITY;
    if (tid == slot) chosen_reg = win;
    if (tid == 0) wsum += ss[win];
    __syncthreads();
  }
  if (tid == 0) sh_scale = route_scale / (wsum + 1e-20f);
  __syncthreads();
  if (tid < K) {
    eid[size_t(t) * size_t(K) + size_t(tid)] = chosen_reg;
    ew[size_t(t) * size_t(K) + size_t(tid)] = ss[chosen_reg] * sh_scale;
  }
}

__device__ __forceinline__ float nvfp4_bit_dot8(uint32_t w, float4 a, float4 b) {
  return e2m1_bit(w) * a.x + e2m1_bit(w >> 4) * a.y +
         e2m1_bit(w >> 8) * a.z + e2m1_bit(w >> 12) * a.w +
         e2m1_bit(w >> 16) * b.x + e2m1_bit(w >> 20) * b.y +
         e2m1_bit(w >> 24) * b.z + e2m1_bit(w >> 28) * b.w;
}

__device__ __forceinline__ float mxfp8_dot4(uint32_t w, const float4 x) {
  return ink2_e4m3(uint8_t(w)) * x.x + ink2_e4m3(uint8_t(w >> 8)) * x.y +
         ink2_e4m3(uint8_t(w >> 16)) * x.z + ink2_e4m3(uint8_t(w >> 24)) * x.w;
}

__global__ void k_moe_reduce8(const float4* pair_out, float4* out, int H, int K_top) {
  const int h4 = H >> 2, col = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (col >= h4) return;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int k = 0; k < K_top; ++k) {
    const float4 v = pair_out[(size_t(t) * size_t(K_top) + size_t(k)) * size_t(h4) + size_t(col)];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  out[size_t(t) * size_t(h4) + size_t(col)] = acc;
}

__global__ void k_moe_route_bucket_small(
    const float* __restrict__ logits, const void* __restrict__ bias, bool bias_f32,
    uint32_t* __restrict__ eid, float* __restrict__ ew,
    int32_t* __restrict__ tile_exp, int32_t* __restrict__ tile_m0, int32_t* __restrict__ tile_cnt,
    int32_t* __restrict__ num_tiles, int32_t* __restrict__ st_tok, int32_t* __restrict__ st_pair,
    float* __restrict__ st_ew, int T, int K_top, int E, float route_scale) {
  __shared__ float sh_b[128], sh_ss[12][128], sh_ew[128];
  __shared__ uint32_t sh_eid[128];
  __shared__ int32_t sh_off[4], sh_act[4];
  const int tid = threadIdx.x, sg = tid >> 5, lane = tid & 31;
  if (tid < 128) {
    sh_b[tid] = bias_f32 ? static_cast<const float*>(bias)[tid]
                         : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[tid]);
  }
  __syncthreads();

  if (sg < T) {
    const float* lg_t = logits + size_t(sg) * size_t(E);
    const int e0 = lane, e1 = lane + 32, e2 = lane + 64, e3 = lane + 96;
    const float s0 = 1.0f / (1.0f + expf(-lg_t[e0]));
    const float s1 = 1.0f / (1.0f + expf(-lg_t[e1]));
    const float s2 = 1.0f / (1.0f + expf(-lg_t[e2]));
    const float s3 = 1.0f / (1.0f + expf(-lg_t[e3]));
    sh_ss[sg][e0] = s0; sh_ss[sg][e1] = s1; sh_ss[sg][e2] = s2; sh_ss[sg][e3] = s3;
    __syncwarp();
    float v0 = s0 + sh_b[e0], v1 = s1 + sh_b[e1], v2 = s2 + sh_b[e2], v3 = s3 + sh_b[e3];
    uint32_t chosen_reg = 0;
    float wsum = 0.0f;
    for (int slot = 0; slot < K_top; ++slot) {
      float bv = v0;
      uint32_t bi = uint32_t(e0);
      if (v1 > bv || (v1 == bv && uint32_t(e1) < bi)) { bv = v1; bi = uint32_t(e1); }
      if (v2 > bv || (v2 == bv && uint32_t(e2) < bi)) { bv = v2; bi = uint32_t(e2); }
      if (v3 > bv || (v3 == bv && uint32_t(e3) < bi)) { bv = v3; bi = uint32_t(e3); }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      const uint32_t win = bi;
      if (uint32_t(e0) == win) v0 = -INFINITY;
      if (uint32_t(e1) == win) v1 = -INFINITY;
      if (uint32_t(e2) == win) v2 = -INFINITY;
      if (uint32_t(e3) == win) v3 = -INFINITY;
      if (lane == slot) chosen_reg = win;
      if (lane == 0) wsum += sh_ss[sg][win];
    }
    const float total_w = __shfl_sync(0xffffffffu, wsum, 0);
    const float scale = route_scale / (total_w + 1e-20f);
    if (lane < K_top) {
      const int p = sg * K_top + lane;
      const float wv = sh_ss[sg][chosen_reg] * scale;
      sh_eid[p] = chosen_reg;
      sh_ew[p] = wv;
      eid[p] = chosen_reg;
      ew[p] = wv;
    }
  }
  __syncthreads();

  const int r_tot = T * K_top;
  int32_t count = 0;
  int32_t is_act = 0;
  int32_t p_incl = 0, e_incl = 0;
  if (tid < 128) {
    if (tid < E) {
      for (int i = 0; i < r_tot; ++i) count += (sh_eid[i] == uint32_t(tid)) ? 1 : 0;
    }
    is_act = (count > 0) ? 1 : 0;
    p_incl = count;
    e_incl = is_act;
    #pragma unroll
    for (int off = 1; off <= 16; off <<= 1) {
      const int32_t up = __shfl_up_sync(0xffffffffu, p_incl, off);
      const int32_t ue = __shfl_up_sync(0xffffffffu, e_incl, off);
      if (lane >= off) { p_incl += up; e_incl += ue; }
    }
    if (lane == 31) { sh_off[sg] = p_incl; sh_act[sg] = e_incl; }
  }
  __syncthreads();
  if (tid < 128) {
    int32_t p_base = 0, e_base = 0;
    for (int s = 0; s < sg; ++s) { p_base += sh_off[s]; e_base += sh_act[s]; }
    const int32_t p_pos0 = p_base + (p_incl - count), e_pos = e_base + (e_incl - is_act);
    if (is_act) {
      tile_exp[e_pos] = tid;
      tile_m0[e_pos] = p_pos0;
      tile_cnt[e_pos] = count;
      int32_t wp = p_pos0;
      for (int i = 0; i < r_tot; ++i) {
        if (sh_eid[i] == uint32_t(tid)) {
          st_tok[wp] = i / K_top;
          st_pair[wp] = i;
          st_ew[wp] = sh_ew[i];
          ++wp;
        }
      }
    }
    if (tid == 0) *num_tiles = sh_act[0] + sh_act[1] + sh_act[2] + sh_act[3];
  }
}

// SMEM __half2 + hardware-decode T=1 NVFP4/MXFP8 kernels (Strata-inspired 128B-coalesced uint4 + LDS.128)
__global__ void __launch_bounds__(128) k_moe_nvfp4_w13_t1_h2(
    const float* __restrict__ x, const uint32_t* __restrict__ eid,
    const uint8_t* __restrict__ w13, const uint8_t* __restrict__ sc,
    const float* __restrict__ outer, float* __restrict__ act, int H, int I) {
  const int tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, feat = blockIdx.x * 16 + sub;
  const int slot = blockIdx.y;
  __shared__ alignas(16) __half2 s_x[1152];
  const float2* src2 = reinterpret_cast<const float2*>(x);
  for (int i = tid; i < (H >> 1); i += 128) {
    const float2 f = src2[i];
    s_x[i] = __floats2half2_rn(f.x, f.y);
  }
  __syncthreads();
  if (feat >= I) return;
  const uint32_t e = eid[slot];
  const int two_i = 2 * I, half_h = H / 2, blocks = H / 16, blocks2 = blocks >> 1;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * feat), wu_row = wg_row + 1;
  const uint4* wgp4 = reinterpret_cast<const uint4*>(w13 + wg_row * size_t(half_h));
  const uint4* wup4 = reinterpret_cast<const uint4*>(w13 + wu_row * size_t(half_h));
  const uint16_t* scg2 = reinterpret_cast<const uint16_t*>(sc + wg_row * size_t(blocks));
  const uint16_t* scu2 = reinterpret_cast<const uint16_t*>(sc + wu_row * size_t(blocks));
  float ag = 0.0f, au = 0.0f;
  for (int i = lg; i < blocks2; i += 8) {
    const uint4 gv = __ldg(wgp4 + i), uv = __ldg(wup4 + i);
    const float2 sg = ink2_e4m3x2(__ldg(scg2 + i));
    const float2 su = ink2_e4m3x2(__ldg(scu2 + i));
    __half2 gw[8], uw[8];
    const __half2* xh = s_x + i * 16;
    sf4_decode16_h2(gv.x, gv.y, gw);
    sf4_decode16_h2(uv.x, uv.y, uw);
    ag = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag);
    au = fmaf(dot16_h2_2acc(uw, xh), su.x, au);
    sf4_decode16_h2(gv.z, gv.w, gw);
    sf4_decode16_h2(uv.z, uv.w, uw);
    ag = fmaf(dot16_h2_2acc(gw, xh + 8), sg.y, ag);
    au = fmaf(dot16_h2_2acc(uw, xh + 8), su.y, au);
  }
  ag += __shfl_xor_sync(0xffffffffu, ag, 4); au += __shfl_xor_sync(0xffffffffu, au, 4);
  ag += __shfl_xor_sync(0xffffffffu, ag, 2); au += __shfl_xor_sync(0xffffffffu, au, 2);
  ag += __shfl_xor_sync(0xffffffffu, ag, 1); au += __shfl_xor_sync(0xffffffffu, au, 1);
  if (lg == 0) {
    const float os = outer[e];
    act[size_t(slot) * size_t(I) + size_t(feat)] = situ_f32(ag * os, au * os);
  }
}

__global__ void __launch_bounds__(128) k_moe_nvfp4_w2_t1_h2(
    const float* __restrict__ act, const uint32_t* __restrict__ eid, const float* __restrict__ ew,
    const uint8_t* __restrict__ w2, const uint8_t* __restrict__ sc, const float* __restrict__ outer,
    float* __restrict__ pair_out, int H, int I) {
  const int tid = threadIdx.x, slot = blockIdx.y, sub = tid >> 3, lg = tid & 7;
  const int pair_r = blockIdx.x * 16 + sub;
  __shared__ alignas(16) __half2 s_h[576];
  const float2* src2 = reinterpret_cast<const float2*>(act + size_t(slot) * size_t(I));
  for (int i = tid; i < (I >> 1); i += 128) {
    const float2 f = src2[i];
    s_h[i] = __floats2half2_rn(f.x, f.y);
  }
  __syncthreads();
  if (2 * pair_r >= H) return;
  const uint32_t e = eid[slot];
  const int half_i = I / 2, blocks = I / 16, blocks2 = blocks >> 1;
  const size_t wrow0 = size_t(e) * size_t(H) + size_t(2 * pair_r), wrow1 = wrow0 + 1;
  const uint4* p0 = reinterpret_cast<const uint4*>(w2 + wrow0 * size_t(half_i));
  const uint4* p1 = reinterpret_cast<const uint4*>(w2 + wrow1 * size_t(half_i));
  const uint16_t* s0 = reinterpret_cast<const uint16_t*>(sc + wrow0 * size_t(blocks));
  const uint16_t* s1 = reinterpret_cast<const uint16_t*>(sc + wrow1 * size_t(blocks));
  float y0 = 0.0f, y1 = 0.0f;
  for (int i = lg; i < blocks2; i += 8) {
    const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
    const float2 sc0 = ink2_e4m3x2(__ldg(s0 + i));
    const float2 sc1 = ink2_e4m3x2(__ldg(s1 + i));
    __half2 w0v[8], w1v[8];
    const __half2* hh = s_h + i * 16;
    sf4_decode16_h2(v0.x, v0.y, w0v);
    sf4_decode16_h2(v1.x, v1.y, w1v);
    y0 = fmaf(dot16_h2_2acc(w0v, hh), sc0.x, y0);
    y1 = fmaf(dot16_h2_2acc(w1v, hh), sc1.x, y1);
    sf4_decode16_h2(v0.z, v0.w, w0v);
    sf4_decode16_h2(v1.z, v1.w, w1v);
    y0 = fmaf(dot16_h2_2acc(w0v, hh + 8), sc0.y, y0);
    y1 = fmaf(dot16_h2_2acc(w1v, hh + 8), sc1.y, y1);
  }
  y0 += __shfl_xor_sync(0xffffffffu, y0, 4); y1 += __shfl_xor_sync(0xffffffffu, y1, 4);
  y0 += __shfl_xor_sync(0xffffffffu, y0, 2); y1 += __shfl_xor_sync(0xffffffffu, y1, 2);
  y0 += __shfl_xor_sync(0xffffffffu, y0, 1); y1 += __shfl_xor_sync(0xffffffffu, y1, 1);
  if (lg == 0) {
    const float rwo = ew[slot] * outer[e];
    *reinterpret_cast<float2*>(pair_out + size_t(slot) * size_t(H) + size_t(2 * pair_r)) =
        make_float2(y0 * rwo, y1 * rwo);
  }
}

__global__ void __launch_bounds__(128) k_moe_mxfp8_w13_t1_h2(
    const float* __restrict__ x, const uint32_t* __restrict__ eid,
    const uint8_t* __restrict__ w13, const uint8_t* __restrict__ sc,
    float* __restrict__ act, int H, int I) {
  const int tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, feat = blockIdx.x * 16 + sub;
  const int slot = blockIdx.y;
  __shared__ alignas(16) __half2 s_x[1152];
  const float2* src2 = reinterpret_cast<const float2*>(x);
  for (int i = tid; i < (H >> 1); i += 128) {
    const float2 f = src2[i];
    s_x[i] = __floats2half2_rn(f.x, f.y);
  }
  __syncthreads();
  if (feat >= I) return;
  const uint32_t e = eid[slot];
  const int two_i = 2 * I, blocks = H / 32, n16 = H / 16;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * feat), wu_row = wg_row + 1;
  const uint4* wgp4 = reinterpret_cast<const uint4*>(w13 + wg_row * size_t(H));
  const uint4* wup4 = reinterpret_cast<const uint4*>(w13 + wu_row * size_t(H));
  const uint8_t* scgp = sc + wg_row * size_t(blocks);
  const uint8_t* scup = sc + wu_row * size_t(blocks);
  float ag = 0.0f, au = 0.0f;
  for (int i = lg; i < n16; i += 8) {
    const float sg = ink2_ue8m0(__ldg(scgp + (i >> 1)));
    const float su = ink2_ue8m0(__ldg(scup + (i >> 1)));
    __half2 gw[8], uw[8];
    mx8_decode16_h2(__ldg(wgp4 + i), gw);
    mx8_decode16_h2(__ldg(wup4 + i), uw);
    const __half2* xh = s_x + i * 8;
    ag = fmaf(dot16_h2_2acc(gw, xh), sg, ag);
    au = fmaf(dot16_h2_2acc(uw, xh), su, au);
  }
  ag += __shfl_xor_sync(0xffffffffu, ag, 4); au += __shfl_xor_sync(0xffffffffu, au, 4);
  ag += __shfl_xor_sync(0xffffffffu, ag, 2); au += __shfl_xor_sync(0xffffffffu, au, 2);
  ag += __shfl_xor_sync(0xffffffffu, ag, 1); au += __shfl_xor_sync(0xffffffffu, au, 1);
  if (lg == 0) act[size_t(slot) * size_t(I) + size_t(feat)] = situ_f32(ag, au);
}

__global__ void __launch_bounds__(128) k_moe_mxfp8_w2_t1_h2(
    const float* __restrict__ act, const uint32_t* __restrict__ eid, const float* __restrict__ ew,
    const uint8_t* __restrict__ w2, const uint8_t* __restrict__ sc, float* __restrict__ pair_out, int H, int I) {
  const int tid = threadIdx.x, slot = blockIdx.y, sub = tid >> 3, lg = tid & 7;
  const int pair_r = blockIdx.x * 16 + sub;
  __shared__ alignas(16) __half2 s_h[576];
  const float2* src2 = reinterpret_cast<const float2*>(act + size_t(slot) * size_t(I));
  for (int i = tid; i < (I >> 1); i += 128) {
    const float2 f = src2[i];
    s_h[i] = __floats2half2_rn(f.x, f.y);
  }
  __syncthreads();
  if (2 * pair_r >= H) return;
  const uint32_t e = eid[slot];
  const int blocks = I / 32, n16 = I / 16;
  const size_t wrow0 = size_t(e) * size_t(H) + size_t(2 * pair_r), wrow1 = wrow0 + 1;
  const uint4* p0 = reinterpret_cast<const uint4*>(w2 + wrow0 * size_t(I));
  const uint4* p1 = reinterpret_cast<const uint4*>(w2 + wrow1 * size_t(I));
  const uint8_t* s0 = sc + wrow0 * size_t(blocks);
  const uint8_t* s1 = sc + wrow1 * size_t(blocks);
  float y0 = 0.0f, y1 = 0.0f;
  for (int i = lg; i < n16; i += 8) {
    const float sc0 = ink2_ue8m0(__ldg(s0 + (i >> 1)));
    const float sc1 = ink2_ue8m0(__ldg(s1 + (i >> 1)));
    __half2 w0v[8], w1v[8];
    mx8_decode16_h2(__ldg(p0 + i), w0v);
    mx8_decode16_h2(__ldg(p1 + i), w1v);
    const __half2* hh = s_h + i * 8;
    y0 = fmaf(dot16_h2_2acc(w0v, hh), sc0, y0);
    y1 = fmaf(dot16_h2_2acc(w1v, hh), sc1, y1);
  }
  y0 += __shfl_xor_sync(0xffffffffu, y0, 4); y1 += __shfl_xor_sync(0xffffffffu, y1, 4);
  y0 += __shfl_xor_sync(0xffffffffu, y0, 2); y1 += __shfl_xor_sync(0xffffffffu, y1, 2);
  y0 += __shfl_xor_sync(0xffffffffu, y0, 1); y1 += __shfl_xor_sync(0xffffffffu, y1, 1);
  if (lg == 0) {
    const float rw = ew[slot];
    *reinterpret_cast<float2*>(pair_out + size_t(slot) * size_t(H) + size_t(2 * pair_r)) =
        make_float2(y0 * rw, y1 * rw);
  }
}

__global__ void k_moe_nvfp4_w13_situ(const float* x, const uint32_t* eid, const uint8_t* w13,
                                     const uint8_t* sc, const float* outer, float* act,
                                     int H, int I, int K_top) {
  const int hw = threadIdx.x >> 4, sub = threadIdx.x & 15, j = blockIdx.x * 16 + hw;
  if (j >= I) return;
  const int slot = blockIdx.y, t = blockIdx.z;
  const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
  const int two_i = 2 * I, half_h = H / 2, blocks = H / 16;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * j), wu_row = wg_row + 1;
  const float4* x4 = reinterpret_cast<const float4*>(x + size_t(t) * size_t(H));
  const uint2* wgp = reinterpret_cast<const uint2*>(w13 + wg_row * size_t(half_h));
  const uint2* wup = reinterpret_cast<const uint2*>(w13 + wu_row * size_t(half_h));
  float ag = 0.0f, au = 0.0f;
  for (int blk = sub; blk < blocks; blk += 16) {
    const float sg_b = ink2_e4m3(__ldg(sc + wg_row * size_t(blocks) + size_t(blk)));
    const float su_b = ink2_e4m3(__ldg(sc + wu_row * size_t(blocks) + size_t(blk)));
    const uint2 g64 = __ldg(wgp + blk), u64 = __ldg(wup + blk);
    const float4 xv0 = x4[blk*4+0], xv1 = x4[blk*4+1], xv2 = x4[blk*4+2], xv3 = x4[blk*4+3];
    ag += (nvfp4_bit_dot8(g64.x, xv0, xv1) + nvfp4_bit_dot8(g64.y, xv2, xv3)) * sg_b;
    au += (nvfp4_bit_dot8(u64.x, xv0, xv1) + nvfp4_bit_dot8(u64.y, xv2, xv3)) * su_b;
  }
  ag += __shfl_xor_sync(0xffffffffu, ag, 8); au += __shfl_xor_sync(0xffffffffu, au, 8);
  ag += __shfl_xor_sync(0xffffffffu, ag, 4); au += __shfl_xor_sync(0xffffffffu, au, 4);
  ag += __shfl_xor_sync(0xffffffffu, ag, 2); au += __shfl_xor_sync(0xffffffffu, au, 2);
  ag += __shfl_xor_sync(0xffffffffu, ag, 1); au += __shfl_xor_sync(0xffffffffu, au, 1);
  if (sub == 0) {
    const float os = outer[e], g = ag * os, u = au * os;
    act[(size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I) + size_t(j)] =
        (4.0f * tanhf(g * 0.25f)) * (1.0f / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
  }
}

__global__ void k_moe_nvfp4_w2_par(const float* act, const uint32_t* eid, const float* ew,
                                   const uint8_t* w2, const uint8_t* sc, const float* outer,
                                   float* pair_out, int H, int I, int K_top) {
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, slot = blockIdx.y, t = blockIdx.z;
  if (n >= H) return;
  const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
  const int half_i = I / 2, blocks = I / 16;
  const size_t wrow = size_t(e) * size_t(H) + size_t(n);
  const float4* a4 = reinterpret_cast<const float4*>(act + (size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I));
  const uint2* wp = reinterpret_cast<const uint2*>(w2 + wrow * size_t(half_i));
  float sum = 0.0f;
  for (int blk = lane; blk < blocks; blk += 32) {
    const float sb = ink2_e4m3(__ldg(sc + wrow * size_t(blocks) + size_t(blk)));
    const uint2 w64 = __ldg(wp + blk);
    sum += (nvfp4_bit_dot8(w64.x, a4[blk * 4 + 0], a4[blk * 4 + 1]) +
            nvfp4_bit_dot8(w64.y, a4[blk * 4 + 2], a4[blk * 4 + 3])) * sb;
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    pair_out[(size_t(t) * size_t(K_top) + size_t(slot)) * size_t(H) + size_t(n)] =
        ew[size_t(t) * size_t(K_top) + size_t(slot)] * outer[e] * sum;
  }
}

// MXFP8 __ldg decode (T=1 & fallback)
__global__ void k_moe_mxfp8_w13_situ(const float* x, const uint32_t* eid, const uint8_t* w13,
                                     const uint8_t* sc, float* act, int H, int I, int K_top) {
  const int hw = threadIdx.x >> 4, sub = threadIdx.x & 15, j = blockIdx.x * 16 + hw;
  if (j >= I) return;
  const int slot = blockIdx.y, t = blockIdx.z;
  const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
  const int two_i = 2 * I, blocks = H / 32;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * j), wu_row = wg_row + 1;
  const float4* x4 = reinterpret_cast<const float4*>(x + size_t(t) * size_t(H));
  const uint4* wgp = reinterpret_cast<const uint4*>(w13 + wg_row * size_t(H));
  const uint4* wup = reinterpret_cast<const uint4*>(w13 + wu_row * size_t(H));
  float ag = 0.0f, au = 0.0f;
  for (int blk = sub; blk < blocks; blk += 16) {
    const float sg_b = ink2_ue8m0(__ldg(sc + wg_row * size_t(blocks) + size_t(blk)));
    const float su_b = ink2_ue8m0(__ldg(sc + wu_row * size_t(blocks) + size_t(blk)));
    const uint4 g0 = __ldg(wgp + blk * 2 + 0), g1 = __ldg(wgp + blk * 2 + 1);
    const uint4 u0 = __ldg(wup + blk * 2 + 0), u1 = __ldg(wup + blk * 2 + 1);
    const float4 xv0 = x4[blk*8+0], xv1 = x4[blk*8+1], xv2 = x4[blk*8+2], xv3 = x4[blk*8+3];
    const float4 xv4 = x4[blk*8+4], xv5 = x4[blk*8+5], xv6 = x4[blk*8+6], xv7 = x4[blk*8+7];
    ag += (mxfp8_dot4(g0.x, xv0) + mxfp8_dot4(g0.y, xv1) + mxfp8_dot4(g0.z, xv2) + mxfp8_dot4(g0.w, xv3) +
           mxfp8_dot4(g1.x, xv4) + mxfp8_dot4(g1.y, xv5) + mxfp8_dot4(g1.z, xv6) + mxfp8_dot4(g1.w, xv7)) * sg_b;
    au += (mxfp8_dot4(u0.x, xv0) + mxfp8_dot4(u0.y, xv1) + mxfp8_dot4(u0.z, xv2) + mxfp8_dot4(u0.w, xv3) +
           mxfp8_dot4(u1.x, xv4) + mxfp8_dot4(u1.y, xv5) + mxfp8_dot4(u1.z, xv6) + mxfp8_dot4(u1.w, xv7)) * su_b;
  }
  ag += __shfl_xor_sync(0xffffffffu, ag, 8); au += __shfl_xor_sync(0xffffffffu, au, 8);
  ag += __shfl_xor_sync(0xffffffffu, ag, 4); au += __shfl_xor_sync(0xffffffffu, au, 4);
  ag += __shfl_xor_sync(0xffffffffu, ag, 2); au += __shfl_xor_sync(0xffffffffu, au, 2);
  ag += __shfl_xor_sync(0xffffffffu, ag, 1); au += __shfl_xor_sync(0xffffffffu, au, 1);
  if (sub == 0) {
    act[(size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I) + size_t(j)] =
        (4.0f * tanhf(ag * 0.25f)) * (1.0f / (1.0f + expf(-ag))) * (25.0f * tanhf(au * 0.04f));
  }
}

__global__ void k_moe_mxfp8_w2_fused8(const float* act, const uint32_t* eid, const float* ew,
                                      const uint8_t* w2, const uint8_t* sc, float* out, int H, int I) {
  __shared__ uint32_t sh_eid[8];
  __shared__ float sh_ew[8];
  __shared__ float4 sh_s[8];
  const int tid = threadIdx.x, slot = tid >> 5, lane = tid & 31, t = blockIdx.y, n0 = blockIdx.x * 4;
  if (lane == 0) {
    sh_eid[slot] = eid[size_t(t) * 8 + size_t(slot)];
    sh_ew[slot] = ew[size_t(t) * 8 + size_t(slot)];
  }
  __syncthreads();
  const uint32_t e = sh_eid[slot];
  const float wt = sh_ew[slot];
  const int blocks = I / 32;
  const float4* a4 = reinterpret_cast<const float4*>(act + (size_t(t) * 8 + size_t(slot)) * size_t(I));
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (int blk = lane; blk < blocks; blk += 32) {
    const float4 xv0 = a4[blk*8+0], xv1 = a4[blk*8+1], xv2 = a4[blk*8+2], xv3 = a4[blk*8+3];
    const float4 xv4 = a4[blk*8+4], xv5 = a4[blk*8+5], xv6 = a4[blk*8+6], xv7 = a4[blk*8+7];
    auto one_r = [&](int nr, float& acc) {
      const size_t wrow = size_t(e) * size_t(H) + size_t(n0 + nr);
      const float sb = ink2_ue8m0(__ldg(sc + wrow * size_t(blocks) + size_t(blk)));
      const uint4* wp = reinterpret_cast<const uint4*>(w2 + wrow * size_t(I));
      const uint4 w0 = __ldg(wp + blk * 2 + 0), w1 = __ldg(wp + blk * 2 + 1);
      acc += (mxfp8_dot4(w0.x, xv0) + mxfp8_dot4(w0.y, xv1) + mxfp8_dot4(w0.z, xv2) + mxfp8_dot4(w0.w, xv3) +
              mxfp8_dot4(w1.x, xv4) + mxfp8_dot4(w1.y, xv5) + mxfp8_dot4(w1.z, xv6) + mxfp8_dot4(w1.w, xv7)) * sb;
    };
    one_r(0, s0); one_r(1, s1); one_r(2, s2); one_r(3, s3);
  }
  s0 = warp_sum(s0) * wt; s1 = warp_sum(s1) * wt; s2 = warp_sum(s2) * wt; s3 = warp_sum(s3) * wt;
  if (lane == 0) sh_s[slot] = make_float4(s0, s1, s2, s3);
  __syncthreads();
  if (slot == 0 && lane == 0) {
    float4 tot = make_float4(0, 0, 0, 0);
    for (int k = 0; k < 8; ++k) {
      tot.x += sh_s[k].x; tot.y += sh_s[k].y; tot.z += sh_s[k].z; tot.w += sh_s[k].w;
    }
    *reinterpret_cast<float4*>(out + size_t(t) * size_t(H) + size_t(n0)) = tot;
  }
}

__global__ void k_moe_mxfp8_w2_par(const float* act, const uint32_t* eid, const float* ew,
                                   const uint8_t* w2, const uint8_t* sc, float* pair_out, int H, int I, int K_top) {
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, slot = blockIdx.y, t = blockIdx.z;
  if (n >= H) return;
  const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
  const int blocks = I / 32;
  const size_t wrow = size_t(e) * size_t(H) + size_t(n);
  const float4* a4 = reinterpret_cast<const float4*>(act + (size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I));
  const uint4* wp = reinterpret_cast<const uint4*>(w2 + wrow * size_t(I));
  float sum = 0.0f;
  for (int blk = lane; blk < blocks; blk += 32) {
    const float sb = ink2_ue8m0(__ldg(sc + wrow * size_t(blocks) + size_t(blk)));
    const uint4 w0 = __ldg(wp + blk * 2 + 0), w1 = __ldg(wp + blk * 2 + 1);
    sum += (mxfp8_dot4(w0.x, a4[blk*8+0]) + mxfp8_dot4(w0.y, a4[blk*8+1]) +
            mxfp8_dot4(w0.z, a4[blk*8+2]) + mxfp8_dot4(w0.w, a4[blk*8+3]) +
            mxfp8_dot4(w1.x, a4[blk*8+4]) + mxfp8_dot4(w1.y, a4[blk*8+5]) +
            mxfp8_dot4(w1.z, a4[blk*8+6]) + mxfp8_dot4(w1.w, a4[blk*8+7])) * sb;
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    pair_out[(size_t(t) * size_t(K_top) + size_t(slot)) * size_t(H) + size_t(n)] =
        ew[size_t(t) * size_t(K_top) + size_t(slot)] * sum;
  }
}

// SMEM-free Small-Batch Grouped MoE Verify (T = 2..12, deduplicating unique active experts)
__global__ void k_moe_bucket_small(const uint32_t* eid, const float* ew,
                                   int32_t* tile_exp, int32_t* tile_m0, int32_t* tile_cnt, int32_t* num_tiles,
                                   int32_t* st_tok, int32_t* st_pair, float* st_ew,
                                   int T, int K_top, int E) {
  __shared__ uint32_t sh_eid[128];
  __shared__ float sh_ew[128];
  __shared__ int32_t sh_off[4], sh_act[4];
  const int tid = threadIdx.x, r_tot = T * K_top;
  if (tid < r_tot) { sh_eid[tid] = eid[tid]; sh_ew[tid] = ew[tid]; }
  __syncthreads();
  int32_t count = 0;
  if (tid < E) {
    for (int i = 0; i < r_tot; ++i) count += (sh_eid[i] == uint32_t(tid)) ? 1 : 0;
  }
  const int32_t is_act = (count > 0) ? 1 : 0;
  const int sg = tid >> 5, lane = tid & 31;
  int32_t p_incl = count, e_incl = is_act;
  #pragma unroll
  for (int off = 1; off <= 16; off <<= 1) {
    const int32_t up = __shfl_up_sync(0xffffffffu, p_incl, off);
    const int32_t ue = __shfl_up_sync(0xffffffffu, e_incl, off);
    if (lane >= off) { p_incl += up; e_incl += ue; }
  }
  if (lane == 31) { sh_off[sg] = p_incl; sh_act[sg] = e_incl; }
  __syncthreads();
  int32_t p_base = 0, e_base = 0;
  for (int s = 0; s < sg; ++s) { p_base += sh_off[s]; e_base += sh_act[s]; }
  const int32_t p_pos0 = p_base + (p_incl - count), e_pos = e_base + (e_incl - is_act);
  if (is_act) {
    tile_exp[e_pos] = tid;
    tile_m0[e_pos] = p_pos0;
    tile_cnt[e_pos] = count;
    int32_t wp = p_pos0;
    for (int i = 0; i < r_tot; ++i) {
      if (sh_eid[i] == uint32_t(tid)) {
        st_tok[wp] = i / K_top;
        st_pair[wp] = i;
        st_ew[wp] = sh_ew[i];
        ++wp;
      }
    }
  }
  if (tid == 0) *num_tiles = sh_act[0] + sh_act[1] + sh_act[2] + sh_act[3];
}

// SMEM __half2 + hardware-decode Small-Batch Grouped MoE (T = 2..12)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_moe_nvfp4_w13_g_h2(
    const float* __restrict__ x, const uint8_t* __restrict__ w13,
    const uint8_t* __restrict__ sc, const float* __restrict__ outer,
    const int32_t* __restrict__ tile_exp, const int32_t* __restrict__ tile_m0,
    const int32_t* __restrict__ tile_cnt, const int32_t* __restrict__ num_tiles,
    const int32_t* __restrict__ st_tok, const int32_t* __restrict__ st_pair,
    float* __restrict__ act, int H, int I) {
  const int eidx = blockIdx.y;
  if (eidx >= *num_tiles) return;
  const int count = tile_cnt[eidx], start = tile_m0[eidx], e = tile_exp[eidx];
  const int tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, feat = blockIdx.x * 16 + sub;
  const bool valid = (feat < I);
  const int two_i = 2 * I, half_h = H / 2, blocks = H / 16, blocks2 = blocks >> 1;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * feat), wu_row = wg_row + 1;
  const uint4* wgp4 = valid ? reinterpret_cast<const uint4*>(w13 + wg_row * size_t(half_h)) : nullptr;
  const uint4* wup4 = valid ? reinterpret_cast<const uint4*>(w13 + wu_row * size_t(half_h)) : nullptr;
  const uint16_t* scg2 = valid ? reinterpret_cast<const uint16_t*>(sc + wg_row * size_t(blocks)) : nullptr;
  const uint16_t* scu2 = valid ? reinterpret_cast<const uint16_t*>(sc + wu_row * size_t(blocks)) : nullptr;
  const float os = outer[e];
  __shared__ alignas(16) __half2 s_x[MAX_M][1152];

  for (int off_m = 0; off_m < count; off_m += MAX_M) {
    const int n_e = min(MAX_M, count - off_m);
    for (int m = 0; m < n_e; ++m) {
      const float2* src2 = reinterpret_cast<const float2*>(x + size_t(st_tok[start + off_m + m]) * size_t(H));
      __half2* dst2 = s_x[m];
      for (int i = tid; i < (H >> 1); i += 128) {
        const float2 f = src2[i];
        dst2[i] = __floats2half2_rn(f.x, f.y);
      }
    }
    __syncthreads();

    if (valid) {
      if (n_e == 1) {
        float ag0 = 0.0f, au0 = 0.0f;
        for (int i = lg; i < blocks2; i += 8) {
          const uint4 gv = __ldg(wgp4 + i), uv = __ldg(wup4 + i);
          const float2 sg = ink2_e4m3x2(__ldg(scg2 + i));
          const float2 su = ink2_e4m3x2(__ldg(scu2 + i));
          __half2 gw[8], uw[8];
          const __half2* xh = s_x[0] + i * 16;
          sf4_decode16_h2(gv.x, gv.y, gw);
          sf4_decode16_h2(uv.x, uv.y, uw);
          ag0 = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag0);
          au0 = fmaf(dot16_h2_2acc(uw, xh), su.x, au0);
          sf4_decode16_h2(gv.z, gv.w, gw);
          sf4_decode16_h2(uv.z, uv.w, uw);
          ag0 = fmaf(dot16_h2_2acc(gw, xh + 8), sg.y, ag0);
          au0 = fmaf(dot16_h2_2acc(uw, xh + 8), su.y, au0);
        }
        ag0 += __shfl_xor_sync(0xffffffffu, ag0, 4); au0 += __shfl_xor_sync(0xffffffffu, au0, 4);
        ag0 += __shfl_xor_sync(0xffffffffu, ag0, 2); au0 += __shfl_xor_sync(0xffffffffu, au0, 2);
        ag0 += __shfl_xor_sync(0xffffffffu, ag0, 1); au0 += __shfl_xor_sync(0xffffffffu, au0, 1);
        if (lg == 0) {
          act[size_t(st_pair[start + off_m]) * size_t(I) + size_t(feat)] = situ_f32(ag0 * os, au0 * os);
        }
      } else {
        float ag[MAX_M] = {}, au[MAX_M] = {};
        for (int i = lg; i < blocks2; i += 8) {
          const uint4 gv = __ldg(wgp4 + i), uv = __ldg(wup4 + i);
          const float2 sg = ink2_e4m3x2(__ldg(scg2 + i));
          const float2 su = ink2_e4m3x2(__ldg(scu2 + i));
          __half2 gw[8], uw[8];
          sf4_decode16_h2(gv.x, gv.y, gw);
          sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
          for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
              const __half2* xh = s_x[m] + i * 16;
              ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag[m]);
              au[m] = fmaf(dot16_h2_2acc(uw, xh), su.x, au[m]);
            }
          }
          sf4_decode16_h2(gv.z, gv.w, gw);
          sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
          for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
              const __half2* xh = s_x[m] + i * 16 + 8;
              ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.y, ag[m]);
              au[m] = fmaf(dot16_h2_2acc(uw, xh), su.y, au[m]);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
          if (m < n_e) {
            float gv = ag[m], uv = au[m];
            gv += __shfl_xor_sync(0xffffffffu, gv, 4); uv += __shfl_xor_sync(0xffffffffu, uv, 4);
            gv += __shfl_xor_sync(0xffffffffu, gv, 2); uv += __shfl_xor_sync(0xffffffffu, uv, 2);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1); uv += __shfl_xor_sync(0xffffffffu, uv, 1);
            if (lg == 0) {
              act[size_t(st_pair[start + off_m + m]) * size_t(I) + size_t(feat)] = situ_f32(gv * os, uv * os);
            }
          }
        }
      }
    }
    if (off_m + MAX_M < count) __syncthreads();
  }
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_moe_nvfp4_w2_g_h2(
    const float* __restrict__ act, const uint8_t* __restrict__ w2,
    const uint8_t* __restrict__ sc, const float* __restrict__ outer,
    const int32_t* __restrict__ tile_exp, const int32_t* __restrict__ tile_m0,
    const int32_t* __restrict__ tile_cnt, const int32_t* __restrict__ num_tiles,
    const int32_t* __restrict__ st_pair, const float* __restrict__ st_ew,
    float* __restrict__ pair_out, int H, int I) {
  const int eidx = blockIdx.y;
  if (eidx >= *num_tiles) return;
  const int count = tile_cnt[eidx], start = tile_m0[eidx], e = tile_exp[eidx];
  const int tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, pair_r = blockIdx.x * 16 + sub;
  const bool valid = (2 * pair_r < H);
  const int half_i = I / 2, blocks = I / 16, blocks2 = blocks >> 1;
  const size_t wrow0 = size_t(e) * size_t(H) + size_t(2 * pair_r), wrow1 = wrow0 + 1;
  const uint4* p0 = valid ? reinterpret_cast<const uint4*>(w2 + wrow0 * size_t(half_i)) : nullptr;
  const uint4* p1 = valid ? reinterpret_cast<const uint4*>(w2 + wrow1 * size_t(half_i)) : nullptr;
  const uint16_t* s0 = valid ? reinterpret_cast<const uint16_t*>(sc + wrow0 * size_t(blocks)) : nullptr;
  const uint16_t* s1 = valid ? reinterpret_cast<const uint16_t*>(sc + wrow1 * size_t(blocks)) : nullptr;
  const float os = outer[e];
  __shared__ alignas(16) __half2 s_h[MAX_M][576];

  for (int off_m = 0; off_m < count; off_m += MAX_M) {
    const int n_e = min(MAX_M, count - off_m);
    for (int m = 0; m < n_e; ++m) {
      const float2* src2 = reinterpret_cast<const float2*>(act + size_t(st_pair[start + off_m + m]) * size_t(I));
      __half2* dst2 = s_h[m];
      for (int i = tid; i < (I >> 1); i += 128) {
        const float2 f = src2[i];
        dst2[i] = __floats2half2_rn(f.x, f.y);
      }
    }
    __syncthreads();

    if (valid) {
      if (n_e == 1) {
        float y0 = 0.0f, y1 = 0.0f;
        for (int i = lg; i < blocks2; i += 8) {
          const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
          const float2 sc0 = ink2_e4m3x2(__ldg(s0 + i));
          const float2 sc1 = ink2_e4m3x2(__ldg(s1 + i));
          __half2 w0v[8], w1v[8];
          const __half2* hh = s_h[0] + i * 16;
          sf4_decode16_h2(v0.x, v0.y, w0v);
          sf4_decode16_h2(v1.x, v1.y, w1v);
          y0 = fmaf(dot16_h2_2acc(w0v, hh), sc0.x, y0);
          y1 = fmaf(dot16_h2_2acc(w1v, hh), sc1.x, y1);
          sf4_decode16_h2(v0.z, v0.w, w0v);
          sf4_decode16_h2(v1.z, v1.w, w1v);
          y0 = fmaf(dot16_h2_2acc(w0v, hh + 8), sc0.y, y0);
          y1 = fmaf(dot16_h2_2acc(w1v, hh + 8), sc1.y, y1);
        }
        y0 += __shfl_xor_sync(0xffffffffu, y0, 4); y1 += __shfl_xor_sync(0xffffffffu, y1, 4);
        y0 += __shfl_xor_sync(0xffffffffu, y0, 2); y1 += __shfl_xor_sync(0xffffffffu, y1, 2);
        y0 += __shfl_xor_sync(0xffffffffu, y0, 1); y1 += __shfl_xor_sync(0xffffffffu, y1, 1);
        if (lg == 0) {
          const int pid = st_pair[start + off_m];
          const float rwo = st_ew[start + off_m] * os;
          *reinterpret_cast<float2*>(pair_out + size_t(pid) * size_t(H) + size_t(2 * pair_r)) =
              make_float2(y0 * rwo, y1 * rwo);
        }
      } else {
        float a0[MAX_M] = {}, a1[MAX_M] = {};
        for (int i = lg; i < blocks2; i += 8) {
          const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
          const float2 sc0 = ink2_e4m3x2(__ldg(s0 + i));
          const float2 sc1 = ink2_e4m3x2(__ldg(s1 + i));
          __half2 w0v[8], w1v[8];
          sf4_decode16_h2(v0.x, v0.y, w0v);
          sf4_decode16_h2(v1.x, v1.y, w1v);
#pragma unroll
          for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
              const __half2* hh = s_h[m] + i * 16;
              a0[m] = fmaf(dot16_h2_2acc(w0v, hh), sc0.x, a0[m]);
              a1[m] = fmaf(dot16_h2_2acc(w1v, hh), sc1.x, a1[m]);
            }
          }
          sf4_decode16_h2(v0.z, v0.w, w0v);
          sf4_decode16_h2(v1.z, v1.w, w1v);
#pragma unroll
          for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
              const __half2* hh = s_h[m] + i * 16 + 8;
              a0[m] = fmaf(dot16_h2_2acc(w0v, hh), sc0.y, a0[m]);
              a1[m] = fmaf(dot16_h2_2acc(w1v, hh), sc1.y, a1[m]);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
          if (m < n_e) {
            float y0 = a0[m], y1 = a1[m];
            y0 += __shfl_xor_sync(0xffffffffu, y0, 4); y1 += __shfl_xor_sync(0xffffffffu, y1, 4);
            y0 += __shfl_xor_sync(0xffffffffu, y0, 2); y1 += __shfl_xor_sync(0xffffffffu, y1, 2);
            y0 += __shfl_xor_sync(0xffffffffu, y0, 1); y1 += __shfl_xor_sync(0xffffffffu, y1, 1);
            if (lg == 0) {
              const int pid = st_pair[start + off_m + m];
              const float rwo = st_ew[start + off_m + m] * os;
              *reinterpret_cast<float2*>(pair_out + size_t(pid) * size_t(H) + size_t(2 * pair_r)) =
                  make_float2(y0 * rwo, y1 * rwo);
            }
          }
        }
      }
    }
    if (off_m + MAX_M < count) __syncthreads();
  }
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_moe_mxfp8_w13_g_h2(
    const float* __restrict__ x, const uint8_t* __restrict__ w13, const uint8_t* __restrict__ sc,
    const int32_t* __restrict__ tile_exp, const int32_t* __restrict__ tile_m0,
    const int32_t* __restrict__ tile_cnt, const int32_t* __restrict__ num_tiles,
    const int32_t* __restrict__ st_tok, const int32_t* __restrict__ st_pair,
    float* __restrict__ act, int H, int I) {
  const int eidx = blockIdx.y;
  if (eidx >= *num_tiles) return;
  const int count = tile_cnt[eidx], start = tile_m0[eidx], e = tile_exp[eidx];
  const int tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, feat = blockIdx.x * 16 + sub;
  const bool valid = (feat < I);
  const int two_i = 2 * I, blocks = H / 32, n16 = H / 16;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * feat), wu_row = wg_row + 1;
  const uint4* wgp4 = valid ? reinterpret_cast<const uint4*>(w13 + wg_row * size_t(H)) : nullptr;
  const uint4* wup4 = valid ? reinterpret_cast<const uint4*>(w13 + wu_row * size_t(H)) : nullptr;
  const uint8_t* scgp = valid ? (sc + wg_row * size_t(blocks)) : nullptr;
  const uint8_t* scup = valid ? (sc + wu_row * size_t(blocks)) : nullptr;
  __shared__ alignas(16) __half2 s_x[MAX_M][1152];

  for (int off_m = 0; off_m < count; off_m += MAX_M) {
    const int n_e = min(MAX_M, count - off_m);
    for (int m = 0; m < n_e; ++m) {
      const float2* src2 = reinterpret_cast<const float2*>(x + size_t(st_tok[start + off_m + m]) * size_t(H));
      __half2* dst2 = s_x[m];
      for (int i = tid; i < (H >> 1); i += 128) {
        const float2 f = src2[i];
        dst2[i] = __floats2half2_rn(f.x, f.y);
      }
    }
    __syncthreads();

    if (valid) {
      if (n_e == 1) {
        float ag0 = 0.0f, au0 = 0.0f;
        for (int i = lg; i < n16; i += 8) {
          const float sg = ink2_ue8m0(__ldg(scgp + (i >> 1)));
          const float su = ink2_ue8m0(__ldg(scup + (i >> 1)));
          __half2 gw[8], uw[8];
          mx8_decode16_h2(__ldg(wgp4 + i), gw);
          mx8_decode16_h2(__ldg(wup4 + i), uw);
          const __half2* xh = s_x[0] + i * 8;
          ag0 = fmaf(dot16_h2_2acc(gw, xh), sg, ag0);
          au0 = fmaf(dot16_h2_2acc(uw, xh), su, au0);
        }
        ag0 += __shfl_xor_sync(0xffffffffu, ag0, 4); au0 += __shfl_xor_sync(0xffffffffu, au0, 4);
        ag0 += __shfl_xor_sync(0xffffffffu, ag0, 2); au0 += __shfl_xor_sync(0xffffffffu, au0, 2);
        ag0 += __shfl_xor_sync(0xffffffffu, ag0, 1); au0 += __shfl_xor_sync(0xffffffffu, au0, 1);
        if (lg == 0) act[size_t(st_pair[start + off_m]) * size_t(I) + size_t(feat)] = situ_f32(ag0, au0);
      } else {
        float ag[MAX_M] = {}, au[MAX_M] = {};
        for (int i = lg; i < n16; i += 8) {
          const float sg = ink2_ue8m0(__ldg(scgp + (i >> 1)));
          const float su = ink2_ue8m0(__ldg(scup + (i >> 1)));
          __half2 gw[8], uw[8];
          mx8_decode16_h2(__ldg(wgp4 + i), gw);
          mx8_decode16_h2(__ldg(wup4 + i), uw);
#pragma unroll
          for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
              const __half2* xh = s_x[m] + i * 8;
              ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg, ag[m]);
              au[m] = fmaf(dot16_h2_2acc(uw, xh), su, au[m]);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
          if (m < n_e) {
            float gv = ag[m], uv = au[m];
            gv += __shfl_xor_sync(0xffffffffu, gv, 4); uv += __shfl_xor_sync(0xffffffffu, uv, 4);
            gv += __shfl_xor_sync(0xffffffffu, gv, 2); uv += __shfl_xor_sync(0xffffffffu, uv, 2);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1); uv += __shfl_xor_sync(0xffffffffu, uv, 1);
            if (lg == 0) act[size_t(st_pair[start + off_m + m]) * size_t(I) + size_t(feat)] = situ_f32(gv, uv);
          }
        }
      }
    }
    if (off_m + MAX_M < count) __syncthreads();
  }
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_moe_mxfp8_w2_g_h2(
    const float* __restrict__ act, const uint8_t* __restrict__ w2, const uint8_t* __restrict__ sc,
    const int32_t* __restrict__ tile_exp, const int32_t* __restrict__ tile_m0,
    const int32_t* __restrict__ tile_cnt, const int32_t* __restrict__ num_tiles,
    const int32_t* __restrict__ st_pair, const float* __restrict__ st_ew,
    float* __restrict__ pair_out, int H, int I) {
  const int eidx = blockIdx.y;
  if (eidx >= *num_tiles) return;
  const int count = tile_cnt[eidx], start = tile_m0[eidx], e = tile_exp[eidx];
  const int tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, pair_r = blockIdx.x * 16 + sub;
  const bool valid = (2 * pair_r < H);
  const int blocks = I / 32, n16 = I / 16;
  const size_t wrow0 = size_t(e) * size_t(H) + size_t(2 * pair_r), wrow1 = wrow0 + 1;
  const uint4* p0 = valid ? reinterpret_cast<const uint4*>(w2 + wrow0 * size_t(I)) : nullptr;
  const uint4* p1 = valid ? reinterpret_cast<const uint4*>(w2 + wrow1 * size_t(I)) : nullptr;
  const uint8_t* s0 = valid ? (sc + wrow0 * size_t(blocks)) : nullptr;
  const uint8_t* s1 = valid ? (sc + wrow1 * size_t(blocks)) : nullptr;
  __shared__ alignas(16) __half2 s_h[MAX_M][576];

  for (int off_m = 0; off_m < count; off_m += MAX_M) {
    const int n_e = min(MAX_M, count - off_m);
    for (int m = 0; m < n_e; ++m) {
      const float2* src2 = reinterpret_cast<const float2*>(act + size_t(st_pair[start + off_m + m]) * size_t(I));
      __half2* dst2 = s_h[m];
      for (int i = tid; i < (I >> 1); i += 128) {
        const float2 f = src2[i];
        dst2[i] = __floats2half2_rn(f.x, f.y);
      }
    }
    __syncthreads();

    if (valid) {
      if (n_e == 1) {
        float y0 = 0.0f, y1 = 0.0f;
        for (int i = lg; i < n16; i += 8) {
          const float sc0 = ink2_ue8m0(__ldg(s0 + (i >> 1)));
          const float sc1 = ink2_ue8m0(__ldg(s1 + (i >> 1)));
          __half2 w0v[8], w1v[8];
          mx8_decode16_h2(__ldg(p0 + i), w0v);
          mx8_decode16_h2(__ldg(p1 + i), w1v);
          const __half2* hh = s_h[0] + i * 8;
          y0 = fmaf(dot16_h2_2acc(w0v, hh), sc0, y0);
          y1 = fmaf(dot16_h2_2acc(w1v, hh), sc1, y1);
        }
        y0 += __shfl_xor_sync(0xffffffffu, y0, 4); y1 += __shfl_xor_sync(0xffffffffu, y1, 4);
        y0 += __shfl_xor_sync(0xffffffffu, y0, 2); y1 += __shfl_xor_sync(0xffffffffu, y1, 2);
        y0 += __shfl_xor_sync(0xffffffffu, y0, 1); y1 += __shfl_xor_sync(0xffffffffu, y1, 1);
        if (lg == 0) {
          const int pid = st_pair[start + off_m];
          const float rw = st_ew[start + off_m];
          *reinterpret_cast<float2*>(pair_out + size_t(pid) * size_t(H) + size_t(2 * pair_r)) =
              make_float2(y0 * rw, y1 * rw);
        }
      } else {
        float a0[MAX_M] = {}, a1[MAX_M] = {};
        for (int i = lg; i < n16; i += 8) {
          const float sc0 = ink2_ue8m0(__ldg(s0 + (i >> 1)));
          const float sc1 = ink2_ue8m0(__ldg(s1 + (i >> 1)));
          __half2 w0v[8], w1v[8];
          mx8_decode16_h2(__ldg(p0 + i), w0v);
          mx8_decode16_h2(__ldg(p1 + i), w1v);
#pragma unroll
          for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
              const __half2* hh = s_h[m] + i * 8;
              a0[m] = fmaf(dot16_h2_2acc(w0v, hh), sc0, a0[m]);
              a1[m] = fmaf(dot16_h2_2acc(w1v, hh), sc1, a1[m]);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
          if (m < n_e) {
            float y0 = a0[m], y1 = a1[m];
            y0 += __shfl_xor_sync(0xffffffffu, y0, 4); y1 += __shfl_xor_sync(0xffffffffu, y1, 4);
            y0 += __shfl_xor_sync(0xffffffffu, y0, 2); y1 += __shfl_xor_sync(0xffffffffu, y1, 2);
            y0 += __shfl_xor_sync(0xffffffffu, y0, 1); y1 += __shfl_xor_sync(0xffffffffu, y1, 1);
            if (lg == 0) {
              const int pid = st_pair[start + off_m + m];
              const float rw = st_ew[start + off_m + m];
              *reinterpret_cast<float2*>(pair_out + size_t(pid) * size_t(H) + size_t(2 * pair_r)) =
                  make_float2(y0 * rw, y1 * rw);
            }
          }
        }
      }
    }
    if (off_m + MAX_M < count) __syncthreads();
  }
}

__global__ void k_moe_mxfp8_w13_g(const float* __restrict__ x, const uint8_t* __restrict__ w13,
                                  const uint8_t* __restrict__ sc,
                                  const int32_t* __restrict__ tile_exp, const int32_t* __restrict__ tile_m0,
                                  const int32_t* __restrict__ tile_cnt, const int32_t* __restrict__ num_tiles,
                                  const int32_t* __restrict__ st_tok, const int32_t* __restrict__ st_pair,
                                  float* __restrict__ act, int H, int I) {
  const int eidx = blockIdx.y;
  if (eidx >= *num_tiles) return;
  const int tid = threadIdx.x, j = blockIdx.x * 8 + (tid >> 3), sub = tid & 7;
  if (j >= I) return;
  const int count = tile_cnt[eidx], start = tile_m0[eidx], e = tile_exp[eidx];
  const int two_i = 2 * I, blocks = H / 32;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * j), wu_row = wg_row + 1;
  const uint4* wgp = reinterpret_cast<const uint4*>(w13 + wg_row * size_t(H));
  const uint4* wup = reinterpret_cast<const uint4*>(w13 + wu_row * size_t(H));
  const uint8_t* scgp = sc + wg_row * size_t(blocks);
  const uint8_t* scup = sc + wu_row * size_t(blocks);

  for (int p0 = 0; p0 < count; p0 += 4) {
    const int pc = min(4, count - p0);
    const int t0 = st_tok[start + p0];
    const int t1 = (pc > 1) ? st_tok[start + p0 + 1] : 0;
    const int t2 = (pc > 2) ? st_tok[start + p0 + 2] : 0;
    const int t3 = (pc > 3) ? st_tok[start + p0 + 3] : 0;
    float ag0 = 0, au0 = 0, ag1 = 0, au1 = 0, ag2 = 0, au2 = 0, ag3 = 0, au3 = 0;
    for (int blk = sub; blk < blocks; blk += 8) {
      const float sg_b = ink2_ue8m0(__ldg(scgp + blk)), su_b = ink2_ue8m0(__ldg(scup + blk));
      const uint4 g0 = __ldg(wgp + blk * 2 + 0), g1 = __ldg(wgp + blk * 2 + 1);
      const uint4 u0 = __ldg(wup + blk * 2 + 0), u1 = __ldg(wup + blk * 2 + 1);
      auto half_blk = [&](const uint4 gw, const uint4 uw, int off) {
        { const float4* a = reinterpret_cast<const float4*>(x + size_t(t0)*size_t(H)) + blk * 8 + off;
          const float4 x0=__ldg(a+0), x1=__ldg(a+1), x2=__ldg(a+2), x3=__ldg(a+3);
          ag0 += (mxfp8_dot4(gw.x, x0) + mxfp8_dot4(gw.y, x1) + mxfp8_dot4(gw.z, x2) + mxfp8_dot4(gw.w, x3)) * sg_b;
          au0 += (mxfp8_dot4(uw.x, x0) + mxfp8_dot4(uw.y, x1) + mxfp8_dot4(uw.z, x2) + mxfp8_dot4(uw.w, x3)) * su_b; }
        if (pc > 1) { const float4* a = reinterpret_cast<const float4*>(x + size_t(t1)*size_t(H)) + blk * 8 + off;
          const float4 x0=__ldg(a+0), x1=__ldg(a+1), x2=__ldg(a+2), x3=__ldg(a+3);
          ag1 += (mxfp8_dot4(gw.x, x0) + mxfp8_dot4(gw.y, x1) + mxfp8_dot4(gw.z, x2) + mxfp8_dot4(gw.w, x3)) * sg_b;
          au1 += (mxfp8_dot4(uw.x, x0) + mxfp8_dot4(uw.y, x1) + mxfp8_dot4(uw.z, x2) + mxfp8_dot4(uw.w, x3)) * su_b; }
        if (pc > 2) { const float4* a = reinterpret_cast<const float4*>(x + size_t(t2)*size_t(H)) + blk * 8 + off;
          const float4 x0=__ldg(a+0), x1=__ldg(a+1), x2=__ldg(a+2), x3=__ldg(a+3);
          ag2 += (mxfp8_dot4(gw.x, x0) + mxfp8_dot4(gw.y, x1) + mxfp8_dot4(gw.z, x2) + mxfp8_dot4(gw.w, x3)) * sg_b;
          au2 += (mxfp8_dot4(uw.x, x0) + mxfp8_dot4(uw.y, x1) + mxfp8_dot4(uw.z, x2) + mxfp8_dot4(uw.w, x3)) * su_b; }
        if (pc > 3) { const float4* a = reinterpret_cast<const float4*>(x + size_t(t3)*size_t(H)) + blk * 8 + off;
          const float4 x0=__ldg(a+0), x1=__ldg(a+1), x2=__ldg(a+2), x3=__ldg(a+3);
          ag3 += (mxfp8_dot4(gw.x, x0) + mxfp8_dot4(gw.y, x1) + mxfp8_dot4(gw.z, x2) + mxfp8_dot4(gw.w, x3)) * sg_b;
          au3 += (mxfp8_dot4(uw.x, x0) + mxfp8_dot4(uw.y, x1) + mxfp8_dot4(uw.z, x2) + mxfp8_dot4(uw.w, x3)) * su_b; }
      };
      half_blk(g0, u0, 0);
      half_blk(g1, u1, 4);
    }
    auto fl = [&](int m, float ag, float au) {
      ag += __shfl_xor_sync(0xffffffffu, ag, 4); au += __shfl_xor_sync(0xffffffffu, au, 4);
      ag += __shfl_xor_sync(0xffffffffu, ag, 2); au += __shfl_xor_sync(0xffffffffu, au, 2);
      ag += __shfl_xor_sync(0xffffffffu, ag, 1); au += __shfl_xor_sync(0xffffffffu, au, 1);
      if (sub == 0) {
        act[size_t(st_pair[start + p0 + m]) * size_t(I) + size_t(j)] =
            (4.0f * tanhf(ag * 0.25f)) * (1.0f / (1.0f + expf(-ag))) * (25.0f * tanhf(au * 0.04f));
      }
    };
    fl(0, ag0, au0); if (pc > 1) fl(1, ag1, au1); if (pc > 2) fl(2, ag2, au2); if (pc > 3) fl(3, ag3, au3);
  }
}

__global__ void k_moe_mxfp8_w2_g(const float* __restrict__ act, const uint8_t* __restrict__ w2,
                                 const uint8_t* __restrict__ sc,
                                 const int32_t* __restrict__ tile_exp, const int32_t* __restrict__ tile_m0,
                                 const int32_t* __restrict__ tile_cnt, const int32_t* __restrict__ num_tiles,
                                 const int32_t* __restrict__ st_pair, const float* __restrict__ st_ew,
                                 float* __restrict__ pair_out, int H, int I) {
  const int eidx = blockIdx.y;
  if (eidx >= *num_tiles) return;
  const int tid = threadIdx.x, n = blockIdx.x * 16 + (tid >> 2), sub = tid & 3;
  if (n >= H) return;
  const int count = tile_cnt[eidx], start = tile_m0[eidx], e = tile_exp[eidx];
  const int blocks = I / 32;
  const size_t wrow = size_t(e) * size_t(H) + size_t(n);
  const uint4* wp = reinterpret_cast<const uint4*>(w2 + wrow * size_t(I));
  const uint8_t* scp = sc + wrow * size_t(blocks);

  for (int p0 = 0; p0 < count; p0 += 4) {
    const int pc = min(4, count - p0);
    const int p_0 = st_pair[start + p0];
    const int p_1 = (pc > 1) ? st_pair[start + p0 + 1] : 0;
    const int p_2 = (pc > 2) ? st_pair[start + p0 + 2] : 0;
    const int p_3 = (pc > 3) ? st_pair[start + p0 + 3] : 0;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int blk = sub; blk < blocks; blk += 4) {
      const float sb = ink2_ue8m0(__ldg(scp + blk));
      const uint4 w0 = __ldg(wp + blk * 2 + 0), w1 = __ldg(wp + blk * 2 + 1);
      auto one_p = [&](int pk, float& sm) {
        const float4* a = reinterpret_cast<const float4*>(act + size_t(pk)*size_t(I)) + blk * 8;
        sm += (mxfp8_dot4(w0.x, __ldg(a+0)) + mxfp8_dot4(w0.y, __ldg(a+1)) + mxfp8_dot4(w0.z, __ldg(a+2)) + mxfp8_dot4(w0.w, __ldg(a+3)) +
               mxfp8_dot4(w1.x, __ldg(a+4)) + mxfp8_dot4(w1.y, __ldg(a+5)) + mxfp8_dot4(w1.z, __ldg(a+6)) + mxfp8_dot4(w1.w, __ldg(a+7))) * sb;
      };
      one_p(p_0, s0); if (pc > 1) one_p(p_1, s1); if (pc > 2) one_p(p_2, s2); if (pc > 3) one_p(p_3, s3);
    }
    auto fl = [&](int m, int p_idx, float sm) {
      sm += __shfl_xor_sync(0xffffffffu, sm, 2); sm += __shfl_xor_sync(0xffffffffu, sm, 1);
      if (sub == 0) pair_out[size_t(p_idx) * size_t(H) + size_t(n)] = st_ew[start + p0 + m] * sm;
    };
    fl(0, p_0, s0); if (pc > 1) fl(1, p_1, s1); if (pc > 2) fl(2, p_2, s2); if (pc > 3) fl(3, p_3, s3);
  }
}

// Blackwell Tensor Core Grouped MoE Prefill (BM=16/32/64)
template <int BM>
__global__ void k_moe_group_build(const uint32_t* eid, int P, int E, int32_t* pair_idx,
                                  int32_t* tile_exp, int32_t* tile_m0, int32_t* tile_cnt, int32_t* num_tiles) {
  __shared__ int32_t sh_cnt[128], sh_cur[128], sh_off[128], sh_toff[128];
  const int tid = threadIdx.x;
  if (tid < E) sh_cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < P; i += 256) atomicAdd(&sh_cnt[eid[i]], 1);
  __syncthreads();
  if (tid == 0) {
    int acc = 0, tacc = 0;
    for (int e = 0; e < E; ++e) {
      const int c = sh_cnt[e];
      sh_off[e] = acc; sh_cur[e] = acc; acc += c;
      sh_toff[e] = tacc; tacc += (c + BM - 1) / BM;
    }
    *num_tiles = tacc;
  }
  __syncthreads();
  for (int i = tid; i < P; i += 256) {
    const int pos = atomicAdd(&sh_cur[eid[i]], 1);
    pair_idx[pos] = i;
  }
  if (tid < E) {
    const int c = sh_cnt[tid], nt = (c + BM - 1) / BM, t0 = sh_toff[tid], m0 = sh_off[tid];
    for (int k = 0; k < nt; ++k) {
      tile_exp[t0 + k] = tid;
      tile_m0[t0 + k] = m0 + k * BM;
      const int rem = c - k * BM;
      tile_cnt[t0 + k] = rem < BM ? rem : BM;
    }
  }
}

template <int BM, bool IS_MXFP8>
__global__ void k_moe_group_w13_tc(const float* x, const float* w13_outer,
                                   const uint8_t* w13, const uint8_t* sc,
                                   const int32_t* pair_idx, const int32_t* tile_exp,
                                   const int32_t* tile_m0, const int32_t* tile_cnt, const int32_t* num_tiles,
                                   float* act, int H, int I, int K_top) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles) return;
  __shared__ __nv_bfloat16 sh_A[BM][64];
  __shared__ __nv_bfloat16 sh_Bg[64][64];
  __shared__ __nv_bfloat16 sh_Bu[64][64];
  __shared__ int sh_pair[BM];
  __shared__ int sh_tok[BM];

  const int tid = threadIdx.x, wg = tid >> 5, lane = tid & 31;
  const int wm = wg >> 2, wn = wg & 3, j0 = blockIdx.y * 64;
  const int e = tile_exp[tile_id], m0 = tile_m0[tile_id], cnt = tile_cnt[tile_id];

  if (tid < BM) {
    if (tid < cnt) {
      const int pr = pair_idx[m0 + tid];
      sh_pair[tid] = pr;
      sh_tok[tid] = pr / K_top;
    } else {
      sh_pair[tid] = -1;
      sh_tok[tid] = -1;
    }
  }
  __syncthreads();

  constexpr int NB = 2;
  float acc_g[2][2][4], acc_u[2][2][4];
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) {
    acc_g[mt][nt][0] = acc_g[mt][nt][1] = acc_g[mt][nt][2] = acc_g[mt][nt][3] = 0.0f;
    acc_u[mt][nt][0] = acc_u[mt][nt][1] = acc_u[mt][nt][2] = acc_u[mt][nt][3] = 0.0f;
  }

  const int two_i = 2 * I;
  const size_t exp_w_base = size_t(e) * size_t(two_i);
  const int w_stride = IS_MXFP8 ? H : (H / 2);
  const int sc_stride = IS_MXFP8 ? (H / 32) : (H / 16);

  for (int k0 = 0; k0 < H; k0 += 64) {
    for (int idx8 = tid; idx8 < BM * 8; idx8 += 256) {
      const int r = idx8 >> 3, c8 = idx8 & 7, csw = c8 ^ (r & 7);
      uint32_t* da = reinterpret_cast<uint32_t*>(&sh_A[r][csw * 8]);
      const int tok = sh_tok[r];
      if (tok >= 0) {
        const float* xp = x + size_t(tok) * size_t(H) + k0 + c8 * 8;
        const float4 v0 = *reinterpret_cast<const float4*>(xp);
        const float4 v1 = *reinterpret_cast<const float4*>(xp + 4);
        da[0] = pack_bf2_f32(v0.x, v0.y); da[1] = pack_bf2_f32(v0.z, v0.w);
        da[2] = pack_bf2_f32(v1.x, v1.y); da[3] = pack_bf2_f32(v1.z, v1.w);
      } else {
        da[0] = 0u; da[1] = 0u; da[2] = 0u; da[3] = 0u;
      }
    }

    for (int idx16 = tid; idx16 < 128 * 4; idx16 += 256) {
      const int r_loc = idx16 >> 2, k_q = idx16 & 3;
      const int j_loc = r_loc >> 1, is_up = r_loc & 1;
      const size_t wrow = exp_w_base + size_t(2 * j0 + r_loc);
      __nv_bfloat16* b_row = is_up ? &sh_Bu[j_loc][0] : &sh_Bg[j_loc][0];
      const int c8_a = (k_q * 2 + 0) ^ (j_loc & 7);
      const int c8_b = (k_q * 2 + 1) ^ (j_loc & 7);
      uint32_t* d0 = reinterpret_cast<uint32_t*>(b_row + c8_a * 8);
      uint32_t* d1 = reinterpret_cast<uint32_t*>(b_row + c8_b * 8);
      if (!IS_MXFP8) {
        const int blk = (k0 >> 4) + k_q;
        const float sc_v = ink2_e4m3(__ldg(sc + wrow * size_t(sc_stride) + size_t(blk)));
        const uint2 w64 = __ldg(reinterpret_cast<const uint2*>(w13 + wrow * size_t(w_stride) + size_t(blk) * 8));
        d0[0] = pack_bf2_f32(e2m1_bit(w64.x) * sc_v, e2m1_bit(w64.x >> 4) * sc_v);
        d0[1] = pack_bf2_f32(e2m1_bit(w64.x >> 8) * sc_v, e2m1_bit(w64.x >> 12) * sc_v);
        d0[2] = pack_bf2_f32(e2m1_bit(w64.x >> 16) * sc_v, e2m1_bit(w64.x >> 20) * sc_v);
        d0[3] = pack_bf2_f32(e2m1_bit(w64.x >> 24) * sc_v, e2m1_bit(w64.x >> 28) * sc_v);
        d1[0] = pack_bf2_f32(e2m1_bit(w64.y) * sc_v, e2m1_bit(w64.y >> 4) * sc_v);
        d1[1] = pack_bf2_f32(e2m1_bit(w64.y >> 8) * sc_v, e2m1_bit(w64.y >> 12) * sc_v);
        d1[2] = pack_bf2_f32(e2m1_bit(w64.y >> 16) * sc_v, e2m1_bit(w64.y >> 20) * sc_v);
        d1[3] = pack_bf2_f32(e2m1_bit(w64.y >> 24) * sc_v, e2m1_bit(w64.y >> 28) * sc_v);
      } else {
        const int blk = (k0 >> 5) + (k_q >> 1);
        const float sc_v = ink2_ue8m0(__ldg(sc + wrow * size_t(sc_stride) + size_t(blk)));
        const uint4 w128 = __ldg(reinterpret_cast<const uint4*>(w13 + wrow * size_t(w_stride) + k0 + k_q * 16));
        d0[0] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.x)) * sc_v, ink2_e4m3(uint8_t(w128.x >> 8)) * sc_v);
        d0[1] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.x >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.x >> 24)) * sc_v);
        d0[2] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.y)) * sc_v, ink2_e4m3(uint8_t(w128.y >> 8)) * sc_v);
        d0[3] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.y >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.y >> 24)) * sc_v);
        d1[0] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.z)) * sc_v, ink2_e4m3(uint8_t(w128.z >> 8)) * sc_v);
        d1[1] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.z >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.z >> 24)) * sc_v);
        d1[2] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.w)) * sc_v, ink2_e4m3(uint8_t(w128.w >> 8)) * sc_v);
        d1[3] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.w >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.w >> 24)) * sc_v);
      }
    }
    __syncthreads();

    if (BM >= 64 || wm == 0) {
      const int m_warp_off = (BM >= 64) ? (wm * 32) : 0;
      const int n_mt = (BM >= 64) ? 2 : (BM / 16);
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t bg_frag[NB][2], bu_frag[NB][2];
        #pragma unroll
        for (int nt = 0; nt < NB; ++nt) {
          const int br = wn * 16 + nt * 8 + (lane & 7);
          const int bc = (ks * 2 + ((lane >> 3) & 1)) ^ (br & 7);
          ldmatrix_x2(bg_frag[nt], &sh_Bg[br][bc * 8]);
          ldmatrix_x2(bu_frag[nt], &sh_Bu[br][bc * 8]);
        }
        #pragma unroll
        for (int mt = 0; mt < n_mt; ++mt) {
          uint32_t a_frag[4];
          const int ar = m_warp_off + mt * 16 + (lane & 15);
          const int ac = (ks * 2 + (lane >> 4)) ^ (ar & 7);
          ldmatrix_x4(a_frag, &sh_A[ar][ac * 8]);
          #pragma unroll
          for (int nt = 0; nt < NB; ++nt) {
            mma_m16n8k16_bf16(acc_g[mt][nt], a_frag, bg_frag[nt], acc_g[mt][nt]);
            mma_m16n8k16_bf16(acc_u[mt][nt], a_frag, bu_frag[nt], acc_u[mt][nt]);
          }
        }
      }
    }
    __syncthreads();
  }

  if (BM < 64 && wm != 0) return;
  const int m_warp_off = (BM >= 64) ? (wm * 32) : 0;
  const int n_mt = (BM >= 64) ? 2 : (BM / 16);
  const float os = IS_MXFP8 ? 1.0f : w13_outer[e];
  const int gid = lane >> 2, t4 = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < n_mt; ++mt) {
    const int r0 = m_warp_off + mt * 16 + gid, r1 = r0 + 8;
    const int pr0 = (r0 < cnt) ? sh_pair[r0] : -1;
    const int pr1 = (r1 < cnt) ? sh_pair[r1] : -1;
    #pragma unroll
    for (int nt = 0; nt < NB; ++nt) {
      const int j_col = j0 + wn * 16 + nt * 8 + t4 * 2;
      if (pr0 >= 0) {
        const float g0 = acc_g[mt][nt][0] * os, u0 = acc_u[mt][nt][0] * os;
        const float g1 = acc_g[mt][nt][1] * os, u1 = acc_u[mt][nt][1] * os;
        act[size_t(pr0) * size_t(I) + size_t(j_col + 0)] =
            (4.0f * tanhf(g0 * 0.25f)) * (1.0f / (1.0f + expf(-g0))) * (25.0f * tanhf(u0 * 0.04f));
        act[size_t(pr0) * size_t(I) + size_t(j_col + 1)] =
            (4.0f * tanhf(g1 * 0.25f)) * (1.0f / (1.0f + expf(-g1))) * (25.0f * tanhf(u1 * 0.04f));
      }
      if (pr1 >= 0) {
        const float g0 = acc_g[mt][nt][2] * os, u0 = acc_u[mt][nt][2] * os;
        const float g1 = acc_g[mt][nt][3] * os, u1 = acc_u[mt][nt][3] * os;
        act[size_t(pr1) * size_t(I) + size_t(j_col + 0)] =
            (4.0f * tanhf(g0 * 0.25f)) * (1.0f / (1.0f + expf(-g0))) * (25.0f * tanhf(u0 * 0.04f));
        act[size_t(pr1) * size_t(I) + size_t(j_col + 1)] =
            (4.0f * tanhf(g1 * 0.25f)) * (1.0f / (1.0f + expf(-g1))) * (25.0f * tanhf(u1 * 0.04f));
      }
    }
  }
}

template <int BM, bool IS_MXFP8>
__global__ void k_moe_group_w2_tc(const float* act, const float* ew, const float* w2_outer,
                                  const uint8_t* w2, const uint8_t* sc,
                                  const int32_t* pair_idx, const int32_t* tile_exp,
                                  const int32_t* tile_m0, const int32_t* tile_cnt, const int32_t* num_tiles,
                                  float* pair_out, int H, int I) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles) return;
  __shared__ __nv_bfloat16 sh_A[BM][64];
  __shared__ __nv_bfloat16 sh_B[128][64];
  __shared__ int sh_pair[BM];

  const int tid = threadIdx.x, wg = tid >> 5, lane = tid & 31;
  const int wm = wg >> 2, wn = wg & 3, n0 = blockIdx.y * 128;
  const int e = tile_exp[tile_id], m0 = tile_m0[tile_id], cnt = tile_cnt[tile_id];

  if (tid < BM) sh_pair[tid] = (tid < cnt) ? pair_idx[m0 + tid] : -1;
  __syncthreads();

  float acc[2][4][4];
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 4; ++nt)
    acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;

  const size_t exp_w_base = size_t(e) * size_t(H);
  const int w_stride = IS_MXFP8 ? I : (I / 2);
  const int sc_stride = IS_MXFP8 ? (I / 32) : (I / 16);

  for (int k0 = 0; k0 < I; k0 += 64) {
    for (int idx8 = tid; idx8 < BM * 8; idx8 += 256) {
      const int r = idx8 >> 3, c8 = idx8 & 7, csw = c8 ^ (r & 7);
      uint32_t* da = reinterpret_cast<uint32_t*>(&sh_A[r][csw * 8]);
      const int pr = sh_pair[r];
      if (pr >= 0) {
        const float* ap = act + size_t(pr) * size_t(I) + k0 + c8 * 8;
        const float4 v0 = *reinterpret_cast<const float4*>(ap);
        const float4 v1 = *reinterpret_cast<const float4*>(ap + 4);
        da[0] = pack_bf2_f32(v0.x, v0.y); da[1] = pack_bf2_f32(v0.z, v0.w);
        da[2] = pack_bf2_f32(v1.x, v1.y); da[3] = pack_bf2_f32(v1.z, v1.w);
      } else {
        da[0] = 0u; da[1] = 0u; da[2] = 0u; da[3] = 0u;
      }
    }

    for (int idx16 = tid; idx16 < 128 * 4; idx16 += 256) {
      const int r_loc = idx16 >> 2, k_q = idx16 & 3;
      const size_t wrow = exp_w_base + size_t(n0 + r_loc);
      const int c8_a = (k_q * 2 + 0) ^ (r_loc & 7);
      const int c8_b = (k_q * 2 + 1) ^ (r_loc & 7);
      uint32_t* d0 = reinterpret_cast<uint32_t*>(&sh_B[r_loc][c8_a * 8]);
      uint32_t* d1 = reinterpret_cast<uint32_t*>(&sh_B[r_loc][c8_b * 8]);
      if (!IS_MXFP8) {
        const int blk = (k0 >> 4) + k_q;
        const float sc_v = ink2_e4m3(__ldg(sc + wrow * size_t(sc_stride) + size_t(blk)));
        const uint2 w64 = __ldg(reinterpret_cast<const uint2*>(w2 + wrow * size_t(w_stride) + size_t(blk) * 8));
        d0[0] = pack_bf2_f32(e2m1_bit(w64.x) * sc_v, e2m1_bit(w64.x >> 4) * sc_v);
        d0[1] = pack_bf2_f32(e2m1_bit(w64.x >> 8) * sc_v, e2m1_bit(w64.x >> 12) * sc_v);
        d0[2] = pack_bf2_f32(e2m1_bit(w64.x >> 16) * sc_v, e2m1_bit(w64.x >> 20) * sc_v);
        d0[3] = pack_bf2_f32(e2m1_bit(w64.x >> 24) * sc_v, e2m1_bit(w64.x >> 28) * sc_v);
        d1[0] = pack_bf2_f32(e2m1_bit(w64.y) * sc_v, e2m1_bit(w64.y >> 4) * sc_v);
        d1[1] = pack_bf2_f32(e2m1_bit(w64.y >> 8) * sc_v, e2m1_bit(w64.y >> 12) * sc_v);
        d1[2] = pack_bf2_f32(e2m1_bit(w64.y >> 16) * sc_v, e2m1_bit(w64.y >> 20) * sc_v);
        d1[3] = pack_bf2_f32(e2m1_bit(w64.y >> 24) * sc_v, e2m1_bit(w64.y >> 28) * sc_v);
      } else {
        const int blk = (k0 >> 5) + (k_q >> 1);
        const float sc_v = ink2_ue8m0(__ldg(sc + wrow * size_t(sc_stride) + size_t(blk)));
        const uint4 w128 = __ldg(reinterpret_cast<const uint4*>(w2 + wrow * size_t(w_stride) + k0 + k_q * 16));
        d0[0] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.x)) * sc_v, ink2_e4m3(uint8_t(w128.x >> 8)) * sc_v);
        d0[1] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.x >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.x >> 24)) * sc_v);
        d0[2] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.y)) * sc_v, ink2_e4m3(uint8_t(w128.y >> 8)) * sc_v);
        d0[3] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.y >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.y >> 24)) * sc_v);
        d1[0] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.z)) * sc_v, ink2_e4m3(uint8_t(w128.z >> 8)) * sc_v);
        d1[1] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.z >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.z >> 24)) * sc_v);
        d1[2] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.w)) * sc_v, ink2_e4m3(uint8_t(w128.w >> 8)) * sc_v);
        d1[3] = pack_bf2_f32(ink2_e4m3(uint8_t(w128.w >> 16)) * sc_v, ink2_e4m3(uint8_t(w128.w >> 24)) * sc_v);
      }
    }
    __syncthreads();

    if (BM >= 64 || wm == 0) {
      const int m_warp_off = (BM >= 64) ? (wm * 32) : 0;
      const int n_mt = (BM >= 64) ? 2 : (BM / 16);
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t b_frag[4][2];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int br = wn * 32 + nt * 8 + (lane & 7);
          const int bc = (ks * 2 + ((lane >> 3) & 1)) ^ (br & 7);
          ldmatrix_x2(b_frag[nt], &sh_B[br][bc * 8]);
        }
        #pragma unroll
        for (int mt = 0; mt < n_mt; ++mt) {
          uint32_t a_frag[4];
          const int ar = m_warp_off + mt * 16 + (lane & 15);
          const int ac = (ks * 2 + (lane >> 4)) ^ (ar & 7);
          ldmatrix_x4(a_frag, &sh_A[ar][ac * 8]);
          #pragma unroll
          for (int nt = 0; nt < 4; ++nt) mma_m16n8k16_bf16(acc[mt][nt], a_frag, b_frag[nt], acc[mt][nt]);
        }
      }
    }
    __syncthreads();
  }

  if (BM < 64 && wm != 0) return;
  const int m_warp_off = (BM >= 64) ? (wm * 32) : 0;
  const int n_mt = (BM >= 64) ? 2 : (BM / 16);
  const float os = IS_MXFP8 ? 1.0f : w2_outer[e];
  const int gid = lane >> 2, t4 = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < n_mt; ++mt) {
    const int r0 = m_warp_off + mt * 16 + gid, r1 = r0 + 8;
    const int pr0 = (r0 < cnt) ? sh_pair[r0] : -1;
    const int pr1 = (r1 < cnt) ? sh_pair[r1] : -1;
    const float wt0 = (pr0 >= 0) ? (ew[pr0] * os) : 0.0f;
    const float wt1 = (pr1 >= 0) ? (ew[pr1] * os) : 0.0f;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int n_col = n0 + wn * 32 + nt * 8 + t4 * 2;
      if (pr0 >= 0) {
        pair_out[size_t(pr0) * size_t(H) + size_t(n_col + 0)] = acc[mt][nt][0] * wt0;
        pair_out[size_t(pr0) * size_t(H) + size_t(n_col + 1)] = acc[mt][nt][1] * wt0;
      }
      if (pr1 >= 0) {
        pair_out[size_t(pr1) * size_t(H) + size_t(n_col + 0)] = acc[mt][nt][2] * wt1;
        pair_out[size_t(pr1) * size_t(H) + size_t(n_col + 1)] = acc[mt][nt][3] * wt1;
      }
    }
  }
}

__global__ void k_add(const float* src, float* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void k_add3(const float* a, const float* b, float* dst, int n) {
  const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  if (i4 < n4) {
    const float4 av = reinterpret_cast<const float4*>(a)[i4];
    const float4 bv = reinterpret_cast<const float4*>(b)[i4];
    reinterpret_cast<float4*>(dst)[i4] = make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
    return;
  }
  const int i = (n4 << 2) + (i4 - n4);
  if (i < n) dst[i] = a[i] + b[i];
}

__global__ void k_softcap(float* x, int n, float cap, float inv_cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] * inv_cap);
}

__global__ void k_suppress_mask(float* logits, int V, const uint8_t* mask_bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < V && mask_bits[i]) logits[i] = -INFINITY;
}

__global__ void k_suppress_mask_rows(float* logits, int V, const uint8_t* mask_bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < V && mask_bits[i]) logits[size_t(blockIdx.y) * size_t(V) + size_t(i)] = -INFINITY;
}

__global__ void k_argmax_p1(const float* logits, int V, float* blk_v, int32_t* blk_i) {
  __shared__ float sv[32];
  __shared__ int32_t si[32];
  const int tid = threadIdx.x, ntg = blockDim.x, sg = tid >> 5, lane = tid & 31;
  const int bid = blockIdx.x, nblk = gridDim.x, row = blockIdx.y;
  const float* src = logits + size_t(row) * size_t(V);
  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;
  for (int i = bid * ntg + tid; i < V; i += nblk * ntg) {
    const float v = src[i];
    if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_v, o);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, o);
    if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
  }
  if (lane == 0) { sv[sg] = best_v; si[sg] = best_i; }
  __syncthreads();
  if (sg == 0) {
    const int nsg = ntg >> 5;
    float v = lane < nsg ? sv[lane] : -INFINITY;
    int32_t id = lane < nsg ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, o);
      const int32_t oi = __shfl_xor_sync(0xffffffffu, id, o);
      if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
    }
    if (lane == 0) {
      const size_t idx = size_t(row) * size_t(nblk) + size_t(bid);
      blk_v[idx] = v;
      blk_i[idx] = id;
    }
  }
}

__global__ void k_argmax_p2(const float* blk_v, const int32_t* blk_i, int nblk, int32_t* d_toks) {
  __shared__ float sv[32];
  __shared__ int32_t si[32];
  const int tid = threadIdx.x, ntg = blockDim.x, sg = tid >> 5, lane = tid & 31;
  const int row = blockIdx.y;
  const size_t base = size_t(row) * size_t(nblk);
  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;
  for (int i = tid; i < nblk; i += ntg) {
    const float v = blk_v[base + i];
    const int32_t id = blk_i[base + i];
    if (v > best_v || (v == best_v && id < best_i)) { best_v = v; best_i = id; }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_v, o);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, o);
    if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
  }
  if (lane == 0) { sv[sg] = best_v; si[sg] = best_i; }
  __syncthreads();
  if (sg == 0) {
    const int nsg = ntg >> 5;
    float v = lane < nsg ? sv[lane] : -INFINITY;
    int32_t id = lane < nsg ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, o);
      const int32_t oi = __shfl_xor_sync(0xffffffffu, id, o);
      if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
    }
    if (lane == 0) d_toks[row] = id;
  }
}

// ---- 12. DSpark MTP Kernels ----
__global__ void k_mtp_fuse_norm(const float* t0, const float* t1, const float* t2, const float* t3, int row_off,
                                float w0, float w1, float w2, float w3,
                                const void* norm_w, bool norm_f32, float* out, int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x, tid = threadIdx.x, ntg = blockDim.x;
  const size_t src_base = size_t(row_off + t) * size_t(H);
  const size_t dst_base = size_t(t) * size_t(H);
  if (H == 2304 && ntg == 192) {
    const int c0 = tid * 4, c1 = (tid + 192) * 4, c2 = (tid + 384) * 4;
    auto ld4 = [](const float* p, size_t off) { return *reinterpret_cast<const float4*>(p + off); };
    const float4 a0 = ld4(t0, src_base + c0), b0 = ld4(t1, src_base + c0), g0 = ld4(t2, src_base + c0), d0 = ld4(t3, src_base + c0);
    const float4 a1 = ld4(t0, src_base + c1), b1 = ld4(t1, src_base + c1), g1 = ld4(t2, src_base + c1), d1 = ld4(t3, src_base + c1);
    const float4 a2 = ld4(t0, src_base + c2), b2 = ld4(t1, src_base + c2), g2 = ld4(t2, src_base + c2), d2 = ld4(t3, src_base + c2);
    const float4 s0 = make_float4(w0*a0.x + w1*b0.x + w2*g0.x + w3*d0.x, w0*a0.y + w1*b0.y + w2*g0.y + w3*d0.y,
                                  w0*a0.z + w1*b0.z + w2*g0.z + w3*d0.z, w0*a0.w + w1*b0.w + w2*g0.w + w3*d0.w);
    const float4 s1 = make_float4(w0*a1.x + w1*b1.x + w2*g1.x + w3*d1.x, w0*a1.y + w1*b1.y + w2*g1.y + w3*d1.y,
                                  w0*a1.z + w1*b1.z + w2*g1.z + w3*d1.z, w0*a1.w + w1*b1.w + w2*g1.w + w3*d1.w);
    const float4 s2 = make_float4(w0*a2.x + w1*b2.x + w2*g2.x + w3*d2.x, w0*a2.y + w1*b2.y + w2*g2.y + w3*d2.y,
                                  w0*a2.z + w1*b2.z + w2*g2.z + w3*d2.z, w0*a2.w + w1*b2.w + w2*g2.w + w3*d2.w);
    const float ss = block_sum_6w(
        (s0.x*s0.x + s0.y*s0.y + s0.z*s0.z + s0.w*s0.w) +
        (s1.x*s1.x + s1.y*s1.y + s1.z*s1.z + s1.w*s1.w) +
        (s2.x*s2.x + s2.y*s2.y + s2.z*s2.z + s2.w*s2.w), sh);
    const float inv = rsqrtf(ss / 2304.0f + eps);
    float4 nw0, nw1, nw2;
    if (norm_f32) {
      const float4* wf = reinterpret_cast<const float4*>(norm_w);
      nw0 = wf[tid]; nw1 = wf[tid + 192]; nw2 = wf[tid + 384];
    } else {
      const uint2* wb = reinterpret_cast<const uint2*>(norm_w);
      const uint2 u0 = __ldg(wb + tid), u1 = __ldg(wb + tid + 192), u2 = __ldg(wb + tid + 384);
      nw0 = make_float4(ink2_bf16_lo(u0.x), ink2_bf16_hi(u0.x), ink2_bf16_lo(u0.y), ink2_bf16_hi(u0.y));
      nw1 = make_float4(ink2_bf16_lo(u1.x), ink2_bf16_hi(u1.x), ink2_bf16_lo(u1.y), ink2_bf16_hi(u1.y));
      nw2 = make_float4(ink2_bf16_lo(u2.x), ink2_bf16_hi(u2.x), ink2_bf16_lo(u2.y), ink2_bf16_hi(u2.y));
    }
    float4* dst4 = reinterpret_cast<float4*>(out + dst_base);
    dst4[tid] = make_float4(s0.x * inv * nw0.x, s0.y * inv * nw0.y, s0.z * inv * nw0.z, s0.w * inv * nw0.w);
    dst4[tid + 192] = make_float4(s1.x * inv * nw1.x, s1.y * inv * nw1.y, s1.z * inv * nw1.z, s1.w * inv * nw1.w);
    dst4[tid + 384] = make_float4(s2.x * inv * nw2.x, s2.y * inv * nw2.y, s2.z * inv * nw2.z, s2.w * inv * nw2.w);
    return;
  }
  float xs[24], ss = 0.0f;
  int k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    const float v = w0 * t0[src_base + c] + w1 * t1[src_base + c] + w2 * t2[src_base + c] + w3 * t3[src_base + c];
    xs[k] = v; ss += v * v;
  }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / float(H) + eps);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    const float nw = norm_f32 ? static_cast<const float*>(norm_w)[c]
                              : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[c]);
    out[dst_base + c] = xs[k] * inv * nw;
  }
}

__global__ void k_mtp_prep_block(const int32_t* d_anchor_tok, const __nv_bfloat16* embed,
                                 const __nv_bfloat16* embed_norm, const __nv_bfloat16* mask_embed,
                                 float* out, int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x, tid = threadIdx.x, ntg = blockDim.x;
  float* dst = out + size_t(t) * size_t(H);
  if (t == 0) {
    const int id = d_anchor_tok[0];
    const __nv_bfloat16* row = embed + size_t(id) * size_t(H);
    if (H == 2304 && ntg == 192) {
      const uint2* r2 = reinterpret_cast<const uint2*>(row);
      const uint2* w2 = reinterpret_cast<const uint2*>(embed_norm);
      const uint2 u0 = __ldg(r2 + tid), u1 = __ldg(r2 + tid + 192), u2 = __ldg(r2 + tid + 384);
      const float4 v0 = make_float4(ink2_bf16_lo(u0.x), ink2_bf16_hi(u0.x), ink2_bf16_lo(u0.y), ink2_bf16_hi(u0.y));
      const float4 v1 = make_float4(ink2_bf16_lo(u1.x), ink2_bf16_hi(u1.x), ink2_bf16_lo(u1.y), ink2_bf16_hi(u1.y));
      const float4 v2 = make_float4(ink2_bf16_lo(u2.x), ink2_bf16_hi(u2.x), ink2_bf16_lo(u2.y), ink2_bf16_hi(u2.y));
      const float ss = block_sum_6w(
          (v0.x*v0.x + v0.y*v0.y + v0.z*v0.z + v0.w*v0.w) +
          (v1.x*v1.x + v1.y*v1.y + v1.z*v1.z + v1.w*v1.w) +
          (v2.x*v2.x + v2.y*v2.y + v2.z*v2.z + v2.w*v2.w), sh);
      const float inv = rsqrtf(ss / 2304.0f + eps);
      const uint2 n0 = __ldg(w2 + tid), n1 = __ldg(w2 + tid + 192), n2 = __ldg(w2 + tid + 384);
      float4* d4 = reinterpret_cast<float4*>(dst);
      d4[tid] = make_float4(v0.x*inv*ink2_bf16_lo(n0.x), v0.y*inv*ink2_bf16_hi(n0.x), v0.z*inv*ink2_bf16_lo(n0.y), v0.w*inv*ink2_bf16_hi(n0.y));
      d4[tid + 192] = make_float4(v1.x*inv*ink2_bf16_lo(n1.x), v1.y*inv*ink2_bf16_hi(n1.x), v1.z*inv*ink2_bf16_lo(n1.y), v1.w*inv*ink2_bf16_hi(n1.y));
      d4[tid + 384] = make_float4(v2.x*inv*ink2_bf16_lo(n2.x), v2.y*inv*ink2_bf16_hi(n2.x), v2.z*inv*ink2_bf16_lo(n2.y), v2.w*inv*ink2_bf16_hi(n2.y));
      return;
    }
    float xs[24], ss = 0.0f;
    int k = 0;
    for (int c = tid; c < H; c += ntg, ++k) { const float v = __bfloat162float(row[c]); xs[k] = v; ss += v * v; }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / float(H) + eps);
    k = 0;
    for (int c = tid; c < H; c += ntg, ++k) dst[c] = xs[k] * inv * __bfloat162float(embed_norm[c]);
    return;
  }
  if (H == 2304 && ntg == 192) {
    const uint2* m2 = reinterpret_cast<const uint2*>(mask_embed);
    const uint2 u0 = __ldg(m2 + tid), u1 = __ldg(m2 + tid + 192), u2 = __ldg(m2 + tid + 384);
    float4* d4 = reinterpret_cast<float4*>(dst);
    d4[tid] = make_float4(ink2_bf16_lo(u0.x), ink2_bf16_hi(u0.x), ink2_bf16_lo(u0.y), ink2_bf16_hi(u0.y));
    d4[tid + 192] = make_float4(ink2_bf16_lo(u1.x), ink2_bf16_hi(u1.x), ink2_bf16_lo(u1.y), ink2_bf16_hi(u1.y));
    d4[tid + 384] = make_float4(ink2_bf16_lo(u2.x), ink2_bf16_hi(u2.x), ink2_bf16_lo(u2.y), ink2_bf16_hi(u2.y));
    return;
  }
  for (int c = tid; c < H; c += ntg) dst[c] = __bfloat162float(mask_embed[c]);
}

template <bool KV_BF16>
__global__ void k_mtp_qk_rope_cache(float* q, float* k, const float* v,
                                    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                    void* kc, void* vc, int Hq, int Hkv, int D, int pos0,
                                    const int32_t* d_pos_ptr, int kv_cap, bool is_swa,
                                    bool draft_mode, float eps, float theta) {
  __shared__ float sh[32];
  __shared__ float sh_vec[128];
  const int tid = threadIdx.x, t = blockIdx.y;
  const int base_pos = d_pos_ptr ? *d_pos_ptr : pos0;
  const int pos = base_pos + t;
  const int half_d = D >> 1, pair = tid < half_d ? tid : (tid - half_d);
  const float inv_freq = expf(-float(2 * pair) / float(D) * logf(theta));
  const float ang = float(pos) * inv_freq;
  const float ca = cosf(ang), sa = sinf(ang);

  if (!draft_mode) {
    const int h = blockIdx.x;
    const size_t off = (size_t(t) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid);
    const float kv = k[off];
    const float ss = block_sum(kv * kv, sh);
    sh_vec[tid] = kv * rsqrtf(ss / float(D) + eps) * __bfloat162float(k_norm[tid]);
    __syncthreads();
    const float cur = sh_vec[tid];
    const float partner = sh_vec[tid < half_d ? (tid + half_d) : (tid - half_d)];
    const float rot = tid < half_d ? (cur * ca - partner * sa) : (partner * sa + cur * ca);
    const int slot = is_swa ? (pos % kv_cap) : pos;
    const size_t dst = (size_t(slot) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid);
    kv_store1<KV_BF16>(kc, dst, rot);
    kv_store1<KV_BF16>(vc, dst, v[off]);
    return;
  }

  if (blockIdx.x < Hq) {
    const int h = blockIdx.x;
    const size_t off = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(tid);
    const float qv = q[off];
    const float ss = block_sum(qv * qv, sh);
    sh_vec[tid] = qv * rsqrtf(ss / float(D) + eps) * __bfloat162float(q_norm[tid]);
    __syncthreads();
    const float cur = sh_vec[tid];
    const float partner = sh_vec[tid < half_d ? (tid + half_d) : (tid - half_d)];
    q[off] = tid < half_d ? (cur * ca - partner * sa) : (partner * sa + cur * ca);
  } else {
    const int h = blockIdx.x - Hq;
    const size_t off = (size_t(t) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid);
    const float kv = k[off];
    const float ss = block_sum(kv * kv, sh);
    sh_vec[tid] = kv * rsqrtf(ss / float(D) + eps) * __bfloat162float(k_norm[tid]);
    __syncthreads();
    const float cur = sh_vec[tid];
    const float partner = sh_vec[tid < half_d ? (tid + half_d) : (tid - half_d)];
    k[off] = tid < half_d ? (cur * ca - partner * sa) : (partner * sa + cur * ca);
  }
}

template <bool KV_BF16>
__global__ void k_mtp_attn(const float* q, const void* kc, const void* vc,
                           const float* k_blk, const float* v_blk, float* out,
                           int B, int P, const int32_t* d_pos_ptr, int kv_cap,
                           int Hq, int Hkv, int D, bool is_swa, int swa, float scale) {
  __shared__ float tgm[8], tgl[8], tgo[8][128];
  const int h = blockIdx.x, b = blockIdx.y;
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31, nsg = blockDim.x >> 5;
  const int hkv = h / (Hq / Hkv), d0 = lane * 4;
  const size_t qoff = (size_t(b) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(d0);
  const float4 qv = *reinterpret_cast<const float4*>(q + qoff);
  const int P_val = d_pos_ptr ? *d_pos_ptr : P;
  const int k_start = (is_swa && P_val > swa) ? (P_val - swa) : 0;

  float mw = -INFINITY, lw = 0.0f;
  float4 ow = make_float4(0, 0, 0, 0);
  for (int kp = k_start + sg; kp < P_val; kp += nsg) {
    const int slot = is_swa ? (kp % kv_cap) : kp;
    const size_t koff = (size_t(slot) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
    const float4 kv = kv_load4<KV_BF16>(kc, koff);
    const float sc = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w) * scale;
    const float4 vv = kv_load4<KV_BF16>(vc, koff);
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = fmaxf(mw, sc);
      const float e1 = expf(mw - nm), e2 = expf(sc - nm);
      ow = make_float4(ow.x * e1 + vv.x * e2, ow.y * e1 + vv.y * e2,
                       ow.z * e1 + vv.z * e2, ow.w * e1 + vv.w * e2);
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  for (int kb = sg; kb < B; kb += nsg) {
    const size_t koff = (size_t(kb) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
    const float4 kv = *reinterpret_cast<const float4*>(k_blk + koff);
    const float sc = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w) * scale;
    const float4 vv = *reinterpret_cast<const float4*>(v_blk + koff);
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = fmaxf(mw, sc);
      const float e1 = expf(mw - nm), e2 = expf(sc - nm);
      ow = make_float4(ow.x * e1 + vv.x * e2, ow.y * e1 + vv.y * e2,
                       ow.z * e1 + vv.z * e2, ow.w * e1 + vv.w * e2);
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0) { tgm[sg] = mw; tgl[sg] = lw; }
  *reinterpret_cast<float4*>(&tgo[sg][d0]) = ow;
  __syncthreads();
  if (sg == 0) {
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = make_float4(0, 0, 0, 0);
    for (int s = 0; s < nsg; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *reinterpret_cast<const float4*>(&tgo[s][d0]);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = fmaxf(mm, m2);
        const float e1 = expf(mm - nm), e2 = expf(m2 - nm);
        oo = make_float4(oo.x * e1 + o2.x * e2, oo.y * e1 + o2.y * e2,
                         oo.z * e1 + o2.z * e2, oo.w * e1 + o2.w * e2);
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const float invl = 1.0f / ll;
    *reinterpret_cast<float4*>(out + qoff) = make_float4(oo.x * invl, oo.y * invl, oo.z * invl, oo.w * invl);
  }
}

template <int MC>
__global__ void k_mtp_gate_up_swiglu_tm(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w_gu_interleaved,
                                        float* __restrict__ out, int N, int K) {
  __shared__ float xs[MC][256];
  float ag[MC], au[MC];
  #pragma unroll
  for (int m = 0; m < MC; ++m) { ag[m] = 0.0f; au[m] = 0.0f; }
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, j0 = blockIdx.x * 4 + warp;
  const bool valid = (j0 < N);
  const uint16_t* wb = reinterpret_cast<const uint16_t*>(w_gu_interleaved);
  const size_t woff_g = size_t(2 * j0) * size_t(K), woff_u = woff_g + size_t(K);
  const int total_v4 = MC * 64;
  for (int kbase = 0; kbase < K; kbase += 256) {
    for (int idx = tid; idx < total_v4; idx += 128) {
      const int m = idx >> 6, c4 = (idx & 63) << 2;
      *reinterpret_cast<float4*>(xs[m] + c4) = __ldg(reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + kbase + c4));
    }
    __syncthreads();
    if (valid) {
      const int kl = lane * 8;
      const uint4 qg = __ldg(reinterpret_cast<const uint4*>(wb + woff_g + kbase + kl));
      const uint4 qu = __ldg(reinterpret_cast<const uint4*>(wb + woff_u + kbase + kl));
      const float g0=ink2_bf16_lo(qg.x), g1=ink2_bf16_hi(qg.x), g2=ink2_bf16_lo(qg.y), g3=ink2_bf16_hi(qg.y);
      const float g4=ink2_bf16_lo(qg.z), g5=ink2_bf16_hi(qg.z), g6=ink2_bf16_lo(qg.w), g7=ink2_bf16_hi(qg.w);
      const float u0=ink2_bf16_lo(qu.x), u1=ink2_bf16_hi(qu.x), u2=ink2_bf16_lo(qu.y), u3=ink2_bf16_hi(qu.y);
      const float u4=ink2_bf16_lo(qu.z), u5=ink2_bf16_hi(qu.z), u6=ink2_bf16_lo(qu.w), u7=ink2_bf16_hi(qu.w);
      #pragma unroll
      for (int m = 0; m < MC; ++m) {
        const float4 x0 = *reinterpret_cast<const float4*>(xs[m] + kl);
        const float4 x1 = *reinterpret_cast<const float4*>(xs[m] + kl + 4);
        ag[m] += (x0.x*g0 + x0.y*g1 + x0.z*g2 + x0.w*g3) + (x1.x*g4 + x1.y*g5 + x1.z*g6 + x1.w*g7);
        au[m] += (x0.x*u0 + x0.y*u1 + x0.z*u2 + x0.w*u3) + (x1.x*u4 + x1.y*u5 + x1.z*u6 + x1.w*u7);
      }
    }
    __syncthreads();
  }
  if (!valid) return;
  #pragma unroll
  for (int m = 0; m < MC; ++m) {
    const float vg = warp_sum(ag[m]), vu = warp_sum(au[m]);
    if (lane == 0) out[size_t(m) * size_t(N) + size_t(j0)] = (vg / (1.0f + expf(-vg))) * vu;
  }
}

__global__ void k_mtp_res_add_bf16(const float* res, const float* upd, float* out, int n) {
  const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  if (i4 < n4) {
    const float4 r = reinterpret_cast<const float4*>(res)[i4];
    const float4 u = reinterpret_cast<const float4*>(upd)[i4];
    auto rbf = [](float v) { return __bfloat162float(__float2bfloat16_rn(v)); };
    reinterpret_cast<float4*>(out)[i4] = make_float4(rbf(r.x) + u.x, rbf(r.y) + u.y, rbf(r.z) + u.z, rbf(r.w) + u.w);
    return;
  }
  const int i = (n4 << 2) + (i4 - n4);
  if (i < n) out[i] = __bfloat162float(__float2bfloat16_rn(res[i])) + upd[i];
}

__global__ void k_mtp_rnn_prep(const float* state, const __nv_bfloat16* markov_w1, const int32_t* d_tok_id,
                                 const float* hidden, const float* rec_feat, float* rnn_in,
                                 int rank, int H, bool first_step) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot = 2 * rank + 2 * H;
  if (i >= tot) return;
  if (i < rank) {
    rnn_in[i] = first_step ? 0.0f : state[i];
  } else if (i < 2 * rank) {
    const int id = d_tok_id[0];
    const uint16_t u = __ldg(reinterpret_cast<const uint16_t*>(markov_w1) + size_t(id) * size_t(rank) + size_t(i - rank));
    rnn_in[i] = ink2_bf16_lo(uint32_t(u));
  } else if (i < 2 * rank + H) {
    rnn_in[i] = hidden[i - 2 * rank];
  } else {
    rnn_in[i] = rec_feat[i - (2 * rank + H)];
  }
}

__global__ void k_mtp_rnn_update(const float* gates, float* state, float* latent, int rank, bool first_step) {
  const int c = threadIdx.x;
  if (c >= rank) return;
  const float prev_s = first_step ? 0.0f : state[c];
  const float k = 1.0f / (1.0f + expf(-gates[c]));
  const float u = tanhf(gates[rank + c]);
  const float l = tanhf(gates[2 * rank + c]);
  state[c] = k * prev_s + (1.0f - k) * u;
  latent[c] = l;
}

__global__ void k_linear_bf16_bias(const float* x, const __nv_bfloat16* w, const __nv_bfloat16* bias,
                                     float* y, int N, int K) {
  const int n = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31, k8 = K >> 3;
  if (n >= N) return;
  const uint4* w8 = reinterpret_cast<const uint4*>(w + size_t(n) * size_t(K));
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc = 0.0f;
  for (int i = lane; i < k8; i += 32) {
    const uint4 wv = __ldg(w8 + i);
    const float4 a = __ldg(x4 + i * 2 + 0), b = __ldg(x4 + i * 2 + 1);
    acc += a.x * ink2_bf16_lo(wv.x) + a.y * ink2_bf16_hi(wv.x) +
           a.z * ink2_bf16_lo(wv.y) + a.w * ink2_bf16_hi(wv.y) +
           b.x * ink2_bf16_lo(wv.z) + b.y * ink2_bf16_hi(wv.z) +
           b.z * ink2_bf16_lo(wv.w) + b.w * ink2_bf16_hi(wv.w);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[n] = acc + __bfloat162float(bias[n]);
}

__global__ void k_mtp_fold_add_bf16(const float* x, const __nv_bfloat16* fold_w, const float* fold_b,
                                      float* y, int V, int R, float softcap) {
  __shared__ float sh_x[256];
  for (int k = threadIdx.x; k < R; k += blockDim.x) sh_x[k] = x[k];
  __syncthreads();
  const int v = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31, r8 = R >> 3;
  if (v >= V) return;
  const uint4* w8 = reinterpret_cast<const uint4*>(fold_w + size_t(v) * size_t(R));
  const float4* x4 = reinterpret_cast<const float4*>(sh_x);
  float acc = 0.0f;
  for (int i = lane; i < r8; i += 32) {
    const uint4 wv = __ldg(w8 + i);
    const float4 a = x4[i * 2 + 0], b = x4[i * 2 + 1];
    acc += a.x * ink2_bf16_lo(wv.x) + a.y * ink2_bf16_hi(wv.x) +
           a.z * ink2_bf16_lo(wv.y) + a.w * ink2_bf16_hi(wv.y) +
           b.x * ink2_bf16_lo(wv.z) + b.y * ink2_bf16_hi(wv.z) +
           b.z * ink2_bf16_lo(wv.w) + b.w * ink2_bf16_hi(wv.w);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    float val = y[v] + acc + fold_b[v];
    if (softcap > 0.0f) val = softcap * tanhf(val / softcap);
    y[v] = val;
  }
}

__global__ void k_mtp_fold_add_q8(const float* x, const int8_t* fold_q, const __half* fold_s, const float* fold_b,
                                    float* y, int V, int R, float softcap) {
  __shared__ float sh_x[256];
  for (int k = threadIdx.x; k < R; k += blockDim.x) sh_x[k] = x[k];
  __syncthreads();
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int v = (blockIdx.x * 4 + sg) * 4 + (lane >> 3), blk = lane & 7;
  float acc = 0.0f;
  if (v < V) {
    const size_t woff = size_t(v) * size_t(R) + size_t(blk * 32);
    const uint4 q0 = __ldg(reinterpret_cast<const uint4*>(fold_q + woff + 0));
    const uint4 q1 = __ldg(reinterpret_cast<const uint4*>(fold_q + woff + 16));
    const float sc = __half2float(__ldg(fold_s + size_t(v) * size_t(R >> 5) + size_t(blk)));
    const float4* x4 = reinterpret_cast<const float4*>(sh_x + blk * 32);
    auto d4 = [](uint32_t u, float4 f) {
      return float(int8_t(u & 0xff)) * f.x + float(int8_t((u >> 8) & 0xff)) * f.y +
             float(int8_t((u >> 16) & 0xff)) * f.z + float(int8_t((u >> 24) & 0xff)) * f.w;
    };
    acc = (d4(q0.x, x4[0]) + d4(q0.y, x4[1]) + d4(q0.z, x4[2]) + d4(q0.w, x4[3]) +
           d4(q1.x, x4[4]) + d4(q1.y, x4[5]) + d4(q1.z, x4[6]) + d4(q1.w, x4[7])) * sc;
  }
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);
  acc += __shfl_xor_sync(0xffffffffu, acc, 2);
  acc += __shfl_xor_sync(0xffffffffu, acc, 4);
  if (blk == 0 && v < V) {
    float val = y[v] + acc + fold_b[v];
    if (softcap > 0.0f) val = softcap * tanhf(val / softcap);
    y[v] = val;
  }
}

template <int BM, bool IS_MXFP8>
static void dispatch_moe_tc(const float* x, const uint32_t* eid, const float* ew,
                            const uint8_t* w13, const uint8_t* w13_sc, const float* w13_outer,
                            const uint8_t* w2, const uint8_t* w2_sc, const float* w2_outer,
                            float* act_scratch, float* pair_out, float* out, int T, int H, int I, int K_top,
                            const MoeGroupBufs* gbuf, int n_experts, cudaStream_t s) {
  const int P = T * K_top;
  const int max_m = n_experts + P / BM;
  k_moe_group_build<BM><<<1, 256, 0, s>>>(eid, P, n_experts, gbuf->pair_idx, gbuf->tile_exp, gbuf->tile_m0,
                                          gbuf->tile_cnt, gbuf->num_tiles);
  k_moe_group_w13_tc<BM, IS_MXFP8><<<dim3(max_m, I / 64, 1), 256, 0, s>>>(
      x, w13_outer, w13, w13_sc, gbuf->pair_idx, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
      act_scratch, H, I, K_top);
  k_moe_group_w2_tc<BM, IS_MXFP8><<<dim3(max_m, H / 128, 1), 256, 0, s>>>(
      act_scratch, ew, w2_outer, w2, w2_sc, gbuf->pair_idx, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt,
      gbuf->num_tiles, pair_out, H, I);
  k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, T, 1), 64, 0, s>>>(
      reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
}

// ---- 13. Screened BF16 Unembed, DSpark W=4 Beam Tree, & Tree Verify/Commit (MIT-inspired by Niko1221/Strata) ----
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ int ink2_f2ord(float f) {
  const uint32_t b = __float_as_uint(f);
  return int(b ^ ((b >> 31) ? 0x7fffffffu : 0u));
}

__device__ __forceinline__ float ink2_ord2f(int o) {
  const uint32_t b = uint32_t(o);
  return __uint_as_float(b ^ ((b >> 31) ? 0x7fffffffu : 0u));
}

__device__ __forceinline__ uint32_t ink2_f2key(float f) {
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ int32_t ink2_dp4a_i8(uint32_t a, uint32_t b, int32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(int(a), int(b), c);
#else
  for (int sh = 0; sh < 32; sh += 8) c += int(int8_t((a >> sh) & 0xffu)) * int(int8_t((b >> sh) & 0xffu));
  return c;
#endif
}

__global__ void __launch_bounds__(256) k_build_unembed_screen_bf16(
    const __nv_bfloat16* __restrict__ W, int8_t* __restrict__ q8_out, float* __restrict__ sc_out,
    float* __restrict__ eps_out, int V, int H) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= V) return;
  const __nv_bfloat16* wr = W + size_t(row) * size_t(H);
  float am = 0.0f;
  for (int k = lane; k < H; k += 32) am = fmaxf(am, fabsf(__bfloat162float(wr[k])));
  am = warp_max(am);
  const float s = am * (1.0f / 127.0f);
  const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
  int8_t* dst = q8_out + size_t(row) * size_t(H);
  float err2 = 0.0f;
  for (int k = lane; k < H; k += 32) {
    const float w = __bfloat162float(wr[k]);
    int q = __float2int_rn(w * inv);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dst[k] = int8_t(q);
    const float d = w - float(q) * s;
    err2 += d * d;
  }
  err2 = warp_sum(err2);
  if (lane == 0) {
    sc_out[row] = s;
    eps_out[row] = sqrtf(err2) * 1.005f + 1e-3f;
  }
}

__global__ void __launch_bounds__(256) k_screen_row_l2norm(
    const float* __restrict__ h, float* __restrict__ out_norm, int* __restrict__ max_lb_bits,
    int32_t* __restrict__ cand_cnt, int M, int H) {
  __shared__ float sh[32];
  const int m = blockIdx.x;
  if (m >= M) return;
  const float* hm = h + size_t(m) * size_t(H);
  float ss = 0.0f;
  for (int d = threadIdx.x; d < H; d += 256) {
    const float v = hm[d];
    ss += v * v;
  }
  ss = block_sum(ss, sh);
  if (threadIdx.x == 0) {
    out_norm[m] = sqrtf(ss);
    if (max_lb_bits) max_lb_bits[m] = ink2_f2ord(-3.0e38f);
    if (cand_cnt) cand_cnt[m] = 0;
  }
}

template <int MAX_M, int KT>
__global__ void __launch_bounds__(256) k_screen_pass1_multiM(
    const int8_t* __restrict__ q8, const float* __restrict__ sc, const float* __restrict__ eps,
    const float* __restrict__ h, const float* __restrict__ h_norm, const uint8_t* __restrict__ suppress,
    float* __restrict__ dot_hat, int* __restrict__ max_lb_bits, int M, int V, int H) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row = blockIdx.x * 8 + warp;

  __shared__ alignas(16) float s_h[MAX_M][KT];
  float acc[MAX_M];
  #pragma unroll
  for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

  for (int k0 = 0; k0 < H; k0 += KT) {
    const int kt = min(KT, H - k0);
    for (int m = 0; m < M; ++m) {
      const float4* src4 = reinterpret_cast<const float4*>(h + size_t(m) * size_t(H) + size_t(k0));
      float4* dst4 = reinterpret_cast<float4*>(s_h[m]);
      for (int i = tid; i < (kt >> 2); i += 256) dst4[i] = src4[i];
    }
    __syncthreads();
    if (row < V) {
      const uint4* qt = reinterpret_cast<const uint4*>(q8 + size_t(row) * size_t(H) + size_t(k0));
      for (int i = lane; i < (kt >> 4); i += 32) {
        const uint4 v = __ldg(qt + i);
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint32_t p = u[q];
          const float f0 = float(int8_t(p & 0xffu));
          const float f1 = float(int8_t((p >> 8) & 0xffu));
          const float f2 = float(int8_t((p >> 16) & 0xffu));
          const float f3 = float(int8_t((p >> 24) & 0xffu));
          const int idx4 = i * 4 + q;
          #pragma unroll
          for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
              const float4 xv = reinterpret_cast<const float4*>(s_h[m])[idx4];
              acc[m] += f0 * xv.x + f1 * xv.y + f2 * xv.z + f3 * xv.w;
            }
          }
        }
      }
    }
    if (k0 + KT < H) __syncthreads();
  }
  if (row >= V) return;
  const float sc_r = sc[row];
  const float eps_r = eps ? eps[row] : 0.0f;
  const bool sup = (suppress != nullptr && suppress[row] != 0);
  #pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m < M) {
      const float d = warp_sum(acc[m]) * sc_r;
      if (lane == 0) {
        dot_hat[size_t(m) * size_t(V) + size_t(row)] = d;
        if (max_lb_bits && !sup) {
          const float lb = d - eps_r * h_norm[m];
          atomicMax(&max_lb_bits[m], ink2_f2ord(lb));
        }
      }
    }
  }
}

__global__ void __launch_bounds__(256) k_screen_pass2(
    const float* __restrict__ eps, const float* __restrict__ h_norm, const float* __restrict__ dot_hat,
    const int* __restrict__ max_lb_bits, const uint8_t* __restrict__ suppress,
    int32_t* __restrict__ cand_ids, int32_t* __restrict__ cand_cnt, int max_cand, int M, int V) {
  const int m = blockIdx.y;
  const int row = blockIdx.x * 256 + threadIdx.x;
  if (m >= M || row >= V) return;
  if (suppress && suppress[row]) return;
  const float lmax = ink2_ord2f(max_lb_bits[m]);
  const float ub = dot_hat[size_t(m) * size_t(V) + size_t(row)] + eps[row] * h_norm[m];
  if (ub >= lmax) {
    const int pos = atomicAdd(&cand_cnt[m], 1);
    if (pos < max_cand) cand_ids[size_t(m) * size_t(max_cand) + size_t(pos)] = row;
  }
}

__global__ void __launch_bounds__(256) k_screen_eval_bf16(
    const __nv_bfloat16* __restrict__ W, const float* __restrict__ h, const int32_t* __restrict__ cand_ids,
    const int32_t* __restrict__ cand_cnt, const uint8_t* __restrict__ suppress, int max_cand,
    float softcap, int32_t* __restrict__ out_argmax, int M, int V, int H) {
  __shared__ alignas(16) float s_h[2304];
  __shared__ float sv[8];
  __shared__ int32_t si[8];
  const int m = blockIdx.x;
  if (m >= M) return;
  const int raw_cnt = cand_cnt[m];
  const bool fallback_all = (raw_cnt <= 0 || raw_cnt > max_cand);
  const int n = fallback_all ? V : raw_cnt;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int K8 = H >> 3;
  const float inv_cap = (softcap > 0.0f) ? (1.0f / softcap) : 0.0f;
  if (H <= 2304) {
    const float4* h4 = reinterpret_cast<const float4*>(h + size_t(m) * size_t(H));
    float4* sh4 = reinterpret_cast<float4*>(s_h);
    for (int i = threadIdx.x; i < (H >> 2); i += 256) sh4[i] = h4[i];
  }
  __syncthreads();
  const float* hm = (H <= 2304) ? s_h : (h + size_t(m) * size_t(H));

  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;
  for (int ci = warp; ci < n; ci += 8) {
    const int32_t row = fallback_all ? ci : cand_ids[size_t(m) * size_t(max_cand) + size_t(ci)];
    if (fallback_all && suppress && suppress[row]) continue;
    const uint4* wr = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(W) + size_t(row) * size_t(H));
    float acc = 0.0f;
    for (int k8 = lane; k8 < K8; k8 += 32) {
      const uint4 u = __ldg(wr + k8);
      const float4 x0 = *reinterpret_cast<const float4*>(hm + k8 * 8 + 0);
      const float4 x1 = *reinterpret_cast<const float4*>(hm + k8 * 8 + 4);
      acc += (x0.x * ink2_bf16_lo(u.x) + x0.y * ink2_bf16_hi(u.x) +
              x0.z * ink2_bf16_lo(u.y) + x0.w * ink2_bf16_hi(u.y)) +
             (x1.x * ink2_bf16_lo(u.z) + x1.y * ink2_bf16_hi(u.z) +
              x1.z * ink2_bf16_lo(u.w) + x1.w * ink2_bf16_hi(u.w));
    }
    acc = warp_sum(acc);
    const float val = (softcap > 0.0f) ? (softcap * tanhf(acc * inv_cap)) : acc;
    if (lane == 0 && (val > best_v || (val == best_v && row < best_i))) {
      best_v = val;
      best_i = row;
    }
  }
  if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float bv = sv[0];
    int32_t bi = si[0];
    for (int w = 1; w < 8; ++w) {
      if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
        bv = sv[w];
        bi = si[w];
      }
    }
    out_argmax[m] = (bi == 0x7fffffff) ? 0 : bi;
  }
}

__global__ void __launch_bounds__(256) k_dspark_quant_h_q8(
    const float* __restrict__ h7, int8_t* __restrict__ h_q8, float* __restrict__ h_sc, int H) {
  __shared__ float sm[8];
  const int m = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const float* src = h7 + size_t(m) * size_t(H);
  float am = 0.0f;
  for (int d = tid; d < H; d += 256) am = fmaxf(am, fabsf(src[d]));
  am = warp_max(am);
  if (lane == 0) sm[warp] = am;
  __syncthreads();
  am = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
             fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
  const float s = am * (1.0f / 127.0f);
  const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
  int8_t* dst = h_q8 + size_t(m) * size_t(H);
  for (int d = tid; d < H; d += 256) {
    int q = __float2int_rn(src[d] * inv);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dst[d] = int8_t(q);
  }
  if (tid == 0) h_sc[m] = s;
}

__global__ void __launch_bounds__(128) k_dspark_screen_dp4a7(
    const int8_t* __restrict__ q8, const float* __restrict__ sc, const int8_t* __restrict__ h_q8,
    const float* __restrict__ h_sc, const uint8_t* __restrict__ suppress,
    float* __restrict__ base_logits, int V, int H) {
  __shared__ alignas(16) uint32_t s_h[7][2304 / 4];
  const int tid = threadIdx.x;
  const int h4 = H >> 2, h16 = H >> 4;
  for (int m = 0; m < 7; ++m) {
    const uint4* src = reinterpret_cast<const uint4*>(h_q8 + size_t(m) * size_t(H));
    uint4* dst = reinterpret_cast<uint4*>(s_h[m]);
    for (int i = tid; i < h16; i += 128) dst[i] = src[i];
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31, row = blockIdx.x * 4 + warp;
  if (row >= V) return;
  const uint4* q16 = reinterpret_cast<const uint4*>(q8 + size_t(row) * size_t(H));
  int32_t acc[7] = {};
  for (int i = lane; i < h16; i += 32) {
    const uint4 v = __ldg(q16 + i);
    const int base = i * 4;
    #pragma unroll
    for (int m = 0; m < 7; ++m) {
      const uint4 hv = *reinterpret_cast<const uint4*>(s_h[m] + base);
      int32_t a = acc[m];
      a = ink2_dp4a_i8(v.x, hv.x, a);
      a = ink2_dp4a_i8(v.y, hv.y, a);
      a = ink2_dp4a_i8(v.z, hv.z, a);
      a = ink2_dp4a_i8(v.w, hv.w, a);
      acc[m] = a;
    }
  }
  const float sc_r = sc[row];
  const bool sup = (suppress != nullptr && suppress[row] != 0);
  #pragma unroll
  for (int m = 0; m < 7; ++m) {
    int32_t s = acc[m];
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) base_logits[size_t(m) * size_t(V) + size_t(row)] = sup ? -1e30f : (float(s) * sc_r * h_sc[m]);
  }
  (void)h4;
}

__global__ void __launch_bounds__(256) k_dspark_top_cand_s1(
    const float* __restrict__ base_logits, int V, int32_t* __restrict__ tmp_ids, float* __restrict__ tmp_vals) {
  const int s = blockIdx.y, ch = blockIdx.x, tid = threadIdx.x;
  const float* lg = base_logits + size_t(s) * size_t(V);
  float b0 = -3.0e38f, b1 = -3.0e38f;
  int32_t i0 = 0, i1 = 0;
  for (int base = ch * 256; base < V; base += 128 * 256) {
    const int idx = base + tid;
    if (idx < V) {
      const float v = lg[idx];
      if (v > b0) { b1 = b0; i1 = i0; b0 = v; i0 = idx; }
      else if (v > b1) { b1 = v; i1 = idx; }
    }
  }
  const int warp = tid >> 5, lane = tid & 31;
  float wv0 = b0;
  int32_t wi0 = i0;
  #pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, wv0, o);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, wi0, o);
    if (ov > wv0 || (ov == wv0 && oi < wi0)) { wv0 = ov; wi0 = oi; }
  }
  float wv1 = (i0 == wi0) ? b1 : b0;
  int32_t wi1 = (i0 == wi0) ? i1 : i0;
  #pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, wv1, o);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, wi1, o);
    if (ov > wv1 || (ov == wv1 && oi < wi1)) { wv1 = ov; wi1 = oi; }
  }
  if (lane == 0) {
    const size_t ob = size_t(s) * 2048 + size_t(ch) * 16 + size_t(warp) * 2;
    tmp_ids[ob + 0] = wi0; tmp_vals[ob + 0] = wv0;
    tmp_ids[ob + 1] = wi1; tmp_vals[ob + 1] = wv1;
  }
}

__global__ void __launch_bounds__(256) k_dspark_top_cand_s2(
    const int32_t* __restrict__ tmp_ids, const float* __restrict__ tmp_vals, int K_cand,
    int32_t* __restrict__ cand_ids, float* __restrict__ cand_base) {
  __shared__ int32_t s_id[2048];
  __shared__ float s_val[2048];
  __shared__ int hist[256];
  __shared__ uint32_t prefix_s, thr_s;
  __shared__ int need_s, cnt;
  const int s = blockIdx.x, tid = threadIdx.x;
  for (int i = tid; i < 2048; i += 256) {
    s_id[i] = tmp_ids[size_t(s) * 2048 + size_t(i)];
    s_val[i] = tmp_vals[size_t(s) * 2048 + size_t(i)];
  }
  if (tid == 0) { prefix_s = 0; need_s = K_cand; }
  __syncthreads();
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    hist[tid] = 0;
    __syncthreads();
    const uint32_t pre = prefix_s;
    for (int i = tid; i < 2048; i += 256) {
      const uint32_t k = ink2_f2key(s_val[i]);
      if (pass == 0 || (k >> (shift + 8)) == (pre >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255u], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int acc = 0, d = 255;
      const int need = need_s;
      for (; d > 0; --d) {
        if (acc + hist[d] >= need) break;
        acc += hist[d];
      }
      need_s = need - acc;
      prefix_s = pre | (uint32_t(d) << shift);
    }
    __syncthreads();
  }
  if (tid == 0) { thr_s = prefix_s; cnt = 0; }
  __syncthreads();
  const uint32_t thr = thr_s;
  int32_t* o_id = cand_ids + size_t(s) * size_t(K_cand);
  float* o_val = cand_base ? (cand_base + size_t(s) * size_t(K_cand)) : nullptr;
  for (int i = tid; i < 2048; i += 256) {
    if (ink2_f2key(s_val[i]) > thr) {
      const int p = atomicAdd(&cnt, 1);
      if (p < K_cand) { o_id[p] = s_id[i]; if (o_val) o_val[p] = s_val[i]; }
    }
  }
  __syncthreads();
  if (cnt < K_cand) {
    for (int i = tid; i < 2048; i += 256) {
      if (ink2_f2key(s_val[i]) == thr) {
        const int p = atomicAdd(&cnt, 1);
        if (p < K_cand) { o_id[p] = s_id[i]; if (o_val) o_val[p] = s_val[i]; }
      }
    }
  }
}

__global__ void __launch_bounds__(256) k_dspark_beam_init(
    const int32_t* __restrict__ anchor_id, float* __restrict__ beam_state,
    int32_t* __restrict__ beam_prev_tok, float* __restrict__ beam_logp, int32_t* __restrict__ beam_node,
    int32_t* __restrict__ pool_tok, int8_t* __restrict__ pool_parent, uint8_t* __restrict__ pool_depth,
    float* __restrict__ pool_logp, int32_t* __restrict__ pool_cnt) {
  constexpr int kBW = 4, kR = 256;
  const int tid = threadIdx.x;
  for (int b = 0; b < kBW; ++b) beam_state[size_t(b) * kR + size_t(tid)] = 0.0f;
  if (tid == 0) {
    const int32_t anc = *anchor_id;
    beam_prev_tok[0] = anc;
    beam_logp[0] = 0.0f;
    beam_node[0] = 0;
    for (int b = 1; b < kBW; ++b) {
      beam_prev_tok[b] = anc;
      beam_logp[b] = -1e30f;
      beam_node[b] = 0;
    }
    pool_tok[0] = anc;
    pool_parent[0] = -1;
    pool_depth[0] = 0;
    pool_logp[0] = 0.0f;
    *pool_cnt = 1;
  }
}

__global__ void __launch_bounds__(256) k_dspark_markov_prep_batch(
    const int32_t* __restrict__ prev_toks, const float* __restrict__ states, const float* __restrict__ hidden,
    const float* __restrict__ rec_feat, const __nv_bfloat16* __restrict__ markov_w1,
    float* __restrict__ joint_out, int nw, int R, int H) {
  const int b = blockIdx.x, tid = threadIdx.x;
  if (b >= nw) return;
  const int32_t pt = prev_toks[b];
  float* jo = joint_out + size_t(b) * size_t(2 * R + 2 * H);
  if (tid < R) {
    jo[tid] = states[size_t(b) * size_t(R) + size_t(tid)];
    const uint16_t u = __ldg(reinterpret_cast<const uint16_t*>(markov_w1) + size_t(pt) * size_t(R) + size_t(tid));
    jo[R + tid] = ink2_bf16_lo(uint32_t(u));
  }
  for (int d = tid; d < H; d += 256) {
    jo[2 * R + d] = hidden[d];
    jo[2 * R + H + d] = rec_feat[d];
  }
}

__global__ void __launch_bounds__(256) k_dspark_markov_update_batch(
    const float* __restrict__ proj, const __nv_bfloat16* __restrict__ rp_b,
    const float* __restrict__ state_in, float* __restrict__ state_next, float* __restrict__ lat_out,
    int nw, int R) {
  const int b = blockIdx.x, c = threadIdx.x;
  if (b >= nw || c >= R) return;
  const float* gp = proj + size_t(b) * size_t(3 * R);
  const float g0 = gp[c] + __bfloat162float(__ldg(rp_b + c));
  const float g1 = gp[R + c] + __bfloat162float(__ldg(rp_b + R + c));
  const float g2 = gp[2 * R + c] + __bfloat162float(__ldg(rp_b + 2 * R + c));
  const float k = 1.0f / (1.0f + expf(-g0));
  const float u = tanhf(g1);
  const float l = tanhf(g2);
  state_next[size_t(b) * size_t(R) + size_t(c)] = k * state_in[size_t(b) * size_t(R) + size_t(c)] + (1.0f - k) * u;
  lat_out[size_t(b) * size_t(R) + size_t(c)] = l;
}

__global__ void __launch_bounds__(256) k_dspark_add_bias_h_batch(
    float* __restrict__ corr, const float* __restrict__ ho_bias, const float* __restrict__ hidden,
    int nw, int H) {
  const int b = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
  if (b >= nw || d >= H) return;
  corr[size_t(b) * size_t(H) + size_t(d)] += ho_bias[d] + hidden[d];
}

__global__ void __launch_bounds__(256) k_dspark_score_cand_dp4a_batch(
    const int8_t* __restrict__ q8, const float* __restrict__ sc, const float* __restrict__ corr,
    const int32_t* __restrict__ cand_ids, const uint8_t* __restrict__ suppress,
    int K_cand, int nw, float softcap, float* __restrict__ cand_scores, int H) {
  constexpr int kBW = 4;
  __shared__ alignas(16) uint32_t s_c[kBW][2304 / 4];
  __shared__ float s_sc[kBW];
  __shared__ float sm[8];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  for (int b = 0; b < nw; ++b) {
    const float* cp = corr + size_t(b) * size_t(H);
    float am = 0.0f;
    for (int d = tid; d < H; d += 256) am = fmaxf(am, fabsf(cp[d]));
    am = warp_max(am);
    if (lane == 0) sm[warp] = am;
    __syncthreads();
    am = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
               fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
    const float csc = am * (1.0f / 127.0f);
    const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
    int8_t* dst = reinterpret_cast<int8_t*>(s_c[b]);
    for (int d = tid; d < H; d += 256) {
      int q = __float2int_rn(cp[d] * inv);
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      dst[d] = int8_t(q);
    }
    if (tid == 0) s_sc[b] = csc;
    __syncthreads();
  }

  const int k = blockIdx.x * 8 + warp;
  if (k >= K_cand) return;
  const int32_t row = cand_ids[k];
  const int h16 = H >> 4;
  const uint4* q16 = reinterpret_cast<const uint4*>(q8 + size_t(row) * size_t(H));
  int32_t acc[kBW] = {};
  for (int i = lane; i < h16; i += 32) {
    const uint4 v = __ldg(q16 + i);
    const int base = i * 4;
    #pragma unroll
    for (int b = 0; b < kBW; ++b) {
      if (b < nw) {
        const uint4 cv = *reinterpret_cast<const uint4*>(s_c[b] + base);
        int32_t a = acc[b];
        a = ink2_dp4a_i8(v.x, cv.x, a);
        a = ink2_dp4a_i8(v.y, cv.y, a);
        a = ink2_dp4a_i8(v.z, cv.z, a);
        a = ink2_dp4a_i8(v.w, cv.w, a);
        acc[b] = a;
      }
    }
  }
  const float sc_r = sc[row];
  const bool sup = (suppress != nullptr && suppress[row] != 0);
  const float inv_cap = (softcap > 0.0f) ? (1.0f / softcap) : 0.0f;
  #pragma unroll
  for (int b = 0; b < kBW; ++b) {
    if (b < nw) {
      int32_t s = acc[b];
      #pragma unroll
      for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (lane == 0) {
        float val = sup ? -1e30f : (float(s) * sc_r * s_sc[b]);
        if (!sup && softcap > 0.0f) val = softcap * tanhf(val * inv_cap);
        cand_scores[size_t(b) * size_t(K_cand) + size_t(k)] = val;
      }
    }
  }
}

__global__ void __launch_bounds__(256) k_dspark_beam_expand(
    const float* __restrict__ cand_scores, const int32_t* __restrict__ cand_ids,
    const float* __restrict__ state_next, int K_cand, int nw, int beam_w, int depth,
    float* __restrict__ state_out, int32_t* __restrict__ beam_prev_tok, float* __restrict__ beam_logp,
    int32_t* __restrict__ beam_node, int32_t* __restrict__ pool_tok, int8_t* __restrict__ pool_parent,
    uint8_t* __restrict__ pool_depth, float* __restrict__ pool_logp, int32_t* __restrict__ pool_cnt) {
  constexpr int kBW = 4, kPoolCap = 64, kR = 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __shared__ float s_wval[8];
  __shared__ int32_t s_widx[8];
  __shared__ float sh_sum[32];
  __shared__ float s_mx, s_logz;
  __shared__ int32_t s_picked[4];
  __shared__ int32_t exp_tok[16], exp_pnode[16], exp_pbeam[16], win_beam[kBW];
  __shared__ float exp_lp[16];

  for (int b = 0; b < nw; ++b) {
    const float* sc = cand_scores + size_t(b) * size_t(K_cand);
    float mv = -3.0e38f;
    for (int k = tid; k < K_cand; k += 256) mv = fmaxf(mv, sc[k]);
    mv = warp_max(mv);
    if (lane == 0) s_wval[warp] = mv;
    __syncthreads();
    if (tid == 0) {
      float g = s_wval[0];
      for (int w = 1; w < 8; ++w) g = fmaxf(g, s_wval[w]);
      s_mx = g;
    }
    __syncthreads();
    const float mx_v = s_mx;
    float z = 0.0f;
    for (int k = tid; k < K_cand; k += 256) z += __expf(sc[k] - mx_v);
    z = block_sum(z, sh_sum);
    if (tid == 0) s_logz = mx_v + logf(fmaxf(z, 1e-30f));
    __syncthreads();
    const float logz = s_logz;
    const float parent_lp = beam_logp[b];
    const int32_t parent_nd = beam_node[b];

    for (int r = 0; r < 4; ++r) {
      float bv = -3.0e38f;
      int32_t bk = -1;
      for (int k = tid; k < K_cand; k += 256) {
        bool skip = false;
        for (int j = 0; j < r; ++j) if (k == s_picked[j]) skip = true;
        if (!skip) {
          const float v = sc[k];
          if (v > bv || (v == bv && (bk < 0 || cand_ids[k] < cand_ids[bk]))) { bv = v; bk = k; }
        }
      }
      #pragma unroll
      for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int32_t ok = __shfl_xor_sync(0xffffffffu, bk, o);
        if (ok >= 0 && (bk < 0 || ov > bv || (ov == bv && cand_ids[ok] < cand_ids[bk]))) { bv = ov; bk = ok; }
      }
      if (lane == 0) { s_wval[warp] = bv; s_widx[warp] = bk; }
      __syncthreads();
      if (tid == 0) {
        float gv = -3.0e38f;
        int32_t gk = -1;
        for (int w = 0; w < 8; ++w) {
          const int32_t ok = s_widx[w];
          const float ov = s_wval[w];
          if (ok >= 0 && (gk < 0 || ov > gv || (ov == gv && cand_ids[ok] < cand_ids[gk]))) { gv = ov; gk = ok; }
        }
        s_picked[r] = gk;
        const int idx = b * 4 + r;
        exp_tok[idx] = (gk >= 0) ? cand_ids[gk] : 0;
        exp_lp[idx] = (gk >= 0) ? (parent_lp + (gv - logz)) : -1e30f;
        exp_pnode[idx] = parent_nd;
        exp_pbeam[idx] = b;
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
    const int n_exp = nw * 4;
    int ord[16];
    for (int i = 0; i < n_exp; ++i) ord[i] = i;
    for (int i = 1; i < n_exp; ++i) {
      const int key = ord[i];
      int j = i - 1;
      while (j >= 0 && exp_lp[ord[j]] < exp_lp[key]) { ord[j + 1] = ord[j]; --j; }
      ord[j + 1] = key;
    }
    const int base = *pool_cnt;
    const int k_rec = min(n_exp, min((depth == 1) ? 4 : 8, kPoolCap - base));
    for (int j = 0; j < k_rec; ++j) {
      const int id = ord[j];
      const int dst = base + j;
      pool_tok[dst] = exp_tok[id];
      pool_parent[dst] = int8_t(exp_pnode[id]);
      pool_depth[dst] = uint8_t(depth);
      pool_logp[dst] = exp_lp[id];
    }
    *pool_cnt = base + k_rec;
    const int k_keep = min(beam_w, k_rec);
    for (int j = 0; j < beam_w; ++j) {
      if (j < k_keep) {
        const int id = ord[j];
        beam_prev_tok[j] = exp_tok[id];
        beam_logp[j] = exp_lp[id];
        beam_node[j] = base + j;
        win_beam[j] = exp_pbeam[id];
      } else {
        beam_prev_tok[j] = exp_tok[ord[0]];
        beam_logp[j] = -1e30f;
        beam_node[j] = base;
        win_beam[j] = exp_pbeam[ord[0]];
      }
    }
  }
  __syncthreads();
  for (int j = 0; j < beam_w; ++j) {
    const int pb = win_beam[j];
    state_out[size_t(j) * kR + size_t(tid)] = state_next[size_t(pb) * kR + size_t(tid)];
  }
}

__global__ void k_sconv_static_tree(
    const float* __restrict__ x, const void* __restrict__ w, bool w_f32, const float* __restrict__ hist,
    float* __restrict__ out, float* __restrict__ stage_raw, int stage_stride, int stage_ch_off,
    const int8_t* __restrict__ p1, const int8_t* __restrict__ p2, const int8_t* __restrict__ p3,
    int M, int C, float scale, const float* __restrict__ res_add) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  const float h0 = hist[0 * size_t(C) + c];
  const float h1 = hist[1 * size_t(C) + c];
  const float h2 = hist[2 * size_t(C) + c];
  float xv[12];
  for (int u = 0; u < M; ++u) {
    const float xt = x[size_t(u) * size_t(C) + size_t(c)];
    xv[u] = xt;
    stage_raw[size_t(u) * size_t(stage_stride) + size_t(stage_ch_off) + size_t(c)] = xt;
  }
  for (int u = 0; u < M; ++u) {
    const int a1 = p1[u], a2 = p2[u], a3 = p3[u];
    const float v1 = (a1 >= 0) ? xv[a1] : h2;
    const float v2 = (a2 >= 0) ? xv[a2] : ((a1 >= 0) ? h2 : h1);
    const float v3 = (a3 >= 0) ? xv[a3] : ((a2 >= 0) ? h2 : ((a1 >= 0) ? h1 : h0));
    const float xt = xv[u];
    const float y = ((w0 * v3 + w1 * v2 + w2 * v1 + w3 * xt) + xt) * scale;
    const size_t idx = size_t(u) * size_t(C) + size_t(c);
    out[idx] = res_add ? (res_add[idx] + y) : y;
  }
}

__global__ void k_moe_reduce8_sconv_static_tree(
    const float* __restrict__ pair_out, const void* __restrict__ w, bool w_f32, const float* __restrict__ hist,
    float* __restrict__ out, float* __restrict__ stage_raw, int stage_stride, int stage_ch_off,
    const int8_t* __restrict__ p1, const int8_t* __restrict__ p2, const int8_t* __restrict__ p3,
    int M, int C, int K_top, float scale, const float* __restrict__ res_add) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  const float h0 = hist[0 * size_t(C) + c];
  const float h1 = hist[1 * size_t(C) + c];
  const float h2 = hist[2 * size_t(C) + c];
  float xv[12];
  for (int u = 0; u < M; ++u) {
    float xt = 0.0f;
    const size_t pbase = size_t(u) * size_t(K_top) * size_t(C) + size_t(c);
    for (int k = 0; k < K_top; ++k) xt += pair_out[pbase + size_t(k) * size_t(C)];
    xv[u] = xt;
    stage_raw[size_t(u) * size_t(stage_stride) + size_t(stage_ch_off) + size_t(c)] = xt;
  }
  for (int u = 0; u < M; ++u) {
    const int a1 = p1[u], a2 = p2[u], a3 = p3[u];
    const float v1 = (a1 >= 0) ? xv[a1] : h2;
    const float v2 = (a2 >= 0) ? xv[a2] : ((a1 >= 0) ? h2 : h1);
    const float v3 = (a3 >= 0) ? xv[a3] : ((a2 >= 0) ? h2 : ((a1 >= 0) ? h1 : h0));
    const float xt = xv[u];
    const float y = ((w0 * v3 + w1 * v2 + w2 * v1 + w3 * xt) + xt) * scale;
    const size_t idx = size_t(u) * size_t(C) + size_t(c);
    out[idx] = res_add ? (res_add[idx] + y) : y;
  }
}

__global__ void k_sconv_static_kv_tree(
    float* __restrict__ xk, const void* __restrict__ wk, bool wk_f32, const float* __restrict__ hk,
    float* __restrict__ xv, const void* __restrict__ wv, bool wv_f32, const float* __restrict__ hv,
    float* __restrict__ stage_raw, int stage_stride,
    const int8_t* __restrict__ p1, const int8_t* __restrict__ p2, const int8_t* __restrict__ p3,
    int M, int C, float scale) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const bool is_v = (blockIdx.y != 0);
  float* x = is_v ? xv : xk;
  const void* w = is_v ? wv : wk;
  const bool w_f32 = is_v ? wv_f32 : wk_f32;
  const float* hist = is_v ? hv : hk;
  float* out = x;
  const int ch_off = is_v ? C : 0;

  float w0, w1, w2, w3;
  if (w_f32) {
    const float4 wf = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + size_t(c) * 4;
    w0 = __bfloat162float(wb[0]); w1 = __bfloat162float(wb[1]);
    w2 = __bfloat162float(wb[2]); w3 = __bfloat162float(wb[3]);
  }
  const float h0 = hist[0 * size_t(C) + c];
  const float h1 = hist[1 * size_t(C) + c];
  const float h2 = hist[2 * size_t(C) + c];
  float xarr[12];
  for (int u = 0; u < M; ++u) {
    const float xt = x[size_t(u) * size_t(C) + size_t(c)];
    xarr[u] = xt;
    stage_raw[size_t(u) * size_t(stage_stride) + size_t(ch_off) + size_t(c)] = xt;
  }
  for (int u = 0; u < M; ++u) {
    const int a1 = p1[u], a2 = p2[u], a3 = p3[u];
    const float v1 = (a1 >= 0) ? xarr[a1] : h2;
    const float v2 = (a2 >= 0) ? xarr[a2] : ((a1 >= 0) ? h2 : h1);
    const float v3 = (a3 >= 0) ? xarr[a3] : ((a2 >= 0) ? h2 : ((a1 >= 0) ? h1 : h0));
    const float xt = xarr[u];
    out[size_t(u) * size_t(C) + size_t(c)] = ((w0 * v3 + w1 * v2 + w2 * v1 + w3 * xt) + xt) * scale;
  }
}

__global__ void k_sconv_dynamic_tree(
    const float* __restrict__ x, const float* __restrict__ dyn, const void* __restrict__ static_w, bool sw_f32,
    const float* __restrict__ hist, float* __restrict__ out, float* __restrict__ stage_raw, int stage_stride,
    int stage_ch_off, const int8_t* __restrict__ p1, const int8_t* __restrict__ p2, const int8_t* __restrict__ p3,
    int M, int C, int n_heads) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int head_dim = C / n_heads, h = c / head_dim;
  float sw[4];
  for (int lag = 0; lag < 4; ++lag) {
    const size_t idx = size_t(lag) * size_t(C) + size_t(c);
    sw[lag] = sw_f32 ? static_cast<const float*>(static_w)[idx]
                     : __bfloat162float(static_cast<const __nv_bfloat16*>(static_w)[idx]);
  }
  const float h0 = hist[0 * size_t(C) + c];
  const float h1 = hist[1 * size_t(C) + c];
  const float h2 = hist[2 * size_t(C) + c];
  float xv[12];
  for (int u = 0; u < M; ++u) {
    const float xt = x[size_t(u) * size_t(C) + size_t(c)];
    xv[u] = xt;
    stage_raw[size_t(u) * size_t(stage_stride) + size_t(stage_ch_off) + size_t(c)] = xt;
  }
  for (int u = 0; u < M; ++u) {
    const int a1 = p1[u], a2 = p2[u], a3 = p3[u];
    const float v1 = (a1 >= 0) ? xv[a1] : h2;
    const float v2 = (a2 >= 0) ? xv[a2] : ((a1 >= 0) ? h2 : h1);
    const float v3 = (a3 >= 0) ? xv[a3] : ((a2 >= 0) ? h2 : ((a1 >= 0) ? h1 : h0));
    const float xt = xv[u];
    const float4 d = *reinterpret_cast<const float4*>(dyn + (size_t(u) * size_t(n_heads) + size_t(h)) * 4);
    const float conv = (sw[0] + d.x) * xt + (sw[1] + d.y) * v1 + (sw[2] + d.z) * v2 + (sw[3] + d.w) * v3;
    out[size_t(u) * size_t(C) + size_t(c)] = (conv + xt) * (1.0f / 3.0f);
  }
}

__global__ void k_qkvr_prep_tree(
    float* q, const float* k, const float* v, float* r, const void* q_scale, const void* r_scale, bool sc_f32,
    float* stage_k, float* stage_v, int Hq, int Hkv, int D, int d_rel, int pos0,
    const int32_t* d_pos_ptr, const uint8_t* d_depth, float qk_clip, float r_clip, float v_eps,
    float log_floor, float log_alpha, bool is_swa) {
  __shared__ float sh[32];
  const int kind_h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int pos = (d_pos_ptr ? *d_pos_ptr : pos0) + int(d_depth[t]);
  const float tau = is_swa ? 1.0f : (1.0f + log_alpha * logf(fmaxf(1.0f, float(pos + 1) / log_floor)));

  if (kind_h < Hq) {
    const int h = kind_h;
    const size_t off = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(tid);
    const size_t soff = size_t(h) * size_t(D) + size_t(tid);
    const float sc = sc_f32 ? static_cast<const float*>(q_scale)[soff]
                            : __bfloat162float(static_cast<const __nv_bfloat16*>(q_scale)[soff]);
    const float val = q[off] * sc;
    const float ss = block_sum(val * val, sh);
    q[off] = val * fminf(1.0f, qk_clip / (sqrtf(ss) + 1e-6f)) * tau;
    return;
  }
  int rem = kind_h - Hq;
  if (rem < Hkv) {
    const int h = rem;
    const size_t off = (size_t(t) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid);
    const float val = k[off];
    const float ss = block_sum(val * val, sh);
    stage_k[off] = val * fminf(1.0f, qk_clip / (sqrtf(ss) + 1e-6f));
    return;
  }
  rem -= Hkv;
  if (rem < Hkv) {
    const int h = rem;
    const size_t off = (size_t(t) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid);
    const float val = v[off];
    const float ss = block_sum(val * val, sh);
    stage_v[off] = val * rsqrtf(ss / float(D) + v_eps);
    return;
  }
  const int h = rem - Hkv;
  const size_t off = (size_t(t) * size_t(Hq) + size_t(h)) * size_t(d_rel) + size_t(tid);
  const size_t soff = size_t(h) * size_t(d_rel) + size_t(tid);
  float val = 0.0f;
  if (tid < d_rel) {
    const float sc = sc_f32 ? static_cast<const float*>(r_scale)[soff]
                            : __bfloat162float(static_cast<const __nv_bfloat16*>(r_scale)[soff]);
    val = r[off] * sc;
  }
  const float ss = block_sum(val * val, sh);
  if (tid < d_rel) r[off] = val * fminf(1.0f, r_clip / (sqrtf(ss) + 1e-6f)) * tau;
}

template <int TILE, bool KV_BF16>
__global__ void k_attn_tree_tm(
    const float* __restrict__ q, const void* __restrict__ kc, const void* __restrict__ vc,
    const float* __restrict__ stage_k, const float* __restrict__ stage_v, const float* __restrict__ rb,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ po,
    int M, int pos0, const int32_t* __restrict__ d_pos_ptr,
    const uint8_t* __restrict__ d_depth, const uint32_t* __restrict__ d_anc_mask,
    int kv_cap, int Hq, int Hkv, int D, int rel_extent, bool is_swa, int swa, int nchunks, float scale) {
  __shared__ float tgm[4], tgl[4], tgo[4][128];
  const int h = blockIdx.x, cid = blockIdx.y, q_base = blockIdx.z * TILE;
  const int nt = min(TILE, M - q_base);
  if (nt <= 0) return;
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cur_pos = d_pos_ptr ? *d_pos_ptr : pos0;
  const int k_min = (is_swa && cur_pos + 1 > swa) ? (cur_pos + 1 - swa) : 0;
  const int k_max = cur_pos;
  const int lk = max(0, k_max - k_min);
  int eff_chunk = max(32, (lk + nchunks - 1) / nchunks);
  eff_chunk = (eff_chunk + 3) / 4 * 4;
  const int c0 = k_min + cid * eff_chunk;
  const int c1 = min(k_max, c0 + eff_chunk);
  if (cid > 0 && c0 >= c1) {
    if (sg == 0 && lane == 0) {
      for (int t = 0; t < nt; ++t) {
        const size_t idx = (size_t(q_base + t) * size_t(Hq) + size_t(h)) * size_t(nchunks) + size_t(cid);
        pm[idx] = -INFINITY;
        pl[idx] = 0.0f;
      }
    }
    return;
  }

  const int hkv = h / (Hq / Hkv);
  const int d0 = lane * 4;
  float4 qv[TILE], ow[TILE];
  float mw[TILE], lw[TILE];
  int q_depth[TILE];
  uint32_t q_mask[TILE];
  for (int t = 0; t < TILE; ++t) {
    if (t < nt) {
      qv[t] = *reinterpret_cast<const float4*>(q + (size_t(q_base + t) * size_t(Hq) + size_t(h)) * size_t(D) + size_t(d0));
      q_depth[t] = int(d_depth[q_base + t]);
      q_mask[t] = d_anc_mask[q_base + t];
    } else {
      qv[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      q_depth[t] = 0;
      q_mask[t] = 0u;
    }
    mw[t] = -INFINITY;
    lw[t] = 0.0f;
    ow[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  for (int kp = c0 + sg; kp < c1; kp += 4) {
    const int slot = is_swa ? (kp % kv_cap) : kp;
    const size_t koff = (size_t(slot) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
    const float4 kv = kv_load4<KV_BF16>(kc, koff);
    const float4 vv = kv_load4<KV_BF16>(vc, koff);
    for (int t = 0; t < nt; ++t) {
      const int qpos = cur_pos + q_depth[t];
      const int t_min = (is_swa && qpos + 1 > swa) ? (qpos + 1 - swa) : 0;
      if (kp < t_min) continue;
      float sc = warp_sum(qv[t].x * kv.x + qv[t].y * kv.y + qv[t].z * kv.z + qv[t].w * kv.w) * scale;
      const int dist = qpos - kp;
      if (dist < rel_extent) sc += rb[(size_t(q_base + t) * size_t(Hq) + size_t(h)) * size_t(rel_extent) + size_t(dist)];
      if (lw[t] == 0.0f) {
        mw[t] = sc; lw[t] = 1.0f; ow[t] = vv;
      } else {
        const float nm = fmaxf(mw[t], sc);
        const float e1 = expf(mw[t] - nm), e2 = expf(sc - nm);
        ow[t] = make_float4(ow[t].x * e1 + vv.x * e2, ow[t].y * e1 + vv.y * e2,
                            ow[t].z * e1 + vv.z * e2, ow[t].w * e1 + vv.w * e2);
        lw[t] = lw[t] * e1 + e2;
        mw[t] = nm;
      }
    }
  }

  if (cid == 0) {
    for (int a = sg; a < M; a += 4) {
      const size_t skoff = (size_t(a) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
      const float4 kv = *reinterpret_cast<const float4*>(stage_k + skoff);
      const float4 vv = *reinterpret_cast<const float4*>(stage_v + skoff);
      const int adepth = int(d_depth[a]);
      for (int t = 0; t < nt; ++t) {
        if (!((q_mask[t] >> a) & 1u)) continue;
        const int dist = q_depth[t] - adepth;
        if (dist < 0 || (is_swa && dist >= swa)) continue;
        float sc = warp_sum(qv[t].x * kv.x + qv[t].y * kv.y + qv[t].z * kv.z + qv[t].w * kv.w) * scale;
        if (dist < rel_extent) sc += rb[(size_t(q_base + t) * size_t(Hq) + size_t(h)) * size_t(rel_extent) + size_t(dist)];
        if (lw[t] == 0.0f) {
          mw[t] = sc; lw[t] = 1.0f; ow[t] = vv;
        } else {
          const float nm = fmaxf(mw[t], sc);
          const float e1 = expf(mw[t] - nm), e2 = expf(sc - nm);
          ow[t] = make_float4(ow[t].x * e1 + vv.x * e2, ow[t].y * e1 + vv.y * e2,
                              ow[t].z * e1 + vv.z * e2, ow[t].w * e1 + vv.w * e2);
          lw[t] = lw[t] * e1 + e2;
          mw[t] = nm;
        }
      }
    }
  }

  for (int t = 0; t < nt; ++t) {
    if (lane == 0) { tgm[sg] = mw[t]; tgl[sg] = lw[t]; }
    *reinterpret_cast<float4*>(&tgo[sg][d0]) = ow[t];
    __syncthreads();
    if (sg == 0) {
      float mm = -INFINITY, ll = 0.0f;
      float4 oo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int s = 0; s < 4; ++s) {
        const float l2 = tgl[s];
        if (l2 == 0.0f) continue;
        const float m2 = tgm[s];
        const float4 o2 = *reinterpret_cast<const float4*>(&tgo[s][d0]);
        if (ll == 0.0f) { mm = m2; ll = l2; oo = o2; }
        else {
          const float nm = fmaxf(mm, m2);
          const float e1 = expf(mm - nm), e2 = expf(m2 - nm);
          oo = make_float4(oo.x * e1 + o2.x * e2, oo.y * e1 + o2.y * e2,
                           oo.z * e1 + o2.z * e2, oo.w * e1 + o2.w * e2);
          ll = ll * e1 + l2 * e2;
          mm = nm;
        }
      }
      const size_t out_idx = (size_t(q_base + t) * size_t(Hq) + size_t(h)) * size_t(nchunks) + size_t(cid);
      if (lane == 0) { pm[out_idx] = mm; pl[out_idx] = ll; }
      *reinterpret_cast<float4*>(po + out_idx * size_t(D) + size_t(d0)) = oo;
    }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(256) k_commit_sconv_all48(
    float* __restrict__ d_conv_base, const float* __restrict__ d_stage_sconv,
    const int32_t* __restrict__ d_path, int n_acc, int tree_cap, int KVD, int H) {
  const int v = blockIdx.x, strm = blockIdx.y, tid = threadIdx.x;
  const int stage_stride = 2 * KVD + 2 * H;
  const int visit_conv_floats = 3 * stage_stride;
  const int st_off[4] = {0, KVD, 2 * KVD, 2 * KVD + H};
  const int conv_off[4] = {0, 3 * KVD, 6 * KVD, 6 * KVD + 3 * H};
  const int st_len[4] = {KVD, KVD, H, H};

  const int C = st_len[strm];
  const int ch_off = st_off[strm];
  float* hist = d_conv_base + size_t(v) * size_t(visit_conv_floats) + size_t(conv_off[strm]);
  const float* st_v = d_stage_sconv + size_t(v) * size_t(tree_cap) * size_t(stage_stride);

  for (int c = tid; c < C; c += 256) {
    float h0 = hist[0 * size_t(C) + size_t(c)];
    float h1 = hist[1 * size_t(C) + size_t(c)];
    float h2 = hist[2 * size_t(C) + size_t(c)];
    for (int k = 0; k < n_acc; ++k) {
      const int node = d_path[k];
      const float cur = st_v[size_t(node) * size_t(stage_stride) + size_t(ch_off) + size_t(c)];
      h0 = h1;
      h1 = h2;
      h2 = cur;
    }
    hist[0 * size_t(C) + size_t(c)] = h0;
    hist[1 * size_t(C) + size_t(c)] = h1;
    hist[2 * size_t(C) + size_t(c)] = h2;
  }
}

template <bool KV_BF16>
__global__ void __launch_bounds__(128) k_commit_kv_all48(
    void* const* __restrict__ d_kc_ptrs, void* const* __restrict__ d_vc_ptrs,
    const int32_t* __restrict__ d_kv_cap, const uint8_t* __restrict__ d_kv_is_swa,
    const float* __restrict__ d_stage_k, const float* __restrict__ d_stage_v,
    const int32_t* __restrict__ d_path, int p0, int n_acc, int tree_cap, int Hkv, int D) {
  const int v = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  void* kc = d_kc_ptrs[v];
  void* vc = d_vc_ptrs[v];
  const int cap = d_kv_cap[v];
  const bool is_swa = (d_kv_is_swa[v] != 0);
  const int kv_dim = Hkv * D;
  for (int k = 0; k < n_acc; ++k) {
    const int node = d_path[k];
    const int p = p0 + k;
    const int slot = is_swa ? (p % cap) : p;
    const size_t sidx = (size_t(v) * size_t(tree_cap) + size_t(node)) * size_t(kv_dim) + size_t(h) * size_t(D) + size_t(d);
    const size_t didx = (size_t(slot) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(d);
    kv_store1<KV_BF16>(kc, didx, d_stage_k[sidx]);
    kv_store1<KV_BF16>(vc, didx, d_stage_v[sidx]);
  }
}

__global__ void __launch_bounds__(256) k_compact_taps_gather(
    const float* t0, const float* t1, const float* t2, const float* t3, float* scratch,
    const int32_t* d_path, int n_acc, int H) {
  const int k = blockIdx.x, ti = blockIdx.y, tid = threadIdx.x;
  const float* srcs[4] = {t0, t1, t2, t3};
  const int node = d_path[k];
  const float* src = srcs[ti] + size_t(node) * size_t(H);
  float* dst = scratch + (size_t(ti) * size_t(n_acc) + size_t(k)) * size_t(H);
  for (int d = tid; d < H; d += 256) dst[d] = src[d];
}

__global__ void __launch_bounds__(256) k_compact_taps_scatter(
    float* t0, float* t1, float* t2, float* t3, const float* scratch, int n_acc, int H) {
  const int k = blockIdx.x, ti = blockIdx.y, tid = threadIdx.x;
  float* dsts[4] = {t0, t1, t2, t3};
  const float* src = scratch + (size_t(ti) * size_t(n_acc) + size_t(k)) * size_t(H);
  float* dst = dsts[ti] + size_t(k) * size_t(H);
  for (int d = tid; d < H; d += 256) dst[d] = src[d];
}

}  // namespace

void launch_embed_norm(const int32_t* tokens, int tok_off, const __nv_bfloat16* embed,
                       const __nv_bfloat16* norm_w, float* out, int T, int H, float eps, cudaStream_t s,
                       const __nv_bfloat16* vis_feats, const int32_t* vis_idx, int32_t img_tok_id) {
  k_embed_norm<<<T, (H == 2304) ? 192 : 128, 0, s>>>(tokens, tok_off, embed, norm_w, out, H, eps,
                                                      vis_feats, vis_idx, img_tok_id);
}

void launch_rmsnorm(const float* x, const void* w, bool w_f32, float* out, int T, int H, float eps, cudaStream_t s) {
  k_rmsnorm<<<T, (H == 2304) ? 192 : 128, 0, s>>>(x, w, w_f32, out, H, eps);
}

void launch_rmsnorm_x3(const float* x0, const float* x1, const float* x2,
                       const void* w0, const void* w1, const void* w2, bool w_f32,
                       float* out0, float* out1, float* out2, int T, int H, float eps, cudaStream_t s) {
  Ptrs3 xs{x0, x1, x2};
  VoidPtrs3 ws{w0, w1, w2};
  FltMutPtrs3 outs{out0, out1, out2};
  k_rmsnorm_x3<<<dim3(T, 3, 1), (H == 2304) ? 192 : 128, 0, s>>>(xs, ws, w_f32, outs, H, eps);
}

void launch_attn_res(const float* const* states, int M, uint32_t norm_mask, const void* queries, bool q_f32, int Q,
                     float* out, int T, int H, float eps, float scale, cudaStream_t s,
                     const void* post_norm_w, bool post_w_f32) {
  Ptrs6 p{};
  for (int m = 0; m < 6; ++m) p.p[m] = m < M ? states[m] : states[0];
  if (H == 2304 && (Q == 1 || !post_norm_w)) {
    k_attn_res_q1_h2304<<<dim3(T, Q, 1), 192, 0, s>>>(p, M, norm_mask, queries, q_f32, out, T, eps, scale,
                                                      post_norm_w, post_w_f32);
  } else {
    k_attn_res<<<T, (H == 2304) ? 192 : 128, 0, s>>>(p, M, norm_mask, queries, q_f32, Q, out, T, H, eps, scale,
                                                     post_norm_w, post_w_f32);
  }
}

void launch_msg_gate(cublasHandle_t cublas, const float* h, const float* m, const __nv_bfloat16* down_w,
                     const __nv_bfloat16* forget_w, const __nv_bfloat16* gain_w,
                     float* nhm_f32, __nv_bfloat16* nhm_bf16, float* g_f32, __nv_bfloat16* g_bf16,
                     float* df_f32, float* dg_f32, float* out, int T, int H, int R, float eps, cudaStream_t s) {
  if (T <= 32 || !cublas || !nhm_bf16 || !g_bf16 || !df_f32 || !dg_f32) {
    k_msg_prep<<<T, 128, 0, s>>>(h, m, nhm_f32, nullptr, H, eps);
    if (T >= 2 && T <= 12 && (H & 7) == 0) {
      switch (T) {
        case 2: k_msg_down_tm<2><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 3: k_msg_down_tm<3><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 4: k_msg_down_tm<4><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 5: k_msg_down_tm<5><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 6: k_msg_down_tm<6><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 7: k_msg_down_tm<7><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 8: k_msg_down_tm<8><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 9: k_msg_down_tm<9><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 10: k_msg_down_tm<10><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 11: k_msg_down_tm<11><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
        case 12: k_msg_down_tm<12><<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, H, R); break;
      }
    } else {
      k_msg_down<<<R, 128, 0, s>>>(nhm_f32, down_w, g_f32, T, H, R);
    }
    const int up_bx = (R == 64) ? ((H + 15) / 16) : ((H + 3) / 4);
    k_msg_up_blend<<<dim3(up_bx, T, 1), 128, 0, s>>>(h, m, g_f32, forget_w, gain_w, out, H, R);
    return;
  }
  k_msg_prep<<<T, 256, 0, s>>>(h, m, nullptr, nhm_bf16, H, eps);
  cublasSetStream(cublas, s);
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasGemmEx(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, 2 * H, &alpha, down_w, CUDA_R_16BF, 2 * H,
      nhm_bf16, CUDA_R_16BF, 2 * H, &beta, g_f32, CUDA_R_32F, R, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in msg_gate down");
  const int n_g = T * R;
  k_f32_to_bf16<<<(n_g + 255) / 256, 256, 0, s>>>(g_f32, g_bf16, n_g);
  st = cublasGemmEx(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, H, T, R, &alpha, forget_w, CUDA_R_16BF, R,
      g_bf16, CUDA_R_16BF, R, &beta, df_f32, CUDA_R_32F, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in msg_gate forget");
  st = cublasGemmEx(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, H, T, R, &alpha, gain_w, CUDA_R_16BF, R,
      g_bf16, CUDA_R_16BF, R, &beta, dg_f32, CUDA_R_32F, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in msg_gate gain");
  const int n_out = T * H;
  k_msg_blend<<<((n_out + 3) / 4 + 255) / 256, 256, 0, s>>>(h, m, df_f32, dg_f32, out, n_out);
}

void launch_msg_gate_x3(const MsgGateItem* items, float* nhm3_f32, float* g3_f32,
                        int T, int H, int R, float eps, cudaStream_t s) {
  MsgItem3 it{};
  for (int i = 0; i < 3; ++i) {
    it.h[i] = items[i].h; it.m[i] = items[i].m;
    it.dw[i] = items[i].down_w; it.fw[i] = items[i].forget_w;
    it.gw[i] = items[i].gain_w; it.out[i] = items[i].out;
  }
  k_msg_prep_x3<<<dim3(T, 3, 1), 128, 0, s>>>(it, nhm3_f32, T, H, eps);
  if (T >= 2 && T <= 12 && (H & 7) == 0) {
    switch (T) {
      case 2: k_msg_down_x3_tm<2><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 3: k_msg_down_x3_tm<3><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 4: k_msg_down_x3_tm<4><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 5: k_msg_down_x3_tm<5><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 6: k_msg_down_x3_tm<6><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 7: k_msg_down_x3_tm<7><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 8: k_msg_down_x3_tm<8><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 9: k_msg_down_x3_tm<9><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 10: k_msg_down_x3_tm<10><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 11: k_msg_down_x3_tm<11><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
      case 12: k_msg_down_x3_tm<12><<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, H, R); break;
    }
  } else {
    k_msg_down_x3<<<dim3(R, 3, 1), 128, 0, s>>>(it, nhm3_f32, g3_f32, T, H, R);
  }
  const int up_bx = (R == 64) ? ((H + 15) / 16) : ((H + 3) / 4);
  k_msg_up_blend_x3<<<dim3(up_bx, T, 3), 128, 0, s>>>(it, g3_f32, T, H, R);
}

void launch_linear(cublasHandle_t cublas, const float* x, const void* w, bool w_f32, __nv_bfloat16* x_bf16_scratch,
                   float* y, int M, int N, int K, bool add, cudaStream_t s) {
  if (M == 1 && !w_f32 && ((K & 7) == 0)) {
    const auto* wb = static_cast<const __nv_bfloat16*>(w);
    if (N <= 384) {
      k_bf16_mv_cta_row<<<N, 128, 0, s>>>(x, wb, y, N, K, add);
    } else if (N <= 4096) {
      k_bf16_mv_m1_1row_4w<<<N, 128, 0, s>>>(x, wb, y, N, K, add);
    } else {
      k_bf16_mv_m1_2row<<<(N + 7) / 8, 128, 0, s>>>(x, wb, y, N, K, add);
    }
    return;
  }
  if (M >= 2 && M <= 12 && N <= 256 && !w_f32 && ((K & 7) == 0)) {
    const auto* wb = static_cast<const __nv_bfloat16*>(w);
    switch (M) {
      case 2: k_router_mw_tm<2><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 3: k_router_mw_tm<3><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 4: k_router_mw_tm<4><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 5: k_router_mw_tm<5><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 6: k_router_mw_tm<6><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 7: k_router_mw_tm<7><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 8: k_router_mw_tm<8><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 9: k_router_mw_tm<9><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 10: k_router_mw_tm<10><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 11: k_router_mw_tm<11><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
      case 12: k_router_mw_tm<12><<<N, 256, 0, s>>>(x, wb, y, N, K, add); break;
    }
    return;
  }
  if (M >= 2 && M <= 12 && !w_f32 && ((K & 7) == 0)) {
    const auto* wb = static_cast<const __nv_bfloat16*>(w);
    if (N >= 128 && (K & 255) == 0) {
      const int gxs = (N + 7) / 8;
      switch (M) {
        case 2: k_bf16_mv_smem<2><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 3: k_bf16_mv_smem<3><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 4: k_bf16_mv_smem<4><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 5: k_bf16_mv_smem<5><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 6: k_bf16_mv_smem<6><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 7: k_bf16_mv_smem<7><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 8: k_bf16_mv_smem<8><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 9: k_bf16_mv_smem<9><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 10: k_bf16_mv_smem<10><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 11: k_bf16_mv_smem<11><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 12: k_bf16_mv_smem<12><<<gxs, 128, 0, s>>>(x, wb, y, N, K, add); break;
      }
      return;
    }
    const int gx2 = (N + 7) / 8, gx1 = (N + 3) / 4;
    if (N >= 8192) {
      switch (M) {
        case 2: k_bf16_mv_ldg<2, 2, 4><<<gx2, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 3: k_bf16_mv_ldg<3, 2, 4><<<gx2, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 4: k_bf16_mv_ldg<4, 2, 4><<<gx2, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 5: k_bf16_mv_ldg<5, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 6: k_bf16_mv_ldg<6, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 7: k_bf16_mv_ldg<7, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 8: k_bf16_mv_ldg<8, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 9: k_bf16_mv_ldg<9, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 10: k_bf16_mv_ldg<10, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 11: k_bf16_mv_ldg<11, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 12: k_bf16_mv_ldg<12, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
      }
    } else {
      switch (M) {
        case 2: k_bf16_mv_ldg<2, 2, 4><<<gx2, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 3: k_bf16_mv_ldg<3, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 4: k_bf16_mv_ldg<4, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 5: k_bf16_mv_ldg<5, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 6: k_bf16_mv_ldg<6, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 7: k_bf16_mv_ldg<7, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 8: k_bf16_mv_ldg<8, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 9: k_bf16_mv_ldg<9, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 10: k_bf16_mv_ldg<10, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 11: k_bf16_mv_ldg<11, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
        case 12: k_bf16_mv_ldg<12, 1, 4><<<gx1, 128, 0, s>>>(x, wb, y, N, K, add); break;
      }
    }
    return;
  }
  const bool use_tc = !w_f32 && cublas && x_bf16_scratch && (M > 32);
  if (!use_tc) {
    for (int m0 = 0; m0 < M; m0 += 8) {
      const int mc = min(8, M - m0);
      k_bf16_mv<<<(N + 3) / 4, 128, 0, s>>>(x + size_t(m0) * size_t(K), w, w_f32,
                                             y + size_t(m0) * size_t(N), mc, N, K, add);
    }
    return;
  }
  const int n_elem = M * K;
  k_f32_to_bf16<<<(n_elem + 255) / 256, 256, 0, s>>>(x, x_bf16_scratch, n_elem);
  cublasSetStream(cublas, s);
  const float alpha = 1.0f, beta = add ? 1.0f : 0.0f;
  const cublasStatus_t st = cublasGemmEx(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16BF, K, x_bf16_scratch, CUDA_R_16BF, K,
      &beta, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in ink2 launch_linear");
}

void launch_linear_multi(cublasHandle_t cublas, const float* x, const __nv_bfloat16* w,
                         __nv_bfloat16* x_bf16_scratch, float* tmp_y,
                         const MultiLinearSeg* segs, int n_segs, int M, int K, cudaStream_t s) {
  SegSpec6 spec{};
  spec.n_segs = n_segs;
  int off = 0;
  float** d_arr[6] = {&spec.d0, &spec.d1, &spec.d2, &spec.d3, &spec.d4, &spec.d5};
  int* s_arr[6] = {&spec.s0, &spec.s1, &spec.s2, &spec.s3, &spec.s4, &spec.s5};
  int* r_arr[6] = {&spec.r0, &spec.r1, &spec.r2, &spec.r3, &spec.r4, &spec.r5};
  for (int i = 0; i < n_segs; ++i) {
    *d_arr[i] = segs[i].dst;
    *s_arr[i] = off;
    *r_arr[i] = segs[i].rows;
    off += segs[i].rows;
  }
  spec.total_rows = off;
  if (M == 1 && (K & 7) == 0) {
    k_bf16_mv_multi_m1_1row_4w<<<off, 128, 0, s>>>(x, w, spec, K);
    return;
  }
  if (M >= 2 && M <= 12 && ((K & 7) == 0)) {
    if (off >= 128 && (K & 255) == 0) {
      const int gxs = (off + 7) / 8;
      switch (M) {
        case 2: k_bf16_mv_smem_multi<2><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 3: k_bf16_mv_smem_multi<3><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 4: k_bf16_mv_smem_multi<4><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 5: k_bf16_mv_smem_multi<5><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 6: k_bf16_mv_smem_multi<6><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 7: k_bf16_mv_smem_multi<7><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 8: k_bf16_mv_smem_multi<8><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 9: k_bf16_mv_smem_multi<9><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 10: k_bf16_mv_smem_multi<10><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 11: k_bf16_mv_smem_multi<11><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
        case 12: k_bf16_mv_smem_multi<12><<<gxs, 128, 0, s>>>(x, w, spec, K); break;
      }
      return;
    }
    const int gx2 = (off + 7) / 8, gx1 = (off + 3) / 4;
    switch (M) {
      case 2: k_bf16_mv_multi_ldg<2, 2, 4><<<gx2, 128, 0, s>>>(x, w, spec, K); break;
      case 3: k_bf16_mv_multi_ldg<3, 2, 4><<<gx2, 128, 0, s>>>(x, w, spec, K); break;
      case 4: k_bf16_mv_multi_ldg<4, 2, 4><<<gx2, 128, 0, s>>>(x, w, spec, K); break;
      case 5: k_bf16_mv_multi_ldg<5, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
      case 6: k_bf16_mv_multi_ldg<6, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
      case 7: k_bf16_mv_multi_ldg<7, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
      case 8: k_bf16_mv_multi_ldg<8, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
      case 9: k_bf16_mv_multi_ldg<9, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
      case 10: k_bf16_mv_multi_ldg<10, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
      case 11: k_bf16_mv_multi_ldg<11, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
      case 12: k_bf16_mv_multi_ldg<12, 1, 4><<<gx1, 128, 0, s>>>(x, w, spec, K); break;
    }
    return;
  }
  const bool use_tc = cublas && x_bf16_scratch && tmp_y && (M > 32);
  if (!use_tc) {
    for (int m0 = 0; m0 < M; m0 += 8) {
      const int mc = min(8, M - m0);
      SegSpec6 sub = spec;
      if (n_segs > 0) sub.d0 += size_t(m0) * size_t(sub.r0);
      if (n_segs > 1) sub.d1 += size_t(m0) * size_t(sub.r1);
      if (n_segs > 2) sub.d2 += size_t(m0) * size_t(sub.r2);
      if (n_segs > 3) sub.d3 += size_t(m0) * size_t(sub.r3);
      if (n_segs > 4) sub.d4 += size_t(m0) * size_t(sub.r4);
      if (n_segs > 5) sub.d5 += size_t(m0) * size_t(sub.r5);
      k_bf16_mv_multi<<<(off + 3) / 4, 128, 0, s>>>(x + size_t(m0) * size_t(K), w, sub, mc, K);
    }
    return;
  }
  const int n_elem = M * K;
  k_f32_to_bf16<<<(n_elem + 255) / 256, 256, 0, s>>>(x, x_bf16_scratch, n_elem);
  cublasSetStream(cublas, s);
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasGemmEx(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, off, M, K, &alpha, w, CUDA_R_16BF, K,
      x_bf16_scratch, CUDA_R_16BF, K, &beta, tmp_y, CUDA_R_32F, off, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in ink2 launch_linear_multi");
  const size_t total = ((off & 3) == 0) ? (size_t(M) * size_t(off >> 2)) : (size_t(M) * size_t(off));
  k_split_multi<<<int((total + 255) / 256), 256, 0, s>>>(tmp_y, spec, M);
}

void launch_quantize_q8_32(const __nv_bfloat16* w, int8_t* q, __half* sc, int N, int K, cudaStream_t s) {
  const int total_warps = N * (K >> 5), wpb = 8;
  k_quantize_q8_32<<<(total_warps + wpb - 1) / wpb, wpb * 32, 0, s>>>(w, q, sc, N, K);
}

void launch_linear_q8_32(const float* x, const int8_t* w_q, const __half* w_sc, float* y, int M, int N, int K,
                         cudaStream_t s) {
  const int gx2 = (N + 7) / 8, gx1 = (N + 3) / 4;
  if (M == 1 && N > 4096) {
    k_q8_32_mv_tm<1, 2><<<gx2, 128, 0, s>>>(x, w_q, w_sc, y, N, K);
    return;
  }
  for (int m0 = 0; m0 < M; m0 += 8) {
    const int mc = min(8, M - m0);
    const float* xm = x + size_t(m0) * size_t(K);
    float* ym = y + size_t(m0) * size_t(N);
    switch (mc) {
      case 1: k_q8_32_mv_tm<1, 1><<<gx1, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
      case 2: k_q8_32_mv_tm<2, 2><<<gx2, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
      case 3: k_q8_32_mv_tm<3, 2><<<gx2, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
      case 4: k_q8_32_mv_tm<4, 2><<<gx2, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
      case 5: k_q8_32_mv_tm<5, 1><<<gx1, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
      case 6: k_q8_32_mv_tm<6, 1><<<gx1, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
      case 7: k_q8_32_mv_tm<7, 1><<<gx1, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
      case 8: k_q8_32_mv_tm<8, 1><<<gx1, 128, 0, s>>>(xm, w_q, w_sc, ym, N, K); break;
    }
  }
}

void launch_sconv_static(const float* x, const void* w, bool w_f32, float* hist, float* out,
                         int T, int C, float scale, cudaStream_t s, const float* res_add, float* pf_scratch,
                         bool save_snap, float* snap, int snap_stride) {
  if (T > 32 && !save_snap && (x != out || pf_scratch != nullptr)) {
    const float* x_ro = x;
    if (x == out) {
      cudaMemcpyAsync(pf_scratch, x, size_t(T) * size_t(C) * sizeof(float), cudaMemcpyDeviceToDevice, s);
      x_ro = pf_scratch;
    }
    k_sconv_static_pf<<<dim3((C + 63) / 64, (T + 63) / 64, 1), 64, 0, s>>>(x_ro, w, w_f32, hist, out, T, C, scale, res_add);
    cudaMemcpyAsync(hist, x_ro + size_t(T - 3) * size_t(C), 3 * size_t(C) * sizeof(float), cudaMemcpyDeviceToDevice, s);
    return;
  }
  k_sconv_static<<<(C + 63) / 64, 64, 0, s>>>(x, w, w_f32, hist, out, T, C, scale, res_add, save_snap, snap, snap_stride);
}

void launch_moe_reduce8_sconv_static(const float* pair_out, const void* w, bool w_f32, float* hist, float* out,
                                     int T, int C, int K_top, float scale, const float* res_add,
                                     bool save_snap, float* snap, int snap_stride, cudaStream_t s) {
  k_moe_reduce8_sconv_static<<<(C + 63) / 64, 64, 0, s>>>(
      pair_out, w, w_f32, hist, out, T, C, K_top, scale, res_add, save_snap, snap, snap_stride);
}

void launch_sconv_static_kv(const float* xk, const void* wk, bool wk_f32, float* hk,
                            const float* xv, const void* wv, bool wv_f32, float* hv,
                            int T, int C, float scale, cudaStream_t s, float* pf_scratch,
                            bool save_snap, float* snap_k, float* snap_v, int snap_stride) {
  if (T > 32 && !save_snap && pf_scratch != nullptr) {
    const size_t bytes = size_t(T) * size_t(C) * sizeof(float);
    float* xk_ro = pf_scratch;
    float* xv_ro = pf_scratch + size_t(T) * size_t(C);
    cudaMemcpyAsync(xk_ro, xk, bytes, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(xv_ro, xv, bytes, cudaMemcpyDeviceToDevice, s);
    k_sconv_static_kv_pf<<<dim3((C + 63) / 64, (T + 63) / 64, 2), 64, 0, s>>>(
        xk_ro, wk, wk_f32, hk, const_cast<float*>(xk),
        xv_ro, wv, wv_f32, hv, const_cast<float*>(xv), T, C, scale);
    cudaMemcpyAsync(hk, xk_ro + size_t(T - 3) * size_t(C), 3 * size_t(C) * sizeof(float), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(hv, xv_ro + size_t(T - 3) * size_t(C), 3 * size_t(C) * sizeof(float), cudaMemcpyDeviceToDevice, s);
    return;
  }
  k_sconv_static_kv<<<dim3((C + 63) / 64, 2, 1), 64, 0, s>>>(
      xk, wk, wk_f32, hk, xv, wv, wv_f32, hv, T, C, scale, save_snap, snap_k, snap_v, snap_stride);
}

void launch_sconv_dynamic(const float* x, const float* dyn, const void* static_w, bool sw_f32, float* hist,
                          float* out, int T, int C, int n_heads, cudaStream_t s, float* pf_scratch,
                          bool save_snap, float* snap, int snap_stride) {
  if (T > 32 && !save_snap && (x != out || pf_scratch != nullptr)) {
    const float* x_ro = x;
    if (x == out) {
      cudaMemcpyAsync(pf_scratch, x, size_t(T) * size_t(C) * sizeof(float), cudaMemcpyDeviceToDevice, s);
      x_ro = pf_scratch;
    }
    k_sconv_dynamic_pf<<<dim3((C + 63) / 64, (T + 63) / 64, 1), 64, 0, s>>>(x_ro, dyn, static_w, sw_f32, hist, out, T, C, n_heads);
    cudaMemcpyAsync(hist, x_ro + size_t(T - 3) * size_t(C), 3 * size_t(C) * sizeof(float), cudaMemcpyDeviceToDevice, s);
    return;
  }
  k_sconv_dynamic<<<(C + 63) / 64, 64, 0, s>>>(x, dyn, static_w, sw_f32, hist, out, T, C, n_heads, save_snap, snap, snap_stride);
}

void launch_qkvr_prep(float* q, const float* k, const float* v, float* r,
                      const void* q_scale, const void* r_scale, bool sc_f32,
                      void* kc, void* vc, int T, int Hq, int Hkv, int D, int d_rel,
                      int pos0, int kv_cap, bool is_swa, float qk_clip, float r_clip,
                      float v_eps, float log_floor, float log_alpha,
                      const int32_t* d_pos_ptr, bool kv_bf16, cudaStream_t s) {
  dim3 grid(2 * Hq + 2 * Hkv, T, 1);
  if (kv_bf16) {
    k_qkvr_prep<true><<<grid, D, 0, s>>>(q, k, v, r, q_scale, r_scale, sc_f32, kc, vc, Hq, Hkv, D, d_rel,
                                         pos0, d_pos_ptr, kv_cap, is_swa, qk_clip, r_clip, v_eps, log_floor, log_alpha);
  } else {
    k_qkvr_prep<false><<<grid, D, 0, s>>>(q, k, v, r, q_scale, r_scale, sc_f32, kc, vc, Hq, Hkv, D, d_rel,
                                          pos0, d_pos_ptr, kv_cap, is_swa, qk_clip, r_clip, v_eps, log_floor, log_alpha);
  }
}

void launch_rel_bias(cublasHandle_t cublas, const float* r, const void* rel_proj, bool proj_f32,
                     __nv_bfloat16* r_bf16_scratch, float* rb, int T, int Hq, int d_rel, int extent, cudaStream_t s) {
  if (T > 32 && !proj_f32 && cublas && r_bf16_scratch) {
    const int n_elem = T * Hq * d_rel;
    k_f32_to_bf16<<<(n_elem + 255) / 256, 256, 0, s>>>(r, r_bf16_scratch, n_elem);
    cublasSetStream(cublas, s);
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, extent, T * Hq, d_rel, &alpha,
        rel_proj, CUDA_R_16BF, extent, r_bf16_scratch, CUDA_R_16BF, d_rel,
        &beta, rb, CUDA_R_32F, extent, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in ink2 launch_rel_bias");
    return;
  }
  dim3 grid((extent + 63) / 64, Hq, T);
  k_rel_bias<<<grid, 64, 0, s>>>(r, rel_proj, proj_f32, rb, Hq, d_rel, extent);
}

void launch_attn_decode(const float* q, const void* kc, const void* vc, const float* rb,
                        const float* sinks, const float* gate, float* part_m, float* part_l, float* part_o,
                        float* out, int pos, int kv_cap, int Hq, int Hkv, int D, int rel_extent,
                        bool is_swa, int swa, const int32_t* d_pos_ptr, bool kv_bf16, cudaStream_t s) {
  int nchunks, chunk;
  if (d_pos_ptr) {
    nchunks = is_swa ? 12 : 24;
    chunk = 0;
  } else {
    const int lk = is_swa ? min(pos + 1, swa) : (pos + 1);
    chunk = max(32, (lk + 63) / 64);
    chunk = (chunk + 3) / 4 * 4;
    nchunks = max(1, (lk + chunk - 1) / chunk);
  }
  const float scale = 1.0f / std::sqrt(float(D));
  if (kv_bf16) {
    k_attn_decode<true><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos, d_pos_ptr,
                                                             kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk,
                                                             nchunks, scale);
  } else {
    k_attn_decode<false><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos, d_pos_ptr,
                                                              kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk,
                                                              nchunks, scale);
  }
  k_attn_decode_reduce<<<Hq, 32, 0, s>>>(part_m, part_l, part_o, sinks, gate, out, D, nchunks);
}

void launch_attn_verify(const float* q, const void* kc, const void* vc, const float* rb,
                        const float* sinks, const float* gate, float* part_m, float* part_l, float* part_o,
                        float* out, int T, int pos0, int kv_cap, int Hq, int Hkv, int D, int rel_extent,
                        bool is_swa, int swa, const int32_t* d_pos_ptr, bool kv_bf16, cudaStream_t s) {
  const int nchunks = is_swa ? 8 : 16;
  const float scale = 1.0f / std::sqrt(float(D));
  auto disp = [&](auto tag) {
    constexpr bool BF = decltype(tag)::value;
    switch (T) {
      case 2: k_attn_verify_tm<2, BF><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos0, d_pos_ptr, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale); break;
      case 3: k_attn_verify_tm<3, BF><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos0, d_pos_ptr, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale); break;
      case 4: k_attn_verify_tm<4, BF><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos0, d_pos_ptr, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale); break;
      case 5: k_attn_verify_tm<5, BF><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos0, d_pos_ptr, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale); break;
      case 6: k_attn_verify_tm<6, BF><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos0, d_pos_ptr, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale); break;
      case 7: k_attn_verify_tm<7, BF><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos0, d_pos_ptr, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale); break;
      default: k_attn_verify_tm<8, BF><<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos0, d_pos_ptr, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale); break;
    }
  };
  if (kv_bf16) disp(std::true_type{});
  else disp(std::false_type{});
  k_attn_verify_reduce<<<dim3(Hq, T, 1), 32, 0, s>>>(part_m, part_l, part_o, sinks, gate, out, Hq, D, nchunks);
}

void launch_attn_prefill(const float* q, const void* kc, const void* vc, const float* rb,
                         const float* sinks, const float* gate, float* out, int T, int pos0, int kv_cap,
                         int Hq, int Hkv, int D, int rel_extent, bool is_swa, int swa, bool kv_bf16, cudaStream_t s) {
  const float scale = 1.0f / std::sqrt(float(D));
  if (T > 32 && Hq == 24 && Hkv == 8 && D == 128) {
    if (kv_bf16) {
      k_attn_prefill_tc<true><<<dim3((T + 15) / 16, Hkv, 1), 96, 0, s>>>(
          q, kc, vc, rb, sinks, gate, out, T, pos0, kv_cap, Hq, Hkv, rel_extent, is_swa, swa, scale);
    } else {
      k_attn_prefill_tc<false><<<dim3((T + 15) / 16, Hkv, 1), 96, 0, s>>>(
          q, kc, vc, rb, sinks, gate, out, T, pos0, kv_cap, Hq, Hkv, rel_extent, is_swa, swa, scale);
    }
    return;
  }
  if (kv_bf16) {
    k_attn_prefill<true><<<dim3(Hq, T, 1), 128, 0, s>>>(q, kc, vc, rb, sinks, gate, out, pos0, kv_cap,
                                                        Hq, Hkv, D, rel_extent, is_swa, swa, scale);
  } else {
    k_attn_prefill<false><<<dim3(Hq, T, 1), 128, 0, s>>>(q, kc, vc, rb, sinks, gate, out, pos0, kv_cap,
                                                         Hq, Hkv, D, rel_extent, is_swa, swa, scale);
  }
}

void launch_moe_route(const float* logits, const void* bias, bool bias_f32, uint32_t* eid, float* ew,
                      int T, int E, int K, float route_scale, cudaStream_t s) {
  k_moe_route<<<T, E, 0, s>>>(logits, bias, bias_f32, eid, ew, E, K, route_scale);
}

void launch_moe_route_bucket_small(const float* logits, const void* bias, bool bias_f32, uint32_t* eid, float* ew,
                                   const MoeGroupBufs* gbuf, int T, int E, int K, float route_scale, cudaStream_t s) {
  const int max_e = min(E, T * K);
  float* st_ew = reinterpret_cast<float*>(gbuf->pair_idx);
  int32_t* st_tok = gbuf->pair_idx + max_e;
  int32_t* st_pair = st_tok + max_e;
  const int nth = max(128, T * 32);
  k_moe_route_bucket_small<<<1, nth, 0, s>>>(
      logits, bias, bias_f32, eid, ew, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
      st_tok, st_pair, st_ew, T, K, E, route_scale);
}

void launch_moe_nvfp4(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc, const float* w13_outer,
                      const uint8_t* w2, const uint8_t* w2_sc, const float* w2_outer,
                      float* act_scratch, float* pair_out, float* out, int T, int H, int I, int K_top,
                      const MoeGroupBufs* gbuf, int n_experts, cudaStream_t s,
                      bool run_reduce8, bool pre_bucketed) {
  if (T > 32 && gbuf != nullptr && H == 2304 && I == 1152 && K_top == 8 && n_experts == 128) {
    if (T >= 768) dispatch_moe_tc<64, false>(x, eid, ew, w13, w13_sc, w13_outer, w2, w2_sc, w2_outer, act_scratch, pair_out, out, T, H, I, K_top, gbuf, n_experts, s);
    else if (T >= 256) dispatch_moe_tc<32, false>(x, eid, ew, w13, w13_sc, w13_outer, w2, w2_sc, w2_outer, act_scratch, pair_out, out, T, H, I, K_top, gbuf, n_experts, s);
    else dispatch_moe_tc<16, false>(x, eid, ew, w13, w13_sc, w13_outer, w2, w2_sc, w2_outer, act_scratch, pair_out, out, T, H, I, K_top, gbuf, n_experts, s);
    return;
  }
  if (T >= 2 && T <= 12 && gbuf != nullptr && H == 2304 && I == 1152 && K_top == 8 && n_experts == 128) {
    const int max_e = min(n_experts, T * K_top);
    float* st_ew = reinterpret_cast<float*>(gbuf->pair_idx);
    int32_t* st_tok = gbuf->pair_idx + max_e;
    int32_t* st_pair = st_tok + max_e;
    if (!pre_bucketed) {
      k_moe_bucket_small<<<1, 128, 0, s>>>(eid, ew, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
                                           st_tok, st_pair, st_ew, T, K_top, n_experts);
    }
    k_moe_nvfp4_w13_g_h2<4><<<dim3((I + 15) / 16, max_e, 1), 128, 0, s>>>(
        x, w13, w13_sc, w13_outer, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
        st_tok, st_pair, act_scratch, H, I);
    k_moe_nvfp4_w2_g_h2<6><<<dim3(((H >> 1) + 15) / 16, max_e, 1), 128, 0, s>>>(
        act_scratch, w2, w2_sc, w2_outer, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
        st_pair, st_ew, pair_out, H, I);
    if (run_reduce8) {
      k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, T, 1), 64, 0, s>>>(
          reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
    }
    return;
  }
  if (T == 1 && H == 2304 && I == 1152 && K_top == 8) {
    k_moe_nvfp4_w13_t1_h2<<<dim3((I + 15) / 16, K_top, 1), 128, 0, s>>>(
        x, eid, w13, w13_sc, w13_outer, act_scratch, H, I);
    k_moe_nvfp4_w2_t1_h2<<<dim3(((H >> 1) + 15) / 16, K_top, 1), 128, 0, s>>>(
        act_scratch, eid, ew, w2, w2_sc, w2_outer, pair_out, H, I);
    if (run_reduce8) {
      k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, 1, 1), 64, 0, s>>>(
          reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
    }
    return;
  }
  k_moe_nvfp4_w13_situ<<<dim3((I + 15) / 16, K_top, T), 256, 0, s>>>(x, eid, w13, w13_sc, w13_outer, act_scratch, H, I, K_top);
  k_moe_nvfp4_w2_par<<<dim3((H + 7) / 8, K_top, T), 256, 0, s>>>(act_scratch, eid, ew, w2, w2_sc, w2_outer, pair_out, H, I, K_top);
  if (run_reduce8) {
    k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, T, 1), 64, 0, s>>>(
        reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
  }
}

void launch_moe_mxfp8(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc,
                      const uint8_t* w2, const uint8_t* w2_sc,
                      float* act_scratch, float* pair_out, float* out, int T, int H, int I, int K_top,
                      const MoeGroupBufs* gbuf, int n_experts, cudaStream_t s,
                      bool run_reduce8, bool pre_bucketed) {
  if (T > 32 && gbuf != nullptr && H == 2304 && I == 1152 && K_top == 8 && n_experts == 128) {
    if (T >= 768) dispatch_moe_tc<64, true>(x, eid, ew, w13, w13_sc, nullptr, w2, w2_sc, nullptr, act_scratch, pair_out, out, T, H, I, K_top, gbuf, n_experts, s);
    else if (T >= 256) dispatch_moe_tc<32, true>(x, eid, ew, w13, w13_sc, nullptr, w2, w2_sc, nullptr, act_scratch, pair_out, out, T, H, I, K_top, gbuf, n_experts, s);
    else dispatch_moe_tc<16, true>(x, eid, ew, w13, w13_sc, nullptr, w2, w2_sc, nullptr, act_scratch, pair_out, out, T, H, I, K_top, gbuf, n_experts, s);
    return;
  }
  if (T >= 2 && T <= 12 && gbuf != nullptr && H == 2304 && I == 1152 && K_top == 8 && n_experts == 128) {
    const int max_e = min(n_experts, T * K_top);
    float* st_ew = reinterpret_cast<float*>(gbuf->pair_idx);
    int32_t* st_tok = gbuf->pair_idx + max_e;
    int32_t* st_pair = st_tok + max_e;
    if (!pre_bucketed) {
      k_moe_bucket_small<<<1, 128, 0, s>>>(eid, ew, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
                                           st_tok, st_pair, st_ew, T, K_top, n_experts);
    }
    k_moe_mxfp8_w13_g_h2<4><<<dim3((I + 15) / 16, max_e, 1), 128, 0, s>>>(
        x, w13, w13_sc, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
        st_tok, st_pair, act_scratch, H, I);
    k_moe_mxfp8_w2_g_h2<6><<<dim3(((H >> 1) + 15) / 16, max_e, 1), 128, 0, s>>>(
        act_scratch, w2, w2_sc, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
        st_pair, st_ew, pair_out, H, I);
    if (run_reduce8) {
      k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, T, 1), 64, 0, s>>>(
          reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
    }
    return;
  }
  if (T >= 2 && T <= 12 && gbuf != nullptr && K_top == 8 && n_experts == 128 && (H & 255) == 0 && (I & 127) == 0) {
    const int max_e = min(n_experts, T * K_top);
    float* st_ew = reinterpret_cast<float*>(gbuf->pair_idx);
    int32_t* st_tok = gbuf->pair_idx + max_e;
    int32_t* st_pair = st_tok + max_e;
    if (!pre_bucketed) {
      k_moe_bucket_small<<<1, 128, 0, s>>>(eid, ew, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
                                           st_tok, st_pair, st_ew, T, K_top, n_experts);
    }
    k_moe_mxfp8_w13_g<<<dim3((I + 7) / 8, max_e, 1), 64, 0, s>>>(
        x, w13, w13_sc, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
        st_tok, st_pair, act_scratch, H, I);
    k_moe_mxfp8_w2_g<<<dim3((H + 15) / 16, max_e, 1), 64, 0, s>>>(
        act_scratch, w2, w2_sc, gbuf->tile_exp, gbuf->tile_m0, gbuf->tile_cnt, gbuf->num_tiles,
        st_pair, st_ew, pair_out, H, I);
    if (run_reduce8) {
      k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, T, 1), 64, 0, s>>>(
          reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
    }
    return;
  }
  if (T == 1 && H == 2304 && I == 1152 && K_top == 8) {
    k_moe_mxfp8_w13_t1_h2<<<dim3((I + 15) / 16, K_top, 1), 128, 0, s>>>(
        x, eid, w13, w13_sc, act_scratch, H, I);
    k_moe_mxfp8_w2_t1_h2<<<dim3(((H >> 1) + 15) / 16, K_top, 1), 128, 0, s>>>(
        act_scratch, eid, ew, w2, w2_sc, pair_out, H, I);
    if (run_reduce8) {
      k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, 1, 1), 64, 0, s>>>(
          reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
    }
    return;
  }
  k_moe_mxfp8_w13_situ<<<dim3((I + 15) / 16, K_top, T), 256, 0, s>>>(x, eid, w13, w13_sc, act_scratch, H, I, K_top);
  if (run_reduce8 && K_top == 8 && (H & 3) == 0 && (I % 32) == 0) {
    k_moe_mxfp8_w2_fused8<<<dim3(H / 4, T, 1), 256, 0, s>>>(act_scratch, eid, ew, w2, w2_sc, out, H, I);
    return;
  }
  k_moe_mxfp8_w2_par<<<dim3((H + 7) / 8, K_top, T), 256, 0, s>>>(act_scratch, eid, ew, w2, w2_sc, pair_out, H, I, K_top);
  if (run_reduce8) {
    k_moe_reduce8<<<dim3(((H >> 2) + 63) / 64, T, 1), 64, 0, s>>>(
        reinterpret_cast<const float4*>(pair_out), reinterpret_cast<float4*>(out), H, K_top);
  }
}

void launch_add(const float* src, float* dst, int n, cudaStream_t s) {
  k_add<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
}

void launch_add3(const float* a, const float* b, float* dst, int n, cudaStream_t s) {
  const int n4 = n >> 2;
  const int work = n4 + (n & 3);
  k_add3<<<(work + 255) / 256, 256, 0, s>>>(a, b, dst, n);
}

void launch_softcap(float* x, int n, float cap, cudaStream_t s) {
  k_softcap<<<(n + 255) / 256, 256, 0, s>>>(x, n, cap, 1.0f / cap);
}

void launch_suppress_mask(float* logits, int V, const uint8_t* mask_bits, cudaStream_t s) {
  k_suppress_mask<<<(V + 255) / 256, 256, 0, s>>>(logits, V, mask_bits);
}

void launch_suppress_mask_rows(float* logits, int R, int V, const uint8_t* mask_bits, cudaStream_t s) {
  k_suppress_mask_rows<<<dim3((V + 255) / 256, R, 1), 256, 0, s>>>(logits, V, mask_bits);
}

void launch_argmax_one(const float* logits, int V, int32_t* d_tok, float* blk_v, int32_t* blk_i, cudaStream_t s) {
  constexpr int NB = 256;
  k_argmax_p1<<<NB, 256, 0, s>>>(logits, V, blk_v, blk_i);
  k_argmax_p2<<<1, 256, 0, s>>>(blk_v, blk_i, NB, d_tok);
}

void launch_argmax_rows(const float* logits, int R, int V, int32_t* d_toks, float* blk_v, int32_t* blk_i, cudaStream_t s) {
  constexpr int NB = 256;
  k_argmax_p1<<<dim3(NB, R, 1), 256, 0, s>>>(logits, V, blk_v, blk_i);
  k_argmax_p2<<<dim3(1, R, 1), 256, 0, s>>>(blk_v, blk_i, NB, d_toks);
}

void launch_mtp_fuse_norm(const float* t0, const float* t1, const float* t2, const float* t3, int row_off,
                          float w0, float w1, float w2, float w3,
                          const void* norm_w, bool norm_f32, float* out, int T, int H, float eps, cudaStream_t s) {
  k_mtp_fuse_norm<<<T, (H == 2304) ? 192 : 128, 0, s>>>(t0, t1, t2, t3, row_off, w0, w1, w2, w3, norm_w, norm_f32, out, H, eps);
}

void launch_mtp_prep_block(const int32_t* d_anchor_tok, const __nv_bfloat16* embed,
                           const __nv_bfloat16* embed_norm, const __nv_bfloat16* mask_embed,
                           float* out, int B, int H, float eps, cudaStream_t s) {
  k_mtp_prep_block<<<B, (H == 2304) ? 192 : 128, 0, s>>>(d_anchor_tok, embed, embed_norm, mask_embed, out, H, eps);
}

void launch_mtp_qk_rope_cache(float* q, float* k, const float* v, const __nv_bfloat16* q_norm,
                               const __nv_bfloat16* k_norm, void* kc, void* vc,
                               int T, int Hq, int Hkv, int D, int pos0, const int32_t* d_pos_ptr,
                               int kv_cap, bool is_swa, bool draft_mode, float eps, float theta,
                               bool kv_bf16, cudaStream_t s) {
  dim3 grid(draft_mode ? (Hq + Hkv) : Hkv, T, 1);
  if (kv_bf16) {
    k_mtp_qk_rope_cache<true><<<grid, D, 0, s>>>(q, k, v, q_norm, k_norm, kc, vc, Hq, Hkv, D, pos0, d_pos_ptr,
                                                 kv_cap, is_swa, draft_mode, eps, theta);
  } else {
    k_mtp_qk_rope_cache<false><<<grid, D, 0, s>>>(q, k, v, q_norm, k_norm, kc, vc, Hq, Hkv, D, pos0, d_pos_ptr,
                                                  kv_cap, is_swa, draft_mode, eps, theta);
  }
}

void launch_mtp_attn(const float* q, const void* kc, const void* vc, const float* k_blk, const float* v_blk,
                     float* out, int B, int P, const int32_t* d_pos_ptr, int kv_cap, int Hq, int Hkv, int D,
                     bool is_swa, int swa, bool kv_bf16, cudaStream_t s) {
  const float scale = 1.0f / std::sqrt(float(D));
  if (kv_bf16) {
    k_mtp_attn<true><<<dim3(Hq, B, 1), 256, 0, s>>>(q, kc, vc, k_blk, v_blk, out, B, P, d_pos_ptr, kv_cap,
                                                    Hq, Hkv, D, is_swa, swa, scale);
  } else {
    k_mtp_attn<false><<<dim3(Hq, B, 1), 256, 0, s>>>(q, kc, vc, k_blk, v_blk, out, B, P, d_pos_ptr, kv_cap,
                                                     Hq, Hkv, D, is_swa, swa, scale);
  }
}

void launch_mtp_gate_up_swiglu(const float* x, const __nv_bfloat16* gate_up_w, float* out,
                               int M, int F, int H, cudaStream_t s) {
  const int gx = (F + 3) / 4;
  switch (M) {
    case 1: k_mtp_gate_up_swiglu_tm<1><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
    case 2: k_mtp_gate_up_swiglu_tm<2><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
    case 3: k_mtp_gate_up_swiglu_tm<3><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
    case 4: k_mtp_gate_up_swiglu_tm<4><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
    case 5: k_mtp_gate_up_swiglu_tm<5><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
    case 6: k_mtp_gate_up_swiglu_tm<6><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
    case 7: k_mtp_gate_up_swiglu_tm<7><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
    default: k_mtp_gate_up_swiglu_tm<8><<<gx, 128, 0, s>>>(x, gate_up_w, out, F, H); break;
  }
}

void launch_mtp_res_add_bf16(const float* res, const float* upd, float* out, int n, cudaStream_t s) {
  const int n4 = n >> 2;
  const int work = n4 + (n & 3);
  k_mtp_res_add_bf16<<<(work + 255) / 256, 256, 0, s>>>(res, upd, out, n);
}

void launch_mtp_rnn_prep(const float* state, const __nv_bfloat16* markov_w1, const int32_t* d_tok_id,
                         const float* hidden, const float* rec_feat, float* rnn_in,
                         int rank, int H, bool first_step, cudaStream_t s) {
  const int tot = 2 * rank + 2 * H;
  k_mtp_rnn_prep<<<(tot + 255) / 256, 256, 0, s>>>(state, markov_w1, d_tok_id, hidden, rec_feat,
                                                    rnn_in, rank, H, first_step);
}

void launch_mtp_rnn_update(const float* gates, float* state, float* latent, int rank, bool first_step, cudaStream_t s) {
  k_mtp_rnn_update<<<1, 256, 0, s>>>(gates, state, latent, rank, first_step);
}

void launch_linear_bf16_bias(const float* x, const __nv_bfloat16* w, const __nv_bfloat16* bias,
                             float* y, int N, int K, cudaStream_t s) {
  k_linear_bf16_bias<<<(N + 3) / 4, 128, 0, s>>>(x, w, bias, y, N, K);
}

void launch_mtp_fold_add_bf16(const float* x, const __nv_bfloat16* fold_w, const float* fold_b,
                              float* y, int V, int R, float softcap, cudaStream_t s) {
  k_mtp_fold_add_bf16<<<(V + 3) / 4, 128, 0, s>>>(x, fold_w, fold_b, y, V, R, softcap);
}

void launch_mtp_fold_add_q8(const float* x, const int8_t* fold_q, const __half* fold_s, const float* fold_b,
                            float* y, int V, int R, float softcap, cudaStream_t s) {
  k_mtp_fold_add_q8<<<(V + 15) / 16, 128, 0, s>>>(x, fold_q, fold_s, fold_b, y, V, R, softcap);
}

void launch_f32_to_bf16_buf(const float* src, __nv_bfloat16* dst, int n, cudaStream_t s) {
  k_f32_to_bf16<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
}

void launch_build_unembed_screen_bf16(const __nv_bfloat16* W, int8_t* q8, float* sc, float* eps, int V, int H,
                                      cudaStream_t s) {
  k_build_unembed_screen_bf16<<<(V + 7) / 8, 256, 0, s>>>(W, q8, sc, eps, V, H);
}

void launch_screen_argmax_bf16(const __nv_bfloat16* W_bf16, const int8_t* q8, const float* sc, const float* eps,
                               const float* h, int M, int V, int H, const uint8_t* suppress, float* d_hnorm,
                               int* d_max_lb, float* d_dot_hat, int32_t* d_cand_ids, int32_t* d_cand_cnt,
                               int max_cand, float softcap, int32_t* d_out_argmax, cudaStream_t s) {
  const int g1 = (V + 7) / 8;
  for (int m0 = 0; m0 < M; m0 += 8) {
    const int mc = min(8, M - m0);
    const float* hm = h + size_t(m0) * size_t(H);
    k_screen_row_l2norm<<<mc, 256, 0, s>>>(hm, d_hnorm, d_max_lb, d_cand_cnt, mc, H);
    if (mc == 1) {
      k_screen_pass1_multiM<1, 2304><<<g1, 256, 0, s>>>(q8, sc, eps, hm, d_hnorm, suppress, d_dot_hat, d_max_lb,
                                                        mc, V, H);
    } else if (mc <= 4) {
      k_screen_pass1_multiM<4, 1152><<<g1, 256, 0, s>>>(q8, sc, eps, hm, d_hnorm, suppress, d_dot_hat, d_max_lb,
                                                        mc, V, H);
    } else {
      k_screen_pass1_multiM<8, 576><<<g1, 256, 0, s>>>(q8, sc, eps, hm, d_hnorm, suppress, d_dot_hat, d_max_lb,
                                                       mc, V, H);
    }
    k_screen_pass2<<<dim3((V + 255) / 256, mc, 1), 256, 0, s>>>(eps, d_hnorm, d_dot_hat, d_max_lb, suppress,
                                                                d_cand_ids, d_cand_cnt, max_cand, mc, V);
    k_screen_eval_bf16<<<mc, 256, 0, s>>>(W_bf16, hm, d_cand_ids, d_cand_cnt, suppress, max_cand, softcap,
                                          d_out_argmax + m0, mc, V, H);
  }
}

void launch_screen_dot_multi(const int8_t* q8, const float* sc, const float* h, float* out_logits, int M, int V,
                             int H, cudaStream_t s) {
  const int g1 = (V + 7) / 8;
  for (int m0 = 0; m0 < M; m0 += 8) {
    const int mc = min(8, M - m0);
    const float* hm = h + size_t(m0) * size_t(H);
    float* ym = out_logits + size_t(m0) * size_t(V);
    if (mc == 1) {
      k_screen_pass1_multiM<1, 2304><<<g1, 256, 0, s>>>(q8, sc, nullptr, hm, nullptr, nullptr, ym, nullptr, mc, V, H);
    } else if (mc <= 4) {
      k_screen_pass1_multiM<4, 1152><<<g1, 256, 0, s>>>(q8, sc, nullptr, hm, nullptr, nullptr, ym, nullptr, mc, V, H);
    } else {
      k_screen_pass1_multiM<8, 576><<<g1, 256, 0, s>>>(q8, sc, nullptr, hm, nullptr, nullptr, ym, nullptr, mc, V, H);
    }
  }
}

void launch_dspark_screen_dp4a7(const int8_t* q8, const float* sc, const float* h7, int8_t* h_q8, float* h_sc,
                                const uint8_t* suppress, float* base_logits, int V, int H, cudaStream_t s) {
  k_dspark_quant_h_q8<<<7, 256, 0, s>>>(h7, h_q8, h_sc, H);
  k_dspark_screen_dp4a7<<<(V + 3) / 4, 128, 0, s>>>(q8, sc, h_q8, h_sc, suppress, base_logits, V, H);
}

void launch_dspark_top_cand(const float* base_logits, int V, int K_cand, int32_t* tmp_ids, float* tmp_vals,
                            int32_t* cand_ids, float* cand_base, cudaStream_t s) {
  k_dspark_top_cand_s1<<<dim3(128, 7, 1), 256, 0, s>>>(base_logits, V, tmp_ids, tmp_vals);
  k_dspark_top_cand_s2<<<7, 256, 0, s>>>(tmp_ids, tmp_vals, K_cand, cand_ids, cand_base);
}

void launch_dspark_beam_init(const int32_t* d_anchor_id, float* beam_state, int32_t* beam_prev_tok, float* beam_logp,
                             int32_t* beam_node, int32_t* pool_tok, int8_t* pool_parent, uint8_t* pool_depth,
                             float* pool_logp, int32_t* pool_cnt, cudaStream_t s) {
  k_dspark_beam_init<<<1, 256, 0, s>>>(d_anchor_id, beam_state, beam_prev_tok, beam_logp, beam_node,
                                       pool_tok, pool_parent, pool_depth, pool_logp, pool_cnt);
}

void launch_dspark_markov_prep_batch(const int32_t* prev_toks, const float* states, const float* hidden,
                                     const float* rec_feat, const __nv_bfloat16* markov_w1, float* joint_out,
                                     int nw, int R, int H, cudaStream_t s) {
  k_dspark_markov_prep_batch<<<nw, 256, 0, s>>>(prev_toks, states, hidden, rec_feat, markov_w1, joint_out, nw, R, H);
}

void launch_dspark_markov_update_batch(const float* proj, const __nv_bfloat16* rp_b, const float* state_in,
                                       float* state_next, float* lat_out, int nw, int R, cudaStream_t s) {
  k_dspark_markov_update_batch<<<nw, R, 0, s>>>(proj, rp_b, state_in, state_next, lat_out, nw, R);
}

void launch_dspark_add_bias_h_batch(float* corr, const float* ho_bias, const float* hidden, int nw, int H,
                                    cudaStream_t s) {
  k_dspark_add_bias_h_batch<<<dim3((H + 255) / 256, nw, 1), 256, 0, s>>>(corr, ho_bias, hidden, nw, H);
}

void launch_dspark_score_cand_dp4a_batch(const int8_t* q8, const float* sc, const float* corr,
                                         const int32_t* cand_ids, const uint8_t* suppress, int K_cand, int nw,
                                         float softcap, float* cand_scores, cudaStream_t s) {
  k_dspark_score_cand_dp4a_batch<<<(K_cand + 7) / 8, 256, 0, s>>>(
      q8, sc, corr, cand_ids, suppress, K_cand, nw, softcap, cand_scores, 2304);
}

void launch_dspark_beam_expand(const float* cand_scores, const int32_t* cand_ids, const float* state_next,
                               int K_cand, int nw, int beam_w, int depth, float* state_out,
                               int32_t* beam_prev_tok, float* beam_logp, int32_t* beam_node, int32_t* pool_tok,
                               int8_t* pool_parent, uint8_t* pool_depth, float* pool_logp, int32_t* pool_cnt,
                               cudaStream_t s) {
  k_dspark_beam_expand<<<1, 256, 0, s>>>(cand_scores, cand_ids, state_next, K_cand, nw, beam_w, depth,
                                         state_out, beam_prev_tok, beam_logp, beam_node, pool_tok,
                                         pool_parent, pool_depth, pool_logp, pool_cnt);
}

void launch_sconv_static_tree(const float* x, const void* w, bool w_f32, const float* hist, float* out,
                              float* stage_raw, int stage_stride, int stage_ch_off, const int8_t* p1,
                              const int8_t* p2, const int8_t* p3, int M, int C, float scale, const float* res_add,
                              cudaStream_t s) {
  k_sconv_static_tree<<<(C + 63) / 64, 64, 0, s>>>(x, w, w_f32, hist, out, stage_raw, stage_stride, stage_ch_off,
                                                    p1, p2, p3, M, C, scale, res_add);
}

void launch_moe_reduce8_sconv_static_tree(const float* pair_out, const void* w, bool w_f32, const float* hist,
                                          float* out, float* stage_raw, int stage_stride, int stage_ch_off,
                                          const int8_t* p1, const int8_t* p2, const int8_t* p3,
                                          int M, int C, int K_top, float scale, const float* res_add, cudaStream_t s) {
  k_moe_reduce8_sconv_static_tree<<<(C + 63) / 64, 64, 0, s>>>(
      pair_out, w, w_f32, hist, out, stage_raw, stage_stride, stage_ch_off, p1, p2, p3, M, C, K_top, scale, res_add);
}

void launch_sconv_static_kv_tree(float* xk, const void* wk, bool wk_f32, const float* hk,
                                 float* xv, const void* wv, bool wv_f32, const float* hv,
                                 float* stage_raw, int stage_stride, const int8_t* p1, const int8_t* p2,
                                 const int8_t* p3, int M, int C, float scale, cudaStream_t s) {
  k_sconv_static_kv_tree<<<dim3((C + 63) / 64, 2, 1), 64, 0, s>>>(
      xk, wk, wk_f32, hk, xv, wv, wv_f32, hv, stage_raw, stage_stride, p1, p2, p3, M, C, scale);
}

void launch_sconv_dynamic_tree(const float* x, const float* dyn, const void* static_w, bool sw_f32,
                               const float* hist, float* out, float* stage_raw, int stage_stride,
                               int stage_ch_off, const int8_t* p1, const int8_t* p2, const int8_t* p3, int M, int C,
                               int n_heads, cudaStream_t s) {
  k_sconv_dynamic_tree<<<(C + 63) / 64, 64, 0, s>>>(x, dyn, static_w, sw_f32, hist, out, stage_raw, stage_stride,
                                                     stage_ch_off, p1, p2, p3, M, C, n_heads);
}

void launch_qkvr_prep_tree(float* q, const float* k, const float* v, float* r,
                           const void* q_scale, const void* r_scale, bool sc_f32,
                           float* stage_k, float* stage_v, int M, int Hq, int Hkv, int D, int d_rel,
                           int pos0, const int32_t* d_pos_ptr, const uint8_t* d_depth,
                           float qk_clip, float r_clip, float v_eps, float log_floor, float log_alpha,
                           bool is_swa, cudaStream_t s) {
  k_qkvr_prep_tree<<<dim3(2 * Hq + 2 * Hkv, M, 1), D, 0, s>>>(
      q, k, v, r, q_scale, r_scale, sc_f32, stage_k, stage_v, Hq, Hkv, D, d_rel, pos0, d_pos_ptr, d_depth,
      qk_clip, r_clip, v_eps, log_floor, log_alpha, is_swa);
}

void launch_attn_tree(const float* q, const void* kc, const void* vc, const float* stage_k, const float* stage_v,
                      const float* rb, const float* sinks, const float* gate, float* part_m, float* part_l,
                      float* part_o, float* out, int M, int pos0, const int32_t* d_pos_ptr, const uint8_t* d_depth,
                      const uint32_t* d_anc_mask, int kv_cap, int Hq, int Hkv, int D, int rel_extent, bool is_swa,
                      int swa, bool kv_bf16, cudaStream_t s) {
  const int nchunks = is_swa ? 8 : 16;
  const float scale = 1.0f / std::sqrt(float(D));
  if (kv_bf16) {
    k_attn_tree_tm<6, true><<<dim3(Hq, nchunks, (M + 5) / 6), 128, 0, s>>>(
        q, kc, vc, stage_k, stage_v, rb, part_m, part_l, part_o, M, pos0, d_pos_ptr, d_depth, d_anc_mask,
        kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale);
  } else {
    k_attn_tree_tm<6, false><<<dim3(Hq, nchunks, (M + 5) / 6), 128, 0, s>>>(
        q, kc, vc, stage_k, stage_v, rb, part_m, part_l, part_o, M, pos0, d_pos_ptr, d_depth, d_anc_mask,
        kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, nchunks, scale);
  }
  k_attn_verify_reduce<<<dim3(Hq, M, 1), 32, 0, s>>>(part_m, part_l, part_o, sinks, gate, out, Hq, D, nchunks);
}

void launch_commit_sconv_all48(float* d_conv_base, const float* d_stage_sconv, const int32_t* d_path, int n_acc,
                               int nv, int tree_cap, int KVD, int H, cudaStream_t s) {
  k_commit_sconv_all48<<<dim3(nv, 4, 1), 256, 0, s>>>(d_conv_base, d_stage_sconv, d_path, n_acc, tree_cap, KVD, H);
}

void launch_commit_kv_all48(void* const* d_kc_ptrs, void* const* d_vc_ptrs, const int32_t* d_kv_cap,
                            const uint8_t* d_kv_is_swa, const float* d_stage_k, const float* d_stage_v,
                            const int32_t* d_path, int p0, int n_acc, int nv, int tree_cap, int Hkv, int D,
                            bool kv_bf16, cudaStream_t s) {
  if (kv_bf16) {
    k_commit_kv_all48<true><<<dim3(nv, Hkv, 1), D, 0, s>>>(
        d_kc_ptrs, d_vc_ptrs, d_kv_cap, d_kv_is_swa, d_stage_k, d_stage_v, d_path, p0, n_acc, tree_cap, Hkv, D);
  } else {
    k_commit_kv_all48<false><<<dim3(nv, Hkv, 1), D, 0, s>>>(
        d_kc_ptrs, d_vc_ptrs, d_kv_cap, d_kv_is_swa, d_stage_k, d_stage_v, d_path, p0, n_acc, tree_cap, Hkv, D);
  }
}

void launch_compact_taps(float* t0, float* t1, float* t2, float* t3, float* scratch, const int32_t* d_path,
                         int n_acc, int H, cudaStream_t s) {
  k_compact_taps_gather<<<dim3(n_acc, 4, 1), 256, 0, s>>>(t0, t1, t2, t3, scratch, d_path, n_acc, H);
  k_compact_taps_scatter<<<dim3(n_acc, 4, 1), 256, 0, s>>>(t0, t1, t2, t3, scratch, n_acc, H);
}

}  // namespace yskn::ink2
