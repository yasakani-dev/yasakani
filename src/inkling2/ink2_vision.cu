// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Image encoder of Inkling-2 Mini on the GPU (interface and rules: ink2_vision.h).
//
// Sequence order inside the encoder: index 0 is the class token, index 1 + 4 t + 2 my + mx is the native 20 x 20
// patch (2 (t / cols) + my, 2 (t % cols) + mx) of image token t (raster order over the rows x cols token grid). With
// this order the 2 x 2 merge after the last block is a plain view: rows 1 .. 4 N of the residual stream, read as
// [N, 4096], are already the merged vectors.
//
// Kernels (all launched on the stream given at create; encode allocates nothing):
//   k_patchify  RGB bytes -> padded, normalised native patches, BF16 [4 N, 1200] in (py, px, c) order
//   k_gemm      C = A W^T with BF16 tensor-core products (mma.sync m16n8k16, FP32 sums) and fused epilogues:
//               FP32 out; + bias with the 2-D rotary on q and k, BF16 out; x += gamma * (C + bias); exact GELU, BF16 out
//   k_embed     + bilinear position grid (32 x 32 table resampled to the native grid), class row
//   k_layernorm LayerNorm with weight and bias, FP32 statistics, FP32 or BF16 out
//   k_attn      bidirectional flash attention, 16 heads of 64, FP32 online softmax, BF16 P and V products
// Every reduction runs in a fixed order (no atomics, no split sums), so one image always gives the same bits.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#define INK2_BUILD  // part of the library (ink2_api.h: export, not import)
#include "ink2_vision.h"

typedef unsigned short u16;

namespace {

// The one supported model (vision_config of Inkling-2 Mini).
constexpr int kD = 1024, kHeads = 16, kHeadDim = 64, kFF = 4096, kLayers = 23, kPatch = 20, kPatchIn = 1200;
constexpr int kGrid = 32, kMergedDim = 4096, kOut = INK2V_OUT_DIM, kTile = INK2V_TILE, kQKV = 3 * kD;
constexpr int kPerLayer = 18, kGlobals = 8;
static_assert(kGlobals + kLayers * kPerLayer == INK2V_TENSORS, "tensor count");

enum {  // global tensors, then kPerLayer per layer in this order (q, k, v weights and biases adjacent: one QKV product)
  T_PATCH, T_POS, T_CLS, T_CLSPOS, T_LNPRE_W, T_LNPRE_B, T_PROJ_IN, T_PROJ_OUT
};
enum {
  L_QW, L_KW, L_VW, L_QB, L_KB, L_VB, L_OW, L_OB, L_LN1W, L_LN1B, L_LN2W, L_LN2B, L_LS1, L_LS2, L_FCW, L_FCB, L_PRW, L_PRB
};

void say(char* err, const char* fmt, ...) {
  if (!err) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, INK2V_ERR_LEN, fmt, ap);
  va_end(ap);
}

struct TensorInfo {
  std::string name;
  long long shape[2];
  int ndim;
  long long bytes, offset;
};

void build_table(std::vector<TensorInfo>& t) {
  auto add = [&](const std::string& n, long long a, long long b) {
    TensorInfo x;
    x.name = n;
    x.shape[0] = a;
    x.shape[1] = b;
    x.ndim = b ? 2 : 1;
    x.bytes = 2 * a * (b ? b : 1);
    x.offset = t.empty() ? 0 : t.back().offset + ((t.back().bytes + 255) / 256) * 256;
    t.push_back(x);
  };
  const std::string e = "model.visual.backbone.embeddings.";
  add(e + "patch_embedding.weight", kD, kPatchIn);
  add(e + "position_embedding", kGrid * kGrid, kD);
  add(e + "class_embedding", kD, 0);
  add(e + "cls_position_embedding", kD, 0);
  add("model.visual.backbone.ln_pre.weight", kD, 0);
  add("model.visual.backbone.ln_pre.bias", kD, 0);
  add("model.visual.proj_in.weight", kMergedDim, kMergedDim);
  add("model.visual.proj_out.weight", kOut, kMergedDim);
  for (int l = 0; l < kLayers; ++l) {
    const std::string p = "model.visual.backbone.layers." + std::to_string(l) + ".";
    add(p + "self_attn.q_proj.weight", kD, kD);
    add(p + "self_attn.k_proj.weight", kD, kD);
    add(p + "self_attn.v_proj.weight", kD, kD);
    add(p + "self_attn.q_proj.bias", kD, 0);
    add(p + "self_attn.k_proj.bias", kD, 0);
    add(p + "self_attn.v_proj.bias", kD, 0);
    add(p + "self_attn.out_proj.weight", kD, kD);
    add(p + "self_attn.out_proj.bias", kD, 0);
    add(p + "ln_1.weight", kD, 0);
    add(p + "ln_1.bias", kD, 0);
    add(p + "ln_2.weight", kD, 0);
    add(p + "ln_2.bias", kD, 0);
    add(p + "ls_1.gamma", kD, 0);
    add(p + "ls_2.gamma", kD, 0);
    add(p + "mlp.c_fc.weight", kFF, kD);
    add(p + "mlp.c_fc.bias", kFF, 0);
    add(p + "mlp.c_proj.weight", kD, kFF);
    add(p + "mlp.c_proj.bias", kD, 0);
  }
}

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ float bf2f(u16 h) { return __uint_as_float((unsigned)h << 16); }
__device__ __forceinline__ u16 f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ unsigned pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);  // lo in the low half
  return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}
__device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}
// 16-byte asynchronous copy global -> shared; with valid false the 16 bytes are zero-filled and nothing is read.
__device__ __forceinline__ void cp16(void* smem, const void* g, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(n));
#endif
}
__device__ __forceinline__ void cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Native patch (i, j) of sequence row s >= 1 (see the order at the top).
__device__ __forceinline__ void native_ij(int s, int cols, int* i, int* j) {
  const int p = s - 1, t = p >> 2, q = p & 3;
  *i = 2 * (t / cols) + (q >> 1);
  *j = 2 * (t % cols) + (q & 1);
}

// Normalised value of padded pixel (y, x), channel c: -1 (on the 0..255 scale) outside the image, then
// (p - 127.5) / 127.5 with an IEEE division.
__device__ __forceinline__ float pixel(const unsigned char* rgb, int w, int h, int y, int x, int c) {
  const float p = (y < h && x < w) ? (float)rgb[((size_t)y * w + x) * 3 + c] : -1.0f;
  return __fdiv_rn(__fsub_rn(p, 127.5f), 127.5f);
}

// ------------------------------------------------------------------------------------------------ K1
__global__ void __launch_bounds__(256) k_patchify(const unsigned char* rgb, int w, int h, int cols, u16* out) {
  int i, j;
  native_ij(blockIdx.x + 1, cols, &i, &j);
  u16* o = out + (size_t)blockIdx.x * kPatchIn;
  for (int e = threadIdx.x; e < kPatchIn; e += blockDim.x) {
    const int py = e / 60, px = (e % 60) / 3, c = e % 3;
    o[e] = f2bf(pixel(rgb, w, h, kPatch * i + py, kPatch * j + px, c));
  }
}

// pixel_values in the layout of the reference processor: [N, 40, 40, 3], token t in raster order.
__global__ void __launch_bounds__(256) k_pixels(const unsigned char* rgb, int w, int h, int cols, float* out) {
  const int t = blockIdx.x, tr = t / cols, tc = t % cols;
  float* o = out + (size_t)t * (kTile * kTile * 3);
  for (int e = threadIdx.x; e < kTile * kTile * 3; e += blockDim.x) {
    const int y = e / (kTile * 3), x = (e % (kTile * 3)) / 3, c = e % 3;
    o[e] = pixel(rgb, w, h, kTile * tr + y, kTile * tc + x, c);
  }
}

// ------------------------------------------------------------------------------------------------ K2
// Bilinear taps of one axis: align_corners false, source clamped to [0, 31], no antialiasing.
__device__ __forceinline__ void taps(int o, int n, int* f0, int* f1, float* w1) {
  float s = ((float)o + 0.5f) * (float)kGrid / (float)n - 0.5f;
  s = fminf(fmaxf(s, 0.0f), (float)(kGrid - 1));
  const int f = (int)floorf(s);
  *f0 = f;
  *f1 = f + 1 < kGrid ? f + 1 : kGrid - 1;
  *w1 = s - (float)f;
}

// x rows 1.. hold the patch projection; adds the resampled position table. Row 0 = class + class position.
__global__ void __launch_bounds__(256) k_embed(float* x, const u16* pos, const u16* cls, const u16* clspos, int rows,
                                                int cols) {
  const int s = blockIdx.x, d = threadIdx.x * 4;
  float4* xr = reinterpret_cast<float4*>(x + (size_t)s * kD + d);
  if (s == 0) {
    *xr = make_float4(bf2f(cls[d]) + bf2f(clspos[d]), bf2f(cls[d + 1]) + bf2f(clspos[d + 1]),
                      bf2f(cls[d + 2]) + bf2f(clspos[d + 2]), bf2f(cls[d + 3]) + bf2f(clspos[d + 3]));
    return;
  }
  int i, j, y0, y1, x0, x1;
  float wy, wx;
  native_ij(s, cols, &i, &j);
  taps(i, 2 * rows, &y0, &y1, &wy);
  taps(j, 2 * cols, &x0, &x1, &wx);
  const u16 *g00 = pos + (size_t)(y0 * kGrid + x0) * kD + d, *g01 = pos + (size_t)(y0 * kGrid + x1) * kD + d;
  const u16 *g10 = pos + (size_t)(y1 * kGrid + x0) * kD + d, *g11 = pos + (size_t)(y1 * kGrid + x1) * kD + d;
  float4 v = *xr;
  float* vv = &v.x;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float top = (1.0f - wx) * bf2f(g00[k]) + wx * bf2f(g01[k]);
    const float bot = (1.0f - wx) * bf2f(g10[k]) + wx * bf2f(g11[k]);
    vv[k] = vv[k] + ((1.0f - wy) * top + wy * bot);
  }
  *xr = v;
}

// ------------------------------------------------------------------------------------------------ K3
__device__ __forceinline__ float block_sum256(float v, float* sh) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  float t = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) t += sh[k];
  return t;
}

// One row of 1024 per block of 256 threads. outf may alias in.
__global__ void __launch_bounds__(256) k_layernorm(const float* in, const u16* w, const u16* b, float eps, float* outf,
                                                    u16* outh) {
  __shared__ float sh[8];
  const size_t r = blockIdx.x;
  const int d = threadIdx.x * 4;
  const float4 v = *reinterpret_cast<const float4*>(in + r * kD + d);
  const float mean = block_sum256((v.x + v.y) + (v.z + v.w), sh) / (float)kD;
  const float a0 = v.x - mean, a1 = v.y - mean, a2 = v.z - mean, a3 = v.w - mean;
  const float var = block_sum256((a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3), sh) / (float)kD;
  const float rs = 1.0f / sqrtf(var + eps);
  const float y0 = a0 * rs * bf2f(w[d]) + bf2f(b[d]), y1 = a1 * rs * bf2f(w[d + 1]) + bf2f(b[d + 1]);
  const float y2 = a2 * rs * bf2f(w[d + 2]) + bf2f(b[d + 2]), y3 = a3 * rs * bf2f(w[d + 3]) + bf2f(b[d + 3]);
  if (outf) {
    *reinterpret_cast<float4*>(outf + r * kD + d) = make_float4(y0, y1, y2, y3);
  } else {
    *reinterpret_cast<uint2*>(outh + r * kD + d) = make_uint2(pack2(y0, y1), pack2(y2, y3));
  }
}

__global__ void k_to_bf16(const float* in, u16* out, long long n2) {  // n2 = number of value pairs
  for (long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x; k < n2; k += (long long)gridDim.x * blockDim.x) {
    const float2 v = reinterpret_cast<const float2*>(in)[k];
    reinterpret_cast<unsigned*>(out)[k] = pack2(v.x, v.y);
  }
}

// ------------------------------------------------------------------------------------------------ G
enum { EP_F32 = 0, EP_QKV_ROPE = 1, EP_RESID = 2, EP_GELU = 3 };

struct GemmArgs {
  const u16* a;  // [M, K] row-major, row stride lda
  const u16* w;  // [N, K] row-major (a PyTorch Linear weight)
  int M, N, K, lda;
  const u16* bias;   // [N] or null
  const u16* gamma;  // [N] (EP_RESID)
  float* of;         // FP32 out (EP_F32) or the residual stream (EP_RESID)
  u16* oh;           // BF16 out (EP_QKV_ROPE, EP_GELU)
  int ldo;
  int cols;          // token grid columns (rotary positions)
  float rope[16];    // rotary frequencies theta^(-k / 16)
};

constexpr int GBM = 128, GBN = 128, GBK = 32, GLD = GBK + 8, GST = 2;  // 80-byte rows: conflict-free ldmatrix

// Block 128 x 128, 8 warps as 2 (rows) x 4 (columns), warp tile 64 x 32; two-stage cp.async pipeline over K in
// steps of 32 (K % 8 == 0; a partial last step is zero-filled).
template <int MODE>
__global__ void __launch_bounds__(256) k_gemm(GemmArgs p) {
  __shared__ __align__(16) u16 sm[GST * (GBM + GBN) * GLD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
  const int g = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
  const int ktiles = (p.K + GBK - 1) / GBK;

  auto load = [&](int slot, int kt) {
    u16* sa = sm + slot * (GBM + GBN) * GLD;
    u16* sb = sa + GBM * GLD;
    const int k0 = kt * GBK;
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int c = tid + it * 256, row = c >> 2, kc = (c & 3) * 8, gk = k0 + kc;
      const bool oka = m0 + row < p.M && gk < p.K;
      cp16(sa + row * GLD + kc, oka ? p.a + (size_t)(m0 + row) * p.lda + gk : p.a, oka);
      const bool okb = n0 + row < p.N && gk < p.K;
      cp16(sb + row * GLD + kc, okb ? p.w + (size_t)(n0 + row) * p.K + gk : p.w, okb);
    }
  };

  float acc[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.0f;

  load(0, 0);
  cp_commit();
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_wait<0>();
    __syncthreads();
    if (kt + 1 < ktiles) load((kt + 1) & 1, kt + 1);
    cp_commit();
    const u16* sa = sm + (kt & 1) * (GBM + GBN) * GLD;
    const u16* sb = sa + GBM * GLD;
#pragma unroll
    for (int kk = 0; kk < GBK / 16; ++kk) {
      unsigned af[4][4], bfr[4][2];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
        ldsm4(af[mi], sa + (wm * 64 + mi * 16 + (lane & 15)) * GLD + kk * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int nj = 0; nj < 2; ++nj) {
        unsigned r[4];
        ldsm4(r, sb + (wn * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * GLD + kk * 16 + ((lane >> 3) & 1) * 8);
        bfr[2 * nj][0] = r[0];
        bfr[2 * nj][1] = r[1];
        bfr[2 * nj + 1][0] = r[2];
        bfr[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) mma16816(acc[mi][ni], af[mi], bfr[ni][0], bfr[ni][1]);
    }
  }

  // Epilogue: thread holds columns (c, c + 1) of rows r and r + 8 of each 16 x 8 tile.
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c = n0 + wn * 32 + ni * 8 + 2 * tig;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = m0 + wm * 64 + mi * 16 + g + 8 * hh;
        if (r >= p.M || c >= p.N) continue;
        float v0 = acc[mi][ni][2 * hh], v1 = acc[mi][ni][2 * hh + 1];
        if (MODE != EP_F32 && p.bias) {
          v0 += bf2f(p.bias[c]);
          v1 += bf2f(p.bias[c + 1]);
        }
        if (MODE == EP_F32) {
          *reinterpret_cast<float2*>(p.of + (size_t)r * p.ldo + c) = make_float2(v0, v1);
        } else if (MODE == EP_RESID) {
          float2* xp = reinterpret_cast<float2*>(p.of + (size_t)r * p.ldo + c);
          float2 x = *xp;
          x.x = x.x + bf2f(p.gamma[c]) * v0;
          x.y = x.y + bf2f(p.gamma[c + 1]) * v1;
          *xp = x;
        } else if (MODE == EP_GELU) {
          *reinterpret_cast<unsigned*>(p.oh + (size_t)r * p.ldo + c) = pack2(gelu_erf(v0), gelu_erf(v1));
        } else {  // EP_QKV_ROPE: columns < 2048 are q and k; the pair (c, c + 1) is the interleaved rotary pair m
          if (c < 2 * kD) {
            float ph = 0.0f, pw = 0.0f;  // class token at (0, 0), native patch (i, j) at (i + 1, j + 1)
            if (r > 0) {
              int i, j;
              native_ij(r, p.cols, &i, &j);
              ph = (float)(i + 1);
              pw = (float)(j + 1);
            }
            const int m = (c & (kHeadDim - 1)) >> 1;  // angles 0..15 follow the column, 16..31 the row
            const float ang = (m < 16 ? pw : ph) * p.rope[m & 15];
            float sn, cs;
            sincosf(ang, &sn, &cs);
            const float y0 = v0 * cs - v1 * sn, y1 = v0 * sn + v1 * cs;
            v0 = y0;
            v1 = y1;
          }
          *reinterpret_cast<unsigned*>(p.oh + (size_t)r * p.ldo + c) = pack2(v0, v1);
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ K5
// qkv: BF16 [S, 3072] (q | k | v, head h at columns h * 64). out: BF16 [S, 1024]. Block = 4 warps x 16 queries of one
// head; keys stream in tiles of 64 (double-buffered cp.async), no mask besides the end of the sequence.
constexpr int AQ = 64, AKT = 64, ALD = kHeadDim + 8;  // 144-byte rows: conflict-free ldmatrix

__global__ void __launch_bounds__(128) k_attn(const u16* qkv, u16* out, int S) {
  __shared__ __align__(16) u16 sq[AQ * ALD];
  __shared__ __align__(16) u16 sk[2][AKT * ALD];
  __shared__ __align__(16) u16 sv[2][AKT * ALD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
  const int h = blockIdx.y, q0 = blockIdx.x * AQ;
  const size_t ld = kQKV;
  const float scale = 0.125f * 1.4426950408889634f;  // 64^-0.5, in log2 units for exp2f
  const float ninf = __int_as_float(0xff800000);

#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int c = tid + it * 128, row = c >> 3, dc = (c & 7) * 8;
    const bool ok = q0 + row < S;
    cp16(sq + row * ALD + dc, ok ? qkv + (size_t)(q0 + row) * ld + h * kHeadDim + dc : qkv, ok);
  }
  auto load_kv = [&](int slot, int kt) {
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int c = tid + it * 128, row = c >> 3, dc = (c & 7) * 8, key = kt * AKT + row;
      const bool ok = key < S;
      const u16* base = qkv + (size_t)key * ld + h * kHeadDim + dc;
      cp16(sk[slot] + row * ALD + dc, ok ? base + kD : qkv, ok);
      cp16(sv[slot] + row * ALD + dc, ok ? base + 2 * kD : qkv, ok);
    }
  };
  load_kv(0, 0);
  cp_commit();

  const int ntiles = (S + AKT - 1) / AKT;
  float o[8][4];
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k][0] = o[k][1] = o[k][2] = o[k][3] = 0.0f;
  float m[2] = {ninf, ninf}, l[2] = {0.0f, 0.0f};
  unsigned qf[4][4];

  for (int kt = 0; kt < ntiles; ++kt) {
    if (kt + 1 < ntiles) load_kv((kt + 1) & 1, kt + 1);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    if (kt == 0) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) ldsm4(qf[kk], sq + (warp * 16 + (lane & 15)) * ALD + kk * 16 + (lane >> 4) * 8);
    }
    const u16* ks = sk[kt & 1];
    const u16* vs = sv[kt & 1];
    float s[8][4];
#pragma unroll
    for (int k = 0; k < 8; ++k) s[k][0] = s[k][1] = s[k][2] = s[k][3] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nj = 0; nj < 4; ++nj) {
        unsigned r[4];
        ldsm4(r, ks + (nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * ALD + kk * 16 + ((lane >> 3) & 1) * 8);
        mma16816(s[2 * nj], qf[kk], r[0], r[1]);
        mma16816(s[2 * nj + 1], qf[kk], r[2], r[3]);
      }
    }
    float mx[2] = {m[0], m[1]};
    const int k0 = kt * AKT;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = k0 + nt * 8 + 2 * tig + (e & 1);
        const float v = key < S ? s[nt][e] * scale : ninf;
        s[nt][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
    }
    float corr[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
      corr[r] = exp2f(m[r] - mx[r]);  // 0 on the first tile (m = -inf); every tile has at least one valid key
      m[r] = mx[r];
      l[r] *= corr[r];
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      o[k][0] *= corr[0];
      o[k][1] *= corr[0];
      o[k][2] *= corr[1];
      o[k][3] *= corr[1];
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float pe = exp2f(s[nt][e] - m[e >> 1]);
        s[nt][e] = pe;
        l[e >> 1] += pe;
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {  // keys 16 kk .. 16 kk + 15: the score tiles 2 kk and 2 kk + 1 are the A operand
      unsigned a[4];
      a[0] = pack2(s[2 * kk][0], s[2 * kk][1]);
      a[1] = pack2(s[2 * kk][2], s[2 * kk][3]);
      a[2] = pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      a[3] = pack2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dj = 0; dj < 4; ++dj) {
        unsigned r[4];
        ldsm4t(r, vs + (kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * ALD + dj * 16 + (lane >> 4) * 8);
        mma16816(o[2 * dj], a, r[0], r[1]);
        mma16816(o[2 * dj + 1], a, r[2], r[3]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int q = q0 + warp * 16 + g + 8 * r;
    if (q >= S) continue;
    const float inv = 1.0f / l[r];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt)
      *reinterpret_cast<unsigned*>(out + (size_t)q * kD + h * kHeadDim + dt * 8 + 2 * tig) =
          pack2(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
  }
}

}  // namespace

// ================================================================================================ host side
struct Ink2Vis {
  Ink2VisionConfig cfg;
  cudaStream_t stream;
  std::vector<TensorInfo> table;
  std::unordered_map<std::string, int> index;
  std::vector<char> have;
  int n_have = 0;
  bool capture = false, finalized = false;
  float rope[16];
  // device memory
  unsigned char* weights = nullptr;
  long long weight_bytes = 0, scratch_bytes = 0;
  float* x = nullptr;        // residual stream FP32 [S, 1024]
  u16* lnb = nullptr;        // LayerNorm / attention output BF16 [S, 1024]
  u16* qkv = nullptr;        // BF16 [S, 3072]; after the blocks: proj_out FP32 [N, 2304]
  u16* mlp = nullptr;        // BF16 [S, 4096]; also the patches [4N, 1200] and stage 0
  unsigned char* rgb = nullptr;  // the image bytes [cap * 1600 * 3]
  float* cap[3] = {nullptr, nullptr, nullptr};  // stages 1..3
  // last encode
  int last_rows = 0, last_cols = 0, last_w = 0, last_h = 0;
  bool encoded = false;

  const u16* w(int t) const { return reinterpret_cast<const u16*>(weights + table[t].offset); }
  const u16* lw(int l, int k) const { return w(kGlobals + l * kPerLayer + k); }
};

static void free_scratch(Ink2Vis* v) {
  cudaFree(v->x);
  cudaFree(v->lnb);
  cudaFree(v->qkv);
  cudaFree(v->mlp);
  cudaFree(v->rgb);
  for (int k = 0; k < 3; ++k) cudaFree(v->cap[k]), v->cap[k] = nullptr;
  v->x = nullptr, v->lnb = nullptr, v->qkv = nullptr, v->mlp = nullptr, v->rgb = nullptr;
  v->scratch_bytes = 0;
}

Ink2Vis* ink2v_create(const Ink2VisionConfig* c, CUstream_st* stream, char* err) {
  if (!c) {
    say(err, "vision config: null pointer");
    return nullptr;
  }
#define INK2V_WANT(field, want)                                                                                \
  if (c->field != (want)) {                                                                                    \
    say(err, "vision config: %s = %d, this engine supports only %d (Inkling-2 Mini)", #field, (int)c->field, \
        (int)(want));                                                                                          \
    return nullptr;                                                                                            \
  }
  INK2V_WANT(hidden, kD)
  INK2V_WANT(layers, kLayers)
  INK2V_WANT(heads, kHeads)
  INK2V_WANT(head_dim, kHeadDim)
  INK2V_WANT(inter, kFF)
  INK2V_WANT(patch, kPatch)
  INK2V_WANT(merge, 2)
  INK2V_WANT(temporal, 2)
  INK2V_WANT(grid, kGrid)
  INK2V_WANT(proj_hidden, kMergedDim)
  INK2V_WANT(text_hidden, kOut)
  INK2V_WANT(projector_bias, 0)
#undef INK2V_WANT
  if (!(c->eps == 1e-5f)) {
    say(err, "vision config: eps = %g, this engine supports only 1e-05", (double)c->eps);
    return nullptr;
  }
  if (!(c->rope_theta == 10000.0f)) {
    say(err, "vision config: rope_theta = %g, this engine supports only 10000", (double)c->rope_theta);
    return nullptr;
  }
  if (c->max_image_tokens < 64 || c->max_image_tokens > 4096) {
    say(err, "vision config: max_image_tokens = %d, allowed 64..4096", c->max_image_tokens);
    return nullptr;
  }
  if (c->max_feature_rows < c->max_image_tokens || c->max_feature_rows > 65536) {
    say(err, "vision config: max_feature_rows = %d, allowed max_image_tokens (%d)..65536", c->max_feature_rows,
        c->max_image_tokens);
    return nullptr;
  }
  int dev = 0, major = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) !=
                                                 cudaSuccess) {
    cudaGetLastError();
    say(err, "vision: no CUDA device");
    return nullptr;
  }
  if (major < 8) {
    say(err, "vision: the image encoder needs a GPU of compute capability 8.0 or newer (this one: %d.x)", major);
    return nullptr;
  }
  cudaFuncAttributes fa{};
  if (cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(&k_gemm<EP_F32>)) != cudaSuccess) {  // no code for this GPU
    cudaGetLastError();
    say(err, "vision: this engine library has no image-encoder code for this GPU (compute capability %d.x)", major);
    return nullptr;
  }
  Ink2Vis* v = new (std::nothrow) Ink2Vis();
  if (!v) {
    say(err, "vision: out of host memory");
    return nullptr;
  }
  v->cfg = *c;
  v->stream = reinterpret_cast<cudaStream_t>(stream);
  build_table(v->table);
  for (int k = 0; k < (int)v->table.size(); ++k) v->index[v->table[k].name] = k;
  v->have.assign(v->table.size(), 0);
  v->weight_bytes = v->table.back().offset + v->table.back().bytes;
  for (int k = 0; k < 16; ++k) v->rope[k] = 1.0f / powf(10000.0f, (float)(2 * k) / 32.0f);
  return v;
}

int ink2v_set_tensor_shaped(Ink2Vis* v, const char* name, const void* bf16, int dtype_bytes, const long long* shape,
                            int ndim, long long nbytes, char* err) {
  if (!v || !name) {
    say(err, "vision set_tensor: null encoder or name");
    return -1;
  }
  if (v->finalized) {
    say(err, "vision set_tensor %s: the encoder is already finalized", name);
    return -1;
  }
  auto it = v->index.find(name);
  if (it == v->index.end()) {
    say(err, "vision set_tensor: unknown tensor name '%.200s'", name);
    return -1;
  }
  const TensorInfo& t = v->table[it->second];
  if (v->have[it->second]) {
    say(err, "vision set_tensor %s: given twice", name);
    return -1;
  }
  if (dtype_bytes != 2) {
    say(err, "vision set_tensor %s: element size %d bytes, expected 2 (BF16)", name, dtype_bytes);
    return -1;
  }
  if (shape) {
    bool same = ndim == t.ndim;
    for (int k = 0; same && k < ndim; ++k) same = shape[k] == t.shape[k];
    if (!same) {
      char got[96] = "", want[64];
      int n = 0;
      for (int k = 0; k < ndim && k < 4 && n < 80; ++k) n += snprintf(got + n, sizeof(got) - n, k ? ", %lld" : "%lld", shape[k]);
      if (t.ndim == 2)
        snprintf(want, sizeof(want), "%lld, %lld", t.shape[0], t.shape[1]);
      else
        snprintf(want, sizeof(want), "%lld", t.shape[0]);
      say(err, "vision set_tensor %s: shape [%s], expected [%s]", name, got, want);
      return -1;
    }
  }
  if (nbytes != t.bytes) {
    say(err, "vision set_tensor %s: %lld bytes, expected %lld (BF16)", name, nbytes, t.bytes);
    return -1;
  }
  if (!bf16) {
    say(err, "vision set_tensor %s: null data", name);
    return -1;
  }
  if (!v->weights) {
    if (cudaMalloc(&v->weights, (size_t)v->weight_bytes) != cudaSuccess) {
      cudaGetLastError();
      v->weights = nullptr;
      say(err, "vision: cannot allocate %lld bytes of GPU memory for the image encoder weights", v->weight_bytes);
      return -1;
    }
  }
  if (cudaMemcpyAsync(v->weights + t.offset, bf16, (size_t)t.bytes, cudaMemcpyHostToDevice, v->stream) != cudaSuccess ||
      cudaStreamSynchronize(v->stream) != cudaSuccess) {
    say(err, "vision set_tensor %s: copy to the GPU failed: %s", name, cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  v->have[it->second] = 1;
  ++v->n_have;
  return 0;
}

int ink2v_set_tensor(Ink2Vis* v, const char* name, const void* bf16, long long nbytes, char* err) {
  return ink2v_set_tensor_shaped(v, name, bf16, 2, nullptr, 0, nbytes, err);
}

int ink2v_set_capture(Ink2Vis* v, int on, char* err) {
  if (!v) {
    say(err, "vision: null encoder");
    return -1;
  }
  if (v->finalized) {
    say(err, "vision set_capture: call it before finalize");
    return -1;
  }
  v->capture = on != 0;
  return 0;
}

int ink2v_finalize(Ink2Vis* v, char* err) {
  if (!v) {
    say(err, "vision: null encoder");
    return -1;
  }
  if (v->finalized) {
    say(err, "vision finalize: already finalized");
    return -1;
  }
  if (v->n_have != (int)v->table.size()) {
    int first = 0;
    while (first < (int)v->have.size() && v->have[first]) ++first;
    say(err, "vision finalize: %d of %d tensors given; missing for example %s", v->n_have, (int)v->table.size(),
        v->table[first].name.c_str());
    return -1;
  }
  const long long cap = v->cfg.max_image_tokens, S = 1 + 4 * cap;
  struct Req {
    void** p;
    long long bytes;
  } req[] = {{(void**)&v->x, S * kD * 4},         {(void**)&v->lnb, S * kD * 2},
             {(void**)&v->qkv, S * kQKV * 2},     {(void**)&v->mlp, S * kFF * 2},
             {(void**)&v->rgb, cap * kTile * kTile * 3},
             {(void**)&v->cap[0], v->capture ? 4 * cap * kD * 4 : 0},
             {(void**)&v->cap[1], v->capture ? S * kD * 4 : 0},
             {(void**)&v->cap[2], v->capture ? S * kD * 4 : 0}};
  long long total = 0;
  for (auto& r : req) {
    if (!r.bytes) continue;
    if (cudaMalloc(r.p, (size_t)r.bytes) != cudaSuccess) {
      cudaGetLastError();
      *r.p = nullptr;
      free_scratch(v);
      say(err, "vision finalize: cannot allocate %lld bytes of GPU scratch (max_image_tokens %lld)", r.bytes, cap);
      return -1;
    }
    total += r.bytes;
  }
  v->scratch_bytes = total;
  v->finalized = true;
  return 0;
}

static void launch_gemm(Ink2Vis* v, int mode, const u16* a, int lda, const u16* w, int M, int N, int K, const u16* bias,
                        const u16* gamma, float* of, u16* oh, int ldo) {
  GemmArgs p;
  p.a = a, p.w = w, p.M = M, p.N = N, p.K = K, p.lda = lda, p.bias = bias, p.gamma = gamma, p.of = of, p.oh = oh;
  p.ldo = ldo, p.cols = v->last_cols;
  memcpy(p.rope, v->rope, sizeof(p.rope));
  dim3 grid((N + GBN - 1) / GBN, (M + GBM - 1) / GBM);
  switch (mode) {
    case EP_F32: k_gemm<EP_F32><<<grid, 256, 0, v->stream>>>(p); break;
    case EP_QKV_ROPE: k_gemm<EP_QKV_ROPE><<<grid, 256, 0, v->stream>>>(p); break;
    case EP_RESID: k_gemm<EP_RESID><<<grid, 256, 0, v->stream>>>(p); break;
    default: k_gemm<EP_GELU><<<grid, 256, 0, v->stream>>>(p); break;
  }
}

int ink2v_encode(Ink2Vis* v, const unsigned char* rgb, int width, int height, long long rgb_bytes, int rows, int cols,
                 unsigned short* dev_out, long long out_rows, char* err) {
  if (!v) {
    say(err, "vision encode: null encoder");
    return -1;
  }
  if (!v->finalized) {
    say(err, "vision encode: the encoder is not finalized");
    return -1;
  }
  if (!rgb || !dev_out) {
    say(err, "vision encode: null image or output pointer");
    return -1;
  }
  if (width < 1 || height < 1) {
    say(err, "vision encode: image size %d x %d", width, height);
    return -1;
  }
  const long long want_rows = ((long long)height + kTile - 1) / kTile, want_cols = ((long long)width + kTile - 1) / kTile;
  if (rows != want_rows || cols != want_cols) {
    say(err, "vision encode: grid %d x %d given, a %d x %d image has %lld x %lld", rows, cols, width, height, want_rows,
        want_cols);
    return -1;
  }
  const long long n = want_rows * want_cols;
  if (n > v->cfg.max_image_tokens) {
    say(err, "vision encode: %lld image tokens, the cap is %d", n, v->cfg.max_image_tokens);
    return -1;
  }
  if (rgb_bytes != (long long)width * height * 3) {
    say(err, "vision encode: %lld bytes of RGB, a %d x %d image has %lld", rgb_bytes, width, height,
        (long long)width * height * 3);
    return -1;
  }
  if (out_rows < n) {
    say(err, "vision encode: the output has room for %lld rows, the image needs %lld", out_rows, n);
    return -1;
  }
  // An error left by earlier work on the device belongs to the engine, which reports it on its next check;
  // refuse without clearing it (clearing would hide it from the engine).
  const cudaError_t before = cudaPeekAtLastError();
  if (before != cudaSuccess) {
    say(err, "vision encode: the GPU reports an earlier error: %s", cudaGetErrorString(before));
    return -1;
  }
  const int N = (int)n, S = 1 + 4 * N;
  v->last_rows = rows, v->last_cols = cols, v->last_w = width, v->last_h = height;
  v->encoded = false;
  if (cudaMemcpyAsync(v->rgb, rgb, (size_t)rgb_bytes, cudaMemcpyHostToDevice, v->stream) != cudaSuccess) {
    say(err, "vision encode: image copy failed: %s", cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  const float eps = v->cfg.eps;
  float* x = v->x;
  // patches, projection into rows 1.. of x, position grid and class row, ln_pre
  k_patchify<<<4 * N, 256, 0, v->stream>>>(v->rgb, width, height, cols, v->mlp);
  launch_gemm(v, EP_F32, v->mlp, kPatchIn, v->w(T_PATCH), 4 * N, kD, kPatchIn, nullptr, nullptr, x + kD, nullptr, kD);
  if (v->capture) cudaMemcpyAsync(v->cap[0], x + kD, (size_t)4 * N * kD * 4, cudaMemcpyDeviceToDevice, v->stream);
  k_embed<<<S, 256, 0, v->stream>>>(x, v->w(T_POS), v->w(T_CLS), v->w(T_CLSPOS), rows, cols);
  if (v->capture) cudaMemcpyAsync(v->cap[1], x, (size_t)S * kD * 4, cudaMemcpyDeviceToDevice, v->stream);
  k_layernorm<<<S, 256, 0, v->stream>>>(x, v->w(T_LNPRE_W), v->w(T_LNPRE_B), eps, x, nullptr);
  const dim3 agrid((S + AQ - 1) / AQ, kHeads);
  for (int l = 0; l < kLayers; ++l) {
    k_layernorm<<<S, 256, 0, v->stream>>>(x, v->lw(l, L_LN1W), v->lw(l, L_LN1B), eps, nullptr, v->lnb);
    launch_gemm(v, EP_QKV_ROPE, v->lnb, kD, v->lw(l, L_QW), S, kQKV, kD, v->lw(l, L_QB), nullptr, nullptr, v->qkv, kQKV);
    k_attn<<<agrid, 128, 0, v->stream>>>(v->qkv, v->lnb, S);
    launch_gemm(v, EP_RESID, v->lnb, kD, v->lw(l, L_OW), S, kD, kD, v->lw(l, L_OB), v->lw(l, L_LS1), x, nullptr, kD);
    k_layernorm<<<S, 256, 0, v->stream>>>(x, v->lw(l, L_LN2W), v->lw(l, L_LN2B), eps, nullptr, v->lnb);
    launch_gemm(v, EP_GELU, v->lnb, kD, v->lw(l, L_FCW), S, kFF, kD, v->lw(l, L_FCB), nullptr, nullptr, v->mlp, kFF);
    launch_gemm(v, EP_RESID, v->mlp, kFF, v->lw(l, L_PRW), S, kD, kFF, v->lw(l, L_PRB), v->lw(l, L_LS2), x, nullptr, kD);
    if (l == 0 && v->capture) cudaMemcpyAsync(v->cap[2], x, (size_t)S * kD * 4, cudaMemcpyDeviceToDevice, v->stream);
  }
  // merge (a view of rows 1..4N as [N, 4096]) and the projector
  const int blocks = 1024;
  k_to_bf16<<<blocks, 256, 0, v->stream>>>(x + kD, v->lnb + kD, (long long)4 * N * kD / 2);
  launch_gemm(v, EP_GELU, v->lnb + kD, kMergedDim, v->w(T_PROJ_IN), N, kMergedDim, kMergedDim, nullptr, nullptr, nullptr,
              v->mlp, kMergedDim);
  float* out32 = reinterpret_cast<float*>(v->qkv);
  launch_gemm(v, EP_F32, v->mlp, kMergedDim, v->w(T_PROJ_OUT), N, kOut, kMergedDim, nullptr, nullptr, out32, nullptr, kOut);
  k_to_bf16<<<blocks, 256, 0, v->stream>>>(out32, dev_out, (long long)N * kOut / 2);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    say(err, "vision encode: kernel launch failed: %s", cudaGetErrorString(e));
    return -1;
  }
  v->encoded = true;
  return 0;
}

long long ink2v_stage_floats(const Ink2Vis* v, int stage) {
  if (!v || !v->encoded) return 0;
  const long long n = (long long)v->last_rows * v->last_cols, S = 1 + 4 * n;
  switch (stage) {
    case 0: return n * kTile * kTile * 3;
    case 1: return v->capture ? 4 * n * kD : 0;
    case 2:
    case 3: return v->capture ? S * kD : 0;
    case 4: return S * kD;
    case 5: return n * kMergedDim;
    case 6: return n * kOut;
    default: return 0;
  }
}

int ink2v_stage(Ink2Vis* v, int stage, float* out, long long out_floats, char* err) {
  if (!v || !out) {
    say(err, "vision stage: null encoder or output");
    return -1;
  }
  if (!v->encoded) {
    say(err, "vision stage: no image encoded yet");
    return -1;
  }
  if (stage < 0 || stage >= INK2V_STAGES) {
    say(err, "vision stage: %d, allowed 0..%d", stage, INK2V_STAGES - 1);
    return -1;
  }
  const long long need = ink2v_stage_floats(v, stage);
  if (!need) {
    say(err, "vision stage %d: not kept (stages 1..3 need set_capture before finalize)", stage);
    return -1;
  }
  if (out_floats < need) {
    say(err, "vision stage %d: room for %lld floats, need %lld", stage, out_floats, need);
    return -1;
  }
  const float* src = nullptr;
  if (stage == 0) {
    const int n = v->last_rows * v->last_cols;
    k_pixels<<<n, 256, 0, v->stream>>>(v->rgb, v->last_w, v->last_h, v->last_cols, reinterpret_cast<float*>(v->mlp));
    src = reinterpret_cast<const float*>(v->mlp);
  } else if (stage <= 3) {
    src = v->cap[stage - 1];
  } else if (stage <= 4) {
    src = v->x;
  } else if (stage == 5) {
    src = v->x + kD;
  } else {
    src = reinterpret_cast<const float*>(v->qkv);
  }
  if (cudaMemcpyAsync(out, src, (size_t)need * 4, cudaMemcpyDeviceToHost, v->stream) != cudaSuccess ||
      cudaStreamSynchronize(v->stream) != cudaSuccess) {
    say(err, "vision stage %d: copy failed: %s", stage, cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  return 0;
}

int ink2v_memory(const Ink2Vis* v, long long* weights, long long* scratch, char* err) {
  if (!v) {
    say(err, "vision: null encoder");
    return -1;
  }
  if (weights) *weights = v->weights ? v->weight_bytes : 0;
  if (scratch) *scratch = v->scratch_bytes;
  return 0;
}

unsigned short* ink2v_alloc_rows(long long bytes, char* err) {
  void* p = nullptr;
  if (bytes <= 0 || cudaMalloc(&p, (size_t)bytes) != cudaSuccess) {
    cudaGetLastError();
    say(err, "the feature buffer of %lld MiB could not be allocated (GPU memory)", bytes >> 20);
    return nullptr;
  }
  if (cudaMemset(p, 0, (size_t)bytes) != cudaSuccess) {
    cudaGetLastError();
    cudaFree(p);
    say(err, "the feature buffer could not be cleared");
    return nullptr;
  }
  return static_cast<unsigned short*>(p);
}

void ink2v_free_rows(unsigned short* p) {
  if (p) cudaFree(p);
}

void ink2v_destroy(Ink2Vis* v) {
  if (!v) return;
  cudaStreamSynchronize(v->stream);
  free_scratch(v);
  cudaFree(v->weights);
  cudaGetLastError();
  delete v;
}
