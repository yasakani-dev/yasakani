// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "ink2_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace yskn::ink2 {
namespace {

__device__ __forceinline__ float ink2_e4m3(uint8_t b) {
  const uint16_t h = uint16_t(((uint32_t(b) & 0x80u) << 8) | ((uint32_t(b) & 0x7Fu) << 7));
  return __half2float(__ushort_as_half(h)) * 256.0f;
}

__device__ __forceinline__ float ink2_ue8m0(uint8_t b) {
  const uint32_t bits = (b == 0u) ? 0x00400000u : (uint32_t(b) << 23);
  return __uint_as_float(bits);
}

__device__ __constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

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
  return sh[0];
}

struct Ptrs6 {
  const float* p[6];
};

// ---- 1. Embed + RMSNorm ----
__global__ void k_embed_norm(const int32_t* tokens, int tok_off, const __nv_bfloat16* embed,
                             const __nv_bfloat16* norm_w, float* out, int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int ntg = blockDim.x;
  const int id = tokens[tok_off + t];
  const __nv_bfloat16* row = embed + size_t(id) * size_t(H);
  float xs[6];
  float ss = 0.0f;
  int k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    const float v = __bfloat162float(row[c]);
    xs[k] = v;
    ss += v * v;
  }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / float(H) + eps);
  float* dst = out + size_t(t) * size_t(H);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) dst[c] = xs[k] * inv * __bfloat162float(norm_w[c]);
}

// ---- 2. Generic RMSNorm (alias safe) ----
__global__ void k_rmsnorm(const float* x, const void* w, bool w_f32, float* out, int H, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int ntg = blockDim.x;
  const float* src = x + size_t(t) * size_t(H);
  float* dst = out + size_t(t) * size_t(H);
  float xs[8];
  float ss = 0.0f;
  int k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    const float v = src[c];
    xs[k] = v;
    ss += v * v;
  }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / float(H) + eps);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    float y = xs[k] * inv;
    if (w) y *= w_f32 ? static_cast<const float*>(w)[c] : __bfloat162float(static_cast<const __nv_bfloat16*>(w)[c]);
    dst[c] = y;
  }
}

// ---- 3. Block Attention Residual ----
__global__ void k_attn_res(Ptrs6 st, int M, const void* queries, bool q_f32, int Q, float* out, int T, int H,
                           float eps, float scale) {
  __shared__ float sh[32];
  __shared__ float tinv[6];
  __shared__ float tsc[6][4];
  __shared__ float tw[6][4];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int ntg = blockDim.x;

  float qv[4][5];
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
    float sv[5];
    float ss = 0.0f;
    int k = 0;
    for (int c = tid; c < H; c += ntg, ++k) {
      const float v = sr[c];
      sv[k] = v;
      ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / float(H) + eps);
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
    float mx = -INFINITY;
    for (int m = 0; m < M; ++m) mx = fmaxf(mx, tsc[m][tid]);
    float exps[6];
    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
      exps[m] = expf(tsc[m][tid] - mx);
      sum += exps[m];
    }
    const float isum = 1.0f / sum;
    for (int m = 0; m < M; ++m) tw[m][tid] = exps[m] * isum;
  }
  __syncthreads();
  for (int c = tid; c < H; c += ntg) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < M; ++m) {
      const float kv = st.p[m][size_t(t) * size_t(H) + size_t(c)] * tinv[m];
      for (int q = 0; q < Q; ++q) acc[q] += tw[m][q] * kv;
    }
    for (int q = 0; q < Q; ++q) out[(size_t(q) * size_t(T) + size_t(t)) * size_t(H) + size_t(c)] = acc[q];
  }
}

// ---- 4. Cross-Pass MessageGate ----
__global__ void k_msg_gate(const float* h, const float* m, const __nv_bfloat16* down_w,
                           const __nv_bfloat16* forget_w, const __nv_bfloat16* gain_w, float* out,
                           int H, int R, float eps) {
  __shared__ float sh[32];
  __shared__ float th_nh[2304];
  __shared__ float th_nm[2304];
  __shared__ float th_g[64];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int ntg = blockDim.x;
  const int sg = tid >> 5;
  const int lane = tid & 31;
  const int nsg = ntg >> 5;

  const float* hr = h + size_t(t) * size_t(H);
  const float* mr = m + size_t(t) * size_t(H);
  float hs[5], ms[5];
  float ssh = 0.0f, ssm = 0.0f;
  int k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    const float hv = hr[c], mv = mr[c];
    hs[k] = hv;
    ms[k] = mv;
    ssh += hv * hv;
    ssm += mv * mv;
  }
  ssh = block_sum(ssh, sh);
  ssm = block_sum(ssm, sh);
  const float invh = rsqrtf(ssh / float(H) + eps);
  const float invm = rsqrtf(ssm / float(H) + eps);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    th_nh[c] = hs[k] * invh;
    th_nm[c] = ms[k] * invm;
  }
  __syncthreads();

  for (int r = sg; r < R; r += nsg) {
    float acc = 0.0f;
    const size_t row_off = size_t(r) * size_t(2 * H);
    for (int c = lane * 4; c < H; c += 128) {
      const float4 nh4 = *reinterpret_cast<const float4*>(th_nh + c);
      const float4 nm4 = *reinterpret_cast<const float4*>(th_nm + c);
      const __nv_bfloat16* b0 = down_w + row_off + c;
      const __nv_bfloat16* b1 = down_w + row_off + H + c;
      acc += nh4.x * __bfloat162float(b0[0]) + nh4.y * __bfloat162float(b0[1]) +
             nh4.z * __bfloat162float(b0[2]) + nh4.w * __bfloat162float(b0[3]);
      acc += nm4.x * __bfloat162float(b1[0]) + nm4.y * __bfloat162float(b1[1]) +
             nm4.z * __bfloat162float(b1[2]) + nm4.w * __bfloat162float(b1[3]);
    }
    acc = warp_sum(acc);
    if (lane == 0) th_g[r] = acc;
  }
  __syncthreads();

  float* dst = out + size_t(t) * size_t(H);
  k = 0;
  for (int c = tid; c < H; c += ntg, ++k) {
    float df = 0.0f, dg = 0.0f;
    const size_t off = size_t(c) * size_t(R);
    for (int r = 0; r < R; ++r) {
      const float g = th_g[r];
      df += g * __bfloat162float(forget_w[off + r]);
      dg += g * __bfloat162float(gain_w[off + r]);
    }
    const float fg = tanhf(df);
    const float gn = tanhf(dg);
    dst[c] = (1.0f - fg) * hs[k] + gn * ms[k];
  }
}

// ---- 5. Linear Projections ----
__global__ void k_bf16_mv(const float* x, const void* w, bool w_f32, float* y,
                          int M, int N, int K, bool add) {
  const int n = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (n >= N) return;
  const int lane = threadIdx.x & 31;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff = size_t(n) * size_t(K);
  for (int k = lane * 4; k < K; k += 128) {
    float4 wv;
    if (w_f32) {
      wv = *reinterpret_cast<const float4*>(static_cast<const float*>(w) + woff + k);
    } else {
      const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(w) + woff + k;
      wv = make_float4(__bfloat162float(wb[0]), __bfloat162float(wb[1]),
                       __bfloat162float(wb[2]), __bfloat162float(wb[3]));
    }
    for (int m = 0; m < M; ++m) {
      const float4 xv = *reinterpret_cast<const float4*>(x + size_t(m) * size_t(K) + k);
      acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
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

__global__ void k_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ---- 6. Short Convolutions ----
__global__ void k_sconv_static(const float* x, const void* w, bool w_f32, float* hist, float* out,
                               int T, int C, float scale) {
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
  for (int t = 0; t < T; ++t) {
    const float xt = x[size_t(t) * size_t(C) + c];
    const float conv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
    out[size_t(t) * size_t(C) + c] = (conv + xt) * scale;
    h0 = h1; h1 = h2; h2 = xt;
  }
  hist[0 * size_t(C) + c] = h0;
  hist[1 * size_t(C) + c] = h1;
  hist[2 * size_t(C) + c] = h2;
}

__global__ void k_sconv_dynamic(const float* x, const float* dyn, const void* static_w, bool sw_f32,
                                float* hist, float* out, int T, int C, int n_heads) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int head_dim = C / n_heads;
  const int h = c / head_dim;
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
  }
  hist[0 * size_t(C) + c] = h0;
  hist[1 * size_t(C) + c] = h1;
  hist[2 * size_t(C) + c] = h2;
}

// ---- 7. QKVR Prep ----
__global__ void k_qkvr_prep(float* q, const float* k, const float* v, float* r,
                            const void* q_scale, const void* r_scale, bool sc_f32,
                            __half* kc, __half* vc, int Hq, int Hkv, int D, int d_rel,
                            int pos0, int kv_cap, bool is_swa, float qk_clip, float r_clip,
                            float v_eps, float log_floor, float log_alpha) {
  __shared__ float sh[32];
  const int kind_h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int pos = pos0 + t;
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
    kc[(size_t(slot) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid)] = __float2half(val * mult);
    return;
  }
  rem -= Hkv;
  if (rem < Hkv) {
    const int h = rem;
    const float val = v[(size_t(t) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid)];
    const float ss = block_sum(val * val, sh);
    const float inv = rsqrtf(ss / float(D) + v_eps);
    vc[(size_t(slot) * size_t(Hkv) + size_t(h)) * size_t(D) + size_t(tid)] = __float2half(val * inv);
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

// ---- 8. Rel Bias ----
__global__ void k_rel_bias(const float* r, const void* rel_proj, bool proj_f32, float* rb,
                           int Hq, int d_rel, int extent) {
  const int col = blockIdx.x * 32 + threadIdx.x;
  if (col >= extent) return;
  const int h = blockIdx.y, t = blockIdx.z;
  const float* rv = r + (size_t(t) * size_t(Hq) + size_t(h)) * size_t(d_rel);
  float acc = 0.0f;
  for (int d = 0; d < d_rel; ++d) {
    const size_t pidx = size_t(d) * size_t(extent) + size_t(col);
    const float pv = proj_f32 ? static_cast<const float*>(rel_proj)[pidx]
                              : __bfloat162float(static_cast<const __nv_bfloat16*>(rel_proj)[pidx]);
    acc += rv[d] * pv;
  }
  rb[(size_t(t) * size_t(Hq) + size_t(h)) * size_t(extent) + size_t(col)] = acc;
}

// ---- 9. Attention ----
__global__ void k_attn_decode(const float* q, const __half* kc, const __half* vc, const float* rb,
                              float* pm, float* pl, float* po, int pos, int kv_cap, int Hq, int Hkv,
                              int D, int rel_extent, bool is_swa, int swa, int chunk, int nchunks,
                              float scale) {
  __shared__ float tgm[4], tgl[4], tgo[4][128];
  const int h = blockIdx.x, cid = blockIdx.y;
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int hkv = h / (Hq / Hkv);
  const int d0 = lane * 4;
  const float4 qv = *reinterpret_cast<const float4*>(q + size_t(h) * size_t(D) + size_t(d0));

  const int lk = is_swa ? min(pos + 1, swa) : (pos + 1);
  const int c0 = (pos + 1 - lk) + cid * chunk;
  const int c1 = min(pos + 1, c0 + chunk);

  float mw = -INFINITY, lw = 0.0f;
  float4 ow = make_float4(0, 0, 0, 0);
  for (int kp = c0 + sg; kp < c1; kp += 4) {
    const int slot = is_swa ? (kp % kv_cap) : kp;
    const size_t koff = (size_t(slot) * size_t(Hkv) + size_t(hkv)) * size_t(D) + size_t(d0);
    const float dot = qv.x * __half2float(kc[koff + 0]) + qv.y * __half2float(kc[koff + 1]) +
                      qv.z * __half2float(kc[koff + 2]) + qv.w * __half2float(kc[koff + 3]);
    float sc = warp_sum(dot) * scale;
    const int dist = pos - kp;
    if (dist < rel_extent) sc += rb[size_t(h) * size_t(rel_extent) + size_t(dist)];
    const float4 vv = make_float4(__half2float(vc[koff + 0]), __half2float(vc[koff + 1]),
                                  __half2float(vc[koff + 2]), __half2float(vc[koff + 3]));
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
  const int h = blockIdx.x, lane = threadIdx.x;
  const int d0 = lane * 4;
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

__global__ void k_attn_prefill(const float* q, const __half* kc, const __half* vc, const float* rb,
                               const float* sinks, const float* gate, float* out, int pos0, int kv_cap,
                               int Hq, int Hkv, int D, int rel_extent, bool is_swa, int swa, float scale) {
  __shared__ float tgm[4], tgl[4], tgo[4][128];
  const int h = blockIdx.x, t = blockIdx.y;
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int hkv = h / (Hq / Hkv);
  const int qpos = pos0 + t;
  const int d0 = lane * 4;
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
    const float dot = qv.x * __half2float(kc[koff + 0]) + qv.y * __half2float(kc[koff + 1]) +
                      qv.z * __half2float(kc[koff + 2]) + qv.w * __half2float(kc[koff + 3]);
    float sc = warp_sum(dot) * scale;
    const int dist = qpos - kp;
    if (dist < rel_extent) sc += rbh[dist];
    const float4 vv = make_float4(__half2float(vc[koff + 0]), __half2float(vc[koff + 1]),
                                  __half2float(vc[koff + 2]), __half2float(vc[koff + 3]));
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

// ---- 10. MoE Router & Experts ----
__global__ void k_moe_route(const float* logits, const void* bias, bool bias_f32,
                            uint32_t* eid, float* ew, int E, int K, float route_scale) {
  __shared__ float ss[128], sb[128];
  const int t = blockIdx.x, tid = threadIdx.x;
  const float l = logits[size_t(t) * size_t(E) + size_t(tid)];
  const float s = 1.0f / (1.0f + expf(-l));
  const float b = bias_f32 ? static_cast<const float*>(bias)[tid]
                           : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[tid]);
  ss[tid] = s;
  sb[tid] = s + b;
  __syncthreads();
  if (tid == 0) {
    uint32_t chosen[8];
    float wsum = 0.0f;
    for (int slot = 0; slot < K; ++slot) {
      float best = -INFINITY;
      uint32_t bi = 0;
      for (int e = 0; e < E; ++e) {
        if (sb[e] > best) { best = sb[e]; bi = uint32_t(e); }
      }
      chosen[slot] = bi;
      sb[bi] = -INFINITY;
      wsum += ss[bi];
    }
    const float inv = route_scale / (wsum + 1e-20f);
    for (int slot = 0; slot < K; ++slot) {
      eid[size_t(t) * size_t(K) + size_t(slot)] = chosen[slot];
      ew[size_t(t) * size_t(K) + size_t(slot)] = ss[chosen[slot]] * inv;
    }
  }
}

__global__ void k_moe_nvfp4_w13_situ(const float* x, const uint32_t* eid, const uint8_t* w13,
                                     const uint8_t* sc, const float* outer, float* act,
                                     int H, int I, int K_top) {
  const int j = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (j >= I) return;
  const int lane = threadIdx.x & 31;
  const int slot = blockIdx.y, t = blockIdx.z;
  const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
  const int two_i = 2 * I, half_h = H / 2, blocks = H / 16;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * j);
  const size_t wu_row = wg_row + 1;
  const float* xp = x + size_t(t) * size_t(H);

  float ag = 0.0f, au = 0.0f;
  for (int blk = lane; blk < blocks; blk += 32) {
    const float sg_b = ink2_e4m3(sc[wg_row * size_t(blocks) + size_t(blk)]);
    const float su_b = ink2_e4m3(sc[wu_row * size_t(blocks) + size_t(blk)]);
    const uint8_t* pg = w13 + wg_row * size_t(half_h) + size_t(blk) * 8;
    const uint8_t* pu = w13 + wu_row * size_t(half_h) + size_t(blk) * 8;
    const int kbase = blk * 16;
    float dg = 0.0f, du = 0.0f;
    for (int b = 0; b < 8; ++b) {
      const float x0 = xp[kbase + 2 * b], x1 = xp[kbase + 2 * b + 1];
      const uint8_t bg = pg[b], bu = pu[b];
      dg += kE2M1[bg & 15u] * x0 + kE2M1[bg >> 4] * x1;
      du += kE2M1[bu & 15u] * x0 + kE2M1[bu >> 4] * x1;
    }
    ag += dg * sg_b;
    au += du * su_b;
  }
  const float os = outer[e];
  const float g = warp_sum(ag) * os;
  const float u = warp_sum(au) * os;
  if (lane == 0) {
    act[(size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I) + size_t(j)] =
        (4.0f * tanhf(g * 0.25f)) * (1.0f / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
  }
}

__global__ void k_moe_nvfp4_w2_acc(const float* act, const uint32_t* eid, const float* ew,
                                   const uint8_t* w2, const uint8_t* sc, const float* outer,
                                   float* out, int H, int I, int K_top) {
  const int n = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (n >= H) return;
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.y;
  const int half_i = I / 2, blocks = I / 16;
  float total = 0.0f;
  for (int slot = 0; slot < K_top; ++slot) {
    const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
    const float wt = ew[size_t(t) * size_t(K_top) + size_t(slot)];
    const size_t wrow = size_t(e) * size_t(H) + size_t(n);
    const float* ap = act + (size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I);
    float sum = 0.0f;
    for (int blk = lane; blk < blocks; blk += 32) {
      const float sb = ink2_e4m3(sc[wrow * size_t(blocks) + size_t(blk)]);
      const uint8_t* pw = w2 + wrow * size_t(half_i) + size_t(blk) * 8;
      const int kbase = blk * 16;
      float d = 0.0f;
      for (int b = 0; b < 8; ++b) {
        const uint8_t bw = pw[b];
        d += kE2M1[bw & 15u] * ap[kbase + 2 * b] + kE2M1[bw >> 4] * ap[kbase + 2 * b + 1];
      }
      sum += d * sb;
    }
    total += wt * (warp_sum(sum) * outer[e]);
  }
  if (lane == 0) out[size_t(t) * size_t(H) + size_t(n)] = total;
}

__global__ void k_moe_mxfp8_w13_situ(const float* x, const uint32_t* eid, const uint8_t* w13,
                                     const uint8_t* sc, float* act, int H, int I, int K_top) {
  const int j = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (j >= I) return;
  const int lane = threadIdx.x & 31;
  const int slot = blockIdx.y, t = blockIdx.z;
  const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
  const int two_i = 2 * I, blocks = H / 32;
  const size_t wg_row = size_t(e) * size_t(two_i) + size_t(2 * j);
  const size_t wu_row = wg_row + 1;
  const float* xp = x + size_t(t) * size_t(H);

  float ag = 0.0f, au = 0.0f;
  for (int blk = lane; blk < blocks; blk += 32) {
    const float sg_b = ink2_ue8m0(sc[wg_row * size_t(blocks) + size_t(blk)]);
    const float su_b = ink2_ue8m0(sc[wu_row * size_t(blocks) + size_t(blk)]);
    const uint8_t* pg = w13 + wg_row * size_t(H) + size_t(blk) * 32;
    const uint8_t* pu = w13 + wu_row * size_t(H) + size_t(blk) * 32;
    const int kbase = blk * 32;
    float dg = 0.0f, du = 0.0f;
    for (int b = 0; b < 32; ++b) {
      const float xv = xp[kbase + b];
      dg += ink2_e4m3(pg[b]) * xv;
      du += ink2_e4m3(pu[b]) * xv;
    }
    ag += dg * sg_b;
    au += du * su_b;
  }
  const float g = warp_sum(ag);
  const float u = warp_sum(au);
  if (lane == 0) {
    act[(size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I) + size_t(j)] =
        (4.0f * tanhf(g * 0.25f)) * (1.0f / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
  }
}

__global__ void k_moe_mxfp8_w2_acc(const float* act, const uint32_t* eid, const float* ew,
                                   const uint8_t* w2, const uint8_t* sc, float* out,
                                   int H, int I, int K_top) {
  const int n = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (n >= H) return;
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.y;
  const int blocks = I / 32;
  float total = 0.0f;
  for (int slot = 0; slot < K_top; ++slot) {
    const uint32_t e = eid[size_t(t) * size_t(K_top) + size_t(slot)];
    const float wt = ew[size_t(t) * size_t(K_top) + size_t(slot)];
    const size_t wrow = size_t(e) * size_t(H) + size_t(n);
    const float* ap = act + (size_t(t) * size_t(K_top) + size_t(slot)) * size_t(I);
    float sum = 0.0f;
    for (int blk = lane; blk < blocks; blk += 32) {
      const float sb = ink2_ue8m0(sc[wrow * size_t(blocks) + size_t(blk)]);
      const uint8_t* pw = w2 + wrow * size_t(I) + size_t(blk) * 32;
      const int kbase = blk * 32;
      float d = 0.0f;
      for (int b = 0; b < 32; ++b) d += ink2_e4m3(pw[b]) * ap[kbase + b];
      sum += d * sb;
    }
    total += wt * warp_sum(sum);
  }
  if (lane == 0) out[size_t(t) * size_t(H) + size_t(n)] = total;
}

__global__ void k_add(const float* src, float* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void k_softcap(float* x, int n, float cap, float inv_cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] * inv_cap);
}

__global__ void k_suppress_mask(float* logits, int V, const uint8_t* mask_bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < V && mask_bits[i]) logits[i] = -INFINITY;
}

__global__ void k_argmax_p1(const float* logits, int V, float* blk_v, int32_t* blk_i) {
  __shared__ float sv[32];
  __shared__ int32_t si[32];
  const int tid = threadIdx.x, ntg = blockDim.x, sg = tid >> 5, lane = tid & 31;
  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;
  for (int i = blockIdx.x * ntg + tid; i < V; i += gridDim.x * ntg) {
    const float v = logits[i];
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
    if (lane == 0) { blk_v[blockIdx.x] = v; blk_i[blockIdx.x] = id; }
  }
}

__global__ void k_argmax_p2(const float* blk_v, const int32_t* blk_i, int n, int32_t* d_tok) {
  __shared__ float sv[32];
  __shared__ int32_t si[32];
  const int tid = threadIdx.x, ntg = blockDim.x, sg = tid >> 5, lane = tid & 31;
  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;
  for (int i = tid; i < n; i += ntg) {
    const float v = blk_v[i];
    const int32_t id = blk_i[i];
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
    if (lane == 0) d_tok[0] = id;
  }
}

}  // namespace

void launch_embed_norm(const int32_t* tokens, int tok_off, const __nv_bfloat16* embed,
                       const __nv_bfloat16* norm_w, float* out, int T, int H, float eps, cudaStream_t s) {
  k_embed_norm<<<T, 512, 0, s>>>(tokens, tok_off, embed, norm_w, out, H, eps);
}

void launch_rmsnorm(const float* x, const void* w, bool w_f32, float* out, int T, int H, float eps,
                    cudaStream_t s) {
  k_rmsnorm<<<T, 512, 0, s>>>(x, w, w_f32, out, H, eps);
}

void launch_attn_res(const float* const* states, int M, const void* queries, bool q_f32, int Q, float* out,
                     int T, int H, float eps, float scale, cudaStream_t s) {
  Ptrs6 p{};
  for (int m = 0; m < 6; ++m) p.p[m] = m < M ? states[m] : states[0];
  k_attn_res<<<T, 512, 0, s>>>(p, M, queries, q_f32, Q, out, T, H, eps, scale);
}

void launch_msg_gate(const float* h, const float* m, const __nv_bfloat16* down_w,
                     const __nv_bfloat16* forget_w, const __nv_bfloat16* gain_w, float* out,
                     int T, int H, int R, float eps, cudaStream_t s) {
  k_msg_gate<<<T, 512, 0, s>>>(h, m, down_w, forget_w, gain_w, out, H, R, eps);
}

void launch_linear(cublasHandle_t cublas, const float* x, const void* w, bool w_f32, __nv_bfloat16* x_bf16_scratch,
                   float* y, int M, int N, int K, bool add, cudaStream_t s) {
  if (M <= 32 || w_f32 || !cublas || !x_bf16_scratch) {
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
  const float alpha = 1.0f;
  const float beta = add ? 1.0f : 0.0f;
  const cublasStatus_t st = cublasGemmEx(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16BF, K, x_bf16_scratch, CUDA_R_16BF, K,
      &beta, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in ink2 launch_linear");
}

void launch_sconv_static(const float* x, const void* w, bool w_f32, float* hist, float* out,
                         int T, int C, float scale, cudaStream_t s) {
  k_sconv_static<<<(C + 255) / 256, 256, 0, s>>>(x, w, w_f32, hist, out, T, C, scale);
}

void launch_sconv_dynamic(const float* x, const float* dyn, const void* static_w, bool sw_f32, float* hist,
                          float* out, int T, int C, int n_heads, cudaStream_t s) {
  k_sconv_dynamic<<<(C + 255) / 256, 256, 0, s>>>(x, dyn, static_w, sw_f32, hist, out, T, C, n_heads);
}

void launch_qkvr_prep(float* q, const float* k, const float* v, float* r,
                      const void* q_scale, const void* r_scale, bool sc_f32,
                      __half* kc, __half* vc, int T, int Hq, int Hkv, int D, int d_rel,
                      int pos0, int kv_cap, bool is_swa, float qk_clip, float r_clip,
                      float v_eps, float log_floor, float log_alpha, cudaStream_t s) {
  dim3 grid(2 * Hq + 2 * Hkv, T, 1);
  k_qkvr_prep<<<grid, D, 0, s>>>(q, k, v, r, q_scale, r_scale, sc_f32, kc, vc, Hq, Hkv, D, d_rel,
                                 pos0, kv_cap, is_swa, qk_clip, r_clip, v_eps, log_floor, log_alpha);
}

void launch_rel_bias(const float* r, const void* rel_proj, bool proj_f32, float* rb,
                     int T, int Hq, int d_rel, int extent, cudaStream_t s) {
  dim3 grid((extent + 31) / 32, Hq, T);
  k_rel_bias<<<grid, 32, 0, s>>>(r, rel_proj, proj_f32, rb, Hq, d_rel, extent);
}

void launch_attn_decode(const float* q, const __half* kc, const __half* vc, const float* rb,
                        const float* sinks, const float* gate, float* part_m, float* part_l, float* part_o,
                        float* out, int pos, int kv_cap, int Hq, int Hkv, int D, int rel_extent,
                        bool is_swa, int swa, cudaStream_t s) {
  const int lk = is_swa ? min(pos + 1, swa) : (pos + 1);
  int chunk = max(32, (lk + 63) / 64);
  chunk = (chunk + 3) / 4 * 4;
  const int nchunks = max(1, (lk + chunk - 1) / chunk);
  const float scale = 1.0f / std::sqrt(float(D));
  k_attn_decode<<<dim3(Hq, nchunks, 1), 128, 0, s>>>(q, kc, vc, rb, part_m, part_l, part_o, pos, kv_cap,
                                                     Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks, scale);
  k_attn_decode_reduce<<<Hq, 32, 0, s>>>(part_m, part_l, part_o, sinks, gate, out, D, nchunks);
}

void launch_attn_prefill(const float* q, const __half* kc, const __half* vc, const float* rb,
                         const float* sinks, const float* gate, float* out, int T, int pos0, int kv_cap,
                         int Hq, int Hkv, int D, int rel_extent, bool is_swa, int swa, cudaStream_t s) {
  const float scale = 1.0f / std::sqrt(float(D));
  k_attn_prefill<<<dim3(Hq, T, 1), 128, 0, s>>>(q, kc, vc, rb, sinks, gate, out, pos0, kv_cap,
                                                Hq, Hkv, D, rel_extent, is_swa, swa, scale);
}

void launch_moe_route(const float* logits, const void* bias, bool bias_f32, uint32_t* eid, float* ew,
                      int T, int E, int K, float route_scale, cudaStream_t s) {
  k_moe_route<<<T, E, 0, s>>>(logits, bias, bias_f32, eid, ew, E, K, route_scale);
}

void launch_moe_nvfp4(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc, const float* w13_outer,
                      const uint8_t* w2, const uint8_t* w2_sc, const float* w2_outer,
                      float* act_scratch, float* out, int T, int H, int I, int K_top, cudaStream_t s) {
  k_moe_nvfp4_w13_situ<<<dim3((I + 3) / 4, K_top, T), 128, 0, s>>>(x, eid, w13, w13_sc, w13_outer,
                                                                   act_scratch, H, I, K_top);
  k_moe_nvfp4_w2_acc<<<dim3((H + 3) / 4, T, 1), 128, 0, s>>>(act_scratch, eid, ew, w2, w2_sc, w2_outer,
                                                              out, H, I, K_top);
}

void launch_moe_mxfp8(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc,
                      const uint8_t* w2, const uint8_t* w2_sc,
                      float* act_scratch, float* out, int T, int H, int I, int K_top, cudaStream_t s) {
  k_moe_mxfp8_w13_situ<<<dim3((I + 3) / 4, K_top, T), 128, 0, s>>>(x, eid, w13, w13_sc, act_scratch, H, I, K_top);
  k_moe_mxfp8_w2_acc<<<dim3((H + 3) / 4, T, 1), 128, 0, s>>>(act_scratch, eid, ew, w2, w2_sc, out, H, I, K_top);
}

void launch_add(const float* src, float* dst, int n, cudaStream_t s) {
  k_add<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
}

void launch_softcap(float* x, int n, float cap, cudaStream_t s) {
  k_softcap<<<(n + 255) / 256, 256, 0, s>>>(x, n, cap, 1.0f / cap);
}

void launch_suppress_mask(float* logits, int V, const uint8_t* mask_bits, cudaStream_t s) {
  k_suppress_mask<<<(V + 255) / 256, 256, 0, s>>>(logits, V, mask_bits);
}

void launch_argmax_one(const float* logits, int V, int32_t* d_tok, float* blk_v, int32_t* blk_i, cudaStream_t s) {
  constexpr int NB = 256;
  k_argmax_p1<<<NB, 256, 0, s>>>(logits, V, blk_v, blk_i);
  k_argmax_p2<<<1, 256, 0, s>>>(blk_v, blk_i, NB, d_tok);
}

}  // namespace yskn::ink2
