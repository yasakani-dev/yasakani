// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "pdl.cuh"
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include "kernels.hpp"
#include <cuda_bf16.h>
#include "a8_prologue.cuh"
namespace yskn {


















template <int MT> __host__ __device__ constexpr int lno_uw() { return MT > 8 ? 4 : 2; }






template <int MT, int NR> __host__ __device__ constexpr int lno_nr_minb() { return MT > 8 ? (NR >= 8 ? 2 : NR >= 4 ? 4 : 8) : (NR >= 8 ? 3 : NR >= 4 ? 5 : 10); }
template <int K, int MT, bool XSM, int LOOK> __host__ __device__ constexpr int lno_minb() { return MT > 32 ? 2 : (MT > 16 ? (XSM ? 2 : 3) : (MT > 8 ? (XSM ? 4 : 6) : (LOOK > 1 ? (K > 6 ? 9 : 11) : (K > 6 ? 12 : 16)))); }

template <int K> __device__ __forceinline__ unsigned lno_field(const unsigned* w, int i) {   
  constexpr int FB = 4 * K; const int bit = FB * i, wi = bit >> 5, sh = bit & 31;
  unsigned v = w[wi] >> sh;
  if (sh + FB > 32) v |= w[wi + 1] << (32 - sh);
  if constexpr (FB >= 32) return v; else return v & ((1u << FB) - 1u);
}
// K = 10 (5 bits per weight): the 12-bit table index at bit o of a lane-row stream of 10 words (o + 12 <= 320; o is a constant after unrolling)
__device__ __forceinline__ unsigned lno10_win(const unsigned* r, int o) {
  const int wi = o >> 5, sh = o & 31;
  return (sh <= 20 ? (r[wi] >> sh) : __funnelshift_r(r[wi], r[wi + 1], sh)) & 0xFFFu;
}
__device__ __forceinline__ void lno_mma(int* d, const unsigned* a, uint2 b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ float lno_madd(float acc, float a, float b) { return __fadd_rn(acc, __fmul_rn(a, b)); }   
__device__ __forceinline__ void lno_touch(const uint4* p) { unsigned x; asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(x) : "l"(p)); asm volatile("" :: "r"(x)); }   

__device__ __forceinline__ float lno_xsm_rm(const float* __restrict__ al, int G) { float m = 0.f; for (int k = 0; k < G; ++k) m = fmaxf(m, __ldg(al + k)); return m > 0.f ? __fdiv_rn(16777216.f, m) : 0.f; }
__device__ __forceinline__ float lno_xsm_us(const float* __restrict__ al, int G) { float m = 0.f; for (int k = 0; k < G; ++k) m = fmaxf(m, __ldg(al + k)); return ldexpf(m, -24); }
__device__ __forceinline__ long long lno_xsm_A(float alpha, float rm) { return (long long)__float2int_rn(__fmul_rn(alpha, rm)); }



template <int K>
__global__ void lnord_build_kernel(const uint4* __restrict__ frag, int N, int Kc, uint4* __restrict__ out) {
  constexpr int NU = K / 2;
  const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int G = Kc / 512, KS = Kc / 256; const long long total = (long long)(N / 16) * G * 2 * 32;
  if (id >= total) return;
  const int lane = (int)(id & 31); long long t = id >> 5; const int blk = (int)(t & 1); t >>= 1;
  const int g = (int)(t % G), tile16 = (int)(t / G), gi = lane >> 2, tig = lane & 3, ks = 2 * g + blk;
  unsigned w[2 * K];
  #pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int row = tile16 * 16 + gi + 8 * rr;
    const uint4* base = frag + (size_t)(row >> 6) * KS * 8 * K * 8 + ((size_t)ks * 8 + ((row >> 3) & 7)) * K * 8 + (row & 7);   
    #pragma unroll
    for (int k = 0; k < K; ++k) { const int wi = K * tig + k; const uint4 q = base[(size_t)(wi >> 2) * 8]; const int c = wi & 3;
      w[rr * K + k] = c == 0 ? q.x : c == 1 ? q.y : c == 2 ? q.z : q.w; }
  }
  uint4* o = out + (((size_t)tile16 * G + g) * 2 + blk) * NU * 32 + lane;
  #pragma unroll
  for (int u = 0; u < NU; ++u) o[32 * u] = make_uint4(w[4 * u], w[4 * u + 1], w[4 * u + 2], w[4 * u + 3]);
}
size_t lnord_bytes(int N, int Kc, int K) { return (size_t)N * Kc * K / 16; }
void lnord_build(const uint4* frag, int N, int Kc, int K, uint4* out, cudaStream_t st) {
  const long long total = (long long)(N / 16) * (Kc / 512) * 2 * 32; const unsigned nb = (unsigned)((total + 255) / 256);
  switch (K) {
    case 4: lnord_build_kernel<4><<<nb, 256, 0, st>>>(frag, N, Kc, out); break;
    case 6: lnord_build_kernel<6><<<nb, 256, 0, st>>>(frag, N, Kc, out); break;
    case 8: lnord_build_kernel<8><<<nb, 256, 0, st>>>(frag, N, Kc, out); break;
    case 10: lnord_build_kernel<10><<<nb, 256, 0, st>>>(frag, N, Kc, out); break;
    default: throw std::runtime_error("lnord_build: K must be 4, 6, 8 or 10");
  }
}








__device__ __forceinline__ float uf_bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t uf_f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float uf_rbf(float f) { return uf_bf2f(uf_f2bf(f)); }
__device__ __forceinline__ void uf_fwht(float (&v)[8], float* xch) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  #pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) { const float q = __shfl_xor_sync(0xffffffffu, v[i], h); v[i] = (lane & h) ? __fsub_rn(q, v[i]) : __fadd_rn(v[i], q); }
  }
  #pragma unroll
  for (int hb = 1; hb < 8; hb <<= 1) {                                  
    #pragma unroll
    for (int i = 0; i < 8; ++i) if (!(i & hb)) { const float lo = v[i], hi = v[i | hb]; v[i] = __fadd_rn(lo, hi); v[i | hb] = __fsub_rn(lo, hi); }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) xch[(w * 8 + i) * 32 + lane] = v[i];      
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 8; ++i) { const float q = xch[((w ^ 1) * 8 + i) * 32 + lane]; v[i] = w ? __fsub_rn(q, v[i]) : __fadd_rn(v[i], q); }
  __syncthreads();
}
__device__ __forceinline__ void lno_fin_res_norm(const GemvLaunch& P, long long* acc, int b, float us, float* fs) {
  float* const xch = fs; float* const s_ss = fs + 512; unsigned* const s_am = reinterpret_cast<unsigned*>(fs + 640);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n0 = b * 512 + 256 * w + lane;
  long long a[8]; float tau[8], hh[8]; signed char so[8], si[8]; uint16_t wg[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { const int n = n0 + 32 * i; a[i] = __ldcg(acc + n); tau[i] = __ldg(P.f_tau + n); so[i] = __ldg(P.f_sg_out + n); }
  #pragma unroll
  for (int i = 0; i < 8; ++i) { const int n = n0 + 32 * i; hh[i] = __ldcg(P.f_h + n); wg[i] = __ldg(P.f_w + n); si[i] = __ldg(P.f_sg_in + n); }
  #pragma unroll
  for (int i = 0; i < 8; ++i) __stcg(acc + n0 + 32 * i, 0ll);           
  float v[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __fmul_rn(__fmul_rn(__ll2float_rn(a[i]), us), tau[i]);
  uf_fwht(v, xch);
  #pragma unroll
  for (int i = 0; i < 8; ++i) { const float y = __fmul_rn(__fmul_rn(v[i], 0.044194173824159216f), (float)so[i]);
    v[i] = uf_rbf(__fadd_rn(hh[i], uf_rbf(y))); P.f_h[n0 + 32 * i] = v[i]; }
  if (w == 0) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) s_ss[k * 32 + lane] = __fadd_rn(__fadd_rn(0.f, __fmul_rn(v[k], v[k])), __fmul_rn(v[k + 4], v[k + 4]));
  }
  __syncthreads();
  if (w == 1) {
    float t[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) { float x = s_ss[k * 32 + lane]; x = __fadd_rn(x, __fmul_rn(v[k], v[k])); x = __fadd_rn(x, __fmul_rn(v[k + 4], v[k + 4]));
      #pragma unroll
      for (int o = 16; o; o >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, o));
      t[k] = x; }
    if (lane == 0) { float sb = 0.f;
      #pragma unroll
      for (int k = 0; k < 4; ++k) sb = __fadd_rn(sb, t[k]);
      P.f_S[b] = sb; }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) { const float u = __fmul_rn(v[i], __fadd_rn(1.f, uf_bf2f(wg[i])));   
    if (P.f_xu) P.f_xu[n0 + 32 * i] = uf_f2bf(u);
    v[i] = __fmul_rn(u, (float)si[i]); }
  uf_fwht(v, xch);
  unsigned mx = 0u;
  #pragma unroll
  for (int i = 0; i < 8; ++i) { v[i] = round_f16(__fmul_rn(v[i], 0.044194173824159216f)); mx = max(mx, __float_as_uint(fabsf(v[i]))); }
  #pragma unroll
  for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (lane == 0) s_am[w] = mx;
  __syncthreads();
  const float al = __fdiv_rn(__uint_as_float(max(s_am[0], s_am[1])), 127.f);
  if (threadIdx.x == 0) P.f_alpha[b] = al;
  #pragma unroll
  for (int i = 0; i < 8; ++i) { const float r = al > 0.f ? __fdiv_rn(v[i], al) : 0.f; int q = __float2int_rn(r); q = q < -127 ? -127 : (q > 127 ? 127 : q);
    P.f_qx[n0 + 32 * i] = (int8_t)q; }
}


template <int K, int MT, bool XSM, int LOOK, int NR = 1, bool FIN = false, int UWX = 0, int MBX = 0, bool LO = false>   
__global__ void __launch_bounds__((NR > 1 ? NR : (UWX ? UWX : lno_uw<MT>())) * 32, NR > 1 ? lno_nr_minb<MT, NR>() : (MBX ? MBX : lno_minb<K, MT, XSM, LOOK>())) lnord_kernel(GemvLaunch P) {   
  static_assert(NR == 1 || XSM, "NR > 1: XSUM only");
  static_assert(!FIN || (XSM && NR == 1 && MT == 8), "FIN: XSUM, M = 1");
  static_assert(!LO || (MT == 8 && NR == 1 && !FIN), "LO: M <= 8");
  constexpr int UW = NR > 1 ? NR : (UWX ? UWX : lno_uw<MT>());
  constexpr int NU = K / 2, T = 12 - K, TS = 3 * K - 12;
  constexpr unsigned PM = (1u << (2 * K)) - 1u, TM = (1u << T) - 1u;
  __shared__ __align__(16) uint16_t s_lut[4096];
  int seg = 0; while (seg + 1 < P.n_seg && (int)blockIdx.x >= P.seg[seg + 1].cta_begin) ++seg;
  const GemvSeg& Sg = P.seg[seg]; const TensorDev& t = Sg.t;
  const int warp = threadIdx.x >> 5;
  const int worker = NR > 1 ? (int)blockIdx.y * NR + warp : (int)blockIdx.y;
  if (NR == 1 && worker >= t.n_workers) { pdl_wait(); return; }   
  const int lane = threadIdx.x & 31, gi = lane >> 2, tig = lane & 3;
  const int tile = NR > 1 ? (int)blockIdx.x - Sg.cta_begin : ((int)blockIdx.x - Sg.cta_begin) * UW + warp, ntiles = t.N / 16, N = t.N, Kc = t.Kc, G = Kc / 512, nw = t.n_workers, M = P.M;
  const bool active = tile < ntiles && worker < nw;
  const int ng = (worker < G && worker < nw) ? (G - worker + nw - 1) / nw : 0, nblk = 2 * ng;    
  const uint4* sp = t.lno + (size_t)(active ? tile : 0) * G * 2 * NU * 32 + lane;
  auto boff = [&](int bi) { return (size_t)((worker + (bi >> 1) * nw) * 2 + (bi & 1)) * (NU * 32); };
  uint4 nx[NU], mx[LOOK > 1 ? NU : 1];
  #pragma unroll
  for (int u = 0; u < NU; ++u) nx[u] = make_uint4(0u, 0u, 0u, 0u);
  #pragma unroll
  for (int u = 0; u < (LOOK > 1 ? NU : 1); ++u) mx[u] = make_uint4(0u, 0u, 0u, 0u);
  auto first_w = [&]() {
  if (active && nblk) {                                               
    #pragma unroll
    for (int u = 0; u < NU; ++u) nx[u] = __ldg(sp + boff(0) + 32 * u);
    if (LOOK > 1 && nblk > 1 && !(P.pre & 4)) {                       
      #pragma unroll
      for (int u = 0; u < NU; ++u) mx[u] = __ldg(sp + boff(1) + 32 * u);
    }
    for (int b = LOOK; b < P.prew && b < nblk; ++b)                     
      for (int u = 0; u < NU; ++u) lno_touch(sp + boff(b) + 32 * u);
  } };
  auto lut_copy = [&]() { for (int i = threadIdx.x; i < 512; i += blockDim.x) reinterpret_cast<uint4*>(s_lut)[i] = __ldg(reinterpret_cast<const uint4*>(t.lut8) + i); };
  if (P.pre & 2) first_w();
  if (P.pre & 1) lut_copy();
  pdl_wait(); pdl_trigger();
  if (!(P.pre & 2)) first_w();
  if (!(P.pre & 1)) lut_copy();
  if (LOOK > 1 && (P.pre & 4) && active && nblk > 1) {
    #pragma unroll
    for (int u = 0; u < NU; ++u) mx[u] = __ldg(sp + boff(1) + 32 * u);
  }
  if (XSM && G > 64 && blockIdx.x == 0 && blockIdx.y == 0 && (int)threadIdx.x < M) P.us_out[threadIdx.x] = lno_xsm_us(P.alpha + (size_t)threadIdx.x * G, G);
  __syncthreads();
  if (NR == 1 && !FIN && (!active || nblk == 0)) return;            
  const int nblk_run = active ? nblk : 0;                              
  constexpr int NB = (MT + 7) / 8;                                    
  const int src = (lane & ~3) | ((tig + 3) & 3);                       
  float pacc[NB][4]; long long acc64[XSM ? NB : 1][4]; int d[NB][4]; float rm[NB][2];
  #pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    #pragma unroll
    for (int q = 0; q < 4; ++q) { pacc[nb][q] = 0.f; d[nb][q] = 0; if (XSM) acc64[nb][q] = 0; }
    if (XSM && G > 64) {
      #pragma unroll
      for (int k = 0; k < 2; ++k) { const int tk = 8 * nb + 2 * tig + k; rm[nb][k] = tk < M ? lno_xsm_rm(P.alpha + (size_t)tk * G, G) : 0.f; }
    }
  }
  float fin_us = 0.f; (void)fin_us;                                    
  if (XSM && G <= 64 && MT > 16) {
    // exact sums for 32 or 64 rows (passes of 17..64 rows): lane l takes rows l and l + 32; the same max and scales as below
    float m0 = 0.f, m1 = 0.f;
    if (lane < M) { const float* al = P.alpha + (size_t)lane * G; for (int gx = 0; gx < G; ++gx) m0 = fmaxf(m0, __ldg(al + gx)); }
    if (MT > 32 && lane + 32 < M) { const float* al = P.alpha + (size_t)(lane + 32) * G; for (int gx = 0; gx < G; ++gx) m1 = fmaxf(m1, __ldg(al + gx)); }
    const float r0 = m0 > 0.f ? __fdiv_rn(16777216.f, m0) : 0.f, r1 = m1 > 0.f ? __fdiv_rn(16777216.f, m1) : 0.f;
    if (blockIdx.x == 0 && blockIdx.y == 0 && warp == 0) { if (lane < M) P.us_out[lane] = ldexpf(m0, -24); if (MT > 32 && lane + 32 < M) P.us_out[lane + 32] = ldexpf(m1, -24); }
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      #pragma unroll
      for (int k = 0; k < 2; ++k) { const int tk = 8 * nb + 2 * tig + k; const float x = __shfl_sync(0xffffffffu, nb < 4 ? r0 : r1, tk & 31); rm[nb][k] = tk < M ? x : 0.f; }
    }
  } else if (XSM && G <= 64) {
    
    
    
    
    constexpr int TM = MT < 16 ? 8 : 16, PP = 32 / TM, NV = 40 / PP;
    const int tkn = lane % TM, part = lane / TM;
    float m = 0.f;
    if (tkn < M) {
      const float* al = P.alpha + (size_t)tkn * G;
      if (G <= 40) {
        float v[NV];
        #pragma unroll
        for (int k = 0; k < NV; ++k) { const int gx = part + k * PP; v[k] = gx < G ? __ldg(al + gx) : 0.f; }
        #pragma unroll
        for (int k = 0; k < NV; ++k) m = fmaxf(m, v[k]);
      } else for (int gx = part; gx < G; gx += PP) m = fmaxf(m, __ldg(al + gx));
    }
    #pragma unroll
    for (int o = TM; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));   
    const float rmv = m > 0.f ? __fdiv_rn(16777216.f, m) : 0.f;
    if constexpr (FIN) fin_us = ldexpf(__shfl_sync(0xffffffffu, m, 0), -24);   
    if (blockIdx.x == 0 && blockIdx.y == 0 && warp == 0 && part == 0 && tkn < M) P.us_out[tkn] = ldexpf(m, -24);
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      #pragma unroll
      for (int k = 0; k < 2; ++k) { const int tk = 8 * nb + 2 * tig + k; const float x = __shfl_sync(0xffffffffu, rmv, tk & 31); rm[nb][k] = tk < M ? x : 0.f; }
    }
  }
  auto pair_a = [&](unsigned p, unsigned q) -> unsigned {             
    const unsigned W = q | (p << T); return unsigned(s_lut[W & 0xFFFu]) | (unsigned(s_lut[(W >> K) & 0xFFFu]) << 16); };
  for (int bi = 0; bi < nblk_run; ++bi) {
    const int g = worker + (bi >> 1) * nw, blk = bi & 1;
    unsigned w[4 * NU];                                               
    #pragma unroll
    for (int u = 0; u < NU; ++u) { w[4 * u] = nx[u].x; w[4 * u + 1] = nx[u].y; w[4 * u + 2] = nx[u].z; w[4 * u + 3] = nx[u].w; }
    if (LOOK == 1) { if (bi + 1 < nblk_run) {
      #pragma unroll
      for (int u = 0; u < NU; ++u) nx[u] = __ldg(sp + boff(bi + 1) + 32 * u); } }
    else {
      #pragma unroll
      for (int u = 0; u < NU; ++u) nx[u] = mx[u];
      if (bi + 2 < nblk_run) {
        #pragma unroll
        for (int u = 0; u < NU; ++u) mx[u] = __ldg(sp + boff(bi + 2) + 32 * u); } }
    const uint4* xp[NB]; uint4 xb[NB];                                  
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) { const int tok = gi + 8 * nb;
      xp[nb] = tok < M ? reinterpret_cast<const uint4*>(LO ? P.qx + (size_t)g * 4096 + blk * 2048 + (tok + P.lo_t0) * 64 + 16 * tig   
                                                          : P.qx + (size_t)tok * Kc + (size_t)g * 512 + blk * 256 + 64 * tig) : nullptr; }
    const unsigned e = (w[K - 1] >> (32 - T)) | ((w[2 * K - 1] >> (32 - T)) << T);   
    const unsigned nbv = __shfl_sync(0xffffffffu, e, src);
    unsigned q0 = tig ? (nbv & TM) : 0u, q1 = tig ? (nbv >> T) : 0u;   
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
      unsigned A[4];
      if constexpr (K == 10) {   // K = 10: indices = 12-bit windows at 40m-2+10j (j = 0..3) of each row's stream; m = 0, j = 0 takes the carry
        const unsigned* r1 = w + K;
        const unsigned a0 = m ? lno10_win(w, 40 * m - 2) : (q0 | ((w[0] & 0x3FFu) << 2)), b0 = m ? lno10_win(r1, 40 * m - 2) : (q1 | ((r1[0] & 0x3FFu) << 2));
        A[0] = unsigned(s_lut[a0]) | (unsigned(s_lut[lno10_win(w, 40 * m + 8)]) << 16);
        A[2] = unsigned(s_lut[lno10_win(w, 40 * m + 18)]) | (unsigned(s_lut[lno10_win(w, 40 * m + 28)]) << 16);
        A[1] = unsigned(s_lut[b0]) | (unsigned(s_lut[lno10_win(r1, 40 * m + 8)]) << 16);
        A[3] = unsigned(s_lut[lno10_win(r1, 40 * m + 18)]) | (unsigned(s_lut[lno10_win(r1, 40 * m + 28)]) << 16);
      } else {
      const unsigned f0 = lno_field<K>(w, m), f1 = lno_field<K>(w + K, m);   
      const unsigned p10 = f0 & PM, p20 = f0 >> (2 * K), p11 = f1 & PM, p21 = f1 >> (2 * K);
      A[0] = pair_a(p10, q0); A[2] = pair_a(p20, p10 >> TS); A[1] = pair_a(p11, q1); A[3] = pair_a(p21, p11 >> TS);
      q0 = p20 >> TS; q1 = p21 >> TS;
      }
      #pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        if ((m & 1) == 0) xb[nb] = xp[nb] ? __ldg(xp[nb] + (LO ? (m >> 1) * 32 : (m >> 1))) : make_uint4(0u, 0u, 0u, 0u);   
        const uint4& v = xb[nb];
        lno_mma(d[nb], A, (m & 1) ? make_uint2(v.z, v.w) : make_uint2(v.x, v.y));
      }
    }
    if (blk == 1) {                                                   
      #pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const int t0 = 8 * nb + 2 * tig, t1 = t0 + 1;
        const float a0 = t0 < M ? __ldg(P.alpha + (size_t)t0 * G + g) : 0.f, a1 = t1 < M ? __ldg(P.alpha + (size_t)t1 * G + g) : 0.f;
        if (XSM) {
          const long long A0 = lno_xsm_A(a0, rm[nb][0]), A1 = lno_xsm_A(a1, rm[nb][1]);
          acc64[nb][0] += A0 * (long long)d[nb][0]; acc64[nb][2] += A0 * (long long)d[nb][2];
          acc64[nb][1] += A1 * (long long)d[nb][1]; acc64[nb][3] += A1 * (long long)d[nb][3];
        } else {
          if (t0 < M) { pacc[nb][0] = lno_madd(pacc[nb][0], a0, (float)d[nb][0]); pacc[nb][2] = lno_madd(pacc[nb][2], a0, (float)d[nb][2]); }
          if (t1 < M) { pacc[nb][1] = lno_madd(pacc[nb][1], a1, (float)d[nb][1]); pacc[nb][3] = lno_madd(pacc[nb][3], a1, (float)d[nb][3]); }
        }
        d[nb][0] = d[nb][1] = d[nb][2] = d[nb][3] = 0;
      }
    }
  }
  const int n = tile * 16 + gi;
  if constexpr (XSM && NR > 1) {                                      
    __shared__ long long s_red[NR][NB * 4][32];
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      #pragma unroll
      for (int q = 0; q < 4; ++q) s_red[warp][nb * 4 + q][lane] = acc64[nb][q];
    }
    __syncthreads();
    if (warp != 0 || tile >= ntiles) return;
    unsigned long long* p64 = reinterpret_cast<unsigned long long*>(Sg.part);
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      const int t0 = 8 * nb + 2 * tig, t1 = t0 + 1; long long sm[4];
      #pragma unroll
      for (int q = 0; q < 4; ++q) { long long v = 0;
        #pragma unroll
        for (int w = 0; w < NR; ++w) v += s_red[w][nb * 4 + q][lane];
        sm[q] = v; }
      if (t0 < M) { atomicAdd(p64 + (size_t)t0 * N + n, (unsigned long long)sm[0]); atomicAdd(p64 + (size_t)t0 * N + n + 8, (unsigned long long)sm[2]); }
      if (t1 < M) { atomicAdd(p64 + (size_t)t1 * N + n, (unsigned long long)sm[1]); atomicAdd(p64 + (size_t)t1 * N + n + 8, (unsigned long long)sm[3]); }
    }
    return;
  }
  if (!FIN || active) {
  #pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    const int t0 = 8 * nb + 2 * tig, t1 = t0 + 1;
    if (XSM) { unsigned long long* p64 = reinterpret_cast<unsigned long long*>(Sg.part);
      if (P.xsm_part) {                                               
        if (t0 < M) { unsigned long long* q = p64 + ((size_t)worker * M + t0) * N + n; q[0] = (unsigned long long)acc64[nb][0]; q[8] = (unsigned long long)acc64[nb][2]; }
        if (t1 < M) { unsigned long long* q = p64 + ((size_t)worker * M + t1) * N + n; q[0] = (unsigned long long)acc64[nb][1]; q[8] = (unsigned long long)acc64[nb][3]; }
      } else {
      if (t0 < M) { atomicAdd(p64 + (size_t)t0 * N + n, (unsigned long long)acc64[nb][0]); atomicAdd(p64 + (size_t)t0 * N + n + 8, (unsigned long long)acc64[nb][2]); }
      if (t1 < M) { atomicAdd(p64 + (size_t)t1 * N + n, (unsigned long long)acc64[nb][1]); atomicAdd(p64 + (size_t)t1 * N + n + 8, (unsigned long long)acc64[nb][3]); }
      }
    } else {
      if (t0 < M) { float* p = Sg.part + ((size_t)worker * M + t0) * N + n; p[0] = pacc[nb][0]; p[8] = pacc[nb][2]; }
      if (t1 < M) { float* p = Sg.part + ((size_t)worker * M + t1) * N + n; p[0] = pacc[nb][1]; p[8] = pacc[nb][3]; }
    }
  }
  }
  if constexpr (FIN) {                                                
    if (P.f_mode == 3) return;                                        
    __threadfence();                                                  
    __syncthreads();                                                  
    int* const s_last = reinterpret_cast<int*>(s_lut) + 1000;
    const int b = ((int)blockIdx.x - Sg.cta_begin) * UW / 32, nwe = nw < G ? nw : G;
    if (threadIdx.x == 0) *s_last = worker < nwe && atomicAdd(P.f_cnt + b, 1u) == (unsigned)(32 / UW * nwe - 1);
    __syncthreads();
    if (!*s_last) return;
    if (P.f_mode == 2) { if (threadIdx.x == 0) P.f_cnt[b] = 0u; return; }   
    __threadfence();
    lno_fin_res_norm(P, reinterpret_cast<long long*>(Sg.part), b, fin_us, reinterpret_cast<float*>(s_lut));
    if (threadIdx.x == 0) P.f_cnt[b] = 0u;
  }
}

static int lnord_k(const GemvLaunch& P, bool xw = false) {          
  if (P.draft || P.e_fuse || P.M < 1 || P.M > 64 || (P.xsum && P.M > 16 && (!xw || P.M > 32 || P.xsm_part || P.lo)) || P.n_seg < 1) return 0;
  if (P.f_mode && (P.M != 1 || !P.xsum || P.xsm_part || P.n_seg != 1 || P.seg[0].t.N % 512 || P.seg[0].t.Kc / 512 > 64)) return 0;
  const int K = P.seg[0].t.k_default;
  // K = 10 only for exact-sum passes with atomic partials: 1..16 rows (LO 1..8, FIN 1, else the 16-row shape)
  // and the wide pass of 17..32 rows (xw, the 32-row shape); anything else is not eligible, and every caller then throws
  if (K != 4 && K != 6 && K != 8 && !(K == 10 && P.xsum && !P.xsm_part && (P.M <= 16 || (xw && P.M <= 32 && !P.lo)))) return 0;
  for (int s = 0; s < P.n_seg; ++s) { const TensorDev& t = P.seg[s].t;
    if (!t.lno || t.k_default != K || t.tile_k || t.group_worker || (t.N % 16) || (t.Kc % 512)) return 0; }
  return K;
}
bool lnord_eligible(const GemvLaunch& P) { return lnord_k(P) != 0; }
bool lnord_xw_eligible(const GemvLaunch& P) { return P.xsum && P.M > 16 && lnord_k(P, true) != 0; }
static int lno_prew_env() { static const int v = [] { const char* e = std::getenv("YSKN_LNO_PREW"); return e ? std::atoi(e) : 1; }(); return v; }
static int lno_look_env() { static const int v = [] { const char* e = std::getenv("YSKN_LNO_LOOK"); return e ? std::atoi(e) : 0; }(); return v; }   
static int lno_pre_env() { static const int v = [] { const char* e = std::getenv("YSKN_LNO_PRE"); return e ? std::atoi(e) : 3; }(); return v; }   
static int lno_nr_env() { static const int v = [] { const char* e = std::getenv("YSKN_LNO_XSMR"); return e ? std::atoi(e) : 1; }(); return v; }   
template <int K, int MT, int NR> static void launch_lno_nr(const GemvLaunch& P0, cudaStream_t st) {   
  GemvLaunch P = P0; P.prew = 1; int ctas = 0;
  for (int s = 0; s < P.n_seg; ++s) { P.seg[s].cta_begin = ctas; ctas += P.seg[s].t.N / 16; }
  launch_k(lnord_kernel<K, MT, true, 1, NR>, dim3(ctas, (P.max_workers + NR - 1) / NR), dim3(NR * 32), 0, st, P);
}
template <int K, int MT, int UWX = 0, int MBX = 0, bool LO = false> static void launch_lno(const GemvLaunch& P0, cudaStream_t st) {
  constexpr int uw = UWX ? UWX : lno_uw<MT>();
  GemvLaunch P = P0; P.prew = lno_prew_env(); P.pre = lno_pre_env(); int ctas = 0, maxblk = 0;                         
  for (int s = 0; s < P.n_seg; ++s) { P.seg[s].cta_begin = ctas; ctas += (P.seg[s].t.N / 16 + uw - 1) / uw;
    const int G = P.seg[s].t.Kc / 512, nw = P.seg[s].t.n_workers; maxblk = std::max(maxblk, 2 * ((G + nw - 1) / nw)); }
  const dim3 grid(ctas, P.max_workers), block(uw * 32);
  (void)maxblk; const int look = lno_look_env() ? lno_look_env() : 1;      
  if constexpr (MT > 16) {
    if (P.xsum) { if constexpr (MT == 32) launch_k(lnord_kernel<K, 32, true, 1>, grid, block, 0, st, P); else throw std::runtime_error("launch_lnord: exact sums up to 32 rows"); }
    else launch_k(lnord_kernel<K, MT, false, 1>, grid, block, 0, st, P);
    return; }   
  else {
  if constexpr (MT == 8 && UWX == 0) { if (P.f_mode) { launch_k(lnord_kernel<K, 8, true, 1, 1, true>, grid, block, 0, st, P); return; } }   
  if (!LO && UWX == 0 && P.xsum && !P.xsm_part && lno_nr_env() > 1) { const int nr = lno_nr_env();   
    if (nr >= 8) launch_lno_nr<K, MT, 8>(P0, st); else if (nr >= 4) launch_lno_nr<K, MT, 4>(P0, st); else launch_lno_nr<K, MT, 2>(P0, st); }
  else if (P.xsum) { if (look == 2 && !LO) launch_k(lnord_kernel<K, MT, true, 2, 1, false, UWX, MBX>, grid, block, 0, st, P); else launch_k(lnord_kernel<K, MT, true, 1, 1, false, UWX, MBX, LO>, grid, block, 0, st, P); }
  else          { if (look == 2) launch_k(lnord_kernel<K, MT, false, 2, 1, false, UWX, MBX>, grid, block, 0, st, P); else launch_k(lnord_kernel<K, MT, false, 1, 1, false, UWX, MBX>, grid, block, 0, st, P); }
  }
}



// width invariance: M <= 16 always uses the 16-row shape (M must not change the arithmetic)
static int lno_mt16_env() { static const int v = [] { const char* e = std::getenv("YSKN_LNO_MT16");
  return e && e[0] ? std::atoi(e) : 0; }(); return v; }
static int lno_cfg_env() { static const int v = [] { const char* e = std::getenv("YSKN_LNO_CFG"); return e && e[0] ? std::atoi(e) : -1; }(); return v; }
static int lno_cfg_of(int M) { static const int tab[9] = {0, 1, 2, 1, 2, 2, 2, 2, 2}; const int f = lno_cfg_env(); return f >= 0 ? f : (M <= 8 ? tab[M] : 0); }
template <int K> static void launch_lno_m(const GemvLaunch& P, cudaStream_t st) {
  const int cfg = P.lno_cfg >= 0 ? P.lno_cfg : lno_cfg_of(P.M);
  if (P.M <= 8 && P.lo) {                                              
    if (!P.xsum || P.f_mode) throw std::runtime_error("launch_lnord: LO needs a XSUM launch without FIN");
    if (cfg == 1) launch_lno<K, 8, 2, 11, true>(P, st); else if (cfg == 2) launch_lno<K, 8, 4, 8, true>(P, st); else launch_lno<K, 8, 0, 0, true>(P, st);
    return; }
  if (lno_mt16_env() && P.M <= 16 && !P.lo) { launch_lno<K, 16>(P, st); return; }
  if (P.M <= 8 && cfg == 1) { launch_lno<K, 8, 2, 11>(P, st); return; }
  if (P.M <= 8 && cfg == 2) { launch_lno<K, 8, 4, 8>(P, st); return; }
  if (P.M <= 8) launch_lno<K, 8>(P, st); else if (P.M <= 16) launch_lno<K, 16>(P, st); else if (P.M <= 32) launch_lno<K, 32>(P, st); else launch_lno<K, 64>(P, st);
}
// The K = 10 shapes.  The integer sums do not depend on the shape.
constexpr int LO10_MB2 = 6;                                          // blocks per SM of the 2..8-row LO shape (4 warps)
template <int MT, int UWX, int MBX, bool LO, bool FIN> static void launch_lno10_t(const GemvLaunch& P0, cudaStream_t st) {
  constexpr int uw = UWX ? UWX : lno_uw<MT>();
  GemvLaunch P = P0; P.prew = lno_prew_env(); P.pre = lno_pre_env(); int ctas = 0;
  for (int s = 0; s < P.n_seg; ++s) { P.seg[s].cta_begin = ctas; ctas += (P.seg[s].t.N / 16 + uw - 1) / uw; }
  launch_k(lnord_kernel<10, MT, true, 1, 1, FIN, UWX, MBX, LO>, dim3(ctas, P.max_workers), dim3(uw * 32), 0, st, P);
}
static void launch_lno10(const GemvLaunch& P, cudaStream_t st) {
  if (!P.xsum || P.xsm_part || P.M < 1 || P.M > 32) throw std::runtime_error("launch_lnord: K = 10 needs an exact-sum pass of 1..32 rows");
  if (lno_look_env() > 1 || lno_nr_env() > 1) throw std::runtime_error("launch_lnord: K = 10 has no YSKN_LNO_LOOK / YSKN_LNO_XSMR shapes");
  if (P.f_mode) { if (P.M != 1 || P.lo) throw std::runtime_error("launch_lnord: K = 10 FIN needs one row");
    launch_lno10_t<8, 0, 0, false, true>(P, st); return; }
  if (P.lo) { if (P.M > 8) throw std::runtime_error("launch_lnord: LO needs <= 8 rows");
    const int cfg = P.lno_cfg >= 0 ? P.lno_cfg : lno_cfg_of(P.M);
    if (cfg == 2) launch_lno10_t<8, 4, LO10_MB2, true, false>(P, st); else launch_lno10_t<8, 2, 11, true, false>(P, st);   // cfg 0 / 1: the 2-warp shape
    return; }
  if (P.M > 16) { launch_lno10_t<32, 0, 0, false, false>(P, st); return; }   // 17..32 rows: the wide pass (= lnord_kernel<K, 32, true, 1>)
  launch_lno10_t<16, 0, 0, false, false>(P, st);                     // 1..16 rows without LO
}
void launch_lnord_xw(const GemvLaunch& P, cudaStream_t st) {
  if (!lnord_xw_eligible(P)) throw std::runtime_error("launch_lnord_xw: launch not eligible");
  switch (lnord_k(P, true)) {
    case 4: launch_lno_m<4>(P, st); break;
    case 6: launch_lno_m<6>(P, st); break;
    case 8: launch_lno_m<8>(P, st); break;
    case 10: launch_lno10(P, st); break;
    default: throw std::runtime_error("launch_lnord_xw: K must be 4, 6, 8 or 10");
  }
}
void launch_lnord(const GemvLaunch& P, cudaStream_t st) {
  switch (lnord_k(P)) {
    case 4: launch_lno_m<4>(P, st); break;
    case 6: launch_lno_m<6>(P, st); break;
    case 8: launch_lno_m<8>(P, st); break;
    case 10: launch_lno10(P, st); break;
    default: throw std::runtime_error("launch_lnord: launch not eligible");
  }
}
} 
