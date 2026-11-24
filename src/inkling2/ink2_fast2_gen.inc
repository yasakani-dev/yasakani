// SPDX-License-Identifier: Apache-2.0
// Fast kernels (second set), CUDA side (see ink2_fast2.h). Included at the end of ink2_cuda.cu, so it is compiled with the
// flags of the general kernels (-fmad=false) and wraps their bodies kb_<name> unchanged: the wrapped kernels give the
// same bits as the kernels they replace. Only the order of things around the body changes:
//   [trigger the dependents] -> [prefetch the block's own weight bytes into L2] -> griddepcontrol.wait -> [trigger]
//   -> the body (activations are read only here).
// The weight bytes never change, so reading them before the wait is safe. Nothing that a kernel before writes is read
// before the wait, except where noted (fx2c in ink2_fast.cu reads the router choice of the kernel two launches back,
// with ld.global.cg, after the kernel in between passed its own wait).
#include "ink2_fast2.h"

#include <vector>  // (the check of the split attention, ink2_f2_attn_check)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define F2_SM90 1
#else
#define F2_SM90 0
#endif

__device__ __forceinline__ void f2_wait() {
#if F2_SM90
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void f2_trigger() {
#if F2_SM90
  asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}
__device__ __forceinline__ void f2_pf_line(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ void f2_pf_bulk(const void* p, unsigned n) {
#if F2_SM90
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(n) : "memory");
#else
  f2_pf_line(p);
#endif
}
// [p, p + n) into L2 by the threads of the block. mode 1: one prefetch per 128-byte line; mode 2: bulk pieces of up
// to 16 KB (16-byte aligned start, size rounded up to 16: the buffers are allocated in 256-byte units).
__device__ __forceinline__ void f2_pf_span(const void* p, size_t n, int mode) {
  if (!p || !n) return;
  const size_t a0 = (size_t)p & ~(size_t)127, a1 = (size_t)p + n;
  if (mode == 2) {
    const size_t b0 = (size_t)p & ~(size_t)15;
    for (size_t o = b0 + (size_t)threadIdx.x * 16384; o < a1; o += (size_t)blockDim.x * 16384) {
      size_t m = a1 - o < 16384 ? a1 - o : 16384;
      f2_pf_bulk((const void*)o, (unsigned)((m + 15) & ~(size_t)15));
    }
    return;
  }
  for (size_t o = a0 + (size_t)threadIdx.x * 128; o < a1; o += (size_t)blockDim.x * 128) f2_pf_line((const void*)o);
}
__device__ __forceinline__ size_t f2_rowb(const GemvJob& J) {
  return J.fmt == INK2_DFMT_NVFP4 ? (size_t)(J.cols >> 1) : J.fmt == INK2_DFMT_MXFP8 ? (size_t)J.cols : (size_t)J.cols * 2;
}
__device__ __forceinline__ size_t f2_scb(const GemvJob& J) {
  return J.fmt == INK2_DFMT_NVFP4 ? (size_t)(J.cols >> 4) : J.fmt == INK2_DFMT_MXFP8 ? (size_t)(J.cols >> 5) : 0;
}
__device__ __forceinline__ void f2_pf_rows(const GemvJob& J, int row0, int n, int mode) {
  if (n <= 0) return;
  const size_t rb = f2_rowb(J), sb = f2_scb(J);
  f2_pf_span((const u8*)J.w + (size_t)row0 * rb, (size_t)n * rb, mode);
  if (sb && J.sc) f2_pf_span(J.sc + (size_t)row0 * sb, (size_t)n * sb, mode);
}
// chunk b of NR rows of the jobs (the block mapping of pgemvc / gemvc / pgemvcm / gemvcm / fgemvm)
__device__ __forceinline__ void f2_pf_chunk(const GemvArgs& a, int b, int NR, int mode) {
  int j = 0;
  for (;;) {
    if (j >= a.njobs) return;
    int nb = (a.job[j].rows + NR - 1) / NR;
    if (b < nb) break;
    b -= nb;
    ++j;
  }
  const int row0 = b * NR;
  int n = a.job[j].rows - row0;
  if (n > NR) n = NR;
  f2_pf_rows(a.job[j], row0, n, mode);
}
// rows [t0, t1) of the rows of all jobs one after the other (gemv / pgemv / gemvw / pgemvw / pgemvn mapping)
__device__ __forceinline__ void f2_pf_trows(const GemvArgs& a, int t0, int t1, int mode) {
  int off = 0;
  for (int j = 0; j < a.njobs && off < t1; ++j) {
    const int lo = t0 > off ? t0 : off, hi = t1 < off + a.job[j].rows ? t1 : off + a.job[j].rows;
    if (lo < hi) f2_pf_rows(a.job[j], lo - off, hi - lo, mode);
    off += a.job[j].rows;
  }
}
// the weight bytes of block b of kernel KID (only the first set of blocks: further right-hand sides read the same)
template <int KID>
__device__ __forceinline__ void f2_pf_block(const GemvArgs& a, int b, int mode) {
  const bool multi = a.nrhs > 1;
  switch (KID) {
    case KID_pgemvc: if (multi && b >= a.blocks) return; f2_pf_chunk(a, b, 16, mode); return;
    case KID_pgemvc8: if (multi && b >= a.blocks) return; f2_pf_chunk(a, b, 8, mode); return;
    case KID_pgemvc4: if (multi && b >= a.blocks) return; f2_pf_chunk(a, b, 4, mode); return;
    case KID_pgemvc2: if (multi && b >= a.blocks) return; f2_pf_chunk(a, b, 2, mode); return;
    case KID_pgemvcm: if (b >= a.blocks) return; f2_pf_chunk(a, b, 16, mode); return;
    case KID_pgemvcm8: if (b >= a.blocks) return; f2_pf_chunk(a, b, 8, mode); return;
    case KID_pgemvcm4: if (b >= a.blocks) return; f2_pf_chunk(a, b, 4, mode); return;
    case KID_pgemvcm2: if (b >= a.blocks) return; f2_pf_chunk(a, b, 2, mode); return;
    case KID_gemvc: f2_pf_chunk(a, b, K_GEMVC_ROWS, mode); return;
    case KID_gemvcm: if (b >= a.blocks) return; f2_pf_chunk(a, b, K_GEMVC_ROWS, mode); return;
    case KID_pgemv: if (multi && b >= a.blocks) return; f2_pf_trows(a, b * 8, b * 8 + 8, mode); return;
    case KID_gemv: f2_pf_trows(a, b * 8, b * 8 + 8, mode); return;
    case KID_pgemvw: if (multi && b >= a.blocks) return; f2_pf_trows(a, b, b + 1, mode); return;
    case KID_gemvw: f2_pf_trows(a, b, b + 1, mode); return;
    case KID_gemvwm: if (b >= a.blocks) return; f2_pf_trows(a, b, b + 1, mode); return;
    case KID_pgemvn: {
      if (multi && b >= a.blocks) return;
      const int lpr = a.job[0].cols >> 3, per = 8 * (32 / lpr);
      f2_pf_trows(a, b * per, b * per + per, mode);
      return;
    }
    default: return;
  }
}
template <int KID>
__device__ __forceinline__ void f2_pf_gemv(const GemvArgs& a, const F2Launch& l) {
  const int b = (int)blockIdx.x;
  f2_pf_block<KID>(a, b, l.pf);
  if (l.extra > 0 && b < l.extra)
    for (int k = 1; k <= l.rounds; ++k) {
      const int bb = b + k * l.extra;
      if (bb >= (int)gridDim.x) break;
      f2_pf_block<KID>(a, bb, l.pf);
    }
}
__device__ __forceinline__ void f2_enter(const F2Launch& l) {
  f2_wait();
  if (!l.trig) f2_trigger();
}

#define F2_WRAP_GEMV(name)                       \
  __global__ void kq_##name(GemvArgs a, F2Launch l) { \
    if (l.trig) f2_trigger();                     \
    if (l.pf) f2_pf_gemv<KID_##name>(a, l);       \
    f2_enter(l);                                  \
    kb_##name(a);                                 \
  }
F2_WRAP_GEMV(pgemvc)
F2_WRAP_GEMV(pgemvc8)
F2_WRAP_GEMV(pgemvc4)
F2_WRAP_GEMV(pgemvc2)
F2_WRAP_GEMV(pgemvcm)
F2_WRAP_GEMV(pgemvcm8)
F2_WRAP_GEMV(pgemvcm4)
F2_WRAP_GEMV(pgemvcm2)
F2_WRAP_GEMV(gemvc)
F2_WRAP_GEMV(gemvcm)
F2_WRAP_GEMV(pgemv)
F2_WRAP_GEMV(gemv)
F2_WRAP_GEMV(pgemvw)
F2_WRAP_GEMV(gemvw)
F2_WRAP_GEMV(gemvwm)
F2_WRAP_GEMV(pgemvn)
#undef F2_WRAP_GEMV
// small kernels that may trigger their dependents first (the weight kernel after them becomes resident earlier)
#define F2_WRAP_SMALL(name, A)                \
  __global__ void kq_##name(A a, F2Launch l) { \
    if (l.trig) f2_trigger();                 \
    f2_enter(l);                              \
    kb_##name(a);                             \
  }
// attn: before the wait, the cached keys / values of the earlier positions into L2 (the newest one is written by
// prep, the kernel before; it is not touched here). One block per kv head (the first query head of the group), row 0.
__global__ void kq_attn(AttnArgs a, F2Launch l) {
  if (l.trig) f2_trigger();
  if (l.pf) {
    int h = KBID, row = 0;
    if (a.rows > 0) {
      row = KBID / a.nh;
      h = KBID - row * a.nh;
    }
    const int grp = a.nh / a.nkv;
    if (row == 0 && h % grp == 0) {
      const int kh = h / grp, C = a.nkv * a.hd;
      const int pos = a.st[0].pos;
      int T = pos + 1;
      const int win = a.window > 0 ? a.window : a.ring;
      if (win && T > win) T = win;
      int slot0 = pos + 1 - T;
      if (a.ring) slot0 = slot0 % a.ring;
      const int lines = (a.hd * 4 + 127) / 128;
      for (int t = KTID; t < (T - 1) * lines; t += KNT) {
        const int key = t / lines, li = t - key * lines;
        int slot = slot0 + key;
        if (a.ring && slot >= a.ring) slot -= a.ring;
        const size_t off = (size_t)slot * C + (size_t)kh * a.hd + (size_t)li * 32;
        f2_pf_line(a.kc + off);
        f2_pf_line(a.vc + off);
      }
    }
  }
  f2_enter(l);
  kb_attn(a);
}
F2_WRAP_SMALL(norm, NormArgs)
F2_WRAP_SMALL(gmix, GmixArgs)
F2_WRAP_SMALL(prep, PrepArgs)
F2_WRAP_SMALL(topk, RouterArgs)
F2_WRAP_SMALL(egroup, GroupArgs)
#undef F2_WRAP_SMALL

// ---------------------------------------------------------------------------
// Attention at long context (INK2_F2_ATTN, ink2_fast2.h): kb_attn spread over more blocks, the same
// bits. kb_attn runs one block of 1024 threads per (row, head): warp w takes the keys t = w, w + 32, ... (lane l the
// elements 4l .. 4l+3 of the head) for the logits and again for its value stripe; thread i adds the exp of the keys
// t = i, i + 1024, ...; k_block_sum adds those sums (k_warp_sum per warp, the warps in order); warp 0 adds the 32
// stripes in order. Every logit and every stripe is one warp's own chain, so they can be spread over blocks:
//   kf2_attn_score (grid rows * nkv * sa, 128 threads): the logits of a range of keys, 8 consecutive keys per warp at a
//     time (their key rows loaded first), for all query heads of the kv head (a key row is read once for the group).
//     Per key and head the arithmetic of kb_attn: the FMA chain of lane l over its 4 elements from 0, times the scale,
//     the relative bias in lanes < drel, the warp sum (k_warp_sum8t: the tree of k_warp_sum, bit for bit). The logits
//     go to the scratch rows as in kb_attn, the largest of the block to pmax.
//   kf2_attn_value (grid rows * nkv * 32 / sw, sw warps): warp w = stripe w for all query heads of the kv head.
//     m = max(the block maxima, sink) (a maximum: any order), e = expf(logit - m) (written back to the scratch rows as
//     kb_attn does), the FMA chain of the stripe over its keys in order from 0 (value rows in batches of 16, the next
//     batch and the next 32 logits in flight while one batch is added). Lane l of stripe w holds the e of the keys w + 32 (32 k + l), k = 0, 1, ...: its sum of them (in order,
//     from 0) is the sum of thread w + 32 l of kb_attn. The last value block of a (row, kv head) then does, per head,
//     k_block_sum on those 1024 sums plus expf(sink - m), adds the 32 stripes in order c = 0 .. 31, divides and gates:
//     the steps of kb_attn after its loops.
// ---------------------------------------------------------------------------
template <int GQ, int NG>
__global__ void __launch_bounds__(128) kf2_attn_score(F2AttnArgs p, F2Launch l) {
  __shared__ float smx[4 * GQ];
  if (l.trig) f2_trigger();
  const AttnArgs& a = p.a;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int b = (int)blockIdx.x;
  const int s = b % p.sa;
  b /= p.sa;
  const int kh = b % a.nkv, row = b / a.nkv;
  const int hd = a.hd, C = a.nkv * hd, h0 = kh * GQ;
  const int pos = a.st[row].pos;  // (uploaded before the launch, not written by a kernel)
  int T = pos + 1;
  const int win = a.window > 0 ? a.window : a.ring;
  if (win && T > win) T = win;
  int slot0 = pos + 1 - T;
  if (a.ring) slot0 = slot0 % a.ring;
  const int ng = (T + 7) >> 3;  // groups of 8 keys
  const int per = (ng + p.sa - 1) / p.sa;
  const int g0 = s * per;
  const int g1 = g0 + per < ng ? g0 + per : ng;
  if (p.pf & 3) {  // before the wait: the key / value rows of this block into L2 (not the newest one: prep writes it)
    const int t1 = g1 * 8 < T - 1 ? g1 * 8 : T - 1;
    const int lines = (hd * 4 + 127) / 128;
    for (int x = threadIdx.x; x < (t1 - g0 * 8) * lines; x += blockDim.x) {
      const int t = g0 * 8 + x / lines, li = x % lines;
      int slot = slot0 + t;
      if (a.ring && slot >= a.ring) slot -= a.ring;
      const size_t off = (size_t)slot * C + (size_t)kh * hd + (size_t)li * 32;
      if (p.pf & 1) f2_pf_line(a.kc + off);
      if (p.pf & 2) f2_pf_line(a.vc + off);
    }
  }
  f2_enter(l);
  const size_t qo = (size_t)row * a.nh * hd;
  KF4 q[GQ];
  float rr[GQ], mx[GQ];
  float* lg[GQ];
#pragma unroll
  for (int g = 0; g < GQ; ++g) {
    const int h = h0 + g;
    q[g] = KLOADF4(a.q + qo + (size_t)h * hd + lane * 4);
    rr[g] = lane < a.drel ? a.r[(size_t)row * a.nh * a.drel + (size_t)h * a.drel + lane] : 0.0f;
    lg[g] = a.scratch + ((size_t)row * a.nh + h) * a.scratch_stride;
    mx[g] = -3.0e38f;
  }
  const bool rel_lane = lane < a.drel;
  const u16* rp = a.rel_proj + (size_t)(rel_lane ? lane : 0) * a.rel_extent;
  const float* kbase = a.kc + (size_t)kh * hd + lane * 4;
  for (int gr = g0 + warp * NG; gr < g1; gr += nw * NG) {
    KF4 k[8 * NG];
#pragma unroll
    for (int i = 0; i < 8 * NG; ++i) {
      const int t = gr * 8 + i;
      k[i] = KF4ZERO;
      if (t < T && gr + i / 8 < g1) {
        int slot = slot0 + t;
        if (a.ring && slot >= a.ring) slot -= a.ring;
        k[i] = KLOADF4(kbase + (size_t)slot * C);
      }
    }
    float bias[8 * NG];
#pragma unroll
    for (int i = 0; i < 8 * NG; ++i) {
      const int t = gr * 8 + i, dist = T - 1 - t;
      bias[i] = (t < T && rel_lane && dist < a.rel_extent) ? k_bf(rp[dist]) : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < NG; ++u) {
      if (gr + u < g1) {
#pragma unroll
        for (int g = 0; g < GQ; ++g) {
          float v[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const KF4 kk = k[u * 8 + i];
            float acc = KFMA(q[g].x, kk.x, 0.0f);
            acc = KFMA(q[g].y, kk.y, acc);
            acc = KFMA(q[g].z, kk.z, acc);
            acc = KFMA(q[g].w, kk.w, acc);
            float vv = acc * a.scale;
            const int t = (gr + u) * 8 + i, dist = T - 1 - t;
            if (t < T && rel_lane && dist < a.rel_extent) vv = KFMA(rr[g], bias[u * 8 + i], vv);
            v[i] = vv;
          }
          const float sum = k_warp_sum8t(v);  // the logit of key (gr + u) * 8 + lane / 4 (lanes 4i .. 4i+3: key i)
          const int t = (gr + u) * 8 + (lane >> 2);
          if (t < T) {
            if ((lane & 3) == 0) lg[g][t] = sum;
            mx[g] = fmaxf(mx[g], sum);
          }
        }
      }
    }
  }
#pragma unroll
  for (int g = 0; g < GQ; ++g) {
    const float wm = k_warp_max(mx[g]);
    if (lane == 0) smx[warp * GQ + g] = wm;
  }
  __syncthreads();
  if ((int)threadIdx.x < GQ) {
    float m = smx[threadIdx.x];
    for (int w = 1; w < nw; ++w) m = fmaxf(m, smx[w * GQ + threadIdx.x]);
    p.pmax[((size_t)row * a.nh + h0 + threadIdx.x) * p.sa + s] = m;
  }
}

template <int GQ>
__global__ void __launch_bounds__(256) kf2_attn_value(F2AttnArgs p, F2Launch l) {
  __shared__ float wsm[GQ][33];
  __shared__ float esm[GQ * 1024];
  __shared__ int is_last;
  if (l.trig) f2_trigger();
  const AttnArgs& a = p.a;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int bpg = 32 / nw;  // value blocks per (row, kv head)
  int b = (int)blockIdx.x;
  const int sb = b % bpg;
  b /= bpg;
  const int kh = b % a.nkv, row = b / a.nkv;
  const int hd = a.hd, C = a.nkv * hd, h0 = kh * GQ;
  const int w = sb * nw + warp;  // the stripe of this warp
  const int pos = a.st[row].pos;
  int T = pos + 1;
  const int win = a.window > 0 ? a.window : a.ring;
  if (win && T > win) T = win;
  int slot0 = pos + 1 - T;
  if (a.ring) slot0 = slot0 % a.ring;
  const int nk = w < T ? (T - 1 - w) / 32 + 1 : 0;  // the keys t = w + 32 j, j < nk
  const float* vbase = a.vc + (size_t)kh * hd + lane * 4;
  auto vrow = [&](int j) {
    int slot = slot0 + w + 32 * j;
    if (a.ring && slot >= a.ring) slot -= a.ring;
    return vbase + (size_t)slot * C;
  };
  if (p.pf & 4) {  // before the wait: the value rows of this stripe into L2 (not the newest one: prep writes it)
    for (int x = lane; x < nk * 4; x += 32) {
      if (w + 32 * (x >> 2) >= T - 1) continue;
      f2_pf_line(vrow(x >> 2) - lane * 4 + (x & 3) * 32);
    }
  }
  f2_enter(l);
  float* lg[GQ];
  float lcur[GQ];
#pragma unroll
  for (int g = 0; g < GQ; ++g) lg[g] = a.scratch + ((size_t)row * a.nh + h0 + g) * a.scratch_stride;
  // in flight first: the value rows of keys 0 .. 15 and the logits of keys 0 .. 31 (lane j: key j)
  KF4 va[16], vb[16];
#pragma unroll
  for (int k = 0; k < 16; ++k)
    if (k < nk) va[k] = KLOADF4(vrow(k));
#pragma unroll
  for (int g = 0; g < GQ; ++g) lcur[g] = lane < nk ? lg[g][w + 32 * lane] : 0.0f;
  float m[GQ], sink[GQ], es[GQ], acc[GQ][4];
#pragma unroll
  for (int g = 0; g < GQ; ++g) {
    const size_t rh = (size_t)row * a.nh + h0 + g;
    float pm = lane < p.sa ? p.pmax[rh * p.sa + lane] : -3.0e38f;
    pm = k_warp_max(pm);
    sink[g] = a.sink[rh];
    m[g] = fmaxf(pm, sink[g]);
    es[g] = 0.0f;
    acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.0f;
  }
  // super batches of 32 keys j0 .. j0 + 31 (two batches of 16: value rows va, vb), the loads of the next batch in flight
  // while one is added; lane k holds the weight of key j0 + k (its exp sum: thread w + 32 k of kb_attn)
  for (int j0 = 0; j0 < nk; j0 += 32) {
#pragma unroll
    for (int k = 0; k < 16; ++k)
      if (j0 + 16 + k < nk) vb[k] = KLOADF4(vrow(j0 + 16 + k));
    float lnext[GQ];
#pragma unroll
    for (int g = 0; g < GQ; ++g) lnext[g] = j0 + 32 + lane < nk ? lg[g][w + 32 * (j0 + 32 + lane)] : 0.0f;
    float e[GQ];
    const bool mine = j0 + lane < nk;
#pragma unroll
    for (int g = 0; g < GQ; ++g) {
      e[g] = 0.0f;
      if (mine) {
        e[g] = expf(lcur[g] - m[g]);
        lg[g][w + 32 * (j0 + lane)] = e[g];
        es[g] += e[g];
      }
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      if (j0 + k < nk) {
#pragma unroll
        for (int g = 0; g < GQ; ++g) {
          const float wgt = KSHFL(e[g], k);
          acc[g][0] = KFMA(wgt, va[k].x, acc[g][0]);
          acc[g][1] = KFMA(wgt, va[k].y, acc[g][1]);
          acc[g][2] = KFMA(wgt, va[k].z, acc[g][2]);
          acc[g][3] = KFMA(wgt, va[k].w, acc[g][3]);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < 16; ++k)
      if (j0 + 32 + k < nk) va[k] = KLOADF4(vrow(j0 + 32 + k));
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      if (j0 + 16 + k < nk) {
#pragma unroll
        for (int g = 0; g < GQ; ++g) {
          const float wgt = KSHFL(e[g], 16 + k);
          acc[g][0] = KFMA(wgt, vb[k].x, acc[g][0]);
          acc[g][1] = KFMA(wgt, vb[k].y, acc[g][1]);
          acc[g][2] = KFMA(wgt, vb[k].z, acc[g][2]);
          acc[g][3] = KFMA(wgt, vb[k].w, acc[g][3]);
        }
      }
    }
#pragma unroll
    for (int g = 0; g < GQ; ++g) lcur[g] = lnext[g];
  }
#pragma unroll
  for (int g = 0; g < GQ; ++g) {
    const size_t rh = (size_t)row * a.nh + h0 + g;
    *(float4*)(p.part + (rh * 32 + w) * hd + lane * 4) = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
    p.es[rh * 1024 + w + 32 * lane] = es[g];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) is_last = atomicAdd(p.cnt + (size_t)row * a.nkv + kh, 1u) == (unsigned)(bpg - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  // the steps of kb_attn after its loops, per head: k_block_sum of the 1024 exp sums plus expf(sink - m), the stripes
  // in order, divide, gate. Every load of a phase issued before its values are used.
  const size_t rh0 = (size_t)row * a.nh + h0;
#pragma unroll 8
  for (int x = threadIdx.x; x < GQ * 1024; x += blockDim.x) esm[x] = __ldcg(p.es + rh0 * 1024 + x);
  __syncthreads();
  for (int x = warp; x < GQ * 32; x += nw) {
    const float v = k_warp_sum(esm[x * 32 + lane]);
    if (lane == 0) wsm[x >> 5][x & 31] = v;
  }
  __syncthreads();
  if ((int)threadIdx.x < GQ) {
    const int g = threadIdx.x;
    float mg = 0.0f, sg = 0.0f;  // (m[g], sink[g] without a dynamic index: the arrays stay in registers)
#pragma unroll
    for (int i = 0; i < GQ; ++i)
      if (g == i) mg = m[i], sg = sink[i];
    float t = 0.0f;
    for (int i = 0; i < 32; ++i) t += wsm[g][i];
    wsm[g][32] = t + expf(sg - mg);
  }
  __syncthreads();
  const size_t qo = (size_t)row * a.nh * hd;
  for (int x = threadIdx.x; x < GQ * hd; x += blockDim.x) {
    const int g = x / hd, j = x - g * hd;
    const float* pp = p.part + ((rh0 + g) * 32) * hd + j;
    float v[32];
#pragma unroll
    for (int c = 0; c < 32; ++c) v[c] = __ldcg(pp + (size_t)c * hd);
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) sum += v[c];
    const size_t o = qo + (size_t)(h0 + g) * hd + j;
    a.out[o] = (sum / wsm[g][32]) * a.gate[o];
  }
  if (threadIdx.x == 0) p.cnt[(size_t)row * a.nkv + kh] = 0;
}

// ---------------------------------------------------------------------------
// kf2_pgemvc (pf 3): pgemvc<NR> for packed jobs (NVFP4 / MXFP8) with the code words and block scales of the block in
// registers BEFORE the wait (the weights of the launch stream in while the kernel before still runs), the inputs after
// it. The arithmetic of k_pgemvc / k_pk8 exactly: FMA chain from 0 over the 8 columns in order, times the block
// scale, k_warp_sum8t per group of 8 rows, warps added in order, k_jout.
// ---------------------------------------------------------------------------
template <int NR>
__global__ void __launch_bounds__(NR >= 16 ? 576 : 1024) kf2_pgemvc(GemvArgs a, F2Launch l) {
  __shared__ float sm[(NR < 8 ? 8 : NR) * 32];
  if (l.trig) f2_trigger();
  const int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int b = KBID, rhs = 0, j = 0;
  if (a.nrhs > 1) {
    rhs = b / a.blocks;
    b -= rhs * a.blocks;
  }
  for (;;) {
    int nb = (a.job[j].rows + NR - 1) / NR;
    if (b < nb) break;
    b -= nb;
    ++j;
  }
  const GemvJob& J = a.job[j];
  const int cols = J.cols, fmt = J.fmt;
  const int row0 = b * NR;
  int nrows = J.rows - row0;
  if (nrows > NR) nrows = NR;
  const int c0 = KTID * 8;
  const bool act = c0 < cols;
  const u8* wb = (const u8*)J.w;
  u64 cw[NR];
  unsigned sc[NR];
  KUNROLL for (int r = 0; r < NR; ++r) {
    cw[r] = 0;
    sc[r] = 0;
    if (act && r < nrows) {
      const size_t row = (size_t)(row0 + r);
      if (fmt == INK2_DFMT_NVFP4) {
        cw[r] = (u64)KLOAD32(wb + row * (size_t)(cols >> 1) + (size_t)(c0 >> 1));
        sc[r] = J.sc[row * (size_t)(cols >> 4) + (size_t)(c0 >> 4)];
      } else {
        cw[r] = KLOAD64(wb + row * (size_t)cols + (size_t)c0);
        sc[r] = J.sc[row * (size_t)(cols >> 5) + (size_t)(c0 >> 5)];
      }
    }
  }
  if (l.extra > 0 && KBID < l.extra && rhs == 0)  // later waves of the launch into L2
    for (int k = 1; k <= l.rounds; ++k) {
      const int bb = KBID + k * l.extra;
      if (bb >= (a.nrhs > 1 ? a.blocks : (int)gridDim.x)) break;
      f2_pf_chunk(a, bb, NR, 1);
    }
  f2_enter(l);
  KF4 xa = KF4ZERO, xb = KF4ZERO;
  if (act) {
    const float* xx = k_xrow(J, rhs, c0);
    xa = KLOADF4(xx);
    xb = KLOADF4(xx + 4);
  }
  float part[NR < 8 ? 8 : NR];
  KUNROLL for (int r = 0; r < (NR < 8 ? 8 : NR); ++r) part[r] = 0.0f;
  if (act) {
    if (fmt == INK2_DFMT_NVFP4) {
      KUNROLL for (int r = 0; r < NR; ++r) if (r < nrows) {
        const unsigned c = (unsigned)cw[r];
        float s = 0.0f;
        K_FP4_8(c, xa, xb)
        part[r] = s * k_e4m3v(sc[r]);
      }
    } else {
      KUNROLL for (int r = 0; r < NR; ++r) if (r < nrows) {
        const u64 c = cw[r];
        float s = 0.0f;
        K_E4M3_8(c, xa, xb)
        part[r] = s * k_exp2v(sc[r]);
      }
    }
  }
  KUNROLL for (int g = 0; g < NR; g += 8) {
    float s = k_warp_sum8t(KCTXA part + g);
    if ((lane & 3) == 0) sm[(g + (lane >> 2)) * 32 + warp] = s;
  }
  KSYNC();
  if (KTID < nrows) {
    float t = 0.0f;
    for (int w = 0; w < nw; ++w) t += sm[KTID * 32 + w];
    J.y[(size_t)rhs * J.rows + row0 + KTID] = k_jout(J, t);
  }
}

// ---------------------------------------------------------------------------
// Router product + top-k in one launch. The blocks run the body of gemvw (R = 1) or gemvwm (R > 1) unchanged; the
// last block to finish (counter) runs the arithmetic of topk for every row of the pass (f2_topk = kb_topk with the
// row as a parameter and the logits read from L2).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void f2_topk(const RouterArgs& a, int row, float* scv, float* raw, int* flag) {
  int i = KTID;
  int E = a.E;
  int* oidx = a.idx + (size_t)row * a.k;
  float* owt = a.wt + (size_t)row * a.k;
  if (i < E) {
    raw[i] = k_sigmoid(__ldcg(a.logits + (size_t)row * E + i));
    scv[i] = raw[i] + a.bias[i];
  }
  KSYNC();
  int chosen = 0;
  if (i < E) {
    int rank = 0;
    float mine = scv[i];
    for (int j = 0; j < E; ++j) rank += (scv[j] > mine || (scv[j] == mine && j < i)) ? 1 : 0;
    chosen = rank < a.k ? 1 : 0;
    flag[i] = chosen;
  }
  KSYNC();
  if (chosen) {
    int pos = 0;
    for (int j = 0; j < i; ++j) pos += flag[j];
    oidx[pos] = i;
    scv[pos] = raw[i];
  }
  KSYNC();
  if (i == 0) {
    float s = 0.0f;
    for (int p = 0; p < a.k; ++p) s += scv[p];
    for (int p = 0; p < a.k; ++p) owt[p] = scv[p] / (s + 1.0e-20f) * a.route_scale;
  }
  KSYNC();  // scv / flag are written again by the next row
}
template <int MULTI>
__global__ void __launch_bounds__(256) kf2_router(F2RouterArgs a, F2Launch l) {
  __shared__ float scv[256], raw[256];
  __shared__ int flag[256];
  __shared__ int last;
  __shared__ float red[33];
  if (l.trig) f2_trigger();
  // pf 3, one row: the bfloat16 words of the block's router row (at most 2 units of 8 per thread) before the wait
  const GemvJob& J0 = a.g.job[0];
  const int nunit = J0.cols >> 3;
  const bool pre = !MULTI && l.pf == 3 && a.g.njobs == 1 && J0.split == J0.cols && nunit <= 2 * (int)blockDim.x;
  u64 w0a = 0, w0b = 0, w1a = 0, w1b = 0;
  if (pre) {
    const u16* wr = J0.w + (size_t)blockIdx.x * (size_t)J0.cols;
    const int u0 = KTID, u1 = KTID + KNT;
    if (u0 < nunit) w0a = KLOAD64(wr + (u0 << 3)), w0b = KLOAD64(wr + (u0 << 3) + 4);
    if (u1 < nunit) w1a = KLOAD64(wr + (u1 << 3)), w1b = KLOAD64(wr + (u1 << 3) + 4);
  } else if (l.pf) {
    if (MULTI) f2_pf_block<KID_gemvwm>(a.g, (int)blockIdx.x, l.pf);
    else f2_pf_block<KID_gemvw>(a.g, (int)blockIdx.x, l.pf);
  }
  if (!MULTI && a.pw) {  // the experts of the previous step: block b takes part b % per of slot b / per
    const int per = (int)gridDim.x / a.r.k;
    const int slot = (int)blockIdx.x / (per > 0 ? per : 1), part = (int)blockIdx.x - slot * per;
    if (per > 0 && slot < a.r.k) {
      const int e = __ldcg(a.r.idx + slot);
      const int rows = (a.prows + per - 1) / per, r0 = part * rows;
      int n = a.prows - r0;
      if (n > rows) n = rows;
      if (n > 0 && e >= 0 && e < a.r.E) {
        const size_t row0 = (size_t)e * a.prows + (size_t)r0;
        f2_pf_span(a.pw + row0 * (size_t)(a.pcols >> 1), (size_t)n * (size_t)(a.pcols >> 1), 1);
        f2_pf_span(a.ps + row0 * (size_t)(a.pcols >> 4), (size_t)n * (size_t)(a.pcols >> 4), 1);
      }
    }
  }
  f2_enter(l);
  if (MULTI) {
    kb_gemvwm(a.g);
  } else if (pre) {  // k_bf_dot over the units KTID, KTID + KNT in order, then the block sum of gemvw
    float acc = 0.0f;
    const int u0 = KTID, u1 = KTID + KNT;
    if (u0 < nunit) {
      const float* xx = J0.x + (u0 << 3);
      KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
      acc = KFMA(k_bf((unsigned)(w0a & 0xffffu)), xa.x, acc);
      acc = KFMA(k_bf((unsigned)((w0a >> 16) & 0xffffu)), xa.y, acc);
      acc = KFMA(k_bf((unsigned)((w0a >> 32) & 0xffffu)), xa.z, acc);
      acc = KFMA(k_bf((unsigned)(w0a >> 48)), xa.w, acc);
      acc = KFMA(k_bf((unsigned)(w0b & 0xffffu)), xb.x, acc);
      acc = KFMA(k_bf((unsigned)((w0b >> 16) & 0xffffu)), xb.y, acc);
      acc = KFMA(k_bf((unsigned)((w0b >> 32) & 0xffffu)), xb.z, acc);
      acc = KFMA(k_bf((unsigned)(w0b >> 48)), xb.w, acc);
    }
    if (u1 < nunit) {
      const float* xx = J0.x + (u1 << 3);
      KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
      acc = KFMA(k_bf((unsigned)(w1a & 0xffffu)), xa.x, acc);
      acc = KFMA(k_bf((unsigned)((w1a >> 16) & 0xffffu)), xa.y, acc);
      acc = KFMA(k_bf((unsigned)((w1a >> 32) & 0xffffu)), xa.z, acc);
      acc = KFMA(k_bf((unsigned)(w1a >> 48)), xa.w, acc);
      acc = KFMA(k_bf((unsigned)(w1b & 0xffffu)), xb.x, acc);
      acc = KFMA(k_bf((unsigned)((w1b >> 16) & 0xffffu)), xb.y, acc);
      acc = KFMA(k_bf((unsigned)((w1b >> 32) & 0xffffu)), xb.z, acc);
      acc = KFMA(k_bf((unsigned)(w1b >> 48)), xb.w, acc);
    }
    float s = k_block_sum(KCTXA acc, red);
    if (KTID == 0) J0.y[blockIdx.x] = k_jout(J0, s);
  } else {
    kb_gemvw(a.g);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(a.done, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int R = a.r.rows > 0 ? a.r.rows : 1;
  for (int r = 0; r < R; ++r) f2_topk(a.r, r, scv, raw, flag);
  if (MULTI && a.gr.idx) kb_egroup(a.gr);  // the topk writes of this block are visible to it after the barriers
  if (threadIdx.x == 0) *a.done = 0u;
}

// ---------------------------------------------------------------------------
// fgemvm: packed dense rows (NVFP4 / MXFP8 jobs, all the same cols, cols / 8 threads) for a pass of R rows, groups of
// K_NRHS = 8 rows of the pass. Block = (group, chunk of nb matrix rows of one job). The inputs of the group stay in
// registers for all nb rows (pgemvcm2 loads them again for every 2 rows), the rows are read in waves of WV, the code
// words of a wave all in flight before the first sum (with prefetch on, the whole chunk is asked from L2 first). Per (matrix row, row of the
// pass): the sums of pgemvc exactly (thread t: the 8 columns 8t .. 8t+7, FMA chain from 0 in column order, times the
// block scale; k_warp_sum8t = k_warp_sum per value; warps added in order; k_jout), so a pass gives bit for bit the
// single steps of pgemvc for any nb.
// ---------------------------------------------------------------------------
#define F2_MNB 32  // largest nb (shared memory: nb * 8 * 32 floats)
template <int WV, int MAXT, int MINB>
__global__ void __launch_bounds__(MAXT, MINB) kf2_gemvm(F2GemvM m, F2Launch l) {
  __shared__ float sm[F2_MNB * K_NRHS * 32];
  const GemvArgs& a = m.g;
  const int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  const int grp = KBID / m.blocks;
  int b = KBID - grp * m.blocks, j = 0;
  if (l.trig) f2_trigger();
  if (l.pf && grp == 0) {
    f2_pf_chunk(a, b, m.nb, l.pf == 3 ? 1 : l.pf);
    if (l.extra > 0 && b < l.extra)
      for (int k = 1; k <= l.rounds; ++k) {
        const int bb = b + k * l.extra;
        if (bb >= m.blocks) break;
        f2_pf_chunk(a, bb, m.nb, l.pf == 3 ? 1 : l.pf);
      }
  }
  for (;;) {
    int nb = (a.job[j].rows + m.nb - 1) / m.nb;
    if (b < nb) break;
    b -= nb;
    ++j;
  }
  const GemvJob& J = a.job[j];
  const int cols = J.cols, fmt = J.fmt;
  const int row0 = b * m.nb;
  int nrows = J.rows - row0;
  if (nrows > m.nb) nrows = m.nb;
  const int r0 = grp * K_NRHS;
  int nr = a.nrhs - r0;
  if (nr > K_NRHS) nr = K_NRHS;
  const int c0 = KTID * 8;
  const bool act = c0 < cols;
  const u8* wb = (const u8*)J.w;
  const size_t rb = fmt == INK2_DFMT_NVFP4 ? (size_t)(cols >> 1) : (size_t)cols;
  const size_t sbb = fmt == INK2_DFMT_NVFP4 ? (size_t)(cols >> 4) : (size_t)(cols >> 5);
  const size_t woff = fmt == INK2_DFMT_NVFP4 ? (size_t)(c0 >> 1) : (size_t)c0;
  const size_t soff = fmt == INK2_DFMT_NVFP4 ? (size_t)(c0 >> 4) : (size_t)(c0 >> 5);
  u64 cw[WV];
  unsigned sc[WV];
  const bool pre = l.pf == 3;  // the code words of wave 0 before the wait
  KUNROLL for (int r = 0; r < WV; ++r) {
    cw[r] = 0;
    sc[r] = 0;
    if (pre && act && r < nrows) {
      const size_t row = (size_t)(row0 + r);
      cw[r] = fmt == INK2_DFMT_NVFP4 ? (u64)KLOAD32(wb + row * rb + woff) : KLOAD64(wb + row * rb + woff);
      sc[r] = J.sc[row * sbb + soff];
    }
  }
  f2_enter(l);
  float x[K_NRHS][8];
  KUNROLL for (int i = 0; i < K_NRHS; ++i) {
    KUNROLL for (int e = 0; e < 8; ++e) x[i][e] = 0.0f;
    if (act && i < nr) {
      const float* xx = k_xrow(J, r0 + i, c0);
      KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
      x[i][0] = xa.x;
      x[i][1] = xa.y;
      x[i][2] = xa.z;
      x[i][3] = xa.w;
      x[i][4] = xb.x;
      x[i][5] = xb.y;
      x[i][6] = xb.z;
      x[i][7] = xb.w;
    }
  }
  KNOUNROLL for (int w0 = 0; w0 < nrows; w0 += WV) {
    KUNROLL for (int r = 0; r < WV; ++r) {  // the code words of the wave
      if (pre && w0 == 0) continue;
      cw[r] = 0;
      sc[r] = 0;
      if (act && w0 + r < nrows) {
        const size_t row = (size_t)(row0 + w0 + r);
        cw[r] = fmt == INK2_DFMT_NVFP4 ? (u64)KLOAD32(wb + row * rb + woff) : KLOAD64(wb + row * rb + woff);
        sc[r] = J.sc[row * sbb + soff];
      }
    }
    KUNROLL for (int r = 0; r < WV; ++r) {
      if (w0 + r < nrows) {
        float acc[K_NRHS];
        KUNROLL for (int i = 0; i < K_NRHS; ++i) acc[i] = 0.0f;
        if (act) {
          float f[8];
          float scl;
          if (fmt == INK2_DFMT_NVFP4) {
            const unsigned c = (unsigned)cw[r];
            KUNROLL for (int e = 0; e < 8; ++e) f[e] = k_fp4x2(c >> (4 * e));
            scl = k_e4m3v(sc[r]);
          } else {
            KUNROLL for (int e = 0; e < 8; ++e) f[e] = k_e4m3v((unsigned)(cw[r] >> (8 * e)));
            scl = k_exp2v(sc[r]);
          }
          KUNROLL for (int i = 0; i < K_NRHS; ++i) {
            if (i < nr) {
              float s = KFMA(f[0], x[i][0], 0.0f);
              s = KFMA(f[1], x[i][1], s);
              s = KFMA(f[2], x[i][2], s);
              s = KFMA(f[3], x[i][3], s);
              s = KFMA(f[4], x[i][4], s);
              s = KFMA(f[5], x[i][5], s);
              s = KFMA(f[6], x[i][6], s);
              s = KFMA(f[7], x[i][7], s);
              acc[i] = s * scl;
            }
          }
        }
        float t = k_warp_sum8t(KCTXA acc);
        if ((lane & 3) == 0) sm[((w0 + r) * K_NRHS + (lane >> 2)) * 32 + warp] = t;
      }
    }
  }
  KSYNC();
  for (int t = KTID; t < nrows * nr; t += KNT) {
    int r = t / nr, i = t - r * nr;
    float s = 0.0f;
    for (int w = 0; w < nw; ++w) s += sm[(r * K_NRHS + i) * 32 + w];
    J.y[(size_t)(r0 + i) * J.rows + row0 + r] = k_jout(J, s);
  }
}

// ---------------------------------------------------------------------------
// fgemvm2 = fgemvm with
//   - the code words of the next wave of WV matrix rows loaded while the current wave is summed (2 WV rows in flight),
//   - NH = 2, 4 or 8 rows of the pass per block (passes of 2 .. 4 rows keep NH x 8 inputs in registers instead of 64),
//   - the groups of right-hand sides of a chunk next to each other in the grid (block = chunk * groups + group), so the
//     later groups read the chunk from L2 right after the first (passes of 9 .. 32 rows, the output projection),
//   - FP4 1: every job NVFP4 (32-bit code words, fewer registers).
// Per (matrix row, row of the pass) the sums of kf2_gemvm / pgemvc exactly: thread t the 8 columns 8t .. 8t+7, FMA chain
// from 0 in column order, times the block scale; k_warp_sumt<NH> (= k_warp_sum per value); warps added in order; k_jout.
// ---------------------------------------------------------------------------
template <int FP4>
struct F2Cw {
  typedef u64 T;
};
template <>
struct F2Cw<1> {
  typedef unsigned T;
};
// m.pgrid > 0 (persistent): a grid of about the resident blocks walks the items (chunk, group) with the stride of the grid
// (a multiple of the groups, so a block keeps its group), the inputs stay in registers while the job's input and the group
// do not change (no per-chunk reload), chunks of m.nb rows; the numbers per (matrix row, row of the pass) do not depend on
// which block computes them.
template <int NH, int WV, int FP4, int MAXT, int MINB>
__global__ void __launch_bounds__(MAXT, MINB) kf2_gemvm2(F2GemvM2 m, F2Launch l) {
  typedef typename F2Cw<FP4>::T CW;
  __shared__ float sm[F2_MNB * NH * 32];
  const GemvArgs& a = m.g;
  const int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  const int groups = (a.nrhs + NH - 1) / NH;
  const int items = m.blocks * groups;
  const int stride = m.pgrid > 0 ? (int)gridDim.x : items;
  const int c0 = KTID * 8;
  if (l.trig) f2_trigger();
  if (l.pf)
    for (int it = KBID; it < items; it += stride)
      if (it % groups == 0) f2_pf_chunk(a, it / groups, m.nb, l.pf == 3 ? 1 : l.pf);
  CW cw[WV], cn[WV];
  unsigned sc[WV], sn[WV];
  float x[NH][8];
  const float* xsrc = nullptr;  // the input rows now in x (job input, group)
  int xgrp = -1;
  bool waited = false;
#define F2_LD2(row, c, s)                                                                                \
  do {                                                                                                   \
    c = 0;                                                                                               \
    s = 0;                                                                                               \
    if (act && (row) < nrows) {                                                                          \
      const size_t rr_ = (size_t)(row0 + (row));                                                         \
      if (fmt == INK2_DFMT_NVFP4) c = (CW)KLOAD32(wb + rr_ * rb + woff);                                 \
      else c = (CW)KLOAD64(wb + rr_ * rb + woff);                                                        \
      s = J.sc[rr_ * sbb + soff];                                                                        \
    }                                                                                                    \
  } while (0)
  KNOUNROLL for (int it = KBID; it < items; it += stride) {
    int b = it / groups;
    const int grp = it - b * groups;
    int j = 0;
    for (;;) {
      int nb = (a.job[j].rows + m.nb - 1) / m.nb;
      if (b < nb) break;
      b -= nb;
      ++j;
    }
    const GemvJob& J = a.job[j];
    const int cols = J.cols;
    const int fmt = FP4 ? INK2_DFMT_NVFP4 : J.fmt;
    const int row0 = b * m.nb;
    int nrows = J.rows - row0;
    if (nrows > m.nb) nrows = m.nb;
    const int r0 = grp * NH;
    int nr = a.nrhs - r0;
    if (nr > NH) nr = NH;
    const bool act = c0 < cols;
    const u8* wb = (const u8*)J.w;
    const size_t rb = fmt == INK2_DFMT_NVFP4 ? (size_t)(cols >> 1) : (size_t)cols;
    const size_t sbb = fmt == INK2_DFMT_NVFP4 ? (size_t)(cols >> 4) : (size_t)(cols >> 5);
    const size_t woff = fmt == INK2_DFMT_NVFP4 ? (size_t)(c0 >> 1) : (size_t)c0;
    const size_t soff = fmt == INK2_DFMT_NVFP4 ? (size_t)(c0 >> 4) : (size_t)(c0 >> 5);
    KUNROLL for (int r = 0; r < WV; ++r) F2_LD2(r, cw[r], sc[r]);  // weights only: the first item's before the wait
    if (!waited) {
      f2_enter(l);
      waited = true;
    }
    if (J.x != xsrc || grp != xgrp) {
      xsrc = J.x;
      xgrp = grp;
      KUNROLL for (int i = 0; i < NH; ++i) {
        KUNROLL for (int e = 0; e < 8; ++e) x[i][e] = 0.0f;
        if (act && i < nr) {
          const float* xx = k_xrow(J, r0 + i, c0);
          KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
          x[i][0] = xa.x;
          x[i][1] = xa.y;
          x[i][2] = xa.z;
          x[i][3] = xa.w;
          x[i][4] = xb.x;
          x[i][5] = xb.y;
          x[i][6] = xb.z;
          x[i][7] = xb.w;
        }
      }
    }
    KNOUNROLL for (int w0 = 0; w0 < nrows; w0 += WV) {
      KUNROLL for (int r = 0; r < WV; ++r) F2_LD2(w0 + WV + r, cn[r], sn[r]);  // the next wave in flight now
      KUNROLL for (int r = 0; r < WV; ++r) {
        if (w0 + r < nrows) {
          float acc[NH];
          KUNROLL for (int i = 0; i < NH; ++i) acc[i] = 0.0f;
          if (act) {
            float f[8];
            float scl;
            if (fmt == INK2_DFMT_NVFP4) {
              const unsigned c = (unsigned)cw[r];
              KUNROLL for (int e = 0; e < 8; ++e) f[e] = k_fp4x2(c >> (4 * e));
              scl = k_e4m3v(sc[r]);
            } else {
              KUNROLL for (int e = 0; e < 8; ++e) f[e] = k_e4m3v((unsigned)((u64)cw[r] >> (8 * e)));
              scl = k_exp2v(sc[r]);
            }
            KUNROLL for (int i = 0; i < NH; ++i) {
              if (i < nr) {
                float s = KFMA(f[0], x[i][0], 0.0f);
                s = KFMA(f[1], x[i][1], s);
                s = KFMA(f[2], x[i][2], s);
                s = KFMA(f[3], x[i][3], s);
                s = KFMA(f[4], x[i][4], s);
                s = KFMA(f[5], x[i][5], s);
                s = KFMA(f[6], x[i][6], s);
                s = KFMA(f[7], x[i][7], s);
                acc[i] = s * scl;
              }
            }
          }
          float t = k_warp_sumt<NH>(KCTXA acc);  // the sum of value i in lanes i * 32 / NH ..
          if ((lane & (32 / NH - 1)) == 0) sm[((w0 + r) * NH + lane / (32 / NH)) * 32 + warp] = t;
        }
      }
      KUNROLL for (int r = 0; r < WV; ++r) {
        cw[r] = cn[r];
        sc[r] = sn[r];
      }
    }
    KSYNC();
    for (int t = KTID; t < nrows * nr; t += KNT) {
      int r = t / nr, i = t - r * nr;
      float s = 0.0f;
      for (int w = 0; w < nw; ++w) s += sm[(r * NH + i) * 32 + w];
      J.y[(size_t)(r0 + i) * J.rows + row0 + r] = k_jout(J, s);
    }
    KSYNC();  // sm free for the next item
  }
#undef F2_LD2
  if (!waited) f2_enter(l);
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
namespace {
template <class A>
void f2_launch(void (*k)(A, F2Launch), const A& a, const F2Launch& l, int grid, int block, cudaStream_t s, bool pdl) {
  if (!pdl) {
    k<<<grid, block, 0, s>>>(a, l);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3((unsigned)block);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, a, l);
}
// resident blocks of a kernel on the whole device (cached per kernel and block size)
template <class A>
int f2_resident(void (*k)(A, F2Launch), int block) {
  struct C {
    const void* k;
    int block, n;
  };
  static C cache[64];
  static int nc = 0;
  for (int i = 0; i < nc; ++i)
    if (cache[i].k == (const void*)k && cache[i].block == block) return cache[i].n;
  int per = 0, dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, block, 0) != cudaSuccess) per = 0;
  cudaGetLastError();
  int n = per * sms;
  if (nc < 64) cache[nc++] = {(const void*)k, block, n};
  return n;
}
size_t f2_weight_bytes(const GemvArgs& a) {
  size_t t = 0;
  for (int j = 0; j < a.njobs; ++j) {
    const GemvJob& J = a.job[j];
    size_t rb = J.fmt == INK2_DFMT_NVFP4 ? (size_t)(J.cols / 2 + J.cols / 16) : J.fmt == INK2_DFMT_MXFP8 ? (size_t)(J.cols + J.cols / 32)
                                                                                                         : (size_t)J.cols * 2;
    t += rb * (size_t)J.rows;
  }
  return t;
}
// the launch details of a weight kernel: own region, and the first wave also the later waves within the budget
template <class A>
F2Launch f2_gemv_launch(void (*k)(A, F2Launch), const GemvArgs& a, int grid, int block, int sets, const F2Opts& o) {
  F2Launch l{};
  l.pf = o.pf;
  if (o.pf && (o.pf_extra & 1)) {
    const int res = f2_resident(k, block);
    const int g1 = sets > 1 ? grid / sets : grid;  // blocks of the first set (the others read the same weights)
    if (res > 0 && g1 > res) {
      const size_t per_block = f2_weight_bytes(a) / (size_t)g1 + 1;
      long long rounds = (g1 + res - 1) / res - 1;
      const long long fit = o.pf_budget / (long long)(per_block * (size_t)res) - 1;  // the first wave itself counts
      if (rounds > fit) rounds = fit;
      if (rounds > 0) {
        l.extra = res;
        l.rounds = (int)rounds;
      }
    }
  }
  return l;
}
}  // namespace

bool f2_all_packed(const GemvArgs& a) {
  for (int j = 0; j < a.njobs; ++j)
    if (a.job[j].fmt != INK2_DFMT_NVFP4 && a.job[j].fmt != INK2_DFMT_MXFP8) return false;
  return true;
}
// Launches kernel kid through its wrapper of the second set; false: no wrapper (the caller launches the general kernel).
bool ink2_f2_run(int kid, const void* args, int grid, int block, cudaStream_t s, bool pdl, const F2Opts& o) {
  F2Launch l{};
  l.trig = o.trig_small;
  switch (kid) {
#define G(name)                                                                                     \
  case KID_##name: {                                                                                \
    if (!o.pf) return false;                                                                        \
    const GemvArgs& g = *(const GemvArgs*)args;                                                     \
    const int sets = g.nrhs > 1 && g.blocks > 0 ? grid / g.blocks : 1;                              \
    f2_launch(kq_##name, g, f2_gemv_launch(kq_##name, g, grid, block, sets, o), grid, block, s, pdl);\
    return true;                                                                                    \
  }
#define P(name, NR)                                                                                 \
  case KID_##name: {                                                                                \
    if (!o.pf || (NR >= 16 && o.pf == 3)) return false; /* 16 rows (output projection): plain */    \
    const GemvArgs& g = *(const GemvArgs*)args;                                                     \
    const int sets = g.nrhs > 1 && g.blocks > 0 ? grid / g.blocks : 1;                              \
    if (o.pf == 3 && (o.reg & 1) && NR <= 8 && f2_all_packed(g)) {  /* 16 rows (output projection): measured slower in registers */                                                            \
      F2Launch lr = f2_gemv_launch(kf2_pgemvc<NR>, g, grid, block, sets, o);                        \
      f2_launch(kf2_pgemvc<NR>, g, lr, grid, block, s, pdl);                                        \
      return true;                                                                                  \
    }                                                                                               \
    f2_launch(kq_##name, g, f2_gemv_launch(kq_##name, g, grid, block, sets, o), grid, block, s, pdl);\
    return true;                                                                                    \
  }
    P(pgemvc, 16)
    P(pgemvc8, 8)
    P(pgemvc4, 4)
    P(pgemvc2, 2)
#undef P
    G(pgemvcm)
    G(pgemvcm8)
    G(pgemvcm4)
    G(pgemvcm2)
    G(gemvc)
    case KID_gemvcm: {  // bfloat16 rows of a pass; o.bfm 0 = the general kernel without the wrapper
      if (!o.pf || !o.bfm) return false;
      const GemvArgs& g = *(const GemvArgs*)args;
      const int sets = g.nrhs > 1 && g.blocks > 0 ? grid / g.blocks : 1;
      f2_launch(kq_gemvcm, g, f2_gemv_launch(kq_gemvcm, g, grid, block, sets, o), grid, block, s, pdl);
      return true;
    }
    G(pgemv)
    G(gemv)
    G(pgemvw)
    G(gemvw)
    case KID_gemvwm: {  // as gemvcm
      if (!o.pf || !o.bfm) return false;
      const GemvArgs& g = *(const GemvArgs*)args;
      const int sets = g.nrhs > 1 && g.blocks > 0 ? grid / g.blocks : 1;
      f2_launch(kq_gemvwm, g, f2_gemv_launch(kq_gemvwm, g, grid, block, sets, o), grid, block, s, pdl);
      return true;
    }
    G(pgemvn)
#undef G
    case KID_attn:
      if (!o.trig_small && !o.pf) return false;
      l.pf = o.pf && (o.reg & 16) ? 1 : 0;  // reg bit 16: the key / value prefetch
      f2_launch(kq_attn, *(const AttnArgs*)args, l, grid, block, s, pdl);
      return true;
    case KID_norm:
      if (!o.trig_small) return false;
      f2_launch(kq_norm, *(const NormArgs*)args, l, grid, block, s, pdl);
      return true;
    case KID_gmix:
      if (!o.trig_small) return false;
      f2_launch(kq_gmix, *(const GmixArgs*)args, l, grid, block, s, pdl);
      return true;
    default:
      return false;
  }
}

namespace {
// attention at long context: the score and the value launches (grid and block from the arguments)
template <int GQ>
void f2_attn_launch(const F2AttnArgs& f, const F2Launch& l, cudaStream_t s, bool pdl) {
  const int R = f.a.rows > 0 ? f.a.rows : 1;
  if (f.ng == 2) f2_launch(kf2_attn_score<GQ, 2>, f, l, R * f.a.nkv * f.sa, 128, s, pdl);
  else f2_launch(kf2_attn_score<GQ, 1>, f, l, R * f.a.nkv * f.sa, 128, s, pdl);
  f2_launch(kf2_attn_value<GQ>, f, l, R * f.a.nkv * (32 / f.sw), 32 * f.sw, s, pdl);
}
int f2_attn_go(const F2AttnArgs& f, const F2Launch& l, cudaStream_t s, bool pdl) {
  const AttnArgs& a = f.a;
  if (a.hd != 128 || a.nkv < 1 || a.nh % a.nkv || a.drel > 32 || f.sa < 1 || f.sa > 32 ||
      (f.sw != 1 && f.sw != 2 && f.sw != 4 && f.sw != 8))
    return (int)cudaErrorInvalidValue;
  switch (a.nh / a.nkv) {
    case 1: f2_attn_launch<1>(f, l, s, pdl); break;
    case 2: f2_attn_launch<2>(f, l, s, pdl); break;
    case 3: f2_attn_launch<3>(f, l, s, pdl); break;
    case 4: f2_attn_launch<4>(f, l, s, pdl); break;
    default: return (int)cudaErrorInvalidValue;
  }
  return (int)cudaGetLastError();
}
}  // namespace

// fid INK2_F2_ROUTER / INK2_F2_GEMVM / INK2_F2_ATTN
int ink2_f2_run_own(int fid, const void* args, int grid, int block, cudaStream_t s, bool pdl, const F2Opts& o) {
  if (fid == INK2_F2_ATTN) {
    F2Launch l{};
    l.trig = o.trig_small;
    return f2_attn_go(*(const F2AttnArgs*)args, l, s, pdl);
  }
  if (fid == INK2_F2_ROUTER) {
    const F2RouterArgs& a = *(const F2RouterArgs*)args;
    F2Launch l{};
    l.pf = pdl ? o.pf : 0;
    if (l.pf == 3 && !(o.reg & 4)) l.pf = 1;
    if (a.g.nrhs > 1) f2_launch(kf2_router<1>, a, l, grid, block, s, pdl);
    else f2_launch(kf2_router<0>, a, l, grid, block, s, pdl);
    return (int)cudaGetLastError();
  }
  if (fid == INK2_F2_GEMVM) {
    const F2GemvM& m = *(const F2GemvM*)args;
    void (*k)(F2GemvM, F2Launch) = block <= 288 ? (m.wv == 4 ? kf2_gemvm<4, 288, 2> : kf2_gemvm<8, 288, 2>)
                                                : (m.wv == 4 ? kf2_gemvm<4, 384, 1> : kf2_gemvm<8, 384, 1>);
    F2Launch l{};
    l.pf = pdl ? o.pf : 0;
    if (l.pf == 3 && !(o.reg & 2)) l.pf = 1;
    l.trig = 0;
    if (o.pf && (o.pf_extra & 1)) {
      const int res = f2_resident(k, block);
      if (res > 0 && m.blocks > res) {
        const size_t per_block = f2_weight_bytes(m.g) / (size_t)m.blocks + 1;
        long long rounds = (m.blocks + res - 1) / res - 1;
        const long long fit = o.pf_budget / (long long)(per_block * (size_t)res) - 1;
        if (rounds > fit) rounds = fit;
        if (rounds > 0) l.extra = res, l.rounds = (int)rounds;
      }
    }
    f2_launch(k, m, l, grid, block, s, pdl);
    return (int)cudaGetLastError();
  }
  if (fid == INK2_F2_GEMVM2) {
    const F2GemvM2& m = *(const F2GemvM2*)args;
    const bool big = block > 288;
    void (*k)(F2GemvM2, F2Launch) = nullptr;
#define F2_GM2(NH, WV, FP)                                                                     \
  if (m.nh == NH && m.wv == WV && m.fp4 == FP)                                                 \
    k = big ? kf2_gemvm2<NH, WV, FP, 384, 1> : m.occ1 ? kf2_gemvm2<NH, WV, FP, 288, 1> : kf2_gemvm2<NH, WV, FP, 288, 2>;
    F2_GM2(2, 4, 1) F2_GM2(4, 4, 1) F2_GM2(8, 4, 1) F2_GM2(2, 8, 1) F2_GM2(4, 8, 1) F2_GM2(8, 8, 1)
    F2_GM2(2, 4, 0) F2_GM2(4, 4, 0) F2_GM2(8, 4, 0) F2_GM2(2, 8, 0) F2_GM2(4, 8, 0) F2_GM2(8, 8, 0)
#undef F2_GM2
    if (!k) return (int)cudaErrorInvalidValue;
    F2Launch l{};
    l.pf = pdl ? o.pf : 0;
    if (l.pf == 3 && !(o.reg & 2)) l.pf = 1;
    l.trig = 0;
    if (m.pgrid > 0) {  // persistent: the resident blocks (times m.pgrid), a multiple of the groups, at most the items
      const int groups = (m.g.nrhs + m.nh - 1) / m.nh;
      int res = f2_resident(k, block) * m.pgrid;
      res -= res % groups;
      if (res >= groups && res < grid) grid = res;
    }
    f2_launch(k, m, l, grid, block, s, pdl);
    return (int)cudaGetLastError();
  }
  return (int)cudaErrorInvalidValue;
}
// largest block of fgemvm with the registers of this build (0: not available)
int ink2_f2_gemvm_max_block() {
  cudaFuncAttributes fa{};
  if (cudaFuncGetAttributes(&fa, kf2_gemvm<8, 384, 1>) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return fa.maxThreadsPerBlock;
}

// ---------------------------------------------------------------------------
// the split attention against the general attn kernel on random inputs.
// cfg[16]: pos0 (row r at position pos0 + r), R, nh, nkv, ring, window, drel, rel_extent, sa, sw, seed, reps (timing,
//          0 = none), pf, qmul (q scaled by qmul / 16), cap (cache slots without a ring), ng
// out[8]:  differing output floats, differing scratch floats, us general, us split, largest |difference| of the outputs,
//          output floats compared, scratch floats compared, cuda error (0 = none)
// The scratch and the outputs are filled with 0xFF bytes before each run, so writes outside the keys also count.
// Timing: L2 flushed (a 192 MB memset) before every launch, the launches alone timed with events.
// ---------------------------------------------------------------------------
namespace {
unsigned f2c_next(unsigned long long& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return (unsigned)(s >> 11);
}
float f2c_uni(unsigned long long& s) { return (float)((double)(f2c_next(s) & 0xFFFFFF) / 16777216.0 * 2.0 - 1.0); }
}  // namespace

#if defined(_WIN32)
#define INK2_F2_EXPORT extern "C" __declspec(dllexport)
#else
#define INK2_F2_EXPORT extern "C" __attribute__((visibility("default")))
#endif
INK2_F2_EXPORT int ink2_f2_attn_check(const int* cfg, double* out) {
  const int pos0 = cfg[0], R = cfg[1], nh = cfg[2], nkv = cfg[3], ring = cfg[4], window = cfg[5], drel = cfg[6];
  const int rext = cfg[7], sa = cfg[8], sw = cfg[9], seed = cfg[10], reps = cfg[11], pf = cfg[12], qmul = cfg[13];
  const int ng = cfg[15];
  for (int i = 0; i < 8; ++i) out[i] = 0;
  const int hd = 128, C = nkv * hd;
  int cap = ring ? ring : cfg[14];
  if (!ring && cap < pos0 + R) cap = pos0 + R;
  const int stride = cap;
  unsigned long long rs = 0x9E3779B97F4A7C15ull ^ ((unsigned long long)(unsigned)seed * 0x100000001B3ull);
  if (!rs) rs = 1;
  const size_t nq = (size_t)R * nh * hd, nr = (size_t)R * nh * (drel > 0 ? drel : 1), ns = (size_t)R * nh;
  const size_t nkvc = (size_t)cap * C, nrp = (size_t)(drel > 0 ? drel : 1) * rext, nscr = (size_t)R * nh * stride;
  std::vector<float> hq(nq), hr(nr), hsink(ns), hgate(nq), hk(nkvc), hv(nkvc);
  std::vector<unsigned short> hrp(nrp);
  std::vector<StepState> hst(R);
  for (auto& x : hq) x = f2c_uni(rs) * (float)qmul / 16.0f;
  for (auto& x : hr) x = f2c_uni(rs);
  for (auto& x : hsink) x = f2c_uni(rs) * 2.0f;
  for (auto& x : hgate) x = (f2c_uni(rs) + 1.0f) * 0.5f;
  for (auto& x : hk) x = f2c_uni(rs);
  for (auto& x : hv) x = f2c_uni(rs);
  for (auto& x : hrp) {
    float f = f2c_uni(rs);
    unsigned u;
    memcpy(&u, &f, 4);
    x = (unsigned short)(u >> 16);
  }
  for (int r = 0; r < R; ++r) hst[r] = StepState{0, pos0 + r, 1.0f, 0};
  float *dq = nullptr, *dr = nullptr, *dsink = nullptr, *dgate = nullptr, *dk = nullptr, *dv = nullptr, *dscr = nullptr;
  float *dout = nullptr, *dpmax = nullptr, *dpart = nullptr, *des = nullptr;
  unsigned short* drp = nullptr;
  StepState* dst = nullptr;
  unsigned* dcnt = nullptr;
  void* flush = nullptr;
  const size_t flush_bytes = (size_t)192 << 20;
  int rc = 0;
  cudaMalloc(&dq, nq * 4);
  cudaMalloc(&dr, nr * 4);
  cudaMalloc(&dsink, ns * 4);
  cudaMalloc(&dgate, nq * 4);
  cudaMalloc(&dk, nkvc * 4);
  cudaMalloc(&dv, nkvc * 4);
  cudaMalloc(&drp, nrp * 2);
  cudaMalloc(&dscr, nscr * 4);
  cudaMalloc(&dout, nq * 4);
  cudaMalloc(&dst, (size_t)R * sizeof(StepState));
  cudaMalloc(&dpmax, (size_t)R * nh * 32 * 4);
  cudaMalloc(&dpart, (size_t)R * nh * 32 * hd * 4);
  cudaMalloc(&des, (size_t)R * nh * 1024 * 4);
  cudaMalloc(&dcnt, (size_t)R * nkv * 4);
  if (reps > 0) cudaMalloc(&flush, flush_bytes);
  rc = (int)cudaGetLastError();
  if (rc) {
    out[7] = rc;
    return rc;
  }
  cudaMemset(dcnt, 0, (size_t)R * nkv * 4);
  cudaMemcpy(dq, hq.data(), nq * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dr, hr.data(), nr * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsink, hsink.data(), ns * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dgate, hgate.data(), nq * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk.data(), nkvc * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), nkvc * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(drp, hrp.data(), nrp * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dst, hst.data(), (size_t)R * sizeof(StepState), cudaMemcpyHostToDevice);
  AttnArgs a{};
  a.q = dq;
  a.r = dr;
  a.sink = dsink;
  a.gate = dgate;
  a.kc = dk;
  a.vc = dv;
  a.rel_proj = drp;
  a.scratch = dscr;
  a.out = dout;
  a.st = dst;
  a.nh = nh;
  a.nkv = nkv;
  a.hd = hd;
  a.drel = drel;
  a.rel_extent = rext;
  a.ring = ring;
  a.window = window;
  a.rows = R;
  a.scratch_stride = stride;
  a.scale = (float)pow((double)hd, -0.5);
  F2AttnArgs f{};
  f.a = a;
  f.pmax = dpmax;
  f.part = dpart;
  f.es = des;
  f.cnt = dcnt;
  f.sa = sa;
  f.sw = sw;
  f.pf = pf;
  f.ng = ng;
  F2Launch l{};
  std::vector<float> ref_out(nq), new_out(nq), ref_scr(nscr), new_scr(nscr);
  // general kernel
  cudaMemset(dscr, 0xFF, nscr * 4);
  cudaMemset(dout, 0xFF, nq * 4);
  kk_attn<<<nh * R, 1024>>>(a);
  cudaDeviceSynchronize();
  cudaMemcpy(ref_out.data(), dout, nq * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(ref_scr.data(), dscr, nscr * 4, cudaMemcpyDeviceToHost);
  // split
  cudaMemset(dscr, 0xFF, nscr * 4);
  cudaMemset(dout, 0xFF, nq * 4);
  rc = f2_attn_go(f, l, 0, false);
  cudaDeviceSynchronize();
  if (!rc) rc = (int)cudaGetLastError();
  cudaMemcpy(new_out.data(), dout, nq * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(new_scr.data(), dscr, nscr * 4, cudaMemcpyDeviceToHost);
  double nd_out = 0, nd_scr = 0, mad = 0;
  for (size_t i = 0; i < nq; ++i) {
    unsigned x, y;
    memcpy(&x, &ref_out[i], 4);
    memcpy(&y, &new_out[i], 4);
    if (x != y) {
      nd_out += 1;
      double d = fabs((double)ref_out[i] - (double)new_out[i]);
      if (d == d && d > mad) mad = d;
    }
  }
  for (size_t i = 0; i < nscr; ++i) {
    unsigned x, y;
    memcpy(&x, &ref_scr[i], 4);
    memcpy(&y, &new_scr[i], 4);
    if (x != y) nd_scr += 1;
  }
  out[0] = nd_out;
  out[1] = nd_scr;
  out[4] = mad;
  out[5] = (double)nq;
  out[6] = (double)nscr;
  // the counters must be back at 0
  {
    std::vector<unsigned> hc((size_t)R * nkv);
    cudaMemcpy(hc.data(), dcnt, hc.size() * 4, cudaMemcpyDeviceToHost);
    for (unsigned c : hc)
      if (c) out[1] += 1e9;  // flagged as a scratch difference
  }
  if (reps > 0 && !rc) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int pass = 0; pass < 2; ++pass) {
      double tot = 0;
      for (int r = -1; r < reps; ++r) {
        cudaMemsetAsync(flush, r & 0xFF, flush_bytes, 0);
        cudaEventRecord(e0, 0);
        if (pass == 0) kk_attn<<<nh * R, 1024>>>(a);
        else f2_attn_go(f, l, 0, false);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (r >= 0) tot += ms;
      }
      out[2 + pass] = tot * 1000.0 / reps;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }
  if (!rc) rc = (int)cudaGetLastError();
  out[7] = rc;
  cudaFree(dq);
  cudaFree(dr);
  cudaFree(dsink);
  cudaFree(dgate);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(drp);
  cudaFree(dscr);
  cudaFree(dout);
  cudaFree(dst);
  cudaFree(dpmax);
  cudaFree(dpart);
  cudaFree(des);
  cudaFree(dcnt);
  if (flush) cudaFree(flush);
  return rc;
}
