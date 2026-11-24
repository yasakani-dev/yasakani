// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Kernel bodies, written once in a small dialect and compiled twice:
//   ink2_cuda.cu : as CUDA kernels
//   ink2_emu.cpp : as host functions run by a thread-block emulator
// Dialect (defined by the including file):
//   KERNEL(name, Args)   kernel with the argument block "a"
//   KINL                 inline helper;  KCTXP / KCTXA  context parameter / argument of helpers
//   KTID KBID KNT        thread index in the block, block index, threads per block
//   KSYNC()              barrier of the block
//   KSHFL_DOWN(v, d)     value of lane + d of the same warp (own value past the end)
//   KSHFL(v, src)        value of lane src
//   KSHARED(T, name, n)  block-shared array (kernel scope only)
//   KLOAD64(p)           8 bytes, little endian (p 8-byte aligned);  KLOAD32(p)  4 bytes (p 4-byte aligned)
//   KF4, KLOADF4(p)      4 floats .x .y .z .w in one load (p 16-byte aligned); KF4ZERO
//   KFMA(a, b, c)        fused a * b + c;  KBITS2F(u)  float32 from its bits
//   KUNROLL / KNOUNROLL  unroll / do not unroll the next loop (CUDA only)
// Rules: a thread may leave early only together with its whole warp when the
// rest of the warp still shuffles; every block size is a multiple of 32.
// Plain a * b + c is never contracted (nvcc -fmad=false, MSVC /fp:precise).
// Shape of the kernels: a thread of the device waits for every load from
// device memory, so the small kernels read each input once, four values per
// load, keep them in registers and use as many threads as there are values.

KINL float k_bf(unsigned b) { return KBITS2F(b << 16); }
KINL float k_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
KINL float k_softcap(float x, float cap) { return tanhf(x / cap) * cap; }
// Table-free decoders of the packed formats. A code placed in the low bits of a float32 exponent is its
// value times a power of two (the subnormal codes included), so one multiplication gives the exact value:
//   E2M1 code (bits 0-3) -> twice its value (as k_e2m1x2);  E4M3 byte -> its value (NaN bytes 0x7F / 0xFF
//   give 480 instead; the quantiser never writes them);  exponent byte b -> 2 ** (b - 127).
KINL float k_fp4x2(unsigned c) { return KBITS2F(((c & 7u) << 22) | ((c & 8u) << 28)) * KBITS2F(0x7F000000u); }
KINL float k_e4m3v(unsigned b) { return KBITS2F(((b & 0x7Fu) << 20) | ((b & 0x80u) << 24)) * KBITS2F(0x7B800000u); }
KINL float k_exp2v(unsigned b) { return KBITS2F(b ? (b << 23) : 0x00400000u); }
// 3-bit code (bits 0-2 of c) of INK2_DFMT_Q3 -> q3[code] = +-1, +-3, +-5, +-7 (exact as float32).
KINL float k_q3v(unsigned c) {
  float m = (float)(((c & 3u) << 1) | 1u);
  return (c & 4u) ? -m : m;
}
// The 24-bit word of the 8 codes of columns c0 .. c0+7 (c0 % 8 == 0) of row r of a q3 matrix (three byte loads).
KINL unsigned k_q3word(const u8* wb, size_t r, int cols, int c0) {
  const u8* p = wb + r * (size_t)((cols >> 3) * 3) + (size_t)((c0 >> 3) * 3);
  return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16);
}

// Sum over the warp, returned to every lane.
KINL float k_warp_sum(KCTXP float v) {
  v += KSHFL_DOWN(v, 16);
  v += KSHFL_DOWN(v, 8);
  v += KSHFL_DOWN(v, 4);
  v += KSHFL_DOWN(v, 2);
  v += KSHFL_DOWN(v, 1);
  return KSHFL(v, 0);
}
// Two sums: lanes 0-15 and lanes 16-31, each returned to its own lanes.
KINL float k_half_sum(KCTXP float v) {
  v += KSHFL_DOWN(v, 8);
  v += KSHFL_DOWN(v, 4);
  v += KSHFL_DOWN(v, 2);
  v += KSHFL_DOWN(v, 1);
  return KSHFL(v, KTID & 16);
}
KINL float k_warp_max(KCTXP float v) {
  v = fmaxf(v, KSHFL_DOWN(v, 16));
  v = fmaxf(v, KSHFL_DOWN(v, 8));
  v = fmaxf(v, KSHFL_DOWN(v, 4));
  v = fmaxf(v, KSHFL_DOWN(v, 2));
  v = fmaxf(v, KSHFL_DOWN(v, 1));
  return KSHFL(v, 0);
}
// Sum / maximum over the block, returned to every thread. sm: shared, 33 floats.
KINL float k_block_sum(KCTXP float v, float* sm) {
  float w = k_warp_sum(KCTXA v);
  if ((KTID & 31) == 0) sm[KTID >> 5] = w;
  KSYNC();
  if (KTID == 0) {
    float t = 0.0f;
    int nw = KNT >> 5;
    for (int i = 0; i < nw; ++i) t += sm[i];
    sm[32] = t;
  }
  KSYNC();
  return sm[32];
}
KINL float k_block_max(KCTXP float v, float* sm) {
  float w = k_warp_max(KCTXA v);
  if ((KTID & 31) == 0) sm[KTID >> 5] = w;
  KSYNC();
  if (KTID == 0) {
    float t = sm[0];
    int nw = KNT >> 5;
    for (int i = 1; i < nw; ++i) t = fmaxf(t, sm[i]);
    sm[32] = t;
  }
  KSYNC();
  return sm[32];
}
// n <= 8 sums over the block at once (v[i] in, the totals out, to every
// thread). sm: shared, 33 * 8 floats.
KINL void k_block_sum_8(KCTXP float* v, int n, float* sm) {
  for (int i = 0; i < 8; ++i) {
    if (i < n) {
      float w = k_warp_sum(KCTXA v[i]);
      if ((KTID & 31) == 0) sm[(KTID >> 5) * 8 + i] = w;
    }
  }
  KSYNC();
  if (KTID < n) {
    float t = 0.0f;
    int nw = KNT >> 5;
    for (int w = 0; w < nw; ++w) t += sm[w * 8 + KTID];
    sm[32 * 8 + KTID] = t;
  }
  KSYNC();
  for (int i = 0; i < 8; ++i)
    if (i < n) v[i] = sm[32 * 8 + i];
}

// Part of a bfloat16 row times x that belongs to one thread: the 8-element
// units u = first, first + stride, ... (cols % 8 == 0). The input may come in
// two pieces: x below column split, x2 from there on.
KINL float k_bf_dot(const u16* wr, const float* x, const float* x2, int split, int cols, int first, int stride) {
  int nunit = cols >> 3;
  float acc = 0.0f;
  for (int u = first; u < nunit; u += stride) {
    const u16* p = wr + (u << 3);
    const float* xx = (u << 3) < split ? x + (u << 3) : x2 + ((u << 3) - split);
    u64 c0 = KLOAD64(p), c1 = KLOAD64(p + 4);
    KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
    acc = KFMA(k_bf((unsigned)(c0 & 0xffffu)), xa.x, acc);
    acc = KFMA(k_bf((unsigned)((c0 >> 16) & 0xffffu)), xa.y, acc);
    acc = KFMA(k_bf((unsigned)((c0 >> 32) & 0xffffu)), xa.z, acc);
    acc = KFMA(k_bf((unsigned)(c0 >> 48)), xa.w, acc);
    acc = KFMA(k_bf((unsigned)(c1 & 0xffffu)), xb.x, acc);
    acc = KFMA(k_bf((unsigned)((c1 >> 16) & 0xffffu)), xb.y, acc);
    acc = KFMA(k_bf((unsigned)((c1 >> 32) & 0xffffu)), xb.z, acc);
    acc = KFMA(k_bf((unsigned)(c1 >> 48)), xb.w, acc);
  }
  return acc;
}
KINL float k_epilogue(float s, int epi, float cap) {
  if (epi == INK2_EPI_SIGMOID) return k_sigmoid(s);
  if (epi == INK2_EPI_SOFTCAP) return k_softcap(s, cap);
  if (epi == INK2_EPI_TANH) return tanhf(s);
  return s;
}
// End of a row sum of a gemv job: the row factor (fac != 0: packed nvfp4, or the decoded nvfp4 values
// in bfloat16 without their matrix factor), then the epilogue.
KINL float k_jout(const GemvJob& J, float s) {
  if (J.fac != 0.0f) s = s * J.fac;
  return k_epilogue(s, J.epi, J.cap);
}

// Slot of position p in a history ring of hr slots (p may be negative).
KINL int k_slot(int p, int hr) { return ((p % hr) + hr) % hr; }
// Static short convolution of one channel (ink2_ref.short_conv for one new
// position): reads the K-1 earlier inputs, stores x as the newest one.
// History ring of hr slots. Row `row` of a multi-row pass reads the inputs of
// the earlier rows of the pass from xrow (the input row of this row, row
// stride C) instead of the ring; row 0 (and a single step) only reads the ring.
KINL float k_conv1(float* hist, int hr, const u16* w, const float* xrow, int row, float x, int c, int C, int K, int pos,
                   float div) {
  int K1 = K - 1;
  float o = 0.0f;
  for (int t = 0; t < K1; ++t) {
    int lag = K1 - t;
    float hv = row >= lag ? xrow[(size_t)c - (size_t)lag * (size_t)C] : hist[(size_t)k_slot(pos - lag, hr) * C + c];
    o = o + hv * k_bf(w[(size_t)c * K + t]);
  }
  o = o + x * k_bf(w[(size_t)c * K + K1]);
  o = (o + x) / div;
  hist[(size_t)(pos % hr) * C + c] = x;
  return o;
}
// The same arithmetic for K = 4 with the three earlier inputs h0 (oldest) ..
// h2 and the four taps of the channel in one 64-bit word.
KINL float k_conv_k4(float h0, float h1, float h2, float x, u64 taps, float div) {
  float o = 0.0f;
  o = o + h0 * k_bf((unsigned)(taps & 0xffffu));
  o = o + h1 * k_bf((unsigned)((taps >> 16) & 0xffffu));
  o = o + h2 * k_bf((unsigned)((taps >> 32) & 0xffffu));
  o = o + x * k_bf((unsigned)(taps >> 48));
  return (o + x) / div;
}

// ---------------------------------------------------------------------------
// gemv: one warp per row (for the large matrices)
// ---------------------------------------------------------------------------
KERNEL(gemv, GemvArgs) {
  int lane = KTID & 31;
  int row = KBID * (KNT >> 5) + (KTID >> 5);
  if (row >= a.total_rows) return;
  int j = 0;
  while (row >= a.job[j].rows) {
    row -= a.job[j].rows;
    ++j;
  }
  int cols = a.job[j].cols;
  float acc = k_bf_dot(a.job[j].w + (size_t)row * (size_t)cols, a.job[j].x, a.job[j].x2, a.job[j].split, cols, lane, 32);
  float s = k_warp_sum(KCTXA acc);
  if (lane == 0) a.job[j].y[row] = k_jout(a.job[j], s);
}

// ---------------------------------------------------------------------------
// gemvw: one block per row (for few long rows, where the time of one row counts)
// ---------------------------------------------------------------------------
KERNEL(gemvw, GemvArgs) {
  KSHARED(float, sm, 33);
  int row = KBID;
  int j = 0;
  while (row >= a.job[j].rows) {
    row -= a.job[j].rows;
    ++j;
  }
  int cols = a.job[j].cols;
  float acc = k_bf_dot(a.job[j].w + (size_t)row * (size_t)cols, a.job[j].x, a.job[j].x2, a.job[j].split, cols, KTID, KNT);
  float s = k_block_sum(KCTXA acc, sm);
  if (KTID == 0) a.job[j].y[row] = k_jout(a.job[j], s);
}

// ---------------------------------------------------------------------------
// norm: RMS norm (optionally of a bfloat16 row) with optional weight; embedding rows of image positions
// are copied from the feature buffer instead (NormArgs.img)
// ---------------------------------------------------------------------------
KERNEL(norm, NormArgs) {
  KSHARED(float, sm, 33);
  int jb = KBID, row = 0;
  if (a.nj > 0) {
    jb = KBID % a.nj;
    row = KBID / a.nj;
  }
  const float* xf = (const float*)a.job[jb].x + (size_t)row * (size_t)a.n;
  const u16* xb = (const u16*)a.job[jb].x;
  const u8* x8 = 0;  // packed table row (mxfp8) and its exponents
  const u8* e8 = 0;
  if (a.embed_stride > 0 && jb == 0) {
    const int pad = a.img ? a.st[row].pad : 0;
    if (pad > 0) {  // image position: the image vector is the embedding (already normalised by the tower)
      const u16* f = a.img + (size_t)(pad - 1) * (size_t)a.n;
      float* o = a.job[0].out + (size_t)row * (size_t)a.n;
      for (int j = KTID; j < a.n; j += KNT) o[j] = k_bf(f[j]);
      return;  // the whole block returns: no barrier is left waiting
    }
    size_t tok = (size_t)a.st[row].token;
    if (a.efmt == INK2_DFMT_MXFP8) {
      x8 = (const u8*)a.job[0].x + tok * (size_t)a.embed_stride;
      e8 = a.esc + tok * (size_t)(a.embed_stride >> 5);
    }
    xb += tok * (size_t)a.embed_stride;
  }
  int isbf = a.job[jb].x_bf16;
  const u16* w = a.job[jb].w;
  float* out = a.job[jb].out + (size_t)row * (size_t)a.n;
  float ss = 0.0f;
  for (int j = KTID; j < a.n; j += KNT) {
    float v = x8 ? k_e4m3v(x8[j]) * k_exp2v(e8[j >> 5]) : isbf ? k_bf(xb[j]) : xf[j];
    ss += v * v;
  }
  float tot = k_block_sum(KCTXA ss, sm);
  float scv = 1.0f / sqrtf(tot / (float)a.n + a.eps);
  for (int j = KTID; j < a.n; j += KNT) {
    float v = (x8 ? k_e4m3v(x8[j]) * k_exp2v(e8[j >> 5]) : isbf ? k_bf(xb[j]) : xf[j]) * scv;
    if (w) v = k_bf(w[j]) * v;
    out[j] = v;
  }
}

// ---------------------------------------------------------------------------
// bar: every query attends over the RMS-normalised states. One block per job,
// H / 4 threads (rounded up to whole warps), thread t owns the elements
// 4t .. 4t+3. Optional first step: the last state is produced here by a short
// convolution (+ residual). Per output: RMS norm with weight in place, or an
// extra RMS-normalised copy.
// ---------------------------------------------------------------------------
#define K_BAR_STATES 6
KERNEL(bar, BarArgs) {
  KSHARED(float, sm, 33 * 8);
  int H = a.H;
  int jb = KBID, row = 0;
  if (a.nj > 0) {
    jb = KBID % a.nj;
    row = KBID / a.nj;
  }
  const BarJob& J = a.job[jb];
  size_t ro = (size_t)row * (size_t)H;
  int ns = J.ns, nq = J.nq;
  int j0 = KTID * 4;
  int act = j0 < H;
  int conv = J.cw != 0;
  float sx[K_BAR_STATES], sy[K_BAR_STATES], sz[K_BAR_STATES], sw[K_BAR_STATES];
  for (int i = 0; i < K_BAR_STATES; ++i) {
    sx[i] = sy[i] = sz[i] = sw[i] = 0.0f;
    if (act && i < ns && !(conv && i == ns - 1)) {
      KF4 t = KLOADF4(J.st[i] + ro + j0);
      sx[i] = t.x;
      sy[i] = t.y;
      sz[i] = t.z;
      sw[i] = t.w;
    }
  }
  if (conv && act) {
    int pos = a.st[row].pos;
    int K = a.K;
    int hr = a.hr > 0 ? a.hr : K - 1;
    float* hist = J.chist;
    const u16* cw = J.cw;
    const float* cres = J.cres ? J.cres + ro : 0;
    float* cout = J.cout + ro;
    const float* cxr = J.cx + ro;
    KF4 x = KLOADF4(cxr + j0);
    float ox, oy, oz, ow;
    if (K == 4) {
      // inputs 3, 2 and 1 steps back: from the pass when they belong to it, else from the ring
      KF4 h0 = row >= 3 ? KLOADF4(cxr - 3 * (size_t)H + j0) : KLOADF4(hist + (size_t)k_slot(pos - 3, hr) * H + j0);
      KF4 h1 = row >= 2 ? KLOADF4(cxr - 2 * (size_t)H + j0) : KLOADF4(hist + (size_t)k_slot(pos - 2, hr) * H + j0);
      KF4 h2 = row >= 1 ? KLOADF4(cxr - (size_t)H + j0) : KLOADF4(hist + (size_t)k_slot(pos - 1, hr) * H + j0);
      const u16* wp = cw + (size_t)j0 * 4;
      ox = k_conv_k4(h0.x, h1.x, h2.x, x.x, KLOAD64(wp), a.cdiv);
      oy = k_conv_k4(h0.y, h1.y, h2.y, x.y, KLOAD64(wp + 4), a.cdiv);
      oz = k_conv_k4(h0.z, h1.z, h2.z, x.z, KLOAD64(wp + 8), a.cdiv);
      ow = k_conv_k4(h0.w, h1.w, h2.w, x.w, KLOAD64(wp + 12), a.cdiv);
      float* hw = hist + (size_t)(pos % hr) * H + j0;
      hw[0] = x.x;
      hw[1] = x.y;
      hw[2] = x.z;
      hw[3] = x.w;
    } else {
      ox = k_conv1(hist, hr, cw, cxr, row, x.x, j0, H, K, pos, a.cdiv);
      oy = k_conv1(hist, hr, cw, cxr, row, x.y, j0 + 1, H, K, pos, a.cdiv);
      oz = k_conv1(hist, hr, cw, cxr, row, x.z, j0 + 2, H, K, pos, a.cdiv);
      ow = k_conv1(hist, hr, cw, cxr, row, x.w, j0 + 3, H, K, pos, a.cdiv);
    }
    if (cres) {
      KF4 r = KLOADF4(cres + j0);
      ox = r.x + ox;
      oy = r.y + oy;
      oz = r.z + oz;
      ow = r.w + ow;
    }
    cout[j0] = ox;
    cout[j0 + 1] = oy;
    cout[j0 + 2] = oz;
    cout[j0 + 3] = ow;
    for (int i = 0; i < K_BAR_STATES; ++i) {
      if (i == ns - 1) {
        sx[i] = ox;
        sy[i] = oy;
        sz[i] = oz;
        sw[i] = ow;
      }
    }
  }
  float v[8];
  for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  for (int i = 0; i < K_BAR_STATES; ++i) {
    float ss = sx[i] * sx[i];
    ss += sy[i] * sy[i];
    ss += sz[i] * sz[i];
    ss += sw[i] * sw[i];
    v[i] = ss;
  }
  k_block_sum_8(KCTXA v, ns, sm);
  for (int i = 0; i < K_BAR_STATES; ++i) {
    if (i < ns) {
      float f = 1.0f / sqrtf(v[i] / (float)H + 1.0e-6f);
      sx[i] = sx[i] * f;
      sy[i] = sy[i] * f;
      sz[i] = sz[i] * f;
      sw[i] = sw[i] * f;
    }
  }
  for (int qi = 0; qi < nq; ++qi) {
    float qx = 0.0f, qy = 0.0f, qz = 0.0f, qw = 0.0f;
    if (act) {
      KF4 q = KLOADF4(J.q + (size_t)qi * H + j0);
      qx = q.x;
      qy = q.y;
      qz = q.z;
      qw = q.w;
    }
    for (int i = 0; i < K_BAR_STATES; ++i) {
      float d = KFMA(qx, sx[i], 0.0f);
      d = KFMA(qy, sy[i], d);
      d = KFMA(qz, sz[i], d);
      d = KFMA(qw, sw[i], d);
      v[i] = d;
    }
    k_block_sum_8(KCTXA v, ns, sm);
    // softmax over the states (every thread computes the same values)
    float m = v[0] * a.hscale;
    for (int i = 0; i < K_BAR_STATES; ++i) {
      if (i < ns) {
        v[i] = v[i] * a.hscale;
        m = fmaxf(m, v[i]);
      }
    }
    float sum = 0.0f;
    for (int i = 0; i < K_BAR_STATES; ++i) {
      if (i < ns) {
        v[i] = expf(v[i] - m);
        sum += v[i];
      }
    }
    float ox = 0.0f, oy = 0.0f, oz = 0.0f, ow = 0.0f;
    for (int i = 0; i < K_BAR_STATES; ++i) {
      if (i < ns) {
        float wgt = v[i] / sum;
        ox = KFMA(wgt, sx[i], ox);
        oy = KFMA(wgt, sy[i], oy);
        oz = KFMA(wgt, sz[i], oz);
        ow = KFMA(wgt, sw[i], ow);
      }
    }
    float* o = J.out[qi] + ro;
    const u16* w = J.normw[qi];
    float* on = J.outn[qi] ? J.outn[qi] + ro : 0;
    if (w || on) {
      float ss = ox * ox;
      ss += oy * oy;
      ss += oz * oz;
      ss += ow * ow;
      float tot = k_block_sum(KCTXA ss, sm);
      float scv = 1.0f / sqrtf(tot / (float)H + a.eps);
      if (act) {
        float nx = ox * scv, ny = oy * scv, nz = oz * scv, nw4 = ow * scv;
        if (on) {
          on[j0] = nx;
          on[j0 + 1] = ny;
          on[j0 + 2] = nz;
          on[j0 + 3] = nw4;
        }
        if (w) {
          u64 wv = KLOAD64(w + j0);
          ox = k_bf((unsigned)(wv & 0xffffu)) * nx;
          oy = k_bf((unsigned)((wv >> 16) & 0xffffu)) * ny;
          oz = k_bf((unsigned)((wv >> 32) & 0xffffu)) * nz;
          ow = k_bf((unsigned)(wv >> 48)) * nw4;
        }
      }
    }
    if (act) {
      o[j0] = ox;
      o[j0 + 1] = oy;
      o[j0 + 2] = oz;
      o[j0 + 3] = ow;
    }
  }
}

// ---------------------------------------------------------------------------
// gmix: last step of a message gate, (1 - f) * hs + g * msg, optional norm.
// (The gate projections Down, Forget and Gain are gemv jobs.) One block per
// job, H / 4 threads, thread t owns the elements 4t .. 4t+3.
// ---------------------------------------------------------------------------
KERNEL(gmix, GmixArgs) {
  KSHARED(float, sm, 33);
  int jb = KBID, row = 0;
  if (a.nj > 0) {
    jb = KBID % a.nj;
    row = KBID / a.nj;
  }
  int H = a.H;
  size_t ro = (size_t)row * (size_t)H;
  const u16* normw = a.job[jb].normw;
  float* out = a.job[jb].out + ro;
  int j0 = KTID * 4;
  int act = j0 < H;
  float ox = 0.0f, oy = 0.0f, oz = 0.0f, ow = 0.0f;
  if (act) {
    KF4 hs = KLOADF4(a.job[jb].hs + ro + j0);
    KF4 msg = KLOADF4(a.job[jb].msg + ro + j0);
    KF4 f = KLOADF4(a.job[jb].f + ro + j0);
    KF4 g = KLOADF4(a.job[jb].g + ro + j0);
    ox = (1.0f - f.x) * hs.x + g.x * msg.x;
    oy = (1.0f - f.y) * hs.y + g.y * msg.y;
    oz = (1.0f - f.z) * hs.z + g.z * msg.z;
    ow = (1.0f - f.w) * hs.w + g.w * msg.w;
  }
  if (normw) {
    float ss = ox * ox;
    ss += oy * oy;
    ss += oz * oz;
    ss += ow * ow;
    float tot = k_block_sum(KCTXA ss, sm);
    float scv = 1.0f / sqrtf(tot / (float)H + a.eps);
    if (act) {
      u64 wv = KLOAD64(normw + j0);
      ox = k_bf((unsigned)(wv & 0xffffu)) * (ox * scv);
      oy = k_bf((unsigned)((wv >> 16) & 0xffffu)) * (oy * scv);
      oz = k_bf((unsigned)((wv >> 32) & 0xffffu)) * (oz * scv);
      ow = k_bf((unsigned)(wv >> 48)) * (ow * scv);
    }
  }
  if (act) {
    out[j0] = ox;
    out[j0 + 1] = oy;
    out[j0 + 2] = oz;
    out[j0 + 3] = ow;
  }
}

// ---------------------------------------------------------------------------
// prep: one warp per head vector. Warps [0, nh): queries, [nh, 2nh): relative
// states, then nkv key heads, then nkv value heads.
// ---------------------------------------------------------------------------
#include "ink2_kv_codec.inc"

KERNEL(prep, PrepArgs) {
  int lane = KTID & 31;
  int row = 0, b = KBID;
  if (a.bpr > 0) {
    row = KBID / a.bpr;
    b = KBID - row * a.bpr;
  }
  int gw = b * (KNT >> 5) + (KTID >> 5);
  int nh = a.nh, nkv = a.nkv;
  if (gw >= 2 * nh + 2 * nkv) return;
  int pos = a.st[row].pos;
  if (gw < 2 * nh) {
    int isr = gw >= nh;
    int h = isr ? gw - nh : gw;
    int n = isr ? a.drel : a.hd;
    float* v = (isr ? a.r + (size_t)row * nh * a.drel : a.q + (size_t)row * nh * a.hd) + (size_t)h * n;
    const u16* scl = (isr ? a.r_scale : a.q_scale) + (size_t)h * n;
    float thr = isr ? a.rel_thr : a.qk_thr;
    float tau = a.st[row].tau;
    float ss = 0.0f;
    for (int j = lane; j < n; j += 32) {
      float x = v[j] * k_bf(scl[j]);
      v[j] = x;
      ss += x * x;
    }
    float tot = k_warp_sum(KCTXA ss);
    float f = thr / (sqrtf(tot) + 1.0e-6f);
    if (f > 1.0f) f = 1.0f;
    for (int j = lane; j < n; j += 32) {
      float x = v[j] * f;
      if (!a.sliding) x = x * tau;
      v[j] = x;
    }
    return;
  }
  int isv = gw >= 2 * nh + nkv;
  int hk = gw - 2 * nh - (isv ? nkv : 0);
  int hd = a.hd, K = a.K, K1 = a.K - 1;
  int C = nkv * hd;
  int hr = a.hr > 0 ? a.hr : K1;
  int cslot = a.ring ? pos % a.ring : pos;
  const float* raw = (isv ? a.vraw : a.kraw) + (size_t)row * C;
  float* hist = isv ? a.vhist : a.khist;
  float* dst = a.kv.mode ? nullptr : (isv ? a.vc : a.kc) + (size_t)cslot * C;
  float values[4];  // both attention head sizes are 128
  const u16* w = isv ? a.v_w : a.k_w;
  const float* dyn = (isv ? a.v_dyn : a.k_dyn) + (a.sliding ? 0 : (size_t)row * nkv * K);
  float ss = 0.0f;
  for (int j = lane; j < hd; j += 32) {
    int c = hk * hd + j;
    float x = raw[c];
    float o = 0.0f;
    // the input lag steps back: from the pass when it belongs to it, else from the ring
    if (a.sliding) {
      for (int t = 0; t < K1; ++t) {
        int lag = K1 - t;
        float hv = row >= lag ? raw[(size_t)c - (size_t)lag * C] : hist[(size_t)k_slot(pos - lag, hr) * C + c];
        o = o + hv * k_bf(w[(size_t)c * K + t]);
      }
      o = o + x * k_bf(w[(size_t)c * K + K1]);
    } else {
      o = x * (k_bf(w[c]) + dyn[hk * K]);
      for (int lag = 1; lag < K; ++lag) {
        float hv = row >= lag ? raw[(size_t)c - (size_t)lag * C] : hist[(size_t)k_slot(pos - lag, hr) * C + c];
        o = o + hv * (k_bf(w[(size_t)lag * C + c]) + dyn[hk * K + lag]);
      }
    }
    o = (o + x) / a.conv_div;
    hist[(size_t)(pos % hr) * C + c] = x;
    values[j >> 5] = o;
    ss += o * o;
  }
  float tot = k_warp_sum(KCTXA ss);
  float f;
  if (isv) {
    f = 1.0f / sqrtf(tot / (float)hd + a.eps);
  } else {
    f = a.qk_thr / (sqrtf(tot) + 1.0e-6f);
    if (f > 1.0f) f = 1.0f;
  }
  for (int j = lane; j < hd; j += 32) {
    int c = hk * hd + j;
    float x = values[j >> 5] * f;
    if (dst) dst[c] = x;
    else kv_store(KCTXA a.kv, isv, pos, cslot, C, c, x);
  }
}

// ---------------------------------------------------------------------------
// attn: one block of 1024 threads per head; head size 128, d_rel <= 32.
// Keys t = 0 .. T-1 are the positions pos+1-T .. pos (T = all, or the window).
//   1. one warp per key, lane l owns the elements 4l .. 4l+3 of the head:
//      scaled dot product + relative bias -> logit
//   2. softmax over the logits and the sink
//   3. weighted sum of the values in 32 stripes of keys, output gate
// ---------------------------------------------------------------------------
KERNEL(attn, AttnArgs) {
  KSHARED(float, sm, 33);
  KSHARED(float, part, 4096);
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int h = KBID, row = 0;
  if (a.rows > 0) {
    row = KBID / a.nh;
    h = KBID - row * a.nh;
  }
  int hd = a.hd;
  int kh = h / (a.nh / a.nkv);
  int C = a.nkv * hd;
  int pos = a.st[row].pos;
  int T = pos + 1;
  int win = a.window > 0 ? a.window : a.ring;  // (old argument blocks: the ring is the window)
  if (win && T > win) T = win;
  int slot0 = pos + 1 - T;
  if (a.ring) slot0 = slot0 % a.ring;
  size_t qo = (size_t)row * a.nh * hd;
  float* lg = a.scratch + ((size_t)row * a.nh + h) * a.scratch_stride;
  const float* kbase = a.kc + (size_t)kh * hd + lane * 4;
  const float* vbase = a.vc + (size_t)kh * hd + lane * 4;
  KF4 q = KLOADF4(a.q + qo + (size_t)h * hd + lane * 4);
  float rr = lane < a.drel ? a.r[(size_t)row * a.nh * a.drel + (size_t)h * a.drel + lane] : 0.0f;
  // transposed table: stride drel per distance; original: stride 1
  const u16* rp = a.rel_proj_t ? a.rel_proj_t + (lane < a.drel ? lane : 0) : a.rel_proj + (size_t)(lane < a.drel ? lane : 0) * a.rel_extent;
  const int rs = a.rel_proj_t ? a.drel : 1;
  float mx = -3.0e38f;
  for (int t = warp; t < T; t += nw) {
    int slot = slot0 + t;
    if (a.ring && slot >= a.ring) slot -= a.ring;
    KF4 k = KLOADF4(kbase + (size_t)slot * C);
    float acc = KFMA(q.x, k.x, 0.0f);
    acc = KFMA(q.y, k.y, acc);
    acc = KFMA(q.z, k.z, acc);
    acc = KFMA(q.w, k.w, acc);
    float v = acc * a.scale;
    int dist = T - 1 - t;
    if (dist < a.rel_extent && lane < a.drel) v = KFMA(rr, k_bf(rp[(size_t)dist * rs]), v);
    float l = k_warp_sum(KCTXA v);
    if (lane == 0) lg[t] = l;
    mx = fmaxf(mx, l);
  }
  float m = k_block_max(KCTXA mx, sm);
  float sink = a.sink[(size_t)row * a.nh + h];
  m = fmaxf(m, sink);
  float es = 0.0f;
  for (int t = KTID; t < T; t += KNT) {
    float e = expf(lg[t] - m);
    lg[t] = e;
    es += e;
  }
  float tot = k_block_sum(KCTXA es, sm) + expf(sink - m);
  // stripe = warp: the keys t = warp, warp + nw, ...; lane l the elements 4l .. 4l+3
  float ox = 0.0f, oy = 0.0f, oz = 0.0f, ow = 0.0f;
  for (int t = warp; t < T; t += nw) {
    int slot = slot0 + t;
    if (a.ring && slot >= a.ring) slot -= a.ring;
    float wgt = lg[t];
    KF4 val = KLOADF4(vbase + (size_t)slot * C);
    ox = KFMA(wgt, val.x, ox);
    oy = KFMA(wgt, val.y, oy);
    oz = KFMA(wgt, val.z, oz);
    ow = KFMA(wgt, val.w, ow);
  }
  float* pp = part + (size_t)KTID * 4;
  pp[0] = ox;
  pp[1] = oy;
  pp[2] = oz;
  pp[3] = ow;
  KSYNC();
  if (warp == 0) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int c = 0; c < nw; ++c) {
      const float* p = part + (size_t)(c * 32 + lane) * 4;
      s0 += p[0];
      s1 += p[1];
      s2 += p[2];
      s3 += p[3];
    }
    KF4 g = KLOADF4(a.gate + qo + (size_t)h * hd + lane * 4);
    float* o = a.out + qo + (size_t)h * hd + lane * 4;
    o[0] = (s0 / tot) * g.x;
    o[1] = (s1 / tot) * g.y;
    o[2] = (s2 / tot) * g.z;
    o[3] = (s3 / tot) * g.w;
  }
}

// ---------------------------------------------------------------------------
#include "ink2_kv_attention.inc"
#include "ink2_splitk_attention.inc"
#include "ink2_flash_attn.inc"

// topk: top-k of sigmoid(logit) + bias, weights from sigmoid(logit).
// One block of at least E threads.
// ---------------------------------------------------------------------------
KERNEL(topk, RouterArgs) {
  KSHARED(float, scv, 1024);
  KSHARED(float, raw, 1024);
  KSHARED(int, flag, 1024);
  int i = KTID;
  int E = a.E;
  int row = a.rows > 0 ? KBID : 0;
  int* oidx = a.idx + (size_t)row * a.k;
  float* owt = a.wt + (size_t)row * a.k;
  if (i < E) {
    raw[i] = k_sigmoid(a.logits[(size_t)row * E + i]);
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
  // every rank is known now, so scv is free: its first k entries take the weights
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
}

// ---------------------------------------------------------------------------
// Experts. A row is cut into 16-element units; 16 lanes share a row, lane l
// owns the units u = l, l + 16, ...
//   nvfp4: unit = 8 bytes of codes (element i in bits 4i..4i+3) and one E4M3
//          block scale; value = e2m1[code] * e4m3[scale] * factor(expert)
//   mxfp8: unit = 16 E4M3 bytes; one exponent byte per 2 units;
//          value = e4m3[byte] * 2 ** (exponent - 127)
// The block scale is applied to the sum of a unit, the expert factor to the
// sum of the row.
// ---------------------------------------------------------------------------
KINL float k_e2m1x2(unsigned code) {
  float v = (float)(int)((0xC8643210u >> ((code & 7u) << 2)) & 15u);
  return (code & 8u) ? -v : v;
}
// four consecutive elements: nvfp4 codes at bit sh of c, mxfp8 bytes at bit sh of c
#define K_NV4(c, sh, X)                                              \
  s = KFMA(k_e2m1x2((unsigned)((c) >> (sh)) & 15u), (X).x, s);       \
  s = KFMA(k_e2m1x2((unsigned)((c) >> ((sh) + 4)) & 15u), (X).y, s); \
  s = KFMA(k_e2m1x2((unsigned)((c) >> ((sh) + 8)) & 15u), (X).z, s); \
  s = KFMA(k_e2m1x2((unsigned)((c) >> ((sh) + 12)) & 15u), (X).w, s);
#define K_MX4(c, sh, X)                                            \
  s = KFMA(e4m3[(unsigned)((c) >> (sh)) & 255u], (X).x, s);        \
  s = KFMA(e4m3[(unsigned)((c) >> ((sh) + 8)) & 255u], (X).y, s);  \
  s = KFMA(e4m3[(unsigned)((c) >> ((sh) + 16)) & 255u], (X).z, s); \
  s = KFMA(e4m3[(unsigned)((c) >> ((sh) + 24)) & 255u], (X).w, s);

KINL float k_exp_lane_dot(const u8* wrow, const u8* srow, const float* x, int cols, int fmt, int l16,
                          const float* e4m3, const float* exp2t) {
  float acc = 0.0f;
  int nunit = cols >> 4;
  if (fmt == INK2_FMT_NVFP4) {
    for (int u = l16; u < nunit; u += 16) {
      const float* xx = x + (u << 4);
      u64 c = KLOAD64(wrow + (u << 3));
      KF4 x0 = KLOADF4(xx), x1 = KLOADF4(xx + 4), x2 = KLOADF4(xx + 8), x3 = KLOADF4(xx + 12);
      float s = 0.0f;
      K_NV4(c, 0, x0)
      K_NV4(c, 16, x1)
      K_NV4(c, 32, x2)
      K_NV4(c, 48, x3)
      acc = KFMA(s, e4m3[srow[u]], acc);
    }
  } else {
    for (int u = l16; u < nunit; u += 16) {
      const float* xx = x + (u << 4);
      u64 c0 = KLOAD64(wrow + (u << 4)), c1 = KLOAD64(wrow + (u << 4) + 8);
      KF4 x0 = KLOADF4(xx), x1 = KLOADF4(xx + 4), x2 = KLOADF4(xx + 8), x3 = KLOADF4(xx + 12);
      float s = 0.0f;
      K_MX4(c0, 0, x0)
      K_MX4(c0, 32, x1)
      K_MX4(c1, 0, x2)
      K_MX4(c1, 32, x3)
      acc = KFMA(s, exp2t[srow[u >> 1]], acc);
    }
  }
  return acc;
}

// w13: one warp per (routed expert, row pair); lanes 0-15 the gate row 2i,
// lanes 16-31 the up row 2i + 1. Writes the activation.
KERNEL(exp13, ExpArgs) {
  int lane = KTID & 31;
  int gw = KBID * (KNT >> 5) + (KTID >> 5);
  int npair = a.rows >> 1;
  if (gw >= a.k * npair) return;
  int e = gw / npair, i = gw - e * npair;
  int id = a.idx[e];
  size_t row = (size_t)id * a.rows + (size_t)(2 * i + (lane >> 4));
  size_t wbytes = a.fmt == INK2_FMT_NVFP4 ? (size_t)(a.cols >> 1) : (size_t)a.cols;
  size_t sbytes = a.fmt == INK2_FMT_NVFP4 ? (size_t)(a.cols >> 4) : (size_t)(a.cols >> 5);
  float acc = k_exp_lane_dot(a.w + row * wbytes, a.sc + row * sbytes, a.x, a.cols, a.fmt, lane & 15, a.e4m3, a.exp2t);
  float s = k_half_sum(KCTXA acc);
  if (a.fmt == INK2_FMT_NVFP4) s = s * a.factor[id];
  float up = KSHFL_DOWN(s, 16);
  if (lane == 0) a.y[(size_t)e * npair + i] = k_softcap(s, a.gate_cap) * k_sigmoid(s) * k_softcap(up, a.up_cap);
}

// w2: one block of 16 * k threads per output row; 16 lanes per routed expert.
// Optional last step: short convolution of the output + residual.
KERNEL(exp2, ExpArgs) {
  KSHARED(float, dots, 64);
  int e = KTID >> 4;
  int h = KBID;
  int id = a.idx[e];
  size_t row = (size_t)id * a.rows + (size_t)h;
  size_t wbytes = a.fmt == INK2_FMT_NVFP4 ? (size_t)(a.cols >> 1) : (size_t)a.cols;
  size_t sbytes = a.fmt == INK2_FMT_NVFP4 ? (size_t)(a.cols >> 4) : (size_t)(a.cols >> 5);
  float acc = k_exp_lane_dot(a.w + row * wbytes, a.sc + row * sbytes, a.x + (size_t)e * a.cols, a.cols, a.fmt,
                             KTID & 15, a.e4m3, a.exp2t);
  float s = k_half_sum(KCTXA acc);
  if (a.fmt == INK2_FMT_NVFP4) s = s * a.factor[id];
  if ((KTID & 15) == 0) dots[e] = s;
  KSYNC();
  if (KTID == 0) {
    float o = 0.0f;
    for (int j = 0; j < a.k; ++j) o = o + dots[j] * a.wt[j];
    a.y[h] = o;
    if (a.cw) {
      float cv = k_conv1(a.chist, a.hr > 0 ? a.hr : a.K - 1, a.cw, 0, 0, o, h, a.rows, a.K, a.st->pos, a.cdiv);
      a.cout[h] = a.cres[h] + cv;
    }
  }
}

// ---------------------------------------------------------------------------
// "Column owner" forms of the two largest products. A thread keeps the x
// values of its columns in registers and walks down the rows of its block, so
// that every load from device memory brings new weight bytes (one coalesced
// load per warp and row) instead of x values.
// ---------------------------------------------------------------------------

// gemvc: one block per chunk of K_GEMVC_ROWS rows of one job; all jobs have
// the same cols; cols / 8 threads, thread t owns the columns 8t .. 8t+7.
#define K_GEMVC_ROWS 16
KERNEL(gemvc, GemvArgs) {
  KSHARED(float, sm, K_GEMVC_ROWS * 32);
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int b = KBID, j = 0;
  for (;;) {
    int nb = (a.job[j].rows + K_GEMVC_ROWS - 1) / K_GEMVC_ROWS;
    if (b < nb) break;
    b -= nb;
    ++j;
  }
  int cols = a.job[j].cols;
  int row0 = b * K_GEMVC_ROWS;
  int nrows = a.job[j].rows - row0;
  if (nrows > K_GEMVC_ROWS) nrows = K_GEMVC_ROWS;
  int c0 = KTID * 8;
  int act = c0 < cols;
  KF4 xa = KF4ZERO, xb = KF4ZERO;
  if (act) {
    const float* xx = c0 < a.job[j].split ? a.job[j].x + c0 : a.job[j].x2 + (c0 - a.job[j].split);
    xa = KLOADF4(xx);
    xb = KLOADF4(xx + 4);
  }
  const u16* wp = a.job[j].w + (size_t)row0 * (size_t)cols + (act ? c0 : 0);
  for (int r = 0; r < nrows; ++r) {
    float acc = 0.0f;
    if (act) {
      const u16* p = wp + (size_t)r * (size_t)cols;
      u64 w0 = KLOAD64(p), w1 = KLOAD64(p + 4);
      acc = KFMA(k_bf((unsigned)(w0 & 0xffffu)), xa.x, acc);
      acc = KFMA(k_bf((unsigned)((w0 >> 16) & 0xffffu)), xa.y, acc);
      acc = KFMA(k_bf((unsigned)((w0 >> 32) & 0xffffu)), xa.z, acc);
      acc = KFMA(k_bf((unsigned)(w0 >> 48)), xa.w, acc);
      acc = KFMA(k_bf((unsigned)(w1 & 0xffffu)), xb.x, acc);
      acc = KFMA(k_bf((unsigned)((w1 >> 16) & 0xffffu)), xb.y, acc);
      acc = KFMA(k_bf((unsigned)((w1 >> 32) & 0xffffu)), xb.z, acc);
      acc = KFMA(k_bf((unsigned)(w1 >> 48)), xb.w, acc);
    }
    float s = k_warp_sum(KCTXA acc);
    if (lane == 0) sm[r * 32 + warp] = s;
  }
  KSYNC();
  if (KTID < nrows) {
    float t = 0.0f;
    for (int w = 0; w < nw; ++w) t += sm[KTID * 32 + w];
    a.job[j].y[row0 + KTID] = k_jout(a.job[j], t);
  }
}

// exp13c: one block per (routed expert, K_EXPC_PAIRS row pairs); cols / 16
// threads (rounded up to whole warps), thread t owns the 16-element unit t.
#define K_EXPC_PAIRS 8
KERNEL(exp13c, ExpArgs) {
  KSHARED(float, sm, 2 * K_EXPC_PAIRS * 32);
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int npair = a.rows >> 1;
  int nb = (npair + K_EXPC_PAIRS - 1) / K_EXPC_PAIRS;
  int e = KBID / nb;
  int p0 = (KBID - e * nb) * K_EXPC_PAIRS;
  int np = npair - p0;
  if (np > K_EXPC_PAIRS) np = K_EXPC_PAIRS;
  int id = a.idx[e];
  int u = KTID;
  int act = u < (a.cols >> 4);
  int nv = a.fmt == INK2_FMT_NVFP4;
  const float* e4m3 = a.e4m3;
  KF4 x0 = KF4ZERO, x1 = KF4ZERO, x2 = KF4ZERO, x3 = KF4ZERO;
  if (act) {
    const float* xx = a.x + (u << 4);
    x0 = KLOADF4(xx);
    x1 = KLOADF4(xx + 4);
    x2 = KLOADF4(xx + 8);
    x3 = KLOADF4(xx + 12);
  }
  size_t wbytes = nv ? (size_t)(a.cols >> 1) : (size_t)a.cols;
  size_t sbytes = nv ? (size_t)(a.cols >> 4) : (size_t)(a.cols >> 5);
  size_t row0 = (size_t)id * a.rows + (size_t)(2 * p0);
  const u8* wp = a.w + row0 * wbytes + (act ? (nv ? (u << 3) : (u << 4)) : 0);
  const u8* sp = a.sc + row0 * sbytes + (act ? (nv ? u : (u >> 1)) : 0);
  for (int r = 0; r < 2 * np; ++r) {
    float part = 0.0f;
    if (act) {
      float s = 0.0f;
      if (nv) {
        u64 c = KLOAD64(wp + (size_t)r * wbytes);
        K_NV4(c, 0, x0)
        K_NV4(c, 16, x1)
        K_NV4(c, 32, x2)
        K_NV4(c, 48, x3)
        part = s * e4m3[sp[(size_t)r * sbytes]];
      } else {
        u64 c0 = KLOAD64(wp + (size_t)r * wbytes), c1 = KLOAD64(wp + (size_t)r * wbytes + 8);
        K_MX4(c0, 0, x0)
        K_MX4(c0, 32, x1)
        K_MX4(c1, 0, x2)
        K_MX4(c1, 32, x3)
        part = s * a.exp2t[sp[(size_t)r * sbytes]];
      }
    }
    float rs = k_warp_sum(KCTXA part);
    if (lane == 0) sm[r * 32 + warp] = rs;
  }
  KSYNC();
  if (KTID < np) {
    float g = 0.0f, up = 0.0f;
    for (int w = 0; w < nw; ++w) {
      g += sm[(2 * KTID) * 32 + w];
      up += sm[(2 * KTID + 1) * 32 + w];
    }
    if (nv) {
      g = g * a.factor[id];
      up = up * a.factor[id];
    }
    a.y[(size_t)e * npair + p0 + KTID] = k_softcap(g, a.gate_cap) * k_sigmoid(g) * k_softcap(up, a.up_cap);
  }
}

// ===========================================================================
// Multi-row pass: R consecutive positions in one call. Every kernel below
// gives a row bit for bit the arithmetic of the single-row kernel named in its
// comment (same partition of the sums over the threads, same order).
// ===========================================================================

// Warp sums of 8 values at once (v[i] of every lane): the sum of value i, bit
// for bit k_warp_sum(v[i]), ends in lanes 4i .. 4i+3 (returned to each lane:
// the value of index lane / 4). 9 shuffles instead of 48: at offsets 16, 8, 4
// a lane keeps half of its values and hands the other half to its partner.
// (The pairs added at every step are those of the tree of k_warp_sum, and a
// + b == b + a, so every lane ends with the same bits as lane 0 there.)
KINL float k_warp_sum8t(KCTXP const float* v) {
  int lane = KTID & 31;
  int b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
  float w0 = (b4 ? v[4] : v[0]) + KSHFL(b4 ? v[0] : v[4], lane ^ 16);
  float w1 = (b4 ? v[5] : v[1]) + KSHFL(b4 ? v[1] : v[5], lane ^ 16);
  float w2 = (b4 ? v[6] : v[2]) + KSHFL(b4 ? v[2] : v[6], lane ^ 16);
  float w3 = (b4 ? v[7] : v[3]) + KSHFL(b4 ? v[3] : v[7], lane ^ 16);
  float u0 = (b3 ? w2 : w0) + KSHFL(b3 ? w0 : w2, lane ^ 8);
  float u1 = (b3 ? w3 : w1) + KSHFL(b3 ? w1 : w3, lane ^ 8);
  float t = (b2 ? u1 : u0) + KSHFL(b2 ? u0 : u1, lane ^ 4);
  t = t + KSHFL(t, lane ^ 2);
  t = t + KSHFL(t, lane ^ 1);
  return t;
}
// The same for 4 values: the sum of value i ends in lanes 8i .. 8i+7.
KINL float k_warp_sum4t(KCTXP const float* v) {
  int lane = KTID & 31;
  int b4 = lane & 16, b3 = lane & 8;
  float w0 = (b4 ? v[2] : v[0]) + KSHFL(b4 ? v[0] : v[2], lane ^ 16);
  float w1 = (b4 ? v[3] : v[1]) + KSHFL(b4 ? v[1] : v[3], lane ^ 16);
  float t = (b3 ? w1 : w0) + KSHFL(b3 ? w0 : w1, lane ^ 8);
  t = t + KSHFL(t, lane ^ 4);
  t = t + KSHFL(t, lane ^ 2);
  t = t + KSHFL(t, lane ^ 1);
  return t;
}

// The 8 bfloat16 weights of a 16-byte unit.
#define K_BF8(w0, w1, f)                          \
  f[0] = k_bf((unsigned)((w0) & 0xffffu));         \
  f[1] = k_bf((unsigned)(((w0) >> 16) & 0xffffu)); \
  f[2] = k_bf((unsigned)(((w0) >> 32) & 0xffffu)); \
  f[3] = k_bf((unsigned)((w0) >> 48));             \
  f[4] = k_bf((unsigned)((w1) & 0xffffu));         \
  f[5] = k_bf((unsigned)(((w1) >> 16) & 0xffffu)); \
  f[6] = k_bf((unsigned)(((w1) >> 32) & 0xffffu)); \
  f[7] = k_bf((unsigned)((w1) >> 48));

#define K_NRHS 8  // right-hand sides held in registers

// Input row i of the group of a multi-row product, 8 columns from column c.
KINL const float* k_xrow(const GemvJob& J, int r, int c) {
  return c < J.split ? J.x + (size_t)r * (size_t)J.split + c
                     : J.x2 + (size_t)r * (size_t)(J.cols - J.split) + (c - J.split);
}

// gemvm: gemv (one warp per matrix row) for up to 8 right-hand sides.
KERNEL(gemvm, GemvArgs) {
  int lane = KTID & 31;
  int grp = KBID / a.blocks;
  int b = KBID - grp * a.blocks;
  int row = b * (KNT >> 5) + (KTID >> 5);
  if (row >= a.total_rows) return;
  int j = 0;
  while (row >= a.job[j].rows) {
    row -= a.job[j].rows;
    ++j;
  }
  const GemvJob& J = a.job[j];
  int r0 = grp * K_NRHS;
  int nr = a.nrhs - r0;
  if (nr > K_NRHS) nr = K_NRHS;
  int cols = J.cols;
  const u16* wr = J.w + (size_t)row * (size_t)cols;
  float acc[K_NRHS];
  for (int i = 0; i < K_NRHS; ++i) acc[i] = 0.0f;
  int nunit = cols >> 3;
  for (int u = lane; u < nunit; u += 32) {
    const u16* p = wr + (u << 3);
    u64 c0 = KLOAD64(p), c1 = KLOAD64(p + 4);
    float f[8];
    K_BF8(c0, c1, f)
    for (int i = 0; i < K_NRHS; ++i) {
      if (i < nr) {
        const float* xx = k_xrow(J, r0 + i, u << 3);
        KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
        float s = acc[i];
        s = KFMA(f[0], xa.x, s);
        s = KFMA(f[1], xa.y, s);
        s = KFMA(f[2], xa.z, s);
        s = KFMA(f[3], xa.w, s);
        s = KFMA(f[4], xb.x, s);
        s = KFMA(f[5], xb.y, s);
        s = KFMA(f[6], xb.z, s);
        s = KFMA(f[7], xb.w, s);
        acc[i] = s;
      }
    }
  }
  float t = k_warp_sum8t(KCTXA acc);
  int i = lane >> 2;
  if ((lane & 3) == 0 && i < nr) J.y[(size_t)(r0 + i) * J.rows + row] = k_jout(J, t);
}

// gemvwm: gemvw (one block per matrix row) for up to 8 right-hand sides.
KERNEL(gemvwm, GemvArgs) {
  KSHARED(float, sm, 32 * K_NRHS);
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int grp = KBID / a.blocks;
  int row = KBID - grp * a.blocks;
  int j = 0;
  while (row >= a.job[j].rows) {
    row -= a.job[j].rows;
    ++j;
  }
  const GemvJob& J = a.job[j];
  int r0 = grp * K_NRHS;
  int nr = a.nrhs - r0;
  if (nr > K_NRHS) nr = K_NRHS;
  int cols = J.cols;
  const u16* wr = J.w + (size_t)row * (size_t)cols;
  float acc[K_NRHS];
  for (int i = 0; i < K_NRHS; ++i) acc[i] = 0.0f;
  int nunit = cols >> 3;
  for (int u = KTID; u < nunit; u += KNT) {
    const u16* p = wr + (u << 3);
    u64 c0 = KLOAD64(p), c1 = KLOAD64(p + 4);
    float f[8];
    K_BF8(c0, c1, f)
    for (int i = 0; i < K_NRHS; ++i) {
      if (i < nr) {
        const float* xx = k_xrow(J, r0 + i, u << 3);
        KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
        float s = acc[i];
        s = KFMA(f[0], xa.x, s);
        s = KFMA(f[1], xa.y, s);
        s = KFMA(f[2], xa.z, s);
        s = KFMA(f[3], xa.w, s);
        s = KFMA(f[4], xb.x, s);
        s = KFMA(f[5], xb.y, s);
        s = KFMA(f[6], xb.z, s);
        s = KFMA(f[7], xb.w, s);
        acc[i] = s;
      }
    }
  }
  float t = k_warp_sum8t(KCTXA acc);
  if ((lane & 3) == 0) sm[warp * K_NRHS + (lane >> 2)] = t;
  KSYNC();
  if (KTID < nr) {
    float s = 0.0f;
    for (int w = 0; w < nw; ++w) s += sm[w * K_NRHS + KTID];
    J.y[(size_t)(r0 + KTID) * J.rows + row] = k_jout(J, s);
  }
}

// gemvcm: gemvc (column owner, K_GEMVC_ROWS rows per block) for up to 8
// right-hand sides; the x values of all of them stay in registers, the
// weights of all rows of the block are loaded before any is used.
KERNEL(gemvcm, GemvArgs) {
  KSHARED(float, sm, K_GEMVC_ROWS * K_NRHS * 32);
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int grp = KBID / a.blocks;
  int b = KBID - grp * a.blocks, j = 0;
  for (;;) {
    int nb = (a.job[j].rows + K_GEMVC_ROWS - 1) / K_GEMVC_ROWS;
    if (b < nb) break;
    b -= nb;
    ++j;
  }
  const GemvJob& J = a.job[j];
  int r0 = grp * K_NRHS;
  int nr = a.nrhs - r0;
  if (nr > K_NRHS) nr = K_NRHS;
  int cols = J.cols;
  int row0 = b * K_GEMVC_ROWS;
  int nrows = J.rows - row0;
  if (nrows > K_GEMVC_ROWS) nrows = K_GEMVC_ROWS;
  int c0 = KTID * 8;
  int act = c0 < cols;
  const u16* wp = J.w + (size_t)row0 * (size_t)cols + (act ? c0 : 0);
  float x[K_NRHS][8];
  for (int i = 0; i < K_NRHS; ++i) {
    for (int e = 0; e < 8; ++e) x[i][e] = 0.0f;
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
  for (int r = 0; r < nrows; ++r) {
    u64 wa = 0, wb = 0;
    if (act) {
      const u16* p = wp + (size_t)r * (size_t)cols;
      wa = KLOAD64(p);
      wb = KLOAD64(p + 4);
    }
    float acc[K_NRHS];
    for (int i = 0; i < K_NRHS; ++i) acc[i] = 0.0f;
    if (act) {
      float f[8];
      K_BF8(wa, wb, f)
      for (int i = 0; i < K_NRHS; ++i) {
        if (i < nr) {
          float s = KFMA(f[0], x[i][0], 0.0f);
          s = KFMA(f[1], x[i][1], s);
          s = KFMA(f[2], x[i][2], s);
          s = KFMA(f[3], x[i][3], s);
          s = KFMA(f[4], x[i][4], s);
          s = KFMA(f[5], x[i][5], s);
          s = KFMA(f[6], x[i][6], s);
          s = KFMA(f[7], x[i][7], s);
          acc[i] = s;
        }
      }
    }
    float t = k_warp_sum8t(KCTXA acc);
    if ((lane & 3) == 0) sm[(r * K_NRHS + (lane >> 2)) * 32 + warp] = t;
  }
  KSYNC();
  for (int t = KTID; t < nrows * nr; t += KNT) {
    int r = t / nr, i = t - r * nr;
    float s = 0.0f;
    for (int w = 0; w < nw; ++w) s += sm[(r * K_NRHS + i) * 32 + w];
    J.y[(size_t)(r0 + i) * J.rows + row0 + r] = k_jout(J, s);
  }
}

// egroup: the distinct experts of a pass and their (row, slot) pairs. One
// block of at least E threads (a power of two >= E); R * k <= 2048.
KERNEL(egroup, GroupArgs) {
  KSHARED(int, sidx, 2048);
  KSHARED(int, cnt, 1024);
  KSHARED(int, start, 1024);
  KSHARED(int, used, 1024);
  int n = a.R * a.k;
  for (int i = KTID; i < n; i += KNT) sidx[i] = a.idx[i];
  KSYNC();
  int e = KTID;
  if (e < a.E) {
    int c = 0;
    for (int i = 0; i < n; ++i) c += sidx[i] == e ? 1 : 0;
    cnt[e] = c;
  }
  KSYNC();
  // inclusive scans of the counts and of "expert used" over the experts (Hillis-Steele, in place)
  int nb = 1;
  while (nb < a.E) nb <<= 1;
  for (int x = KTID; x < nb; x += KNT) {
    start[x] = x < a.E ? cnt[x] : 0;
    used[x] = x < a.E && cnt[x] > 0 ? 1 : 0;
  }
  KSYNC();
  for (int d = 1; d < nb; d <<= 1) {
    int v0 = 0, v1 = 0;
    if (e < nb && e >= d) {
      v0 = start[e - d];
      v1 = used[e - d];
    }
    KSYNC();
    if (e < nb && e >= d) {
      start[e] += v0;
      used[e] += v1;
    }
    KSYNC();
  }
  if (e < a.E) {
    int off = start[e] - cnt[e];  // exclusive
    if (cnt[e] > 0) {
      int u = used[e] - 1;
      a.gexp[u] = e;
      a.goff[u] = off;
    }
    if (e == a.E - 1) {
      a.goff[used[e]] = start[e];
      a.gcount[0] = used[e];
    }
  }
  KSYNC();
  if (e < a.E) start[e] -= cnt[e];
  KSYNC();
  if (e < a.E && cnt[e] > 0) {
    int p = start[e];
    for (int i = 0; i < n; ++i)
      if (sidx[i] == e) a.gpairs[p++] = i;
  }
}

// Activations staged in shared memory: float4 q of the 16-element unit u is
// kept at 4u + (q ^ ((u >> 1) & 3)), so that 8 lanes reading units u .. u+7
// hit 8 different bank groups.
KINL int k_swz(int c4) { return (c4 & ~3) | ((c4 & 3) ^ ((c4 >> 3) & 3)); }

// The 16 values of an expert unit (nvfp4: 8 code bytes; mxfp8: 16 E4M3 bytes).
KINL void k_unit16(const u8* wp, int nv, const float* e4m3, float* d) {
  if (nv) {
    u64 c = KLOAD64(wp);
    for (int e = 0; e < 16; ++e) d[e] = k_e2m1x2((unsigned)(c >> (4 * e)) & 15u);
  } else {
    u64 c0 = KLOAD64(wp), c1 = KLOAD64(wp + 8);
    for (int e = 0; e < 8; ++e) d[e] = e4m3[(unsigned)(c0 >> (8 * e)) & 255u];
    for (int e = 0; e < 8; ++e) d[8 + e] = e4m3[(unsigned)(c1 >> (8 * e)) & 255u];
  }
}
// s = sum of d[e] * x[e] over the 16 elements, in the order of K_NV4 / K_MX4.
KINL float k_dot16(const float* d, KF4 x0, KF4 x1, KF4 x2, KF4 x3) {
  float s = 0.0f;
  s = KFMA(d[0], x0.x, s);
  s = KFMA(d[1], x0.y, s);
  s = KFMA(d[2], x0.z, s);
  s = KFMA(d[3], x0.w, s);
  s = KFMA(d[4], x1.x, s);
  s = KFMA(d[5], x1.y, s);
  s = KFMA(d[6], x1.z, s);
  s = KFMA(d[7], x1.w, s);
  s = KFMA(d[8], x2.x, s);
  s = KFMA(d[9], x2.y, s);
  s = KFMA(d[10], x2.z, s);
  s = KFMA(d[11], x2.w, s);
  s = KFMA(d[12], x3.x, s);
  s = KFMA(d[13], x3.y, s);
  s = KFMA(d[14], x3.z, s);
  s = KFMA(d[15], x3.w, s);
  return s;
}

// Sums of 8 values over each 16-lane half at once: the sum of value i, bit
// for bit k_half_sum(v[i]), ends in lanes 2i, 2i+1 of the half (returned to
// each lane: the value of index (lane & 15) / 2).
KINL float k_half_sum8t(KCTXP const float* v) {
  int lane = KTID & 31;
  int b3 = lane & 8, b2 = lane & 4, b1 = lane & 2;
  float w0 = (b3 ? v[4] : v[0]) + KSHFL(b3 ? v[0] : v[4], lane ^ 8);
  float w1 = (b3 ? v[5] : v[1]) + KSHFL(b3 ? v[1] : v[5], lane ^ 8);
  float w2 = (b3 ? v[6] : v[2]) + KSHFL(b3 ? v[2] : v[6], lane ^ 8);
  float w3 = (b3 ? v[7] : v[3]) + KSHFL(b3 ? v[3] : v[7], lane ^ 8);
  float u0 = (b2 ? w2 : w0) + KSHFL(b2 ? w0 : w2, lane ^ 4);
  float u1 = (b2 ? w3 : w1) + KSHFL(b2 ? w1 : w3, lane ^ 4);
  float t = (b1 ? u1 : u0) + KSHFL(b1 ? u0 : u1, lane ^ 2);
  t = t + KSHFL(t, lane ^ 1);
  return t;
}

// exp13cm: exp13c for the experts of a pass. exp13c gives thread t the
// 16-element unit t, sums the products of a warp (k_warp_sum) and then the
// warps in order. Here one warp walks a row pair with lane l taking the unit
// l + 32 m in slice m, so that slice m holds what warp m of exp13c sums (same
// lanes, same tree); the slices are added in order. Item = (distinct expert,
// K_E13_PAIRS row pairs), warp v takes the row pairs v (and v + 8). For up to
// K_E13_X pairs of the expert at a time, slice m of their inputs is staged in
// shared memory (two buffers: slice m + 1 and its weights are loaded while
// slice m is computed), a weight unit is decoded once for all of them, and
// the sum of pair i is kept in lanes 4i .. 4i+3.
#define K_E13_X 8                   // pairs at a time
#define K_E13_RP 2                  // row pairs per warp (1 or 2; 2 measured faster)
#define K_E13_PAIRS (8 * K_E13_RP)  // row pairs per item (8 warps, 256 threads)
// The weights of one slice of a row: the code words and the block scale.
struct K_W16 {
  u64 c0, c1;
  float sc;
};
KINL K_W16 k_w16_load(const ExpMArgs& a, size_t row, int un, int ok, int nv, size_t wbytes, size_t sbytes) {
  K_W16 w;
  w.c0 = w.c1 = 0;
  w.sc = 0.0f;
  if (ok) {
    const u8* wp = a.w + row * wbytes + (nv ? ((size_t)un << 3) : ((size_t)un << 4));
    w.c0 = KLOAD64(wp);
    if (!nv) w.c1 = KLOAD64(wp + 8);
    unsigned sb = a.sc[row * sbytes + (nv ? un : (un >> 1))];
    w.sc = nv ? a.e4m3[sb] : a.exp2t[sb];
  }
  return w;
}
// Decoding through a table in shared memory (k_dtab_fill): nvfp4 byte b ->
// (value of its low nibble, value of its high nibble) at 2b, 2b + 1; mxfp8
// byte b -> e4m3[b]. The same values as k_e2m1x2 / the e4m3 table.
KINL void k_dtab_fill(KCTXP float* dtab, int nv, const float* e4m3) {
  for (int b = KTID; b < 256; b += KNT) {
    if (nv) {
      dtab[2 * b] = k_e2m1x2((unsigned)b & 15u);
      dtab[2 * b + 1] = k_e2m1x2((unsigned)b >> 4);
    } else {
      dtab[b] = e4m3[b];
    }
  }
}
KINL void k_w16_decode(const K_W16& w, int nv, const float* dtab, float* d) {
  if (nv) {
    for (int j = 0; j < 8; ++j) {
      unsigned b = (unsigned)(w.c0 >> (8 * j)) & 255u;
      d[2 * j] = dtab[2 * b];
      d[2 * j + 1] = dtab[2 * b + 1];
    }
  } else {
    for (int e = 0; e < 8; ++e) d[e] = dtab[(unsigned)(w.c0 >> (8 * e)) & 255u];
    for (int e = 0; e < 8; ++e) d[8 + e] = dtab[(unsigned)(w.c1 >> (8 * e)) & 255u];
  }
}
// Stages slice m of the inputs of ng pairs (rows of x, row stride cols, pair
// p -> input row p / div) into buf: pair i at i * per float4, swizzled.
KINL void k_stage(KCTXP KF4* buf, const ExpMArgs& a, int g, int ng, int div, int cols, int ufirst, int nu, int per) {
  for (int q = KTID; q < ng * nu * 4; q += KNT) {
    int pi = q / (nu * 4), cc = q - pi * (nu * 4);
    buf[pi * per + k_swz(cc)] = KLOADF4(a.x + (size_t)(a.gpairs[g + pi] / div) * cols + 16 * ufirst + 4 * cc);
  }
}
KERNEL(exp13cm, ExpMArgs) {
  KSHARED(KF4, xs4, 2 * K_E13_X * 32 * 4);
  KSHARED(float, dtab, 512);
  int lane = KTID & 31, warp = KTID >> 5;
  int npair = a.rows >> 1;
  int cols = a.cols, nunit = cols >> 4;
  int nslice = (nunit + 31) >> 5;  // warps of exp13c
  int items = a.gcount[0] * a.nchunk;
  int nv = a.fmt == INK2_FMT_NVFP4;
  size_t wbytes = nv ? (size_t)(cols >> 1) : (size_t)cols;
  size_t sbytes = nv ? (size_t)(cols >> 4) : (size_t)(cols >> 5);
  int me = lane >> 2;  // the pair whose sums this lane keeps
  int sw = (lane >> 1) & 3;
  const int per = 32 * 4;  // float4 of a pair in a slice
  k_dtab_fill(KCTXA dtab, nv, a.e4m3);  // (read after the first barrier below)
  for (int it = KBID; it < items; it += a.grid) {
    int ue = it / a.nchunk;
    int p0 = (it - ue * a.nchunk) * K_E13_PAIRS;
    int id = a.gexp[ue];
    int pb = a.goff[ue], pe = a.goff[ue + 1];
    float fac = nv ? a.factor[id] : 1.0f;
    int pr0 = p0 + warp, pr1 = p0 + warp + 8;  // the row pairs of this warp
    int live0 = pr0 < npair, live1 = K_E13_RP > 1 && pr1 < npair;
    size_t rg0 = (size_t)id * a.rows + (size_t)(2 * (live0 ? pr0 : 0));
    size_t rg1 = (size_t)id * a.rows + (size_t)(2 * (live1 ? pr1 : 0));
    for (int g = pb; g < pe; g += K_E13_X) {
      int ng = pe - g;
      if (ng > K_E13_X) ng = K_E13_X;
      float gl0 = 0.0f, ul0 = 0.0f, gl1 = 0.0f, ul1 = 0.0f;
      KSYNC();  // buffer 0 is no longer read
      {
        int nu = nunit < 32 ? nunit : 32;
        k_stage(KCTXA xs4, a, g, ng, a.k, cols, 0, nu, per);
      }
      int un = lane;
      K_W16 wg0 = k_w16_load(a, rg0, un, live0 && un < nunit, nv, wbytes, sbytes);
      K_W16 wu0 = k_w16_load(a, rg0 + 1, un, live0 && un < nunit, nv, wbytes, sbytes);
      K_W16 wg1 = k_w16_load(a, rg1, un, live1 && un < nunit, nv, wbytes, sbytes);
      K_W16 wu1 = k_w16_load(a, rg1 + 1, un, live1 && un < nunit, nv, wbytes, sbytes);
      KSYNC();
      for (int m = 0; m < nslice; ++m) {
        KF4* cur = xs4 + (m & 1) * (K_E13_X * per);
        int un2 = un + 32;  // the unit of the next slice
        K_W16 ng0, nu0, ng1, nu1;
        if (m + 1 < nslice) {  // the next slice: inputs into the other buffer, weights into registers
          int nu = nunit - 32 * (m + 1);
          if (nu > 32) nu = 32;
          k_stage(KCTXA xs4 + ((m + 1) & 1) * (K_E13_X * per), a, g, ng, a.k, cols, 32 * (m + 1), nu, per);
        }
        ng0 = k_w16_load(a, rg0, un2, live0 && m + 1 < nslice && un2 < nunit, nv, wbytes, sbytes);
        nu0 = k_w16_load(a, rg0 + 1, un2, live0 && m + 1 < nslice && un2 < nunit, nv, wbytes, sbytes);
        ng1 = k_w16_load(a, rg1, un2, live1 && m + 1 < nslice && un2 < nunit, nv, wbytes, sbytes);
        nu1 = k_w16_load(a, rg1 + 1, un2, live1 && m + 1 < nslice && un2 < nunit, nv, wbytes, sbytes);
        const KF4* xx0 = cur + 4 * lane;
        KUNROLL for (int rp = 0; rp < 2; ++rp) {
          int live = rp ? live1 : live0;
          if (!live) continue;  // uniform over the warp
          float pg[K_E13_X], pu[K_E13_X];
          KUNROLL for (int pi = 0; pi < K_E13_X; ++pi) pg[pi] = pu[pi] = 0.0f;
          if (un < nunit) {
            float d[16];
            k_w16_decode(rp ? wg1 : wg0, nv, dtab, d);
            float sgv = rp ? wg1.sc : wg0.sc;
            KUNROLL for (int pi = 0; pi < K_E13_X; ++pi) {
              if (pi < ng) {
                const KF4* xx = xx0 + pi * per;
                pg[pi] = k_dot16(d, xx[0 ^ sw], xx[1 ^ sw], xx[2 ^ sw], xx[3 ^ sw]) * sgv;
              }
            }
            k_w16_decode(rp ? wu1 : wu0, nv, dtab, d);
            float suv = rp ? wu1.sc : wu0.sc;
            KUNROLL for (int pi = 0; pi < K_E13_X; ++pi) {
              if (pi < ng) {
                const KF4* xx = xx0 + pi * per;
                pu[pi] = k_dot16(d, xx[0 ^ sw], xx[1 ^ sw], xx[2 ^ sw], xx[3 ^ sw]) * suv;
              }
            }
          }
          float tg = k_warp_sum8t(KCTXA pg);
          float tu = k_warp_sum8t(KCTXA pu);
          if (rp) {
            gl1 += tg;
            ul1 += tu;
          } else {
            gl0 += tg;
            ul0 += tu;
          }
        }
        KSYNC();  // this slice is read, the next one is staged
        wg0 = ng0;
        wu0 = nu0;
        wg1 = ng1;
        wu1 = nu1;
        un = un2;
      }
      KUNROLL for (int rp = 0; rp < 2; ++rp) {
        int pr = rp ? pr1 : pr0;
        if ((rp ? live1 : live0) && (lane & 3) == 0 && me < ng) {
          float gv = rp ? gl1 : gl0, up = rp ? ul1 : ul0;
          if (nv) {
            gv = gv * fac;
            up = up * fac;
          }
          a.y[(size_t)a.gpairs[g + me] * npair + pr] = k_softcap(gv, a.gate_cap) * k_sigmoid(gv) * k_softcap(up, a.up_cap);
        }
      }
    }
  }
}

// exp2m: the w2 products of exp2 (16 lanes per (pair, output row), lane l the
// units l, l + 16, ... in one fused multiply-add chain, then k_half_sum) for
// the experts of a pass. Item = (distinct expert, K_EXP2M_H output rows), one
// 16-lane group per row. For up to K_EXP2M_P pairs of the expert at a time,
// slice m (units 16 m .. 16 m + 15) of their activations is staged in shared
// memory (two buffers, the next slice and its weights are loaded while slice
// m is computed) and every lane extends the chains of all of them by its unit.
#define K_EXP2M_P 8   // pairs at a time
#define K_EXP2M_H 16  // output rows per item (16 lanes each, 256 threads)
KERNEL(exp2m, ExpMArgs) {
  KSHARED(KF4, xs4, 2 * K_EXP2M_P * 16 * 4);
  KSHARED(float, dtab, 512);
  int l16 = KTID & 15, gi = KTID >> 4;
  int cols = a.cols, nunit = cols >> 4;
  int nslice = (nunit + 15) >> 4;
  int items = a.gcount[0] * a.nchunk;
  int nv = a.fmt == INK2_FMT_NVFP4;
  size_t wbytes = nv ? (size_t)(cols >> 1) : (size_t)cols;
  size_t sbytes = nv ? (size_t)(cols >> 4) : (size_t)(cols >> 5);
  int me = l16 >> 1;  // the pair whose sum this lane ends with
  int sw = (l16 >> 1) & 3;
  const int per = 16 * 4;
  k_dtab_fill(KCTXA dtab, nv, a.e4m3);  // (read after the first barrier below)
  for (int it = KBID; it < items; it += a.grid) {
    int ue = it / a.nchunk;
    int h = (it - ue * a.nchunk) * K_EXP2M_H + gi;
    int liveh = h < a.rows;
    int id = a.gexp[ue];
    int pb = a.goff[ue], pe = a.goff[ue + 1];
    size_t row = (size_t)id * a.rows + (size_t)(liveh ? h : 0);
    for (int g = pb; g < pe; g += K_EXP2M_P) {
      int ng = pe - g;
      if (ng > K_EXP2M_P) ng = K_EXP2M_P;
      float acc[K_EXP2M_P];
      KUNROLL for (int pi = 0; pi < K_EXP2M_P; ++pi) acc[pi] = 0.0f;
      KSYNC();  // buffer 0 is no longer read
      {
        int nu = nunit < 16 ? nunit : 16;
        k_stage(KCTXA xs4, a, g, ng, 1, cols, 0, nu, per);
      }
      int un = l16;
      K_W16 w = k_w16_load(a, row, un, liveh && un < nunit, nv, wbytes, sbytes);
      KSYNC();
      for (int m = 0; m < nslice; ++m) {
        KF4* cur = xs4 + (m & 1) * (K_EXP2M_P * per);
        int un2 = un + 16;
        if (m + 1 < nslice) {
          int nu = nunit - 16 * (m + 1);
          if (nu > 16) nu = 16;
          k_stage(KCTXA xs4 + ((m + 1) & 1) * (K_EXP2M_P * per), a, g, ng, 1, cols, 16 * (m + 1), nu, per);
        }
        K_W16 wn = k_w16_load(a, row, un2, liveh && m + 1 < nslice && un2 < nunit, nv, wbytes, sbytes);
        if (liveh && un < nunit) {
          float d[16];
          k_w16_decode(w, nv, dtab, d);
          const KF4* xx0 = cur + 4 * l16;
          KUNROLL for (int pi = 0; pi < K_EXP2M_P; ++pi) {
            if (pi < ng) {
              const KF4* xx = xx0 + pi * per;
              float s = k_dot16(d, xx[0 ^ sw], xx[1 ^ sw], xx[2 ^ sw], xx[3 ^ sw]);
              acc[pi] = KFMA(s, w.sc, acc[pi]);
            }
          }
        }
        KSYNC();  // this slice is read, the next one is staged
        w = wn;
        un = un2;
      }
      float s = k_half_sum8t(KCTXA acc);
      if (nv) s = s * a.factor[id];
      if (liveh && (l16 & 1) == 0 && me < ng) a.y[(size_t)a.gpairs[g + me] * a.rows + h] = s;
    }
  }
}

// ecomb: the end of exp2 for every (row, channel): sum of the k products in
// slot order, then the short convolution of that sum + residual. The sums of
// the K - 1 rows before are recomputed here when they belong to the pass, so
// that the rows do not wait for each other.
#define K_CONV_MAX 8
KERNEL(ecomb, ECombArgs) {
  int t = KBID * KNT + KTID;
  int H = a.H;
  if (t >= a.R * H) return;
  int r = t / H, h = t - r * H;
  int K1 = a.K - 1;
  float ov[K_CONV_MAX];
  for (int m = 0; m <= K1 && m < K_CONV_MAX; ++m) {
    ov[m] = 0.0f;
    int rr = r - m;
    if (rr < 0 || (m > 0 && !a.cw)) continue;
    float o = 0.0f;
    for (int j = 0; j < a.k; ++j) o = o + a.dots[((size_t)rr * a.k + j) * H + h] * a.wt[(size_t)rr * a.k + j];
    ov[m] = o;
  }
  float x = ov[0];
  a.y[(size_t)r * H + h] = x;
  if (a.cw) {
    int pos = a.st[r].pos;
    float o = 0.0f;
    for (int tt = 0; tt < K1; ++tt) {
      int lag = K1 - tt;
      float hv = r >= lag ? ov[lag] : a.chist[(size_t)k_slot(pos - lag, a.hr) * H + h];
      o = o + hv * k_bf(a.cw[(size_t)h * a.K + tt]);
    }
    o = o + x * k_bf(a.cw[(size_t)h * a.K + K1]);
    o = (o + x) / a.cdiv;
    a.chist[(size_t)(pos % a.hr) * H + h] = x;
    a.cout[(size_t)r * H + h] = a.cres[(size_t)r * H + h] + o;
  }
}

// ---------------------------------------------------------------------------
// Experts of a pass of few rows (R <= 8: verification passes). There most distinct experts serve one or
// two (row, slot) pairs, and exp13cm / exp2m (made for up to 128 rows: inputs staged slice by slice in
// shared memory, a barrier per slice) spend their time on the staging. exp13s / exp2s keep the block and
// the arithmetic of the single-row kernels exp13c / exp2 per (distinct expert, chunk of matrix rows):
// every thread first loads the code words and scales of all its rows (all loads in flight at once), then
// for each pair of the expert multiplies the input row of that pair with exactly the sums of exp13c /
// exp2 (same unit per thread, same FMA chains, same warp trees, warps added in order). So a pass gives
// bit for bit the single steps. NV = 1: nvfp4 experts, 0: mxfp8 (the decode tables live in shared memory,
// with the values of the device tables).
// ---------------------------------------------------------------------------
// exp13s: exp13c for the distinct experts of a pass of few rows. The block of exp13c (H / 16 threads
// rounded up to whole warps; thread t the 16-element unit t; K_EXPC_PAIRS row pairs per item) and its loop
// over the rows, once for every (row, slot) pair that chose the expert: the first pair reads the weights
// from device memory, the further pairs of the same expert find them in the caches. Item = (distinct
// expert it / nchunk, row pairs (it % nchunk) * K_EXPC_PAIRS ..). Per pair the sums of exp13c exactly.
template <int NV>
KINL void k_exp13s(KCTXP const ExpMArgs& a, float* sm) {
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int npair = a.rows >> 1;
  int u = KTID;
  int act = u < (a.cols >> 4);
  const float* e4m3 = a.e4m3;
  size_t wbytes = NV ? (size_t)(a.cols >> 1) : (size_t)a.cols;
  size_t sbytes = NV ? (size_t)(a.cols >> 4) : (size_t)(a.cols >> 5);
  int items = a.gcount[0] * a.nchunk;
  for (int it = KBID; it < items; it += a.grid) {
    int ue = it / a.nchunk;
    int p0 = (it - ue * a.nchunk) * K_EXPC_PAIRS;
    int np = npair - p0;
    if (np > K_EXPC_PAIRS) np = K_EXPC_PAIRS;
    int id = a.gexp[ue];
    int pb = a.goff[ue], pe = a.goff[ue + 1];
    size_t row0 = (size_t)id * a.rows + (size_t)(2 * p0);
    const u8* wp = a.w + row0 * wbytes + (act ? (NV ? ((size_t)u << 3) : ((size_t)u << 4)) : 0);
    const u8* sp = a.sc + row0 * sbytes + (act ? (NV ? (size_t)u : (size_t)(u >> 1)) : 0);
    for (int q = pb; q < pe; ++q) {
      int pr = a.gpairs[q];
      KF4 x0 = KF4ZERO, x1 = KF4ZERO, x2 = KF4ZERO, x3 = KF4ZERO;
      if (act) {
        const float* xx = a.x + (size_t)(pr / a.k) * a.cols + (u << 4);
        x0 = KLOADF4(xx);
        x1 = KLOADF4(xx + 4);
        x2 = KLOADF4(xx + 8);
        x3 = KLOADF4(xx + 12);
      }
      for (int r = 0; r < 2 * np; ++r) {  // the loop of exp13c
        float part = 0.0f;
        if (act) {
          float s = 0.0f;
          if (NV) {
            u64 c = KLOAD64(wp + (size_t)r * wbytes);
            K_NV4(c, 0, x0)
            K_NV4(c, 16, x1)
            K_NV4(c, 32, x2)
            K_NV4(c, 48, x3)
            part = s * e4m3[sp[(size_t)r * sbytes]];
          } else {
            u64 c0 = KLOAD64(wp + (size_t)r * wbytes), c1 = KLOAD64(wp + (size_t)r * wbytes + 8);
            K_MX4(c0, 0, x0)
            K_MX4(c0, 32, x1)
            K_MX4(c1, 0, x2)
            K_MX4(c1, 32, x3)
            part = s * a.exp2t[sp[(size_t)r * sbytes]];
          }
        }
        float rs = k_warp_sum(KCTXA part);
        if (lane == 0) sm[r * 32 + warp] = rs;
      }
      KSYNC();
      if (KTID < np) {
        float g = 0.0f, up = 0.0f;
        for (int w = 0; w < nw; ++w) {
          g += sm[(2 * KTID) * 32 + w];
          up += sm[(2 * KTID + 1) * 32 + w];
        }
        if (NV) {
          g = g * a.factor[id];
          up = up * a.factor[id];
        }
        a.y[(size_t)pr * npair + p0 + KTID] = k_softcap(g, a.gate_cap) * k_sigmoid(g) * k_softcap(up, a.up_cap);
      }
      KSYNC();  // sm is written again by the next pair
    }
  }
}
KERNEL(exp13s, ExpMArgs) {
  KSHARED(float, sm, 2 * K_EXPC_PAIRS * 32);
  k_exp13s<1>(KCTXA a, sm);
}
KERNEL(exp13sx, ExpMArgs) {
  KSHARED(float, sm, 2 * K_EXPC_PAIRS * 32);
  k_exp13s<0>(KCTXA a, sm);
}

// exp2s: the w2 products of exp2 for the experts of a pass of few rows. Item = (distinct expert,
// K_EXP2M_H output rows), 256 threads: the 16-lane group gi takes output row h = K_EXP2M_H * chunk + gi,
// lane l the units l, l + 16, ... of that row (as the 16 lanes of a slot in exp2). Their code words and
// scales go to shared memory first (all loads in flight; each thread reads back only its own), then for
// every pair of the expert the FMA chain of exp2 over the units in order, k_half_sum, the expert factor
// -> y [pair, h] (ecomb adds the k products in slot order as exp2 does).
#define K_EXP2S_U 5  // units per lane: cols <= 16 * 16 * K_EXP2S_U (the engine allows I <= 1280)
template <int NV>
KINL void k_exp2s(KCTXP const ExpMArgs& a, float* ctab, float* stab, u64* wsh, u64* wsh1, float* ssh) {
  int l16 = KTID & 15, gi = KTID >> 4;
  int nunit = a.cols >> 4;
  for (int b = KTID; b < 256; b += KNT) {
    stab[b] = NV ? a.e4m3[b] : a.exp2t[b];
    ctab[b] = a.e4m3[b];
  }
  KSYNC();
  const float* e4m3 = ctab;
  size_t wbytes = NV ? (size_t)(a.cols >> 1) : (size_t)a.cols;
  size_t sbytes = NV ? (size_t)(a.cols >> 4) : (size_t)(a.cols >> 5);
  int items = a.gcount[0] * a.nchunk;
  KNOUNROLL for (int it = KBID; it < items; it += a.grid) {
    int ue = it / a.nchunk;
    int h = (it - ue * a.nchunk) * K_EXP2M_H + gi;
    int liveh = h < a.rows;
    int id = a.gexp[ue];
    int pb = a.goff[ue], pe = a.goff[ue + 1];
    size_t row = (size_t)id * a.rows + (size_t)(liveh ? h : 0);
    const u8* wrow = a.w + row * wbytes;
    const u8* srow = a.sc + row * sbytes;
    KUNROLL for (int i = 0; i < K_EXP2S_U; ++i) {  // all loads of the item in flight at once
      int un = l16 + 16 * i;
      if (liveh && un < nunit) {
        if (NV) {
          wsh[i * 256 + KTID] = KLOAD64(wrow + ((size_t)un << 3));
          ssh[i * 256 + KTID] = stab[srow[un]];
        } else {
          wsh[i * 256 + KTID] = KLOAD64(wrow + ((size_t)un << 4));
          wsh1[i * 256 + KTID] = KLOAD64(wrow + ((size_t)un << 4) + 8);
          ssh[i * 256 + KTID] = stab[srow[un >> 1]];
        }
      }
    }
    KNOUNROLL for (int q = pb; q < pe; ++q) {
      int pr = a.gpairs[q];
      const float* xr = a.x + (size_t)pr * a.cols;
      float acc = 0.0f;
      KNOUNROLL for (int i = 0; i < K_EXP2S_U; ++i) {
        int un = l16 + 16 * i;
        if (liveh && un < nunit) {
          const float* xx = xr + (un << 4);
          KF4 x0 = KLOADF4(xx), x1 = KLOADF4(xx + 4), x2 = KLOADF4(xx + 8), x3 = KLOADF4(xx + 12);
          float s = 0.0f;
          u64 c = wsh[i * 256 + KTID];
          if (NV) {
            K_NV4(c, 0, x0)
            K_NV4(c, 16, x1)
            K_NV4(c, 32, x2)
            K_NV4(c, 48, x3)
          } else {
            u64 cb = wsh1[i * 256 + KTID];
            K_MX4(c, 0, x0)
            K_MX4(c, 32, x1)
            K_MX4(cb, 0, x2)
            K_MX4(cb, 32, x3)
          }
          acc = KFMA(s, ssh[i * 256 + KTID], acc);
        }
      }
      float s = k_half_sum(KCTXA acc);
      if (NV) s = s * a.factor[id];
      if (liveh && l16 == 0) a.y[(size_t)pr * a.rows + h] = s;
    }
  }
}
KERNEL(exp2s, ExpMArgs) {
  KSHARED(float, ctab, 256);
  KSHARED(float, stab, 256);
  KSHARED(u64, wsh, K_EXP2S_U * 256);
  KSHARED(float, ssh, K_EXP2S_U * 256);
  k_exp2s<1>(KCTXA a, ctab, stab, wsh, wsh, ssh);
}
KERNEL(exp2sx, ExpMArgs) {
  KSHARED(float, ctab, 256);
  KSHARED(float, stab, 256);
  KSHARED(u64, wsh, K_EXP2S_U * 256);
  KSHARED(u64, wsh1, K_EXP2S_U * 256);
  KSHARED(float, ssh, K_EXP2S_U * 256);
  k_exp2s<0>(KCTXA a, ctab, stab, wsh, wsh1, ssh);
}

// Index ordering matches numpy.argmax, including ties and the first NaN.
KINL bool k_pick_before(float v, int id, float best, int winner) {
  if (v != v) return best == best || id < winner;
  return v > best || (v == best && id < winner);
}
KERNEL(pick_tile, PickArgs) {
  KSHARED(float, warp_val, 32);
  KSHARED(int, warp_idx, 32);
  int begin = a.indices ? 0 : KBID * ink2_pick_tile;
  int end = !a.indices && a.n - begin > ink2_pick_tile ? begin + ink2_pick_tile : a.n;
  float best = KBITS2F(0xff800000u);
  int winner = 0x00ffffff;  // the engine's supported vocabulary indices are exact float32 integers
  for (int i = begin + KTID; i < end; i += KNT) {
    int id = a.indices ? a.indices[i] : i;
    float v = a.values[i];
    if (k_pick_before(v,id,best,winner)) { best = v; winner = id; }
  }
  for (int delta = 16; delta; delta >>= 1) {
    float v = KSHFL_DOWN(best,delta);
    int id = (int)KSHFL_DOWN((float)winner,delta);
    if (k_pick_before(v,id,best,winner)) { best = v; winner = id; }
  }
  if (!(KTID & 31)) {
    warp_val[KTID >> 5] = best;
    warp_idx[KTID >> 5] = winner;
  }
  KSYNC();
  if (!KTID) {
    best = warp_val[0];
    winner = warp_idx[0];
    for (int w = 1; w < KNT / 32; ++w) {
      if (k_pick_before(warp_val[w],warp_idx[w],best,winner)) {
        best = warp_val[w];
        winner = warp_idx[w];
      }
    }
    a.maxima[KBID] = best;
    a.winners[KBID] = winner;
  }
}

// argmax: largest logit of a row and its index (the first on ties).
KERNEL(argmax, ArgmaxArgs) {
  KSHARED(float, sv, 32);
  KSHARED(float, si, 32);
  const float* l = a.logits + (size_t)KBID * a.n;
  float best = -3.0e38f, bi = (float)a.n;  // indices < 2^24 are exact as floats
  for (int i = KTID; i < a.n; i += KNT) {
    float v = l[i];
    if (v > best) {
      best = v;
      bi = (float)i;
    }
  }
  for (int d = 16; d >= 1; d >>= 1) {
    float ov = KSHFL_DOWN(best, d), oi = KSHFL_DOWN(bi, d);
    if (ov > best || (ov == best && oi < bi)) {
      best = ov;
      bi = oi;
    }
  }
  if ((KTID & 31) == 0) {
    sv[KTID >> 5] = best;
    si[KTID >> 5] = bi;
  }
  KSYNC();
  if (KTID == 0) {
    int nw = KNT >> 5;
    best = sv[0];
    bi = si[0];
    for (int w = 1; w < nw; ++w) {
      if (sv[w] > best || (sv[w] == best && si[w] < bi)) {
        best = sv[w];
        bi = si[w];
      }
    }
    a.idx[KBID] = (int)bi;
    a.val[KBID] = best;
  }
}

// ===========================================================================
// Packed dense rows: pgemv / pgemvw / pgemvc are gemv / gemvw / gemvc on rows
// stored as nvfp4 or mxfp8 (GemvJob.fmt; bfloat16 jobs may be mixed in). A
// thread multiplies 8 consecutive columns c0 .. c0+7 (c0 % 8 == 0) of a row:
//   bf16 : 16 bytes;
//   nvfp4: 4 code bytes; the E4M3 scale of their 16-column block multiplies
//          the sum of the 8 products;
//   mxfp8: 8 E4M3 bytes; the exponent of their 32-column block multiplies
//          the sum of the 8 products.
// The row sum is multiplied by fac (nvfp4: amax / 2688 / 2) before the
// epilogue (k_jout). Several right-hand sides (nrhs > 1): blocks * nrhs
// blocks, block b of right-hand side r does exactly the work of block b of a
// single row on input row r (x row stride split, x2 row stride cols - split,
// y row stride rows), so that a pass gives bit for bit the single steps; the
// weights are read once per right-hand side (from L2 after the first).
// ===========================================================================
#define K_FP4_8(c, xa, xb)                \
  s = KFMA(k_fp4x2((c)), (xa).x, s);      \
  s = KFMA(k_fp4x2((c) >> 4), (xa).y, s); \
  s = KFMA(k_fp4x2((c) >> 8), (xa).z, s); \
  s = KFMA(k_fp4x2((c) >> 12), (xa).w, s); \
  s = KFMA(k_fp4x2((c) >> 16), (xb).x, s); \
  s = KFMA(k_fp4x2((c) >> 20), (xb).y, s); \
  s = KFMA(k_fp4x2((c) >> 24), (xb).z, s); \
  s = KFMA(k_fp4x2((c) >> 28), (xb).w, s);
#define K_E4M3_8(c, xa, xb)                            \
  s = KFMA(k_e4m3v((unsigned)(c)), (xa).x, s);         \
  s = KFMA(k_e4m3v((unsigned)((c) >> 8)), (xa).y, s);  \
  s = KFMA(k_e4m3v((unsigned)((c) >> 16)), (xa).z, s); \
  s = KFMA(k_e4m3v((unsigned)((c) >> 24)), (xa).w, s); \
  s = KFMA(k_e4m3v((unsigned)((c) >> 32)), (xb).x, s); \
  s = KFMA(k_e4m3v((unsigned)((c) >> 40)), (xb).y, s); \
  s = KFMA(k_e4m3v((unsigned)((c) >> 48)), (xb).z, s); \
  s = KFMA(k_e4m3v((unsigned)((c) >> 56)), (xb).w, s);

// 8 columns from c0 of row r of one matrix (wb = code bytes, sb = scale bytes) times xa, xb
KINL float k_pk8(const u8* wb, const u8* sb, int fmt, size_t r, int cols, int c0, KF4 xa, KF4 xb) {
  float s = 0.0f;
  if (fmt == INK2_DFMT_NVFP4) {
    unsigned c = KLOAD32(wb + r * (size_t)(cols >> 1) + (size_t)(c0 >> 1));
    K_FP4_8(c, xa, xb)
    return s * k_e4m3v(sb[r * (size_t)(cols >> 4) + (size_t)(c0 >> 4)]);
  }
  if (fmt == INK2_DFMT_MXFP8) {
    u64 c = KLOAD64(wb + r * (size_t)cols + (size_t)c0);
    K_E4M3_8(c, xa, xb)
    return s * k_exp2v(sb[r * (size_t)(cols >> 5) + (size_t)(c0 >> 5)]);
  }
  if (fmt == INK2_DFMT_Q3) {
    unsigned c = k_q3word(wb, r, cols, c0);
    s = KFMA(k_q3v(c), xa.x, s);
    s = KFMA(k_q3v(c >> 3), xa.y, s);
    s = KFMA(k_q3v(c >> 6), xa.z, s);
    s = KFMA(k_q3v(c >> 9), xa.w, s);
    s = KFMA(k_q3v(c >> 12), xb.x, s);
    s = KFMA(k_q3v(c >> 15), xb.y, s);
    s = KFMA(k_q3v(c >> 18), xb.z, s);
    s = KFMA(k_q3v(c >> 21), xb.w, s);
    return s * k_e4m3v(sb[r * (size_t)(cols >> 4) + (size_t)(c0 >> 4)]);
  }
  const u16* p = (const u16*)wb + r * (size_t)cols + (size_t)c0;
  u64 w0 = KLOAD64(p), w1 = KLOAD64(p + 4);
  s = KFMA(k_bf((unsigned)(w0 & 0xffffu)), xa.x, s);
  s = KFMA(k_bf((unsigned)((w0 >> 16) & 0xffffu)), xa.y, s);
  s = KFMA(k_bf((unsigned)((w0 >> 32) & 0xffffu)), xa.z, s);
  s = KFMA(k_bf((unsigned)(w0 >> 48)), xa.w, s);
  s = KFMA(k_bf((unsigned)(w1 & 0xffffu)), xb.x, s);
  s = KFMA(k_bf((unsigned)((w1 >> 16) & 0xffffu)), xb.y, s);
  s = KFMA(k_bf((unsigned)((w1 >> 32) & 0xffffu)), xb.z, s);
  s = KFMA(k_bf((unsigned)(w1 >> 48)), xb.w, s);
  return s;
}
// Part of row r of job J times input row rhs: the 8-column units u = first, first + stride, ...
KINL float k_pk_dot(const GemvJob& J, int rhs, size_t r, int first, int stride) {
  float acc = 0.0f;
  int nunit = J.cols >> 3;
  for (int u = first; u < nunit; u += stride) {
    const float* xx = k_xrow(J, rhs, u << 3);
    KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
    acc += k_pk8((const u8*)J.w, J.sc, J.fmt, r, J.cols, u << 3, xa, xb);
  }
  return acc;
}

// pgemv: one warp per row (as gemv)
KERNEL(pgemv, GemvArgs) {
  int lane = KTID & 31;
  int b = KBID, rhs = 0;
  if (a.nrhs > 1) {
    rhs = b / a.blocks;
    b -= rhs * a.blocks;
  }
  int row = b * (KNT >> 5) + (KTID >> 5);
  if (row >= a.total_rows) return;
  int j = 0;
  while (row >= a.job[j].rows) {
    row -= a.job[j].rows;
    ++j;
  }
  const GemvJob& J = a.job[j];
  float s = k_warp_sum(KCTXA k_pk_dot(J, rhs, (size_t)row, lane, 32));
  if (lane == 0) J.y[(size_t)rhs * J.rows + row] = k_jout(J, s);
}

// pgemvw: one block per row (as gemvw)
KERNEL(pgemvw, GemvArgs) {
  KSHARED(float, sm, 33);
  int row = KBID, rhs = 0;
  if (a.nrhs > 1) {
    rhs = row / a.blocks;
    row -= rhs * a.blocks;
  }
  int j = 0;
  while (row >= a.job[j].rows) {
    row -= a.job[j].rows;
    ++j;
  }
  const GemvJob& J = a.job[j];
  float s = k_block_sum(KCTXA k_pk_dot(J, rhs, (size_t)row, KTID, KNT), sm);
  if (KTID == 0) J.y[(size_t)rhs * J.rows + row] = k_jout(J, s);
}

// pgemvc: column owner (as gemvc). One block per chunk of NR rows of one job
// (pgemvc: 16, pgemvc8: 8, pgemvc4: 4, pgemvc2: 2); all jobs have the same cols; cols / 8
// threads, thread t owns the columns 8t .. 8t+7 (x in registers). The loads of
// all rows of the chunk are issued before the sums; the row sums of a warp go
// through k_warp_sum8t.
#define K_PGEMVC_ROWS 16
template <int NR>
KINL void k_pgemvc(KCTXP const GemvArgs& a, float* sm) {
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
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
  int cols = J.cols;
  int row0 = b * NR;
  int nrows = J.rows - row0;
  if (nrows > NR) nrows = NR;
  int c0 = KTID * 8;
  int act = c0 < cols;
  KF4 xa = KF4ZERO, xb = KF4ZERO;
  if (act) {
    const float* xx = k_xrow(J, rhs, c0);
    xa = KLOADF4(xx);
    xb = KLOADF4(xx + 4);
  }
  const u8* wb = (const u8*)J.w;
  const u8* sb = J.sc;
  float part[NR < 8 ? 8 : NR];  // k_warp_sum8t sums 8 at a time
  KUNROLL for (int r = 0; r < (NR < 8 ? 8 : NR); ++r) part[r] = 0.0f;
  if (act) {
    if (J.fmt == INK2_DFMT_NVFP4) {
      KUNROLL for (int r = 0; r < NR; ++r) if (r < nrows) part[r] = k_pk8(wb, sb, INK2_DFMT_NVFP4, (size_t)(row0 + r), cols, c0, xa, xb);
    } else if (J.fmt == INK2_DFMT_MXFP8) {
      KUNROLL for (int r = 0; r < NR; ++r) if (r < nrows) part[r] = k_pk8(wb, sb, INK2_DFMT_MXFP8, (size_t)(row0 + r), cols, c0, xa, xb);
    } else if (J.fmt == INK2_DFMT_Q3) {
      KUNROLL for (int r = 0; r < NR; ++r) if (r < nrows) part[r] = k_pk8(wb, sb, INK2_DFMT_Q3, (size_t)(row0 + r), cols, c0, xa, xb);
    } else {
      KUNROLL for (int r = 0; r < NR; ++r) if (r < nrows) part[r] = k_pk8(wb, sb, INK2_DFMT_BF16, (size_t)(row0 + r), cols, c0, xa, xb);
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
KERNEL(pgemvc, GemvArgs) {
  KSHARED(float, sm, 16 * 32);
  k_pgemvc<16>(KCTXA a, sm);
}
KERNEL(pgemvc8, GemvArgs) {
  KSHARED(float, sm, 8 * 32);
  k_pgemvc<8>(KCTXA a, sm);
}
KERNEL(pgemvc4, GemvArgs) {
  KSHARED(float, sm, 8 * 32);
  k_pgemvc<4>(KCTXA a, sm);
}
KERNEL(pgemvc2, GemvArgs) {
  KSHARED(float, sm, 8 * 32);
  k_pgemvc<2>(KCTXA a, sm);
}

// pgemvn: narrow rows (cols <= 256, all jobs the same cols, cols / 8 a power
// of two): lpr = cols / 8 lanes per row, 32 / lpr rows per warp, 8 warps per
// block; lane l of a row group multiplies the 8 columns 8 l .. 8 l + 7.
KERNEL(pgemvn, GemvArgs) {
  int lane = KTID & 31;
  int b = KBID, rhs = 0;
  if (a.nrhs > 1) {
    rhs = b / a.blocks;
    b -= rhs * a.blocks;
  }
  int lpr = a.job[0].cols >> 3, rpw = 32 / lpr;
  int sub = lane / lpr, u = lane - sub * lpr;
  int row = (b * (KNT >> 5) + (KTID >> 5)) * rpw + sub;
  int live = row < a.total_rows;
  int j = 0, r = live ? row : 0;
  while (r >= a.job[j].rows) {
    r -= a.job[j].rows;
    ++j;
  }
  const GemvJob& J = a.job[j];
  float s = 0.0f;
  if (live) {
    const float* xx = k_xrow(J, rhs, u << 3);
    KF4 xa = KLOADF4(xx), xb = KLOADF4(xx + 4);
    s = k_pk8((const u8*)J.w, J.sc, J.fmt, (size_t)r, J.cols, u << 3, xa, xb);
  }
  for (int o = lpr >> 1; o > 0; o >>= 1) s = s + KSHFL(s, lane ^ o);
  if (live && u == 0) J.y[(size_t)rhs * J.rows + r] = k_jout(J, s);
}

// ---------------------------------------------------------------------------
// pgemvcm: pgemvc (column owner, NR rows per block) for up to K_NRHS right-
// hand sides per block, so that a pass of R rows reads the packed weights
// once per group of 8 rows instead of once per row. Block b of group g does
// the work of block b of pgemvc on the input rows 8g .. 8g+7: thread t owns
// the columns 8t .. 8t+7, the 8 weights of a row are decoded once (k_pkload /
// k_pkdec) and multiplied with every input row in the order of k_pk8 (FMA
// chain from 0, then the block scale), the sums go through k_warp_sum8t (bit
// for bit k_warp_sum) and the warps are added in order, as in pgemvc. So a
// pass gives bit for bit the single steps with any NR. All code words of the
// block are loaded before the first is used.
// ---------------------------------------------------------------------------
struct K_PKW {
  u64 a, b;    // code bytes (nvfp4: 4 in a, mxfp8: 8 in a, q3: 3 in a, bfloat16: 16 in a and b)
  unsigned s;  // block scale byte (packed formats)
};
KINL K_PKW k_pkload(const u8* wb, const u8* sb, int fmt, size_t r, int cols, int c0) {
  K_PKW w;
  w.b = 0;
  w.s = 0;
  if (fmt == INK2_DFMT_NVFP4) {
    w.a = (u64)KLOAD32(wb + r * (size_t)(cols >> 1) + (size_t)(c0 >> 1));
    w.s = sb[r * (size_t)(cols >> 4) + (size_t)(c0 >> 4)];
  } else if (fmt == INK2_DFMT_MXFP8) {
    w.a = KLOAD64(wb + r * (size_t)cols + (size_t)c0);
    w.s = sb[r * (size_t)(cols >> 5) + (size_t)(c0 >> 5)];
  } else if (fmt == INK2_DFMT_Q3) {
    w.a = (u64)k_q3word(wb, r, cols, c0);
    w.s = sb[r * (size_t)(cols >> 4) + (size_t)(c0 >> 4)];
  } else {
    const u16* p = (const u16*)wb + r * (size_t)cols + (size_t)c0;
    w.a = KLOAD64(p);
    w.b = KLOAD64(p + 4);
  }
  return w;
}
// The 8 weights as floats (the values k_pk8 multiplies) -> f; returns the block scale (1 for bfloat16).
KINL float k_pkdec(const K_PKW& w, int fmt, float* f) {
  if (fmt == INK2_DFMT_NVFP4) {
    unsigned c = (unsigned)w.a;
    for (int e = 0; e < 8; ++e) f[e] = k_fp4x2(c >> (4 * e));
    return k_e4m3v(w.s);
  }
  if (fmt == INK2_DFMT_MXFP8) {
    for (int e = 0; e < 8; ++e) f[e] = k_e4m3v((unsigned)(w.a >> (8 * e)));
    return k_exp2v(w.s);
  }
  if (fmt == INK2_DFMT_Q3) {
    unsigned c = (unsigned)w.a;
    for (int e = 0; e < 8; ++e) f[e] = k_q3v(c >> (3 * e));
    return k_e4m3v(w.s);
  }
  K_BF8(w.a, w.b, f)
  return 1.0f;
}
// k_warp_sum of NH values at once (NH = 8, 4, 2): value i ends in the lanes i * 32 / NH .. (each bit for bit
// k_warp_sum: the pairs of its tree, a + b == b + a).
KINL float k_warp_sum2t(KCTXP const float* v) {
  int lane = KTID & 31;
  int b4 = lane & 16;
  float t = (b4 ? v[1] : v[0]) + KSHFL(b4 ? v[0] : v[1], lane ^ 16);
  t = t + KSHFL(t, lane ^ 8);
  t = t + KSHFL(t, lane ^ 4);
  t = t + KSHFL(t, lane ^ 2);
  t = t + KSHFL(t, lane ^ 1);
  return t;
}
template <int NH>
KINL float k_warp_sumt(KCTXP const float* v) {
  return NH == 8 ? k_warp_sum8t(KCTXA v) : NH == 4 ? k_warp_sum4t(KCTXA v) : k_warp_sum2t(KCTXA v);
}
template <int NR, int NH>
KINL void k_pgemvcm(KCTXP const GemvArgs& a, float* sm) {
  int lane = KTID & 31, warp = KTID >> 5, nw = KNT >> 5;
  int grp = KBID / a.blocks;
  int b = KBID - grp * a.blocks, j = 0;
  for (;;) {
    int nb = (a.job[j].rows + NR - 1) / NR;
    if (b < nb) break;
    b -= nb;
    ++j;
  }
  const GemvJob& J = a.job[j];
  int r0 = grp * NH;
  int nr = a.nrhs - r0;
  if (nr > NH) nr = NH;
  int cols = J.cols;
  int row0 = b * NR;
  int nrows = J.rows - row0;
  if (nrows > NR) nrows = NR;
  int c0 = KTID * 8;
  int act = c0 < cols;
  int fmt = J.fmt;
  float x[NH][8];
  for (int i = 0; i < NH; ++i) {
    for (int e = 0; e < 8; ++e) x[i][e] = 0.0f;
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
  const u8* wb = (const u8*)J.w;
  K_PKW wr[NR];
  KUNROLL for (int r = 0; r < NR; ++r) {
    wr[r].a = wr[r].b = 0;
    wr[r].s = 0;
    if (act && r < nrows) wr[r] = k_pkload(wb, J.sc, fmt, (size_t)(row0 + r), cols, c0);
  }
  KUNROLL for (int r = 0; r < NR; ++r) {
    if (r < nrows) {
      float acc[NH];
      for (int i = 0; i < NH; ++i) acc[i] = 0.0f;
      if (act) {
        float f[8];
        float scl = k_pkdec(wr[r], fmt, f);
        for (int i = 0; i < NH; ++i) {
          if (i < nr) {
            float s = KFMA(f[0], x[i][0], 0.0f);
            s = KFMA(f[1], x[i][1], s);
            s = KFMA(f[2], x[i][2], s);
            s = KFMA(f[3], x[i][3], s);
            s = KFMA(f[4], x[i][4], s);
            s = KFMA(f[5], x[i][5], s);
            s = KFMA(f[6], x[i][6], s);
            s = KFMA(f[7], x[i][7], s);
            acc[i] = fmt == INK2_DFMT_BF16 ? s : s * scl;
          }
        }
      }
      float t = k_warp_sumt<NH>(KCTXA acc);  // the sum of value i in lanes i * 32 / NH ..
      if ((lane & (32 / NH - 1)) == 0) sm[(r * NH + lane / (32 / NH)) * 32 + warp] = t;
    }
  }
  KSYNC();
  for (int t = KTID; t < nrows * nr; t += KNT) {
    int r = t / nr, i = t - r * nr;
    float s = 0.0f;
    for (int w = 0; w < nw; ++w) s += sm[(r * NH + i) * 32 + w];
    J.y[(size_t)(r0 + i) * J.rows + row0 + r] = k_jout(J, s);
  }
}
KERNEL(pgemvcm, GemvArgs) {
  KSHARED(float, sm, 16 * K_NRHS * 32);
  k_pgemvcm<16, K_NRHS>(KCTXA a, sm);
}
KERNEL(pgemvcm8, GemvArgs) {
  KSHARED(float, sm, 8 * K_NRHS * 32);
  k_pgemvcm<8, K_NRHS>(KCTXA a, sm);
}
KERNEL(pgemvcm4, GemvArgs) {
  KSHARED(float, sm, 4 * K_NRHS * 32);
  k_pgemvcm<4, K_NRHS>(KCTXA a, sm);
}
KERNEL(pgemvcm2, GemvArgs) {
  KSHARED(float, sm, 2 * K_NRHS * 32);
  k_pgemvcm<2, K_NRHS>(KCTXA a, sm);
}

// pgemvcm for passes of 2 (h2) or 3 .. 4 (h4) rows: NH right-hand sides per block, so that the inputs held
// in registers and the sums match the rows of the pass (same arithmetic as pgemvcm, i.e. as pgemvc).
#define K_PGCM(NR, NH)                                       KERNEL(pgcm##NR##h##NH, GemvArgs) {                          KSHARED(float, sm, NR * NH * 32);                          k_pgemvcm<NR, NH>(KCTXA a, sm);                          }
K_PGCM(16, 2)
K_PGCM(8, 2)
K_PGCM(4, 2)
K_PGCM(2, 2)
K_PGCM(16, 4)
K_PGCM(8, 4)
K_PGCM(4, 4)
K_PGCM(2, 4)

#include "ink2_drafter_kernels.inc"  // drafter hook
