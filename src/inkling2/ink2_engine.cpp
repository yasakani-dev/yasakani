// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Inkling-2 Mini engine: the orchestration shared by both backends (schedule
// of the layer visits, caches, the sequence of kernels of a visit) and the C
// API. The arithmetic follows our CPU reference implementation (ink2_ref, not part of this repository) (mode (b), one token per
// call, float32); the comments name the functions of that file.
// A pass may also advance R consecutive positions at once (ink2_feed_many,
// ink2_step_many): every row then gets bit for bit the arithmetic of a single
// step by default (the multi-row kernels keep the order of every sum), the rows see the
// earlier rows of the pass through the caches and convolution histories, and
// every weight matrix is read once for all rows (experts: once per distinct
// expert of the pass). The optional NVFP4 Tensor Core prompt mode instead rounds activations.
// Image input (ink2_vision_api.h): the image tower writes image vectors into a feature buffer; a prompt
// position that names a feature row takes that row as its embedding (the norm kernel, NormArgs.img) and is
// an ordinary position for everything after it.
#define INK2_BUILD
#include "ink2_api.h"
#include "ink2_vision_api.h"
#ifdef INK2_WITH_VISION
#include "ink2_vision.h"  // the image tower (ink2_vision.cu): ink2v_*
#else
struct Ink2Vis;  // no image tower in this build: the ink2_vision_* calls refuse with a message
#endif

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "ink2_kargs.h"
#include "ink2_fast.h"   // fast kernels (RTX 30 and later)
#include "ink2_fast2.h"  // fast kernels, second set
#include <stdlib.h>

int ink2_run_selftest(Backend* ref, Backend* dut, std::string& report);
int ink2_cuda_microbench(char* report, int cap);

// ink2_ckpt.make_e4m3_table and the exponent table of decode_mxfp8
void ink2_make_tables(float* e4m3, float* exp2t) {
  for (int b = 0; b < 256; ++b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    int e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = ldexpf((float)m / 8.0f, -6);
    else if (e == 15 && m == 7) v = nanf("");
    else v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    e4m3[b] = sign * v;
    exp2t[b] = ldexpf(1.0f, b - 127);
  }
}

namespace {

std::string g_last_error;

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

struct Ten {
  void* p = nullptr;
  long long nbytes = 0;
  std::vector<unsigned char> host;  // copy of small tensors
};

// A dense (non-expert) matrix: bfloat16, or packed (nvfp4 / mxfp8) with its scale bytes.
// fac != 0 multiplies every row sum (packed nvfp4: amax / 2688 / 2; bfloat16 rows with a
// "_factor" tensor: amax / 2688, the decoded nvfp4 values without the matrix factor).
struct DMat {
  const void* w = nullptr;
  const u8* sc = nullptr;
  float fac = 0.0f;
  int fmt = INK2_DFMT_BF16;
};

struct Layer {
  bool sliding = true, loop = false;
  int nh = 0, nkv = 0, hd = 0, rel_extent = 0, nq = 1;
  int fmt13 = 0, fmt2 = 0;
  const float *attn_query = nullptr, *mlp_query = nullptr, *router_bias = nullptr;
  const u16 *norm_q = nullptr, *norm_k = nullptr, *norm_v = nullptr, *mlp_norm = nullptr;
  DMat wq, wk, wv, wr, wo, w_gate, w_sink, k_wproj, v_wproj, router_w;
  const u16 *q_scale = nullptr, *rel_scale = nullptr, *rel_proj = nullptr;
  u16* rel_t = nullptr;  // rel_proj as [rel_extent, d_rel], built when splitk_attn is set
  const u16* rel_proj_t = nullptr;  // [rel_extent, drel], uploaded by the option rel_t
  const u16 *k_w = nullptr, *v_w = nullptr;
  const u16 *attn_sconv = nullptr, *mlp_sconv = nullptr;
  const u8 *w13 = nullptr, *w13s = nullptr, *w2 = nullptr, *w2s = nullptr;
  const float *f13 = nullptr, *f2 = nullptr;
  DMat g_down[4], g_forget[4], g_gain[4];
};

// What one layer visit keeps: the key / value cache (cap slots; on a sliding
// layer a ring of window + margin slots) and the four convolution histories
// (rings of hr slots), and its last outputs for every row of a pass.
struct Visit {
  KvStorage kv{};
  int layer = 0, pass = 0, cap = 0;
  float *kc = nullptr, *vc = nullptr, *khist = nullptr, *vhist = nullptr, *ahist = nullptr, *mhist = nullptr;
  float *rlog = nullptr, *rwt = nullptr, *hid = nullptr;  // [max_rows, ...]
  int* ridx = nullptr;
};

struct Capture {
  std::string name;
  std::vector<float> data;
};

struct Drafter;  // drafter hook (ink2_drafter.inc)

struct Engine {
  Ink2Config c{};
  Backend* be = nullptr;
  std::string err;
  std::map<std::string, Ten> ten;
  std::vector<void*> owned;
  bool finalized = false;
  int pos = 0;

  // rows of a pass and the margin of the rings (options before finalize)
  int max_rows = 128;  // largest R of a multi-row pass
  int rollback = 64;   // positions ink2_truncate can always drop
  int margin = 0;      // max(max_rows, rollback): extra slots of the rings
  // KV cache: INK2_KV_FP32 (default) or INK2_KV_MXFP8 (option kv_mxfp8; full-attention visits only, see ink2_kv_mxfp8.inc)
  int kv_mode = INK2_KV_FP32, kv_hot = 128, kv_split = 0, global_splits = 1;
  // Tiled Flash-GQA attention of prompt passes: 0 off (default), 1 FP32 (8 rows x 3 heads per block),
  // 2 BF16 tensor cores (16 rows x 3 heads), 3 BF16 tensor cores (32 rows x 3 heads)
  int flash_attn = 0, flash_rel_cap = 0, flash_sms = 0;
  int flash_min_rows = 8;  // smallest pass for flash (9 keeps the 8-row draft checks on the cache kernels)
  float *flash_rel = nullptr, *flash_partial = nullptr;
  enum { flash_max_splits = 16 };
  bool flash_hook(AttnArgs& a, const Layer& L, const Visit& V, int R) {
    if (!flash_attn || R < flash_min_rows || L.nh != 3 * L.nkv || L.rel_extent > flash_rel_cap)
      return false;
    if (exact_logit_pass(R) && splitk_use(L, R)) return false;  // rows with logits stay on the single-step kernel
    a.kv = V.kv;
    a.rel_logits = flash_rel;
    a.rel_stride = flash_rel_cap;
    run(KID_attn_rel, &a, R * L.nh, 256, INK2_PROF_ATTENTION);
    const int per = flash_attn == 3 ? 32 : flash_attn == 2 ? 16 : 8;
    const int tiles = (R + per - 1) / per, blocks = tiles * L.nkv;
    int splits = flash_sms ? (2 * flash_sms + blocks - 1) / blocks : 1;
    a.splits = splits < 1 ? 1 : splits > flash_max_splits ? flash_max_splits : splits;
    a.partial = flash_partial;
    a.flash_tc = flash_attn == 3 ? 6 : flash_attn == 2 ? 3 : 0;
    run(KID_attn_flash, &a, blocks * a.splits, a.flash_tc ? 32 * a.flash_tc : 256, INK2_PROF_ATTENTION);
    if (a.splits > 1) run(KID_kv_merge, &a, R * L.nh, 128, INK2_PROF_ATTENTION);
    return true;
  }
  bool rel_t = false;  // attention reads the transposed rel_proj (same bits)
  size_t kv_cache_bytes = 0, kv_work_bytes = 0;
  float *kv_partial = nullptr, *capture_k = nullptr, *capture_v = nullptr;
  // GQA-shared Split-K attention for short passes (option splitk_attn, default off)
  int splitk_attn = 0, splitk_splits = 0, splitk_chunk = 64;
  int splitk_max_rows = 7;  // largest pass for splitk (8 = the 8-row draft checks too)
  bool splitk_rel = true;
  float* splitk_partial = nullptr;  // [splitk_rows, nh_full, 32 splits, 132]
  // pass_exact (default 1): every row of a pass with logits (ink2_step_many: verification and copy passes) gets bit for
  // bit the logits of a single step at its position. Split-K then cuts every row's history into the splits a single
  // step uses (the count of R = 1, not of R), and with splitk_attn on, passes with logits of up to splitk_rows rows
  // take the Split-K kernel of the single step (never flash or the pass kernels). Prompt passes (no logits) keep their
  // kernels. Acts only with splitk_attn.
  int pass_exact = 1;
  bool cur_logits = false;  // the current pass reads logits (set by pass(), cleared after it)
  int splitk_rows = 8;      // rows the Split-K partials hold (max(8, logit_rows) once allocated)
  bool exact_logit_pass(int R) const { return pass_exact && cur_logits && R > 1 && R <= splitk_rows; }
  bool splitk_use(const Layer& L, int R) const {
    return splitk_attn && splitk_partial && (R <= splitk_max_rows || exact_logit_pass(R)) && L.hd == 128 && L.nkv > 0 &&
           L.nh % L.nkv == 0 && L.nh / L.nkv <= 4 && c.d_rel <= 32;
  }
  int splitk_count(int R, int nkv) const {
    if (splitk_splits) return splitk_splits;
    if (pass_exact) R = 1;  // the splits of a single step for every row (sum order independent of the pass size)
    int sm = be->sm_count(), n = ((sm ? 2 * sm : 64) + R * nkv - 1) / (R * nkv);
    return n < 1 ? 1 : n > 32 ? 32 : n;
  }
  void splitk_prepare() {
    if (!splitk_partial) {
      size_t start = be->bytes_allocated();
      splitk_rows = (std::max)(8, logit_rows);  // passes with logits up to logit_rows rows
      splitk_partial = alloc<float>((size_t)splitk_rows * c.nh_full * 32 * 132);
      kv_work_bytes += be->bytes_allocated() - start;
    }
    for (Layer& L : layers) {
      if (L.rel_t || !L.rel_proj || !L.rel_extent) continue;
      const size_t n = (size_t)c.d_rel * L.rel_extent;
      std::vector<u16> src(n), dst(n);
      be->download(src.data(), L.rel_proj, n * 2);
      for (int d = 0; d < c.d_rel; ++d)
        for (int t = 0; t < L.rel_extent; ++t) dst[(size_t)t * c.d_rel + d] = src[(size_t)d * L.rel_extent + t];
      L.rel_t = alloc<u16>(n);
      be->upload(L.rel_t, dst.data(), n * 2);
    }
    check_backend();
  }
  int hr = 0;          // slots of a convolution history ring (K - 1 + margin)
  int logit_rows = 0;  // largest R of ink2_step_many (rows with logits)
  int chunk = 128;     // rows per pass of ink2_feed_many
  int nvfp4_planes = 0;
  u8* nvfp4_scratch = nullptr;
  int decode_tiles = 0;

  std::vector<Layer> layers;
  std::vector<Visit> visits;
  const u16 *embed_norm = nullptr, *final_norm = nullptr;
  DMat embed, unembed;
  const float *output_query = nullptr, *msg_q = nullptr;
  float *e4m3 = nullptr, *exp2t = nullptr;
  StepState* st = nullptr;  // [max_rows]
  std::vector<StepState> st_host;
  // work buffers, [max_rows, ...]
  float *emb = nullptr, *qin = nullptr, *kin = nullptr, *vin = nullptr;
  float *xs[3] = {}, *xsn[3] = {};  // attention inputs before the gates, and their normalised copies
  float *msg[4] = {}, *msgn[4] = {};
  float *gd[4] = {}, *gf[4] = {}, *gg[4] = {};  // gate projections
  float *qv = nullptr, *rv = nullptr, *gatev = nullptr, *sinkv = nullptr, *kraw = nullptr, *vraw = nullptr;
  float *kdyn = nullptr, *vdyn = nullptr, *att = nullptr, *ao = nullptr, *resid = nullptr;
  float *xm = nullptr, *xm_n = nullptr, *xmn = nullptr;
  float *act = nullptr, *ey = nullptr, *fh = nullptr, *logits = nullptr, *scratch = nullptr;
  float* ydots = nullptr;  // expert products of a pass [max_rows * top_k, H]
  int *gcount = nullptr, *gexp = nullptr, *goff = nullptr, *gpairs = nullptr;
  int* amax_idx = nullptr;
  float* amax_val = nullptr;
  int* pick_indices = nullptr;
  float* pick_values = nullptr;
  bool device_greedy = false;
  int scratch_stride = 0;
  std::vector<float> logits_host;

  bool use_graph = false;
  bool use_gemvc = true, use_exp13c = true;  // column-owner forms of the large products
  int packed_dense = 0;                       // dense matrices stored packed (nvfp4 / mxfp8)
  int pgemvc_rows = 0;         // rows per block of the packed column-owner product: 0 (by size), 2, 4, 8, 16
  int pgemvc_min_blocks = 680;  // the size rule of pgemvc_rows = 0 (4 blocks per SM of an RTX 5090)
  int pgemvw_rows = 2;          // packed one-block-per-row products: 0 = pgemvw, else pgemvc with these rows per block
  bool use_pgemvn = true;       // narrow packed rows (cols <= 256): several rows per warp
  bool use_pgemvcm = true;      // passes of several rows: packed column owner for 8 rows at a time (else once per row)
  // Rows per block of pgemvcm (any choice gives the same sums): 0 = by the rows of the pass, 2 up to 8 rows
  // (verification passes), 16 above (prompt passes); else 2, 4, 8, 16. Passes of fewer than pgemvcm_min_rows
  // rows use the single-row kernels once per row (faster for 2 .. 5 rows: A/B in README.md).
  int pgemvcm_rows = 0;
  int pgemvcm_min_rows = 6;
  // Passes of 2 .. small_rows rows (verification passes; 0 = off): experts through exp13s / exp2s, packed
  // dense products through pgemvcm with 2 / 4 / 8 right-hand sides per block (as many as the pass needs);
  // small_dense_rows = rows per block of those (0: the size rule of pgemvc, else 2, 4, 8, 16).
  int small_rows = 4;  // measured (A/B in one process): exp13s pays at R = 2 .. 4; 0 = the grouped kernels always
  bool use_small_experts = true, use_small_dense = false, use_small_w2 = false;  // (measured: exp2s and the h2 / h4 dense
  // products are not faster than exp2m and the single-row products on the RTX 5090; kept as options)
  int small_dense_rows = 0;
  void* graph[4] = {nullptr, nullptr, nullptr, nullptr};  // without / with the output projection (+ 2: the split attention of the fast kernels)
  // Row count/output flag, attention variant, and the optional fixed-depth draft shape.
  std::map<std::tuple<int, bool, int>, void*> mgraph;
  bool joined_draft_graph = false;
  int graphs_built = 0;
  bool prof = false;
  double prof_t[INK2_PROF_COUNT] = {};
  double prof_mark = 0;
  int bench_reps = 0;
  double bench_t[KID_COUNT] = {};
  long long bench_n[KID_COUNT] = {};
  double t_enqueue = 0, t_wait = 0;
  long long t_steps = 0, n_launch = 0;
  int cap_visit = -1;
  std::vector<Capture> caps;
  Drafter* drf = nullptr;  // drafter hook: defined in ink2_drafter.inc
  // Image input (ink2_vision_api.h). img_feat is fixed from ink2_vision_finalize to ink2_vision_unload, so the
  // recorded kernel sequences stay valid in between (the rows are named per position through st[].pad).
  Ink2Vis* vis = nullptr;               // the image tower (ink2_vision.cu)
  Ink2VisionConfig vis_cfg{};
  bool vis_ready = false;               // ink2_vision_finalize done
  bool vis_capture = false;             // option "vision_capture": keep stages 1 .. 3 of every encode (tests)
  u16* img_feat = nullptr;              // feature buffer, bfloat16 [vis_cfg.max_feature_rows, hidden]
  std::vector<unsigned char> img_valid;  // per feature row: written since the last ink2_reset
  void vision_release();
#include "ink2_fast_engine.inc"   // fast kernels
#include "ink2_fast2_engine.inc"  // fast kernels, second set
  void drf_commit(int R);
  int drf_before_pass(int R, bool can_join);
  void drf_record_pending(int depth);
  void drf_free_graphs();
  void drf_destroy();

  ~Engine() {
    if (be) {
      drf_destroy();  // drafter hook
      vision_release();
      free_graphs();
      for (void* p : owned) be->release(p);
      for (auto& kv : ten)
        if (kv.second.p) be->release(kv.second.p);
      delete be;
    }
  }

  void free_graphs() {
    for (void*& g : graph) {
      if (g) be->graph_free(g);
      g = nullptr;
    }
    for (auto& kv : mgraph)
      if (kv.second) be->graph_free(kv.second);
    mgraph.clear();
  }
  void check_backend() {
    const char* e = be->error();
    if (e) fail(e);
  }
  template <class T>
  T* alloc(size_t count) {
    void* p = be->alloc(count * sizeof(T));
    if (!p) fail(be->error() ? be->error() : "allocation failed");
    owned.push_back(p);
    return (T*)p;
  }
  // the transposed relative projection of every layer (once; the same values)
  void upload_rel_t() {
    for (Layer& L : layers) {
      if (L.rel_proj_t || !L.rel_proj) continue;
      const size_t n = (size_t)c.d_rel * L.rel_extent;
      std::vector<u16> src(n), dst(n);
      be->download(src.data(), L.rel_proj, n * sizeof(u16));
      for (int d = 0; d < c.d_rel; ++d)
        for (int t = 0; t < L.rel_extent; ++t) dst[(size_t)t * c.d_rel + d] = src[(size_t)d * L.rel_extent + t];
      u16* p = alloc<u16>(n);
      be->upload(p, dst.data(), n * sizeof(u16));
      L.rel_proj_t = p;
    }
    be->sync();
    check_backend();
  }
  void reserve_kv_partials(int splits) {
    if (splits > 1 && !kv_partial) {
      // Split count can change while the cache is live. Allocate once, keep
      // its address stable, and reuse it across all global visits.
      kv_partial = alloc<float>((size_t)max_rows * c.nh_full * 32 * 130);
    }
  }
  void run(int kid, const void* args, int grid, int block, int cat) {
    f2_flush();  // fast kernels, second set
    if (bench_reps > 0) {  // timing only: the same kernel many times in a row (the results are not valid)
      be->sync();
      double t0 = now_s();
      for (int r = 0; r < bench_reps; ++r) be->run(kid, args, grid, block);
      be->sync();
      bench_t[kid] += (now_s() - t0) / bench_reps;
      ++bench_n[kid];
      return;
    }
    be->run(kid, args, grid, block);
    ++n_launch;
    if (prof) {
      be->sync();
      double t = now_s();
      prof_t[cat] += t - prof_mark;
      prof_mark = t;
    }
  }
  void capture(int vi, const char* name, const void* p, size_t nfloats) {
    if (vi != cap_visit) return;
    Capture cp;
    cp.name = name;
    cp.data.resize(nfloats);
    be->download(cp.data.data(), p, nfloats * 4);
    caps.push_back(std::move(cp));
  }

  // -- tensors -------------------------------------------------------------
  const void* need(const std::string& name, long long nbytes) {
    auto it = ten.find(name);
    if (it == ten.end()) fail("missing tensor " + name);
    if (it->second.nbytes != nbytes)
      fail(fmt("tensor %s has %lld bytes, expected %lld", name.c_str(), it->second.nbytes, nbytes));
    return it->second.p;
  }
  bool has(const std::string& name) { return ten.find(name) != ten.end(); }
  const u16* bf(const std::string& name, long long count) { return (const u16*)need(name, count * 2); }
  const float* f32(const std::string& name, long long count) { return (const float*)need(name, count * 4); }
  float host_f32(const std::string& name) {
    need(name, 4);
    float v;
    memcpy(&v, ten[name].host.data(), 4);
    return v;
  }
  // A dense matrix [rows, cols]: "name" alone = bfloat16; with "name_scale" = packed, nvfp4 when
  // "name_factor" (float32 [1], amax / 2688) is there, else mxfp8; "name" bfloat16 with
  // "name_factor" = decoded nvfp4 values without the matrix factor (used to check the packed path).
  DMat dmat(const std::string& name, long long rows, long long cols) {
    DMat m;
    long long n = rows * cols;
    bool hasf = has(name + "_factor");
    if (has(name + "_scale")) {
      if (cols % 32) fail("packed matrix " + name + ": columns must be a multiple of 32");
      if (hasf) {
        m.fmt = INK2_DFMT_NVFP4;
        m.w = need(name, n / 2);
        m.sc = (const u8*)need(name + "_scale", n / 16);
        m.fac = host_f32(name + "_factor") * 0.5f;  // the kernels decode the codes as twice the E2M1 value
      } else {
        m.fmt = INK2_DFMT_MXFP8;
        m.w = need(name, n);
        m.sc = (const u8*)need(name + "_scale", n / 32);
      }
    } else {
      m.w = need(name, n * 2);
      if (hasf) m.fac = host_f32(name + "_factor");
    }
    if (m.fmt != INK2_DFMT_BF16) ++packed_dense;
    return m;
  }

  void load_experts(const std::string& base, long long rows, long long cols, int* fmt_out, const u8** w, const u8** s,
                    const float** factor) {
    long long E = c.experts;
    if (has(base + "_amax")) {
      *fmt_out = INK2_FMT_NVFP4;
      *w = (const u8*)need(base, E * rows * cols / 2);
      *s = (const u8*)need(base + "_scale", E * rows * cols / 16);
      need(base + "_amax", E * 4);
      need(base + "_den", 4);
      const Ten& amax = ten[base + "_amax"];
      const Ten& den = ten[base + "_den"];
      std::vector<float> f(E);
      float d;
      memcpy(&d, den.host.data(), 4);
      for (long long e = 0; e < E; ++e) {
        float am;
        memcpy(&am, amax.host.data() + 4 * e, 4);
        f[e] = (am / d) * 0.5f;  // the codes are decoded as 2 * e2m1
      }
      float* dev = alloc<float>(E);
      be->upload(dev, f.data(), E * 4);
      *factor = dev;
    } else {
      *fmt_out = INK2_FMT_MXFP8;
      *w = (const u8*)need(base, E * rows * cols);
      *s = (const u8*)need(base + "_scale", E * rows * cols / 32);
      *factor = nullptr;
    }
  }

  void finalize() {
    if (finalized) fail("finalize called twice");
    const long long H = c.hidden, I = c.expert_inner, E = c.experts, K = c.conv_k, R = c.gate_rank;
    if (c.layers < 1 || c.layers > 64) fail("layers out of range");
    if (H % 32 || I % 32) fail("hidden and expert sizes must be multiples of 32");
    if (c.top_k < 2 || c.top_k % 2 || c.top_k > 32 || c.top_k > E) fail("top_k must be even, 2..32");
    if (E > 1024) fail("more than 1024 experts");
    if (R % 8) fail("gate rank must be a multiple of 8");
    if (K < 2 || K > 8) fail("convolution kernel must be 2..8");
    if (c.loop_start < 1 || c.loop_end > c.layers || c.loop_start >= c.loop_end) fail("bad loop range");
    if (c.loop_start % c.bar_block || c.loop_end % c.bar_block) fail("loop range must be whole blocks");
    if ((c.layers + c.bar_block - 1) / c.bar_block + 1 > 6) fail("too many attention-residual blocks");
    if (H > 4096) fail("hidden size above 4096");
    if (c.max_positions < 1) fail("max_positions must be positive");
    if (I > 16 * 16 * 5 || H > 32 * 16 * 5) fail("expert inner size above 1280 or hidden size above 2560 (multi-row expert kernels)");
    if (max_rows < 1 || (long long)max_rows * c.top_k > 2048) fail("max_rows must be 1 .. 2048 / top_k");
    if (rollback < 0 || rollback > 0x7fffffff - 256) fail("rollback out of range");
    margin = max_rows > rollback ? max_rows : rollback;
    if (kv_split) global_splits = kv_split;
    else if (kv_mode != INK2_KV_FP32) {
      int sm = be->sm_count();
      global_splits = sm >= 150 ? 24 : sm >= 100 ? 16 : sm ? 12 : 1;
    }
    hr = (int)(K - 1) + margin;
    logit_rows = max_rows < 32 ? max_rows : 32;
    if (chunk > max_rows) chunk = max_rows;
    const size_t MR = (size_t)max_rows;

    embed = dmat("embed", (long long)c.embed_rows, H);
    if (embed.fmt == INK2_DFMT_NVFP4 || embed.fac != 0.0f) fail("the embedding may be bfloat16 or mxfp8 only");
    embed_norm = bf("embed_norm", H);
    final_norm = bf("norm", H);
    unembed = dmat("unembed", (long long)c.vocab, H);
    output_query = f32("output_query", H);
    int nloop = c.loop_end - c.loop_start;
    if (c.loop_passes > 1) msg_q = f32("msg_q", (long long)nloop * 4 * H);

    layers.resize(c.layers);
    int max_nh = 0, max_hd = 0, max_nkv = 0;
    for (int i = 0; i < c.layers; ++i) {
      Layer& L = layers[i];
      std::string p = fmt("L%d.", i);
      L.sliding = c.sliding[i] != 0;
      L.loop = i >= c.loop_start && i < c.loop_end;
      L.nh = L.sliding ? c.nh_sw : c.nh_full;
      L.nkv = L.sliding ? c.nkv_sw : c.nkv_full;
      L.hd = L.sliding ? c.hd_sw : c.hd_full;
      L.rel_extent = L.sliding ? c.window : c.rel_extent;
      L.nq = L.sliding ? 1 : 3;
      if (L.nh % L.nkv) fail("heads must be a multiple of key/value heads");
      if ((L.nh * L.hd) % 8) fail("attention width must be a multiple of 8");
      if (L.hd != 128 || c.d_rel > 32) fail("the attention kernel needs head size 128 and d_rel <= 32");
      long long A = (long long)L.nh * L.hd, C = (long long)L.nkv * L.hd;
      max_nh = L.nh > max_nh ? L.nh : max_nh;
      max_hd = L.hd > max_hd ? L.hd : max_hd;
      max_nkv = L.nkv > max_nkv ? L.nkv : max_nkv;
      if (i > 0) L.attn_query = f32(p + "attn_query", L.nq * H);
      L.mlp_query = f32(p + "mlp_query", H);
      if (L.sliding) {
        L.norm_q = L.norm_k = L.norm_v = bf(p + "attn_norm", H);
        L.k_w = bf(p + "k_conv", C * K);
        L.v_w = bf(p + "v_conv", C * K);
      } else {
        L.norm_q = bf(p + "q_norm", H);
        L.norm_k = bf(p + "k_norm", H);
        L.norm_v = bf(p + "v_norm", H);
        L.k_w = bf(p + "k_static", K * C);
        L.v_w = bf(p + "v_static", K * C);
        L.k_wproj = dmat(p + "k_wproj", (long long)L.nkv * K, H);
        L.v_wproj = dmat(p + "v_wproj", (long long)L.nkv * K, H);
      }
      L.mlp_norm = bf(p + "mlp_norm", H);
      L.wq = dmat(p + "wq", A, H);
      L.wk = dmat(p + "wk", C, H);
      L.wv = dmat(p + "wv", C, H);
      L.wr = dmat(p + "wr", (long long)L.nh * c.d_rel, H);
      L.wo = dmat(p + "wo", H, A);
      L.w_gate = dmat(p + "w_gate", A, H);
      L.w_sink = dmat(p + "w_sink", (long long)L.nh, H);
      L.q_scale = bf(p + "q_scale", A);
      L.rel_scale = bf(p + "rel_scale", (long long)L.nh * c.d_rel);
      L.rel_proj = bf(p + "rel_proj", (long long)c.d_rel * L.rel_extent);
      L.attn_sconv = bf(p + "attn_sconv", H * K);
      L.mlp_sconv = bf(p + "mlp_sconv", H * K);
      L.router_w = dmat(p + "router_w", E, H);
      L.router_bias = f32(p + "router_bias", E);
      load_experts(p + "w13", 2 * I, H, &L.fmt13, &L.w13, &L.w13s, &L.f13);
      load_experts(p + "w2", H, I, &L.fmt2, &L.w2, &L.w2s, &L.f2);
      if (L.loop && c.loop_passes > 1) {
        static const char* gname[4] = {"q", "k", "v", "mlp"};
        for (int g = 0; g < 4; ++g) {
          std::string gp = p + "gate_" + gname[g];
          L.g_down[g] = dmat(gp + "_down", R, 2 * H);
          L.g_forget[g] = dmat(gp + "_forget", H, R);
          L.g_gain[g] = dmat(gp + "_gain", H, R);
        }
      }
    }

    // schedule (ink2_ref.Config.schedule)
    std::vector<std::pair<int, int>> sched;
    for (int i = 0; i < c.loop_end; ++i) sched.push_back({i, 0});
    for (int l = 1; l < c.loop_passes; ++l)
      for (int i = c.loop_start; i < c.loop_end; ++i) sched.push_back({i, l});
    for (int i = c.loop_end; i < c.layers; ++i) sched.push_back({i, 0});

    int max_cap = 0;
    visits.resize(sched.size());
    for (size_t v = 0; v < sched.size(); ++v) {
      Visit& V = visits[v];
      V.layer = sched[v].first;
      V.pass = sched[v].second;
      const Layer& L = layers[V.layer];
      long long C = (long long)L.nkv * L.hd;
      // sliding layers: a ring of window + margin slots, so that the rows of a
      // pass and a rollback of `margin` positions find every key they see
      long long ring = (long long)c.window + margin;
      V.cap = L.sliding ? (int)(ring < c.max_positions ? ring : c.max_positions) : c.max_positions;
      max_cap = V.cap > max_cap ? V.cap : max_cap;
      size_t start = be->bytes_allocated(), count = (size_t)V.cap * C;
      KvStorage& k = V.kv;
      k.mode = kv_mode;
      if (kv_mode != INK2_KV_MXFP8 || L.sliding) {  // FP32 (the sliding visits stay FP32 with MXFP8)
        k.mode = INK2_KV_FP32;
        V.kc = alloc<float>(count);
        V.vc = alloc<float>(count);
      } else {  // MXFP8: E4M3 codes, one E8M0 byte per 32, BF16 copies of the sinks and the recent window
        k.key = alloc<u8>(count);
        k.value = alloc<u8>(count);
        k.ks = alloc<u8>(count / 32);
        k.vs = alloc<u8>(count / 32);
        k.sinks = V.cap < 8 ? V.cap : 8;
        k.hot = kv_hot;
        if (k.hot && V.cap > k.sinks) {
          k.hot_slots = k.hot + margin;
          if (k.hot_slots > V.cap - k.sinks) k.hot_slots = V.cap - k.sinks;
          k.tags = alloc<int>(k.hot_slots);
        }
        size_t near_elems = (size_t)(k.sinks + k.hot_slots) * C;
        k.near_k = alloc<u16>(near_elems);
        k.near_v = alloc<u16>(near_elems);
      }
      kv_cache_bytes += be->bytes_allocated() - start;
      V.khist = alloc<float>((size_t)hr * C);
      V.vhist = alloc<float>((size_t)hr * C);
      V.ahist = alloc<float>((size_t)hr * H);
      V.mhist = alloc<float>((size_t)hr * H);
      V.rlog = alloc<float>(MR * E);
      V.rwt = alloc<float>(MR * c.top_k);
      V.ridx = alloc<int>(MR * c.top_k);
      V.hid = alloc<float>(MR * H);
    }

    long long maxA = (long long)max_nh * max_hd, maxC = (long long)max_nkv * max_hd;
    st = alloc<StepState>(MR);
    st_host.resize(MR);
    emb = alloc<float>(MR * H);
    qin = alloc<float>(MR * H);
    kin = alloc<float>(MR * H);
    vin = alloc<float>(MR * H);
    for (int g = 0; g < 3; ++g) xs[g] = alloc<float>(MR * H), xsn[g] = alloc<float>(MR * H);
    for (int g = 0; g < 4; ++g) {
      msg[g] = alloc<float>(MR * H);
      msgn[g] = alloc<float>(MR * H);
      gd[g] = alloc<float>(MR * R);
      gf[g] = alloc<float>(MR * H);
      gg[g] = alloc<float>(MR * H);
    }
    qv = alloc<float>(MR * maxA);
    rv = alloc<float>(MR * max_nh * c.d_rel);
    gatev = alloc<float>(MR * maxA);
    sinkv = alloc<float>(MR * max_nh);
    kraw = alloc<float>(MR * maxC);
    vraw = alloc<float>(MR * maxC);
    kdyn = alloc<float>(MR * max_nkv * K);
    vdyn = alloc<float>(MR * max_nkv * K);
    att = alloc<float>(MR * maxA);
    ao = alloc<float>(MR * H);
    resid = alloc<float>(MR * H);
    xm = alloc<float>(MR * H);
    xm_n = alloc<float>(MR * H);
    xmn = alloc<float>(MR * H);
    act = alloc<float>(MR * c.top_k * I);
    ey = alloc<float>(MR * H);
    fh = alloc<float>(MR * H);
    logits = alloc<float>((size_t)logit_rows * c.vocab);
    ydots = alloc<float>(MR * c.top_k * H);
    gcount = alloc<int>(1);
    gexp = alloc<int>(E);
    goff = alloc<int>(E + 1);
    gpairs = alloc<int>(MR * c.top_k);
    amax_idx = alloc<int>(logit_rows);
    amax_val = alloc<float>(logit_rows);
    size_t work_start = be->bytes_allocated();
    if (kv_mode == INK2_KV_FP32 && !kv_split) {
      scratch_stride = max_cap;
      scratch = alloc<float>(MR * max_nh * scratch_stride);
    } else {
      if (kv_mode == INK2_KV_MXFP8 && !kv_split) {  // the FP32 sliding visits keep the original attention kernel
        scratch_stride = 0;
        for (const Visit& v : visits)
          if (layers[v.layer].sliding && v.cap > scratch_stride) scratch_stride = v.cap;
        if (scratch_stride) scratch = alloc<float>(MR * max_nh * scratch_stride);
      }
      reserve_kv_partials(global_splits);
    }
    if (kv_mode != INK2_KV_FP32) {
      capture_k = alloc<float>(maxC);
      capture_v = alloc<float>(maxC);
    }
    kv_work_bytes = be->bytes_allocated() - work_start;
    e4m3 = alloc<float>(256);
    exp2t = alloc<float>(256);
    float t1[256], t2[256];
    ink2_make_tables(t1, t2);
    be->upload(e4m3, t1, sizeof t1);
    be->upload(exp2t, t2, sizeof t2);
    logits_host.resize(c.vocab);
    for (auto& kv : ten) std::vector<unsigned char>().swap(kv.second.host);
    be->sync();
    check_backend();
    if (rel_t) upload_rel_t();
    finalized = true;
    if (splitk_attn) splitk_prepare();
    pos = 0;
    fast_resolve();  // fast kernels
    f2_resolve();    // fast kernels, second set
  }

  void reset() {
    const long long H = c.hidden;
    for (Visit& V : visits) {
      const Layer& L = layers[V.layer];
      long long C = (long long)L.nkv * L.hd;
      be->zero(V.khist, (size_t)hr * C * 4);
      be->zero(V.vhist, (size_t)hr * C * 4);
      be->zero(V.ahist, (size_t)hr * H * 4);
      be->zero(V.mhist, (size_t)hr * H * 4);
      if (V.kv.tags) be->zero(V.kv.tags, (size_t)V.kv.hot_slots * sizeof(int));
    }
    pos = 0;
    check_backend();
  }

  // Drops the positions from n on. The caches keep what positions below n
  // see as long as at most `margin` positions are dropped: the key / value
  // rings and the histories still hold them (later passes overwrite the
  // dropped slots). Before position K - 1 the histories must read zeros.
  void truncate(int n) {
    if (!finalized) fail("truncate before finalize");
    if (n < 0 || n > pos) fail(fmt("truncate to %d: only %d positions seen", n, pos));
    if (n == pos) return;
    if (n == 0) {
      reset();
      return;
    }
    if (pos - n > margin)
      fail(fmt("truncate drops %d positions, the caches keep %d (set the option \"rollback\" before finalize, or reset and feed again)",
               pos - n, margin));
    const int K1 = c.conv_k - 1;
    if (n < K1) {  // the slots of the positions -K1 .. -1 (hr - K1 .. hr - 1) must be zero again
      const long long H = c.hidden;
      for (Visit& V : visits) {
        const Layer& L = layers[V.layer];
        long long C = (long long)L.nkv * L.hd;
        be->zero(V.khist + (size_t)(hr - K1) * C, (size_t)K1 * C * 4);
        be->zero(V.vhist + (size_t)(hr - K1) * C, (size_t)K1 * C * 4);
        be->zero(V.ahist + (size_t)(hr - K1) * H, (size_t)K1 * H * 4);
        be->zero(V.mhist + (size_t)(hr - K1) * H, (size_t)K1 * H * 4);
      }
      check_backend();
    }
    pos = n;
  }

  // -- kernels with their launch shapes -------------------------------------
  static bool any_packed(const GemvArgs& a) {
    for (int j = 0; j < a.njobs; ++j)
      if (a.job[j].fmt != INK2_DFMT_BF16) return true;
    return false;
  }
  // y = W x for every job; R right-hand sides (rows of the pass)
  void k_gemv(GemvArgs& a, int cat, int R = 1) {
    a.total_rows = 0;
    bool same_cols = true;
    int chunks = 0;
    for (int j = 0; j < a.njobs; ++j) {
      a.total_rows += a.job[j].rows;
      same_cols = same_cols && a.job[j].cols == a.job[0].cols;
      chunks += (a.job[j].rows + 15) / 16;  // K_GEMVC_ROWS
    }
    int cols = a.job[0].cols;
    bool colowner = use_gemvc && same_cols && cols % 256 == 0 && cols / 8 <= 1024;
    if (any_packed(a)) {  // packed rows; R > 1: one set of blocks per row
      a.nrhs = R;
      int lpr = cols / 8;
      if (colowner) {
        k_pgemvc(a, pgemvc_rows, cat, R);
      } else if (use_pgemvn && same_cols && lpr <= 32 && (lpr & (lpr - 1)) == 0) {
        a.blocks = (a.total_rows + 8 * (32 / lpr) - 1) / (8 * (32 / lpr));
        run(KID_pgemvn, &a, a.blocks * R, 256, cat);
      } else {
        a.blocks = (a.total_rows + 7) / 8;
        run(KID_pgemv, &a, a.blocks * R, 256, cat);
      }
      return;
    }
    if (R == 1) {
      if (colowner) run(KID_gemvc, &a, chunks, cols / 8, cat);  // column-owner form
      else run(KID_gemv, &a, (a.total_rows + 7) / 8, 256, cat);
      return;
    }
    int groups = (R + 7) / 8;  // K_NRHS
    a.nrhs = R;
    if (colowner) {
      a.blocks = chunks;
      run(KID_gemvcm, &a, chunks * groups, cols / 8, cat);
    } else {
      a.blocks = (a.total_rows + 7) / 8;
      run(KID_gemvm, &a, a.blocks * groups, 256, cat);
    }
  }
  // the same with one block per row: for few rows, where the time of a single row counts
  void k_gemvw(GemvArgs& a, int cat, int R = 1) {
    a.total_rows = 0;
    for (int j = 0; j < a.njobs; ++j) a.total_rows += a.job[j].rows;
    if (any_packed(a)) {
      a.nrhs = R;
      bool same_cols = true;
      for (int j = 0; j < a.njobs; ++j) same_cols = same_cols && a.job[j].cols == a.job[0].cols;
      int cols = a.job[0].cols;
      if (pgemvw_rows > 0 && same_cols && cols % 256 == 0 && cols / 8 <= 1024) {
        k_pgemvc(a, pgemvw_rows, cat, R);  // column owner with few rows per block
        return;
      }
      a.blocks = a.total_rows;
      run(KID_pgemvw, &a, a.total_rows * R, 256, cat);
      return;
    }
    if (R == 1) {
      run(KID_gemvw, &a, a.total_rows, 256, cat);
      return;
    }
    a.nrhs = R;
    a.blocks = a.total_rows;
    run(KID_gemvwm, &a, a.total_rows * ((R + 7) / 8), 256, cat);
  }
  int quad_block() const { return ((c.hidden / 4) + 31) / 32 * 32; }  // H / 4 threads in whole warps
  // pgemvc with nr rows per block (2, 4, 8, 16); nr = 0: the largest of 16, 8, 4 that still gives
  // pgemvc_min_blocks blocks (few rows: more blocks, so that the whole matrix is in flight at once)
  void k_pgemvc(GemvArgs& a, int nr, int cat, int R) {
    auto nblocks = [&](int n) {
      int b = 0;
      for (int j = 0; j < a.njobs; ++j) b += (a.job[j].rows + n - 1) / n;
      return b;
    };
    bool explicit_nr = nr != 0;
    if (nr == 0) {
      nr = 16;
      while (nr > 4 && nblocks(nr) < pgemvc_min_blocks) nr >>= 1;
    }
    a.blocks = nblocks(nr);
    // The vocabulary projection is much larger than cache. Read its weights
    // together for the verifier's rows even when layer GEMVs run individually.
    const bool packed_head = (decode_tiles & 1) && a.njobs == 1 && a.job[0].w == unembed.w;
    if (R > 1 && !packed_head && f2_dense_multi(a, cat, R)) return;  // fast kernels, second set (decode_tiles 1 keeps its head)
    if (R > 1 && R <= 8 && ((R <= small_rows && use_small_dense) || packed_head)) {
      int nh = R <= 2 ? 2 : R <= 4 ? 4 : 8;
      int mr = nr;
      if (!explicit_nr && small_dense_rows > 0) mr = small_dense_rows;
      auto skid = [nh](int n) {
        if (nh == 2) return n == 2 ? KID_pgcm2h2 : n == 4 ? KID_pgcm4h2 : n == 8 ? KID_pgcm8h2 : KID_pgcm16h2;
        if (nh == 4) return n == 2 ? KID_pgcm2h4 : n == 4 ? KID_pgcm4h4 : n == 8 ? KID_pgcm8h4 : KID_pgcm16h4;
        return n == 2 ? KID_pgemvcm2 : n == 4 ? KID_pgemvcm4 : n == 8 ? KID_pgemvcm8 : KID_pgemvcm;
      };
      while (mr > 2 && be->max_block(skid(mr)) < a.job[0].cols / 8) mr >>= 1;  // fewer rows: fewer registers
      if (be->max_block(skid(mr)) >= a.job[0].cols / 8) {
        a.blocks = nblocks(mr);
        a.nrhs = R;
        run(skid(mr), &a, a.blocks * ((R + nh - 1) / nh), a.job[0].cols / 8, cat);
        return;
      }
    }
    if (R > 1 && use_pgemvcm && R >= pgemvcm_min_rows) {  // 8 rows of the pass per block (same sums as R single blocks, for any rows per block)
      int mr = pgemvcm_rows > 0 ? pgemvcm_rows : R <= 8 ? 2 : 16;
      auto mkid = [](int n) { return n == 2 ? KID_pgemvcm2 : n == 4 ? KID_pgemvcm4 : n == 8 ? KID_pgemvcm8 : KID_pgemvcm; };
      while (mr > 2 && be->max_block(mkid(mr)) < a.job[0].cols / 8) mr >>= 1;  // fewer rows: fewer registers
      if (be->max_block(mkid(mr)) >= a.job[0].cols / 8) {
        a.blocks = nblocks(mr);
        a.nrhs = R;
        run(mkid(mr), &a, a.blocks * ((R + 7) / 8), a.job[0].cols / 8, cat);  // K_NRHS
        return;
      }
    }
    int kid = nr == 2 ? KID_pgemvc2 : nr == 4 ? KID_pgemvc4 : nr == 8 ? KID_pgemvc8 : KID_pgemvc;
    run(kid, &a, a.blocks * R, a.job[0].cols / 8, cat);
  }
  static GemvJob job(const DMat& m, const float* x, float* y, int rows, int cols, int epi = INK2_EPI_NONE, float cap = 0) {
    GemvJob j{};
    j.w = (const u16*)m.w;
    j.sc = m.sc;
    j.fac = m.fac;
    j.fmt = m.fmt;
    j.x = x;
    j.split = cols;
    j.y = y;
    j.rows = rows;
    j.cols = cols;
    j.epi = epi;
    j.cap = cap;
    return j;
  }
  // ink2_ref.block_attention_residual; per output an RMS norm with weight
  // (normw) or an extra unweighted RMS-normalised copy (outn)
  static BarJob bar_job(const std::vector<const float*>& blocks, const float* partial, const float* q, int nq,
                        float* const* outs, const u16* const* normw, float* const* outn) {
    BarJob j{};
    j.ns = (int)blocks.size() + 1;
    for (size_t i = 0; i < blocks.size(); ++i) j.st[i] = blocks[i];
    j.st[blocks.size()] = partial;
    j.q = q;
    j.nq = nq;
    for (int i = 0; i < nq; ++i) {
      j.out[i] = outs[i];
      j.normw[i] = normw ? normw[i] : nullptr;
      j.outn[i] = outn ? outn[i] : nullptr;
    }
    return j;
  }
  void k_bar(const BarJob& j0, const BarJob* j1, int R) {
    BarArgs a{};
    a.job[0] = j0;
    if (j1) a.job[1] = *j1;
    a.st = st;
    a.H = c.hidden;
    a.K = c.conv_k;
    a.hscale = (float)pow((double)c.hidden, -0.5);
    a.eps = c.eps;
    a.cdiv = (float)sqrt((double)(c.conv_k + 1));
    a.nj = j1 ? 2 : 1;
    a.rows = R;
    a.hr = hr;
    if (f2_bar_comb(a, R)) return;  // fast kernels, second set
    if (fast_bar(a)) run_fx(INK2_FBAR, &a, a.nj * R, quad_block(), 0, INK2_PROF_MIX_NORM);  // fast kernels
    else run(KID_bar, &a, a.nj * R, quad_block(), INK2_PROF_MIX_NORM);
  }
  // ink2_ref.Ink2Model.message_gate for n gates (g0 = index of the first gate
  // of the layer), followed by rms_norm with the given weights.
  // hs_n / msgn: the unweighted RMS-normalised inputs (written by k_bar).
  void k_gates(const Layer& L, int g0, int n, const float* const* hs, const float* const* hs_n, const u16* const* normw,
               float* const* outs, int R) {
    const int H = c.hidden, GR = c.gate_rank;
    GemvArgs a{};
    for (int j = 0; j < n; ++j) {
      a.job[j] = job(L.g_down[g0 + j], hs_n[j], gd[g0 + j], GR, 2 * H);
      a.job[j].x2 = msgn[g0 + j];
      a.job[j].split = H;
    }
    a.njobs = n;
    k_gemvw(a, INK2_PROF_GATES, R);
    GemvArgs b{};
    for (int j = 0; j < n; ++j) {
      b.job[2 * j] = job(L.g_forget[g0 + j], gd[g0 + j], gf[g0 + j], H, GR, INK2_EPI_TANH);
      b.job[2 * j + 1] = job(L.g_gain[g0 + j], gd[g0 + j], gg[g0 + j], H, GR, INK2_EPI_TANH);
    }
    b.njobs = 2 * n;
    k_gemv(b, INK2_PROF_GATES, R);
    GmixArgs m{};
    for (int j = 0; j < n; ++j) {
      m.job[j].hs = hs[j];
      m.job[j].msg = msg[g0 + j];
      m.job[j].f = gf[g0 + j];
      m.job[j].g = gg[g0 + j];
      m.job[j].normw = normw[j];
      m.job[j].out = outs[j];
    }
    m.H = H;
    m.eps = c.eps;
    m.nj = n;
    m.rows = R;
    run(KID_gmix, &m, n * R, quad_block(), INK2_PROF_GATES);
  }

  // One layer visit (ink2_ref.Ink2Model.layer, attention and moe) for the R
  // rows of a pass. msg_job: the job that computes the four messages of this
  // visit (loop passes after the first), or NULL.
  const float* visit(int vi, const float* hidden_in, std::vector<const float*>& blocks, const BarJob* msg_job, int R) {
    Visit& V = visits[vi];
    const int li = V.layer;
    const Layer& L = layers[li];
    const int H = c.hidden, K = c.conv_k, A = L.nh * L.hd, C = L.nkv * L.hd;
    const bool boundary = li % c.bar_block == 0;
    const bool with_msg = msg_job != nullptr;

    // 1. attention inputs
    float* ins[3] = {qin, kin, vin};
    const u16* norms[3] = {L.norm_q, L.norm_k, L.norm_v};
    const float *q_in = qin, *k_in = qin, *v_in = qin;
    if (!L.sliding || with_msg) {
      k_in = kin;
      v_in = vin;
    }
    if (!with_msg) {
      if (li > 0) {
        k_bar(bar_job(blocks, hidden_in, L.attn_query, L.nq, ins, norms, nullptr), nullptr, R);
      } else {
        NormArgs n{};
        int nj = L.sliding ? 1 : 3;
        for (int j = 0; j < nj; ++j) {
          n.job[j].x = hidden_in;
          n.job[j].w = norms[j];
          n.job[j].out = ins[j];
        }
        n.st = st;
        n.n = H;
        n.eps = c.eps;
        n.nj = nj;
        n.rows = R;
        run(KID_norm, &n, nj * R, 256, INK2_PROF_MIX_NORM);
      }
    } else {
      if (li == 0) fail("a loop layer cannot be layer 0");
      k_bar(bar_job(blocks, hidden_in, L.attn_query, L.nq, xs, nullptr, xsn), msg_job, R);
      const float* hs[3] = {xs[0], xs[0], xs[0]};
      const float* hs_n[3] = {xsn[0], xsn[0], xsn[0]};
      if (!L.sliding) hs[1] = xs[1], hs[2] = xs[2], hs_n[1] = xsn[1], hs_n[2] = xsn[2];
      k_gates(L, 0, 3, hs, hs_n, norms, ins, R);
    }
    capture(vi, "q_in", q_in, H);
    capture(vi, "k_in", k_in, H);
    capture(vi, "v_in", v_in, H);

    // 2. projections of the attention inputs
    {
      GemvArgs a{};
      int n = 0;
      a.job[n++] = job(L.wq, q_in, qv, A, H);
      a.job[n++] = job(L.wr, q_in, rv, L.nh * c.d_rel, H);
      a.job[n++] = job(L.w_gate, q_in, gatev, A, H, INK2_EPI_SIGMOID);
      a.job[n++] = job(L.w_sink, q_in, sinkv, L.nh, H);
      a.job[n++] = job(L.wk, k_in, kraw, C, H);
      a.job[n++] = job(L.wv, v_in, vraw, C, H);
      if (!L.sliding) {
        a.job[n++] = job(L.k_wproj, k_in, kdyn, L.nkv * K, H);
        a.job[n++] = job(L.v_wproj, v_in, vdyn, L.nkv * K, H);
      }
      a.njobs = n;
      k_gemv(a, INK2_PROF_MATRIX, R);
    }
    capture(vi, "q_proj", qv, A);
    capture(vi, "rel_proj", rv, L.nh * c.d_rel);
    capture(vi, "out_gate", gatev, A);
    capture(vi, "sink", sinkv, L.nh);
    capture(vi, "k_proj", kraw, C);
    capture(vi, "v_proj", vraw, C);

    // 3. clipping, key / value convolution, cache write
    const int ring = L.sliding ? V.cap : 0;
    {
      PrepArgs a{};
      a.q = qv;
      a.q_scale = L.q_scale;
      a.r = rv;
      a.r_scale = L.rel_scale;
      a.kraw = kraw;
      a.vraw = vraw;
      a.khist = V.khist;
      a.vhist = V.vhist;
      a.kc = V.kc;
      a.vc = V.vc;
      a.kv = V.kv;
      a.k_w = L.k_w;
      a.v_w = L.v_w;
      a.k_dyn = kdyn;
      a.v_dyn = vdyn;
      a.st = st;
      a.nh = L.nh;
      a.nkv = L.nkv;
      a.hd = L.hd;
      a.drel = c.d_rel;
      a.K = K;
      a.sliding = L.sliding ? 1 : 0;
      a.ring = ring;
      a.qk_thr = c.qk_thr;
      a.rel_thr = c.rel_thr;
      a.eps = c.eps;
      a.conv_div = (float)sqrt((double)(L.sliding ? K + 1 : 2 * K + 1));
      int warps = 2 * L.nh + 2 * L.nkv;
      a.bpr = (warps + 7) / 8;
      a.rows = R;
      a.hr = hr;
      if (fast_prep(a)) run_fx(INK2_FPREP, &a, a.bpr * R, 256, 0, INK2_PROF_ATTENTION);  // fast kernels
      else run(KID_prep, &a, a.bpr * R, 256, INK2_PROF_ATTENTION);
    }
    if (vi == cap_visit) {
      const int slot = ring ? pos % ring : pos;
      capture(vi, "q", qv, A);
      capture(vi, "rel", rv, L.nh * c.d_rel);
      if (V.kv.mode) {  // per visit (MXFP8 keeps FP32 on the sliding visits)
        KvCaptureArgs ca{V.kv, st, capture_k, capture_v, ring, (int)C};
        run(KID_kv_capture, &ca, ((int)C + 511) / 512, 128, INK2_PROF_ATTENTION);
        capture(vi, "k_new", capture_k, C);
        capture(vi, "v_new", capture_v, C);
      } else {
        capture(vi, "k_new", V.kc + (size_t)slot * C, C);
        capture(vi, "v_new", V.vc + (size_t)slot * C, C);
      }
    }

    // 4. attention over the cache of this visit
    {
      AttnArgs a{};
      a.q = qv;
      a.r = rv;
      a.sink = sinkv;
      a.gate = gatev;
      a.kc = V.kc;
      a.vc = V.vc;
      a.rel_proj = L.rel_proj;
      a.rel_proj_t = rel_t ? L.rel_proj_t : nullptr;
      a.scratch = scratch;
      a.out = att;
      a.st = st;
      a.nh = L.nh;
      a.nkv = L.nkv;
      a.hd = L.hd;
      a.drel = c.d_rel;
      a.rel_extent = L.rel_extent;
      a.ring = ring;
      a.window = L.sliding ? c.window : 0;
      a.rows = R;
      a.scratch_stride = scratch_stride;
      a.scale = (float)pow((double)L.hd, -0.5);
      if (flash_hook(a, L, V, R)) {
      } else if (splitk_use(L, R)) {  // FP32 or MXFP8 (kv_read4 / kv_near)
        a.kv = V.kv;
        a.partial = splitk_partial;
        a.rel_t = splitk_rel ? L.rel_t : nullptr;
        a.splits = splitk_count(R, L.nkv);
        a.sk_chunk = splitk_chunk;
        run(KID_attn_gqa_splitk, &a, R * L.nkv * a.splits, 256, INK2_PROF_ATTENTION);
        if (a.splits > 1) run(KID_attn_splitk_reduce, &a, R * L.nh, 32, INK2_PROF_ATTENTION);
      } else if ((kv_mode == INK2_KV_FP32 || L.sliding) && !kv_split) {  // MXFP8: the sliding visits are FP32
        if (!f2_attn_run(a, R)) run(KID_attn, &a, L.nh * R, 1024, INK2_PROF_ATTENTION);  // fast kernels: split attention
      } else {
        a.kv = V.kv;
        a.row_tile = R == 1 ? 1 : R <= 4 ? 4 : 8;
        a.splits = L.sliding ? 1 : global_splits;
        a.partial = kv_partial;
        int bundles = (a.row_tile * (L.nh / L.nkv) + 23) / 24;
        int grid = ((R + a.row_tile - 1) / a.row_tile) * bundles * a.splits * L.nkv;
        if (R <= 8 && L.nh == 3 * L.nkv)
          run(KID_kv_stripes, &a, R * a.splits * L.nkv, 256, INK2_PROF_ATTENTION);
        else
          run(KID_kv_attention, &a, grid, 256, INK2_PROF_ATTENTION);
        if (a.splits > 1) run(KID_kv_merge, &a, R * L.nh, 128, INK2_PROF_ATTENTION);
      }
    }
    capture(vi, "attn", att, A);
    {
      GemvArgs a{};
      a.job[0] = job(L.wo, att, ao, H, A);
      a.njobs = 1;
      k_gemv(a, INK2_PROF_MATRIX, R);
    }
    capture(vi, "attn_out", ao, H);

    // 5. short convolution of the attention output (+ residual) and the input
    // of the MLP through the attention residual in one kernel, (message gate)
    if (boundary) blocks.push_back(hidden_in);
    {
      float* o1[1] = {with_msg ? xm : xmn};
      float* on1[1] = {xm_n};
      const u16* n1[1] = {L.mlp_norm};
      BarJob j = bar_job(blocks, resid, L.mlp_query, 1, o1, with_msg ? nullptr : n1, with_msg ? on1 : nullptr);
      j.cx = ao;
      j.cw = L.attn_sconv;
      j.chist = V.ahist;
      j.cres = boundary ? nullptr : hidden_in;
      j.cout = resid;
      k_bar(j, nullptr, R);
    }
    capture(vi, "residual", resid, H);
    if (with_msg) {
      const float* hs[1] = {xm};
      const float* hs_n[1] = {xm_n};
      const u16* n1[1] = {L.mlp_norm};
      float* outs[1] = {xmn};
      k_gates(L, 3, 1, hs, hs_n, n1, outs, R);
    }
    capture(vi, "mlp_in", xmn, H);

    // 6. router and experts; the last kernel also applies the short
    // convolution of the MLP output and adds the residual
    // the NVFP4 prompt planes and decode_tiles 2 keep their own expert kernels
    const bool use_fx = fast_experts(L) && !(R > logit_rows && nvfp4_planes) && !(R <= 8 && (decode_tiles & 2));
    {
      GemvArgs a{};
      a.job[0] = job(L.router_w, xmn, V.rlog, c.experts, H);
      a.njobs = 1;
      RouterArgs r{};
      r.logits = V.rlog;
      r.bias = L.router_bias;
      r.idx = V.ridx;
      r.wt = V.rwt;
      r.E = c.experts;
      r.k = c.top_k;
      r.route_scale = c.route_scale;
      r.rows = R;
      f2_layer = &L;  // fast kernels, second set
      if (!f2_router_topk(a, r, R, use_fx)) {  // fast kernels, second set: both in one launch
        k_gemvw(a, INK2_PROF_CONV_ROUTER, R);
        run(KID_topk, &r, R, (c.experts + 31) / 32 * 32, INK2_PROF_CONV_ROUTER);
      }
    }
    capture(vi, "router_logits", V.rlog, c.experts);
    capture(vi, "router_weights", V.rwt, c.top_k);
    if (use_fx) {  // fast kernels
      experts_fast(V, L, R);
    } else if (R == 1) {
      ExpArgs a{};
      a.w = L.w13;
      a.sc = L.w13s;
      a.factor = L.f13;
      a.idx = V.ridx;
      a.wt = V.rwt;
      a.x = xmn;
      a.y = act;
      a.e4m3 = e4m3;
      a.exp2t = exp2t;
      a.fmt = L.fmt13;
      a.rows = 2 * c.expert_inner;
      a.cols = H;
      a.k = c.top_k;
      a.gate_cap = c.gate_cap;
      a.up_cap = c.up_cap;
      a.hr = hr;
      if (use_exp13c && H / 16 <= 1024)  // column-owner form, K_EXPC_PAIRS = 8 row pairs per block
        run(KID_exp13c, &a, c.top_k * ((c.expert_inner + 7) / 8), (H / 16 + 31) / 32 * 32, INK2_PROF_EXPERTS);
      else
        run(KID_exp13, &a, (c.top_k * c.expert_inner + 7) / 8, 256, INK2_PROF_EXPERTS);
      capture(vi, "expert_act", act, (size_t)c.top_k * c.expert_inner);
      a.w = L.w2;
      a.sc = L.w2s;
      a.factor = L.f2;
      a.x = act;
      a.y = ey;
      a.fmt = L.fmt2;
      a.rows = H;
      a.cols = c.expert_inner;
      a.cw = L.mlp_sconv;
      a.chist = V.mhist;
      a.cres = resid;
      a.cout = V.hid;
      a.st = st;
      a.K = K;
      a.cdiv = (float)sqrt((double)(K + 1));
      run(KID_exp2, &a, H, 16 * c.top_k, INK2_PROF_EXPERTS);
    } else {
      experts_multi(V, L, R);
    }
    capture(vi, "expert_out", ey, H);
    capture(vi, "hidden", V.hid, H);
    return V.hid;
  }

  // The experts of an R-row pass: group the (row, slot) pairs by expert, then
  // every distinct expert is read once for all the rows that chose it.
  void experts_multi(Visit& V, const Layer& L, int R) {
    const int H = c.hidden, I = c.expert_inner, k = c.top_k;
    {
      GroupArgs g{};
      g.idx = V.ridx;
      g.gcount = gcount;
      g.gexp = gexp;
      g.goff = goff;
      g.gpairs = gpairs;
      g.R = R;
      g.k = k;
      g.E = c.experts;
      int nb = 32;  // a power of two >= E threads
      while (nb < c.experts) nb <<= 1;
      run(KID_egroup, &g, 1, nb, INK2_PROF_EXPERTS);
    }
    // Blocks for about 1.3 times the distinct experts expected (about k + 0.38 k per further row
    // on real text, at most R * k and E); with more, the blocks take several items each.
    int slots = R * k < c.experts ? R * k : c.experts;
    if (R <= small_rows && use_small_experts && use_exp13c && H / 16 <= 1024 && (I / 16 + 15) / 16 <= 5 /* K_EXP2S_U */ &&
        I % 16 == 0) {
      // few rows: one block per possible item (slots = the most distinct experts R rows can have)
      ExpMArgs s{};
      s.e4m3 = e4m3;
      s.exp2t = exp2t;
      s.gcount = gcount;
      s.gexp = gexp;
      s.goff = goff;
      s.gpairs = gpairs;
      s.expert_slots = slots;
      s.register_down = (decode_tiles & 2) != 0;
      s.k = k;
      s.gate_cap = c.gate_cap;
      s.up_cap = c.up_cap;
      s.w = L.w13;
      s.sc = L.w13s;
      s.factor = L.f13;
      s.fmt = L.fmt13;
      s.rows = 2 * I;
      s.cols = H;
      s.x = xmn;
      s.y = act;
      s.nchunk = (I + 7) / 8;  // K_EXPC_PAIRS row pairs per item, as exp13c
      s.grid = slots * s.nchunk;
      run(L.fmt13 == INK2_FMT_NVFP4 ? KID_exp13s : KID_exp13sx, &s, s.grid, (H / 16 + 31) / 32 * 32, INK2_PROF_EXPERTS);
      if (use_small_w2) {
        s.w = L.w2;
        s.sc = L.w2s;
        s.factor = L.f2;
        s.fmt = L.fmt2;
        s.rows = H;
        s.cols = I;
        s.x = act;
        s.y = ydots;
        s.nchunk = (H + 15) / 16;  // K_EXP2M_H output rows per item
        s.grid = slots * s.nchunk;
        run(L.fmt2 == INK2_FMT_NVFP4 ? KID_exp2s : KID_exp2sx, &s, s.grid, 256, INK2_PROF_EXPERTS);
      } else {
        s.w = L.w2;
        s.sc = L.w2s;
        s.factor = L.f2;
        s.fmt = L.fmt2;
        s.rows = H;
        s.cols = I;
        s.x = act;
        s.y = ydots;
        s.nchunk = (H + 15) / 16;  // K_EXP2M_H
        int ex = (int)ceil(1.3 * (k + 0.38 * k * (R - 1)));
        s.grid = (ex < slots ? ex : slots) * s.nchunk;
        run(KID_exp2m, &s, s.grid, 256, INK2_PROF_EXPERTS);
      }
      experts_comb(V, L, R);
      return;
    }
    int expect = (int)ceil(1.3 * (k + 0.38 * k * (R - 1)));
    if (expect < slots) slots = expect;
    ExpMArgs a{};
    a.e4m3 = e4m3;
    a.exp2t = exp2t;
    a.gcount = gcount;
    a.gexp = gexp;
    a.goff = goff;
    a.gpairs = gpairs;
    a.k = k;
    a.gate_cap = c.gate_cap;
    a.up_cap = c.up_cap;
    // w13 -> activations [R * k, I]
    // Passes larger than the verification limit contain prompt tokens only.
    if (R > logit_rows) {
      a.nvfp4_planes = nvfp4_planes;
      a.nvfp4_scratch = nvfp4_scratch;
    }
    a.input_rows = R;
    a.expert_slots = (std::min)(R * k, c.experts);
    a.register_down = R <= 8 && (decode_tiles & 2);
    a.w = L.w13;
    a.sc = L.w13s;
    a.factor = L.f13;
    a.fmt = L.fmt13;
    a.rows = 2 * I;
    a.cols = H;
    a.x = xmn;
    a.y = act;
    a.nchunk = (I + 15) / 16;  // K_E13_PAIRS
    a.grid = slots * a.nchunk;  // one block per possible item (blocks past the distinct experts end at once)
    run(KID_exp13cm, &a, a.grid, 256, INK2_PROF_EXPERTS);
    // w2 -> products [R * k, H]
    a.input_rows = R * k;
    a.w = L.w2;
    a.sc = L.w2s;
    a.factor = L.f2;
    a.fmt = L.fmt2;
    a.rows = H;
    a.cols = I;
    a.x = act;
    a.y = ydots;
    a.nchunk = (H + 15) / 16;  // K_EXP2M_H
    a.grid = slots * a.nchunk;
    run(KID_exp2m, &a, a.grid, 256, INK2_PROF_EXPERTS);
    experts_comb(V, L, R);
  }
  // ecomb: the k products of every row in slot order, convolution, residual
  void experts_comb(Visit& V, const Layer& L, int R) {
    const int H = c.hidden, k = c.top_k;
    ECombArgs e{};
    e.dots = ydots;
    e.wt = V.rwt;
    e.y = ey;
    e.cw = L.mlp_sconv;
    e.chist = V.mhist;
    e.cres = resid;
    e.cout = V.hid;
    e.st = st;
    e.R = R;
    e.k = k;
    e.H = H;
    e.K = c.conv_k;
    e.hr = hr;
    e.cdiv = (float)sqrt((double)(c.conv_k + 1));
    run(KID_ecomb, &e, (R * H + 255) / 256, 256, INK2_PROF_EXPERTS);
  }

  // All kernels of one pass of R rows (ink2_ref.Ink2Model.forward). Nothing
  // here may depend on the tokens or the positions: those are read from `st`
  // by the kernels, so that the same sequence can be captured once and replayed.
  void forward(bool with_logits, int R) {
    const int H = c.hidden;
    {
      NormArgs n{};
      n.job[0].x = embed.w;
      n.esc = embed.sc;
      n.efmt = embed.fmt;
      n.job[0].x_bf16 = 1;
      n.job[0].w = embed_norm;
      n.job[0].out = emb;
      n.st = st;
      n.embed_stride = H;
      n.n = H;
      n.eps = c.eps;
      n.nj = 1;
      n.rows = R;
      n.img = img_feat;  // image positions (st[row].pad > 0) copy their feature row; NULL without the image tower
      run(KID_norm, &n, R, 256, INK2_PROF_MIX_NORM);
    }
    const float* hidden = emb;
    std::vector<const float*> blocks;
    int vi = 0;
    for (int i = 0; i < c.loop_start; ++i, ++vi) hidden = visit(vi, hidden, blocks, nullptr, R);
    const float* pre_hidden = hidden;
    std::vector<const float*> pre_blocks = blocks;
    size_t npre = pre_blocks.size();
    for (int i = c.loop_start; i < c.loop_end; ++i, ++vi) hidden = visit(vi, hidden, blocks, nullptr, R);
    for (int l = 1; l < c.loop_passes; ++l) {
      // +1 skips the block appended by the first loop layer (it equals the pre-loop hidden state)
      std::vector<const float*> prev_blocks(blocks.begin() + npre + 1, blocks.end());
      const float* prev_hidden = hidden;
      hidden = pre_hidden;
      blocks = pre_blocks;
      for (int i = c.loop_start; i < c.loop_end; ++i, ++vi) {
        BarJob mj = bar_job(prev_blocks, prev_hidden, msg_q + (size_t)(i - c.loop_start) * 4 * H, 4, msg, nullptr, msgn);
        hidden = visit(vi, hidden, blocks, &mj, R);
      }
    }
    for (int i = c.loop_end; i < c.layers; ++i, ++vi) hidden = visit(vi, hidden, blocks, nullptr, R);
    if (!with_logits && !drf) {  // prompt tokens whose prediction nobody reads (drafter hook: it reads fh)
      f2_flush();  // fast kernels, second set (a deferred fxcomb of the last visit)
      return;
    }

    float* o1[1] = {fh};
    const u16* n1[1] = {final_norm};
    k_bar(bar_job(blocks, hidden, output_query, 1, o1, n1, nullptr), nullptr, R);
    if (drf) drf_commit(R);  // drafter hook
    if (!with_logits) return;
    GemvArgs a{};
    a.job[0] = job(unembed, fh, logits, c.vocab, H, INK2_EPI_SOFTCAP, c.logit_cap);
    a.njobs = 1;
    k_gemv(a, INK2_PROF_OUTPUT, R);
    if (R > 1) {
      ArgmaxArgs m{};
      m.logits = logits;
      m.idx = amax_idx;
      m.val = amax_val;
      m.n = c.vocab;
      run(KID_argmax, &m, R, 1024, INK2_PROF_OUTPUT);
    }
  }

  StepState state_at(int token, int p) const {
    // log_scale: tau = 1 + alpha * log(max((position + 1) / n_floor, 1))
    float ratio = (float)(p + 1) / (float)c.n_floor;
    if (ratio < 1.0f) ratio = 1.0f;
    return StepState{token, p, 1.0f + c.alpha * logf(ratio), 0};
  }

  // img_row >= 0: an image position that takes feature row img_row (checked by the caller).
  void step(int token, float* logits_out, bool with_logits, int* top1 = nullptr, int img_row = -1) {
    if (!finalized) fail("step before finalize");
    if (token < 0 || token >= c.embed_rows) fail("token id out of range");
    if (pos >= c.max_positions) fail(fmt("context full: %d positions", c.max_positions));
    caps.clear();
    StepState s = state_at(token, pos);
    s.pad = img_row + 1;  // 0: text
    be->upload(st, &s, sizeof s);
    if (prof) {
      be->sync();
      prof_mark = now_s();
    }
    double t0 = now_s();
    f2_attn_lv = kv_mode == INK2_KV_FP32 && !kv_split && f2_attn_use(pos, 1);  // fast kernels: split attention
    bool direct = !(use_graph && be->graph_supported()) || prof || cap_visit >= 0 || bench_reps > 0;
    if (direct) {
      forward(with_logits, 1);
    } else {
      void*& g = graph[(f2_attn_lv ? 2 : 0) + (with_logits ? 1 : 0)];
      if (!g) {
        be->graph_begin();
        forward(with_logits, 1);
        g = be->graph_end();
        check_backend();
        ++graphs_built;
      }
      be->graph_launch(g);
    }
    double t1 = now_s();
    if (top1) {
      int tiles = (c.vocab + ink2_pick_tile - 1) / ink2_pick_tile;
      PickArgs filter{logits, nullptr, pick_values, pick_indices, c.vocab};
      run(KID_pick_tile, &filter, tiles, 256, INK2_PROF_OUTPUT);
      PickArgs pick{pick_values, pick_indices, amax_val, amax_idx, tiles};
      run(KID_pick_tile, &pick, 1, 256, INK2_PROF_OUTPUT);
      be->download(top1, amax_idx, sizeof(*top1));
    } else if (with_logits) {
      be->download(logits_out ? logits_out : logits_host.data(), logits, (size_t)c.vocab * 4);
    }
    double t2 = now_s();
    t_enqueue += t1 - t0;
    t_wait += t2 - t1;
    ++t_steps;
    if (prof) {
      prof_t[INK2_PROF_COPY] += t2 - prof_mark;
      prof_mark = t2;
    }
    check_backend();
    ++pos;
  }

  // One pass over R consecutive positions (R >= 2). img (may be NULL): per row -1 for text, else the
  // feature row of an image position (checked by the caller).
  void pass(const int* ids, int R, bool with_logits, float* logits_out, int* argmax_out, float* max_out,
            const int* img = nullptr) {
    if (!finalized) fail("pass before finalize");
    if (R < 2 || R > max_rows) fail(fmt("rows of a pass must be 2 .. %d", max_rows));
    if (with_logits && R > logit_rows) fail(fmt("at most %d rows with logits", logit_rows));
    if (pos + R > c.max_positions) fail(fmt("context full: %d positions", c.max_positions));
    for (int r = 0; r < R; ++r) {
      if (ids[r] < 0 || ids[r] >= c.embed_rows) fail("token id out of range");
      st_host[r] = state_at(ids[r], pos + r);
      st_host[r].pad = img ? img[r] + 1 : 0;  // every row is written: 0 for text
    }
    caps.clear();
    be->upload(st, st_host.data(), (size_t)R * sizeof(StepState));
    const bool direct = !(use_graph && be->graph_supported()) || prof || bench_reps > 0;
    const int joint_key = drf ? drf_before_pass(R, joined_draft_graph && !direct && with_logits) : 0;
    if (prof) {
      be->sync();
      prof_mark = now_s();
    }
    double t0 = now_s();
    f2_attn_lv = kv_mode == INK2_KV_FP32 && !kv_split && f2_attn_use(pos, R);  // fast kernels: split attention
    cur_logits = with_logits;  // read by exact_logit_pass while the pass is recorded / run
    struct ClearLogits { bool& f; ~ClearLogits() { f = false; } } clear_logits{cur_logits};
    if (direct) {
      forward(with_logits, R);
    } else {
      const int key = 4 * R + (f2_attn_lv ? 2 : 0) + int(with_logits);  // + 2: split attention of the fast kernels
      void*& g = mgraph[std::make_tuple(key, false, joint_key)];
      if (!g) {
        be->graph_begin();
        if (joint_key) drf_record_pending(R - 1);
        forward(with_logits, R);
        g = be->graph_end();
        check_backend();
        ++graphs_built;
      }
      be->graph_launch(g);
    }
    double t1 = now_s();
    if (with_logits) {
      if (argmax_out) be->download(argmax_out, amax_idx, (size_t)R * 4);
      if (max_out) be->download(max_out, amax_val, (size_t)R * 4);
      if (logits_out) be->download(logits_out, logits, (size_t)R * c.vocab * 4);
    }
    double t2 = now_s();
    t_enqueue += t1 - t0;
    t_wait += t2 - t1;
    t_steps += R;
    if (prof) {
      prof_t[INK2_PROF_COPY] += t2 - prof_mark;
      prof_mark = t2;
    }
    check_backend();
    pos += R;
  }

  // Prompt tokens: passes of `chunk` rows; the rest in passes of falling
  // powers of two (so that only a few pass sizes are ever recorded).
  // img (may be NULL): the feature rows of image positions, -1 for text (checked by the caller).
  void feed_many(const int* ids, int n, const int* img = nullptr) {
    if (n < 0) fail("negative count");
    if (pos + n > c.max_positions) fail(fmt("context full: %d positions", c.max_positions));
    int i = 0;
    while (i < n) {
      int left = n - i, R = chunk;
      if (left < R) {
        R = 1;
        while (R * 2 <= left) R *= 2;
      }
      if (R <= 1) step(ids[i], nullptr, false, nullptr, img ? img[i] : -1);
      else pass(ids + i, R, false, nullptr, nullptr, nullptr, img ? img + i : nullptr);
      i += R;
    }
  }
};

Backend* make_backend(const char* name) {
  std::string n = name ? name : "host";
  if (n == "host") return ink2_make_host_backend();
  if (n == "cuda") {
#ifdef INK2_WITH_CUDA
    char err[300] = "";
    Backend* b = ink2_make_cuda_backend(err, sizeof err);
    if (!b) fail(err);
    return b;
#else
    fail("this build has no cuda backend");
#endif
  }
  fail("unknown backend " + n);
}

}  // namespace

#define API_BEGIN(h)        \
  Engine* e = (Engine*)(h); \
  try {
#define API_END                              \
  }                                          \
  catch (const std::exception& ex) {         \
    (e ? e->err : g_last_error) = ex.what(); \
    return -1;                               \
  }

INK2_API const char* ink2_build_info(void) {
#ifdef INK2_WITH_CUDA
  return "ink2 engine, backends: host cuda";
#else
  return "ink2 engine, backends: host";
#endif
}

INK2_API int ink2_abi_version(void) { return INK2_ABI_VERSION; }

INK2_API int ink2_capabilities(void) {
  return INK2_CAP_PACKED_DENSE | INK2_CAP_DRAFTER | INK2_CAP_DRAFTER_PACKED | INK2_CAP_Q3 | INK2_CAP_KV
#ifdef INK2_WITH_VISION
         | INK2_CAP_VISION
#endif
      ;
}

// After ink2_finalize: 0 = the general kernels, else the compute capability (86, 89, 120, ...) the fast kernels run on
// (ink2_fast.h; option "fast").
INK2_API int ink2_fast_path(void* h) {
  Engine* e = (Engine*)h;
  return e && e->finalized ? e->fast : 0;
}

#ifdef INK2_WITH_CUDA
int ink2_cuda_nvfp4_prompt();  // ink2_cuda.cu
#endif
// 1 when this library has the NVFP4 tensor-core prompt path (option nvfp4_prefill) and the current CUDA device can run it.
INK2_API int ink2_nvfp4_prompt(void) {
#ifdef INK2_WITH_CUDA
  return ink2_cuda_nvfp4_prompt();
#else
  return 0;
#endif
}

INK2_API int ink2_packed_dense(void* h) {
  Engine* e = (Engine*)h;
  return e && e->finalized ? e->packed_dense : 0;
}

// Values of the config that a division, a loop or a kernel launch depends on. Checked in ink2_create, so a config.json
// the engine cannot take gives a clear error instead of a crash (finalize checks the tensor sizes later).
static void check_config(const Ink2Config& c) {
  auto need = [](bool ok, const char* what) {
    if (!ok) fail(std::string("config: ") + what);
  };
  need(c.layers >= 1 && c.layers <= 64, "the number of layers must be 1 .. 64");
  need(c.hidden >= 32 && c.hidden % 32 == 0, "hidden_size must be a positive multiple of 32");
  need(c.vocab >= 1 && c.embed_rows >= c.vocab, "the vocabulary must be at least 1 and not larger than the embedding table");
  bool any_full = false, any_sw = false;
  for (int i = 0; i < c.layers; ++i) (c.sliding[i] ? any_sw : any_full) = true;
  if (any_full) {
    need(c.nh_full >= 1 && c.nkv_full >= 1, "the head counts of the full-attention layers must be at least 1");
    need(c.nh_full % c.nkv_full == 0, "the heads of the full-attention layers must be a multiple of their key/value heads");
    need(c.hd_full == 128, "the head size of the full-attention layers must be 128");
    need(c.rel_extent >= 1, "rel_extent must be at least 1");
  }
  if (any_sw) {
    need(c.nh_sw >= 1 && c.nkv_sw >= 1, "the head counts of the sliding-window layers must be at least 1");
    need(c.nh_sw % c.nkv_sw == 0, "the heads of the sliding-window layers must be a multiple of their key/value heads");
    need(c.hd_sw == 128, "the head size of the sliding-window layers must be 128");
    need(c.window >= 1 && c.window <= 65536, "sliding_window_size must be 1 .. 65536");
  }
  need(c.d_rel >= 1 && c.d_rel <= 32, "d_rel must be 1 .. 32");
  need(c.conv_k >= 2 && c.conv_k <= 8, "conv_kernel_size must be 2 .. 8");
  need(c.expert_inner >= 32 && c.expert_inner % 32 == 0, "moe_intermediate_size must be a positive multiple of 32");
  need(c.experts >= 1 && c.experts <= 1024, "n_routed_experts must be 1 .. 1024");
  need(c.top_k >= 2 && c.top_k % 2 == 0 && c.top_k <= 32 && c.top_k <= c.experts,
       "num_experts_per_tok must be even, 2 .. 32 and not more than the experts");
  need(c.bar_block >= 1, "the layers per attention-residual block must be at least 1");
  need(c.loop_passes >= 1, "the loop passes must be at least 1");
  need(c.loop_start >= 1 && c.loop_end <= c.layers && c.loop_start < c.loop_end, "the loop range must lie inside the layers");
  need(c.gate_rank >= 8 && c.gate_rank % 8 == 0, "the gate rank must be a positive multiple of 8");
  need(c.n_floor >= 1, "the position floor of the log scaling must be at least 1");
  need(c.max_positions >= 1, "max_positions must be positive");
}

INK2_API void* ink2_create(const Ink2Config* cfg, const char* backend) {
  Engine* e = nullptr;
  try {
    if (!cfg) fail("no config");
    check_config(*cfg);
    e = new Engine();
    e->c = *cfg;
    e->be = make_backend(backend);
    return e;
  } catch (const std::exception& ex) {
    g_last_error = ex.what();
    delete e;
    return nullptr;
  }
}

INK2_API void ink2_destroy(void* h) { delete (Engine*)h; }

INK2_API const char* ink2_last_error(void* h) { return h ? ((Engine*)h)->err.c_str() : g_last_error.c_str(); }

INK2_API int ink2_set_tensor(void* h, const char* name, const void* data, long long nbytes) {
  API_BEGIN(h)
  if (e->finalized) fail("set_tensor after finalize");
  if (nbytes <= 0) fail(std::string("empty tensor ") + name);
  Ten& t = e->ten[name];
  if (t.p) fail(std::string("tensor set twice: ") + name);
  t.p = e->be->alloc((size_t)nbytes);
  if (!t.p) fail(e->be->error() ? e->be->error() : "allocation failed");
  t.nbytes = nbytes;
  e->be->upload(t.p, data, (size_t)nbytes);
  if (nbytes <= 65536) t.host.assign((const unsigned char*)data, (const unsigned char*)data + nbytes);
  e->check_backend();
  return 0;
  API_END
}

INK2_API int ink2_finalize(void* h) {
  API_BEGIN(h)
  e->finalize();
  return 0;
  API_END
}

INK2_API int ink2_reset(void* h) {
  API_BEGIN(h)
  e->reset();
  std::fill(e->img_valid.begin(), e->img_valid.end(), (unsigned char)0);  // image rows must be written again
  return 0;
  API_END
}

INK2_API int ink2_step(void* h, int token, float* logits_out) {
  API_BEGIN(h)
  e->step(token, logits_out, true);
  return 0;
  API_END
}
INK2_API int ink2_step_top1(void* h, int token, int* winner) {
  API_BEGIN(h)
  if (!winner || !e->device_greedy) fail("step_top1 requires device_greedy and a destination");
  e->step(token, nullptr, true, winner);
  return 0;
  API_END
}

INK2_API int ink2_feed(void* h, int token) {
  API_BEGIN(h)
  e->step(token, nullptr, false);
  return 0;
  API_END
}

INK2_API int ink2_feed_many(void* h, const int* ids, int n) {
  API_BEGIN(h)
  e->feed_many(ids, n);
  return 0;
  API_END
}

INK2_API int ink2_step_many(void* h, const int* ids, int n, float* logits_out, int* argmax_out, float* max_out) {
  API_BEGIN(h)
  if (n == 1) {  // one row: the single step (bit for bit the same results)
    float* lo = logits_out ? logits_out : e->logits_host.data();
    e->step(ids[0], lo, true);
    int best = 0;
    for (int i = 1; i < e->c.vocab; ++i)
      if (lo[i] > lo[best]) best = i;
    if (argmax_out) argmax_out[0] = best;
    if (max_out) max_out[0] = lo[best];
  } else {
    e->pass(ids, n, true, logits_out, argmax_out, max_out);
  }
  return 0;
  API_END
}

INK2_API int ink2_truncate(void* h, int n_positions) {
  API_BEGIN(h)
  e->truncate(n_positions);
  return 0;
  API_END
}

INK2_API int ink2_position(void* h) { return ((Engine*)h)->pos; }

INK2_API int ink2_limits(void* h, int* out4) {
  Engine* e = (Engine*)h;
  out4[0] = e->max_rows;
  out4[1] = e->finalized ? e->logit_rows : (e->max_rows < 32 ? e->max_rows : 32);
  out4[2] = e->finalized ? e->margin : (e->max_rows > e->rollback ? e->max_rows : e->rollback);
  out4[3] = e->c.max_positions;
  return 0;
}

INK2_API int ink2_set_option(void* h, const char* name, int value) {
  API_BEGIN(h)
  std::string n = name;
  if (n == "graph") e->use_graph = value != 0;
  else if (n == "kv_split") {
    if (value < 1 || value > 32) fail("kv_split must be 1 .. 32");
    if (e->finalized) {
      size_t start = e->be->bytes_allocated();
      e->reserve_kv_partials(value);
      e->kv_work_bytes += e->be->bytes_allocated() - start;
      e->global_splits = value;
    }
    e->kv_split = value;
  }
  else if (n == "pass_exact") {
    if (value != 0 && value != 1) fail("pass_exact must be 0 or 1");
    e->pass_exact = value;
    if (e->finalized) e->free_graphs();
  }
  else if (n == "splitk_max_rows") {
    if (value < 1 || value > 8) fail("splitk_max_rows must be 1 .. 8");
    e->splitk_max_rows = value;
    if (e->finalized) e->free_graphs();
  }
  else if (n == "splitk_attn" || n == "splitk_splits" || n == "splitk_chunk" || n == "splitk_rel") {
    if (n == "splitk_attn" && (value == 0 || value == 1)) e->splitk_attn = value;
    else if (n == "splitk_splits" && value >= 0 && value <= 32) e->splitk_splits = value;
    else if (n == "splitk_chunk" && value >= 8 && value <= 65536) e->splitk_chunk = value;
    else if (n == "splitk_rel" && (value == 0 || value == 1)) e->splitk_rel = value != 0;
    else fail("option " + n + " out of range");
    if (e->splitk_attn && e->finalized) e->splitk_prepare();
    if (e->finalized) e->free_graphs();
  }
  else if (n == "kv_mxfp8" || n == "kv_hot") {
    if (e->finalized) fail("option " + n + " must be set before finalize");
    if (n == "kv_mxfp8" && (value == 0 || value == 1)) e->kv_mode = value ? INK2_KV_MXFP8 : INK2_KV_FP32;
    else if (n == "kv_hot" && (value == 0 || (value >= 16 && value <= 256 && !(value & (value - 1))))) e->kv_hot = value;
    else fail("option " + n + " out of range");
  }
  else if (n == "rel_t" && (value == 0 || value == 1)) {
    if (value && e->finalized) e->upload_rel_t();
    e->rel_t = value != 0;
  }
  else if (n == "gemvc") e->use_gemvc = value != 0;
  else if (n == "pgemvc_rows" && (value == 0 || value == 2 || value == 4 || value == 8 || value == 16)) e->pgemvc_rows = value;
  else if (n == "pgemvw_rows" && (value == 0 || value == 2 || value == 4 || value == 8 || value == 16)) e->pgemvw_rows = value;
  else if (n == "pgemvc_min_blocks" && value >= 1) e->pgemvc_min_blocks = value;
  else if (n == "pgemvn") e->use_pgemvn = value != 0;
  else if (n == "pgemvcm") e->use_pgemvcm = value != 0;
  else if (n == "pgemvcm_rows" && (value == 0 || value == 2 || value == 4 || value == 8 || value == 16)) e->pgemvcm_rows = value;
  else if (n == "pgemvcm_min_rows" && value >= 2) e->pgemvcm_min_rows = value;
  else if (n == "exp13c") e->use_exp13c = value != 0;
  else if (n == "small_rows" && value >= 0 && value <= 8) e->small_rows = value;
  else if (n == "small_experts") e->use_small_experts = value != 0;
  else if (n == "small_dense") e->use_small_dense = value != 0;
  else if (n == "joined_draft_graph" && (value == 0 || value == 1)) e->joined_draft_graph = value != 0;
  else if (n == "decode_tiles" && value >= 0 && value <= 3) {
    e->decode_tiles = value;
    e->drf_free_graphs();
  }
  else if (n == "draft_l2_mib") {
    if (value < 0 || value > 64 || !e->be->set_draft_l2(value))
      fail("draft_l2_mib requires 0 .. 64 MiB and a reserved CUDA persisting L2 region when enabled");
    e->drf_free_graphs();
  }
  else if (n == "packed_prefill") {
    if ((value != 0 && value != 1) || !e->be->set_packed_prompt(value != 0))
      fail("packed_prefill requires 0 or 1 and a CUDA backend when enabled");
  }
  else if (n == "flash_min_rows") {
    if (value < 8 || value > 1 << 20) fail("flash_min_rows must be 8 or more");
    e->flash_min_rows = value;
    if (e->finalized) e->free_graphs();
  }
  else if (n == "flash_attn") {
    if (value < 0 || value > 3) fail("flash_attn must be 0 .. 3");
    if (!e->finalized) fail("set flash_attn after finalize");
    if (value >= 2 && std::string(e->be->name()) != "cuda") fail("flash_attn 2 and 3 need the CUDA backend");
    if (value && !e->flash_rel) {
      int nh = (std::max)(e->c.nh_full, e->c.nh_sw);
      e->flash_rel_cap = (std::max)(e->c.window, e->c.rel_extent);
      e->flash_rel = e->alloc<float>((size_t)e->max_rows * nh * e->flash_rel_cap);
      e->flash_partial = e->alloc<float>((size_t)e->max_rows * nh * Engine::flash_max_splits * 130);
      e->flash_sms = e->be->sm_count();
    }
    e->flash_attn = value;
  }
  else if (n == "device_greedy" && (value == 0 || value == 1)) {
    if (!e->finalized) fail("set device_greedy after finalize");
    if (value) {
      if (e->c.vocab >= 0x00ffffff) fail("device_greedy needs vocabulary indices below 2^24 - 1");
      int tiles = (e->c.vocab + ink2_pick_tile - 1) / ink2_pick_tile;
      if (!e->pick_indices) e->pick_indices = e->alloc<int>(tiles);
      if (!e->pick_values) e->pick_values = e->alloc<float>(tiles);
    }
    e->device_greedy = value != 0;
    return 0;  // selects only the read-back after the existing graph
  }
  else if (n == "nvfp4_prefill") {
    if (value < 0 || value > 4) fail("nvfp4_prefill must be 0 .. 4");
    if (!e->finalized || e->pos != 0) fail("set nvfp4_prefill on an empty finalized engine");
    if (value && !e->be->supports_nvfp4_prefill()) fail("nvfp4_prefill requires the sm_120a CUDA build and device");
    if (value && !e->nvfp4_scratch) {
      size_t width = (std::max)((size_t)e->c.hidden, (size_t)e->c.top_k * e->c.expert_inner);
      // Per plane: one nibble per element and one scale byte per 16.
      e->nvfp4_scratch = e->alloc<u8>(4 * (width / 16) * 9 * e->max_rows);
    }
    e->nvfp4_planes = value;
  }
  else if (n == "small_w2") e->use_small_w2 = value != 0;
  else if (n == "small_dense_rows" && (value == 0 || value == 2 || value == 4 || value == 8 || value == 16)) e->small_dense_rows = value;
  else if (n == "fast" && value >= -1 && value <= 1) e->fast_opt = value, e->fast_resolve();  // fast kernels
  else if (n == "fx13_p" && (value == 4 || value == 8)) e->fx13_p = value;
  else if (n == "fx2_e" && (value == 4 || value == 8)) e->fx2_e = value;
  else if (n == "fx2_hb" && value >= 1 && value <= 8) e->fx2_hb = value, e->fast_resolve();
  else if (n == "fast_small" && value >= 0 && value <= 3) e->fsmall = value;
  else if (n == "fx_lut") e->fx_lut = value != 0;
  else if (n == "pdl") e->pdl_opt = value != 0, e->fast_resolve();
  else if (n == "fx2_mode" && (value == 0 || value == 1)) e->fx2_mode = value;
  else if (n == "fx2c_p" && (value == 4 || value == 8)) e->fx2c_p = value;
  else if (n == "f2_pf" && value >= 0 && value <= 3) e->f2_pf = value;  // fast kernels, second set
  else if (n == "f2_extra" && value >= 0 && value <= 7) e->f2_extra = value;
  else if (n == "f2_budget" && value >= 0 && value <= 1024) e->f2_budget = value;
  else if (n == "f2_trig") e->f2_trig = value != 0;
  else if (n == "f2_gemvm") e->f2_gemvm = value != 0;
  else if (n == "f2_router") e->f2_router = value != 0;
  else if (n == "f2_nb" && (value == 0 || value == 8 || value == 16 || value == 24 || value == 32)) e->f2_nb = value;
  else if (n == "f2_min_blocks" && value >= 1) e->f2_min_blocks = value;
  else if (n == "f2_wv" && (value == 4 || value == 8)) e->f2_wv = value;
  else if (n == "f2_max_rows" && value >= 2) e->f2_max_rows = value;
  else if (n == "f2_min_rows" && value >= 2) e->f2_min_rows = value;
  else if (n == "f2_fuse") e->f2_fuse = value != 0;
  else if (n == "f2_ccomb") e->f2_ccomb = value != 0;
  else if (n == "f2_spec") e->f2_spec = value != 0;
  else if (n == "f2_npf" && value >= 0 && value <= 1024) e->f2_npf = value;
  else if (n == "f2_pfel") e->f2_pfel = value != 0;
  else if (n == "f2_pipe" && value >= 0 && value <= 3) e->f2_pipe = value;
  else if (n == "f2_pwaves" && value >= 1 && value <= 8) e->f2_pwaves = value;
  else if (n == "f2_reg" && value >= 0 && value <= 31) e->f2_reg = value;  // bit 16: attn key / value prefetch
  else if (n == "f2_prepb" && (value == 1 || value == 2)) e->f2_prepb = value;
  else if (n == "f2_router_rows" && value >= 1 && value <= 16) e->f2_router_rows = value;
  else if (n == "f2_attn") e->f2_attn = value != 0;  // split attention of single steps
  else if (n == "f2_attn_from" && value >= 0) e->f2_attn_from = value;
  else if (n == "f2_attn_rows" && value >= 1 && value <= 16) e->f2_attn_rows = value;
  else if (n == "f2_attn_sa" && value >= 1 && value <= 32) e->f2_attn_sa = value;
  else if (n == "f2_attn_sw" && (value == 1 || value == 2 || value == 4 || value == 8)) e->f2_attn_sw = value;
  else if (n == "f2_attn_pf" && value >= 0 && value <= 7) e->f2_attn_pf = value;
  else if (n == "f2_attn_ng" && (value == 1 || value == 2)) e->f2_attn_ng = value;
  else if (n == "f2_gm2") e->f2_gm2 = value != 0;  // dense products of passes
  else if (n == "f2_gm2_min" && value >= 2 && value <= 128) e->f2_gm2_min = value;
  else if (n == "f2_gm2_max" && value >= 2 && value <= 128) e->f2_gm2_max = value;
  else if (n == "f2_gm2_wv" && (value == 4 || value == 8)) e->f2_gm2_wv = value;
  else if (n == "f2_gm2_mb" && value >= 1) e->f2_gm2_mb = value;
  else if (n == "f2_bfm") e->f2_bfm = value != 0;
  else if (n == "f2_gm2_nh" && (value == 4 || value == 8)) e->f2_gm2_nh = value;
  else if (n == "f2_gm2_occ" && value >= 0 && value <= 7) e->f2_gm2_occ = value;
  else if (n == "f2_gm2_nh8" && value >= 5 && value <= 129) e->f2_gm2_nh8 = value;
  else if (n == "f2_gm2_pg" && value >= 0 && value <= 8) e->f2_gm2_pg = value;
  else if (n == "f2_gm2_mid") e->f2_gm2_mid = value != 0;
  else if (n == "f2_gm2_pgmin" && value >= 2 && value <= 129) e->f2_gm2_pgmin = value;
  else if (n == "f2_gm2_pnb" && (value == 8 || value == 16 || value == 24 || value == 32)) e->f2_gm2_pnb = value;
  else if (n == "fx_g" && (value == 1 || value == 2 || value == 4)) e->fx_g = value;
  else if (n == "fx_gp" && (value == 1 || value == 2 || value == 4)) e->fx_gp = value;
  else if (n == "fx_gp_rows" && value >= 2 && value <= 2048) e->fx_gp_rows = value;
  else if (n == "fx_peel") e->fx_peel = value != 0;
  else if (n == "max_rows" || n == "rollback") {
    if (e->finalized) fail("option " + n + " must be set before finalize");
    if (value < (n == "max_rows" ? 1 : 0)) fail("option " + n + " out of range");
    (n == "max_rows" ? e->max_rows : e->rollback) = value;
  } else if (n == "chunk") {
    if (value < 1 || (e->finalized && value > e->max_rows)) fail("chunk must be 1 .. max_rows");
    e->chunk = value;
    return 0;
  } else if (n == "vision_capture") {
    if (value != 0 && value != 1) fail("vision_capture must be 0 or 1");
    if (e->vis_ready) fail("option vision_capture must be set before ink2_vision_finalize");
    e->vis_capture = value != 0;
    return 0;
  } else fail("unknown option " + n);
  e->f2_resolve();  // fast kernels, second set
  e->free_graphs();  // the recorded kernel sequences depend on the options
  return 0;
  API_END
}

INK2_API int ink2_timing(void* h, double* out4) {
  Engine* e = (Engine*)h;
  out4[0] = e->t_enqueue;
  out4[1] = e->t_wait;
  out4[2] = (double)e->t_steps;
  out4[3] = (double)e->n_launch;
  e->t_enqueue = e->t_wait = 0;
  e->t_steps = e->n_launch = 0;
  return e->graphs_built;
}

INK2_API int ink2_num_visits(void* h) { return (int)((Engine*)h)->visits.size(); }

INK2_API int ink2_visit_info(void* h, int visit, int* layer, int* pass) {
  API_BEGIN(h)
  if (visit < 0 || visit >= (int)e->visits.size()) fail("visit out of range");
  *layer = e->visits[visit].layer;
  *pass = e->visits[visit].pass;
  return 0;
  API_END
}

INK2_API int ink2_get_row(void* h, int what, int visit, int row, void* out, long long out_bytes) {
  API_BEGIN(h)
  if (!e->finalized) fail("get before finalize");
  const void* src = nullptr;
  long long n = 0;
  bool per_visit = what <= INK2_GET_ROUTER_LOGITS;
  if (per_visit && (visit < 0 || visit >= (int)e->visits.size())) fail("visit out of range");
  int rows = what >= INK2_GET_FINAL_HIDDEN ? e->logit_rows : e->max_rows;
  if (row < 0 || row >= rows) fail("row out of range");
  const size_t r = (size_t)row;
  const Ink2Config& c = e->c;
  switch (what) {
    case INK2_GET_HIDDEN: src = e->visits[visit].hid + r * c.hidden, n = 4LL * c.hidden; break;
    case INK2_GET_ROUTER_INDEX: src = e->visits[visit].ridx + r * c.top_k, n = 4LL * c.top_k; break;
    case INK2_GET_ROUTER_WEIGHT: src = e->visits[visit].rwt + r * c.top_k, n = 4LL * c.top_k; break;
    case INK2_GET_ROUTER_LOGITS: src = e->visits[visit].rlog + r * c.experts, n = 4LL * c.experts; break;
    case INK2_GET_EMBED: src = e->emb + r * c.hidden, n = 4LL * c.hidden; break;
    case INK2_GET_FINAL_HIDDEN: src = e->fh + r * c.hidden, n = 4LL * c.hidden; break;
    case INK2_GET_LOGITS: src = e->logits + r * c.vocab, n = 4LL * c.vocab; break;
    default: fail("unknown buffer");
  }
  if (out_bytes != n) fail(fmt("buffer has %lld bytes, the data %lld", out_bytes, n));
  e->be->download(out, src, (size_t)n);
  e->check_backend();
  return 0;
  API_END
}

INK2_API int ink2_get(void* h, int what, int visit, void* out, long long out_bytes) {
  return ink2_get_row(h, what, visit, 0, out, out_bytes);
}

INK2_API long long ink2_memory_used(void* h) { return (long long)((Engine*)h)->be->bytes_allocated(); }

INK2_API int ink2_kv_memory(void* h, long long* cache, long long* workspace) {
  API_BEGIN(h)
  if (!e->finalized || !cache || !workspace) fail("kv_memory needs a finalized engine and two outputs");
  *cache = (long long)e->kv_cache_bytes;
  *workspace = (long long)e->kv_work_bytes;
  return 0;
  API_END
}

INK2_API int ink2_device_memory(void* h, long long* free_bytes, long long* total_bytes) {
  return ((Engine*)h)->be->device_memory(free_bytes, total_bytes) ? 0 : -1;
}

INK2_API int ink2_profile(void* h, int enable, double* seconds_out, int n) {
  Engine* e = (Engine*)h;
  for (int i = 0; i < n && i < INK2_PROF_COUNT; ++i) seconds_out[i] = e->prof_t[i];
  for (int i = 0; i < INK2_PROF_COUNT; ++i) e->prof_t[i] = 0;
  e->prof = enable != 0;
  return 0;
}

INK2_API const char* ink2_kernel_name(int kid) {
  static const char* names[] = {
#define X(name, A) #name,
      INK2_KERNEL_LIST(X)
#undef X
  };
  return kid >= 0 && kid < KID_COUNT ? names[kid] : nullptr;
}

// reps > 0: every kernel of the following steps runs reps times in a row and
// is timed alone (the results of those steps are not valid; reset afterwards).
// Returns the seconds of one run and the launch counts per kernel, and clears them.
INK2_API int ink2_bench(void* h, int reps, double* seconds_out, long long* counts_out, int n) {
  Engine* e = (Engine*)h;
  for (int i = 0; i < n && i < KID_COUNT; ++i) {
    seconds_out[i] = e->bench_t[i];
    counts_out[i] = e->bench_n[i];
  }
  for (int i = 0; i < KID_COUNT; ++i) e->bench_t[i] = 0, e->bench_n[i] = 0;
  e->bench_reps = reps;
  return KID_COUNT;
}

INK2_API int ink2_capture(void* h, int visit) {
  Engine* e = (Engine*)h;
  e->cap_visit = visit;
  e->caps.clear();
  return 0;
}

INK2_API int ink2_capture_count(void* h) { return (int)((Engine*)h)->caps.size(); }

INK2_API int ink2_capture_get(void* h, int i, char* name_out, int name_cap, float* data_out, long long max_floats) {
  Engine* e = (Engine*)h;
  if (i < 0 || i >= (int)e->caps.size()) return -1;
  const Capture& c = e->caps[i];
  if (name_out && name_cap > 0) snprintf(name_out, name_cap, "%s", c.name.c_str());
  long long n = (long long)c.data.size();
  if (data_out) memcpy(data_out, c.data.data(), (size_t)(n < max_floats ? n : max_floats) * 4);
  return (int)n;
}

INK2_API int ink2_microbench(char* report, int report_cap) {
#ifdef INK2_WITH_CUDA
  return ink2_cuda_microbench(report, report_cap);
#else
  snprintf(report, report_cap, "microbench: this build has no cuda backend\n");
  return -1;
#endif
}

INK2_API int ink2_selftest(const char* backend, char* report, int report_cap) {
  Backend *a = nullptr, *b = nullptr;
  int rc = -1;
  std::string text;
  try {
    a = ink2_make_host_backend();
    b = make_backend(backend);
    rc = ink2_run_selftest(a, b, text);
  } catch (const std::exception& ex) {
    g_last_error = ex.what();
    text += std::string("self-test error: ") + ex.what() + "\n";
    rc = -1;
  }
  delete a;
  delete b;
  if (report && report_cap > 0) snprintf(report, report_cap, "%s", text.c_str());
  return rc;
}

// ---------------------------------------------------------------------------
// Image input (ink2_vision_api.h). These calls own the feature buffer and every check of the arguments; the
// image tower itself (ink2_vision.cu, built with INK2_WITH_VISION) only sees arguments that passed them.
// ---------------------------------------------------------------------------
namespace {

void Engine::vision_release() {
#ifdef INK2_WITH_VISION
  if (vis) ink2v_destroy(vis);  // waits for the stream
  if (img_feat) ink2v_free_rows(img_feat);
#endif
  vis = nullptr;
  img_feat = nullptr;
  vis_ready = false;
  img_valid.clear();
}

Engine* vision_engine(void* h, bool need_ready) {
  Engine* e = (Engine*)h;
  if (!e) fail("no engine");
  if (!e->vis) fail("no image tower: call ink2_vision_create first");
  if (need_ready && !e->vis_ready) fail("the image tower is not finalized: call ink2_vision_finalize first");
  return e;
}

// [row0, row0 + n) inside the feature buffer
void check_feature_range(const Engine* e, long long row0, long long n) {
  const long long cap = e->vis_cfg.max_feature_rows;
  if (n < 1) fail("at least one feature row is needed");
  if (row0 < 0 || row0 + n > cap)
    fail(fmt("feature rows %lld .. %lld lie outside the buffer of %lld rows (max_feature_rows)", row0, row0 + n - 1, cap));
}

[[noreturn]] void tower_fail(const char* what, const char* err) {
  fail(std::string("image tower: ") + what + ": " + (err && err[0] ? err : "failed without a message"));
}

// The published values of the Inkling-2 Mini image tower; anything else is refused with the key named.
void check_vision_config(const Ink2VisionConfig& v, int text_hidden) {
  auto need = [](bool ok, const char* key, const char* expected) {
    if (!ok) fail(std::string("vision config: ") + key + " must be " + expected);
  };
  need(v.hidden == 1024, "hidden_size", "1024");
  need(v.layers == 23, "num_hidden_layers", "23");
  need(v.heads == 16, "num_attention_heads", "16");
  need(v.head_dim == 64, "head_dim", "64 (hidden_size / num_attention_heads)");
  need(v.inter == 4096, "the MLP size", "4096 (mlp_ratio 4.0)");
  need(v.patch == 20, "native_patch_size", "20");
  need(v.merge == 2, "adapter_merge_size", "2");
  need(v.temporal == 2, "temporal_patch_size", "2");
  need(v.grid == 32, "posemb_grid_size", "32");
  need(v.proj_hidden == 4096, "projection_hidden_dim", "4096");
  need(v.projector_bias == 0, "projector_bias", "false");
  need(v.text_hidden == text_hidden, "text_hidden_size", "the hidden size of the text model");
  need(v.eps == 1e-5f, "layer_norm_eps", "1e-5");
  need(v.rope_theta == 10000.0f, "rope_theta", "10000");
  need(v.max_image_tokens >= 64 && v.max_image_tokens <= 4096, "max_image_tokens", "64 .. 4096");
  need(v.max_feature_rows >= v.max_image_tokens && v.max_feature_rows <= 65536, "max_feature_rows",
       "max_image_tokens .. 65536");
}

}  // namespace

INK2_API int ink2_vision_create(void* h, const Ink2VisionConfig* cfg) {
  API_BEGIN(h)
  if (!e) fail("no engine");
  if (std::string(e->be->name()) != "cuda") fail("image input needs the cuda backend");
  if (!e->finalized) fail("ink2_vision_create needs the finalized text model (ink2_finalize first)");
  if (e->vis) fail("the image tower is already created (ink2_vision_unload first)");
  if (!cfg) fail("no vision config");
#ifdef INK2_WITH_VISION
  check_vision_config(*cfg, e->c.hidden);
  char err[INK2V_ERR_LEN] = "";
  // The engine's own stream: the encoder's kernels and the engine's passes are then ordered by the stream itself.
  Ink2Vis* v = ink2v_create(cfg, (CUstream_st*)e->be->native_stream(), err);
  if (!v) tower_fail("create", err);
  e->vis = v;
  e->vis_cfg = *cfg;
  e->vis_ready = false;
  return 0;
#else
  fail("this library was built without the image tower");
#endif
  API_END
}

INK2_API int ink2_vision_set_tensor(void* h, const char* name, const void* bf16, long long nbytes) {
  API_BEGIN(h)
  vision_engine(h, false);
  if (e->vis_ready) fail("ink2_vision_set_tensor after ink2_vision_finalize");
  if (!name || !bf16) fail("ink2_vision_set_tensor: no name or no data");
  if (nbytes <= 0) fail(std::string("empty tensor ") + name);
#ifdef INK2_WITH_VISION
  char err[INK2V_ERR_LEN] = "";
  if (ink2v_set_tensor(e->vis, name, bf16, nbytes, err) != 0) tower_fail(name, err);
  return 0;
#else
  fail("this library was built without the image tower");
#endif
  API_END
}

INK2_API int ink2_vision_set_tensor_shaped(void* h, const char* name, const char* dtype, const long long* shape, int ndim,
                                           const void* bf16, long long nbytes) {
  API_BEGIN(h)
  vision_engine(h, false);
  if (e->vis_ready) fail("ink2_vision_set_tensor_shaped after ink2_vision_finalize");
  if (!name || !bf16) fail("ink2_vision_set_tensor_shaped: no name or no data");
  if (!dtype || strcmp(dtype, "BF16") != 0)
    fail(fmt("tensor %.200s: dtype %.16s, expected BF16", name, dtype ? dtype : "(none)"));
  if (!shape || ndim < 1 || ndim > 4) fail(fmt("tensor %.200s: %d dimensions given, expected 1 .. 4 with a shape", name, ndim));
  if (nbytes <= 0) fail(std::string("empty tensor ") + name);
#ifdef INK2_WITH_VISION
  char err[INK2V_ERR_LEN] = "";
  if (ink2v_set_tensor_shaped(e->vis, name, bf16, 2, shape, ndim, nbytes, err) != 0) tower_fail(name, err);
  return 0;
#else
  fail("this library was built without the image tower");
#endif
  API_END
}

INK2_API int ink2_vision_finalize(void* h) {
  API_BEGIN(h)
  vision_engine(h, false);
  if (e->vis_ready) fail("the image tower is already finalized");
#ifdef INK2_WITH_VISION
  // The feature buffer first (with a look at the free memory), then the encoder scratch. Both are taken outside the
  // backend: a failed allocation here refuses image input and leaves the text engine working.
  const size_t bytes = (size_t)e->vis_cfg.max_feature_rows * (size_t)e->c.hidden * 2;
  long long free_b = 0, total_b = 0;
  if (e->be->device_memory(&free_b, &total_b) && free_b < (long long)bytes + (64ll << 20))
    fail(fmt("the feature buffer needs %lld MiB, %lld MiB are free", (long long)(bytes >> 20), free_b >> 20));
  char err[INK2V_ERR_LEN] = "";
  if (ink2v_set_capture(e->vis, e->vis_capture ? 1 : 0, err) != 0) tower_fail("finalize", err);
  if (e->img_feat) ink2v_free_rows(e->img_feat);  // not reached today: a failed finalize frees its buffer
  e->img_feat = nullptr;
  u16* buf = ink2v_alloc_rows((long long)bytes, err);
  if (!buf) tower_fail("finalize", err);
  if (ink2v_finalize(e->vis, err) != 0) {
    ink2v_free_rows(buf);
    tower_fail("finalize", err);
  }
  e->img_feat = buf;
  e->img_valid.assign((size_t)e->vis_cfg.max_feature_rows, 0);
  e->vis_ready = true;
  e->free_graphs();  // the embedding kernel now reads the feature buffer: record the passes again
  return 0;
#else
  fail("this library was built without the image tower");
#endif
  API_END
}

INK2_API int ink2_vision_encode(void* h, const unsigned char* rgb, int width, int height, long long rgb_bytes, int rows,
                                int cols, int row0) {
  API_BEGIN(h)
  vision_engine(h, true);
  if (!rgb) fail("ink2_vision_encode: no pixels");
  if (width < 1 || height < 1) fail(fmt("image size %d x %d: both sides must be at least 1", width, height));
  if (rgb_bytes != (long long)width * (long long)height * 3)
    fail(fmt("image %d x %d: %lld bytes given, %lld expected (RGB)", width, height, rgb_bytes,
             (long long)width * (long long)height * 3));
  const int side = e->vis_cfg.patch * e->vis_cfg.merge;  // pixels per image vector and side (40)
  const long long want_rows = ((long long)height + side - 1) / side, want_cols = ((long long)width + side - 1) / side;
  if (rows != want_rows || cols != want_cols)
    fail(fmt("image %d x %d has a grid of %lld x %lld, not %d x %d", width, height, want_rows, want_cols, rows, cols));
  const long long n = want_rows * want_cols;
  if (n > e->vis_cfg.max_image_tokens)
    fail(fmt("image %d x %d gives %lld image vectors, at most %d per image (max_image_tokens)", width, height, n,
             e->vis_cfg.max_image_tokens));
  check_feature_range(e, row0, n);
#ifdef INK2_WITH_VISION
  char err[INK2V_ERR_LEN] = "";
  // The tower checks again and refuses before it writes anything, so a refusal leaves the rows as they were.
  if (ink2v_encode(e->vis, rgb, width, height, rgb_bytes, rows, cols, e->img_feat + (size_t)row0 * (size_t)e->c.hidden,
                   (long long)e->vis_cfg.max_feature_rows - row0, err) != 0)
    tower_fail("encode", err);
  // Queued on the engine's stream; a launch failure means the rows may be partly written: no longer valid.
  std::fill(e->img_valid.begin() + row0, e->img_valid.begin() + row0 + n, (unsigned char)0);
  e->check_backend();
  std::fill(e->img_valid.begin() + row0, e->img_valid.begin() + row0 + n, (unsigned char)1);
  return 0;
#else
  fail("this library was built without the image tower");
#endif
  API_END
}

INK2_API int ink2_vision_set_rows(void* h, const unsigned short* bf16, int row0, int n, long long nbytes) {
  API_BEGIN(h)
  vision_engine(h, true);
  if (!bf16) fail("ink2_vision_set_rows: no data");
  check_feature_range(e, row0, n);
  const long long want = (long long)n * e->c.hidden * 2;
  if (nbytes != want) fail(fmt("ink2_vision_set_rows: %lld bytes given, %lld expected (%d rows of %d bfloat16)", nbytes, want, n,
                               e->c.hidden));
  e->be->upload(e->img_feat + (size_t)row0 * (size_t)e->c.hidden, bf16, (size_t)want);
  e->check_backend();
  std::fill(e->img_valid.begin() + row0, e->img_valid.begin() + row0 + n, (unsigned char)1);
  return 0;
  API_END
}

INK2_API int ink2_vision_get_rows(void* h, float* out, int row0, int n, long long out_floats) {
  API_BEGIN(h)
  vision_engine(h, true);
  if (!out) fail("ink2_vision_get_rows: no destination");
  check_feature_range(e, row0, n);
  const size_t H = (size_t)e->c.hidden, count = (size_t)n * H;
  if (out_floats < (long long)count) fail(fmt("ink2_vision_get_rows: room for %lld floats, %lld needed", out_floats, (long long)count));
  for (int r = row0; r < row0 + n; ++r)
    if (!e->img_valid[r]) fail(fmt("feature row %d has not been written since the last reset", r));
  std::vector<u16> tmp(count);
  e->be->download(tmp.data(), e->img_feat + (size_t)row0 * H, count * 2);
  e->check_backend();
  for (size_t i = 0; i < count; ++i) {
    unsigned u = (unsigned)tmp[i] << 16;
    memcpy(out + i, &u, 4);
  }
  return 0;
  API_END
}

INK2_API int ink2_vision_stage(void* h, int stage, float* out, long long out_floats) {
  API_BEGIN(h)
  vision_engine(h, true);
  if (!out) fail("ink2_vision_stage: no destination");
  if (stage < 0 || stage > 6) fail(fmt("ink2_vision_stage: stage %d, expected 0 .. 6", stage));
#ifdef INK2_WITH_VISION
  char err[INK2V_ERR_LEN] = "";
  if (ink2v_stage(e->vis, stage, out, out_floats, err) != 0) tower_fail("stage", err);
  return 0;
#else
  fail("this library was built without the image tower");
#endif
  API_END
}

INK2_API int ink2_vision_memory(void* h, long long* weights, long long* scratch, long long* features) {
  API_BEGIN(h)
  if (!e) fail("no engine");
  long long w = 0, s = 0;
#ifdef INK2_WITH_VISION
  char err[INK2V_ERR_LEN] = "";
  if (e->vis && ink2v_memory(e->vis, &w, &s, err) != 0) tower_fail("memory", err);
#endif
  if (weights) *weights = w;
  if (scratch) *scratch = s;
  if (features) *features = e->img_feat ? (long long)e->vis_cfg.max_feature_rows * e->c.hidden * 2 : 0;
  return 0;
  API_END
}

INK2_API int ink2_vision_unload(void* h) {
  API_BEGIN(h)
  if (!e) fail("no engine");
  const bool had_buffer = e->img_feat != nullptr;
  e->vision_release();
  if (had_buffer) e->free_graphs();  // the recorded passes read the released buffer
  return 0;
  API_END
}

INK2_API int ink2_feed_many_img(void* h, const int* ids, const int* img_row, int n) {
  API_BEGIN(h)
  if (!e) fail("no engine");
  if (!e->finalized) fail("feed before finalize");
  if (n < 0) fail("negative count");
  if (n > 0 && !ids) fail("ink2_feed_many_img: no ids");
  if ((long long)e->pos + n > e->c.max_positions) fail(fmt("context full: %d positions", e->c.max_positions));
  // Every row is checked before the first pass runs, so a refused call changes nothing.
  int last = -1, first_img = -1;
  for (int i = 0; i < n; ++i) {
    if (ids[i] < 0 || ids[i] >= e->c.embed_rows) fail(fmt("position %d: token id %d out of range", i, ids[i]));
    const int r = img_row ? img_row[i] : -1;
    if (r == -1) continue;
    if (r < -1) fail(fmt("position %d: feature row %d (expected -1 for text, else a row >= 0)", i, r));
    if (!e->vis_ready) fail(fmt("position %d names feature row %d, but no image tower is finalized", i, r));
    if (ids[i] != INK2_IMAGE_TOKEN)
      fail(fmt("position %d: an image position must carry the token %d, not %d", i, INK2_IMAGE_TOKEN, ids[i]));
    if (r >= e->vis_cfg.max_feature_rows)
      fail(fmt("position %d: feature row %d lies outside the buffer of %d rows", i, r, e->vis_cfg.max_feature_rows));
    if (!e->img_valid[r]) fail(fmt("position %d: feature row %d has not been written since the last reset", i, r));
    if (r <= last) fail(fmt("position %d: feature row %d after row %d (the rows of one call must increase)", i, r, last));
    if (first_img < 0) first_img = i;
    last = r;
  }
  e->feed_many(ids, n, first_img >= 0 ? img_row : nullptr);
  return 0;
  API_END
}

#include "ink2_drafter.inc"  // drafter hook
