// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Argument blocks of the kernels. Plain data, passed by value to a CUDA
// kernel and by reference to the host emulation of the same kernel.
#pragma once
#include <stddef.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned long long u64;

enum { INK2_EPI_NONE = 0, INK2_EPI_SIGMOID = 1, INK2_EPI_SOFTCAP = 2, INK2_EPI_TANH = 3 };
enum { INK2_FMT_NVFP4 = 0, INK2_FMT_MXFP8 = 1 };
// Storage of a dense (non-expert) matrix: GemvJob.fmt, NormArgs.efmt.
// INK2_DFMT_Q3: 3-bit codes (the converted drafter files): value = (q3[code] * e4m3[block scale]) * factor,
// q3[c] = (2 (c & 3) + 1) * (c & 4 ? -1 : 1); 8 values of a row in 3 bytes, one E4M3 scale per 16 values.
enum { INK2_DFMT_BF16 = 0, INK2_DFMT_NVFP4 = 1, INK2_DFMT_MXFP8 = 2, INK2_DFMT_Q3 = 3 };

// What changes from token to token. The kernels read it from the backend
// memory, so that the argument blocks of a step are the same for every token
// (a step can then be replayed as one captured graph). A pass over R
// consecutive positions has one StepState per row (array of R).
//
// Rows. Every kernel that takes `rows` (or `nrhs`) handles R consecutive
// positions of one pass; per-row buffers are [R, n] with the row stride n of
// the buffer named in the comment. rows == 0 means one row (old argument
// blocks of the self-test). The arithmetic of a row is the same as with one
// row, so that a pass over R rows gives bit for bit the results of R single
// steps on the same backend.
// Short convolution histories are rings of hr >= K - 1 slots, position p in
// slot p % hr; a row whose earlier inputs lie inside the pass reads them from
// the input buffer of the pass instead (hr >= R + K - 1 keeps the slots read
// and written by one kernel apart).
struct StepState {
  int token, pos;
  float tau;  // log scaling factor of this position
  int pad;    // 0: text position; k > 0: image position, feature row k - 1 (NormArgs.img)
};

// y = W x, W bfloat16 [rows, cols] (cols % 8 == 0), x and y float32.
// The input may come in two pieces: x for the columns below split, x2 for the rest.
struct GemvJob {
  const u16* w;  // bfloat16 [rows, cols]; packed (fmt != BF16): the code bytes, see below
  const float* x;
  const float* x2;
  float* y;
  int rows, cols, split, epi;
  float cap;
  // Packed dense rows (kernels pgemv, pgemvw, pgemvc; fmt INK2_DFMT_*):
  //   nvfp4: w = codes [rows, cols/2] (even element in the low nibble), sc = E4M3 block scales [rows, cols/16];
  //   mxfp8: w = E4M3 bytes [rows, cols], sc = exponent bytes [rows, cols/32] (value * 2 ** (byte - 127));
  //   q3:    w = codes [rows, 3 cols/8] (8 values in 3 bytes, value e in bits 3e .. 3e+2 of the little-endian
  //          24-bit word), sc = E4M3 block scales [rows, cols/16]; fac = the matrix factor.
  // fac != 0: factor of every row sum before the epilogue (nvfp4: amax / 2688 / 2, the codes being
  // decoded as twice the E2M1 value); also applied by gemv, gemvw and gemvc to bfloat16 rows.
  const u8* sc;
  float fac;
  int fmt;
};
// Multi-row forms (gemvm, gemvwm, gemvcm): nrhs right-hand sides, x row
// stride split, x2 row stride cols - split, y row stride rows; groups of up to
// 8 right-hand sides, `blocks` blocks per group.
struct GemvArgs {
  GemvJob job[8];
  int njobs, total_rows;
  int nrhs, blocks;
};

// out = w * (x * rsqrt(mean(x^2) + eps)); w == NULL: no weight. One block per job.
struct NormJob {
  const void* x;  // float32, or bfloat16 bits when x_bf16
  const u16* w;
  float* out;
  int x_bf16;
};
struct NormArgs {
  NormJob job[4];
  const StepState* st;
  int embed_stride;  // > 0: job 0 reads row st[row].token of the bfloat16 table x
  int n;
  float eps;
  int rows, nj;  // block = row * nj + job; x and out [rows, n]
  // efmt == INK2_DFMT_MXFP8: the table of job 0 (embed_stride > 0) is packed: x = E4M3 bytes
  // [table rows, n], esc = exponent bytes [table rows, n / 32]
  const u8* esc;
  int efmt;
  // Image positions (ink2_vision_api.h): with img != NULL, a row of job 0 (embed_stride > 0) whose
  // st[row].pad = k > 0 takes the bfloat16 feature row img[k - 1] (n values) as it is: no table row, no norm.
  const u16* img;
};

// Top-k routing from the router logits. One block of >= E threads.
struct RouterArgs {
  const float* logits;  // [rows, E]
  const float* bias;
  int* idx;   // [rows, k], increasing
  float* wt;  // [rows, k]
  int E, k;
  float route_scale;
  int rows;  // one block per row
};

// Block attention residual: nq queries over ns <= 6 states. One block of H / 4
// threads per job.
// Optional first step: the last state st[ns-1] (= cout) is produced by a short
// convolution of cx (+ residual cres). Per output: RMS norm with weight in
// place (normw), or an extra RMS-normalised copy (outn).
struct BarJob {
  const float* st[8];
  const float* q;  // [nq, H]
  float* out[4];
  float* outn[4];
  const u16* normw[4];
  const float* cx;
  const u16* cw;  // [H, K] or NULL
  float* chist;   // [(K-1), H]
  const float* cres;
  float* cout;
  int ns, nq;
};
// Rows: block = row * nj + job; every per-row pointer of a job ([rows, H]:
// states, outputs, cx, cres, cout) advances by H per row, q is shared.
struct BarArgs {
  BarJob job[2];
  const StepState* st;
  int H, K;
  float hscale;  // H ** -0.5
  float eps;     // of the optional norms
  float cdiv;
  int rows, nj, hr;
};

// Last step of a message gate: (1 - f) * hs + g * msg, then the optional RMS
// norm with weight. One block per job.
struct GmixJob {
  const float* hs;
  const float* msg;
  const float* f;  // tanh(forget projection)
  const float* g;  // tanh(gain projection)
  const u16* normw;
  float* out;
};
struct GmixArgs {
  GmixJob job[4];
  int H;
  float eps;
  int rows, nj;  // block = row * nj + job; all pointers [rows, H]
};

// Query / relative state scaling and clipping, key / value convolution,
// clipping / normalisation and the write into the cache slot.
// KV cache of a visit: FP32 (kc / vc of the visit) or MXFP8 (ink2_kv_mxfp8.inc).
enum { INK2_KV_FP32 = 0, INK2_KV_MXFP8 = 8 };
struct KvStorage {
  int mode, sinks, hot, hot_slots;
  u8 *key, *value;           // E4M3 codes [cap, nkv * hd] (MXFP8)
  u8 *ks, *vs;               // E8M0 exponents, one per group of 32 codes
  u16 *near_k, *near_v;      // BF16 copies [sinks + hot_slots, nkv, hd]
  int* tags;                 // logical position + 1 of a recent slot; zero is invalid
};

struct PrepArgs {
  float* q;  // [nh, hd] in place
  const u16* q_scale;
  float* r;  // [nh, drel] in place
  const u16* r_scale;
  const float* kraw;  // [nkv * hd]
  const float* vraw;
  float* khist;  // [(K-1), nkv * hd]
  float* vhist;
  float* kc;  // cache [cap, nkv * hd]; the slot of the position is written
  float* vc;
  const u16* k_w;  // sliding: [C, K]; full: static [K, C]
  const u16* v_w;
  const float* k_dyn;  // full: [nkv, K]
  const float* v_dyn;
  const StepState* st;
  int nh, nkv, hd, drel, K, sliding, ring;
  float qk_thr, rel_thr, eps, conv_div;
  // rows: q [rows, nh*hd], r [rows, nh*drel], kraw/vraw [rows, nkv*hd],
  // k_dyn/v_dyn [rows, nkv*K]; bpr blocks per row; hr history ring slots
  int rows, bpr, hr;
  KvStorage kv;
};

// Attention over the cached keys. One block per head.
struct AttnArgs {
  const float* q;
  const float* r;
  const float* sink;  // [nh]
  const float* gate;  // [nh * hd], sigmoid already applied
  const float* kc;    // [cap, nkv * hd]
  const float* vc;
  const u16* rel_proj;  // [drel, rel_extent]
  float* scratch;       // [nh, scratch_stride]
  float* out;           // [nh * hd]
  const StepState* st;
  int nh, nkv, hd, drel, rel_extent, ring, scratch_stride;
  float scale;  // hd ** -0.5
  // window: keys visible to a query (0 = all); ring: slots of the cache (0 =
  // no wrap). rows: block = row * nh + head; q, gate, out [rows, nh*hd],
  // r [rows, nh*drel], sink [rows, nh], scratch [rows * nh, scratch_stride]
  int window, rows;
  KvStorage kv;
  int splits, row_tile;
  float* partial;  // [row, head, split, 130]: vector, max, exponential sum
  // attn_gqa_splitk: min keys per split; rel_proj as [rel_extent, drel] (or null)
  int sk_chunk;
  const u16* rel_t;
  // tiled Flash-GQA prompt attention (attn_rel + attn_flash): relative logits per
  // (row, head, distance) [rows * nh, rel_stride]; flash_tc = warps of the tensor-core form (0 = FP32 form)
  float* rel_logits;
  int rel_stride, flash_tc;
  // rel_proj transposed to [rel_extent, drel] (one 32-byte read per key); null = use rel_proj
  const u16* rel_proj_t;
};

struct KvCaptureArgs {
  KvStorage kv;
  const StepState* st;
  float *k, *v;
  int ring, channels;
};

// Experts, multiplied straight from the packed bytes.
struct ExpArgs {
  const u8* w;          // [E, rows, cols/2] (nvfp4) or [E, rows, cols] (mxfp8)
  const u8* sc;         // [E, rows, cols/16] or [E, rows, cols/32]
  const float* factor;  // nvfp4: [E] amax / denominator / 2
  const int* idx;       // [k]
  const float* wt;      // [k] (w2 only)
  const float* x;       // w13: [cols]; w2: [k, cols]
  float* y;             // w13: act [k, rows/2]; w2: [rows]
  const float* e4m3;    // [256]
  const float* exp2t;   // [256]
  // w2 only, optional: short convolution of the output + residual -> cout
  const u16* cw;  // [rows, K] or NULL
  float* chist;   // [(K-1), rows]
  const float* cres;
  float* cout;
  const StepState* st;
  int fmt, rows, cols, k, K;
  float gate_cap, up_cap, cdiv;
  int hr;
};

// Experts of a multi-row pass, grouped by expert: egroup lists the distinct
// experts of the pass and for each the (row, slot) pairs p = row * k + slot
// that chose it (in increasing p). Every needed expert is then read once.
struct GroupArgs {
  const int* idx;  // [R, k]
  int* gcount;     // [1] distinct experts
  int* gexp;       // [<= E] their ids, increasing
  int* goff;       // [<= E + 1] first pair of each in gpairs
  int* gpairs;     // [R * k]
  int R, k, E;
};
struct ExpMArgs {
  const u8* w;
  const u8* sc;
  const float* factor;
  const float* e4m3;
  const float* exp2t;
  const int* gcount;
  const int* gexp;
  const int* goff;
  const int* gpairs;
  const float* x;  // w13: [R, cols]; w2: activations [R * k, cols]
  float* y;        // w13: activations [R * k, rows / 2]; w2: products [R * k, rows]
  int fmt, rows, cols, k;
  int nchunk;  // blocks of matrix rows per expert
  int grid;    // blocks of the launch (the items distinct * nchunk are spread over them)
  float gate_cap, up_cap;
  // Wide prompt passes can pack the inputs once for all routed experts.
  int input_rows, expert_slots, nvfp4_planes;
  u8* nvfp4_scratch;
  bool register_down;
};
// Sum of the k expert products of every row (in slot order, as exp2), then
// the short convolution + residual. One thread per (row, channel).
struct ECombArgs {
  const float* dots;  // [R * k, H]
  const float* wt;    // [R, k]
  float* y;           // [R, H]
  const u16* cw;      // [H, K] or NULL
  float* chist;       // [hr, H]
  const float* cres;  // [R, H]
  float* cout;        // [R, H]
  const StepState* st;
  int R, k, H, K, hr;
  float cdiv;
};
// Largest logit of every row (first index on ties). One block per row.
struct ArgmaxArgs {
  const float* logits;  // [rows, n]
  int* idx;             // [rows]
  float* val;           // [rows]
  int n;
};
constexpr int ink2_pick_tile = 1024;
struct PickArgs {
  const float* values;
  const int* indices;  // null on the vocabulary pass
  float* maxima;
  int* winners;
  int n;
};

#include "ink2_drafter_kargs.h"  // drafter hook

#define INK2_KERNEL_LIST(X) \
  X(gemv, GemvArgs) \
  X(gemvw, GemvArgs) \
  X(gemvc, GemvArgs) \
  X(norm, NormArgs) \
  X(bar, BarArgs) \
  X(gmix, GmixArgs) \
  X(prep, PrepArgs) \
  X(attn, AttnArgs) \
  X(topk, RouterArgs) \
  X(exp13, ExpArgs) \
  X(exp13c, ExpArgs) \
  X(exp2, ExpArgs) \
  X(gemvm, GemvArgs) \
  X(gemvwm, GemvArgs) \
  X(gemvcm, GemvArgs) \
  X(egroup, GroupArgs) \
  X(exp13cm, ExpMArgs) \
  X(exp2m, ExpMArgs) \
  X(ecomb, ECombArgs) \
  X(argmax, ArgmaxArgs)   X(pgemv, GemvArgs)   X(pgemvw, GemvArgs)   X(pgemvc, GemvArgs)   X(pgemvc8, GemvArgs)   X(pgemvc4, GemvArgs)   X(pgemvc2, GemvArgs)   X(pgemvn, GemvArgs)   X(pgemvcm, GemvArgs)   X(pgemvcm8, GemvArgs)   X(pgemvcm4, GemvArgs)   X(pgemvcm2, GemvArgs) \
  X(exp13s, ExpMArgs) X(exp13sx, ExpMArgs) X(exp2s, ExpMArgs) X(exp2sx, ExpMArgs) \
  X(pgcm16h2, GemvArgs) X(pgcm8h2, GemvArgs) X(pgcm4h2, GemvArgs) X(pgcm2h2, GemvArgs) \
  X(pgcm16h4, GemvArgs) X(pgcm8h4, GemvArgs) X(pgcm4h4, GemvArgs) X(pgcm2h4, GemvArgs) \
  INK2_DRAFT_KERNEL_LIST(X)  \
  X(kv_attention, AttnArgs) X(kv_merge, AttnArgs) X(kv_capture, KvCaptureArgs) X(kv_stripes, AttnArgs) \
  X(pick_tile, PickArgs) X(attn_gqa_splitk, AttnArgs) X(attn_splitk_reduce, AttnArgs) \
  X(attn_rel, AttnArgs) X(attn_flash, AttnArgs)

enum {
#define X(name, A) KID_##name,
  INK2_KERNEL_LIST(X)
#undef X
  KID_COUNT
};

// The interface between the shared engine and a backend.
struct Backend {
  virtual ~Backend() {}
  virtual const char* name() = 0;
  virtual void* alloc(size_t bytes) = 0;  // zero filled, 256-byte aligned
  virtual void release(void* p) = 0;
  virtual void upload(void* dst, const void* host_src, size_t bytes) = 0;
  virtual void download(void* host_dst, const void* src, size_t bytes) = 0;
  virtual void zero(void* dst, size_t bytes) = 0;
  virtual void sync() = 0;
  virtual void run(int kid, const void* args, int grid, int block) = 0;
  // largest block of kernel kid that can be launched (registers of the build for this device)
  virtual int max_block(int kid) { return 1024; }
  virtual int sm_count() { return 0; }
  virtual size_t bytes_allocated() = 0;
  virtual bool device_memory(long long* free_bytes, long long* total_bytes) = 0;
  virtual const char* error() = 0;  // NULL when no error happened
  // Optional: record the kernels of a step once and replay them with one call.
  virtual bool graph_supported() { return false; }
  virtual bool supports_nvfp4_prefill() { return false; }
  virtual bool set_packed_prompt(bool enabled) { return !enabled; }
  virtual bool set_draft_l2(int mib) { return mib == 0; }
  virtual void graph_begin() {}
  virtual void* graph_end() { return nullptr; }
  virtual void graph_launch(void*) {}
  virtual void graph_free(void*) {}
  // Fast kernels (ink2_fast.h, CUDA only): compute capability of the device when the fast kernels can run there
  // (86 / 89 / 120 ...), else 0; run_fast launches fast kernel fid with an FxArgs block.
  virtual int fast_level() { return 0; }
  virtual void run_fast(int fid, const void* args, int grid, int block, int smem) {}
  virtual int fast_smem() { return 0; }  // largest dynamic shared memory of a fast kernel launch (bytes)
  virtual bool set_pdl(bool on) { return false; }  // programmatic dependent launch of every kernel (true: now on)
  // Fast kernels, second set (ink2_fast2.h, CUDA only): options of the wrapped launches, own kernels (fid INK2_F2_*;
  // returns 0 or a CUDA error code, -1 without the kernels), largest block of fgemvm (0: none).
  virtual void set_f2(const struct F2Opts* o) {}
  virtual int run_f2(int fid, const void* args, int grid, int block) { return -1; }
  virtual int f2_gemvm_block() { return 0; }
  // CUDA: the stream (cudaStream_t) the kernels run on, for work launched outside this interface (the image
  // tower). NULL: the legacy default stream, which is ordered with the blocking stream of the CUDA backend.
  virtual void* native_stream() { return nullptr; }
};

Backend* ink2_make_host_backend();
Backend* ink2_make_cuda_backend(char* err, int err_cap);  // NULL on failure
