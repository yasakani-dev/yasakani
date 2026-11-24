// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// C API of the Inkling-2 Mini engine. One token per call
// (ink2_step / ink2_feed), or R consecutive positions in one pass
// (ink2_feed_many / ink2_step_many, ABI version 2); dense matrices bfloat16 or packed
// (ABI version 3); image input (ABI version 4, ink2_vision_api.h).
#pragma once

// 1: one token per call; 2: adds ink2_abi_version, ink2_feed_many, ink2_step_many,
// ink2_truncate, ink2_position, ink2_limits, ink2_get_row and the options
// "max_rows", "rollback", "chunk". Functions of version 1 are unchanged.
// 3: adds ink2_capabilities and ink2_packed_dense: dense (non-expert) matrices may be
// given packed (nvfp4 / mxfp8, see "Packed dense matrices" below), and the options
// "pgemvc_rows", "pgemvw_rows", "pgemvc_min_blocks", "pgemvn". Nothing of version 2 is
// removed or changed: a model sent as in version 2 (bfloat16 dense matrices) runs as before.
// The drafter calls (ink2_drafter_api.h) are an addition within version 3, shown by INK2_CAP_DRAFTER.
// ink2_create checks the config values (head counts, sizes, experts and top-k, kernel and window sizes)
// and fails with a message in ink2_last_error(NULL) for values the kernels cannot take.
// 4: adds the image input of ink2_vision_api.h (ink2_vision_* and ink2_feed_many_img; the calls exist in every
// library of version 4, INK2_CAP_VISION says whether this one has the image tower). Text calls are unchanged.
#define INK2_ABI_VERSION 4

// Bits of ink2_capabilities().
enum {
  INK2_CAP_PACKED_DENSE = 1,  // dense matrices are accepted packed (nvfp4 / mxfp8), multiplied as stored
  INK2_CAP_DRAFTER = 2,       // the model's drafter (ink2_drafter_api.h: ink2_draft_abi, ink2_spec_pass, ...)
  INK2_CAP_DRAFTER_PACKED = 4,  // drafter matrices and folded head accepted packed (nvfp4 / mxfp8), ink2_draft_abi() >= 2
  INK2_CAP_Q3 = 8,              // the 3-bit format q3 (INK2_DFMT_Q3) for the drafter matrices
  INK2_CAP_KV = 16,             // options kv_mxfp8, kv_hot, kv_split and ink2_kv_memory
  INK2_CAP_VISION = 32          // the image tower and image positions (ink2_vision_api.h)
};

#if defined(_WIN32)
#ifdef INK2_BUILD
#define INK2_API extern "C" __declspec(dllexport)
#else
#define INK2_API extern "C" __declspec(dllimport)
#endif
#else
#define INK2_API extern "C" __attribute__((visibility("default")))
#endif

typedef struct Ink2Config {
  int vocab;       // rows of the output projection that are used (unpadded vocabulary)
  int embed_rows;  // rows of the embedding table
  int hidden, layers;
  int nh_full, nkv_full, hd_full;  // full-attention layers: heads, key/value heads, head size
  int nh_sw, nkv_sw, hd_sw;        // sliding-window layers
  int window, d_rel, rel_extent;
  int conv_k, expert_inner, experts, top_k;
  int bar_block;  // layers per attention-residual block
  int loop_start, loop_end, loop_passes, gate_rank;
  int n_floor;        // log scaling starts at this position count
  int max_positions;  // context limit (full-attention caches)
  float eps, qk_thr, rel_thr, alpha, route_scale, gate_cap, up_cap, logit_cap;
  int sliding[64];  // 1 = sliding-window layer
} Ink2Config;

enum {
  INK2_GET_HIDDEN = 0,         // float32 [hidden], output of a visit
  INK2_GET_ROUTER_INDEX = 1,   // int32 [top_k], increasing
  INK2_GET_ROUTER_WEIGHT = 2,  // float32 [top_k], same order
  INK2_GET_ROUTER_LOGITS = 3,  // float32 [experts]
  INK2_GET_EMBED = 4,          // float32 [hidden]
  INK2_GET_FINAL_HIDDEN = 5,   // float32 [hidden]
  INK2_GET_LOGITS = 6          // float32 [vocab]
};

// Profile slots of ink2_profile (seconds, accumulated while the profile is on).
enum {
  INK2_PROF_EXPERTS = 0,
  INK2_PROF_ATTENTION = 1,
  INK2_PROF_MATRIX = 2,
  INK2_PROF_OUTPUT = 3,
  INK2_PROF_MIX_NORM = 4,
  INK2_PROF_GATES = 5,
  INK2_PROF_CONV_ROUTER = 6,
  INK2_PROF_COPY = 7,
  INK2_PROF_COUNT = 8
};

INK2_API const char* ink2_build_info(void);
INK2_API void* ink2_create(const Ink2Config* cfg, const char* backend);  // "host" or "cuda"; NULL on error
INK2_API void ink2_destroy(void* h);
INK2_API const char* ink2_last_error(void* h);  // h may be NULL (errors of ink2_create / ink2_selftest)
INK2_API int ink2_set_tensor(void* h, const char* name, const void* data, long long nbytes);
INK2_API int ink2_finalize(void* h);
INK2_API int ink2_reset(void* h);
INK2_API int ink2_step(void* h, int token, float* logits_out);  // logits_out may be NULL
INK2_API int ink2_step_top1(void* h, int token, int* winner);  // with device_greedy, read only the first argmax
// A token whose prediction is not needed (all prompt tokens but the last): no output projection,
// no copy, no wait. INK2_GET_FINAL_HIDDEN and INK2_GET_LOGITS are not valid after it.
INK2_API int ink2_feed(void* h, int token);
// Options: "graph" 1 = record the kernels of a step once and replay them with one call (cuda only);
// "gemvc" / "exp13c" 0 = use the row-per-warp kernels instead of the column-owner forms (default 1;
// passes of several rows always use the column-owner expert form);
// before ink2_finalize only: "max_rows" = largest R of a pass (default 128, R * top_k <= 2048),
// "rollback" = positions ink2_truncate can always drop (default 64; the rings keep
// max(max_rows, rollback) extra positions); "chunk" = rows per pass of ink2_feed_many (default 128).
// With INK2_CAP_KV, before finalize: "kv_mxfp8" = 1 stores the keys and values of the full-attention
// visits as MXFP8 (E4M3 codes, one power-of-two scale per 32 values; default 0 = FP32), "kv_hot" = the
// recent positions that also keep a BF16 copy (default 128; 0 or a power of two from 16 through 256).
// "kv_split" selects 1..32 partitions of global attention; it may be changed between passes after finalize.
INK2_API int ink2_set_option(void* h, const char* name, int value);

// Packed dense matrices (version 3, INK2_CAP_PACKED_DENSE). A dense matrix "<name>" of [rows, cols]
// (cols a multiple of 32) is bfloat16 when only "<name>" is set (version 2), else:
//   nvfp4: "<name>" = codes u8 [rows, cols / 2] (two E2M1 codes per byte, the even column in the low
//          nibble), "<name>_scale" = E4M3 bytes [rows, cols / 16] (one per 16 consecutive columns),
//          "<name>_factor" = float32 [1] = amax / denominator (the matrix factor); value =
//          e2m1[code] * e4m3[scale] * factor;
//   mxfp8: "<name>" = E4M3 bytes [rows, cols], "<name>_scale" = u8 exponents [rows, cols / 32];
//          value = e4m3[byte] * 2 ** (scale - 127); no "_factor".
// The embedding may be bfloat16 or mxfp8; the router and the small tensors stay bfloat16. Every
// other matrix of the model may be either format, independently ("unembed" with the unpadded rows).
// "<name>" bfloat16 together with "<name>_factor" multiplies every row sum by the factor (bfloat16
// values of nvfp4 without their matrix factor: the same numbers as packed, for checks).

// ---- version 2 ----
INK2_API int ink2_abi_version(void);  // INK2_ABI_VERSION of the DLL
// n prompt tokens whose predictions are not needed, in passes of "chunk" rows (no output
// projection, no copy, no wait). Same results as n calls of ink2_feed.
INK2_API int ink2_feed_many(void* h, const int* ids, int n);
// n consecutive tokens in one pass with the logits of every row (n <= max step rows, see
// ink2_limits): same results as n calls of ink2_step. Any output may be NULL:
// logits_out [n, vocab], argmax_out [n] (first index of the largest logit), max_out [n] (that logit).
INK2_API int ink2_step_many(void* h, const int* ids, int n, float* logits_out, int* argmax_out, float* max_out);
// Drops the positions from n_positions on (rejected drafted tokens, a chat continued from an
// earlier point): at most `rollback` positions (ink2_limits out4[2]) without recomputing; more
// gives an error (reset and feed again). n_positions = 0 is a reset.
INK2_API int ink2_truncate(void* h, int n_positions);
INK2_API int ink2_position(void* h);  // positions seen so far
// out4: max rows of a pass, max rows of ink2_step_many, positions ink2_truncate can drop, max_positions.
INK2_API int ink2_limits(void* h, int* out4);
// As ink2_get, for row `row` of the last pass (row 0 = the token of the last single step).
INK2_API int ink2_get_row(void* h, int what, int visit, int row, void* out, long long out_bytes);

// ---- version 3 ----
INK2_API int ink2_capabilities(void);  // INK2_CAP_* bits of this DLL
INK2_API int ink2_packed_dense(void* h);  // after ink2_finalize: dense matrices loaded packed (0 = all bfloat16)
INK2_API int ink2_fast_path(void* h);  // after finalize: 0 = general kernels, else the compute capability the fast kernels run on
INK2_API int ink2_nvfp4_prompt(void);  // 1: the NVFP4 tensor-core prompt path (option "nvfp4_prefill") runs on this device
// out4: seconds spent queueing kernels, seconds waiting for the logits, steps, kernel launches (cleared).
// Returns the number of graphs recorded so far.
INK2_API int ink2_timing(void* h, double* out4);
INK2_API int ink2_num_visits(void* h);
INK2_API int ink2_visit_info(void* h, int visit, int* layer, int* pass);
INK2_API int ink2_get(void* h, int what, int visit, void* out, long long out_bytes);
INK2_API long long ink2_memory_used(void* h);  // bytes allocated by the engine on its backend
INK2_API int ink2_device_memory(void* h, long long* free_bytes, long long* total_bytes);
INK2_API int ink2_profile(void* h, int enable, double* seconds_out, int n);  // returns and clears the slots
// Timing of every kernel alone: with reps > 0 each kernel of the following steps runs reps times in a
// row (those steps give no valid result; reset afterwards). Returns the number of kernels; fills the
// seconds of one run and the launch counts per kernel since the last call.
INK2_API const char* ink2_kernel_name(int kid);
INK2_API int ink2_bench(void* h, int reps, double* seconds_out, long long* counts_out, int n);
// Debug capture: the buffers written by every operation of one visit (-1 = off).
INK2_API int ink2_capture(void* h, int visit);
INK2_API int ink2_capture_count(void* h);
INK2_API int ink2_capture_get(void* h, int i, char* name_out, int name_cap, float* data_out, long long max_floats);
// Device measurements that are independent of the model (launch cost, barrier cost); cuda build only.
INK2_API int ink2_microbench(char* report, int report_cap);
// Every kernel on random inputs, host emulation against the CUDA build.
// Returns the number of kernels over the tolerance (0 = pass), < 0 on error.
INK2_API int ink2_selftest(const char* backend, char* report, int report_cap);
// Allocated KV payload (including precision metadata and history tail) and attention workspace, in bytes.
INK2_API int ink2_kv_memory(void* h, long long* cache, long long* workspace);
