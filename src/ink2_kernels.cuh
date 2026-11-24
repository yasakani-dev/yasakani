// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace yskn::ink2 {

void launch_embed_norm(const int32_t* tokens, int tok_off, const __nv_bfloat16* embed,
                       const __nv_bfloat16* norm_w, float* out, int T, int H, float eps, cudaStream_t s);

void launch_rmsnorm(const float* x, const void* w, bool w_f32, float* out, int T, int H, float eps,
                    cudaStream_t s);

void launch_rmsnorm_x3(const float* x0, const float* x1, const float* x2,
                       const void* w0, const void* w1, const void* w2, bool w_f32,
                       float* out0, float* out1, float* out2, int T, int H, float eps, cudaStream_t s);

void launch_attn_res(const float* const* states, int M, uint32_t norm_mask, const void* queries, bool q_f32, int Q,
                     float* out, int T, int H, float eps, float scale, cudaStream_t s,
                     const void* post_norm_w = nullptr, bool post_w_f32 = false);

void launch_msg_gate(cublasHandle_t cublas, const float* h, const float* m, const __nv_bfloat16* down_w,
                     const __nv_bfloat16* forget_w, const __nv_bfloat16* gain_w,
                     float* nhm_f32, __nv_bfloat16* nhm_bf16, float* g_f32, __nv_bfloat16* g_bf16,
                     float* df_f32, float* dg_f32, float* out, int T, int H, int R, float eps, cudaStream_t s);

struct MsgGateItem {
  const float* h;
  const float* m;
  const __nv_bfloat16* down_w;
  const __nv_bfloat16* forget_w;
  const __nv_bfloat16* gain_w;
  float* out;
};
void launch_msg_gate_x3(const MsgGateItem* items, float* nhm3_f32, float* g3_f32,
                        int T, int H, int R, float eps, cudaStream_t s);

// Linear Y [M, N] = X [M, K] @ W [N, K]^T (__ldg MV up to M<=32, cuBLAS BF16 GEMM for M>32)
void launch_linear(cublasHandle_t cublas, const float* x, const void* w, bool w_f32, __nv_bfloat16* x_bf16_scratch,
                   float* y, int M, int N, int K, bool add, cudaStream_t s);

void launch_quantize_q8_32(const __nv_bfloat16* w, int8_t* q, __half* sc, int N, int K, cudaStream_t s);
void launch_linear_q8_32(const float* x, const int8_t* w_q, const __half* w_sc, float* y, int M, int N, int K,
                         cudaStream_t s);

struct MultiLinearSeg {
  float* dst = nullptr;
  int rows = 0;
};
void launch_linear_multi(cublasHandle_t cublas, const float* x, const __nv_bfloat16* w,
                         __nv_bfloat16* x_bf16_scratch, float* tmp_y,
                         const MultiLinearSeg* segs, int n_segs, int M, int K, cudaStream_t s);

void launch_sconv_static(const float* x, const void* w, bool w_f32, float* hist, float* out,
                         int T, int C, float scale, cudaStream_t s,
                         const float* res_add = nullptr, float* pf_scratch = nullptr,
                         bool save_snap = false, float* snap = nullptr, int snap_stride = 0);

void launch_moe_reduce8_sconv_static(const float* pair_out, const void* w, bool w_f32, float* hist, float* out,
                                     int T, int C, int K_top, float scale, const float* res_add,
                                     bool save_snap, float* snap, int snap_stride, cudaStream_t s);

void launch_sconv_static_kv(const float* xk, const void* wk, bool wk_f32, float* hk,
                            const float* xv, const void* wv, bool wv_f32, float* hv,
                            int T, int C, float scale, cudaStream_t s, float* pf_scratch = nullptr,
                            bool save_snap = false, float* snap_k = nullptr, float* snap_v = nullptr,
                            int snap_stride = 0);

void launch_sconv_dynamic(const float* x, const float* dyn, const void* static_w, bool sw_f32, float* hist,
                          float* out, int T, int C, int n_heads, cudaStream_t s, float* pf_scratch = nullptr,
                          bool save_snap = false, float* snap = nullptr, int snap_stride = 0);

void launch_qkvr_prep(float* q, const float* k, const float* v, float* r,
                      const void* q_scale, const void* r_scale, bool sc_f32,
                      void* kc, void* vc, int T, int Hq, int Hkv, int D, int d_rel,
                      int pos0, int kv_cap, bool is_swa, float qk_clip, float r_clip,
                      float v_eps, float log_floor, float log_alpha,
                      const int32_t* d_pos_ptr, bool kv_bf16, cudaStream_t s);

void launch_rel_bias(cublasHandle_t cublas, const float* r, const void* rel_proj, bool proj_f32,
                     __nv_bfloat16* r_bf16_scratch, float* rb, int T, int Hq, int d_rel, int extent, cudaStream_t s);

void launch_attn_decode(const float* q, const void* kc, const void* vc, const float* rb,
                        const float* sinks, const float* gate, float* part_m, float* part_l, float* part_o,
                        float* out, int pos, int kv_cap, int Hq, int Hkv, int D, int rel_extent,
                        bool is_swa, int swa, const int32_t* d_pos_ptr, bool kv_bf16, cudaStream_t s);

void launch_attn_verify(const float* q, const void* kc, const void* vc, const float* rb,
                        const float* sinks, const float* gate, float* part_m, float* part_l, float* part_o,
                        float* out, int T, int pos0, int kv_cap, int Hq, int Hkv, int D, int rel_extent,
                        bool is_swa, int swa, const int32_t* d_pos_ptr, bool kv_bf16, cudaStream_t s);

void launch_attn_prefill(const float* q, const void* kc, const void* vc, const float* rb,
                         const float* sinks, const float* gate, float* out, int T, int pos0, int kv_cap,
                         int Hq, int Hkv, int D, int rel_extent, bool is_swa, int swa, bool kv_bf16, cudaStream_t s);

void launch_moe_route(const float* logits, const void* bias, bool bias_f32, uint32_t* eid, float* ew,
                      int T, int E, int K, float route_scale, cudaStream_t s);

struct MoeGroupBufs {
  int32_t* pair_idx;
  int32_t* tile_exp;
  int32_t* tile_m0;
  int32_t* tile_cnt;
  int32_t* num_tiles;
};

void launch_moe_route_bucket_small(const float* logits, const void* bias, bool bias_f32, uint32_t* eid, float* ew,
                                   const MoeGroupBufs* gbuf, int T, int E, int K, float route_scale, cudaStream_t s);

void launch_moe_nvfp4(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc, const float* w13_outer,
                      const uint8_t* w2, const uint8_t* w2_sc, const float* w2_outer,
                      float* act_scratch, float* pair_out, float* out, int T, int H, int I, int K_top,
                      const MoeGroupBufs* gbuf, int n_experts, cudaStream_t s,
                      bool run_reduce8 = true, bool pre_bucketed = false);

void launch_moe_mxfp8(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc,
                      const uint8_t* w2, const uint8_t* w2_sc,
                      float* act_scratch, float* pair_out, float* out, int T, int H, int I, int K_top,
                      const MoeGroupBufs* gbuf, int n_experts, cudaStream_t s,
                      bool run_reduce8 = true, bool pre_bucketed = false);

void launch_add(const float* src, float* dst, int n, cudaStream_t s);
void launch_add3(const float* a, const float* b, float* dst, int n, cudaStream_t s);
void launch_softcap(float* x, int n, float cap, cudaStream_t s);
void launch_suppress_mask(float* logits, int V, const uint8_t* mask_bits, cudaStream_t s);
void launch_suppress_mask_rows(float* logits, int R, int V, const uint8_t* mask_bits, cudaStream_t s);
void launch_argmax_one(const float* logits, int V, int32_t* d_tok, float* blk_v, int32_t* blk_i, cudaStream_t s);
void launch_argmax_rows(const float* logits, int R, int V, int32_t* d_toks, float* blk_v, int32_t* blk_i, cudaStream_t s);

// DSpark MTP speculative drafter kernels
void launch_mtp_fuse_norm(const float* t0, const float* t1, const float* t2, const float* t3, int row_off,
                          float w0, float w1, float w2, float w3,
                          const void* norm_w, bool norm_f32, float* out, int T, int H, float eps, cudaStream_t s);

void launch_mtp_prep_block(const int32_t* d_anchor_tok, const __nv_bfloat16* embed,
                           const __nv_bfloat16* embed_norm, const __nv_bfloat16* mask_embed,
                           float* out, int B, int H, float eps, cudaStream_t s);

void launch_mtp_qk_rope_cache(float* q, float* k, const float* v, const __nv_bfloat16* q_norm,
                               const __nv_bfloat16* k_norm, void* kc, void* vc,
                               int T, int Hq, int Hkv, int D, int pos0, const int32_t* d_pos_ptr,
                               int kv_cap, bool is_swa, bool draft_mode, float eps, float theta,
                               bool kv_bf16, cudaStream_t s);

void launch_mtp_attn(const float* q, const void* kc, const void* vc, const float* k_blk, const float* v_blk,
                     float* out, int B, int P, const int32_t* d_pos_ptr, int kv_cap, int Hq, int Hkv, int D,
                     bool is_swa, int swa, bool kv_bf16, cudaStream_t s);

void launch_mtp_gate_up_swiglu(const float* x, const __nv_bfloat16* gate_up_w, float* out,
                               int M, int F, int H, cudaStream_t s);

void launch_mtp_res_add_bf16(const float* res, const float* upd, float* out, int n, cudaStream_t s);

void launch_mtp_rnn_prep(const float* state, const __nv_bfloat16* markov_w1, const int32_t* d_tok_id,
                         const float* hidden, const float* rec_feat, float* rnn_in,
                         int rank, int H, bool first_step, cudaStream_t s);

void launch_mtp_rnn_update(const float* gates, float* state, float* latent, int rank, bool first_step, cudaStream_t s);

void launch_linear_bf16_bias(const float* x, const __nv_bfloat16* w, const __nv_bfloat16* bias,
                             float* y, int N, int K, cudaStream_t s);

void launch_mtp_fold_add_bf16(const float* x, const __nv_bfloat16* fold_w, const float* fold_b,
                              float* y, int V, int R, float softcap, cudaStream_t s);

void launch_mtp_fold_add_q8(const float* x, const int8_t* fold_q, const __half* fold_s, const float* fold_b,
                            float* y, int V, int R, float softcap, cudaStream_t s);

void launch_f32_to_bf16_buf(const float* src, __nv_bfloat16* dst, int n, cudaStream_t s);

// Screened exact BF16 unembed & L2-resident DSpark W=4 Markov beam tree (MIT-inspired by Niko1221/Strata)
void launch_build_unembed_screen_bf16(const __nv_bfloat16* W, int8_t* q8, float* sc, float* eps, int V, int H,
                                      cudaStream_t s);

void launch_screen_argmax_bf16(const __nv_bfloat16* W_bf16, const int8_t* q8, const float* sc, const float* eps,
                               const float* h, int M, int V, int H, const uint8_t* suppress, float* d_hnorm,
                               int* d_max_lb, float* d_dot_hat, int32_t* d_cand_ids, int32_t* d_cand_cnt,
                               int max_cand, float softcap, int32_t* d_out_argmax, cudaStream_t s);

void launch_screen_dot_multi(const int8_t* q8, const float* sc, const float* h, float* out_logits, int M, int V,
                             int H, cudaStream_t s);

void launch_dspark_screen_dp4a7(const int8_t* q8, const float* sc, const float* h7, int8_t* h_q8, float* h_sc,
                                const uint8_t* suppress, float* base_logits, int V, int H, cudaStream_t s);

void launch_dspark_top_cand(const float* base_logits, int V, int K_cand, int32_t* tmp_ids, float* tmp_vals,
                            int32_t* cand_ids, float* cand_base, cudaStream_t s);

void launch_dspark_beam_init(const int32_t* d_anchor_id, float* beam_state, int32_t* beam_prev_tok, float* beam_logp,
                             int32_t* beam_node, int32_t* pool_tok, int8_t* pool_parent, uint8_t* pool_depth,
                             float* pool_logp, int32_t* pool_cnt, cudaStream_t s);

void launch_dspark_markov_prep_batch(const int32_t* prev_toks, const float* states, const float* hidden,
                                     const float* rec_feat, const __nv_bfloat16* markov_w1, float* joint_out,
                                     int nw, int R, int H, cudaStream_t s);

void launch_dspark_markov_update_batch(const float* proj, const __nv_bfloat16* rp_b, const float* state_in,
                                       float* state_next, float* lat_out, int nw, int R, cudaStream_t s);

void launch_dspark_add_bias_h_batch(float* corr, const float* ho_bias, const float* hidden, int nw, int H,
                                    cudaStream_t s);

void launch_dspark_score_cand_dp4a_batch(const int8_t* q8, const float* sc, const float* corr,
                                         const int32_t* cand_ids, const uint8_t* suppress, int K_cand, int nw,
                                         float softcap, float* cand_scores, cudaStream_t s);

void launch_dspark_beam_expand(const float* cand_scores, const int32_t* cand_ids, const float* state_next,
                               int K_cand, int nw, int beam_w, int depth, float* state_out,
                               int32_t* beam_prev_tok, float* beam_logp, int32_t* beam_node, int32_t* pool_tok,
                               int8_t* pool_parent, uint8_t* pool_depth, float* pool_logp, int32_t* pool_cnt,
                               cudaStream_t s);

// Tree verify & 2-launch all-48-visit commit kernels
void launch_sconv_static_tree(const float* x, const void* w, bool w_f32, const float* hist, float* out,
                              float* stage_raw, int stage_stride, int stage_ch_off, const int8_t* p1,
                              const int8_t* p2, const int8_t* p3, int M, int C, float scale, const float* res_add,
                              cudaStream_t s);

void launch_moe_reduce8_sconv_static_tree(const float* pair_out, const void* w, bool w_f32, const float* hist,
                                          float* out, float* stage_raw, int stage_stride, int stage_ch_off,
                                          const int8_t* p1, const int8_t* p2, const int8_t* p3,
                                          int M, int C, int K_top, float scale, const float* res_add, cudaStream_t s);

void launch_sconv_static_kv_tree(float* xk, const void* wk, bool wk_f32, const float* hk,
                                 float* xv, const void* wv, bool wv_f32, const float* hv,
                                 float* stage_raw, int stage_stride, const int8_t* p1, const int8_t* p2,
                                 const int8_t* p3, int M, int C, float scale, cudaStream_t s);

void launch_sconv_dynamic_tree(const float* x, const float* dyn, const void* static_w, bool sw_f32,
                               const float* hist, float* out, float* stage_raw, int stage_stride,
                               int stage_ch_off, const int8_t* p1, const int8_t* p2, const int8_t* p3, int M, int C,
                               int n_heads, cudaStream_t s);

void launch_qkvr_prep_tree(float* q, const float* k, const float* v, float* r,
                           const void* q_scale, const void* r_scale, bool sc_f32,
                           float* stage_k, float* stage_v, int M, int Hq, int Hkv, int D, int d_rel,
                           int pos0, const int32_t* d_pos_ptr, const uint8_t* d_depth,
                           float qk_clip, float r_clip, float v_eps, float log_floor, float log_alpha,
                           bool is_swa, cudaStream_t s);

void launch_attn_tree(const float* q, const void* kc, const void* vc, const float* stage_k, const float* stage_v,
                      const float* rb, const float* sinks, const float* gate, float* part_m, float* part_l,
                      float* part_o, float* out, int M, int pos0, const int32_t* d_pos_ptr, const uint8_t* d_depth,
                      const uint32_t* d_anc_mask, int kv_cap, int Hq, int Hkv, int D, int rel_extent, bool is_swa,
                      int swa, bool kv_bf16, cudaStream_t s);

void launch_commit_sconv_all48(float* d_conv_base, const float* d_stage_sconv, const int32_t* d_path, int n_acc,
                               int nv, int tree_cap, int KVD, int H, cudaStream_t s);

void launch_commit_kv_all48(void* const* d_kc_ptrs, void* const* d_vc_ptrs, const int32_t* d_kv_cap,
                            const uint8_t* d_kv_is_swa, const float* d_stage_k, const float* d_stage_v,
                            const int32_t* d_path, int p0, int n_acc, int nv, int tree_cap, int Hkv, int D,
                            bool kv_bf16, cudaStream_t s);

void launch_compact_taps(float* t0, float* t1, float* t2, float* t3, float* scratch, const int32_t* d_path,
                         int n_acc, int H, cudaStream_t s);

// DFlash2 parallel block-diffusion drafter kernels
void launch_dflash_concat_taps(const float* t0, const float* t1, const float* t2, const float* t3,
                               int row_off, float* out_cat, int m, int H, cudaStream_t s);

void launch_dflash_prep_tokens(const int32_t* d_anchor_tok, int32_t* d_blk_toks,
                               int B, int32_t mask_token_id, cudaStream_t s);

void launch_dflash_dyn_conv(const float* in, const float* dyn, const __nv_bfloat16* base_kernel,
                            const float* residual, float* out, int B, int H, int group_size,
                            bool finish, cudaStream_t s);

void launch_dflash_attn(const float* q, const void* kc, const void* vc, const float* k_blk, const float* v_blk,
                        float* out, int B, int P, const int32_t* d_pos_ptr, int kv_cap, int Hq, int Hkv, int D,
                        int swa, bool kv_bf16, cudaStream_t s);

void launch_dflash_top16(const float* logits, int rows, int V, float* tmp_vals, int32_t* tmp_ids,
                         float* out_vals, int32_t* out_ids, cudaStream_t s);

void launch_dflash_edge_table(const int32_t* d_anchor_tok, const int32_t* cand_ids, const float* sel_h,
                              const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb,
                              float* d_edges, int rows, int K_sel, int R, cudaStream_t s);

void launch_dflash_select_chain_greedy(const int32_t* d_anchor_tok, const int32_t* cand_ids, const float* cand_unary,
                                       const float* sel_h, const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb,
                                       int32_t* d_out_toks, int Kd, int K_sel, int R,
                                       float softcap, bool after_correction, cudaStream_t s);

}  // namespace yskn::ink2
