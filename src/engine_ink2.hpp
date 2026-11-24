// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace yskn {

struct Ink2Config {
  int hidden = 2304;
  int num_layers = 24;
  int vocab_padded = 201024;
  int vocab = 200058;
  int hq = 24;
  int hkv = 8;
  int head_dim = 128;
  int d_rel = 16;
  int rel_extent = 1024;
  int swa = 1024;
  int block_sz = 6;
  int core_start = 6;
  int core_end = 18;
  int num_passes = 3;
  int n_experts = 128;
  int top_k = 8;
  int moe_I = 1152;
  int gate_rank = 64;
  float rms_eps = 1e-6f;
  float qk_clip = 22.627416997969522f;
  float rel_clip = 8.0f;
  float log_floor = 8192.0f;
  float log_alpha = 0.1f;
  float route_scale = 8.0f;
  float logit_softcap = 30.0f;
  std::vector<bool> is_local;
  std::vector<int> schedule;  // length 48
};

struct Ink2Tensor {
  void* d_ptr = nullptr;
  int rows = 0;
  int cols = 0;
  bool f32 = false;
  size_t bytes = 0;
};

struct Ink2Ctrl {
  Ink2Tensor down;
  Ink2Tensor forget;
  Ink2Tensor gain;
};

struct Ink2Layer {
  bool is_swa = true;
  bool has_ctrl = false;
  bool is_mxfp8 = false;
  bool fused_qkvrsg = false;
  bool fused_qrsg = false;
  bool fused_kvp = false;
  const __nv_bfloat16* w_qkvrsg = nullptr;
  const __nv_bfloat16* w_kkp = nullptr;
  const __nv_bfloat16* w_vvp = nullptr;
  Ink2Tensor wq_du, wk_dv, wv_dv, wr_du, attn_sink_proj, gate_attn, wo_ud;
  Ink2Tensor q_scale, rel_scale, rel_proj;
  Ink2Tensor attn_norm;
  Ink2Tensor q_norm, k_norm, v_norm;
  Ink2Tensor k_sconv, v_sconv;
  Ink2Tensor k_static, k_proj, v_static, v_proj;
  Ink2Tensor attn_sconv, mlp_sconv, mlp_norm;
  Ink2Ctrl ctrl_attn, ctrl_k, ctrl_v, ctrl_mlp;
  Ink2Tensor router_w, router_bias;
  uint8_t* w13 = nullptr;
  uint8_t* w13_sc = nullptr;
  uint8_t* w2 = nullptr;
  uint8_t* w2_sc = nullptr;
  float* w13_outer = nullptr;
  float* w2_outer = nullptr;
};

struct Ink2MtpConfig {
  int hidden = 2304;
  int num_layers = 3;
  int hq = 24;
  int hkv = 8;
  int head_dim = 128;
  int ffn = 9216;
  int swa = 1024;
  int block = 8;
  int markov_rank = 256;
  float rope_theta = 1e6f;
  float rms_eps = 1e-6f;
  bool is_local[3] = {true, false, true};
  int taps[4] = {5, 23, 41, 48};
};

struct Ink2MtpLayer {
  bool is_local = true;
  Ink2Tensor dflare_norm, in_norm, q_norm, k_norm, k_norm_ctx;
  Ink2Tensor k_proj_ctx, v_proj_ctx, o_proj, ff_norm, down_proj;
  const __nv_bfloat16* w_kv_ctx = nullptr;
  const __nv_bfloat16* w_qkv = nullptr;
  const __nv_bfloat16* w_gu = nullptr;
  int qkv_rows = 0;
};

struct Ink2GenStats {
  int64_t produced = 0;
  int eos = 0;
  int cached = 0;
  int passes = 0;
  int64_t drafted = 0;
  int64_t accepted = 0;
  double prefill_ms = 0.0;
  double gen_ms = 0.0;
};

// CPU speculative acceptance helper (greedy exact & unbiased sampled rejection with independent residual salt).
// logits rows are post-softcap & post-suppress. Returns accepted draft count a in [0..K], and writes tokens[a]..
int ink2_accept_tokens(int32_t* out_accepted_and_next, const int32_t* draft_tokens, int K, int V,
                       const float* verify_logits, const float* draft_logits, const int32_t* verify_argmax,
                       float temp, int top_k, float top_p, float min_p, uint64_t seed, int64_t step);

class EngineInk2 {
 public:
  static constexpr int kPrefillChunk = 1024;
  static constexpr int kMtpBlock = 8;
  static constexpr int kMaxTree = 12;
  static constexpr int kNumTreeBuckets = 8;  // {2, 3, 4, 5, 6, 8, 10, 12}
  static constexpr int kDsparkKCand = 512;
  static constexpr int kBeamW = 4;
  static constexpr int kBeamPoolCap = 64;
  static constexpr int kScreenMaxCand = 8192;

  EngineInk2(const std::string& path, int max_ctx, int flags);
  ~EngineInk2();

  EngineInk2(const EngineInk2&) = delete;
  EngineInk2& operator=(const EngineInk2&) = delete;

  static bool is_inkling2_dir(const std::string& path);
  static std::string resolve_dir(const std::string& path);

  int vocab() const { return cfg_.vocab; }
  int max_ctx() const { return max_ctx_; }
  int position() const { return pos_; }

  void reset();
  int32_t prefill(const int32_t* ids, int n);
  void decode(int32_t* out, int steps);
  void logits(float* out);
  void set_opt(int key, int value);
  void set_suppress(const int32_t* token_ids, int n_tokens);
  void generate(const int32_t* prompt, int n_prompt, int n_gen, int eos1, int eos2,
                const std::atomic<int>* cancel, int32_t* out_ids, Ink2GenStats& st);

 private:
  struct TreeSpec {
    int M = 0;
    int max_depth = 0;
    int32_t tok[kMaxTree] = {};
    int8_t p1[kMaxTree] = {};
    int8_t p2[kMaxTree] = {};
    int8_t p3[kMaxTree] = {};
    uint8_t depth[kMaxTree] = {};
    uint32_t anc_mask[kMaxTree] = {};
  };

  void forward_tokens(const int32_t* ids, int T, int pos0, bool need_logits,
                      bool capture_taps = false, bool save_snaps = false, bool need_all_logits = false,
                      const int32_t* d_pos_ptr = nullptr, bool greedy_argmax_only = false);
  void forward_tree_verify(int M, int pos0, const int32_t* d_pos_ptr);
  void draft_beam_tree(int32_t anchor_tok, int pos0, int max_nodes, int room, TreeSpec& out_tree);
  void commit_mtp_context(int pos0, int m, const int32_t* d_pos_ptr, bool restore_conv);
  void mtp_draft_step(int pos0, int Kd, const int32_t* d_pos_ptr, bool greedy, int64_t step_pos);
  void mtp_verify_step(int pos0, int R, const int32_t* d_pos_ptr, bool greedy);
  void destroy_graphs();
  int32_t sample_logits_row(const float* h_row, int64_t step_pos) const;
  int32_t sample_logits(int64_t step_pos);
  void* alloc_gpu(size_t bytes);
  void free_all() noexcept;  // what the destructor frees (also when the constructor fails)

  std::string dir_;
  Ink2Config cfg_;
  int sm_count_ = 48;
  int max_ctx_ = 32768;
  int ring_rows_ = 0;
  int pos_ = 0;
  int32_t last_tok_ = -1;
  bool logits_valid_ = false;
  std::vector<int32_t> live_tokens_;

  // Sampling options (matching Engine35 option keys)
  float temp_ = 0.0f;
  float top_p_ = 0.95f;
  int top_k_ = 20;
  float min_p_ = 0.0f;
  uint64_t seed_ = 0;
  int sampler_mode_ = 3;

  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  std::vector<void*> gpu_allocs_;

  // Model weights
  Ink2Tensor embed_, embed_norm_, final_norm_, unembed_;
  bool use_q8_head_ = false;
  std::string head_mode_ = "auto";
  int8_t* d_unembed_q8_ = nullptr;
  __half* d_unembed_q8_sc_ = nullptr;
  int8_t* d_draft_u_q8_ = nullptr;
  __half* d_draft_u_sc_ = nullptr;
  bool use_screen_head_ = false;
  int8_t* d_screen_q8_ = nullptr;
  float* d_screen_sc_ = nullptr;
  float* d_screen_eps_ = nullptr;
  float* d_screen_hnorm_ = nullptr;
  int* d_screen_max_lb_ = nullptr;
  float* d_screen_dot_hat_ = nullptr;
  int32_t* d_screen_cand_ids_ = nullptr;
  int32_t* d_screen_cand_cnt_ = nullptr;
  std::vector<Ink2Tensor> attn_queries_, mlp_queries_;
  Ink2Tensor output_query_;
  float* d_msg_queries_ = nullptr;
  std::vector<Ink2Layer> layers_;

  // KV caches (FP32 default for exact parity & short-ctx speed; BF16 opt-in via YSKN_INK2_KV=bf16) & short-conv history across 48 visits
  bool kv_bf16_ = false;
  bool use_suffix_draft_ = true;
  std::vector<void*> d_kc_, d_vc_;
  float* d_conv_base_ = nullptr;
  std::vector<float*> d_conv_;
  float* d_verify_conv_ = nullptr;

  // Reusable activations
  int32_t* d_tokens_ = nullptr;
  int32_t* d_tok_out_ = nullptr;
  float* d_blk_v_ = nullptr;
  int32_t* d_blk_i_ = nullptr;
  __nv_bfloat16* d_x_bf16_ = nullptr;

  float *d_x_ = nullptr, *d_x_pre_ = nullptr, *d_inp_sa_ = nullptr, *d_ffn_inp_ = nullptr, *d_h_norm_ = nullptr;
  float* d_blocks_[6] = {};
  float* d_pre_blocks_[4] = {};
  float* d_tap_[3] = {};
  float *d_ar_out_ = nullptr, *d_msg_out_ = nullptr;
  float *d_mg_nhm_ = nullptr, *d_mg_g_ = nullptr, *d_mg_df_ = nullptr, *d_mg_dg_ = nullptr;
  __nv_bfloat16 *d_mg_nhm_bf16_ = nullptr, *d_mg_g_bf16_ = nullptr;
  float* d_qkvrsg_tmp_ = nullptr;
  int32_t* d_step_pos_ = nullptr;
  int32_t* d_commit_pos_ = nullptr;
  float *d_n0_ = nullptr, *d_n1_ = nullptr, *d_n2_ = nullptr;
  float *d_q_ = nullptr, *d_k_ = nullptr, *d_v_ = nullptr, *d_r_ = nullptr;
  float *d_sinks_ = nullptr, *d_gate_attn_ = nullptr, *d_att_out_ = nullptr, *d_a_out_ = nullptr;
  float *d_dyn_ = nullptr, *d_sconv_tmp_ = nullptr, *d_rb_ = nullptr;
  float *d_part_m_ = nullptr, *d_part_l_ = nullptr, *d_part_o_ = nullptr;
  float *d_h_mlp_ = nullptr, *d_x_mlp_ = nullptr, *d_rlog_ = nullptr, *d_ew_ = nullptr;
  uint32_t* d_eid_ = nullptr;
  float *d_moe_act_ = nullptr, *d_moe_pair_out_ = nullptr, *d_moe_out_ = nullptr;
  float *d_logits_ = nullptr, *d_verify_logits_ = nullptr;
  int32_t *d_moe_pair_idx_ = nullptr, *d_moe_tile_exp_ = nullptr, *d_moe_tile_m0_ = nullptr;
  int32_t *d_moe_tile_cnt_ = nullptr, *d_moe_num_tiles_ = nullptr;

  // Token suppression bitmask (byte-per-token on GPU, 200058 bytes) & pinned host staging for sampling/argmax
  uint8_t* d_suppress_ = nullptr;
  std::vector<uint8_t> h_suppress_;
  bool suppress_active_ = false;
  float* h_logits_ = nullptr;
  float* h_verify_logits_ = nullptr;
  int32_t* h_tok_ = nullptr;
  int32_t* d_verify_samples_ = nullptr;
  int32_t* h_pair_toks_ = nullptr;
  int32_t* h_tree_next_ = nullptr;

  // DSpark MTP drafter & W=4 Bayes-Optimal Beam Tree
  bool mtp_enabled_ = false;
  bool mtp_q8_ = true;
  bool use_tree_ = false;
  float moe_ratio_ = 12.0f;
  int tree_max_ = kMaxTree;
  Ink2MtpConfig mtp_cfg_;
  float mtp_fuse_[4][4] = {};
  Ink2Tensor mtp_mask_embed_, mtp_norm_, mtp_rec_norm_;
  Ink2Tensor mtp_markov_w1_, mtp_rnn_proj_w_, mtp_rnn_proj_b_, mtp_hidden_out_w_;
  float* d_mtp_ho_b_ = nullptr;
  const __nv_bfloat16* d_mtp_fold_w_ = nullptr;
  int8_t* d_mtp_fold_q_ = nullptr;
  __half* d_mtp_fold_s_ = nullptr;
  float* d_mtp_fold_b_ = nullptr;
  std::vector<Ink2MtpLayer> mtp_layers_;
  void *d_mtp_kc_[3] = {}, *d_mtp_vc_[3] = {};
  float *d_mtp_rec_feat_ = nullptr, *d_mtp_ctx_h_ = nullptr, *d_mtp_ctx_k_ = nullptr, *d_mtp_ctx_v_ = nullptr;
  float *d_mtp_blk_x_ = nullptr, *d_mtp_blk_res_ = nullptr, *d_mtp_blk_norm_ = nullptr;
  float *d_mtp_blk_q_ = nullptr, *d_mtp_blk_k_ = nullptr, *d_mtp_blk_v_ = nullptr, *d_mtp_blk_att_ = nullptr;
  float *d_mtp_blk_gate_ = nullptr, *d_mtp_blk_out_ = nullptr, *d_mtp_qkv_tmp_ = nullptr;
  float *d_mtp_state_ = nullptr, *d_mtp_latent_ = nullptr, *d_mtp_rnn_in_ = nullptr, *d_mtp_rnn_gates_ = nullptr;
  float *d_mtp_logits_ = nullptr;
  float* h_mtp_draft_logits_ = nullptr;

  // Beam-tree draft + staging + all-48-visit commit buffers
  int8_t* d_beam_h_q8_ = nullptr;
  float* d_beam_h_sc_ = nullptr;
  int32_t* d_beam_top_tmp_ids_ = nullptr;
  float* d_beam_top_tmp_vals_ = nullptr;
  int32_t* d_beam_cand_ids_ = nullptr;
  float* d_beam_cand_base_ = nullptr;
  float *d_beam_state_ = nullptr, *d_beam_state_next_ = nullptr;
  int32_t *d_beam_prev_tok_ = nullptr, *d_beam_node_ = nullptr;
  float* d_beam_logp_ = nullptr;
  float *d_beam_joint_ = nullptr, *d_beam_proj_ = nullptr, *d_beam_lat_ = nullptr;
  float *d_beam_corr_ = nullptr, *d_beam_cand_scores_ = nullptr;
  int32_t *d_pool_tok_ = nullptr, *d_pool_cnt_ = nullptr;
  int8_t* d_pool_parent_ = nullptr;
  uint8_t* d_pool_depth_ = nullptr;
  float* d_pool_logp_ = nullptr;
  int32_t* h_pool_tok_ = nullptr;
  int8_t* h_pool_parent_ = nullptr;
  uint8_t* h_pool_depth_ = nullptr;
  float* h_pool_logp_ = nullptr;
  int32_t* h_pool_cnt_ = nullptr;

  float *d_stage_k_ = nullptr, *d_stage_v_ = nullptr, *d_stage_sconv_ = nullptr;
  float* d_tap_scratch_ = nullptr;
  int8_t *d_tree_p1_ = nullptr, *d_tree_p2_ = nullptr, *d_tree_p3_ = nullptr;
  uint8_t* d_tree_depth_ = nullptr;
  uint32_t* d_tree_anc_mask_ = nullptr;
  int32_t* d_acc_path_ = nullptr;
  void **d_kc_ptrs_ = nullptr, **d_vc_ptrs_ = nullptr;
  int32_t* d_kv_cap_ = nullptr;
  uint8_t* d_kv_is_swa_ = nullptr;

  // CUDA Graphs:
  // - T=1 plain decode ([argmax_only][sup])
  // - linear MTP draft+verify ([K-1][sup])
  // - pipelined MTP context commit ([restore_conv][m-1])
  // - W=4 beam tree draft ([sup]) & tree verify ([bucket][sup])
  bool use_graph_ = true;
  cudaGraph_t decode_graph_[2][2] = {};
  cudaGraphExec_t decode_graph_exec_[2][2] = {};
  cudaGraph_t dv_graph_[7][2] = {};
  cudaGraphExec_t dv_graph_exec_[7][2] = {};
  cudaGraph_t commit_graph_[2][8] = {};
  cudaGraphExec_t commit_graph_exec_[2][8] = {};
  cudaGraph_t beam_draft_graph_[2] = {};
  cudaGraphExec_t beam_draft_graph_exec_[2] = {};
  cudaGraph_t tree_verify_graph_[kNumTreeBuckets][2] = {};
  cudaGraphExec_t tree_verify_graph_exec_[kNumTreeBuckets][2] = {};
};

}  // namespace yskn
