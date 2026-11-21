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

class EngineInk2 {
 public:
  static constexpr int kPrefillChunk = 512;

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
                const std::atomic<int>* cancel, int32_t* out_ids,
                int64_t& produced_out, int& eos_out, int& cached_out, int& passes_out,
                double& prefill_ms, double& gen_ms);

 private:
  void forward_tokens(const int32_t* ids, int T, int pos0, bool need_logits);
  int32_t sample_logits(int64_t step_pos);
  void* alloc_gpu(size_t bytes);

  std::string dir_;
  Ink2Config cfg_;
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
  std::vector<Ink2Tensor> attn_queries_, mlp_queries_;
  Ink2Tensor output_query_;
  float* d_msg_queries_ = nullptr;
  std::vector<Ink2Layer> layers_;

  // KV caches & short-conv history across 48 visits
  std::vector<__half*> d_kc_, d_vc_;
  std::vector<float*> d_conv_;

  // Reusable activations
  int32_t* d_tokens_ = nullptr;
  int32_t* d_tok_out_ = nullptr;
  float* d_blk_v_ = nullptr;
  int32_t* d_blk_i_ = nullptr;
  __nv_bfloat16* d_x_bf16_ = nullptr;

  float *d_x_ = nullptr, *d_x_pre_ = nullptr, *d_inp_sa_ = nullptr, *d_ffn_inp_ = nullptr, *d_h_norm_ = nullptr;
  float* d_blocks_[6] = {};
  float* d_pre_blocks_[4] = {};
  float *d_ar_out_ = nullptr, *d_mg0_ = nullptr, *d_mg1_ = nullptr, *d_mg2_ = nullptr;
  float *d_n0_ = nullptr, *d_n1_ = nullptr, *d_n2_ = nullptr;
  float *d_q_ = nullptr, *d_k_ = nullptr, *d_v_ = nullptr, *d_r_ = nullptr;
  float *d_sinks_ = nullptr, *d_gate_attn_ = nullptr, *d_att_out_ = nullptr, *d_a_out_ = nullptr;
  float *d_dyn_ = nullptr, *d_rb_ = nullptr, *d_part_m_ = nullptr, *d_part_l_ = nullptr, *d_part_o_ = nullptr;
  float *d_h_mlp_ = nullptr, *d_x_mlp_ = nullptr, *d_rlog_ = nullptr, *d_ew_ = nullptr;
  uint32_t* d_eid_ = nullptr;
  float *d_moe_act_ = nullptr, *d_moe_out_ = nullptr, *d_logits_ = nullptr;

  // Token suppression bitmask (byte-per-token on GPU, 200058 bytes) & pinned host staging for sampling/argmax
  uint8_t* d_suppress_ = nullptr;
  bool suppress_active_ = false;
  float* h_logits_ = nullptr;
  int32_t* h_tok_ = nullptr;
};

}  // namespace yskn
