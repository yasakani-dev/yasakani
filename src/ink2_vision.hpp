// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace yskn {

struct Ink2VisionConfig {
  int patch_size = 20;
  int merge_size = 2;
  int hidden_size = 1024;
  int num_hidden_layers = 23;
  int num_attention_heads = 16;
  int head_dim = 64;
  int intermediate_size = 4096;
  int posemb_grid_size = 32;
  int projection_hidden_dim = 4096;
  int text_hidden_size = 2304;
  bool projector_bias = false;
  float layer_norm_eps = 1e-5f;
  float rope_theta = 10000.0f;
};

struct Ink2VisionLayerWeights {
  const __nv_bfloat16* ln1_w = nullptr;
  const __nv_bfloat16* ln1_b = nullptr;
  const __nv_bfloat16* ln2_w = nullptr;
  const __nv_bfloat16* ln2_b = nullptr;
  const __nv_bfloat16* ls1_gamma = nullptr;
  const __nv_bfloat16* ls2_gamma = nullptr;
  const __nv_bfloat16* qkv_w = nullptr;      // packed [3*H, H]
  const __nv_bfloat16* qkv_b = nullptr;      // packed [3*H]
  const __nv_bfloat16* out_w = nullptr;      // [H, H]
  const __nv_bfloat16* out_b = nullptr;      // [H]
  const __nv_bfloat16* fc1_w = nullptr;      // [I, H]
  const __nv_bfloat16* fc1_b = nullptr;      // [I]
  const __nv_bfloat16* fc2_w = nullptr;      // [H, I]
  const __nv_bfloat16* fc2_b = nullptr;      // [H]
};

class Ink2VisionEncoder {
 public:
  static constexpr int kTargetBatchSeq = 8192;

  Ink2VisionEncoder(const std::string& dir, cudaStream_t stream, cublasHandle_t cublas);
  ~Ink2VisionEncoder();

  Ink2VisionEncoder(const Ink2VisionEncoder&) = delete;
  Ink2VisionEncoder& operator=(const Ink2VisionEncoder&) = delete;

  static bool has_vision_config(const std::string& dir);
  const Ink2VisionConfig& config() const { return cfg_; }

  // Encode one or more contiguous RGB uint8 frames [H_i, W_i, 3] on GPU into BF16 embeddings [n_tokens_tot, text_hidden_size].
  // out_bf16 must hold at least sum_i(ceil(H_i/40)*ceil(W_i/40)) * text_hidden_size uint16_t elements.
  // out_grids (optional, length 2*n_frames) receives [rows_i, cols_i] per frame.
  // out_n_tokens (optional) receives total image token count.
  void encode_rgb(const uint8_t* const* frames, const int32_t* widths, const int32_t* heights, int n_frames,
                  uint16_t* out_bf16, int32_t* out_grids, int32_t* out_n_tokens);

 private:
  void ensure_loaded();
  void ensure_scratch(int need_S, int need_pat, int need_img, size_t need_rgb_bytes, int need_frames);
  void free_scratch();
  void* alloc_weight(size_t bytes);
  void linear_f32(const float* x_f32, const __nv_bfloat16* w_bf16, float* y_f32, int M, int N, int K);

  std::string dir_;
  Ink2VisionConfig cfg_;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  bool loaded_ = false;
  bool fp32_gemm_ = true;

  std::vector<void*> weight_allocs_;
  const __nv_bfloat16* class_emb_ = nullptr;
  const __nv_bfloat16* cls_pos_emb_ = nullptr;
  const __nv_bfloat16* patch_emb_w_ = nullptr;
  const __nv_bfloat16* patch_emb_b_ = nullptr;
  const __nv_bfloat16* pos_emb_ = nullptr;
  const __nv_bfloat16* ln_pre_w_ = nullptr;
  const __nv_bfloat16* ln_pre_b_ = nullptr;
  std::vector<Ink2VisionLayerWeights> layers_;
  const __nv_bfloat16* proj_in_w_ = nullptr;
  const __nv_bfloat16* proj_in_b_ = nullptr;
  const __nv_bfloat16* proj_out_w_ = nullptr;
  const __nv_bfloat16* proj_out_b_ = nullptr;

  // Reusable scratch buffers
  int cap_S_ = 0;
  int cap_pat_ = 0;
  int cap_img_ = 0;
  int cap_frames_ = 0;
  size_t cap_rgb_bytes_ = 0;

  uint8_t* d_rgb_ = nullptr;
  void* d_desc_ = nullptr;
  float* d_hidden_ = nullptr;
  float* d_norm_ = nullptr;
  float* d_qkv_ = nullptr;
  float* d_q_nh_ = nullptr;
  float* d_k_nh_ = nullptr;
  float* d_v_nh_ = nullptr;
  float* d_proj_ = nullptr;
  float* d_mlp_ = nullptr;       // sized for max(cap_S * I, cap_pat * patch_in_dim, cap_img * proj_hidden_dim)
  float* d_merged_ = nullptr;    // sized cap_img * M*M*H
  float* d_out_f32_ = nullptr;   // sized cap_img * text_hidden_size
  __nv_bfloat16* d_out_bf16_ = nullptr;
  __nv_bfloat16* d_x_bf16_ = nullptr;
  float* d_w_f32_ = nullptr;
  size_t cap_w_f32_elems_ = 0;
  std::vector<uint8_t> h_rgb_stage_;
};

}  // namespace yskn
