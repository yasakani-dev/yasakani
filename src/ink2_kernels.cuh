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

void launch_attn_res(const float* const* states, int M, const void* queries, bool q_f32, int Q, float* out,
                     int T, int H, float eps, float scale, cudaStream_t s);

void launch_msg_gate(const float* h, const float* m, const __nv_bfloat16* down_w,
                     const __nv_bfloat16* forget_w, const __nv_bfloat16* gain_w, float* out,
                     int T, int H, int R, float eps, cudaStream_t s);

// Linear Y [M, N] = X [M, K] @ W [N, K]^T (FP32 exact MV up to M<=32, cuBLAS BF16 GEMM for larger M)
void launch_linear(cublasHandle_t cublas, const float* x, const void* w, bool w_f32, __nv_bfloat16* x_bf16_scratch,
                   float* y, int M, int N, int K, bool add, cudaStream_t s);

void launch_sconv_static(const float* x, const void* w, bool w_f32, float* hist, float* out,
                         int T, int C, float scale, cudaStream_t s);

void launch_sconv_dynamic(const float* x, const float* dyn, const void* static_w, bool sw_f32, float* hist,
                          float* out, int T, int C, int n_heads, cudaStream_t s);

void launch_qkvr_prep(float* q, const float* k, const float* v, float* r,
                      const void* q_scale, const void* r_scale, bool sc_f32,
                      __half* kc, __half* vc, int T, int Hq, int Hkv, int D, int d_rel,
                      int pos0, int kv_cap, bool is_swa, float qk_clip, float r_clip,
                      float v_eps, float log_floor, float log_alpha, cudaStream_t s);

void launch_rel_bias(const float* r, const void* rel_proj, bool proj_f32, float* rb,
                     int T, int Hq, int d_rel, int extent, cudaStream_t s);

void launch_attn_decode(const float* q, const __half* kc, const __half* vc, const float* rb,
                        const float* sinks, const float* gate, float* part_m, float* part_l, float* part_o,
                        float* out, int pos, int kv_cap, int Hq, int Hkv, int D, int rel_extent,
                        bool is_swa, int swa, cudaStream_t s);

void launch_attn_prefill(const float* q, const __half* kc, const __half* vc, const float* rb,
                         const float* sinks, const float* gate, float* out, int T, int pos0, int kv_cap,
                         int Hq, int Hkv, int D, int rel_extent, bool is_swa, int swa, cudaStream_t s);

void launch_moe_route(const float* logits, const void* bias, bool bias_f32, uint32_t* eid, float* ew,
                      int T, int E, int K, float route_scale, cudaStream_t s);

void launch_moe_nvfp4(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc, const float* w13_outer,
                      const uint8_t* w2, const uint8_t* w2_sc, const float* w2_outer,
                      float* act_scratch, float* out, int T, int H, int I, int K_top, cudaStream_t s);

void launch_moe_mxfp8(const float* x, const uint32_t* eid, const float* ew,
                      const uint8_t* w13, const uint8_t* w13_sc,
                      const uint8_t* w2, const uint8_t* w2_sc,
                      float* act_scratch, float* out, int T, int H, int I, int K_top, cudaStream_t s);

void launch_add(const float* src, float* dst, int n, cudaStream_t s);
void launch_softcap(float* x, int n, float cap, cudaStream_t s);
void launch_suppress_mask(float* logits, int V, const uint8_t* mask_bits, cudaStream_t s);
void launch_argmax_one(const float* logits, int V, int32_t* d_tok, float* blk_v, int32_t* blk_i, cudaStream_t s);

}  // namespace yskn::ink2
