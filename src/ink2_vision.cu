// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "ink2_vision.hpp"

#include "safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace yskn {
namespace {

constexpr int64_t kMaxVisionPixels = 40000000LL;
constexpr int kMaxVisionSide = 32768;

struct VisFrameDesc {
  int32_t w;
  int32_t h;
  int32_t rows;
  int32_t cols;
  int32_t seq_off;
  int32_t pat_off;
  int32_t img_off;
  uint64_t rgb_off;
};

inline void vis_cuda_ok(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string("ink2_vision: ") + what + ": " + cudaGetErrorString(err));
}

std::string slurp_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("ink2_vision: cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

Ink2VisionConfig parse_vision_config(const std::string& dir) {
  const stjson::Val root = stjson::parse(slurp_text(dir + "/config.json"));
  if (!root.has("vision_config")) throw std::runtime_error("ink2_vision: config.json missing vision_config");
  const stjson::Val& v = root.at("vision_config");
  auto gi = [](const stjson::Val& val, int def) { return val.kind == stjson::Val::Num ? int(val.i) : def; };
  auto gf = [](const stjson::Val& val, float def) { return val.kind == stjson::Val::Num ? float(val.num) : def; };
  auto gb = [](const stjson::Val& val, bool def) { return val.kind == stjson::Val::Bool ? val.b : def; };

  Ink2VisionConfig c;
  c.patch_size = v.has("native_patch_size") ? gi(v.at("native_patch_size"), 20) : gi(v.at("patch_size"), 20);
  c.merge_size = v.has("adapter_merge_size") ? gi(v.at("adapter_merge_size"), 2) : gi(v.at("spatial_merge_size"), 2);
  c.hidden_size = gi(v.at("hidden_size"), 1024);
  c.num_hidden_layers = gi(v.at("num_hidden_layers"), 23);
  c.num_attention_heads = gi(v.at("num_attention_heads"), 16);
  if (c.num_attention_heads <= 0 || c.hidden_size % c.num_attention_heads != 0)
    throw std::runtime_error("ink2_vision: invalid hidden_size / num_attention_heads");
  c.head_dim = c.hidden_size / c.num_attention_heads;
  if (c.head_dim != 64) throw std::runtime_error("ink2_vision: unsupported vision head_dim (expected 64)");
  if (v.has("intermediate_size") && v.at("intermediate_size").kind == stjson::Val::Num) {
    c.intermediate_size = int(v.at("intermediate_size").i);
  } else {
    c.intermediate_size = int(std::llround(double(c.hidden_size) * double(gf(v.at("mlp_ratio"), 4.0f))));
  }
  c.posemb_grid_size = gi(v.at("posemb_grid_size"), 32);
  c.projection_hidden_dim = gi(v.at("projection_hidden_dim"), 4096);
  c.text_hidden_size = gi(v.at("text_hidden_size"), 2304);
  c.projector_bias = gb(v.at("projector_bias"), false);
  c.layer_norm_eps = gf(v.at("layer_norm_eps"), 1e-5f);
  if (v.has("rope_parameters") && v.at("rope_parameters").has("rope_theta")) {
    c.rope_theta = gf(v.at("rope_parameters").at("rope_theta"), 10000.0f);
  }
  return c;
}

__device__ __forceinline__ float vis_warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float vis_warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float vis_block_sum(float v, float* sh) {
  const uint32_t tid = threadIdx.x;
  const uint32_t sg = tid >> 5;
  const uint32_t lane = tid & 31;
  const uint32_t nsg = (blockDim.x + 31) >> 5;
  v = vis_warp_sum(v);
  if (lane == 0) sh[sg] = v;
  __syncthreads();
  if (sg == 0) {
    float s = (lane < nsg) ? sh[lane] : 0.0f;
    s = vis_warp_sum(s);
    if (lane == 0) sh[0] = s;
  }
  __syncthreads();
  const float res = sh[0];
  __syncthreads();
  return res;
}

__device__ __forceinline__ int find_frame_by_pat(const VisFrameDesc* desc, int n_frames, int p_global) {
  int lo = 0, hi = n_frames - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (desc[mid].pat_off <= p_global) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ int find_frame_by_seq(const VisFrameDesc* desc, int n_frames, int s_global) {
  int lo = 0, hi = n_frames - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (desc[mid].seq_off <= s_global) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ int find_frame_by_img(const VisFrameDesc* desc, int n_frames, int img_global) {
  int lo = 0, hi = n_frames - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (desc[mid].img_off <= img_global) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// Block-major native patch extraction + [-1 on 0..255] padding + rescale 1/255 + normalize mean=0.5 std=0.5
__global__ void k_vis_extract_patches(const uint8_t* rgb, const VisFrameDesc* desc, int n_frames,
                                      int P, int M, float* out) {
  const int p_global = blockIdx.x;
  const int tid = threadIdx.x;
  const int b = find_frame_by_pat(desc, n_frames, p_global);
  const VisFrameDesc f = desc[b];
  const int p = p_global - f.pat_off;
  const int mm = M * M;
  const int tok = p / mm;
  const int sub = p % mm;
  const int br = tok / f.cols;
  const int bc = tok % f.cols;
  const int ir = sub / M;
  const int ic = sub % M;
  const int r = br * M + ir;
  const int c = bc * M + ic;
  const int in_dim = P * P * 3;
  float* dst = out + size_t(p_global) * size_t(in_dim);
  const uint8_t* frgb = rgb + f.rgb_off;
  const float rescale = float(1.0 / 255.0);
  for (int k = tid; k < in_dim; k += blockDim.x) {
    const int ch = k % 3;
    const int px = (k / 3) % P;
    const int py = (k / 3) / P;
    const int y = r * P + py;
    const int x = c * P + px;
    const float raw = (y < f.h && x < f.w) ? float(frgb[(size_t(y) * size_t(f.w) + size_t(x)) * 3 + ch]) : -1.0f;
    dst[k] = (raw * rescale - 0.5f) / 0.5f;
  }
}

// Bilinear (align_corners=False) pos_emb interpolation + [CLS] prepend
__global__ void k_vis_pos_cls(const float* patch_out, const __nv_bfloat16* patch_bias,
                              const __nv_bfloat16* class_emb, const __nv_bfloat16* cls_pos_emb,
                              const __nv_bfloat16* pos_emb, const VisFrameDesc* desc, int n_frames,
                              int H, int M, int G, float* hidden_out) {
  const int s_global = blockIdx.x;
  const int tid = threadIdx.x;
  const int b = find_frame_by_seq(desc, n_frames, s_global);
  const VisFrameDesc f = desc[b];
  const int s_local = s_global - f.seq_off;
  float* dst = hidden_out + size_t(s_global) * size_t(H);
  if (s_local == 0) {
    for (int d = tid; d < H; d += blockDim.x) {
      dst[d] = __bfloat162float(class_emb[d]) + __bfloat162float(cls_pos_emb[d]);
    }
    return;
  }
  const int p_local = s_local - 1;
  const int p_global = f.pat_off + p_local;
  const int mm = M * M;
  const int tok = p_local / mm;
  const int sub = p_local % mm;
  const int r = (tok / f.cols) * M + (sub / M);
  const int c = (tok % f.cols) * M + (sub % M);
  const int gh = f.rows * M;
  const int gw = f.cols * M;

  const float src_h = (float(r) + 0.5f) * float(G) / float(gh) - 0.5f;
  const float fh = floorf(src_h);
  const int ih = int(fh);
  const int h0 = max(0, min(G - 1, ih));
  const int h1 = max(0, min(G - 1, ih + 1));
  const float wh0 = fmaxf(0.0f, 1.0f - fabsf(src_h - fh));
  const float wh1 = fmaxf(0.0f, 1.0f - fabsf(src_h - fh - 1.0f));

  const float src_w = (float(c) + 0.5f) * float(G) / float(gw) - 0.5f;
  const float fw = floorf(src_w);
  const int iw = int(fw);
  const int w0 = max(0, min(G - 1, iw));
  const int w1 = max(0, min(G - 1, iw + 1));
  const float ww0 = fmaxf(0.0f, 1.0f - fabsf(src_w - fw));
  const float ww1 = fmaxf(0.0f, 1.0f - fabsf(src_w - fw - 1.0f));

  const float c00 = wh0 * ww0, c01 = wh0 * ww1, c10 = wh1 * ww0, c11 = wh1 * ww1;
  const __nv_bfloat16* p00 = pos_emb + size_t(h0 * G + w0) * size_t(H);
  const __nv_bfloat16* p01 = pos_emb + size_t(h0 * G + w1) * size_t(H);
  const __nv_bfloat16* p10 = pos_emb + size_t(h1 * G + w0) * size_t(H);
  const __nv_bfloat16* p11 = pos_emb + size_t(h1 * G + w1) * size_t(H);
  const float* src = patch_out + size_t(p_global) * size_t(H);
  for (int d = tid; d < H; d += blockDim.x) {
    const float pb = patch_bias ? __bfloat162float(patch_bias[d]) : 0.0f;
    const float pe = __bfloat162float(p00[d]) * c00 + __bfloat162float(p01[d]) * c01 +
                     __bfloat162float(p10[d]) * c10 + __bfloat162float(p11[d]) * c11;
    dst[d] = src[d] + pb + pe;
  }
}

// Standard FP32 LayerNorm (H == 1024 fast register path + generic fallback)
__global__ void k_vis_layernorm(const float* x, const __nv_bfloat16* w, const __nv_bfloat16* b,
                                float* out, int H, float eps) {
  __shared__ float sh[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* src = x + size_t(row) * size_t(H);
  float* dst = out + size_t(row) * size_t(H);
  if (H == 1024 && blockDim.x == 256) {
    const float4 v = reinterpret_cast<const float4*>(src)[tid];
    const float mean = vis_block_sum(v.x + v.y + v.z + v.w, sh) * (1.0f / 1024.0f);
    const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
    const float var = vis_block_sum(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3, sh) * (1.0f / 1024.0f);
    const float inv = rsqrtf(var + eps);
    const uint2 wu = reinterpret_cast<const uint2*>(w)[tid];
    const uint2 bu = reinterpret_cast<const uint2*>(b)[tid];
    const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wu);
    const __nv_bfloat16* bb = reinterpret_cast<const __nv_bfloat16*>(&bu);
    reinterpret_cast<float4*>(dst)[tid] = make_float4(
        d0 * inv * __bfloat162float(wb[0]) + __bfloat162float(bb[0]),
        d1 * inv * __bfloat162float(wb[1]) + __bfloat162float(bb[1]),
        d2 * inv * __bfloat162float(wb[2]) + __bfloat162float(bb[2]),
        d3 * inv * __bfloat162float(wb[3]) + __bfloat162float(bb[3]));
    return;
  }
  float sum = 0.0f;
  for (int c = tid; c < H; c += blockDim.x) sum += src[c];
  const float mean = vis_block_sum(sum, sh) / float(H);
  float vsum = 0.0f;
  for (int c = tid; c < H; c += blockDim.x) {
    const float df = src[c] - mean;
    vsum += df * df;
  }
  const float inv = rsqrtf(vis_block_sum(vsum, sh) / float(H) + eps);
  for (int c = tid; c < H; c += blockDim.x) {
    dst[c] = (src[c] - mean) * inv * __bfloat162float(w[c]) + __bfloat162float(b[c]);
  }
}

// QKV bias add + 2D axial RoPE (CLS=(0,0), patches 1-indexed, W then H interleaved pairs) + transpose to [NH, S, HD]
__global__ void k_vis_qkv_bias_rope(const float* qkv, const __nv_bfloat16* qkv_b,
                                    const VisFrameDesc* desc, int n_frames, int S_tot,
                                    int H, int NH, int HD, int M, float rope_theta,
                                    float* q_nh, float* k_nh, float* v_nh) {
  __shared__ int sh_coords[2];
  const int s = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid == 0) {
    const int b = find_frame_by_seq(desc, n_frames, s);
    const VisFrameDesc f = desc[b];
    const int s_local = s - f.seq_off;
    if (s_local == 0) {
      sh_coords[0] = 0;
      sh_coords[1] = 0;
    } else {
      const int p_local = s_local - 1;
      const int mm = M * M;
      const int tok = p_local / mm;
      const int sub = p_local % mm;
      sh_coords[0] = (tok / f.cols) * M + (sub / M) + 1;
      sh_coords[1] = (tok % f.cols) * M + (sub % M) + 1;
    }
  }
  __syncthreads();

  const int h_coord = sh_coords[0];
  const int w_coord = sh_coords[1];
  const int h = tid >> 5;
  const int pair = tid & 31;
  if (h >= NH) return;

  const int col0 = h * HD + 2 * pair;
  const int col1 = col0 + 1;
  const float* row = qkv + size_t(s) * size_t(3 * H);

  const float q0 = row[col0] + __bfloat162float(qkv_b[col0]);
  const float q1 = row[col1] + __bfloat162float(qkv_b[col1]);
  const float k0 = row[H + col0] + __bfloat162float(qkv_b[H + col0]);
  const float k1 = row[H + col1] + __bfloat162float(qkv_b[H + col1]);
  const float v0 = row[2 * H + col0] + __bfloat162float(qkv_b[2 * H + col0]);
  const float v1 = row[2 * H + col1] + __bfloat162float(qkv_b[2 * H + col1]);

  const int half_pairs = HD >> 2;  // 16
  const int coord = (pair < half_pairs) ? w_coord : h_coord;
  const int j = (pair < half_pairs) ? pair : (pair - half_pairs);
  const float inv_freq = 1.0f / powf(rope_theta, float(2 * j) / float(HD >> 1));
  float s_val = 0.0f, c_val = 1.0f;
  sincosf(float(coord) * inv_freq, &s_val, &c_val);

  const size_t nh_off = (size_t(h) * size_t(S_tot) + size_t(s)) * size_t(HD) + size_t(2 * pair);
  q_nh[nh_off] = q0 * c_val - q1 * s_val;
  q_nh[nh_off + 1] = q1 * c_val + q0 * s_val;
  k_nh[nh_off] = k0 * c_val - k1 * s_val;
  k_nh[nh_off + 1] = k1 * c_val + k0 * s_val;
  v_nh[nh_off] = v0;
  v_nh[nh_off + 1] = v1;
}

// Tiled bidirectional FP32 FlashAttention for HD=64 (4 warps/CTA, BK=32 keys, 0 smem bank conflicts)
__global__ void k_vis_flash_attn_d64(const float* q_nh, const float* k_nh, const float* v_nh,
                                     const VisFrameDesc* desc, int S_tot, int NH, int M,
                                     float scale, float* out) {
  constexpr int BQ = 4;
  constexpr int BK = 32;
  __shared__ float K_smem[BK][65];
  __shared__ float V_smem[BK][65];

  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const VisFrameDesc f = desc[b];
  const int s0 = f.seq_off;
  const int slen = 1 + M * M * f.rows * f.cols;
  const int q_start = q_tile * BQ;
  if (q_start >= slen) return;

  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int q_local = q_start + w;
  const bool q_valid = (q_local < slen);

  float q0 = 0.0f, q1 = 0.0f;
  if (q_valid) {
    const size_t q_off = (size_t(h) * size_t(S_tot) + size_t(s0 + q_local)) * 64u;
    q0 = q_nh[q_off + lane];
    q1 = q_nh[q_off + lane + 32];
  }

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o0 = 0.0f, o1 = 0.0f;

  const size_t kv_head_base = size_t(h) * size_t(S_tot) * 64u;
  for (int k0 = 0; k0 < slen; k0 += BK) {
    const int tile_len = min(BK, slen - k0);
    for (int idx = tid; idx < BK * 64; idx += 128) {
      const int kr = idx >> 6;
      const int d = idx & 63;
      if (kr < tile_len) {
        const size_t kv_off = kv_head_base + size_t(s0 + k0 + kr) * 64u + size_t(d);
        K_smem[kr][d] = k_nh[kv_off];
        V_smem[kr][d] = v_nh[kv_off];
      } else {
        K_smem[kr][d] = 0.0f;
        V_smem[kr][d] = 0.0f;
      }
    }
    __syncthreads();

    float my_s = -INFINITY;
    for (int k = 0; k < tile_len; ++k) {
      const float prod = q0 * K_smem[k][lane] + q1 * K_smem[k][lane + 32];
      const float dot = vis_warp_sum(prod);
      if (lane == k) my_s = dot * scale;
    }

    const float m_tile = vis_warp_max(my_s);
    const float m_new = fmaxf(m_i, m_tile);
    const float alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
    const float p = (lane < tile_len) ? expf(my_s - m_new) : 0.0f;
    const float l_tile = vis_warp_sum(p);

    o0 *= alpha;
    o1 *= alpha;
    l_i = l_i * alpha + l_tile;
    m_i = m_new;

    for (int k = 0; k < tile_len; ++k) {
      const float p_k = __shfl_sync(0xffffffffu, p, k);
      o0 += p_k * V_smem[k][lane];
      o1 += p_k * V_smem[k][lane + 32];
    }
    __syncthreads();
  }

  if (q_valid) {
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    const size_t out_off = size_t(s0 + q_local) * size_t(NH * 64) + size_t(h) * 64u;
    out[out_off + lane] = o0 * inv_l;
    out[out_off + lane + 32] = o1 * inv_l;
  }
}

__global__ void k_vis_bias_layerscale_add(const float* upd, const __nv_bfloat16* bias,
                                          const __nv_bfloat16* ls_gamma, float* hidden, int S, int H) {
  const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  const size_t tot = size_t(S) * size_t(H);
  if (idx >= tot) return;
  const int c = int(idx % size_t(H));
  const float u = upd[idx] + __bfloat162float(bias[c]);
  hidden[idx] += u * __bfloat162float(ls_gamma[c]);
}

__global__ void k_vis_bias_gelu(float* x, const __nv_bfloat16* bias, int M, int N) {
  const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  const size_t tot = size_t(M) * size_t(N);
  if (idx >= tot) return;
  float v = x[idx];
  if (bias) v += __bfloat162float(bias[idx % size_t(N)]);
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  x[idx] = 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

__global__ void k_vis_add_bias(float* x, const __nv_bfloat16* bias, int M, int N) {
  const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  const size_t tot = size_t(M) * size_t(N);
  if (idx >= tot) return;
  x[idx] += __bfloat162float(bias[idx % size_t(N)]);
}

// Drop [CLS] row per frame and pack 2x2 merged block-major patch embeddings [n_img_tot, M*M*H]
__global__ void k_vis_drop_cls(const float* hidden, const VisFrameDesc* desc, int n_frames,
                               int H, int M, float* merged) {
  const int img_global = blockIdx.x;
  const int tid = threadIdx.x;
  const int b = find_frame_by_img(desc, n_frames, img_global);
  const VisFrameDesc f = desc[b];
  const int img_local = img_global - f.img_off;
  const int mm = M * M;
  const int src_row0 = f.seq_off + 1 + img_local * mm;
  const int merge_dim = mm * H;
  const float4* src4 = reinterpret_cast<const float4*>(hidden + size_t(src_row0) * size_t(H));
  float4* dst4 = reinterpret_cast<float4*>(merged + size_t(img_global) * size_t(merge_dim));
  const int n4 = merge_dim >> 2;
  for (int i = tid; i < n4; i += blockDim.x) dst4[i] = src4[i];
}

__global__ void k_vis_f32_to_bf16_rn(const float* src, __nv_bfloat16* dst, size_t n) {
  const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  if (idx < n) dst[idx] = __float2bfloat16_rn(src[idx]);
}

__global__ void k_vis_bf16_to_f32(const __nv_bfloat16* src, float* dst, size_t n) {
  const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  if (idx < n) dst[idx] = __bfloat162float(src[idx]);
}

}  // namespace

bool Ink2VisionEncoder::has_vision_config(const std::string& dir) {
  namespace fs = std::filesystem;
  try {
    const std::string cfg = dir + "/config.json";
    if (!fs::exists(cfg)) return false;
    const stjson::Val root = stjson::parse(slurp_text(cfg));
    return root.has("vision_config") && root.at("vision_config").kind == stjson::Val::Obj;
  } catch (...) {
    return false;
  }
}

Ink2VisionEncoder::Ink2VisionEncoder(const std::string& dir, cudaStream_t stream, cublasHandle_t cublas)
    : dir_(dir), stream_(stream), cublas_(cublas) {
  cfg_ = parse_vision_config(dir_);
  if (const char* env = std::getenv("YSKN_INK2_VISION_FP32_GEMM")) {
    fp32_gemm_ = (std::strcmp(env, "0") != 0);
  }
}

Ink2VisionEncoder::~Ink2VisionEncoder() {
  if (stream_) cudaStreamSynchronize(stream_);
  free_scratch();
  for (void* p : weight_allocs_) cudaFree(p);
  weight_allocs_.clear();
}

void* Ink2VisionEncoder::alloc_weight(size_t bytes) {
  void* ptr = nullptr;
  vis_cuda_ok(cudaMalloc(&ptr, std::max<size_t>(bytes, 16)), "cudaMalloc weight");
  weight_allocs_.push_back(ptr);
  return ptr;
}

void Ink2VisionEncoder::free_scratch() {
  auto cf = [](auto*& p) {
    if (p) {
      cudaFree(p);
      p = nullptr;
    }
  };
  cf(d_rgb_);
  cf(d_desc_);
  cf(d_hidden_);
  cf(d_norm_);
  cf(d_qkv_);
  cf(d_q_nh_);
  cf(d_k_nh_);
  cf(d_v_nh_);
  cf(d_proj_);
  cf(d_mlp_);
  cf(d_merged_);
  cf(d_out_f32_);
  cf(d_out_bf16_);
  cf(d_x_bf16_);
  cf(d_w_f32_);
  cap_S_ = cap_pat_ = cap_img_ = cap_frames_ = 0;
  cap_rgb_bytes_ = cap_w_f32_elems_ = 0;
}

void Ink2VisionEncoder::ensure_loaded() {
  if (loaded_) return;
  SafetensorsReader reader;
  reader.open_dir(dir_, "model.visual.");

  auto upload_bf16 = [&](const std::string& name) -> const __nv_bfloat16* {
    const StTensorInfo& info = reader.get(name);
    if (info.dtype != "BF16") throw std::runtime_error("ink2_vision: expected BF16 tensor for " + name);
    void* p = alloc_weight(info.nbytes);
    reader.upload_gpu(info, p, stream_);
    return static_cast<const __nv_bfloat16*>(p);
  };

  auto try_upload_bf16 = [&](const std::string& name) -> const __nv_bfloat16* {
    try {
      const StTensorInfo& info = reader.get(name);
      if (info.dtype != "BF16") throw std::runtime_error("ink2_vision: expected BF16 tensor for " + name);
      void* p = alloc_weight(info.nbytes);
      reader.upload_gpu(info, p, stream_);
      return static_cast<const __nv_bfloat16*>(p);
    } catch (const std::exception&) {
      return nullptr;
    }
  };

  auto pack_bf16_3 = [&](const std::string& n0, const std::string& n1, const std::string& n2) -> const __nv_bfloat16* {
    const StTensorInfo& i0 = reader.get(n0);
    const StTensorInfo& i1 = reader.get(n1);
    const StTensorInfo& i2 = reader.get(n2);
    if (i0.dtype != "BF16" || i1.dtype != "BF16" || i2.dtype != "BF16")
      throw std::runtime_error("ink2_vision: expected BF16 QKV tensor");
    uint8_t* p = static_cast<uint8_t*>(alloc_weight(i0.nbytes + i1.nbytes + i2.nbytes));
    reader.upload_gpu(i0, p, stream_);
    reader.upload_gpu(i1, p + i0.nbytes, stream_);
    reader.upload_gpu(i2, p + i0.nbytes + i1.nbytes, stream_);
    return reinterpret_cast<const __nv_bfloat16*>(p);
  };

  const std::string emb = "model.visual.backbone.embeddings.";
  class_emb_ = upload_bf16(emb + "class_embedding");
  cls_pos_emb_ = upload_bf16(emb + "cls_position_embedding");
  patch_emb_w_ = upload_bf16(emb + "patch_embedding.weight");
  patch_emb_b_ = try_upload_bf16(emb + "patch_embedding.bias");
  pos_emb_ = upload_bf16(emb + "position_embedding");
  ln_pre_w_ = upload_bf16("model.visual.backbone.ln_pre.weight");
  ln_pre_b_ = upload_bf16("model.visual.backbone.ln_pre.bias");

  layers_.resize(size_t(cfg_.num_hidden_layers));
  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    Ink2VisionLayerWeights& L = layers_[size_t(i)];
    const std::string p = "model.visual.backbone.layers." + std::to_string(i) + ".";
    L.ln1_w = upload_bf16(p + "ln_1.weight");
    L.ln1_b = upload_bf16(p + "ln_1.bias");
    L.ln2_w = upload_bf16(p + "ln_2.weight");
    L.ln2_b = upload_bf16(p + "ln_2.bias");
    L.ls1_gamma = upload_bf16(p + "ls_1.gamma");
    L.ls2_gamma = upload_bf16(p + "ls_2.gamma");
    L.qkv_w = pack_bf16_3(p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight");
    L.qkv_b = pack_bf16_3(p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias", p + "self_attn.v_proj.bias");
    L.out_w = upload_bf16(p + "self_attn.out_proj.weight");
    L.out_b = upload_bf16(p + "self_attn.out_proj.bias");
    L.fc1_w = upload_bf16(p + "mlp.c_fc.weight");
    L.fc1_b = upload_bf16(p + "mlp.c_fc.bias");
    L.fc2_w = upload_bf16(p + "mlp.c_proj.weight");
    L.fc2_b = upload_bf16(p + "mlp.c_proj.bias");
  }

  proj_in_w_ = upload_bf16("model.visual.proj_in.weight");
  proj_out_w_ = upload_bf16("model.visual.proj_out.weight");
  if (cfg_.projector_bias) {
    proj_in_b_ = upload_bf16("model.visual.proj_in.bias");
    proj_out_b_ = upload_bf16("model.visual.proj_out.bias");
  }
  reader.close_all();
  vis_cuda_ok(cudaStreamSynchronize(stream_), "sync weights");
  loaded_ = true;
}

void Ink2VisionEncoder::ensure_scratch(int need_S, int need_pat, int need_img, size_t need_rgb_bytes, int need_frames) {
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  const int patch_in_dim = cfg_.patch_size * cfg_.patch_size * 3;
  const int merge_dim = cfg_.merge_size * cfg_.merge_size * H;

  const bool need_seq_realloc = (need_S > cap_S_ || need_pat > cap_pat_ || need_img > cap_img_);
  if (need_seq_realloc) {
    const int new_S = std::max(cap_S_, need_S);
    const int new_pat = std::max(cap_pat_, need_pat);
    const int new_img = std::max(cap_img_, need_img);
    auto realloc_buf = [&](auto*& ptr, size_t bytes, const char* what) {
      if (ptr) cudaFree(ptr);
      ptr = nullptr;
      vis_cuda_ok(cudaMalloc(&ptr, std::max<size_t>(bytes, 16)), what);
    };
    realloc_buf(d_hidden_, size_t(new_S) * size_t(H) * sizeof(float), "alloc d_hidden");
    realloc_buf(d_norm_, size_t(new_S) * size_t(H) * sizeof(float), "alloc d_norm");
    realloc_buf(d_qkv_, size_t(new_S) * size_t(3 * H) * sizeof(float), "alloc d_qkv");
    realloc_buf(d_q_nh_, size_t(new_S) * size_t(H) * sizeof(float), "alloc d_q_nh");
    realloc_buf(d_k_nh_, size_t(new_S) * size_t(H) * sizeof(float), "alloc d_k_nh");
    realloc_buf(d_v_nh_, size_t(new_S) * size_t(H) * sizeof(float), "alloc d_v_nh");
    realloc_buf(d_proj_, size_t(new_S) * size_t(H) * sizeof(float), "alloc d_proj");
    const size_t mlp_elems = std::max({size_t(new_S) * size_t(I), size_t(new_pat) * size_t(patch_in_dim),
                                       size_t(new_img) * size_t(cfg_.projection_hidden_dim)});
    realloc_buf(d_mlp_, mlp_elems * sizeof(float), "alloc d_mlp");
    realloc_buf(d_merged_, size_t(new_img) * size_t(merge_dim) * sizeof(float), "alloc d_merged");
    realloc_buf(d_out_f32_, size_t(new_img) * size_t(cfg_.text_hidden_size) * sizeof(float), "alloc d_out_f32");
    realloc_buf(d_out_bf16_, size_t(new_img) * size_t(cfg_.text_hidden_size) * sizeof(__nv_bfloat16), "alloc d_out_bf16");
    if (!fp32_gemm_) {
      const size_t x_bf16_elems = std::max({size_t(new_S) * size_t(I), size_t(new_pat) * size_t(patch_in_dim),
                                            size_t(new_img) * size_t(std::max(merge_dim, cfg_.projection_hidden_dim))});
      realloc_buf(d_x_bf16_, x_bf16_elems * sizeof(__nv_bfloat16), "alloc d_x_bf16");
    }
    cap_S_ = new_S;
    cap_pat_ = new_pat;
    cap_img_ = new_img;
  }

  if (fp32_gemm_) {
    const size_t max_w_elems = std::max({size_t(3 * H) * size_t(H), size_t(H) * size_t(I),
                                         size_t(cfg_.projection_hidden_dim) * size_t(merge_dim),
                                         size_t(cfg_.text_hidden_size) * size_t(cfg_.projection_hidden_dim),
                                         size_t(H) * size_t(patch_in_dim)});
    if (max_w_elems > cap_w_f32_elems_) {
      if (d_w_f32_) cudaFree(d_w_f32_);
      d_w_f32_ = nullptr;
      vis_cuda_ok(cudaMalloc(&d_w_f32_, max_w_elems * sizeof(float)), "alloc d_w_f32");
      cap_w_f32_elems_ = max_w_elems;
    }
  }

  if (need_rgb_bytes > cap_rgb_bytes_) {
    if (d_rgb_) cudaFree(d_rgb_);
    d_rgb_ = nullptr;
    vis_cuda_ok(cudaMalloc(&d_rgb_, std::max<size_t>(need_rgb_bytes, 16)), "alloc d_rgb");
    cap_rgb_bytes_ = need_rgb_bytes;
  }

  if (need_frames > cap_frames_) {
    if (d_desc_) cudaFree(d_desc_);
    d_desc_ = nullptr;
    vis_cuda_ok(cudaMalloc(&d_desc_, size_t(need_frames) * sizeof(VisFrameDesc)), "alloc d_desc");
    cap_frames_ = need_frames;
  }
}

void Ink2VisionEncoder::linear_f32(const float* x_f32, const __nv_bfloat16* w_bf16, float* y_f32,
                                   int M, int N, int K) {
  cublasSetStream(cublas_, stream_);
  const float alpha = 1.0f, beta = 0.0f;
  if (fp32_gemm_) {
    const size_t w_elems = size_t(N) * size_t(K);
    k_vis_bf16_to_f32<<<int((w_elems + 255) / 256), 256, 0, stream_>>>(w_bf16, d_w_f32_, w_elems);
    const cublasStatus_t st = cublasGemmEx(
        cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_w_f32_, CUDA_R_32F, K, x_f32, CUDA_R_32F, K,
        &beta, y_f32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("ink2_vision: cublasGemmEx FP32 failed");
    return;
  }
  const size_t x_elems = size_t(M) * size_t(K);
  k_vis_f32_to_bf16_rn<<<int((x_elems + 255) / 256), 256, 0, stream_>>>(x_f32, d_x_bf16_, x_elems);
  const cublasStatus_t st = cublasGemmEx(
      cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w_bf16, CUDA_R_16BF, K, d_x_bf16_, CUDA_R_16BF, K,
      &beta, y_f32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("ink2_vision: cublasGemmEx BF16 failed");
}

void Ink2VisionEncoder::encode_rgb(const uint8_t* const* frames, const int32_t* widths, const int32_t* heights,
                                   int n_frames, uint16_t* out_bf16, int32_t* out_grids, int32_t* out_n_tokens) {
  if (n_frames <= 0) throw std::runtime_error("ink2_vision: n_frames must be >= 1");
  if (!frames || !widths || !heights || !out_bf16) throw std::runtime_error("ink2_vision: null input/output pointer");

  const int P = cfg_.patch_size;
  const int M = cfg_.merge_size;
  const int side = P * M;
  const int H = cfg_.hidden_size;
  const int NH = cfg_.num_attention_heads;
  const int HD = cfg_.head_dim;
  const int I = cfg_.intermediate_size;
  const int G = cfg_.posemb_grid_size;
  const int patch_in_dim = P * P * 3;
  const int merge_dim = M * M * H;

  std::vector<int32_t> f_rows(static_cast<size_t>(n_frames));
  std::vector<int32_t> f_cols(static_cast<size_t>(n_frames));
  std::vector<int32_t> f_seq(static_cast<size_t>(n_frames));
  int64_t total_img_tokens = 0;

  for (int i = 0; i < n_frames; ++i) {
    if (!frames[i]) throw std::runtime_error("ink2_vision: null frame pointer");
    const int32_t w = widths[i];
    const int32_t h = heights[i];
    if (w <= 0 || h <= 0 || w > kMaxVisionSide || h > kMaxVisionSide || int64_t(w) * int64_t(h) > kMaxVisionPixels)
      throw std::runtime_error("ink2_vision: invalid frame dimensions");
    const int32_t rows = (h + side - 1) / side;
    const int32_t cols = (w + side - 1) / side;
    f_rows[size_t(i)] = rows;
    f_cols[size_t(i)] = cols;
    const int64_t n_img = int64_t(rows) * int64_t(cols);
    f_seq[size_t(i)] = 1 + int32_t(n_img) * M * M;
    total_img_tokens += n_img;
    if (out_grids) {
      out_grids[size_t(2 * i)] = rows;
      out_grids[size_t(2 * i + 1)] = cols;
    }
  }
  if (total_img_tokens > std::numeric_limits<int32_t>::max())
    throw std::runtime_error("ink2_vision: total image tokens overflow");
  if (out_n_tokens) *out_n_tokens = int32_t(total_img_tokens);

  ensure_loaded();

  int b0 = 0;
  int64_t out_tok_off = 0;
  while (b0 < n_frames) {
    int b1 = b0;
    int batch_S = 0;
    int batch_pat = 0;
    int batch_img = 0;
    size_t batch_rgb_bytes = 0;

    while (b1 < n_frames) {
      const int s_len = f_seq[size_t(b1)];
      if (b1 > b0 && batch_S + s_len > kTargetBatchSeq) break;
      const int n_img = f_rows[size_t(b1)] * f_cols[size_t(b1)];
      batch_S += s_len;
      batch_pat += n_img * M * M;
      batch_img += n_img;
      batch_rgb_bytes += size_t(widths[b1]) * size_t(heights[b1]) * 3u;
      ++b1;
    }

    const int n_batch = b1 - b0;
    ensure_scratch(batch_S, batch_pat, batch_img, batch_rgb_bytes, n_batch);

    std::vector<VisFrameDesc> descs(static_cast<size_t>(n_batch));
    if (h_rgb_stage_.size() < batch_rgb_bytes) h_rgb_stage_.resize(batch_rgb_bytes);
    int s_cur = 0, p_cur = 0, i_cur = 0;
    size_t rgb_cur = 0;
    int max_s_len = 0;
    for (int bi = 0; bi < n_batch; ++bi) {
      const int fi = b0 + bi;
      const int w = widths[fi], h = heights[fi];
      const int rows = f_rows[size_t(fi)], cols = f_cols[size_t(fi)];
      const int n_img = rows * cols;
      const int n_pat = n_img * M * M;
      const int s_len = 1 + n_pat;
      const size_t fbytes = size_t(w) * size_t(h) * 3u;
      std::memcpy(h_rgb_stage_.data() + rgb_cur, frames[fi], fbytes);
      descs[size_t(bi)] = VisFrameDesc{w, h, rows, cols, s_cur, p_cur, i_cur, uint64_t(rgb_cur)};
      s_cur += s_len;
      p_cur += n_pat;
      i_cur += n_img;
      rgb_cur += fbytes;
      max_s_len = std::max(max_s_len, s_len);
    }

    vis_cuda_ok(cudaMemcpyAsync(d_rgb_, h_rgb_stage_.data(), batch_rgb_bytes, cudaMemcpyHostToDevice, stream_),
                "upload rgb");
    vis_cuda_ok(cudaMemcpyAsync(d_desc_, descs.data(), size_t(n_batch) * sizeof(VisFrameDesc), cudaMemcpyHostToDevice, stream_),
                "upload frame descs");

    const VisFrameDesc* d_desc_ptr = static_cast<const VisFrameDesc*>(d_desc_);

    // 1. Extract 20x20x3 block-major patches + normalize -> d_mlp_ [batch_pat, 1200]
    k_vis_extract_patches<<<batch_pat, 256, 0, stream_>>>(d_rgb_, d_desc_ptr, n_batch, P, M, d_mlp_);

    // 2. Patch embedding [batch_pat, 1200] -> d_proj_ [batch_pat, H]
    linear_f32(d_mlp_, patch_emb_w_, d_proj_, batch_pat, H, patch_in_dim);

    // 3. Bilinear pos_emb + [CLS] prepend -> d_hidden_ [batch_S, H]
    k_vis_pos_cls<<<batch_S, 256, 0, stream_>>>(d_proj_, patch_emb_b_, class_emb_, cls_pos_emb_, pos_emb_,
                                                d_desc_ptr, n_batch, H, M, G, d_hidden_);

    // 4. ln_pre -> d_hidden_ [batch_S, H]
    k_vis_layernorm<<<batch_S, 256, 0, stream_>>>(d_hidden_, ln_pre_w_, ln_pre_b_, d_hidden_, H, cfg_.layer_norm_eps);

    // 5. 23 ViT layers
    const float attn_scale = 1.0f / std::sqrt(float(HD));
    const int q_tiles = (max_s_len + 3) / 4;
    const dim3 attn_grid(q_tiles, NH, n_batch);
    const int qkv_threads = NH * 32;  // 512

    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
      const Ink2VisionLayerWeights& L = layers_[size_t(l)];

      k_vis_layernorm<<<batch_S, 256, 0, stream_>>>(d_hidden_, L.ln1_w, L.ln1_b, d_norm_, H, cfg_.layer_norm_eps);
      linear_f32(d_norm_, L.qkv_w, d_qkv_, batch_S, 3 * H, H);
      k_vis_qkv_bias_rope<<<batch_S, qkv_threads, 0, stream_>>>(
          d_qkv_, L.qkv_b, d_desc_ptr, n_batch, batch_S, H, NH, HD, M, cfg_.rope_theta, d_q_nh_, d_k_nh_, d_v_nh_);
      k_vis_flash_attn_d64<<<attn_grid, 128, 0, stream_>>>(
          d_q_nh_, d_k_nh_, d_v_nh_, d_desc_ptr, batch_S, NH, M, attn_scale, d_norm_);
      linear_f32(d_norm_, L.out_w, d_proj_, batch_S, H, H);
      {
        const size_t n_tot = size_t(batch_S) * size_t(H);
        k_vis_bias_layerscale_add<<<int((n_tot + 255) / 256), 256, 0, stream_>>>(
            d_proj_, L.out_b, L.ls1_gamma, d_hidden_, batch_S, H);
      }

      k_vis_layernorm<<<batch_S, 256, 0, stream_>>>(d_hidden_, L.ln2_w, L.ln2_b, d_norm_, H, cfg_.layer_norm_eps);
      linear_f32(d_norm_, L.fc1_w, d_mlp_, batch_S, I, H);
      {
        const size_t n_tot = size_t(batch_S) * size_t(I);
        k_vis_bias_gelu<<<int((n_tot + 255) / 256), 256, 0, stream_>>>(d_mlp_, L.fc1_b, batch_S, I);
      }
      linear_f32(d_mlp_, L.fc2_w, d_proj_, batch_S, H, I);
      {
        const size_t n_tot = size_t(batch_S) * size_t(H);
        k_vis_bias_layerscale_add<<<int((n_tot + 255) / 256), 256, 0, stream_>>>(
            d_proj_, L.fc2_b, L.ls2_gamma, d_hidden_, batch_S, H);
      }
    }

    // 6. Drop [CLS] and pack 2x2 merger blocks -> d_merged_ [batch_img, 4096]
    k_vis_drop_cls<<<batch_img, 256, 0, stream_>>>(d_hidden_, d_desc_ptr, n_batch, H, M, d_merged_);

    // 7. Merger: proj_in -> GELU -> proj_out -> BF16
    linear_f32(d_merged_, proj_in_w_, d_mlp_, batch_img, cfg_.projection_hidden_dim, merge_dim);
    {
      const size_t n_tot = size_t(batch_img) * size_t(cfg_.projection_hidden_dim);
      k_vis_bias_gelu<<<int((n_tot + 255) / 256), 256, 0, stream_>>>(
          d_mlp_, proj_in_b_, batch_img, cfg_.projection_hidden_dim);
    }
    linear_f32(d_mlp_, proj_out_w_, d_out_f32_, batch_img, cfg_.text_hidden_size, cfg_.projection_hidden_dim);
    {
      const size_t n_tot = size_t(batch_img) * size_t(cfg_.text_hidden_size);
      if (proj_out_b_) {
        k_vis_add_bias<<<int((n_tot + 255) / 256), 256, 0, stream_>>>(
            d_out_f32_, proj_out_b_, batch_img, cfg_.text_hidden_size);
      }
      k_vis_f32_to_bf16_rn<<<int((n_tot + 255) / 256), 256, 0, stream_>>>(d_out_f32_, d_out_bf16_, n_tot);
      vis_cuda_ok(cudaMemcpyAsync(out_bf16 + out_tok_off * int64_t(cfg_.text_hidden_size), d_out_bf16_,
                                  n_tot * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream_),
                  "copy vis_out");
    }
    vis_cuda_ok(cudaStreamSynchronize(stream_), "sync batch encode");
    out_tok_off += batch_img;
    b0 = b1;
  }
}

}  // namespace yskn
