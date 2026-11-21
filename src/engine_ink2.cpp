// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "engine_ink2.hpp"

#include "ink2_kernels.cuh"
#include "safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace yskn {
namespace {

constexpr int kVisitConvFloats = 3 * (1024 + 1024 + 2304 + 2304);
constexpr int kMaxAttnChunks = 64;

inline void cuda_ok(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline float bf16_bits_to_f32(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

std::string slurp_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline uint64_t splitmix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

inline double sample_uniform(uint64_t seed, int64_t pos) {
  const uint64_t h = splitmix64(splitmix64(seed) ^ uint64_t(pos));
  return double(h >> 11) * (1.0 / 9007199254740992.0);
}

Ink2Config parse_ink2_config(const std::string& dir) {
  const stjson::Val root = stjson::parse(slurp_file(dir + "/config.json"));
  const stjson::Val& t = root.has("text_config") ? root.at("text_config") : root;
  auto gi = [](const stjson::Val& v, int def) { return v.kind == stjson::Val::Num ? int(v.i) : def; };
  auto gf = [](const stjson::Val& v, float def) { return v.kind == stjson::Val::Num ? float(v.num) : def; };

  Ink2Config c;
  c.hidden = gi(t.at("hidden_size"), 2304);
  c.num_layers = gi(t.at("num_hidden_layers"), 24);
  c.vocab_padded = gi(t.at("vocab_size"), 201024);
  c.vocab = gi(t.at("unpadded_vocab_size"), c.vocab_padded);
  c.hq = gi(t.at("num_attention_heads"), 24);
  c.hkv = gi(t.at("num_key_value_heads"), 8);
  c.head_dim = gi(t.at("head_dim"), 128);
  c.d_rel = gi(t.at("d_rel"), 16);
  c.rel_extent = gi(t.at("rel_extent"), 1024);
  c.swa = gi(t.at("sliding_window_size"), 1024);
  c.block_sz = gi(t.at("n_layers_per_attn_res_block"), 6);
  c.n_experts = gi(t.at("n_routed_experts"), 128);
  c.top_k = gi(t.at("num_experts_per_tok"), 8);
  c.moe_I = gi(t.at("intermediate_size"), 1152);
  c.rms_eps = gf(t.at("rms_norm_eps"), 1e-6f);
  c.qk_clip = gf(t.at("qk_l2_clip_threshold"), 22.627416997969522f);
  c.rel_clip = gf(t.at("rel_l2_clip_threshold"), 8.0f);
  c.log_floor = gf(t.at("log_scaling_n_floor"), 8192.0f);
  c.log_alpha = gf(t.at("log_scaling_alpha"), 0.1f);
  c.route_scale = gf(t.at("route_scale"), 8.0f);
  c.logit_softcap = gf(t.at("final_logit_softcapping"), 30.0f);

  const stjson::Val& loop = t.at("loop_config");
  c.core_start = gi(loop.at("core_start"), 6);
  c.core_end = gi(loop.at("core_end"), 18);
  c.num_passes = gi(loop.at("num_passes"), 3);
  c.gate_rank = gi(loop.at("gate_rank"), 64);

  c.is_local.assign(size_t(c.num_layers), false);
  const stjson::Val& lids = t.at("local_layer_ids");
  if (lids.kind == stjson::Val::Arr) {
    for (const auto& lid : lids.arr) {
      if (lid.i >= 0 && lid.i < c.num_layers) c.is_local[size_t(lid.i)] = true;
    }
  }
  for (int i = 0; i < c.core_start; ++i) c.schedule.push_back(i);
  for (int pass = 0; pass < c.num_passes; ++pass)
    for (int i = c.core_start; i < c.core_end; ++i) c.schedule.push_back(i);
  for (int i = c.core_end; i < c.num_layers; ++i) c.schedule.push_back(i);
  return c;
}

}  // namespace

std::string EngineInk2::resolve_dir(const std::string& path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path p(path);
  if (fs::is_regular_file(p, ec)) p = p.parent_path();
  std::string d = p.string();
  while (d.size() > 1 && d.back() == '/') d.pop_back();
  return d;
}

bool EngineInk2::is_inkling2_dir(const std::string& path) {
  namespace fs = std::filesystem;
  try {
    const std::string dir = resolve_dir(path);
    const std::string cfg = dir + "/config.json";
    if (!fs::exists(cfg)) return false;
    const stjson::Val root = stjson::parse(slurp_file(cfg));
    if (root.at("model_type").s == "inkling2_loop") return true;
    return root.has("text_config") && root.at("text_config").at("model_type").s == "inkling2_loop_text";
  } catch (...) {
    return false;
  }
}

void* EngineInk2::alloc_gpu(size_t bytes) {
  void* ptr = nullptr;
  const size_t sz = std::max<size_t>(bytes, 16);
  cuda_ok(cudaMalloc(&ptr, sz), "cudaMalloc");
  gpu_allocs_.push_back(ptr);
  return ptr;
}

EngineInk2::EngineInk2(const std::string& path, int max_ctx, int /*flags*/) {
  dir_ = resolve_dir(path);
  if (!is_inkling2_dir(dir_)) throw std::runtime_error(path + " is not an Inkling-2 checkpoint");
  cfg_ = parse_ink2_config(dir_);
  max_ctx_ = max_ctx > 0 ? max_ctx : 32768;
  ring_rows_ = (cfg_.swa + kPrefillChunk + 15) / 16 * 16;

  cuda_ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");

  SafetensorsReader reader;
  reader.open_dir(dir_, "model.llm.");

  auto upload_tensor = [&](const std::string& name) -> Ink2Tensor {
    const StTensorInfo& info = reader.get(name);
    Ink2Tensor t;
    t.f32 = (info.dtype == "F32");
    if (!t.f32 && info.dtype != "BF16") throw std::runtime_error("unexpected dtype " + info.dtype + " for " + name);
    if (info.shape.size() == 1) {
      t.rows = int(info.shape[0]);
      t.cols = 1;
    } else if (info.shape.size() >= 2) {
      t.rows = int(info.shape[0]);
      size_t col_prod = 1;
      for (size_t i = 1; i < info.shape.size(); ++i) col_prod *= size_t(info.shape[i]);
      t.cols = int(col_prod);
    }
    t.bytes = info.nbytes;
    t.d_ptr = alloc_gpu(info.nbytes);
    reader.upload_gpu(info, t.d_ptr, stream_);
    return t;
  };

  auto upload_raw = [&](const std::string& name) -> uint8_t* {
    const StTensorInfo& info = reader.get(name);
    uint8_t* p = static_cast<uint8_t*>(alloc_gpu(info.nbytes));
    reader.upload_gpu(info, p, stream_);
    return p;
  };

  auto read_f32_vec = [&](const StTensorInfo& info) -> std::vector<float> {
    const size_t n = info.numel();
    std::vector<float> out(n);
    if (info.dtype == "F32") {
      reader.read_host(info, out.data());
    } else if (info.dtype == "BF16") {
      std::vector<uint16_t> tmp(n);
      reader.read_host(info, tmp.data());
      for (size_t i = 0; i < n; ++i) out[i] = bf16_bits_to_f32(tmp[i]);
    } else {
      throw std::runtime_error("unsupported scalar/vec dtype " + info.dtype);
    }
    return out;
  };

  embed_ = upload_tensor("model.llm.embed.weight");
  embed_norm_ = upload_tensor("model.llm.embed_norm.weight");
  final_norm_ = upload_tensor("model.llm.norm.weight");
  unembed_ = upload_tensor("model.llm.unembed.weight");

  attn_queries_.resize(size_t(cfg_.num_layers - 1));
  for (int i = 0; i < cfg_.num_layers - 1; ++i)
    attn_queries_[size_t(i)] = upload_tensor("model.llm.attn_res.attn_queries." + std::to_string(i));
  mlp_queries_.resize(size_t(cfg_.num_layers));
  for (int i = 0; i < cfg_.num_layers; ++i)
    mlp_queries_[size_t(i)] = upload_tensor("model.llm.attn_res.mlp_queries." + std::to_string(i));
  output_query_ = upload_tensor("model.llm.attn_res.output_query");

  // Pre-interleave [ncore, 4, H] middle cycle cross-pass queries: (attn, k, v, mlp)
  const int ncore = cfg_.core_end - cfg_.core_start;
  const std::vector<float> qam = read_f32_vec(reader.get("model.llm.middle_cycle_cross_pass_queries"));
  const std::vector<float> qkv = read_f32_vec(reader.get("model.llm.middle_cycle_cross_pass_kv_queries"));
  const size_t H = size_t(cfg_.hidden);
  std::vector<float> mq(size_t(ncore) * 4 * H);
  for (int c = 0; c < ncore; ++c) {
    for (size_t j = 0; j < H; ++j) {
      mq[(size_t(c) * 4 + 0) * H + j] = qam[(size_t(c) * 2 + 0) * H + j];
      mq[(size_t(c) * 4 + 1) * H + j] = qkv[(size_t(c) * 2 + 0) * H + j];
      mq[(size_t(c) * 4 + 2) * H + j] = qkv[(size_t(c) * 2 + 1) * H + j];
      mq[(size_t(c) * 4 + 3) * H + j] = qam[(size_t(c) * 2 + 1) * H + j];
    }
  }
  d_msg_queries_ = static_cast<float*>(alloc_gpu(mq.size() * sizeof(float)));
  cuda_ok(cudaMemcpyAsync(d_msg_queries_, mq.data(), mq.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
          "upload msg_queries");
  cuda_ok(cudaStreamSynchronize(stream_), "sync msg_queries");

  auto make_outer = [&](const std::string& base) -> float* {
    const std::vector<float> amax = read_f32_vec(reader.get(base + "_amax"));
    const std::vector<float> denom = read_f32_vec(reader.get(base + "_outer_scale_denominator"));
    std::vector<float> out(size_t(cfg_.n_experts));
    for (int e = 0; e < cfg_.n_experts; ++e) out[size_t(e)] = amax[size_t(e)] / denom[0];
    float* dp = static_cast<float*>(alloc_gpu(out.size() * sizeof(float)));
    cuda_ok(cudaMemcpyAsync(dp, out.data(), out.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
            "upload outer_scale");
    cuda_ok(cudaStreamSynchronize(stream_), "sync outer_scale");
    return dp;
  };

  layers_.resize(size_t(cfg_.num_layers));
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Ink2Layer& L = layers_[size_t(i)];
    const std::string p = "model.llm.layers." + std::to_string(i) + ".";
    L.is_swa = cfg_.is_local[size_t(i)];
    L.has_ctrl = (i >= cfg_.core_start && i < cfg_.core_end);
    L.wq_du = upload_tensor(p + "attn.wq_du.weight");
    L.wk_dv = upload_tensor(p + "attn.wk_dv.weight");
    L.wv_dv = upload_tensor(p + "attn.wv_dv.weight");
    L.wr_du = upload_tensor(p + "attn.wr_du.weight");
    L.attn_sink_proj = upload_tensor(p + "attn.attn_sink_proj.weight");
    L.gate_attn = upload_tensor(p + "attn.gate_attn.weight");
    L.wo_ud = upload_tensor(p + "attn.wo_ud.weight");
    L.q_scale = upload_tensor(p + "attn.q_l2_clip.scale");
    L.rel_scale = upload_tensor(p + "attn.rel_l2_clip.scale");
    L.rel_proj = upload_tensor(p + "attn.rel_logits_proj.proj");
    L.mlp_norm = upload_tensor(p + "mlp_norm.weight");
    L.attn_sconv = upload_tensor(p + "attn_sconv.weight");
    L.mlp_sconv = upload_tensor(p + "mlp_sconv.weight");
    if (L.is_swa) {
      L.attn_norm = upload_tensor(p + "attn_norm.weight");
      L.k_sconv = upload_tensor(p + "attn.k_sconv.weight");
      L.v_sconv = upload_tensor(p + "attn.v_sconv.weight");
    } else {
      L.q_norm = upload_tensor(p + "q_input_norm.weight");
      L.k_norm = upload_tensor(p + "k_input_norm.weight");
      L.v_norm = upload_tensor(p + "v_input_norm.weight");
      L.k_static = upload_tensor(p + "attn.k_sconv.static_w");
      L.k_proj = upload_tensor(p + "attn.k_sconv.weight_proj");
      L.v_static = upload_tensor(p + "attn.v_sconv.static_w");
      L.v_proj = upload_tensor(p + "attn.v_sconv.weight_proj");
    }
    if (L.has_ctrl) {
      auto load_ctrl = [&](const char* name) -> Ink2Ctrl {
        const std::string c = p + "controllers." + name + ".";
        return Ink2Ctrl{upload_tensor(c + "down_weight"), upload_tensor(c + "forget_weight"),
                        upload_tensor(c + "gain_weight")};
      };
      L.ctrl_attn = load_ctrl("attn");
      L.ctrl_k = load_ctrl("k");
      L.ctrl_v = load_ctrl("v");
      L.ctrl_mlp = load_ctrl("mlp");
    }
    L.router_w = upload_tensor(p + "mlp.gate.weight");
    L.router_bias = upload_tensor(p + "mlp.gate.bias");

    const std::string m = p + "mlp.experts.";
    const StTensorInfo& w13_info = reader.get(m + "w13_weight");
    L.is_mxfp8 = (w13_info.dtype == "F8_E4M3");
    if (!L.is_mxfp8 && w13_info.dtype != "U8")
      throw std::runtime_error("unexpected expert dtype " + w13_info.dtype);
    L.w13 = upload_raw(m + "w13_weight");
    L.w13_sc = upload_raw(m + "w13_weight_scale");
    L.w2 = upload_raw(m + "w2_weight");
    L.w2_sc = upload_raw(m + "w2_weight_scale");
    if (!L.is_mxfp8) {
      L.w13_outer = make_outer(m + "w13_weight");
      L.w2_outer = make_outer(m + "w2_weight");
    }
  }
  reader.close_all();

  // Allocate KV caches & conv histories for 48 visits
  const size_t nv = cfg_.schedule.size();
  const size_t kv_row_elems = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  d_kc_.resize(nv);
  d_vc_.resize(nv);
  d_conv_.resize(nv);
  for (size_t v = 0; v < nv; ++v) {
    const int ip = cfg_.schedule[v];
    const size_t cap = cfg_.is_local[size_t(ip)] ? size_t(ring_rows_) : size_t(max_ctx_);
    d_kc_[v] = static_cast<__half*>(alloc_gpu(cap * kv_row_elems * sizeof(__half)));
    d_vc_[v] = static_cast<__half*>(alloc_gpu(cap * kv_row_elems * sizeof(__half)));
    d_conv_[v] = static_cast<float*>(alloc_gpu(size_t(kVisitConvFloats) * sizeof(float)));
  }

  // Allocate activations sized for kPrefillChunk rows
  const size_t T = size_t(kPrefillChunk);
  const size_t QD = size_t(cfg_.hq) * size_t(cfg_.head_dim);
  const size_t KVD = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  const size_t V = size_t(cfg_.vocab);

  d_tokens_ = static_cast<int32_t*>(alloc_gpu(T * sizeof(int32_t)));
  d_tok_out_ = static_cast<int32_t*>(alloc_gpu(sizeof(int32_t)));
  d_blk_v_ = static_cast<float*>(alloc_gpu(256 * sizeof(float)));
  d_blk_i_ = static_cast<int32_t*>(alloc_gpu(256 * sizeof(int32_t)));
  d_x_bf16_ = static_cast<__nv_bfloat16*>(alloc_gpu(T * size_t(cfg_.hq * cfg_.head_dim) * sizeof(__nv_bfloat16)));

  d_x_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_x_pre_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_inp_sa_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_ffn_inp_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_h_norm_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  for (int i = 0; i < 6; ++i) d_blocks_[i] = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  for (int i = 0; i < 4; ++i) d_pre_blocks_[i] = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_ar_out_ = static_cast<float*>(alloc_gpu(4 * T * H * sizeof(float)));
  d_mg0_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_mg1_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_mg2_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_n0_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_n1_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_n2_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));

  d_q_ = static_cast<float*>(alloc_gpu(T * QD * sizeof(float)));
  d_k_ = static_cast<float*>(alloc_gpu(T * KVD * sizeof(float)));
  d_v_ = static_cast<float*>(alloc_gpu(T * KVD * sizeof(float)));
  d_r_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.hq) * size_t(cfg_.d_rel) * sizeof(float)));
  d_sinks_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.hq) * sizeof(float)));
  d_gate_attn_ = static_cast<float*>(alloc_gpu(T * QD * sizeof(float)));
  d_att_out_ = static_cast<float*>(alloc_gpu(T * QD * sizeof(float)));
  d_a_out_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));

  d_dyn_ = static_cast<float*>(alloc_gpu(T * 32 * sizeof(float)));
  d_rb_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.hq) * size_t(cfg_.rel_extent) * sizeof(float)));
  d_part_m_ = static_cast<float*>(alloc_gpu(size_t(cfg_.hq) * kMaxAttnChunks * sizeof(float)));
  d_part_l_ = static_cast<float*>(alloc_gpu(size_t(cfg_.hq) * kMaxAttnChunks * sizeof(float)));
  d_part_o_ = static_cast<float*>(alloc_gpu(size_t(cfg_.hq) * kMaxAttnChunks * size_t(cfg_.head_dim) * sizeof(float)));

  d_h_mlp_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_x_mlp_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_rlog_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.n_experts) * sizeof(float)));
  d_eid_ = static_cast<uint32_t*>(alloc_gpu(T * size_t(cfg_.top_k) * sizeof(uint32_t)));
  d_ew_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.top_k) * sizeof(float)));
  d_moe_act_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.top_k) * size_t(cfg_.moe_I) * sizeof(float)));
  d_moe_out_ = static_cast<float*>(alloc_gpu(T * H * sizeof(float)));
  d_logits_ = static_cast<float*>(alloc_gpu(V * sizeof(float)));
  d_suppress_ = static_cast<uint8_t*>(alloc_gpu(V));

  cuda_ok(cudaMallocHost(&h_logits_, V * sizeof(float)), "cudaMallocHost logits");
  cuda_ok(cudaMallocHost(&h_tok_, sizeof(int32_t)), "cudaMallocHost tok");
  cuda_ok(cudaMemsetAsync(d_suppress_, 0, V, stream_), "clear suppress");
  reset();
}

EngineInk2::~EngineInk2() {
  if (stream_) cudaStreamSynchronize(stream_);
  if (h_logits_) cudaFreeHost(h_logits_);
  if (h_tok_) cudaFreeHost(h_tok_);
  if (cublas_) cublasDestroy(cublas_);
  for (void* p : gpu_allocs_) cudaFree(p);
  if (stream_) cudaStreamDestroy(stream_);
}

void EngineInk2::reset() {
  for (float* p : d_conv_)
    cuda_ok(cudaMemsetAsync(p, 0, size_t(kVisitConvFloats) * sizeof(float), stream_), "memset conv");
  cuda_ok(cudaStreamSynchronize(stream_), "reset sync");
  pos_ = 0;
  last_tok_ = -1;
  logits_valid_ = false;
  live_tokens_.clear();
}

void EngineInk2::set_opt(int key, int value) {
  switch (key) {
    case 10: temp_ = float(value) / 1000.0f; break;
    case 11: top_p_ = float(value) / 1000.0f; break;
    case 12: top_k_ = value; break;
    case 13: seed_ = uint64_t(uint32_t(value)); break;
    case 16: sampler_mode_ = value; break;
    case 29: min_p_ = float(value) / 1000.0f; break;
    default: break;
  }
}

void EngineInk2::set_suppress(const int32_t* token_ids, int n_tokens) {
  const size_t V = size_t(cfg_.vocab);
  if (!token_ids || n_tokens <= 0) {
    suppress_active_ = false;
    cuda_ok(cudaMemsetAsync(d_suppress_, 0, V, stream_), "clear suppress");
    cuda_ok(cudaStreamSynchronize(stream_), "sync suppress clear");
    return;
  }
  std::vector<uint8_t> bits(V, 0);
  for (int i = 0; i < n_tokens; ++i) {
    const int32_t id = token_ids[i];
    if (id < 0 || id >= cfg_.vocab) throw std::runtime_error("suppress token ID out of range");
    bits[size_t(id)] = 1;
  }
  cuda_ok(cudaMemcpyAsync(d_suppress_, bits.data(), V, cudaMemcpyHostToDevice, stream_), "upload suppress");
  cuda_ok(cudaStreamSynchronize(stream_), "sync suppress");
  suppress_active_ = true;
}

void EngineInk2::forward_tokens(const int32_t* ids, int T, int pos0, bool need_logits) {
  if (T <= 0 || T > kPrefillChunk) throw std::runtime_error("invalid Inkling2 chunk size");
  if (pos0 + T > max_ctx_) throw std::runtime_error("context full (" + std::to_string(max_ctx_) + ")");

  const int H = cfg_.hidden;
  const int QD = cfg_.hq * cfg_.head_dim;
  const int KVD = cfg_.hkv * cfg_.head_dim;
  const size_t row_bytes = size_t(T) * size_t(H) * sizeof(float);
  const float sconv_static_scale = 1.0f / std::sqrt(5.0f);
  const float ar_scale = 1.0f / std::sqrt(float(H));

  cuda_ok(cudaMemcpyAsync(d_tokens_, ids, size_t(T) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "upload tokens");
  ink2::launch_embed_norm(d_tokens_, 0, static_cast<const __nv_bfloat16*>(embed_.d_ptr),
                          static_cast<const __nv_bfloat16*>(embed_norm_.d_ptr), d_x_, T, H, cfg_.rms_eps, stream_);

  std::vector<const float*> blocks_vec;
  bool have_prelude = false;
  bool have_prev = false;
  int n_prelude_blocks = 0;

  for (size_t v_idx = 0; v_idx < cfg_.schedule.size(); ++v_idx) {
    const int ip = cfg_.schedule[v_idx];
    const Ink2Layer& L = layers_[size_t(ip)];
    float* conv_k = d_conv_[v_idx] + 0;
    float* conv_v = d_conv_[v_idx] + 3 * KVD;
    float* conv_a = d_conv_[v_idx] + 6 * KVD;
    float* conv_m = d_conv_[v_idx] + 6 * KVD + 3 * H;

    if (ip == cfg_.core_start) {
      if (!have_prelude) {
        cuda_ok(cudaMemcpyAsync(d_x_pre_, d_x_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "copy x_pre");
        n_prelude_blocks = int(blocks_vec.size());
        for (int b = 0; b < n_prelude_blocks; ++b)
          cuda_ok(cudaMemcpyAsync(d_pre_blocks_[b], blocks_vec[size_t(b)], row_bytes, cudaMemcpyDeviceToDevice, stream_),
                  "copy pre_block");
        have_prelude = true;
      } else {
        cuda_ok(cudaMemcpyAsync(d_blocks_[4], blocks_vec[size_t(n_prelude_blocks + 1)], row_bytes,
                                cudaMemcpyDeviceToDevice, stream_),
                "copy bank0");
        cuda_ok(cudaMemcpyAsync(d_blocks_[5], d_x_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "copy bank1");
        cuda_ok(cudaMemcpyAsync(d_x_, d_x_pre_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "restore x_pre");
        blocks_vec.clear();
        for (int b = 0; b < n_prelude_blocks; ++b) {
          cuda_ok(cudaMemcpyAsync(d_blocks_[b], d_pre_blocks_[b], row_bytes, cudaMemcpyDeviceToDevice, stream_),
                  "restore pre_block");
          blocks_vec.push_back(d_blocks_[b]);
        }
        have_prev = true;
      }
    }

    const bool has_msgs = have_prev && ip < cfg_.core_end;
    float* msg0 = d_ar_out_ + 0 * size_t(T) * size_t(H);
    float* msg1 = d_ar_out_ + 1 * size_t(T) * size_t(H);
    float* msg2 = d_ar_out_ + 2 * size_t(T) * size_t(H);
    float* msg3 = d_ar_out_ + 3 * size_t(T) * size_t(H);
    if (has_msgs) {
      const float* bank[2] = {d_blocks_[4], d_blocks_[5]};
      const float* qv = d_msg_queries_ + size_t(ip - cfg_.core_start) * 4 * size_t(H);
      ink2::launch_attn_res(bank, 2, qv, true, 4, d_ar_out_, T, H, cfg_.rms_eps, ar_scale, stream_);
      cuda_ok(cudaMemcpyAsync(d_mg0_, msg0, row_bytes, cudaMemcpyDeviceToDevice, stream_), "save mg0");
      cuda_ok(cudaMemcpyAsync(d_mg1_, msg1, row_bytes, cudaMemcpyDeviceToDevice, stream_), "save mg1");
      cuda_ok(cudaMemcpyAsync(d_mg2_, msg2, row_bytes, cudaMemcpyDeviceToDevice, stream_), "save mg2");
      cuda_ok(cudaMemcpyAsync(d_h_norm_, msg3, row_bytes, cudaMemcpyDeviceToDevice, stream_), "save msg3");
    }

    cuda_ok(cudaMemcpyAsync(d_inp_sa_, d_x_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "copy inp_sa");

    const float* in_ptrs[3];
    if (ip == 0) {
      in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = d_inp_sa_;
    } else {
      std::vector<const float*> st = blocks_vec;
      st.push_back(d_inp_sa_);
      const Ink2Tensor& aq = attn_queries_[size_t(ip - 1)];
      const int Q = (aq.cols == 1) ? 1 : (aq.rows / H == 3 ? 3 : 1);
      // Note: upload_tensor collapsed [3, 2304] into rows=3, cols=2304 and [2304] into rows=2304, cols=1!
      const int nq = (aq.cols == 1) ? 1 : aq.rows;
      ink2::launch_attn_res(st.data(), int(st.size()), aq.d_ptr, aq.f32, nq, d_ar_out_, T, H, cfg_.rms_eps, ar_scale,
                            stream_);
      (void)Q;
      if (nq == 1) {
        in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = d_ar_out_;
      } else {
        in_ptrs[0] = msg0;
        in_ptrs[1] = msg1;
        in_ptrs[2] = msg2;
      }
    }

    if (has_msgs) {
      auto call_mg = [&](const Ink2Ctrl& c, const float* h_in, const float* m_in, float* out) {
        ink2::launch_msg_gate(h_in, m_in, static_cast<const __nv_bfloat16*>(c.down.d_ptr),
                              static_cast<const __nv_bfloat16*>(c.forget.d_ptr),
                              static_cast<const __nv_bfloat16*>(c.gain.d_ptr), out, T, H, cfg_.gate_rank, cfg_.rms_eps,
                              stream_);
      };
      call_mg(L.ctrl_attn, in_ptrs[0], d_mg0_, d_n0_);
      call_mg(L.ctrl_k, in_ptrs[1], d_mg1_, d_n1_);
      call_mg(L.ctrl_v, in_ptrs[2], d_mg2_, d_n2_);
      if (L.is_swa) {
        ink2::launch_rmsnorm(d_n0_, L.attn_norm.d_ptr, L.attn_norm.f32, d_n0_, T, H, cfg_.rms_eps, stream_);
        ink2::launch_rmsnorm(d_n1_, L.attn_norm.d_ptr, L.attn_norm.f32, d_n1_, T, H, cfg_.rms_eps, stream_);
        ink2::launch_rmsnorm(d_n2_, L.attn_norm.d_ptr, L.attn_norm.f32, d_n2_, T, H, cfg_.rms_eps, stream_);
      } else {
        ink2::launch_rmsnorm(d_n0_, L.q_norm.d_ptr, L.q_norm.f32, d_n0_, T, H, cfg_.rms_eps, stream_);
        ink2::launch_rmsnorm(d_n1_, L.k_norm.d_ptr, L.k_norm.f32, d_n1_, T, H, cfg_.rms_eps, stream_);
        ink2::launch_rmsnorm(d_n2_, L.v_norm.d_ptr, L.v_norm.f32, d_n2_, T, H, cfg_.rms_eps, stream_);
      }
    } else if (L.is_swa) {
      ink2::launch_rmsnorm(in_ptrs[0], L.attn_norm.d_ptr, L.attn_norm.f32, d_n0_, T, H, cfg_.rms_eps, stream_);
    } else {
      ink2::launch_rmsnorm(in_ptrs[0], L.q_norm.d_ptr, L.q_norm.f32, d_n0_, T, H, cfg_.rms_eps, stream_);
      ink2::launch_rmsnorm(in_ptrs[1], L.k_norm.d_ptr, L.k_norm.f32, d_n1_, T, H, cfg_.rms_eps, stream_);
      ink2::launch_rmsnorm(in_ptrs[2], L.v_norm.d_ptr, L.v_norm.f32, d_n2_, T, H, cfg_.rms_eps, stream_);
    }

    const float* kn = (has_msgs || !L.is_swa) ? d_n1_ : d_n0_;
    const float* vn = (has_msgs || !L.is_swa) ? d_n2_ : d_n0_;

    auto mm = [&](const Ink2Tensor& w, const float* x, float* y, int n_lim = 0) {
      const int N = n_lim ? n_lim : w.rows;
      ink2::launch_linear(cublas_, x, w.d_ptr, w.f32, d_x_bf16_, y, T, N, w.cols, false, stream_);
    };
    mm(L.wq_du, d_n0_, d_q_);
    mm(L.wk_dv, kn, d_k_);
    mm(L.wv_dv, vn, d_v_);
    mm(L.wr_du, d_n0_, d_r_);
    mm(L.attn_sink_proj, d_n0_, d_sinks_);
    mm(L.gate_attn, d_n0_, d_gate_attn_);

    if (L.is_swa) {
      ink2::launch_sconv_static(d_k_, L.k_sconv.d_ptr, L.k_sconv.f32, conv_k, d_k_, T, KVD, sconv_static_scale, stream_);
      ink2::launch_sconv_static(d_v_, L.v_sconv.d_ptr, L.v_sconv.f32, conv_v, d_v_, T, KVD, sconv_static_scale, stream_);
    } else {
      mm(L.k_proj, kn, d_dyn_);
      ink2::launch_sconv_dynamic(d_k_, d_dyn_, L.k_static.d_ptr, L.k_static.f32, conv_k, d_k_, T, KVD, cfg_.hkv,
                                 stream_);
      mm(L.v_proj, vn, d_dyn_);
      ink2::launch_sconv_dynamic(d_v_, d_dyn_, L.v_static.d_ptr, L.v_static.f32, conv_v, d_v_, T, KVD, cfg_.hkv,
                                 stream_);
    }

    const int kv_cap = L.is_swa ? ring_rows_ : max_ctx_;
    ink2::launch_qkvr_prep(d_q_, d_k_, d_v_, d_r_, L.q_scale.d_ptr, L.rel_scale.d_ptr, L.q_scale.f32, d_kc_[v_idx],
                           d_vc_[v_idx], T, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.d_rel, pos0, kv_cap, L.is_swa,
                           cfg_.qk_clip, cfg_.rel_clip, cfg_.rms_eps, cfg_.log_floor, cfg_.log_alpha, stream_);
    ink2::launch_rel_bias(d_r_, L.rel_proj.d_ptr, L.rel_proj.f32, d_rb_, T, cfg_.hq, cfg_.d_rel, cfg_.rel_extent,
                          stream_);

    if (T == 1) {
      ink2::launch_attn_decode(d_q_, d_kc_[v_idx], d_vc_[v_idx], d_rb_, d_sinks_, d_gate_attn_, d_part_m_, d_part_l_,
                               d_part_o_, d_att_out_, pos0, kv_cap, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.rel_extent,
                               L.is_swa, cfg_.swa, stream_);
    } else {
      ink2::launch_attn_prefill(d_q_, d_kc_[v_idx], d_vc_[v_idx], d_rb_, d_sinks_, d_gate_attn_, d_att_out_, T, pos0,
                                kv_cap, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.rel_extent, L.is_swa, cfg_.swa, stream_);
    }

    mm(L.wo_ud, d_att_out_, d_a_out_);
    ink2::launch_sconv_static(d_a_out_, L.attn_sconv.d_ptr, L.attn_sconv.f32, conv_a, d_a_out_, T, H,
                              sconv_static_scale, stream_);

    if (ip % cfg_.block_sz == 0) {
      const int bslot = ip / cfg_.block_sz;
      cuda_ok(cudaMemcpyAsync(d_blocks_[bslot], d_inp_sa_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "save block");
      blocks_vec.push_back(d_blocks_[bslot]);
      cuda_ok(cudaMemcpyAsync(d_ffn_inp_, d_a_out_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "ffn_inp");
    } else {
      cuda_ok(cudaMemcpyAsync(d_ffn_inp_, d_inp_sa_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "ffn_inp copy");
      ink2::launch_add(d_a_out_, d_ffn_inp_, T * H, stream_);
    }

    {
      std::vector<const float*> st = blocks_vec;
      st.push_back(d_ffn_inp_);
      const Ink2Tensor& mq = mlp_queries_[size_t(ip)];
      ink2::launch_attn_res(st.data(), int(st.size()), mq.d_ptr, mq.f32, 1, d_h_mlp_, T, H, cfg_.rms_eps, ar_scale,
                            stream_);
    }
    if (has_msgs) {
      ink2::launch_msg_gate(d_h_mlp_, d_h_norm_, static_cast<const __nv_bfloat16*>(L.ctrl_mlp.down.d_ptr),
                            static_cast<const __nv_bfloat16*>(L.ctrl_mlp.forget.d_ptr),
                            static_cast<const __nv_bfloat16*>(L.ctrl_mlp.gain.d_ptr), d_h_mlp_, T, H, cfg_.gate_rank,
                            cfg_.rms_eps, stream_);
    }
    ink2::launch_rmsnorm(d_h_mlp_, L.mlp_norm.d_ptr, L.mlp_norm.f32, d_x_mlp_, T, H, cfg_.rms_eps, stream_);

    mm(L.router_w, d_x_mlp_, d_rlog_);
    ink2::launch_moe_route(d_rlog_, L.router_bias.d_ptr, L.router_bias.f32, d_eid_, d_ew_, T, cfg_.n_experts,
                           cfg_.top_k, cfg_.route_scale, stream_);
    if (!L.is_mxfp8) {
      ink2::launch_moe_nvfp4(d_x_mlp_, d_eid_, d_ew_, L.w13, L.w13_sc, L.w13_outer, L.w2, L.w2_sc, L.w2_outer,
                             d_moe_act_, d_moe_out_, T, H, cfg_.moe_I, cfg_.top_k, stream_);
    } else {
      ink2::launch_moe_mxfp8(d_x_mlp_, d_eid_, d_ew_, L.w13, L.w13_sc, L.w2, L.w2_sc, d_moe_act_, d_moe_out_, T, H,
                             cfg_.moe_I, cfg_.top_k, stream_);
    }

    ink2::launch_sconv_static(d_moe_out_, L.mlp_sconv.d_ptr, L.mlp_sconv.f32, conv_m, d_moe_out_, T, H,
                              sconv_static_scale, stream_);
    cuda_ok(cudaMemcpyAsync(d_x_, d_ffn_inp_, row_bytes, cudaMemcpyDeviceToDevice, stream_), "x residual");
    ink2::launch_add(d_moe_out_, d_x_, T * H, stream_);
  }

  if (!need_logits) return;

  {
    std::vector<const float*> st = blocks_vec;
    st.push_back(d_x_);
    ink2::launch_attn_res(st.data(), int(st.size()), output_query_.d_ptr, output_query_.f32, 1, d_ar_out_, T, H,
                          cfg_.rms_eps, ar_scale, stream_);
  }
  ink2::launch_rmsnorm(d_ar_out_, final_norm_.d_ptr, final_norm_.f32, d_h_norm_, T, H, cfg_.rms_eps, stream_);

  const float* h_last = d_h_norm_ + size_t(T - 1) * size_t(H);
  ink2::launch_linear(cublas_, h_last, unembed_.d_ptr, unembed_.f32, d_x_bf16_, d_logits_, 1, cfg_.vocab, H, false,
                      stream_);
  ink2::launch_softcap(d_logits_, cfg_.vocab, cfg_.logit_softcap, stream_);
  if (suppress_active_) ink2::launch_suppress_mask(d_logits_, cfg_.vocab, d_suppress_, stream_);
  logits_valid_ = true;
}

int32_t EngineInk2::sample_logits(int64_t step_pos) {
  if (temp_ <= 0.0f || top_k_ == 1) {
    ink2::launch_argmax_one(d_logits_, cfg_.vocab, d_tok_out_, d_blk_v_, d_blk_i_, stream_);
    cuda_ok(cudaMemcpyAsync(h_tok_, d_tok_out_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_), "copy tok");
    cuda_ok(cudaStreamSynchronize(stream_), "sync argmax");
    return *h_tok_;
  }

  const int V = cfg_.vocab;
  cuda_ok(cudaMemcpyAsync(h_logits_, d_logits_, size_t(V) * sizeof(float), cudaMemcpyDeviceToHost, stream_),
          "copy logits for sampling");
  cuda_ok(cudaStreamSynchronize(stream_), "sync sample logits");

  const int k = top_k_ <= 0 ? V : std::min(top_k_, V);
  std::vector<std::pair<float, int32_t>> cand;
  cand.reserve(size_t(V));
  for (int i = 0; i < V; ++i) {
    const float v = h_logits_[i];
    if (std::isfinite(v)) cand.emplace_back(v, i);
  }
  if (cand.empty()) return 0;
  const int take = std::min(k, int(cand.size()));
  auto cmp = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  };
  if (take < int(cand.size())) {
    std::partial_sort(cand.begin(), cand.begin() + take, cand.end(), cmp);
    cand.resize(size_t(take));
  } else {
    std::sort(cand.begin(), cand.end(), cmp);
  }

  std::vector<double> w(static_cast<size_t>(take));
  double tot = 0.0;
  const double max_l = cand[0].first;
  for (int j = 0; j < take; ++j) {
    w[size_t(j)] = std::exp((double(cand[size_t(j)].first) - max_l) / double(temp_));
    tot += w[size_t(j)];
  }
  int n = take;
  if (top_p_ > 0.0f && top_p_ < 1.0f) {
    double run = 0.0;
    for (int j = 0; j < take; ++j) {
      run += w[size_t(j)];
      if (run >= double(top_p_) * tot) {
        n = j + 1;
        break;
      }
    }
  }
  if (min_p_ > 0.0f) {
    int m = 1;
    while (m < n && w[size_t(m)] >= double(min_p_)) ++m;
    n = m;
  }
  double kept = 0.0;
  for (int j = 0; j < n; ++j) kept += w[size_t(j)];
  const double u = sample_uniform(seed_, step_pos) * kept;
  double run = 0.0;
  int pick = n - 1;
  for (int j = 0; j < n; ++j) {
    run += w[size_t(j)];
    if (u < run) {
      pick = j;
      break;
    }
  }
  return cand[size_t(pick)].second;
}

int32_t EngineInk2::prefill(const int32_t* ids, int n) {
  if (!ids || n <= 0) throw std::runtime_error("prefill requires >= 1 token");
  if (pos_ + n > max_ctx_) throw std::runtime_error("prompt exceeds max_ctx");
  int off = 0;
  while (off < n) {
    const int c = std::min(kPrefillChunk, n - off);
    const bool last = (off + c == n);
    forward_tokens(ids + off, c, pos_, last);
    pos_ += c;
    off += c;
  }
  live_tokens_.insert(live_tokens_.end(), ids, ids + n);
  last_tok_ = sample_logits(pos_);
  return last_tok_;
}

void EngineInk2::decode(int32_t* out, int steps) {
  if (last_tok_ < 0) throw std::runtime_error("decode called before prefill");
  for (int i = 0; i < steps; ++i) {
    const int32_t tok = last_tok_;
    forward_tokens(&tok, 1, pos_, true);
    ++pos_;
    live_tokens_.push_back(tok);
    last_tok_ = sample_logits(pos_);
    if (out) out[i] = last_tok_;
  }
}

void EngineInk2::logits(float* out) {
  if (!logits_valid_) throw std::runtime_error("logits requested before forward pass");
  cuda_ok(cudaMemcpyAsync(out, d_logits_, size_t(cfg_.vocab) * sizeof(float), cudaMemcpyDeviceToHost, stream_),
          "read logits");
  cuda_ok(cudaStreamSynchronize(stream_), "sync logits");
}

void EngineInk2::generate(const int32_t* prompt, int n_prompt, int n_gen, int eos1, int eos2,
                          const std::atomic<int>* cancel, int32_t* out_ids, int64_t& produced_out, int& eos_out,
                          int& cached_out, int& passes_out, double& prefill_ms, double& gen_ms) {
  using clk = std::chrono::steady_clock;
  using ms = std::chrono::duration<double, std::milli>;

  if (!prompt || n_prompt <= 0) throw std::runtime_error("empty prompt");
  if (n_prompt >= max_ctx_) throw std::runtime_error("prompt exceeds max_ctx");

  const auto is_eos = [&](int32_t t) { return (eos1 != 0x7fffffff && t == eos1) || (eos2 != 0x7fffffff && t == eos2); };

  const auto t0 = clk::now();
  int cached = 0;
  if (!live_tokens_.empty() && int(live_tokens_.size()) < n_prompt &&
      std::equal(live_tokens_.begin(), live_tokens_.end(), prompt)) {
    cached = int(live_tokens_.size());
  } else if (!live_tokens_.empty() || pos_ != 0) {
    reset();
  }
  cached_out = cached;

  const int rem = n_prompt - cached;
  int off = 0;
  int passes = 0;
  while (off < rem) {
    if (cancel && cancel->load(std::memory_order_relaxed)) break;
    const int c = std::min(kPrefillChunk, rem - off);
    const bool last = (off + c == rem);
    forward_tokens(prompt + cached + off, c, pos_, last);
    pos_ += c;
    off += c;
    ++passes;
  }
  live_tokens_.insert(live_tokens_.end(), prompt + cached, prompt + cached + off);

  produced_out = 0;
  eos_out = 0;
  if (off < rem || (cancel && cancel->load(std::memory_order_relaxed))) {
    const auto t1 = clk::now();
    prefill_ms = ms(t1 - t0).count();
    gen_ms = 0.0;
    passes_out = passes;
    return;
  }

  int32_t cur = sample_logits(pos_);
  last_tok_ = cur;
  __atomic_store_n(&out_ids[0], cur, __ATOMIC_RELEASE);
  const auto t1 = clk::now();
  prefill_ms = ms(t1 - t0).count();

  if (is_eos(cur)) {
    eos_out = 1;
    gen_ms = 0.0;
    passes_out = passes;
    return;
  }

  for (int i = 0; i < n_gen; ++i) {
    if (cancel && cancel->load(std::memory_order_relaxed)) break;
    if (pos_ + 1 >= max_ctx_) break;
    forward_tokens(&cur, 1, pos_, true);
    ++pos_;
    live_tokens_.push_back(cur);
    ++passes;
    const int32_t nxt = sample_logits(pos_);
    last_tok_ = nxt;
    __atomic_store_n(&out_ids[1 + i], nxt, __ATOMIC_RELEASE);
    produced_out = i + 1;
    cur = nxt;
    if (is_eos(nxt)) {
      eos_out = 1;
      break;
    }
  }
  const auto t2 = clk::now();
  gen_ms = ms(t2 - t1).count();
  passes_out = passes;
}

}  // namespace yskn
