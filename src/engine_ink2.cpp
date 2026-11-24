// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "engine_ink2.hpp"

#include "ink2_kernels.cuh"
#include "safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
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

constexpr uint64_t kSaltDraft = 0x27D4EB2F165667C5ull;
constexpr uint64_t kSaltAccept = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kSaltResidual = 0x517CC1B727220A95ull;

struct SparseRowDist {
  std::vector<std::pair<int32_t, double>> items;
  double prob(int32_t id) const {
    for (const auto& kv : items) {
      if (kv.first == id) return kv.second;
    }
    return 0.0;
  }
};

// Matches EngineInk2::sample_logits truncation & normalization (top_k -> temperature/top_p -> min_p -> normalize).
SparseRowDist build_row_dist(const float* logits, int V, float temp, int top_k, float top_p, float min_p) {
  SparseRowDist d;
  if (!logits || V <= 0) return d;
  const int k = top_k <= 0 ? V : std::min(top_k, V);
  struct Cand {
    float v;
    int32_t id;
  };
  auto better = [](const Cand& a, const Cand& b) { return a.v > b.v || (a.v == b.v && a.id < b.id); };

  std::vector<Cand> cand;
  if (k < V && k <= 256) {
    cand.reserve(size_t(k));
    for (int i = 0; i < V; ++i) {
      const float v = logits[i];
      if (!std::isfinite(v)) continue;
      const Cand c{v, int32_t(i)};
      if (int(cand.size()) < k) {
        cand.push_back(c);
        std::push_heap(cand.begin(), cand.end(), better);
      } else if (better(c, cand[0])) {
        std::pop_heap(cand.begin(), cand.end(), better);
        cand.back() = c;
        std::push_heap(cand.begin(), cand.end(), better);
      }
    }
    std::sort(cand.begin(), cand.end(), better);
  } else {
    cand.reserve(size_t(V));
    for (int i = 0; i < V; ++i) {
      const float v = logits[i];
      if (std::isfinite(v)) cand.emplace_back(Cand{v, int32_t(i)});
    }
    const int take = std::min(k, int(cand.size()));
    if (take < int(cand.size())) {
      std::partial_sort(cand.begin(), cand.begin() + take, cand.end(), better);
      cand.resize(size_t(take));
    } else {
      std::sort(cand.begin(), cand.end(), better);
    }
  }
  if (cand.empty()) {
    throw std::runtime_error("Inkling-2: no finite logit to sample from (the weights or a kernel gave NaN or inf)");
  }

  const int take = int(cand.size());
  std::vector<double> w(static_cast<size_t>(take));
  double tot = 0.0;
  const double max_l = cand[0].v;
  for (int j = 0; j < take; ++j) {
    w[size_t(j)] = std::exp((double(cand[size_t(j)].v) - max_l) / double(temp));
    tot += w[size_t(j)];
  }
  int n = take;
  if (top_p > 0.0f && top_p < 1.0f) {
    double run = 0.0;
    for (int j = 0; j < take; ++j) {
      run += w[size_t(j)];
      if (run >= double(top_p) * tot) {
        n = j + 1;
        break;
      }
    }
  }
  if (min_p > 0.0f) {
    int m = 1;
    while (m < n && w[size_t(m)] >= double(min_p)) ++m;
    n = m;
  }
  double kept = 0.0;
  for (int j = 0; j < n; ++j) kept += w[size_t(j)];
  const double inv = kept > 0.0 ? (1.0 / kept) : 1.0;
  d.items.reserve(size_t(n));
  for (int j = 0; j < n; ++j) d.items.emplace_back(cand[size_t(j)].id, w[size_t(j)] * inv);
  return d;
}

int32_t draw_from_dist(const SparseRowDist& d, uint64_t seed, int64_t step) {
  if (d.items.empty()) return 0;
  const double u = sample_uniform(seed, step);
  double run = 0.0;
  for (size_t j = 0; j + 1 < d.items.size(); ++j) {
    run += d.items[j].second;
    if (u < run) return d.items[j].first;
  }
  return d.items.back().first;
}

bool parse_mtp_config(const std::string& dir, Ink2MtpConfig& out) {
  const stjson::Val root = stjson::parse(slurp_file(dir + "/config.json"));
  const stjson::Val* mp = nullptr;
  if (root.has("mtp_config")) mp = &root.at("mtp_config");
  else if (root.has("text_config") && root.at("text_config").has("mtp_config")) mp = &root.at("text_config").at("mtp_config");
  if (!mp || mp->kind != stjson::Val::Obj) return false;
  auto gi = [](const stjson::Val& v, int def) { return v.kind == stjson::Val::Num ? int(v.i) : def; };
  auto gf = [](const stjson::Val& v, float def) { return v.kind == stjson::Val::Num ? float(v.num) : def; };
  out.hidden = gi(mp->at("hidden_size"), 2304);
  out.num_layers = gi(mp->at("num_hidden_layers"), 3);
  out.hq = gi(mp->at("num_attention_heads"), 24);
  out.hkv = gi(mp->at("num_key_value_heads"), 8);
  out.head_dim = gi(mp->at("head_dim"), 128);
  out.ffn = gi(mp->at("intermediate_size"), 9216);
  out.swa = gi(mp->at("sliding_window_size"), 1024);
  out.block = gi(mp->at("block_size"), 8);
  out.markov_rank = gi(mp->at("markov_rank"), 256);
  out.rms_eps = gf(mp->at("rms_norm_eps"), 1e-6f);
  out.rope_theta = gf(mp->at("rope_theta"), 1e6f);
  for (int i = 0; i < 3; ++i) out.is_local[i] = false;
  const stjson::Val& ll = mp->at("local_layer_ids");
  if (ll.kind == stjson::Val::Arr) {
    for (const auto& lid : ll.arr) {
      if (lid.i >= 0 && lid.i < 3) out.is_local[lid.i] = true;
    }
  }
  const stjson::Val& tl = mp->at("target_layer_ids");
  if (tl.kind == stjson::Val::Arr && tl.arr.size() >= 4) {
    for (int i = 0; i < 4; ++i) out.taps[i] = int(tl.arr[size_t(i)].i);
  }
  return out.num_layers == 3 && out.block == 8;
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

// The kernels and buffers of this engine are written for one model shape: Inkling-2 Mini (fixed-size arrays in
// ink2_kernels.cu and engine_ink2.hpp, the visit schedule, the expert formats). A config.json of another shape is
// refused before anything is loaded.
void check_mini_shape(const Ink2Config& c, const stjson::Val& t, const std::string& dir) {
  const Ink2Config m;  // the defaults are the values of Inkling-2 Mini
  std::string bad;
  auto eq = [&](const char* key, int got, int want) {
    if (got != want)
      bad += std::string(bad.empty() ? "" : ", ") + key + " " + std::to_string(got) + " (needs " +
             std::to_string(want) + ")";
  };
  eq("hidden_size", c.hidden, m.hidden);
  eq("num_hidden_layers", c.num_layers, m.num_layers);
  eq("vocab_size", c.vocab_padded, m.vocab_padded);
  eq("unpadded_vocab_size", c.vocab, m.vocab);
  eq("num_attention_heads", c.hq, m.hq);
  eq("num_key_value_heads", c.hkv, m.hkv);
  eq("head_dim", c.head_dim, m.head_dim);
  eq("d_rel", c.d_rel, m.d_rel);
  eq("rel_extent", c.rel_extent, m.rel_extent);
  eq("sliding_window_size", c.swa, m.swa);
  eq("n_layers_per_attn_res_block", c.block_sz, m.block_sz);
  eq("n_routed_experts", c.n_experts, m.n_experts);
  eq("num_experts_per_tok", c.top_k, m.top_k);
  eq("intermediate_size", c.moe_I, m.moe_I);
  eq("loop_config.core_start", c.core_start, m.core_start);
  eq("loop_config.core_end", c.core_end, m.core_end);
  eq("loop_config.num_passes", c.num_passes, m.num_passes);
  eq("loop_config.gate_rank", c.gate_rank, m.gate_rank);
  auto opt_eq = [&](const stjson::Val& v, const char* key, int want) {
    if (v.kind == stjson::Val::Num) eq(key, int(v.i), want);
    else if (v.kind == stjson::Val::Bool) eq(key, v.b ? 1 : 0, want);
  };
  opt_eq(t.at("sconv_kernel_size"), "sconv_kernel_size", 4);
  opt_eq(t.at("n_shared_experts"), "n_shared_experts", 0);
  opt_eq(t.at("loop_config").at("is_varloop"), "loop_config.is_varloop", 0);
  if (bad.empty()) {  // the full-attention layers of Mini: 5, 11, 17, 23 (one per block of 6); every other layer is local
    for (int i = 0; i < c.num_layers; ++i)
      if (c.is_local[size_t(i)] != (i % 6 != 5)) {
        bad = "local_layer_ids (needs every layer but 5, 11, 17 and 23)";
        break;
      }
  }
  if (!bad.empty())
    throw std::runtime_error("config.json in " + dir + ": this Inkling-2 engine runs the shape of Inkling-2 Mini only; " +
                             bad);
}

void check_mtp_shape(const Ink2MtpConfig& mc, const std::string& dir) {
  const Ink2MtpConfig m;
  std::string bad;
  auto eq = [&](const char* key, int got, int want) {
    if (got != want)
      bad += std::string(bad.empty() ? "" : ", ") + key + " " + std::to_string(got) + " (needs " +
             std::to_string(want) + ")";
  };
  eq("mtp_config.hidden_size", mc.hidden, m.hidden);
  eq("mtp_config.num_hidden_layers", mc.num_layers, m.num_layers);
  eq("mtp_config.num_attention_heads", mc.hq, m.hq);
  eq("mtp_config.num_key_value_heads", mc.hkv, m.hkv);
  eq("mtp_config.head_dim", mc.head_dim, m.head_dim);
  eq("mtp_config.intermediate_size", mc.ffn, m.ffn);
  eq("mtp_config.sliding_window_size", mc.swa, m.swa);
  eq("mtp_config.block_size", mc.block, m.block);
  eq("mtp_config.markov_rank", mc.markov_rank, m.markov_rank);
  if (bad.empty()) {
    for (int i = 0; i < m.num_layers; ++i)
      if (mc.is_local[i] != m.is_local[i]) {
        bad = "mtp_config.local_layer_ids (needs [0, 2])";
        break;
      }
  }
  if (bad.empty()) {
    for (int i = 0; i < 4; ++i)
      if (mc.taps[i] < 0 || mc.taps[i] > 48) {
        bad = "mtp_config.target_layer_ids out of range [0..48]";
        break;
      }
  }
  if (!bad.empty())
    throw std::runtime_error("config.json in " + dir +
                             ": this Inkling-2 MTP drafter runs the shape of Inkling-2 Mini only; " + bad);
}

// The dtype and shape the engine needs for one tensor (it sizes its buffers and kernels by the model shape, not the file).
void expect_tensor(const StTensorInfo& t, const std::string& name, const char* dtype,
                   std::initializer_list<int64_t> shape) {
  if (t.dtype == dtype && t.shape.size() == shape.size() &&
      std::equal(t.shape.begin(), t.shape.end(), shape.begin()) &&
      t.nbytes == t.numel() * st_dtype_bytes(t.dtype))
    return;
  auto dims = [](const std::vector<int64_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
  };
  throw std::runtime_error("Inkling-2 weights: " + name + " is " + t.dtype + " " + dims(t.shape) + " (" +
                           std::to_string(t.nbytes) + " bytes); this engine needs " + dtype + " " +
                           dims(std::vector<int64_t>(shape)) + " (Inkling-2 Mini)");
}

struct DevGuard {
  void* ptr = nullptr;
  DevGuard() = default;
  ~DevGuard() {
    if (ptr) cudaFree(ptr);
  }
  DevGuard(const DevGuard&) = delete;
  DevGuard& operator=(const DevGuard&) = delete;
};

}  // namespace

int ink2_accept_tokens(int32_t* out_accepted_and_next, const int32_t* draft_tokens, int K, int V,
                       const float* verify_logits, const float* draft_logits, const int32_t* verify_argmax,
                       float temp, int top_k, float top_p, float min_p, uint64_t seed, int64_t step) {
  if (K <= 0) {
    if (out_accepted_and_next) {
      if (temp <= 0.0f || top_k == 1) out_accepted_and_next[0] = verify_argmax[0];
      else out_accepted_and_next[0] = draw_from_dist(build_row_dist(verify_logits, V, temp, top_k, top_p, min_p), seed, step);
    }
    return 0;
  }

  if (temp <= 0.0f || top_k == 1) {
    int a = 0;
    while (a < K && draft_tokens[a] == verify_argmax[a]) {
      out_accepted_and_next[a] = draft_tokens[a];
      ++a;
    }
    out_accepted_and_next[a] = verify_argmax[a];
    return a;
  }

  int a = 0;
  for (; a < K; ++a) {
    const int32_t dt = draft_tokens[a];
    const SparseRowDist p = build_row_dist(verify_logits + size_t(a) * size_t(V), V, temp, top_k, top_p, min_p);
    const SparseRowDist q = build_row_dist(draft_logits + size_t(a) * size_t(V), V, temp, top_k, top_p, min_p);
    const double pa = p.prob(dt);
    const double qa = q.prob(dt);
    const double u_acc = sample_uniform(seed ^ kSaltAccept, step + a);
    if (qa > 0.0 && u_acc * qa < pa) {
      out_accepted_and_next[a] = dt;
      continue;
    }

    // Rejected: draw residual max(0, p(x) - q(x)) using independent residual salt distinct from draft & u_acc.
    std::vector<std::pair<int32_t, double>> res;
    res.reserve(p.items.size());
    double rsum = 0.0;
    for (const auto& kv : p.items) {
      const double r = kv.second - q.prob(kv.first);
      if (r > 0.0) {
        res.emplace_back(kv.first, r);
        rsum += r;
      }
    }
    int32_t chosen = p.items.empty() ? 0 : p.items[0].first;
    if (rsum > 0.0) {
      const double target = sample_uniform(seed ^ kSaltResidual, step + a) * rsum;
      double run = 0.0;
      chosen = res.back().first;
      for (const auto& kv : res) {
        run += kv.second;
        if (target < run) {
          chosen = kv.first;
          break;
        }
      }
    }
    out_accepted_and_next[a] = chosen;
    return a;
  }

  const SparseRowDist p_bonus =
      build_row_dist(verify_logits + size_t(K) * size_t(V), V, temp, top_k, top_p, min_p);
  out_accepted_and_next[K] = draw_from_dist(p_bonus, seed, step + K);
  return K;
}

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
  try {
  dir_ = resolve_dir(path);
  if (!is_inkling2_dir(dir_)) throw std::runtime_error(path + " is not an Inkling-2 checkpoint");
  cfg_ = parse_ink2_config(dir_);
  {
    const stjson::Val root = stjson::parse(slurp_file(dir_ + "/config.json"));
    check_mini_shape(cfg_, root.has("text_config") ? root.at("text_config") : root, dir_);
  }
  max_ctx_ = max_ctx > 0 ? max_ctx : 32768;
  ring_rows_ = (cfg_.swa + kPrefillChunk + 15) / 16 * 16;

  int dev = 0;
  cuda_ok(cudaGetDevice(&dev), "cudaGetDevice");
  cuda_ok(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
  cuda_ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");

  SafetensorsReader reader;
  reader.open_dir(dir_, "model.llm.");

  const int64_t H = cfg_.hidden, QD = int64_t(cfg_.hq) * cfg_.head_dim, KVD = int64_t(cfg_.hkv) * cfg_.head_dim;
  const int64_t NE = cfg_.n_experts, I2 = 2 * int64_t(cfg_.moe_I);
  auto upload_tensor = [&](const std::string& name, const char* dtype, std::initializer_list<int64_t> shape) -> Ink2Tensor {
    const StTensorInfo& info = reader.get(name);
    expect_tensor(info, name, dtype, shape);
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

  auto upload_raw = [&](const std::string& name, const char* dtype, std::initializer_list<int64_t> shape) -> uint8_t* {
    const StTensorInfo& info = reader.get(name);
    expect_tensor(info, name, dtype, shape);
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

  embed_ = upload_tensor("model.llm.embed.weight", "BF16", {cfg_.vocab_padded, H});
  embed_norm_ = upload_tensor("model.llm.embed_norm.weight", "BF16", {H});
  final_norm_ = upload_tensor("model.llm.norm.weight", "BF16", {H});

  if (const char* eh = std::getenv("YSKN_INK2_HEAD")) head_mode_ = eh;
  use_q8_head_ = (head_mode_ == "q8_0" || head_mode_ == "Q8_0");

  bool mtp_want = std::filesystem::exists(dir_ + "/mtp.safetensors");
  if (const char* em = std::getenv("YSKN_INK2_MTP")) {
    if (std::strcmp(em, "0") == 0) mtp_want = false;
  }
  if (mtp_want) mtp_enabled_ = parse_mtp_config(dir_, mtp_cfg_);
  if (const char* mq8 = std::getenv("YSKN_INK2_MTP_Q8")) mtp_q8_ = (std::strcmp(mq8, "0") != 0);
  if (mtp_cfg_.markov_rank != 256) mtp_q8_ = false;

  bool screen_want = !use_q8_head_ && (cfg_.hidden == 2304);
  if (const char* es = std::getenv("YSKN_INK2_SCREEN_HEAD")) {
    if (std::strcmp(es, "0") == 0) screen_want = false;
  }
  use_screen_head_ = screen_want && !use_q8_head_ && (cfg_.hidden == 2304);

  bool tree_want = mtp_enabled_ && use_screen_head_ && !use_q8_head_ && (mtp_cfg_.markov_rank == 256) && (cfg_.hidden == 2304);
  if (const char* et = std::getenv("YSKN_INK2_TREE")) {
    if (std::strcmp(et, "0") == 0) tree_want = false;
  }
  use_tree_ = tree_want && use_screen_head_ && mtp_enabled_ && !use_q8_head_ && (mtp_cfg_.markov_rank == 256) && (cfg_.hidden == 2304);
  // Across 48 visits, reading all 128 NVFP4/MXFP8 experts moves ~33 GB vs ~2.5 GB fixed dense+head overhead (~13x byte ratio);
  // bandwidth-bound low-SM GPUs (e.g. GB10 with 48 SMs) benefit from tighter trees (M<=6) and higher MoE cost weighting.
  moe_ratio_ = (sm_count_ <= 64) ? 15.0f : 12.0f;
  tree_max_ = (sm_count_ <= 64) ? 6 : kMaxTree;
  if (const char* er = std::getenv("YSKN_INK2_MOE_RATIO")) {
    const float r = std::strtof(er, nullptr);
    if (std::isfinite(r) && r > 0.0f) moe_ratio_ = r;
  }
  if (const char* emx = std::getenv("YSKN_INK2_TREE_MAX")) {
    const int m = std::atoi(emx);
    if (m >= 2) tree_max_ = std::min(kMaxTree, m);
  }
  if (const char* ekv = std::getenv("YSKN_INK2_KV")) {
    kv_bf16_ = (std::strcmp(ekv, "bf16") == 0 || std::strcmp(ekv, "BF16") == 0 || std::strcmp(ekv, "1") == 0);
  }
  if (const char* esuf = std::getenv("YSKN_INK2_SUFFIX_DRAFT")) {
    use_suffix_draft_ = (std::strcmp(esuf, "0") != 0);
  }

  DevGuard tmp_u_bf16;
  const __nv_bfloat16* u_bf16_for_fold = nullptr;
  if (use_q8_head_) {
    const StTensorInfo& uinfo = reader.get("model.llm.unembed.weight");
    expect_tensor(uinfo, "model.llm.unembed.weight", "BF16", {cfg_.vocab_padded, H});
    if ((cfg_.hidden & 31) != 0) {
      use_q8_head_ = false;
    } else {
      cuda_ok(cudaMalloc(&tmp_u_bf16.ptr, uinfo.nbytes), "cudaMalloc tmp unembed");
      reader.upload_gpu(uinfo, tmp_u_bf16.ptr, stream_);
      u_bf16_for_fold = static_cast<const __nv_bfloat16*>(tmp_u_bf16.ptr);
      const int V = cfg_.vocab;
      const int H_dim = cfg_.hidden;
      d_unembed_q8_ = static_cast<int8_t*>(alloc_gpu(size_t(V) * size_t(H_dim)));
      d_unembed_q8_sc_ = static_cast<__half*>(alloc_gpu(size_t(V) * size_t(H_dim / 32) * sizeof(__half)));
      ink2::launch_quantize_q8_32(static_cast<const __nv_bfloat16*>(tmp_u_bf16.ptr), d_unembed_q8_, d_unembed_q8_sc_, V,
                                  H_dim, stream_);
      cuda_ok(cudaStreamSynchronize(stream_), "sync quantize q8 unembed");
    }
  }
  if (!use_q8_head_) {
    unembed_ = upload_tensor("model.llm.unembed.weight", "BF16", {cfg_.vocab_padded, H});
    u_bf16_for_fold = static_cast<const __nv_bfloat16*>(unembed_.d_ptr);
    if (use_screen_head_ || use_tree_) {
      const int V_scr = cfg_.vocab;
      const int H_scr = cfg_.hidden;
      d_screen_q8_ = static_cast<int8_t*>(alloc_gpu(size_t(V_scr) * size_t(H_scr)));
      d_screen_sc_ = static_cast<float*>(alloc_gpu(size_t(V_scr) * sizeof(float)));
      d_screen_eps_ = static_cast<float*>(alloc_gpu(size_t(V_scr) * sizeof(float)));
      ink2::launch_build_unembed_screen_bf16(u_bf16_for_fold, d_screen_q8_, d_screen_sc_, d_screen_eps_, V_scr, H_scr,
                                             stream_);
      if (use_screen_head_) {
        d_screen_hnorm_ = static_cast<float*>(alloc_gpu(8 * sizeof(float)));
        d_screen_max_lb_ = static_cast<int*>(alloc_gpu(8 * sizeof(int)));
        d_screen_dot_hat_ = static_cast<float*>(alloc_gpu(8 * size_t(V_scr) * sizeof(float)));
        d_screen_cand_ids_ = static_cast<int32_t*>(alloc_gpu(8 * size_t(kScreenMaxCand) * sizeof(int32_t)));
        d_screen_cand_cnt_ = static_cast<int32_t*>(alloc_gpu(8 * sizeof(int32_t)));
      }
      cuda_ok(cudaStreamSynchronize(stream_), "sync build unembed screen");
    }
  }

  attn_queries_.resize(size_t(cfg_.num_layers - 1));
  for (int i = 0; i < cfg_.num_layers - 1; ++i) {  // the queries before a full-attention layer read 3 inputs (q, k, v)
    const std::string n = "model.llm.attn_res.attn_queries." + std::to_string(i);
    attn_queries_[size_t(i)] =
        cfg_.is_local[size_t(i + 1)] ? upload_tensor(n, "F32", {H}) : upload_tensor(n, "F32", {3, H});
  }
  mlp_queries_.resize(size_t(cfg_.num_layers));
  for (int i = 0; i < cfg_.num_layers; ++i)
    mlp_queries_[size_t(i)] = upload_tensor("model.llm.attn_res.mlp_queries." + std::to_string(i), "F32", {H});
  output_query_ = upload_tensor("model.llm.attn_res.output_query", "F32", {H});

  // Pre-interleave [ncore, 4, H] middle cycle cross-pass queries: (attn, k, v, mlp)
  const int ncore = cfg_.core_end - cfg_.core_start;
  const StTensorInfo& qam_info = reader.get("model.llm.middle_cycle_cross_pass_queries");
  const StTensorInfo& qkv_info = reader.get("model.llm.middle_cycle_cross_pass_kv_queries");
  expect_tensor(qam_info, "model.llm.middle_cycle_cross_pass_queries", "F32", {2 * int64_t(ncore), H});
  expect_tensor(qkv_info, "model.llm.middle_cycle_cross_pass_kv_queries", "F32", {2 * int64_t(ncore), H});
  const std::vector<float> qam = read_f32_vec(qam_info);
  const std::vector<float> qkv = read_f32_vec(qkv_info);
  std::vector<float> mq(size_t(ncore) * 4 * size_t(H));
  for (int c = 0; c < ncore; ++c) {
    for (size_t j = 0; j < size_t(H); ++j) {
      const size_t h = size_t(H);
      mq[(size_t(c) * 4 + 0) * h + j] = qam[(size_t(c) * 2 + 0) * h + j];
      mq[(size_t(c) * 4 + 1) * h + j] = qkv[(size_t(c) * 2 + 0) * h + j];
      mq[(size_t(c) * 4 + 2) * h + j] = qkv[(size_t(c) * 2 + 1) * h + j];
      mq[(size_t(c) * 4 + 3) * h + j] = qam[(size_t(c) * 2 + 1) * h + j];
    }
  }
  d_msg_queries_ = static_cast<float*>(alloc_gpu(mq.size() * sizeof(float)));
  cuda_ok(cudaMemcpyAsync(d_msg_queries_, mq.data(), mq.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
          "upload msg_queries");
  cuda_ok(cudaStreamSynchronize(stream_), "sync msg_queries");

  auto make_outer = [&](const std::string& base) -> float* {
    const StTensorInfo& amax_info = reader.get(base + "_amax");
    const StTensorInfo& denom_info = reader.get(base + "_outer_scale_denominator");
    expect_tensor(amax_info, base + "_amax", "F32", {NE});
    expect_tensor(denom_info, base + "_outer_scale_denominator", "F32", {1});
    const std::vector<float> amax = read_f32_vec(amax_info);
    const std::vector<float> denom = read_f32_vec(denom_info);
    std::vector<float> out(size_t(cfg_.n_experts));
    for (int e = 0; e < cfg_.n_experts; ++e) out[size_t(e)] = amax[size_t(e)] / denom[0];
    float* dp = static_cast<float*>(alloc_gpu(out.size() * sizeof(float)));
    cuda_ok(cudaMemcpyAsync(dp, out.data(), out.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
            "upload outer_scale");
    cuda_ok(cudaStreamSynchronize(stream_), "sync outer_scale");
    return dp;
  };

  auto upload_into = [&](const std::string& name, void* dst, const char* dtype,
                         std::initializer_list<int64_t> shape) -> Ink2Tensor {
    const StTensorInfo& info = reader.get(name);
    expect_tensor(info, name, dtype, shape);
    Ink2Tensor t;
    t.f32 = false;
    t.rows = int(info.shape[0]);
    size_t col_prod = 1;
    for (size_t i = 1; i < info.shape.size(); ++i) col_prod *= size_t(info.shape[i]);
    t.cols = int(col_prod);
    t.bytes = info.nbytes;
    t.d_ptr = dst;
    reader.upload_gpu(info, t.d_ptr, stream_);
    return t;
  };

  layers_.resize(size_t(cfg_.num_layers));
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Ink2Layer& L = layers_[size_t(i)];
    const std::string p = "model.llm.layers." + std::to_string(i) + ".";
    L.is_swa = cfg_.is_local[size_t(i)];
    L.has_ctrl = (i >= cfg_.core_start && i < cfg_.core_end);
    const int64_t HQ = cfg_.hq, DREL = cfg_.d_rel, GR = cfg_.gate_rank;
    // Pack BF16 dense projections sharing normalized input so decode/prefill fuse into 1 launch.
    const auto& iq = reader.get(p + "attn.wq_du.weight");
    const auto& ir = reader.get(p + "attn.wr_du.weight");
    const auto& isink = reader.get(p + "attn.attn_sink_proj.weight");
    const auto& ig = reader.get(p + "attn.gate_attn.weight");
    const auto& ik = reader.get(p + "attn.wk_dv.weight");
    const auto& iv = reader.get(p + "attn.wv_dv.weight");
    expect_tensor(iq, p + "attn.wq_du.weight", "BF16", {QD, H});
    expect_tensor(ir, p + "attn.wr_du.weight", "BF16", {HQ * DREL, H});
    expect_tensor(isink, p + "attn.attn_sink_proj.weight", "BF16", {HQ, H});
    expect_tensor(ig, p + "attn.gate_attn.weight", "BF16", {QD, H});
    expect_tensor(ik, p + "attn.wk_dv.weight", "BF16", {KVD, H});
    expect_tensor(iv, p + "attn.wv_dv.weight", "BF16", {KVD, H});
    if (L.is_swa) {
      const size_t total_bytes = iq.nbytes + ir.nbytes + isink.nbytes + ig.nbytes + ik.nbytes + iv.nbytes;
      uint8_t* base = static_cast<uint8_t*>(alloc_gpu(total_bytes));
      size_t off = 0;
      L.wq_du = upload_into(p + "attn.wq_du.weight", base + off, "BF16", {QD, H}); off += iq.nbytes;
      L.wr_du = upload_into(p + "attn.wr_du.weight", base + off, "BF16", {HQ * DREL, H}); off += ir.nbytes;
      L.attn_sink_proj = upload_into(p + "attn.attn_sink_proj.weight", base + off, "BF16", {HQ, H}); off += isink.nbytes;
      L.gate_attn = upload_into(p + "attn.gate_attn.weight", base + off, "BF16", {QD, H}); off += ig.nbytes;
      L.wk_dv = upload_into(p + "attn.wk_dv.weight", base + off, "BF16", {KVD, H}); off += ik.nbytes;
      L.wv_dv = upload_into(p + "attn.wv_dv.weight", base + off, "BF16", {KVD, H});
      L.w_qkvrsg = reinterpret_cast<const __nv_bfloat16*>(base);
      L.fused_qkvrsg = true;
      L.fused_qrsg = true;
    } else {
      const size_t qrsg_bytes = iq.nbytes + ir.nbytes + isink.nbytes + ig.nbytes;
      uint8_t* base = static_cast<uint8_t*>(alloc_gpu(qrsg_bytes));
      size_t off = 0;
      L.wq_du = upload_into(p + "attn.wq_du.weight", base + off, "BF16", {QD, H}); off += iq.nbytes;
      L.wr_du = upload_into(p + "attn.wr_du.weight", base + off, "BF16", {HQ * DREL, H}); off += ir.nbytes;
      L.attn_sink_proj = upload_into(p + "attn.attn_sink_proj.weight", base + off, "BF16", {HQ, H}); off += isink.nbytes;
      L.gate_attn = upload_into(p + "attn.gate_attn.weight", base + off, "BF16", {QD, H});
      L.w_qkvrsg = reinterpret_cast<const __nv_bfloat16*>(base);
      L.fused_qrsg = true;
    }
    if (!L.fused_qkvrsg) {
      const auto& ikp = reader.get(p + "attn.k_sconv.weight_proj");
      const auto& ivp = reader.get(p + "attn.v_sconv.weight_proj");
      expect_tensor(ikp, p + "attn.k_sconv.weight_proj", "BF16", {int64_t(cfg_.hkv) * 4, H});
      expect_tensor(ivp, p + "attn.v_sconv.weight_proj", "BF16", {int64_t(cfg_.hkv) * 4, H});
      uint8_t* bk = static_cast<uint8_t*>(alloc_gpu(ik.nbytes + ikp.nbytes));
      L.wk_dv = upload_into(p + "attn.wk_dv.weight", bk, "BF16", {KVD, H});
      L.k_proj = upload_into(p + "attn.k_sconv.weight_proj", bk + ik.nbytes, "BF16", {int64_t(cfg_.hkv) * 4, H});
      L.w_kkp = reinterpret_cast<const __nv_bfloat16*>(bk);
      uint8_t* bv = static_cast<uint8_t*>(alloc_gpu(iv.nbytes + ivp.nbytes));
      L.wv_dv = upload_into(p + "attn.wv_dv.weight", bv, "BF16", {KVD, H});
      L.v_proj = upload_into(p + "attn.v_sconv.weight_proj", bv + iv.nbytes, "BF16", {int64_t(cfg_.hkv) * 4, H});
      L.w_vvp = reinterpret_cast<const __nv_bfloat16*>(bv);
      L.fused_kvp = true;
    }
    L.wo_ud = upload_tensor(p + "attn.wo_ud.weight", "BF16", {H, QD});
    L.q_scale = upload_tensor(p + "attn.q_l2_clip.scale", "BF16", {HQ, cfg_.head_dim});
    L.rel_scale = upload_tensor(p + "attn.rel_l2_clip.scale", "BF16", {HQ, DREL});
    L.rel_proj = upload_tensor(p + "attn.rel_logits_proj.proj", "BF16", {DREL, cfg_.rel_extent});
    L.mlp_norm = upload_tensor(p + "mlp_norm.weight", "BF16", {H});
    L.attn_sconv = upload_tensor(p + "attn_sconv.weight", "BF16", {H, 1, 4});
    L.mlp_sconv = upload_tensor(p + "mlp_sconv.weight", "BF16", {H, 1, 4});
    if (L.is_swa) {
      L.attn_norm = upload_tensor(p + "attn_norm.weight", "BF16", {H});
      L.k_sconv = upload_tensor(p + "attn.k_sconv.weight", "BF16", {KVD, 1, 4});
      L.v_sconv = upload_tensor(p + "attn.v_sconv.weight", "BF16", {KVD, 1, 4});
    } else {
      L.q_norm = upload_tensor(p + "q_input_norm.weight", "BF16", {H});
      L.k_norm = upload_tensor(p + "k_input_norm.weight", "BF16", {H});
      L.v_norm = upload_tensor(p + "v_input_norm.weight", "BF16", {H});
      L.k_static = upload_tensor(p + "attn.k_sconv.static_w", "BF16", {4, KVD});
      L.v_static = upload_tensor(p + "attn.v_sconv.static_w", "BF16", {4, KVD});
      if (!L.fused_kvp) {
        L.k_proj = upload_tensor(p + "attn.k_sconv.weight_proj", "BF16", {int64_t(cfg_.hkv) * 4, H});
        L.v_proj = upload_tensor(p + "attn.v_sconv.weight_proj", "BF16", {int64_t(cfg_.hkv) * 4, H});
      }
    }
    if (L.has_ctrl) {
      auto load_ctrl = [&](const char* name) -> Ink2Ctrl {
        const std::string c = p + "controllers." + name + ".";
        Ink2Ctrl r;
        r.down = upload_tensor(c + "down_weight", "BF16", {GR, 2 * H});
        r.forget = upload_tensor(c + "forget_weight", "BF16", {H, GR});
        r.gain = upload_tensor(c + "gain_weight", "BF16", {H, GR});
        return r;
      };
      L.ctrl_attn = load_ctrl("attn");
      L.ctrl_k = load_ctrl("k");
      L.ctrl_v = load_ctrl("v");
      L.ctrl_mlp = load_ctrl("mlp");
    }
    L.router_w = upload_tensor(p + "mlp.gate.weight", "BF16", {NE, H});
    L.router_bias = upload_tensor(p + "mlp.gate.bias", "F32", {NE});

    const std::string m = p + "mlp.experts.";
    const StTensorInfo& w13_info = reader.get(m + "w13_weight");
    L.is_mxfp8 = (w13_info.dtype == "F8_E4M3");
    if (!L.is_mxfp8 && w13_info.dtype != "U8")
      throw std::runtime_error("unexpected expert dtype " + w13_info.dtype);
    // NVFP4 (two values per byte, an E4M3 scale per 16) or MXFP8 (one E4M3 value per byte, an E8M0 scale byte per 32)
    const int64_t I = cfg_.moe_I;
    if (L.is_mxfp8) {
      L.w13 = upload_raw(m + "w13_weight", "F8_E4M3", {NE, I2, H});
      L.w13_sc = upload_raw(m + "w13_weight_scale", "U8", {NE, I2, H / 32});
      L.w2 = upload_raw(m + "w2_weight", "F8_E4M3", {NE, H, I});
      L.w2_sc = upload_raw(m + "w2_weight_scale", "U8", {NE, H, I / 32});
    } else {
      L.w13 = upload_raw(m + "w13_weight", "U8", {NE, I2, H / 2});
      L.w13_sc = upload_raw(m + "w13_weight_scale", "F8_E4M3", {NE, I2, H / 16});
      L.w2 = upload_raw(m + "w2_weight", "U8", {NE, H, I / 2});
      L.w2_sc = upload_raw(m + "w2_weight_scale", "F8_E4M3", {NE, H, I / 16});
    }
    if (!L.is_mxfp8) {
      L.w13_outer = make_outer(m + "w13_weight");
      L.w2_outer = make_outer(m + "w2_weight");
    }
  }
  reader.close_all();

  // KV caches (FP32 default, BF16 opt-in via YSKN_INK2_KV=bf16) & contiguous short-conv history across 48 visits
  const size_t nv = cfg_.schedule.size();
  const size_t kv_row_elems = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  const size_t kv_eb = kv_bf16_ ? sizeof(__nv_bfloat16) : sizeof(float);
  d_kc_.resize(nv);
  d_vc_.resize(nv);
  d_conv_.resize(nv);
  d_conv_base_ = static_cast<float*>(alloc_gpu(nv * size_t(kVisitConvFloats) * sizeof(float)));
  for (size_t v = 0; v < nv; ++v) {
    const int ip = cfg_.schedule[v];
    const size_t cap = cfg_.is_local[size_t(ip)] ? size_t(ring_rows_) : size_t(max_ctx_);
    d_kc_[v] = alloc_gpu(cap * kv_row_elems * kv_eb);
    d_vc_[v] = alloc_gpu(cap * kv_row_elems * kv_eb);
    d_conv_[v] = d_conv_base_ + v * size_t(kVisitConvFloats);
  }
  d_verify_conv_ = static_cast<float*>(alloc_gpu(size_t(kMtpBlock) * nv * size_t(kVisitConvFloats) * sizeof(float)));

  // Allocate activations sized for kPrefillChunk rows
  const size_t T = size_t(kPrefillChunk);
  const size_t V = size_t(cfg_.vocab);

  d_tokens_ = static_cast<int32_t*>(alloc_gpu(T * sizeof(int32_t)));
  d_tok_out_ = static_cast<int32_t*>(alloc_gpu(sizeof(int32_t)));
  d_blk_v_ = static_cast<float*>(alloc_gpu(size_t(kMaxTree) * 256 * sizeof(float)));
  d_blk_i_ = static_cast<int32_t*>(alloc_gpu(size_t(kMaxTree) * 256 * sizeof(int32_t)));
  d_x_bf16_ = static_cast<__nv_bfloat16*>(alloc_gpu(T * size_t(cfg_.hq * cfg_.head_dim) * sizeof(__nv_bfloat16)));

  d_x_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_x_pre_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_inp_sa_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_ffn_inp_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_h_norm_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  for (int i = 0; i < 6; ++i) d_blocks_[i] = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  for (int i = 0; i < 4; ++i) d_pre_blocks_[i] = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  for (int i = 0; i < 3; ++i) d_tap_[i] = mtp_enabled_ ? static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float))) : nullptr;
  d_ar_out_ = static_cast<float*>(alloc_gpu(4 * T * size_t(H) * sizeof(float)));
  d_msg_out_ = static_cast<float*>(alloc_gpu(4 * T * size_t(H) * sizeof(float)));
  d_mg_nhm_ = static_cast<float*>(alloc_gpu(T * 2 * size_t(H) * sizeof(float)));
  d_mg_nhm_bf16_ = static_cast<__nv_bfloat16*>(alloc_gpu(T * 2 * size_t(H) * sizeof(__nv_bfloat16)));
  d_mg_g_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.gate_rank) * sizeof(float)));
  d_mg_g_bf16_ = static_cast<__nv_bfloat16*>(alloc_gpu(T * size_t(cfg_.gate_rank) * sizeof(__nv_bfloat16)));
  d_mg_df_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_mg_dg_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_qkvrsg_tmp_ = static_cast<float*>(
      alloc_gpu(T * size_t(2 * QD + 2 * KVD + size_t(cfg_.hq) * size_t(cfg_.d_rel) + size_t(cfg_.hq)) * sizeof(float)));
  d_step_pos_ = static_cast<int32_t*>(alloc_gpu(sizeof(int32_t)));
  d_commit_pos_ = static_cast<int32_t*>(alloc_gpu(sizeof(int32_t)));
  d_n0_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_n1_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_n2_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));

  d_q_ = static_cast<float*>(alloc_gpu(T * size_t(QD) * sizeof(float)));
  d_k_ = static_cast<float*>(alloc_gpu(T * size_t(KVD) * sizeof(float)));
  d_v_ = static_cast<float*>(alloc_gpu(T * size_t(KVD) * sizeof(float)));
  d_r_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.hq) * size_t(cfg_.d_rel) * sizeof(float)));
  d_sinks_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.hq) * sizeof(float)));
  d_gate_attn_ = static_cast<float*>(alloc_gpu(T * size_t(QD) * sizeof(float)));
  d_att_out_ = static_cast<float*>(alloc_gpu(T * size_t(QD) * sizeof(float)));
  d_a_out_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));

  d_dyn_ = static_cast<float*>(alloc_gpu(T * 32 * sizeof(float)));
  d_sconv_tmp_ = static_cast<float*>(alloc_gpu(T * std::max<size_t>(size_t(H), 2 * size_t(KVD)) * sizeof(float)));
  d_rb_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.hq) * size_t(cfg_.rel_extent) * sizeof(float)));
  d_part_m_ = static_cast<float*>(alloc_gpu(size_t(kMaxTree) * size_t(cfg_.hq) * kMaxAttnChunks * sizeof(float)));
  d_part_l_ = static_cast<float*>(alloc_gpu(size_t(kMaxTree) * size_t(cfg_.hq) * kMaxAttnChunks * sizeof(float)));
  d_part_o_ = static_cast<float*>(
      alloc_gpu(size_t(kMaxTree) * size_t(cfg_.hq) * kMaxAttnChunks * size_t(cfg_.head_dim) * sizeof(float)));

  d_h_mlp_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_x_mlp_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_rlog_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.n_experts) * sizeof(float)));
  d_eid_ = static_cast<uint32_t*>(alloc_gpu(T * size_t(cfg_.top_k) * sizeof(uint32_t)));
  d_ew_ = static_cast<float*>(alloc_gpu(T * size_t(cfg_.top_k) * sizeof(float)));
  const size_t P_max = T * size_t(cfg_.top_k);
  d_moe_act_ = static_cast<float*>(alloc_gpu(P_max * size_t(cfg_.moe_I) * sizeof(float)));
  d_moe_pair_out_ = static_cast<float*>(alloc_gpu(P_max * size_t(H) * sizeof(float)));
  d_moe_out_ = static_cast<float*>(alloc_gpu(T * size_t(H) * sizeof(float)));
  d_moe_pair_idx_ =
      static_cast<int32_t*>(alloc_gpu(std::max<size_t>(P_max, size_t(kMaxTree) * size_t(cfg_.top_k) * 3) * sizeof(int32_t)));
  const size_t max_tiles = size_t(cfg_.n_experts) + P_max / 16;
  d_moe_tile_exp_ = static_cast<int32_t*>(alloc_gpu(max_tiles * sizeof(int32_t)));
  d_moe_tile_m0_ = static_cast<int32_t*>(alloc_gpu(max_tiles * sizeof(int32_t)));
  d_moe_tile_cnt_ = static_cast<int32_t*>(alloc_gpu(max_tiles * sizeof(int32_t)));
  d_moe_num_tiles_ = static_cast<int32_t*>(alloc_gpu(sizeof(int32_t)));

  d_logits_ = static_cast<float*>(alloc_gpu(V * sizeof(float)));
  d_verify_logits_ = static_cast<float*>(alloc_gpu(size_t(kMtpBlock) * V * sizeof(float)));
  d_suppress_ = static_cast<uint8_t*>(alloc_gpu(V));
  d_verify_samples_ = static_cast<int32_t*>(alloc_gpu(size_t(kMaxTree) * sizeof(int32_t)));

  cuda_ok(cudaMallocHost(&h_logits_, V * sizeof(float)), "cudaMallocHost logits");
  cuda_ok(cudaMallocHost(&h_verify_logits_, size_t(kMtpBlock) * V * sizeof(float)), "cudaMallocHost verify_logits");
  cuda_ok(cudaMallocHost(&h_tok_, sizeof(int32_t)), "cudaMallocHost tok");
  cuda_ok(cudaMallocHost(&h_pair_toks_, 2 * size_t(kMtpBlock) * sizeof(int32_t)), "cudaMallocHost pair_toks");
  cuda_ok(cudaMallocHost(&h_tree_next_, size_t(kMaxTree) * sizeof(int32_t)), "cudaMallocHost tree_next");
  cuda_ok(cudaMemsetAsync(d_suppress_, 0, V, stream_), "clear suppress");

  if (mtp_enabled_) {
    check_mtp_shape(mtp_cfg_, dir_);
    SafetensorsReader mreader;
    mreader.open_file(dir_ + "/mtp.safetensors", nullptr);
    const int64_t MH = mtp_cfg_.hidden;
    const int64_t MQD = int64_t(mtp_cfg_.hq) * mtp_cfg_.head_dim;
    const int64_t MKVD = int64_t(mtp_cfg_.hkv) * mtp_cfg_.head_dim;
    const int64_t MHD = mtp_cfg_.head_dim;
    const int64_t MF = mtp_cfg_.ffn;
    const int64_t MR = mtp_cfg_.markov_rank;

    auto m_upload = [&](const std::string& name, const char* dtype, std::initializer_list<int64_t> shape) -> Ink2Tensor {
      const StTensorInfo& info = mreader.get(name);
      expect_tensor(info, name, dtype, shape);
      Ink2Tensor t;
      t.f32 = (info.dtype == "F32");
      if (!t.f32 && info.dtype != "BF16") throw std::runtime_error("unexpected mtp dtype " + info.dtype + " for " + name);
      if (info.shape.size() == 1) {
        t.rows = int(info.shape[0]);
        t.cols = 1;
      } else {
        t.rows = int(info.shape[0]);
        size_t cp = 1;
        for (size_t i = 1; i < info.shape.size(); ++i) cp *= size_t(info.shape[i]);
        t.cols = int(cp);
      }
      t.bytes = info.nbytes;
      t.d_ptr = alloc_gpu(info.nbytes);
      mreader.upload_gpu(info, t.d_ptr, stream_);
      return t;
    };
    auto m_f32_vec = [&](const StTensorInfo& info) -> std::vector<float> {
      const size_t n = info.numel();
      std::vector<float> out(n);
      if (info.dtype == "F32") {
        mreader.read_host(info, out.data());
      } else if (info.dtype == "BF16") {
        std::vector<uint16_t> tmp(n);
        mreader.read_host(info, tmp.data());
        for (size_t i = 0; i < n; ++i) out[i] = bf16_bits_to_f32(tmp[i]);
      } else {
        throw std::runtime_error("unsupported mtp vec dtype " + info.dtype);
      }
      return out;
    };

    const StTensorInfo& fw_info = mreader.get("fuse_weight");
    expect_tensor(fw_info, "fuse_weight", "BF16", {4, 4});
    const std::vector<float> fw = m_f32_vec(fw_info);
    for (int r = 0; r < 4; ++r) {
      float mx = fw[size_t(r) * 4 + 0];
      for (int c = 1; c < 4; ++c) mx = std::max(mx, fw[size_t(r) * 4 + size_t(c)]);
      float sum = 0.0f;
      for (int c = 0; c < 4; ++c) {
        mtp_fuse_[r][c] = std::exp(fw[size_t(r) * 4 + size_t(c)] - mx);
        sum += mtp_fuse_[r][c];
      }
      for (int c = 0; c < 4; ++c) mtp_fuse_[r][c] /= sum;
    }

    mtp_mask_embed_ = m_upload("mask_embed", "BF16", {MH});
    mtp_norm_ = m_upload("norm.weight", "BF16", {MH});
    mtp_rec_norm_ = m_upload("markov_head.dflare_norm.weight", "BF16", {MH});
    mtp_markov_w1_ = m_upload("markov_head.markov_w1.weight", "BF16", {cfg_.vocab_padded, MR});
    mtp_rnn_proj_w_ = m_upload("markov_head.rnn_proj.weight", "BF16", {3 * MR, 2 * MR + 2 * MH});
    mtp_rnn_proj_b_ = m_upload("markov_head.rnn_proj.bias", "BF16", {3 * MR});

    mtp_layers_.resize(size_t(mtp_cfg_.num_layers));
    for (int i = 0; i < mtp_cfg_.num_layers; ++i) {
      Ink2MtpLayer& L = mtp_layers_[size_t(i)];
      const std::string p = "layers." + std::to_string(i) + ".";
      const std::string a = p + "self_attn.";
      L.is_local = mtp_cfg_.is_local[i];
      L.dflare_norm = m_upload(p + "dflare_norm.weight", "BF16", {MH});
      L.in_norm = m_upload(p + "input_layernorm.weight", "BF16", {MH});
      L.q_norm = m_upload(a + "q_norm.weight", "BF16", {MHD});
      L.k_norm = m_upload(a + "k_norm.weight", "BF16", {MHD});
      L.k_norm_ctx = m_upload(a + "k_norm_ctx.weight", "BF16", {MHD});
      const auto& ikc = mreader.get(a + "k_proj_ctx.weight");
      const auto& ivc = mreader.get(a + "v_proj_ctx.weight");
      expect_tensor(ikc, a + "k_proj_ctx.weight", "BF16", {MKVD, MH});
      expect_tensor(ivc, a + "v_proj_ctx.weight", "BF16", {MKVD, MH});
      {
        uint8_t* kvc_base = static_cast<uint8_t*>(alloc_gpu(ikc.nbytes + ivc.nbytes));
        auto m_into = [&](const StTensorInfo& info, void* dst) {
          Ink2Tensor t;
          t.f32 = false;
          t.rows = int(info.shape[0]);
          size_t cp = 1;
          for (size_t j = 1; j < info.shape.size(); ++j) cp *= size_t(info.shape[j]);
          t.cols = int(cp);
          t.bytes = info.nbytes;
          t.d_ptr = dst;
          mreader.upload_gpu(info, dst, stream_);
          return t;
        };
        L.k_proj_ctx = m_into(ikc, kvc_base);
        L.v_proj_ctx = m_into(ivc, kvc_base + ikc.nbytes);
        L.w_kv_ctx = reinterpret_cast<const __nv_bfloat16*>(kvc_base);
      }
      L.o_proj = m_upload(a + "o_proj.weight", "BF16", {MH, MQD});
      L.ff_norm = m_upload(p + "post_attention_layernorm.weight", "BF16", {MH});
      L.down_proj = m_upload(p + "mlp.down_proj.weight", "BF16", {MH, MF});

      // Pack Q, K, V projections contiguously [MQD + 2*MKVD, MH] for 1-launch QKV MV
      const auto& iq = mreader.get(a + "q_proj.weight");
      const auto& ik = mreader.get(a + "k_proj.weight");
      const auto& iv = mreader.get(a + "v_proj.weight");
      expect_tensor(iq, a + "q_proj.weight", "BF16", {MQD, MH});
      expect_tensor(ik, a + "k_proj.weight", "BF16", {MKVD, MH});
      expect_tensor(iv, a + "v_proj.weight", "BF16", {MKVD, MH});
      L.qkv_rows = int(MQD + 2 * MKVD);
      uint8_t* qkv_base = static_cast<uint8_t*>(alloc_gpu(iq.nbytes + ik.nbytes + iv.nbytes));
      mreader.upload_gpu(iq, qkv_base, stream_);
      mreader.upload_gpu(ik, qkv_base + iq.nbytes, stream_);
      mreader.upload_gpu(iv, qkv_base + iq.nbytes + ik.nbytes, stream_);
      L.w_qkv = reinterpret_cast<const __nv_bfloat16*>(qkv_base);

      // Interleave gate_proj & up_proj [2*MF, MH] rows for single-pass k_mtp_gate_up_swiglu
      const auto& ig = mreader.get(p + "mlp.gate_proj.weight");
      const auto& iu = mreader.get(p + "mlp.up_proj.weight");
      expect_tensor(ig, p + "mlp.gate_proj.weight", "BF16", {MF, MH});
      expect_tensor(iu, p + "mlp.up_proj.weight", "BF16", {MF, MH});
      std::vector<uint8_t> hg(ig.nbytes), hu(iu.nbytes), hgu(ig.nbytes + iu.nbytes);
      mreader.read_host(ig, hg.data());
      mreader.read_host(iu, hu.data());
      const size_t row_b = size_t(MH) * 2;
      for (size_t j = 0; j < size_t(MF); ++j) {
        std::memcpy(hgu.data() + (2 * j + 0) * row_b, hg.data() + j * row_b, row_b);
        std::memcpy(hgu.data() + (2 * j + 1) * row_b, hu.data() + j * row_b, row_b);
      }
      void* d_gu = alloc_gpu(hgu.size());
      cuda_ok(cudaMemcpyAsync(d_gu, hgu.data(), hgu.size(), cudaMemcpyHostToDevice, stream_), "upload mtp w_gu");
      cuda_ok(cudaStreamSynchronize(stream_), "sync mtp w_gu");
      L.w_gu = reinterpret_cast<const __nv_bfloat16*>(d_gu);
    }

    // Pre-fold markov_head.hidden_out [H, R] + bias [H] with BF16 unembed [V, H] on GPU
    const auto& how = mreader.get("markov_head.hidden_out.weight");
    const auto& hob = mreader.get("markov_head.hidden_out.bias");
    expect_tensor(how, "markov_head.hidden_out.weight", "BF16", {MH, MR});
    expect_tensor(hob, "markov_head.hidden_out.bias", "BF16", {MH});
    const std::vector<float> hb_vec = m_f32_vec(hob);
    if (use_tree_) {
      mtp_hidden_out_w_ = m_upload("markov_head.hidden_out.weight", "BF16", {MH, MR});
      d_mtp_ho_b_ = static_cast<float*>(alloc_gpu(hb_vec.size() * sizeof(float)));
      cuda_ok(cudaMemcpyAsync(d_mtp_ho_b_, hb_vec.data(), hb_vec.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
              "upload mtp ho bias");
    }
    std::vector<uint16_t> wt_bf16(size_t(MR) * size_t(MH));
    {
      std::vector<uint16_t> hw(size_t(MH) * size_t(MR));
      mreader.read_host(how, hw.data());
      for (size_t h = 0; h < size_t(MH); ++h)
        for (size_t r = 0; r < size_t(MR); ++r) wt_bf16[r * size_t(MH) + h] = hw[h * size_t(MR) + r];
    }
    DevGuard d_wt, d_hb, d_fold_f32, tmp_fold_bf16;
    cuda_ok(cudaMalloc(&d_wt.ptr, wt_bf16.size() * sizeof(uint16_t)), "malloc mtp tmp wt");
    cuda_ok(cudaMalloc(&d_hb.ptr, hb_vec.size() * sizeof(float)), "malloc mtp tmp hb");
    cuda_ok(cudaMalloc(&d_fold_f32.ptr, V * size_t(MR) * sizeof(float)), "malloc mtp tmp fold_f32");
    cuda_ok(cudaMemcpyAsync(d_wt.ptr, wt_bf16.data(), wt_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_),
            "upload mtp tmp wt");
    cuda_ok(cudaMemcpyAsync(d_hb.ptr, hb_vec.data(), hb_vec.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
            "upload mtp tmp hb");

    d_mtp_fold_b_ = static_cast<float*>(alloc_gpu(V * sizeof(float)));
    ink2::launch_linear(cublas_, static_cast<const float*>(d_hb.ptr), u_bf16_for_fold, false, d_x_bf16_, d_mtp_fold_b_,
                        1, int(V), int(MH), false, stream_);
    cublasSetStream(cublas_, stream_);
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        cublas_, CUBLAS_OP_T, CUBLAS_OP_N, int(MR), int(V), int(MH), &alpha, d_wt.ptr, CUDA_R_16BF, int(MH),
        u_bf16_for_fold, CUDA_R_16BF, int(MH), &beta, d_fold_f32.ptr, CUDA_R_32F, int(MR), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed in MTP fold_w");
    __nv_bfloat16* d_fold_bf16 = nullptr;
    if (mtp_q8_) {
      cuda_ok(cudaMalloc(&tmp_fold_bf16.ptr, V * size_t(MR) * sizeof(__nv_bfloat16)), "malloc mtp tmp fold_bf16");
      d_fold_bf16 = static_cast<__nv_bfloat16*>(tmp_fold_bf16.ptr);
      ink2::launch_f32_to_bf16_buf(static_cast<const float*>(d_fold_f32.ptr), d_fold_bf16, int(V * size_t(MR)), stream_);
      d_mtp_fold_q_ = static_cast<int8_t*>(alloc_gpu(V * size_t(MR)));
      d_mtp_fold_s_ = static_cast<__half*>(alloc_gpu(V * size_t(MR / 32) * sizeof(__half)));
      ink2::launch_quantize_q8_32(d_fold_bf16, d_mtp_fold_q_, d_mtp_fold_s_, int(V), int(MR), stream_);
    } else {
      d_fold_bf16 = static_cast<__nv_bfloat16*>(alloc_gpu(V * size_t(MR) * sizeof(__nv_bfloat16)));
      ink2::launch_f32_to_bf16_buf(static_cast<const float*>(d_fold_f32.ptr), d_fold_bf16, int(V * size_t(MR)), stream_);
      d_mtp_fold_w_ = d_fold_bf16;
    }
    cuda_ok(cudaStreamSynchronize(stream_), "sync mtp fold");
    mreader.close_all();

    // Draft-only Q8 unembed (reuse d_screen_q8_ when built to avoid duplicate ~505 MB allocation)
    if (use_q8_head_) {
      d_draft_u_q8_ = d_unembed_q8_;
      d_draft_u_sc_ = d_unembed_q8_sc_;
    } else if (!d_screen_q8_ && (MH & 31) == 0 && mtp_q8_) {
      d_draft_u_q8_ = static_cast<int8_t*>(alloc_gpu(V * size_t(MH)));
      d_draft_u_sc_ = static_cast<__half*>(alloc_gpu(V * size_t(MH / 32) * sizeof(__half)));
      ink2::launch_quantize_q8_32(u_bf16_for_fold, d_draft_u_q8_, d_draft_u_sc_, int(V), int(MH), stream_);
      cuda_ok(cudaStreamSynchronize(stream_), "sync draft u_q8");
    }

    // MTP context KV caches & draft scratch buffers
    const size_t mb = size_t(mtp_cfg_.block);
    const size_t mtp_kv_row = size_t(MKVD);
    const size_t mtp_kv_eb = kv_bf16_ ? sizeof(__nv_bfloat16) : sizeof(float);
    for (int i = 0; i < 3; ++i) {
      const size_t cap = mtp_cfg_.is_local[i] ? size_t(ring_rows_) : size_t(max_ctx_);
      d_mtp_kc_[i] = alloc_gpu(cap * mtp_kv_row * mtp_kv_eb);
      d_mtp_vc_[i] = alloc_gpu(cap * mtp_kv_row * mtp_kv_eb);
    }
    d_mtp_rec_feat_ = static_cast<float*>(alloc_gpu(size_t(MH) * sizeof(float)));
    d_mtp_ctx_h_ = static_cast<float*>(alloc_gpu(T * size_t(MH) * sizeof(float)));
    d_mtp_ctx_k_ = static_cast<float*>(alloc_gpu(T * size_t(MKVD) * sizeof(float)));
    d_mtp_ctx_v_ = static_cast<float*>(alloc_gpu(T * size_t(MKVD) * sizeof(float)));
    d_mtp_blk_x_ = static_cast<float*>(alloc_gpu(mb * size_t(MH) * sizeof(float)));
    d_mtp_blk_res_ = static_cast<float*>(alloc_gpu(mb * size_t(MH) * sizeof(float)));
    d_mtp_blk_norm_ = static_cast<float*>(alloc_gpu(mb * size_t(MH) * sizeof(float)));
    d_mtp_blk_q_ = static_cast<float*>(alloc_gpu(mb * size_t(MQD) * sizeof(float)));
    d_mtp_blk_k_ = static_cast<float*>(alloc_gpu(mb * size_t(MKVD) * sizeof(float)));
    d_mtp_blk_v_ = static_cast<float*>(alloc_gpu(mb * size_t(MKVD) * sizeof(float)));
    d_mtp_blk_att_ = static_cast<float*>(alloc_gpu(mb * size_t(MQD) * sizeof(float)));
    d_mtp_blk_gate_ = static_cast<float*>(alloc_gpu(mb * size_t(MF) * sizeof(float)));
    d_mtp_blk_out_ = static_cast<float*>(alloc_gpu(mb * size_t(MH) * sizeof(float)));
    d_mtp_qkv_tmp_ = static_cast<float*>(alloc_gpu(mb * size_t(MQD + 2 * MKVD) * sizeof(float)));
    d_mtp_state_ = static_cast<float*>(alloc_gpu(size_t(MR) * sizeof(float)));
    d_mtp_latent_ = static_cast<float*>(alloc_gpu(size_t(MR) * sizeof(float)));
    d_mtp_rnn_in_ = static_cast<float*>(alloc_gpu(size_t(2 * MR + 2 * MH) * sizeof(float)));
    d_mtp_rnn_gates_ = static_cast<float*>(alloc_gpu(3 * size_t(MR) * sizeof(float)));
    d_mtp_logits_ = static_cast<float*>(alloc_gpu((mb - 1) * V * sizeof(float)));
    cuda_ok(cudaMallocHost(&h_mtp_draft_logits_, (mb - 1) * V * sizeof(float)), "cudaMallocHost mtp draft_logits");

    if (use_tree_) {
      d_beam_h_q8_ = static_cast<int8_t*>(alloc_gpu(7 * size_t(MH)));
      d_beam_h_sc_ = static_cast<float*>(alloc_gpu(7 * sizeof(float)));
      d_beam_top_tmp_ids_ = static_cast<int32_t*>(alloc_gpu(7 * 2048 * sizeof(int32_t)));
      d_beam_top_tmp_vals_ = static_cast<float*>(alloc_gpu(7 * 2048 * sizeof(float)));
      d_beam_cand_ids_ = static_cast<int32_t*>(alloc_gpu(7 * size_t(kDsparkKCand) * sizeof(int32_t)));
      d_beam_cand_base_ = static_cast<float*>(alloc_gpu(7 * size_t(kDsparkKCand) * sizeof(float)));
      d_beam_state_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * size_t(MR) * sizeof(float)));
      d_beam_state_next_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * size_t(MR) * sizeof(float)));
      d_beam_prev_tok_ = static_cast<int32_t*>(alloc_gpu(size_t(kBeamW) * sizeof(int32_t)));
      d_beam_node_ = static_cast<int32_t*>(alloc_gpu(size_t(kBeamW) * sizeof(int32_t)));
      d_beam_logp_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * sizeof(float)));
      d_beam_joint_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * size_t(2 * MR + 2 * MH) * sizeof(float)));
      d_beam_proj_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * 3 * size_t(MR) * sizeof(float)));
      d_beam_lat_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * size_t(MR) * sizeof(float)));
      d_beam_corr_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * size_t(MH) * sizeof(float)));
      d_beam_cand_scores_ = static_cast<float*>(alloc_gpu(size_t(kBeamW) * size_t(kDsparkKCand) * sizeof(float)));

      d_pool_tok_ = static_cast<int32_t*>(alloc_gpu(size_t(kBeamPoolCap) * sizeof(int32_t)));
      d_pool_parent_ = static_cast<int8_t*>(alloc_gpu(size_t(kBeamPoolCap) * sizeof(int8_t)));
      d_pool_depth_ = static_cast<uint8_t*>(alloc_gpu(size_t(kBeamPoolCap) * sizeof(uint8_t)));
      d_pool_logp_ = static_cast<float*>(alloc_gpu(size_t(kBeamPoolCap) * sizeof(float)));
      d_pool_cnt_ = static_cast<int32_t*>(alloc_gpu(sizeof(int32_t)));
      cuda_ok(cudaMallocHost(&h_pool_tok_, size_t(kBeamPoolCap) * sizeof(int32_t)), "cudaMallocHost pool_tok");
      cuda_ok(cudaMallocHost(&h_pool_parent_, size_t(kBeamPoolCap) * sizeof(int8_t)), "cudaMallocHost pool_parent");
      cuda_ok(cudaMallocHost(&h_pool_depth_, size_t(kBeamPoolCap) * sizeof(uint8_t)), "cudaMallocHost pool_depth");
      cuda_ok(cudaMallocHost(&h_pool_logp_, size_t(kBeamPoolCap) * sizeof(float)), "cudaMallocHost pool_logp");
      cuda_ok(cudaMallocHost(&h_pool_cnt_, sizeof(int32_t)), "cudaMallocHost pool_cnt");

      const size_t nv_t = cfg_.schedule.size();
      const size_t stage_sc_stride = size_t(2 * KVD + 2 * H);
      d_stage_k_ = static_cast<float*>(alloc_gpu(nv_t * size_t(kMaxTree) * size_t(KVD) * sizeof(float)));
      d_stage_v_ = static_cast<float*>(alloc_gpu(nv_t * size_t(kMaxTree) * size_t(KVD) * sizeof(float)));
      d_stage_sconv_ = static_cast<float*>(alloc_gpu(nv_t * size_t(kMaxTree) * stage_sc_stride * sizeof(float)));
      d_tap_scratch_ = static_cast<float*>(alloc_gpu(4 * size_t(kMaxTree) * size_t(H) * sizeof(float)));
      d_tree_p1_ = static_cast<int8_t*>(alloc_gpu(size_t(kMaxTree) * sizeof(int8_t)));
      d_tree_p2_ = static_cast<int8_t*>(alloc_gpu(size_t(kMaxTree) * sizeof(int8_t)));
      d_tree_p3_ = static_cast<int8_t*>(alloc_gpu(size_t(kMaxTree) * sizeof(int8_t)));
      d_tree_depth_ = static_cast<uint8_t*>(alloc_gpu(size_t(kMaxTree) * sizeof(uint8_t)));
      d_tree_anc_mask_ = static_cast<uint32_t*>(alloc_gpu(size_t(kMaxTree) * sizeof(uint32_t)));
      d_acc_path_ = static_cast<int32_t*>(alloc_gpu(size_t(kMaxTree) * sizeof(int32_t)));

      d_kc_ptrs_ = static_cast<void**>(alloc_gpu(nv_t * sizeof(void*)));
      d_vc_ptrs_ = static_cast<void**>(alloc_gpu(nv_t * sizeof(void*)));
      d_kv_cap_ = static_cast<int32_t*>(alloc_gpu(nv_t * sizeof(int32_t)));
      d_kv_is_swa_ = static_cast<uint8_t*>(alloc_gpu(nv_t * sizeof(uint8_t)));
      std::vector<void*> kc_h(nv_t), vc_h(nv_t);
      std::vector<int32_t> kv_cap_h(nv_t);
      std::vector<uint8_t> kv_swa_h(nv_t);
      for (size_t v_i = 0; v_i < nv_t; ++v_i) {
        const bool is_s = cfg_.is_local[size_t(cfg_.schedule[v_i])];
        kc_h[v_i] = d_kc_[v_i];
        vc_h[v_i] = d_vc_[v_i];
        kv_cap_h[v_i] = is_s ? ring_rows_ : max_ctx_;
        kv_swa_h[v_i] = is_s ? 1 : 0;
      }
      cuda_ok(cudaMemcpyAsync(d_kc_ptrs_, kc_h.data(), nv_t * sizeof(void*), cudaMemcpyHostToDevice, stream_), "kc ptrs");
      cuda_ok(cudaMemcpyAsync(d_vc_ptrs_, vc_h.data(), nv_t * sizeof(void*), cudaMemcpyHostToDevice, stream_), "vc ptrs");
      cuda_ok(cudaMemcpyAsync(d_kv_cap_, kv_cap_h.data(), nv_t * sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "kv cap");
      cuda_ok(cudaMemcpyAsync(d_kv_is_swa_, kv_swa_h.data(), nv_t * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_), "kv swa");
      cuda_ok(cudaStreamSynchronize(stream_), "sync tree tables");
    }
  }

  if (const char* eg = std::getenv("YSKN_INK2_GRAPH")) use_graph_ = (std::strcmp(eg, "0") != 0);
  reset();
  } catch (...) {  // a refused or damaged folder: give back what was taken before the error
    free_all();
    throw;
  }
}

void EngineInk2::destroy_graphs() {
  for (int a = 0; a < 2; ++a) {
    for (int i = 0; i < 2; ++i) {
      if (decode_graph_exec_[a][i]) {
        cudaGraphExecDestroy(decode_graph_exec_[a][i]);
        decode_graph_exec_[a][i] = nullptr;
      }
      if (decode_graph_[a][i]) {
        cudaGraphDestroy(decode_graph_[a][i]);
        decode_graph_[a][i] = nullptr;
      }
    }
  }
  for (int k = 0; k < 7; ++k) {
    for (int s = 0; s < 2; ++s) {
      if (dv_graph_exec_[k][s]) {
        cudaGraphExecDestroy(dv_graph_exec_[k][s]);
        dv_graph_exec_[k][s] = nullptr;
      }
      if (dv_graph_[k][s]) {
        cudaGraphDestroy(dv_graph_[k][s]);
        dv_graph_[k][s] = nullptr;
      }
    }
  }
  for (int r = 0; r < 2; ++r) {
    for (int m = 0; m < 8; ++m) {
      if (commit_graph_exec_[r][m]) {
        cudaGraphExecDestroy(commit_graph_exec_[r][m]);
        commit_graph_exec_[r][m] = nullptr;
      }
      if (commit_graph_[r][m]) {
        cudaGraphDestroy(commit_graph_[r][m]);
        commit_graph_[r][m] = nullptr;
      }
    }
  }
  for (int s = 0; s < 2; ++s) {
    if (beam_draft_graph_exec_[s]) {
      cudaGraphExecDestroy(beam_draft_graph_exec_[s]);
      beam_draft_graph_exec_[s] = nullptr;
    }
    if (beam_draft_graph_[s]) {
      cudaGraphDestroy(beam_draft_graph_[s]);
      beam_draft_graph_[s] = nullptr;
    }
  }
  for (int b = 0; b < kNumTreeBuckets; ++b) {
    for (int s = 0; s < 2; ++s) {
      if (tree_verify_graph_exec_[b][s]) {
        cudaGraphExecDestroy(tree_verify_graph_exec_[b][s]);
        tree_verify_graph_exec_[b][s] = nullptr;
      }
      if (tree_verify_graph_[b][s]) {
        cudaGraphDestroy(tree_verify_graph_[b][s]);
        tree_verify_graph_[b][s] = nullptr;
      }
    }
  }
}

void EngineInk2::free_all() noexcept {
  if (stream_) cudaStreamSynchronize(stream_);
  destroy_graphs();
  if (h_logits_) cudaFreeHost(h_logits_);
  if (h_verify_logits_) cudaFreeHost(h_verify_logits_);
  if (h_tok_) cudaFreeHost(h_tok_);
  if (h_pair_toks_) cudaFreeHost(h_pair_toks_);
  if (h_tree_next_) cudaFreeHost(h_tree_next_);
  if (h_mtp_draft_logits_) cudaFreeHost(h_mtp_draft_logits_);
  if (h_pool_tok_) cudaFreeHost(h_pool_tok_);
  if (h_pool_parent_) cudaFreeHost(h_pool_parent_);
  if (h_pool_depth_) cudaFreeHost(h_pool_depth_);
  if (h_pool_logp_) cudaFreeHost(h_pool_logp_);
  if (h_pool_cnt_) cudaFreeHost(h_pool_cnt_);
  if (cublas_) cublasDestroy(cublas_);
  for (void* p : gpu_allocs_) cudaFree(p);
  if (stream_) cudaStreamDestroy(stream_);
  h_logits_ = nullptr;
  h_verify_logits_ = nullptr;
  h_tok_ = nullptr;
  h_pair_toks_ = nullptr;
  h_tree_next_ = nullptr;
  h_mtp_draft_logits_ = nullptr;
  h_pool_tok_ = nullptr;
  h_pool_parent_ = nullptr;
  h_pool_depth_ = nullptr;
  h_pool_logp_ = nullptr;
  h_pool_cnt_ = nullptr;
  cublas_ = nullptr;
  gpu_allocs_.clear();
  stream_ = nullptr;
}

EngineInk2::~EngineInk2() { free_all(); }

void EngineInk2::reset() {
  cuda_ok(cudaMemsetAsync(d_conv_base_, 0, cfg_.schedule.size() * size_t(kVisitConvFloats) * sizeof(float), stream_),
          "memset conv");
  if (d_mtp_rec_feat_) cuda_ok(cudaMemsetAsync(d_mtp_rec_feat_, 0, size_t(cfg_.hidden) * sizeof(float), stream_), "memset rec_feat");
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
    h_suppress_.clear();
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
  h_suppress_ = bits;
  cuda_ok(cudaMemcpyAsync(d_suppress_, bits.data(), V, cudaMemcpyHostToDevice, stream_), "upload suppress");
  cuda_ok(cudaStreamSynchronize(stream_), "sync suppress");
  suppress_active_ = true;
}

void EngineInk2::forward_tokens(const int32_t* ids, int T, int pos0, bool need_logits,
                                bool capture_taps, bool save_snaps, bool need_all_logits,
                                const int32_t* d_pos_ptr_arg, bool greedy_argmax_only) {
  if (T <= 0 || T > kPrefillChunk) throw std::runtime_error("invalid Inkling2 chunk size");
  if (pos0 + T > max_ctx_) throw std::runtime_error("context full (" + std::to_string(max_ctx_) + ")");

  const int H = cfg_.hidden;
  const int QD = cfg_.hq * cfg_.head_dim;
  const int KVD = cfg_.hkv * cfg_.head_dim;
  const int snap_stride = save_snaps ? (int(cfg_.schedule.size()) * kVisitConvFloats) : 0;
  const size_t row_bytes = size_t(T) * size_t(H) * sizeof(float);
  const float sconv_static_scale = 1.0f / std::sqrt(5.0f);
  const float ar_scale = 1.0f / std::sqrt(float(H));
  const bool use_scr_argmax = greedy_argmax_only && use_screen_head_ && !need_all_logits;

  if (ids != nullptr) {
    cuda_ok(cudaMemcpyAsync(d_tokens_, ids, size_t(T) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
            "upload tokens");
  }

  auto run_forward_body = [&](const int32_t* d_pos_ptr) {
    ink2::launch_embed_norm(d_tokens_, 0, static_cast<const __nv_bfloat16*>(embed_.d_ptr),
                            static_cast<const __nv_bfloat16*>(embed_norm_.d_ptr), d_x_, T, H, cfg_.rms_eps, stream_);

    const float* blocks_vec[6] = {};
    float* bslot_ptr[6] = {d_blocks_[0], d_blocks_[1], d_blocks_[2], d_blocks_[3], d_blocks_[4], d_blocks_[5]};
    const float* bank0_ptr = d_blocks_[4];
    float* cur_x = d_x_;
    int n_blocks = 0;
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
      const size_t cbase = v_idx * size_t(kVisitConvFloats);
      float* snap_k = save_snaps ? (d_verify_conv_ + cbase + 0) : nullptr;
      float* snap_v = save_snaps ? (d_verify_conv_ + cbase + 3 * KVD) : nullptr;
      float* snap_a = save_snaps ? (d_verify_conv_ + cbase + 6 * KVD) : nullptr;
      float* snap_m = save_snaps ? (d_verify_conv_ + cbase + 6 * KVD + 3 * H) : nullptr;

      if (ip == cfg_.core_start) {
        if (!have_prelude) {
          if (cur_x != d_x_pre_) {
            cuda_ok(cudaMemcpyAsync(d_x_pre_, cur_x, row_bytes, cudaMemcpyDeviceToDevice, stream_), "copy x_pre");
            cur_x = d_x_pre_;
          }
          n_prelude_blocks = n_blocks;
          if (n_prelude_blocks > cfg_.core_start / cfg_.block_sz) {
            for (int b = 0; b < n_prelude_blocks; ++b)
              cuda_ok(cudaMemcpyAsync(d_pre_blocks_[b], blocks_vec[b], row_bytes, cudaMemcpyDeviceToDevice, stream_),
                      "copy pre_block");
          }
          have_prelude = true;
        } else {
          const float* prev_mid = blocks_vec[n_prelude_blocks + 1];
          if (n_prelude_blocks <= cfg_.core_start / cfg_.block_sz && prev_mid == bslot_ptr[2]) {
            std::swap(bslot_ptr[2], bslot_ptr[4]);
            bank0_ptr = bslot_ptr[4];
          } else {
            cuda_ok(cudaMemcpyAsync(bslot_ptr[4], prev_mid, row_bytes, cudaMemcpyDeviceToDevice, stream_), "copy bank0");
            bank0_ptr = bslot_ptr[4];
          }
          ink2::launch_rmsnorm(cur_x, nullptr, true, bslot_ptr[5], T, H, cfg_.rms_eps, stream_);
          cur_x = d_x_pre_;
          n_blocks = n_prelude_blocks;
          if (n_prelude_blocks > cfg_.core_start / cfg_.block_sz) {
            for (int b = 0; b < n_prelude_blocks; ++b) {
              cuda_ok(cudaMemcpyAsync(d_blocks_[b], d_pre_blocks_[b], row_bytes, cudaMemcpyDeviceToDevice, stream_),
                      "restore pre_block");
              blocks_vec[b] = d_blocks_[b];
            }
          }
          have_prev = true;
        }
      }

      if (capture_taps && mtp_enabled_) {
        for (int ti = 0; ti < 3; ++ti) {
          if (int(v_idx) == mtp_cfg_.taps[ti]) {
            cuda_ok(cudaMemcpyAsync(d_tap_[ti], cur_x, row_bytes, cudaMemcpyDeviceToDevice, stream_), "capture tap");
          }
        }
      }

      const bool has_msgs = have_prev && ip < cfg_.core_end;
      const float* msg_m0 = d_msg_out_ + 0 * size_t(T) * size_t(H);
      const float* msg_m1 = d_msg_out_ + 1 * size_t(T) * size_t(H);
      const float* msg_m2 = d_msg_out_ + 2 * size_t(T) * size_t(H);
      const float* msg_mlp = d_msg_out_ + 3 * size_t(T) * size_t(H);
      if (has_msgs) {
        const float* bank[2] = {bank0_ptr, bslot_ptr[5]};
        const float* qv = d_msg_queries_ + size_t(ip - cfg_.core_start) * 4 * size_t(H);
        ink2::launch_attn_res(bank, 2, 0x3u, qv, true, 4, d_msg_out_, T, H, cfg_.rms_eps, ar_scale, stream_);
      }

      const float* in_ptrs[3];
      bool fused_attn_norm = false;
      if (ip == 0) {
        in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = cur_x;
      } else {
        const float* st[6];
        for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
        st[n_blocks] = cur_x;
        const Ink2Tensor& aq = attn_queries_[size_t(ip - 1)];
        const int nq = (aq.cols == 1) ? 1 : aq.rows;
        const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
        if (!has_msgs && L.is_swa && nq == 1) {
          ink2::launch_attn_res(st, n_blocks + 1, st_mask, aq.d_ptr, aq.f32, 1, d_n0_, T, H, cfg_.rms_eps, ar_scale,
                                stream_, L.attn_norm.d_ptr, L.attn_norm.f32);
          fused_attn_norm = true;
          in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = d_n0_;
        } else {
          ink2::launch_attn_res(st, n_blocks + 1, st_mask, aq.d_ptr, aq.f32, nq, d_ar_out_, T, H, cfg_.rms_eps,
                                ar_scale, stream_);
          if (nq == 1) {
            in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = d_ar_out_;
          } else {
            in_ptrs[0] = d_ar_out_ + 0 * size_t(T) * size_t(H);
            in_ptrs[1] = d_ar_out_ + 1 * size_t(T) * size_t(H);
            in_ptrs[2] = d_ar_out_ + 2 * size_t(T) * size_t(H);
          }
        }
      }

      if (has_msgs) {
        if (T <= 32) {
          const ink2::MsgGateItem items[3] = {
              {in_ptrs[0], msg_m0, static_cast<const __nv_bfloat16*>(L.ctrl_attn.down.d_ptr),
               static_cast<const __nv_bfloat16*>(L.ctrl_attn.forget.d_ptr),
               static_cast<const __nv_bfloat16*>(L.ctrl_attn.gain.d_ptr), d_n0_},
              {in_ptrs[1], msg_m1, static_cast<const __nv_bfloat16*>(L.ctrl_k.down.d_ptr),
               static_cast<const __nv_bfloat16*>(L.ctrl_k.forget.d_ptr),
               static_cast<const __nv_bfloat16*>(L.ctrl_k.gain.d_ptr), d_n1_},
              {in_ptrs[2], msg_m2, static_cast<const __nv_bfloat16*>(L.ctrl_v.down.d_ptr),
               static_cast<const __nv_bfloat16*>(L.ctrl_v.forget.d_ptr),
               static_cast<const __nv_bfloat16*>(L.ctrl_v.gain.d_ptr), d_n2_},
          };
          ink2::launch_msg_gate_x3(items, d_mg_nhm_, d_mg_g_, T, H, cfg_.gate_rank, cfg_.rms_eps, stream_);
        } else {
          auto call_mg = [&](const Ink2Ctrl& c, const float* h_in, const float* m_in, float* out) {
            ink2::launch_msg_gate(cublas_, h_in, m_in, static_cast<const __nv_bfloat16*>(c.down.d_ptr),
                                  static_cast<const __nv_bfloat16*>(c.forget.d_ptr),
                                  static_cast<const __nv_bfloat16*>(c.gain.d_ptr),
                                  d_mg_nhm_, d_mg_nhm_bf16_, d_mg_g_, d_mg_g_bf16_, d_mg_df_, d_mg_dg_,
                                  out, T, H, cfg_.gate_rank, cfg_.rms_eps, stream_);
          };
          call_mg(L.ctrl_attn, in_ptrs[0], msg_m0, d_n0_);
          call_mg(L.ctrl_k, in_ptrs[1], msg_m1, d_n1_);
          call_mg(L.ctrl_v, in_ptrs[2], msg_m2, d_n2_);
        }
        if (L.is_swa) {
          ink2::launch_rmsnorm_x3(d_n0_, d_n1_, d_n2_, L.attn_norm.d_ptr, L.attn_norm.d_ptr, L.attn_norm.d_ptr,
                                  L.attn_norm.f32, d_n0_, d_n1_, d_n2_, T, H, cfg_.rms_eps, stream_);
        } else {
          ink2::launch_rmsnorm_x3(d_n0_, d_n1_, d_n2_, L.q_norm.d_ptr, L.k_norm.d_ptr, L.v_norm.d_ptr,
                                  L.q_norm.f32, d_n0_, d_n1_, d_n2_, T, H, cfg_.rms_eps, stream_);
        }
      } else if (L.is_swa) {
        if (!fused_attn_norm) {
          ink2::launch_rmsnorm(in_ptrs[0], L.attn_norm.d_ptr, L.attn_norm.f32, d_n0_, T, H, cfg_.rms_eps, stream_);
        }
      } else {
        ink2::launch_rmsnorm_x3(in_ptrs[0], in_ptrs[1], in_ptrs[2], L.q_norm.d_ptr, L.k_norm.d_ptr, L.v_norm.d_ptr,
                                L.q_norm.f32, d_n0_, d_n1_, d_n2_, T, H, cfg_.rms_eps, stream_);
      }

      const float* kn = (has_msgs || !L.is_swa) ? d_n1_ : d_n0_;
      const float* vn = (has_msgs || !L.is_swa) ? d_n2_ : d_n0_;

      auto mm = [&](const Ink2Tensor& w, const float* x, float* y, int n_lim = 0) {
        const int N = n_lim ? n_lim : w.rows;
        ink2::launch_linear(cublas_, x, w.d_ptr, w.f32, d_x_bf16_, y, T, N, w.cols, false, stream_);
      };

      if (L.is_swa && !has_msgs && L.fused_qkvrsg) {
        const ink2::MultiLinearSeg segs[6] = {
            {d_q_, L.wq_du.rows},
            {d_r_, L.wr_du.rows},
            {d_sinks_, L.attn_sink_proj.rows},
            {d_gate_attn_, L.gate_attn.rows},
            {d_k_, L.wk_dv.rows},
            {d_v_, L.wv_dv.rows},
        };
        ink2::launch_linear_multi(cublas_, d_n0_, L.w_qkvrsg, d_x_bf16_, d_qkvrsg_tmp_, segs, 6, T, H, stream_);
      } else {
        if (L.fused_qrsg) {
          const ink2::MultiLinearSeg segs[4] = {
              {d_q_, L.wq_du.rows},
              {d_r_, L.wr_du.rows},
              {d_sinks_, L.attn_sink_proj.rows},
              {d_gate_attn_, L.gate_attn.rows},
          };
          ink2::launch_linear_multi(cublas_, d_n0_, L.w_qkvrsg, d_x_bf16_, d_qkvrsg_tmp_, segs, 4, T, H, stream_);
        } else {
          mm(L.wq_du, d_n0_, d_q_);
          mm(L.wr_du, d_n0_, d_r_);
          mm(L.attn_sink_proj, d_n0_, d_sinks_);
          mm(L.gate_attn, d_n0_, d_gate_attn_);
        }
        if (L.is_swa || !L.fused_kvp) {
          mm(L.wk_dv, kn, d_k_);
          mm(L.wv_dv, vn, d_v_);
        }
      }

      if (L.is_swa) {
        ink2::launch_sconv_static_kv(d_k_, L.k_sconv.d_ptr, L.k_sconv.f32, conv_k,
                                     d_v_, L.v_sconv.d_ptr, L.v_sconv.f32, conv_v,
                                     T, KVD, sconv_static_scale, stream_, d_sconv_tmp_,
                                     save_snaps, snap_k, snap_v, snap_stride);
      } else {
        if (L.fused_kvp) {
          const ink2::MultiLinearSeg ksegs[2] = {{d_k_, L.wk_dv.rows}, {d_dyn_, L.k_proj.rows}};
          ink2::launch_linear_multi(cublas_, kn, L.w_kkp, d_x_bf16_, d_qkvrsg_tmp_, ksegs, 2, T, H, stream_);
        } else {
          mm(L.k_proj, kn, d_dyn_);
        }
        ink2::launch_sconv_dynamic(d_k_, d_dyn_, L.k_static.d_ptr, L.k_static.f32, conv_k, d_k_, T, KVD, cfg_.hkv,
                                   stream_, d_sconv_tmp_, save_snaps, snap_k, snap_stride);
        if (L.fused_kvp) {
          const ink2::MultiLinearSeg vsegs[2] = {{d_v_, L.wv_dv.rows}, {d_dyn_, L.v_proj.rows}};
          ink2::launch_linear_multi(cublas_, vn, L.w_vvp, d_x_bf16_, d_qkvrsg_tmp_, vsegs, 2, T, H, stream_);
        } else {
          mm(L.v_proj, vn, d_dyn_);
        }
        ink2::launch_sconv_dynamic(d_v_, d_dyn_, L.v_static.d_ptr, L.v_static.f32, conv_v, d_v_, T, KVD, cfg_.hkv,
                                   stream_, d_sconv_tmp_, save_snaps, snap_v, snap_stride);
      }

      const int kv_cap = L.is_swa ? ring_rows_ : max_ctx_;
      ink2::launch_qkvr_prep(d_q_, d_k_, d_v_, d_r_, L.q_scale.d_ptr, L.rel_scale.d_ptr, L.q_scale.f32, d_kc_[v_idx],
                             d_vc_[v_idx], T, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.d_rel, pos0, kv_cap, L.is_swa,
                             cfg_.qk_clip, cfg_.rel_clip, cfg_.rms_eps, cfg_.log_floor, cfg_.log_alpha, d_pos_ptr,
                             kv_bf16_, stream_);
      ink2::launch_rel_bias(cublas_, d_r_, L.rel_proj.d_ptr, L.rel_proj.f32, d_x_bf16_, d_rb_, T, cfg_.hq,
                            cfg_.d_rel, cfg_.rel_extent, stream_);

      if (T == 1) {
        ink2::launch_attn_decode(d_q_, d_kc_[v_idx], d_vc_[v_idx], d_rb_, d_sinks_, d_gate_attn_, d_part_m_, d_part_l_,
                                 d_part_o_, d_att_out_, pos0, kv_cap, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.rel_extent,
                                 L.is_swa, cfg_.swa, d_pos_ptr, kv_bf16_, stream_);
      } else if (T <= kMtpBlock) {
        ink2::launch_attn_verify(d_q_, d_kc_[v_idx], d_vc_[v_idx], d_rb_, d_sinks_, d_gate_attn_, d_part_m_, d_part_l_,
                                 d_part_o_, d_att_out_, T, pos0, kv_cap, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.rel_extent,
                                 L.is_swa, cfg_.swa, d_pos_ptr, kv_bf16_, stream_);
      } else {
        ink2::launch_attn_prefill(d_q_, d_kc_[v_idx], d_vc_[v_idx], d_rb_, d_sinks_, d_gate_attn_, d_att_out_, T, pos0,
                                  kv_cap, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.rel_extent, L.is_swa, cfg_.swa,
                                  kv_bf16_, stream_);
      }

      mm(L.wo_ud, d_att_out_, d_a_out_);
      const float* ffn_ptr = d_a_out_;
      if (ip % cfg_.block_sz == 0) {
        ink2::launch_sconv_static(d_a_out_, L.attn_sconv.d_ptr, L.attn_sconv.f32, conv_a, d_a_out_, T, H,
                                  sconv_static_scale, stream_, nullptr, d_sconv_tmp_, save_snaps, snap_a, snap_stride);
        const int bslot = ip / cfg_.block_sz;
        ink2::launch_rmsnorm(cur_x, nullptr, true, bslot_ptr[bslot], T, H, cfg_.rms_eps, stream_);
        blocks_vec[size_t(n_blocks++)] = bslot_ptr[bslot];
      } else {
        ink2::launch_sconv_static(d_a_out_, L.attn_sconv.d_ptr, L.attn_sconv.f32, conv_a, d_ffn_inp_, T, H,
                                  sconv_static_scale, stream_, cur_x, nullptr, save_snaps, snap_a, snap_stride);
        ffn_ptr = d_ffn_inp_;
      }

      if (has_msgs) {
        {
          const float* st[6];
          for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
          st[n_blocks] = ffn_ptr;
          const Ink2Tensor& mq = mlp_queries_[size_t(ip)];
          const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
          ink2::launch_attn_res(st, n_blocks + 1, st_mask, mq.d_ptr, mq.f32, 1, d_h_mlp_, T, H, cfg_.rms_eps, ar_scale,
                                stream_);
        }
        ink2::launch_msg_gate(cublas_, d_h_mlp_, msg_mlp, static_cast<const __nv_bfloat16*>(L.ctrl_mlp.down.d_ptr),
                              static_cast<const __nv_bfloat16*>(L.ctrl_mlp.forget.d_ptr),
                              static_cast<const __nv_bfloat16*>(L.ctrl_mlp.gain.d_ptr),
                              d_mg_nhm_, d_mg_nhm_bf16_, d_mg_g_, d_mg_g_bf16_, d_mg_df_, d_mg_dg_,
                              d_h_mlp_, T, H, cfg_.gate_rank, cfg_.rms_eps, stream_);
        ink2::launch_rmsnorm(d_h_mlp_, L.mlp_norm.d_ptr, L.mlp_norm.f32, d_x_mlp_, T, H, cfg_.rms_eps, stream_);
      } else {
        const float* st[6];
        for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
        st[n_blocks] = ffn_ptr;
        const Ink2Tensor& mq = mlp_queries_[size_t(ip)];
        const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
        ink2::launch_attn_res(st, n_blocks + 1, st_mask, mq.d_ptr, mq.f32, 1, d_x_mlp_, T, H, cfg_.rms_eps, ar_scale,
                              stream_, L.mlp_norm.d_ptr, L.mlp_norm.f32);
      }

      mm(L.router_w, d_x_mlp_, d_rlog_);
      const ink2::MoeGroupBufs gbuf{d_moe_pair_idx_, d_moe_tile_exp_, d_moe_tile_m0_, d_moe_tile_cnt_, d_moe_num_tiles_};
      const bool small_grouped = (T >= 2 && T <= 12 && H == 2304 && cfg_.moe_I == 1152 && cfg_.top_k == 8 && cfg_.n_experts == 128);
      if (small_grouped) {
        ink2::launch_moe_route_bucket_small(d_rlog_, L.router_bias.d_ptr, L.router_bias.f32, d_eid_, d_ew_,
                                            &gbuf, T, cfg_.n_experts, cfg_.top_k, cfg_.route_scale, stream_);
      } else {
        ink2::launch_moe_route(d_rlog_, L.router_bias.d_ptr, L.router_bias.f32, d_eid_, d_ew_, T, cfg_.n_experts,
                               cfg_.top_k, cfg_.route_scale, stream_);
      }
      const bool fuse_mlp_sconv = (T <= 32 && H == 2304 && cfg_.moe_I == 1152 && cfg_.top_k == 8);
      if (!L.is_mxfp8) {
        ink2::launch_moe_nvfp4(d_x_mlp_, d_eid_, d_ew_, L.w13, L.w13_sc, L.w13_outer, L.w2, L.w2_sc, L.w2_outer,
                               d_moe_act_, d_moe_pair_out_, d_moe_out_, T, H, cfg_.moe_I, cfg_.top_k,
                               &gbuf, cfg_.n_experts, stream_, !fuse_mlp_sconv, small_grouped);
      } else {
        ink2::launch_moe_mxfp8(d_x_mlp_, d_eid_, d_ew_, L.w13, L.w13_sc, L.w2, L.w2_sc,
                               d_moe_act_, d_moe_pair_out_, d_moe_out_, T, H, cfg_.moe_I, cfg_.top_k,
                               &gbuf, cfg_.n_experts, stream_, !fuse_mlp_sconv, small_grouped);
      }

      float* next_x = (!have_prelude && int(v_idx) + 1 < int(cfg_.schedule.size()) &&
                       cfg_.schedule[v_idx + 1] == cfg_.core_start)
                          ? d_x_pre_
                          : d_x_;
      if (fuse_mlp_sconv) {
        ink2::launch_moe_reduce8_sconv_static(d_moe_pair_out_, L.mlp_sconv.d_ptr, L.mlp_sconv.f32, conv_m, next_x,
                                              T, H, cfg_.top_k, sconv_static_scale, ffn_ptr,
                                              save_snaps, snap_m, snap_stride, stream_);
      } else {
        ink2::launch_sconv_static(d_moe_out_, L.mlp_sconv.d_ptr, L.mlp_sconv.f32, conv_m, next_x, T, H,
                                  sconv_static_scale, stream_, ffn_ptr, nullptr, save_snaps, snap_m, snap_stride);
      }
      cur_x = next_x;
    }

    if (!need_logits && !need_all_logits && !(capture_taps && mtp_enabled_)) return;

    {
      const float* st[6];
      for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
      st[n_blocks] = cur_x;
      const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
      ink2::launch_attn_res(st, n_blocks + 1, st_mask, output_query_.d_ptr, output_query_.f32, 1, d_h_norm_, T, H,
                            cfg_.rms_eps, ar_scale, stream_, final_norm_.d_ptr, final_norm_.f32);
    }

    if (!need_logits && !need_all_logits) return;

    if (need_all_logits) {
      if (use_q8_head_) {
        ink2::launch_linear_q8_32(d_h_norm_, d_unembed_q8_, d_unembed_q8_sc_, d_verify_logits_, T, cfg_.vocab, H, stream_);
      } else {
        ink2::launch_linear(cublas_, d_h_norm_, unembed_.d_ptr, unembed_.f32, d_x_bf16_, d_verify_logits_, T,
                            cfg_.vocab, H, false, stream_);
      }
      ink2::launch_softcap(d_verify_logits_, T * cfg_.vocab, cfg_.logit_softcap, stream_);
      if (suppress_active_) ink2::launch_suppress_mask_rows(d_verify_logits_, T, cfg_.vocab, d_suppress_, stream_);
    } else if (use_scr_argmax) {
      const float* h_last = d_h_norm_ + size_t(T - 1) * size_t(H);
      ink2::launch_screen_argmax_bf16(
          static_cast<const __nv_bfloat16*>(unembed_.d_ptr), d_screen_q8_, d_screen_sc_, d_screen_eps_, h_last, 1,
          cfg_.vocab, H, suppress_active_ ? d_suppress_ : nullptr, d_screen_hnorm_, d_screen_max_lb_,
          d_screen_dot_hat_, d_screen_cand_ids_, d_screen_cand_cnt_, kScreenMaxCand, cfg_.logit_softcap, d_tok_out_,
          stream_);
    } else {
      const float* h_last = d_h_norm_ + size_t(T - 1) * size_t(H);
      if (use_q8_head_) {
        ink2::launch_linear_q8_32(h_last, d_unembed_q8_, d_unembed_q8_sc_, d_logits_, 1, cfg_.vocab, H, stream_);
      } else {
        ink2::launch_linear(cublas_, h_last, unembed_.d_ptr, unembed_.f32, d_x_bf16_, d_logits_, 1, cfg_.vocab, H,
                            false, stream_);
      }
      ink2::launch_softcap(d_logits_, cfg_.vocab, cfg_.logit_softcap, stream_);
      if (suppress_active_) ink2::launch_suppress_mask(d_logits_, cfg_.vocab, d_suppress_, stream_);
    }
  };

  const int argmax_idx = use_scr_argmax ? 1 : 0;
  if (d_pos_ptr_arg != nullptr) {
    run_forward_body(d_pos_ptr_arg);
  } else if (T == 1 && need_logits && !need_all_logits && !capture_taps && !save_snaps && use_graph_) {
    const int sup_idx = suppress_active_ ? 1 : 0;
    const int32_t sp = int32_t(pos0);
    cuda_ok(cudaMemcpyAsync(d_step_pos_, &sp, sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "upload step pos");
    if (!decode_graph_exec_[argmax_idx][sup_idx]) {
      cuda_ok(cudaStreamSynchronize(stream_), "sync before graph capture");
      cuda_ok(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin graph capture");
      run_forward_body(d_step_pos_);
      cuda_ok(cudaStreamEndCapture(stream_, &decode_graph_[argmax_idx][sup_idx]), "end graph capture");
      cuda_ok(cudaGraphInstantiate(&decode_graph_exec_[argmax_idx][sup_idx], decode_graph_[argmax_idx][sup_idx],
                                   nullptr, nullptr, 0),
              "instantiate graph");
    }
    cuda_ok(cudaGraphLaunch(decode_graph_exec_[argmax_idx][sup_idx], stream_), "launch decode graph");
  } else {
    run_forward_body(nullptr);
  }
  if (need_logits && !need_all_logits) logits_valid_ = !use_scr_argmax;
}

void EngineInk2::forward_tree_verify(int M, int pos0, const int32_t* d_pos_ptr) {
  const int H = cfg_.hidden;
  const int QD = cfg_.hq * cfg_.head_dim;
  const int KVD = cfg_.hkv * cfg_.head_dim;
  const int stage_stride = 2 * KVD + 2 * H;
  const float sconv_static_scale = 1.0f / std::sqrt(5.0f);
  const float ar_scale = 1.0f / std::sqrt(float(H));

  ink2::launch_embed_norm(d_tokens_, 0, static_cast<const __nv_bfloat16*>(embed_.d_ptr),
                          static_cast<const __nv_bfloat16*>(embed_norm_.d_ptr), d_x_, M, H, cfg_.rms_eps, stream_);

  const float* blocks_vec[6] = {};
  float* bslot_ptr[6] = {d_blocks_[0], d_blocks_[1], d_blocks_[2], d_blocks_[3], d_blocks_[4], d_blocks_[5]};
  const float* bank0_ptr = d_blocks_[4];
  float* cur_x = d_x_;
  int n_blocks = 0;
  bool have_prelude = false;
  bool have_prev = false;
  int n_prelude_blocks = 0;
  const size_t row_bytes = size_t(M) * size_t(H) * sizeof(float);

  for (size_t v_idx = 0; v_idx < cfg_.schedule.size(); ++v_idx) {
    const int ip = cfg_.schedule[v_idx];
    const Ink2Layer& L = layers_[size_t(ip)];
    float* conv_k = d_conv_[v_idx] + 0;
    float* conv_v = d_conv_[v_idx] + 3 * KVD;
    float* conv_a = d_conv_[v_idx] + 6 * KVD;
    float* conv_m = d_conv_[v_idx] + 6 * KVD + 3 * H;
    float* stage_v_sc = d_stage_sconv_ + v_idx * size_t(kMaxTree) * size_t(stage_stride);
    float* stage_k_v = d_stage_k_ + v_idx * size_t(kMaxTree) * size_t(KVD);
    float* stage_v_v = d_stage_v_ + v_idx * size_t(kMaxTree) * size_t(KVD);

    if (ip == cfg_.core_start) {
      if (!have_prelude) {
        if (cur_x != d_x_pre_) {
          cuda_ok(cudaMemcpyAsync(d_x_pre_, cur_x, row_bytes, cudaMemcpyDeviceToDevice, stream_), "copy tree x_pre");
          cur_x = d_x_pre_;
        }
        n_prelude_blocks = n_blocks;
        if (n_prelude_blocks > cfg_.core_start / cfg_.block_sz) {
          for (int b = 0; b < n_prelude_blocks; ++b)
            cuda_ok(cudaMemcpyAsync(d_pre_blocks_[b], blocks_vec[b], row_bytes, cudaMemcpyDeviceToDevice, stream_),
                    "copy tree pre_block");
        }
        have_prelude = true;
      } else {
        const float* prev_mid = blocks_vec[n_prelude_blocks + 1];
        if (n_prelude_blocks <= cfg_.core_start / cfg_.block_sz && prev_mid == bslot_ptr[2]) {
          std::swap(bslot_ptr[2], bslot_ptr[4]);
          bank0_ptr = bslot_ptr[4];
        } else {
          cuda_ok(cudaMemcpyAsync(bslot_ptr[4], prev_mid, row_bytes, cudaMemcpyDeviceToDevice, stream_), "copy tree bank0");
          bank0_ptr = bslot_ptr[4];
        }
        ink2::launch_rmsnorm(cur_x, nullptr, true, bslot_ptr[5], M, H, cfg_.rms_eps, stream_);
        cur_x = d_x_pre_;
        n_blocks = n_prelude_blocks;
        if (n_prelude_blocks > cfg_.core_start / cfg_.block_sz) {
          for (int b = 0; b < n_prelude_blocks; ++b) {
            cuda_ok(cudaMemcpyAsync(d_blocks_[b], d_pre_blocks_[b], row_bytes, cudaMemcpyDeviceToDevice, stream_),
                    "restore tree pre_block");
            blocks_vec[b] = d_blocks_[b];
          }
        }
        have_prev = true;
      }
    }

    if (mtp_enabled_) {
      for (int ti = 0; ti < 3; ++ti) {
        if (int(v_idx) == mtp_cfg_.taps[ti]) {
          cuda_ok(cudaMemcpyAsync(d_tap_[ti], cur_x, row_bytes, cudaMemcpyDeviceToDevice, stream_), "capture tree tap");
        }
      }
    }

    const bool has_msgs = have_prev && ip < cfg_.core_end;
    const float* msg_m0 = d_msg_out_ + 0 * size_t(M) * size_t(H);
    const float* msg_m1 = d_msg_out_ + 1 * size_t(M) * size_t(H);
    const float* msg_m2 = d_msg_out_ + 2 * size_t(M) * size_t(H);
    const float* msg_mlp = d_msg_out_ + 3 * size_t(M) * size_t(H);
    if (has_msgs) {
      const float* bank[2] = {bank0_ptr, bslot_ptr[5]};
      const float* qv = d_msg_queries_ + size_t(ip - cfg_.core_start) * 4 * size_t(H);
      ink2::launch_attn_res(bank, 2, 0x3u, qv, true, 4, d_msg_out_, M, H, cfg_.rms_eps, ar_scale, stream_);
    }

    const float* in_ptrs[3];
    bool fused_attn_norm = false;
    if (ip == 0) {
      in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = cur_x;
    } else {
      const float* st[6];
      for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
      st[n_blocks] = cur_x;
      const Ink2Tensor& aq = attn_queries_[size_t(ip - 1)];
      const int nq = (aq.cols == 1) ? 1 : aq.rows;
      const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
      if (!has_msgs && L.is_swa && nq == 1) {
        ink2::launch_attn_res(st, n_blocks + 1, st_mask, aq.d_ptr, aq.f32, 1, d_n0_, M, H, cfg_.rms_eps, ar_scale,
                              stream_, L.attn_norm.d_ptr, L.attn_norm.f32);
        fused_attn_norm = true;
        in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = d_n0_;
      } else {
        ink2::launch_attn_res(st, n_blocks + 1, st_mask, aq.d_ptr, aq.f32, nq, d_ar_out_, M, H, cfg_.rms_eps,
                              ar_scale, stream_);
        if (nq == 1) {
          in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = d_ar_out_;
        } else {
          in_ptrs[0] = d_ar_out_ + 0 * size_t(M) * size_t(H);
          in_ptrs[1] = d_ar_out_ + 1 * size_t(M) * size_t(H);
          in_ptrs[2] = d_ar_out_ + 2 * size_t(M) * size_t(H);
        }
      }
    }

    if (has_msgs) {
      const ink2::MsgGateItem items[3] = {
          {in_ptrs[0], msg_m0, static_cast<const __nv_bfloat16*>(L.ctrl_attn.down.d_ptr),
           static_cast<const __nv_bfloat16*>(L.ctrl_attn.forget.d_ptr),
           static_cast<const __nv_bfloat16*>(L.ctrl_attn.gain.d_ptr), d_n0_},
          {in_ptrs[1], msg_m1, static_cast<const __nv_bfloat16*>(L.ctrl_k.down.d_ptr),
           static_cast<const __nv_bfloat16*>(L.ctrl_k.forget.d_ptr),
           static_cast<const __nv_bfloat16*>(L.ctrl_k.gain.d_ptr), d_n1_},
          {in_ptrs[2], msg_m2, static_cast<const __nv_bfloat16*>(L.ctrl_v.down.d_ptr),
           static_cast<const __nv_bfloat16*>(L.ctrl_v.forget.d_ptr),
           static_cast<const __nv_bfloat16*>(L.ctrl_v.gain.d_ptr), d_n2_},
      };
      ink2::launch_msg_gate_x3(items, d_mg_nhm_, d_mg_g_, M, H, cfg_.gate_rank, cfg_.rms_eps, stream_);
      if (L.is_swa) {
        ink2::launch_rmsnorm_x3(d_n0_, d_n1_, d_n2_, L.attn_norm.d_ptr, L.attn_norm.d_ptr, L.attn_norm.d_ptr,
                                L.attn_norm.f32, d_n0_, d_n1_, d_n2_, M, H, cfg_.rms_eps, stream_);
      } else {
        ink2::launch_rmsnorm_x3(d_n0_, d_n1_, d_n2_, L.q_norm.d_ptr, L.k_norm.d_ptr, L.v_norm.d_ptr,
                                L.q_norm.f32, d_n0_, d_n1_, d_n2_, M, H, cfg_.rms_eps, stream_);
      }
    } else if (L.is_swa) {
      if (!fused_attn_norm) {
        ink2::launch_rmsnorm(in_ptrs[0], L.attn_norm.d_ptr, L.attn_norm.f32, d_n0_, M, H, cfg_.rms_eps, stream_);
      }
    } else {
      ink2::launch_rmsnorm_x3(in_ptrs[0], in_ptrs[1], in_ptrs[2], L.q_norm.d_ptr, L.k_norm.d_ptr, L.v_norm.d_ptr,
                              L.q_norm.f32, d_n0_, d_n1_, d_n2_, M, H, cfg_.rms_eps, stream_);
    }

    const float* kn = (has_msgs || !L.is_swa) ? d_n1_ : d_n0_;
    const float* vn = (has_msgs || !L.is_swa) ? d_n2_ : d_n0_;
    auto mm = [&](const Ink2Tensor& w, const float* x, float* y) {
      ink2::launch_linear(cublas_, x, w.d_ptr, w.f32, d_x_bf16_, y, M, w.rows, w.cols, false, stream_);
    };

    if (L.is_swa && !has_msgs && L.fused_qkvrsg) {
      const ink2::MultiLinearSeg segs[6] = {
          {d_q_, L.wq_du.rows}, {d_r_, L.wr_du.rows}, {d_sinks_, L.attn_sink_proj.rows},
          {d_gate_attn_, L.gate_attn.rows}, {d_k_, L.wk_dv.rows}, {d_v_, L.wv_dv.rows},
      };
      ink2::launch_linear_multi(cublas_, d_n0_, L.w_qkvrsg, d_x_bf16_, d_qkvrsg_tmp_, segs, 6, M, H, stream_);
    } else {
      if (L.fused_qrsg) {
        const ink2::MultiLinearSeg segs[4] = {
            {d_q_, L.wq_du.rows}, {d_r_, L.wr_du.rows}, {d_sinks_, L.attn_sink_proj.rows}, {d_gate_attn_, L.gate_attn.rows},
        };
        ink2::launch_linear_multi(cublas_, d_n0_, L.w_qkvrsg, d_x_bf16_, d_qkvrsg_tmp_, segs, 4, M, H, stream_);
      } else {
        mm(L.wq_du, d_n0_, d_q_);
        mm(L.wr_du, d_n0_, d_r_);
        mm(L.attn_sink_proj, d_n0_, d_sinks_);
        mm(L.gate_attn, d_n0_, d_gate_attn_);
      }
      if (L.is_swa || !L.fused_kvp) {
        mm(L.wk_dv, kn, d_k_);
        mm(L.wv_dv, vn, d_v_);
      }
    }

    if (L.is_swa) {
      ink2::launch_sconv_static_kv_tree(d_k_, L.k_sconv.d_ptr, L.k_sconv.f32, conv_k,
                                        d_v_, L.v_sconv.d_ptr, L.v_sconv.f32, conv_v,
                                        stage_v_sc, stage_stride, d_tree_p1_, d_tree_p2_, d_tree_p3_,
                                        M, KVD, sconv_static_scale, stream_);
    } else {
      if (L.fused_kvp) {
        const ink2::MultiLinearSeg ksegs[2] = {{d_k_, L.wk_dv.rows}, {d_dyn_, L.k_proj.rows}};
        ink2::launch_linear_multi(cublas_, kn, L.w_kkp, d_x_bf16_, d_qkvrsg_tmp_, ksegs, 2, M, H, stream_);
      } else {
        mm(L.k_proj, kn, d_dyn_);
      }
      ink2::launch_sconv_dynamic_tree(d_k_, d_dyn_, L.k_static.d_ptr, L.k_static.f32, conv_k, d_k_,
                                      stage_v_sc, stage_stride, 0, d_tree_p1_, d_tree_p2_, d_tree_p3_,
                                      M, KVD, cfg_.hkv, stream_);
      if (L.fused_kvp) {
        const ink2::MultiLinearSeg vsegs[2] = {{d_v_, L.wv_dv.rows}, {d_dyn_, L.v_proj.rows}};
        ink2::launch_linear_multi(cublas_, vn, L.w_vvp, d_x_bf16_, d_qkvrsg_tmp_, vsegs, 2, M, H, stream_);
      } else {
        mm(L.v_proj, vn, d_dyn_);
      }
      ink2::launch_sconv_dynamic_tree(d_v_, d_dyn_, L.v_static.d_ptr, L.v_static.f32, conv_v, d_v_,
                                      stage_v_sc, stage_stride, KVD, d_tree_p1_, d_tree_p2_, d_tree_p3_,
                                      M, KVD, cfg_.hkv, stream_);
    }

    const int kv_cap = L.is_swa ? ring_rows_ : max_ctx_;
    ink2::launch_qkvr_prep_tree(d_q_, d_k_, d_v_, d_r_, L.q_scale.d_ptr, L.rel_scale.d_ptr, L.q_scale.f32,
                                stage_k_v, stage_v_v, M, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.d_rel,
                                pos0, d_pos_ptr, d_tree_depth_, cfg_.qk_clip, cfg_.rel_clip, cfg_.rms_eps,
                                cfg_.log_floor, cfg_.log_alpha, L.is_swa, stream_);
    ink2::launch_rel_bias(cublas_, d_r_, L.rel_proj.d_ptr, L.rel_proj.f32, d_x_bf16_, d_rb_, M, cfg_.hq,
                          cfg_.d_rel, cfg_.rel_extent, stream_);
    ink2::launch_attn_tree(d_q_, d_kc_[v_idx], d_vc_[v_idx], stage_k_v, stage_v_v, d_rb_, d_sinks_, d_gate_attn_,
                           d_part_m_, d_part_l_, d_part_o_, d_att_out_, M, pos0, d_pos_ptr, d_tree_depth_,
                           d_tree_anc_mask_, kv_cap, cfg_.hq, cfg_.hkv, cfg_.head_dim, cfg_.rel_extent,
                           L.is_swa, cfg_.swa, kv_bf16_, stream_);

    mm(L.wo_ud, d_att_out_, d_a_out_);
    const float* ffn_ptr = d_a_out_;
    if (ip % cfg_.block_sz == 0) {
      ink2::launch_sconv_static_tree(d_a_out_, L.attn_sconv.d_ptr, L.attn_sconv.f32, conv_a, d_a_out_,
                                     stage_v_sc, stage_stride, 2 * KVD, d_tree_p1_, d_tree_p2_, d_tree_p3_,
                                     M, H, sconv_static_scale, nullptr, stream_);
      const int bslot = ip / cfg_.block_sz;
      ink2::launch_rmsnorm(cur_x, nullptr, true, bslot_ptr[bslot], M, H, cfg_.rms_eps, stream_);
      blocks_vec[size_t(n_blocks++)] = bslot_ptr[bslot];
    } else {
      ink2::launch_sconv_static_tree(d_a_out_, L.attn_sconv.d_ptr, L.attn_sconv.f32, conv_a, d_ffn_inp_,
                                     stage_v_sc, stage_stride, 2 * KVD, d_tree_p1_, d_tree_p2_, d_tree_p3_,
                                     M, H, sconv_static_scale, cur_x, stream_);
      ffn_ptr = d_ffn_inp_;
    }

    if (has_msgs) {
      const float* st[6];
      for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
      st[n_blocks] = ffn_ptr;
      const Ink2Tensor& mq = mlp_queries_[size_t(ip)];
      const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
      ink2::launch_attn_res(st, n_blocks + 1, st_mask, mq.d_ptr, mq.f32, 1, d_h_mlp_, M, H, cfg_.rms_eps, ar_scale, stream_);
      ink2::launch_msg_gate(cublas_, d_h_mlp_, msg_mlp, static_cast<const __nv_bfloat16*>(L.ctrl_mlp.down.d_ptr),
                            static_cast<const __nv_bfloat16*>(L.ctrl_mlp.forget.d_ptr),
                            static_cast<const __nv_bfloat16*>(L.ctrl_mlp.gain.d_ptr),
                            d_mg_nhm_, d_mg_nhm_bf16_, d_mg_g_, d_mg_g_bf16_, d_mg_df_, d_mg_dg_,
                            d_h_mlp_, M, H, cfg_.gate_rank, cfg_.rms_eps, stream_);
      ink2::launch_rmsnorm(d_h_mlp_, L.mlp_norm.d_ptr, L.mlp_norm.f32, d_x_mlp_, M, H, cfg_.rms_eps, stream_);
    } else {
      const float* st[6];
      for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
      st[n_blocks] = ffn_ptr;
      const Ink2Tensor& mq = mlp_queries_[size_t(ip)];
      const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
      ink2::launch_attn_res(st, n_blocks + 1, st_mask, mq.d_ptr, mq.f32, 1, d_x_mlp_, M, H, cfg_.rms_eps, ar_scale,
                            stream_, L.mlp_norm.d_ptr, L.mlp_norm.f32);
    }

    mm(L.router_w, d_x_mlp_, d_rlog_);
    const ink2::MoeGroupBufs gbuf{d_moe_pair_idx_, d_moe_tile_exp_, d_moe_tile_m0_, d_moe_tile_cnt_, d_moe_num_tiles_};
    const bool small_grouped = (M >= 2 && M <= 12 && H == 2304 && cfg_.moe_I == 1152 && cfg_.top_k == 8 && cfg_.n_experts == 128);
    if (small_grouped) {
      ink2::launch_moe_route_bucket_small(d_rlog_, L.router_bias.d_ptr, L.router_bias.f32, d_eid_, d_ew_,
                                          &gbuf, M, cfg_.n_experts, cfg_.top_k, cfg_.route_scale, stream_);
    } else {
      ink2::launch_moe_route(d_rlog_, L.router_bias.d_ptr, L.router_bias.f32, d_eid_, d_ew_, M, cfg_.n_experts,
                             cfg_.top_k, cfg_.route_scale, stream_);
    }
    const bool fuse_mlp_sconv = (M <= 32 && H == 2304 && cfg_.moe_I == 1152 && cfg_.top_k == 8);
    if (!L.is_mxfp8) {
      ink2::launch_moe_nvfp4(d_x_mlp_, d_eid_, d_ew_, L.w13, L.w13_sc, L.w13_outer, L.w2, L.w2_sc, L.w2_outer,
                             d_moe_act_, d_moe_pair_out_, d_moe_out_, M, H, cfg_.moe_I, cfg_.top_k,
                             &gbuf, cfg_.n_experts, stream_, !fuse_mlp_sconv, small_grouped);
    } else {
      ink2::launch_moe_mxfp8(d_x_mlp_, d_eid_, d_ew_, L.w13, L.w13_sc, L.w2, L.w2_sc,
                             d_moe_act_, d_moe_pair_out_, d_moe_out_, M, H, cfg_.moe_I, cfg_.top_k,
                             &gbuf, cfg_.n_experts, stream_, !fuse_mlp_sconv, small_grouped);
    }

    float* next_x = (!have_prelude && int(v_idx) + 1 < int(cfg_.schedule.size()) &&
                     cfg_.schedule[v_idx + 1] == cfg_.core_start)
                        ? d_x_pre_
                        : d_x_;
    if (fuse_mlp_sconv) {
      ink2::launch_moe_reduce8_sconv_static_tree(d_moe_pair_out_, L.mlp_sconv.d_ptr, L.mlp_sconv.f32, conv_m, next_x,
                                                 stage_v_sc, stage_stride, 2 * KVD + H, d_tree_p1_, d_tree_p2_,
                                                 d_tree_p3_, M, H, cfg_.top_k, sconv_static_scale, ffn_ptr, stream_);
    } else {
      ink2::launch_sconv_static_tree(d_moe_out_, L.mlp_sconv.d_ptr, L.mlp_sconv.f32, conv_m, next_x,
                                     stage_v_sc, stage_stride, 2 * KVD + H, d_tree_p1_, d_tree_p2_, d_tree_p3_,
                                     M, H, sconv_static_scale, ffn_ptr, stream_);
    }
    cur_x = next_x;
  }

  {
    const float* st[6];
    for (int b = 0; b < n_blocks; ++b) st[b] = blocks_vec[b];
    st[n_blocks] = cur_x;
    const uint32_t st_mask = (n_blocks > 0) ? ((1u << n_blocks) - 1u) : 0u;
    ink2::launch_attn_res(st, n_blocks + 1, st_mask, output_query_.d_ptr, output_query_.f32, 1, d_h_norm_, M, H,
                          cfg_.rms_eps, ar_scale, stream_, final_norm_.d_ptr, final_norm_.f32);
  }

  ink2::launch_screen_argmax_bf16(
      static_cast<const __nv_bfloat16*>(unembed_.d_ptr), d_screen_q8_, d_screen_sc_, d_screen_eps_, d_h_norm_, M,
      cfg_.vocab, H, suppress_active_ ? d_suppress_ : nullptr, d_screen_hnorm_, d_screen_max_lb_,
      d_screen_dot_hat_, d_screen_cand_ids_, d_screen_cand_cnt_, kScreenMaxCand, cfg_.logit_softcap,
      d_verify_samples_, stream_);
}

void EngineInk2::draft_beam_tree(int32_t anchor_tok, int pos0, int max_nodes, int room, TreeSpec& out_tree) {
  const int B = mtp_cfg_.block;
  const int H = mtp_cfg_.hidden;
  const int R = mtp_cfg_.markov_rank;
  const int V = cfg_.vocab;
  const int QD = mtp_cfg_.hq * mtp_cfg_.head_dim;
  const int KVD = mtp_cfg_.hkv * mtp_cfg_.head_dim;
  const int sup_idx = suppress_active_ ? 1 : 0;

  cuda_ok(cudaMemcpyAsync(d_tokens_, &anchor_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "upload beam anc");
  const int32_t sp = int32_t(pos0);
  cuda_ok(cudaMemcpyAsync(d_step_pos_, &sp, sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "upload beam pos");

  auto emit_beam_draft = [&]() {
    ink2::launch_mtp_prep_block(d_tokens_, static_cast<const __nv_bfloat16*>(embed_.d_ptr),
                                static_cast<const __nv_bfloat16*>(embed_norm_.d_ptr),
                                static_cast<const __nv_bfloat16*>(mtp_mask_embed_.d_ptr),
                                d_mtp_blk_x_, B, H, mtp_cfg_.rms_eps, stream_);
    for (int i = 0; i < mtp_cfg_.num_layers; ++i) {
      const Ink2MtpLayer& L = mtp_layers_[size_t(i)];
      ink2::launch_rmsnorm(d_mtp_blk_x_, L.in_norm.d_ptr, L.in_norm.f32, d_mtp_blk_norm_, B, H, mtp_cfg_.rms_eps, stream_);
      const ink2::MultiLinearSeg qkv_segs[3] = {{d_mtp_blk_q_, QD}, {d_mtp_blk_k_, KVD}, {d_mtp_blk_v_, KVD}};
      ink2::launch_linear_multi(cublas_, d_mtp_blk_norm_, L.w_qkv, d_x_bf16_, d_mtp_qkv_tmp_, qkv_segs, 3, B, H, stream_);
      const int kv_cap = L.is_local ? ring_rows_ : max_ctx_;
      ink2::launch_mtp_qk_rope_cache(d_mtp_blk_q_, d_mtp_blk_k_, d_mtp_blk_v_,
                                     static_cast<const __nv_bfloat16*>(L.q_norm.d_ptr),
                                     static_cast<const __nv_bfloat16*>(L.k_norm.d_ptr),
                                     d_mtp_kc_[i], d_mtp_vc_[i], B, mtp_cfg_.hq, mtp_cfg_.hkv, mtp_cfg_.head_dim,
                                     0, d_step_pos_, kv_cap, L.is_local, true, mtp_cfg_.rms_eps, mtp_cfg_.rope_theta,
                                     kv_bf16_, stream_);
      ink2::launch_mtp_attn(d_mtp_blk_q_, d_mtp_kc_[i], d_mtp_vc_[i], d_mtp_blk_k_, d_mtp_blk_v_, d_mtp_blk_att_,
                            B, 0, d_step_pos_, kv_cap, mtp_cfg_.hq, mtp_cfg_.hkv, mtp_cfg_.head_dim,
                            L.is_local, mtp_cfg_.swa, kv_bf16_, stream_);
      ink2::launch_linear(cublas_, d_mtp_blk_att_, L.o_proj.d_ptr, L.o_proj.f32, d_x_bf16_, d_mtp_blk_out_,
                          B, H, QD, false, stream_);
      ink2::launch_mtp_res_add_bf16(d_mtp_blk_x_, d_mtp_blk_out_, d_mtp_blk_res_, B * H, stream_);
      ink2::launch_rmsnorm(d_mtp_blk_res_, L.ff_norm.d_ptr, L.ff_norm.f32, d_mtp_blk_norm_, B, H, mtp_cfg_.rms_eps, stream_);
      ink2::launch_mtp_gate_up_swiglu(d_mtp_blk_norm_, L.w_gu, d_mtp_blk_gate_, B, mtp_cfg_.ffn, H, stream_);
      ink2::launch_linear(cublas_, d_mtp_blk_gate_, L.down_proj.d_ptr, L.down_proj.f32, d_x_bf16_, d_mtp_blk_out_,
                          B, H, mtp_cfg_.ffn, false, stream_);
      ink2::launch_mtp_res_add_bf16(d_mtp_blk_res_, d_mtp_blk_out_, d_mtp_blk_x_, B * H, stream_);
    }
    ink2::launch_rmsnorm(d_mtp_blk_x_, mtp_norm_.d_ptr, mtp_norm_.f32, d_mtp_blk_norm_, B, H, mtp_cfg_.rms_eps, stream_);
    ink2::launch_dspark_screen_dp4a7(d_screen_q8_, d_screen_sc_, d_mtp_blk_norm_ + H, d_beam_h_q8_, d_beam_h_sc_,
                                     suppress_active_ ? d_suppress_ : nullptr, d_mtp_logits_, V, H, stream_);
    ink2::launch_dspark_top_cand(d_mtp_logits_, V, kDsparkKCand, d_beam_top_tmp_ids_, d_beam_top_tmp_vals_,
                                 d_beam_cand_ids_, nullptr, stream_);
    ink2::launch_dspark_beam_init(d_tokens_, d_beam_state_, d_beam_prev_tok_, d_beam_logp_, d_beam_node_,
                                  d_pool_tok_, d_pool_parent_, d_pool_depth_, d_pool_logp_, d_pool_cnt_, stream_);
    for (int s = 0; s < 7; ++s) {
      const int nw = (s == 0) ? 1 : kBeamW;
      const float* hs = d_mtp_blk_norm_ + size_t(1 + s) * size_t(H);
      ink2::launch_dspark_markov_prep_batch(
          d_beam_prev_tok_, d_beam_state_, hs, d_mtp_rec_feat_,
          static_cast<const __nv_bfloat16*>(mtp_markov_w1_.d_ptr), d_beam_joint_, nw, R, H, stream_);
      ink2::launch_linear(cublas_, d_beam_joint_, mtp_rnn_proj_w_.d_ptr, mtp_rnn_proj_w_.f32, d_x_bf16_, d_beam_proj_,
                          nw, 3 * R, 2 * R + 2 * H, false, stream_);
      ink2::launch_dspark_markov_update_batch(
          d_beam_proj_, static_cast<const __nv_bfloat16*>(mtp_rnn_proj_b_.d_ptr),
          d_beam_state_, d_beam_state_next_, d_beam_lat_, nw, R, stream_);
      ink2::launch_linear(cublas_, d_beam_lat_, mtp_hidden_out_w_.d_ptr, mtp_hidden_out_w_.f32, d_x_bf16_, d_beam_corr_,
                          nw, H, R, false, stream_);
      ink2::launch_dspark_add_bias_h_batch(d_beam_corr_, d_mtp_ho_b_, hs, nw, H, stream_);
      ink2::launch_dspark_score_cand_dp4a_batch(
          d_screen_q8_, d_screen_sc_, d_beam_corr_, d_beam_cand_ids_ + size_t(s) * size_t(kDsparkKCand),
          suppress_active_ ? d_suppress_ : nullptr, kDsparkKCand, nw, cfg_.logit_softcap, d_beam_cand_scores_, stream_);
      ink2::launch_dspark_beam_expand(
          d_beam_cand_scores_, d_beam_cand_ids_ + size_t(s) * size_t(kDsparkKCand), d_beam_state_next_, kDsparkKCand,
          nw, kBeamW, s + 1, d_beam_state_, d_beam_prev_tok_, d_beam_logp_, d_beam_node_, d_pool_tok_,
          d_pool_parent_, d_pool_depth_, d_pool_logp_, d_pool_cnt_, stream_);
    }
  };

  if (use_graph_) {
    if (!beam_draft_graph_exec_[sup_idx]) {
      cuda_ok(cudaStreamSynchronize(stream_), "sync before beam draft graph");
      cuda_ok(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin beam draft graph");
      emit_beam_draft();
      cuda_ok(cudaStreamEndCapture(stream_, &beam_draft_graph_[sup_idx]), "end beam draft graph");
      cuda_ok(cudaGraphInstantiate(&beam_draft_graph_exec_[sup_idx], beam_draft_graph_[sup_idx],
                                   nullptr, nullptr, 0),
              "instantiate beam draft graph");
    }
    cuda_ok(cudaGraphLaunch(beam_draft_graph_exec_[sup_idx], stream_), "launch beam draft graph");
  } else {
    emit_beam_draft();
  }

  cuda_ok(cudaMemcpyAsync(h_pool_cnt_, d_pool_cnt_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_), "copy pool cnt");
  cuda_ok(cudaMemcpyAsync(h_pool_tok_, d_pool_tok_, size_t(kBeamPoolCap) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_), "copy pool tok");
  cuda_ok(cudaMemcpyAsync(h_pool_parent_, d_pool_parent_, size_t(kBeamPoolCap) * sizeof(int8_t), cudaMemcpyDeviceToHost, stream_), "copy pool par");
  cuda_ok(cudaMemcpyAsync(h_pool_depth_, d_pool_depth_, size_t(kBeamPoolCap) * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_), "copy pool dep");
  cuda_ok(cudaMemcpyAsync(h_pool_logp_, d_pool_logp_, size_t(kBeamPoolCap) * sizeof(float), cudaMemcpyDeviceToHost, stream_), "copy pool lp");
  cuda_ok(cudaStreamSynchronize(stream_), "sync beam pool");

  const int cnt = std::min(kBeamPoolCap, std::max(1, *h_pool_cnt_));
  int ord[kBeamPoolCap];
  int n_cand = 0;
  for (int i = 1; i < cnt; ++i) {
    if (int(h_pool_depth_[i]) <= room && h_pool_logp_[i] > -1e20f) ord[n_cand++] = i;
  }
  std::sort(ord, ord + n_cand, [&](int a, int b) {
    if (h_pool_logp_[a] != h_pool_logp_[b]) return h_pool_logp_[a] > h_pool_logp_[b];
    if (h_pool_depth_[a] != h_pool_depth_[b]) return h_pool_depth_[a] < h_pool_depth_[b];
    return a < b;
  });

  const int max_usable = std::min(max_nodes, 1 + n_cand);
  static constexpr int kBuckets[kNumTreeBuckets] = {2, 3, 4, 5, 6, 8, 10, 12};
  double pref_p[kBeamPoolCap + 1];
  pref_p[0] = 1.0;
  for (int i = 0; i < n_cand; ++i) {
    const float lp = h_pool_logp_[ord[i]];
    pref_p[i + 1] = pref_p[i] + ((lp > -20.0f) ? std::exp(double(lp)) : 0.0);
  }
  double best_rate = -1.0;
  int best_m = 0;
  for (int mb : kBuckets) {
    if (mb > max_usable) continue;
    const double e_tok = pref_p[mb - 1];
    const double exp_frac = 1.0 - std::pow(15.0 / 16.0, double(mb));
    const double rate = e_tok / (1.0 + double(moe_ratio_) * exp_frac);
    if (rate > best_rate) {
      best_rate = rate;
      best_m = mb;
    }
  }
  if (best_m < 2) {
    out_tree = TreeSpec{};
    return;
  }

  uint8_t keep[kBeamPoolCap] = {};
  keep[0] = 1;
  int n_kept = 1;
  for (int i = 0; i < n_cand && n_kept < best_m; ++i) {
    const int u = ord[i];
    if (!keep[u]) {
      const int par = int(h_pool_parent_[u]);
      if (par >= 0 && keep[par]) {
        keep[u] = 1;
        ++n_kept;
      }
    }
  }
  // If pruning to connected nodes produced fewer than best_m, snap down to the largest valid bucket <= n_kept
  if (n_kept < best_m) {
    int snapped = 0;
    for (int mb : kBuckets) {
      if (mb <= n_kept) snapped = mb;
    }
    if (snapped < 2) {
      out_tree = TreeSpec{};
      return;
    }
    if (n_kept > snapped) {
      int dropped = 0;
      const int need_drop = n_kept - snapped;
      for (int i = n_cand - 1; i >= 0 && dropped < need_drop; --i) {
        const int u = ord[i];
        if (!keep[u]) continue;
        bool has_kept_child = false;
        for (int p = 1; p < cnt; ++p) {
          if (keep[p] && int(h_pool_parent_[p]) == u) { has_kept_child = true; break; }
        }
        if (!has_kept_child) { keep[u] = 0; ++dropped; --n_kept; }
      }
    }
  }

  out_tree = TreeSpec{};
  int8_t remap[kBeamPoolCap];
  std::memset(remap, -1, sizeof(remap));
  for (int p = 0; p < cnt && out_tree.M < kMaxTree; ++p) {
    if (!keep[p]) continue;
    const int nid = out_tree.M++;
    remap[p] = int8_t(nid);
    out_tree.tok[nid] = h_pool_tok_[p];
    const int8_t par = (p == 0) ? -1 : remap[int(h_pool_parent_[p])];
    out_tree.p1[nid] = par;
    out_tree.p2[nid] = (par >= 0) ? out_tree.p1[par] : -1;
    out_tree.p3[nid] = (par >= 0 && out_tree.p1[par] >= 0) ? out_tree.p1[out_tree.p1[par]] : -1;
    out_tree.depth[nid] = h_pool_depth_[p];
    out_tree.max_depth = std::max(out_tree.max_depth, int(out_tree.depth[nid]));
    uint32_t mask = (1u << nid);
    if (par >= 0) mask |= out_tree.anc_mask[par];
    out_tree.anc_mask[nid] = mask;
  }
}

void EngineInk2::commit_mtp_context(int pos0, int m, const int32_t* d_pos_ptr, bool restore_conv) {
  if (!mtp_enabled_ || m <= 0) return;
  if (restore_conv) {
    const size_t cbytes = cfg_.schedule.size() * size_t(kVisitConvFloats) * sizeof(float);
    cuda_ok(cudaMemcpyAsync(d_conv_base_, d_verify_conv_ + size_t(m - 1) * cfg_.schedule.size() * size_t(kVisitConvFloats),
                            cbytes, cudaMemcpyDeviceToDevice, stream_),
            "restore sconv");
  }
  const int H = mtp_cfg_.hidden;
  const int KVD = mtp_cfg_.hkv * mtp_cfg_.head_dim;
  for (int i = 0; i < mtp_cfg_.num_layers; ++i) {
    const Ink2MtpLayer& L = mtp_layers_[size_t(i)];
    ink2::launch_mtp_fuse_norm(d_tap_[0], d_tap_[1], d_tap_[2], d_h_norm_, 0,
                               mtp_fuse_[i][0], mtp_fuse_[i][1], mtp_fuse_[i][2], mtp_fuse_[i][3],
                               L.dflare_norm.d_ptr, L.dflare_norm.f32, d_mtp_ctx_h_, m, H, mtp_cfg_.rms_eps, stream_);
    if (L.w_kv_ctx) {
      const ink2::MultiLinearSeg kv_segs[2] = {{d_mtp_ctx_k_, KVD}, {d_mtp_ctx_v_, KVD}};
      ink2::launch_linear_multi(cublas_, d_mtp_ctx_h_, L.w_kv_ctx, d_x_bf16_, d_qkvrsg_tmp_, kv_segs, 2, m, H, stream_);
    } else {
      ink2::launch_linear(cublas_, d_mtp_ctx_h_, L.k_proj_ctx.d_ptr, L.k_proj_ctx.f32, d_x_bf16_, d_mtp_ctx_k_,
                          m, L.k_proj_ctx.rows, H, false, stream_);
      ink2::launch_linear(cublas_, d_mtp_ctx_h_, L.v_proj_ctx.d_ptr, L.v_proj_ctx.f32, d_x_bf16_, d_mtp_ctx_v_,
                          m, L.v_proj_ctx.rows, H, false, stream_);
    }
    const int kv_cap = L.is_local ? ring_rows_ : max_ctx_;
    ink2::launch_mtp_qk_rope_cache(d_mtp_blk_q_, d_mtp_ctx_k_, d_mtp_ctx_v_,
                                   static_cast<const __nv_bfloat16*>(L.q_norm.d_ptr),
                                   static_cast<const __nv_bfloat16*>(L.k_norm_ctx.d_ptr),
                                   d_mtp_kc_[i], d_mtp_vc_[i], m, mtp_cfg_.hq, mtp_cfg_.hkv, mtp_cfg_.head_dim,
                                   pos0, d_pos_ptr, kv_cap, L.is_local, false, mtp_cfg_.rms_eps, mtp_cfg_.rope_theta,
                                   kv_bf16_, stream_);
  }
  ink2::launch_mtp_fuse_norm(d_tap_[0], d_tap_[1], d_tap_[2], d_h_norm_, m - 1,
                             mtp_fuse_[3][0], mtp_fuse_[3][1], mtp_fuse_[3][2], mtp_fuse_[3][3],
                             mtp_rec_norm_.d_ptr, mtp_rec_norm_.f32, d_mtp_rec_feat_, 1, H, mtp_cfg_.rms_eps, stream_);
}

void EngineInk2::mtp_draft_step(int pos0, int Kd, const int32_t* d_pos_ptr, bool greedy, int64_t step_pos) {
  const int B = mtp_cfg_.block;
  const int H = mtp_cfg_.hidden;
  const int R = mtp_cfg_.markov_rank;
  const int V = cfg_.vocab;
  const int QD = mtp_cfg_.hq * mtp_cfg_.head_dim;
  const int KVD = mtp_cfg_.hkv * mtp_cfg_.head_dim;

  ink2::launch_mtp_prep_block(d_tokens_, static_cast<const __nv_bfloat16*>(embed_.d_ptr),
                              static_cast<const __nv_bfloat16*>(embed_norm_.d_ptr),
                              static_cast<const __nv_bfloat16*>(mtp_mask_embed_.d_ptr),
                              d_mtp_blk_x_, B, H, mtp_cfg_.rms_eps, stream_);
  for (int i = 0; i < mtp_cfg_.num_layers; ++i) {
    const Ink2MtpLayer& L = mtp_layers_[size_t(i)];
    ink2::launch_rmsnorm(d_mtp_blk_x_, L.in_norm.d_ptr, L.in_norm.f32, d_mtp_blk_norm_, B, H, mtp_cfg_.rms_eps, stream_);
    const ink2::MultiLinearSeg qkv_segs[3] = {{d_mtp_blk_q_, QD}, {d_mtp_blk_k_, KVD}, {d_mtp_blk_v_, KVD}};
    ink2::launch_linear_multi(cublas_, d_mtp_blk_norm_, L.w_qkv, d_x_bf16_, d_mtp_qkv_tmp_, qkv_segs, 3, B, H, stream_);
    const int kv_cap = L.is_local ? ring_rows_ : max_ctx_;
    ink2::launch_mtp_qk_rope_cache(d_mtp_blk_q_, d_mtp_blk_k_, d_mtp_blk_v_,
                                   static_cast<const __nv_bfloat16*>(L.q_norm.d_ptr),
                                   static_cast<const __nv_bfloat16*>(L.k_norm.d_ptr),
                                   d_mtp_kc_[i], d_mtp_vc_[i], B, mtp_cfg_.hq, mtp_cfg_.hkv, mtp_cfg_.head_dim,
                                   pos0, d_pos_ptr, kv_cap, L.is_local, true, mtp_cfg_.rms_eps, mtp_cfg_.rope_theta,
                                   kv_bf16_, stream_);
    ink2::launch_mtp_attn(d_mtp_blk_q_, d_mtp_kc_[i], d_mtp_vc_[i], d_mtp_blk_k_, d_mtp_blk_v_, d_mtp_blk_att_,
                          B, pos0, d_pos_ptr, kv_cap, mtp_cfg_.hq, mtp_cfg_.hkv, mtp_cfg_.head_dim,
                          L.is_local, mtp_cfg_.swa, kv_bf16_, stream_);
    ink2::launch_linear(cublas_, d_mtp_blk_att_, L.o_proj.d_ptr, L.o_proj.f32, d_x_bf16_, d_mtp_blk_out_,
                        B, H, QD, false, stream_);
    ink2::launch_mtp_res_add_bf16(d_mtp_blk_x_, d_mtp_blk_out_, d_mtp_blk_res_, B * H, stream_);
    ink2::launch_rmsnorm(d_mtp_blk_res_, L.ff_norm.d_ptr, L.ff_norm.f32, d_mtp_blk_norm_, B, H, mtp_cfg_.rms_eps, stream_);
    ink2::launch_mtp_gate_up_swiglu(d_mtp_blk_norm_, L.w_gu, d_mtp_blk_gate_, B, mtp_cfg_.ffn, H, stream_);
    ink2::launch_linear(cublas_, d_mtp_blk_gate_, L.down_proj.d_ptr, L.down_proj.f32, d_x_bf16_, d_mtp_blk_out_,
                        B, H, mtp_cfg_.ffn, false, stream_);
    ink2::launch_mtp_res_add_bf16(d_mtp_blk_res_, d_mtp_blk_out_, d_mtp_blk_x_, B * H, stream_);
  }
  ink2::launch_rmsnorm(d_mtp_blk_x_, mtp_norm_.d_ptr, mtp_norm_.f32, d_mtp_blk_norm_, B, H, mtp_cfg_.rms_eps, stream_);
  if (d_draft_u_q8_) {
    ink2::launch_linear_q8_32(d_mtp_blk_norm_ + H, d_draft_u_q8_, d_draft_u_sc_, d_mtp_logits_, Kd, V, H, stream_);
  } else if (d_screen_q8_) {
    ink2::launch_screen_dot_multi(d_screen_q8_, d_screen_sc_, d_mtp_blk_norm_ + H, d_mtp_logits_, Kd, V, H, stream_);
  } else {
    ink2::launch_linear(cublas_, d_mtp_blk_norm_ + H, unembed_.d_ptr, unembed_.f32, d_x_bf16_, d_mtp_logits_,
                        Kd, V, H, false, stream_);
  }

  for (int i = 0; i < Kd; ++i) {
    ink2::launch_mtp_rnn_prep(d_mtp_state_, static_cast<const __nv_bfloat16*>(mtp_markov_w1_.d_ptr),
                              d_tokens_ + i, d_mtp_blk_norm_ + size_t(1 + i) * size_t(H), d_mtp_rec_feat_,
                              d_mtp_rnn_in_, R, H, i == 0, stream_);
    ink2::launch_linear_bf16_bias(d_mtp_rnn_in_, static_cast<const __nv_bfloat16*>(mtp_rnn_proj_w_.d_ptr),
                                  static_cast<const __nv_bfloat16*>(mtp_rnn_proj_b_.d_ptr), d_mtp_rnn_gates_,
                                  3 * R, 2 * R + 2 * H, stream_);
    ink2::launch_mtp_rnn_update(d_mtp_rnn_gates_, d_mtp_state_, d_mtp_latent_, R, i == 0, stream_);
    float* lrow = d_mtp_logits_ + size_t(i) * size_t(V);
    if (mtp_q8_) {
      ink2::launch_mtp_fold_add_q8(d_mtp_latent_, d_mtp_fold_q_, d_mtp_fold_s_, d_mtp_fold_b_, lrow,
                                   V, R, cfg_.logit_softcap, stream_);
    } else {
      ink2::launch_mtp_fold_add_bf16(d_mtp_latent_, d_mtp_fold_w_, d_mtp_fold_b_, lrow,
                                     V, R, cfg_.logit_softcap, stream_);
    }
    if (suppress_active_) ink2::launch_suppress_mask(lrow, V, d_suppress_, stream_);
    if (greedy) {
      ink2::launch_argmax_one(lrow, V, d_tokens_ + 1 + i, d_blk_v_, d_blk_i_, stream_);
    } else {
      float* h_drow = h_mtp_draft_logits_ + size_t(i) * size_t(V);
      cuda_ok(cudaMemcpyAsync(h_drow, lrow, size_t(V) * sizeof(float), cudaMemcpyDeviceToHost, stream_),
              "copy draft logits row");
      cuda_ok(cudaStreamSynchronize(stream_), "sync draft row");
      const SparseRowDist q = build_row_dist(h_drow, V, temp_, top_k_, top_p_, min_p_);
      const int32_t dtok = draw_from_dist(q, seed_ ^ kSaltDraft, step_pos + i);
      h_pair_toks_[1 + i] = dtok;
      cuda_ok(cudaMemcpyAsync(d_tokens_ + 1 + i, &h_pair_toks_[1 + i], sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
              "upload draft tok");
    }
  }
}

void EngineInk2::mtp_verify_step(int pos0, int R, const int32_t* d_pos_ptr, bool greedy) {
  if (greedy && use_screen_head_ && !use_q8_head_) {
    forward_tokens(nullptr, R, pos0, false, true, true, false, d_pos_ptr);
    ink2::launch_screen_argmax_bf16(
        static_cast<const __nv_bfloat16*>(unembed_.d_ptr), d_screen_q8_, d_screen_sc_, d_screen_eps_, d_h_norm_, R,
        cfg_.vocab, cfg_.hidden, suppress_active_ ? d_suppress_ : nullptr, d_screen_hnorm_, d_screen_max_lb_,
        d_screen_dot_hat_, d_screen_cand_ids_, d_screen_cand_cnt_, kScreenMaxCand, cfg_.logit_softcap,
        d_verify_samples_, stream_);
    return;
  }
  forward_tokens(nullptr, R, pos0, false, true, true, true, d_pos_ptr);
  if (greedy) {
    ink2::launch_argmax_rows(d_verify_logits_, R, cfg_.vocab, d_verify_samples_, d_blk_v_, d_blk_i_, stream_);
  }
}

int32_t EngineInk2::sample_logits_row(const float* h_row, int64_t step_pos) const {
  return draw_from_dist(build_row_dist(h_row, cfg_.vocab, temp_, top_k_, top_p_, min_p_), seed_, step_pos);
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
  return sample_logits_row(h_logits_, step_pos);
}

int32_t EngineInk2::prefill(const int32_t* ids, int n) {
  if (!ids || n <= 0) throw std::runtime_error("prefill requires >= 1 token");
  if (pos_ + n > max_ctx_) throw std::runtime_error("prompt exceeds max_ctx");
  int off = 0;
  while (off < n) {
    const int c = std::min(kPrefillChunk, n - off);
    const bool last = (off + c == n);
    forward_tokens(ids + off, c, pos_, last, mtp_enabled_, false, false);
    if (mtp_enabled_) commit_mtp_context(pos_, c, nullptr, false);
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
    forward_tokens(&tok, 1, pos_, true, mtp_enabled_, false, false);
    if (mtp_enabled_) commit_mtp_context(pos_, 1, nullptr, false);
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
                          const std::atomic<int>* cancel, int32_t* out_ids, Ink2GenStats& st) {
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
  st.cached = cached;

  const int rem = n_prompt - cached;
  int off = 0;
  int passes = 0;
  while (off < rem) {
    if (cancel && cancel->load(std::memory_order_relaxed)) break;
    const int c = std::min(kPrefillChunk, rem - off);
    const bool last = (off + c == rem);
    forward_tokens(prompt + cached + off, c, pos_, last, mtp_enabled_, false, false);
    if (mtp_enabled_) commit_mtp_context(pos_, c, nullptr, false);
    pos_ += c;
    off += c;
    ++passes;
  }
  live_tokens_.insert(live_tokens_.end(), prompt + cached, prompt + cached + off);

  st.produced = 0;
  st.eos = 0;
  st.drafted = 0;
  st.accepted = 0;
  if (off < rem || (cancel && cancel->load(std::memory_order_relaxed))) {
    const auto t1 = clk::now();
    st.prefill_ms = ms(t1 - t0).count();
    st.gen_ms = 0.0;
    st.passes = passes;
    return;
  }

  int32_t cur = sample_logits(pos_);
  last_tok_ = cur;
  __atomic_store_n(&out_ids[0], cur, __ATOMIC_RELEASE);
  const auto t1 = clk::now();
  st.prefill_ms = ms(t1 - t0).count();

  if (is_eos(cur) || n_gen <= 0) {
    st.eos = is_eos(cur) ? 1 : 0;
    st.gen_ms = 0.0;
    st.passes = passes;
    return;
  }

  // Adaptive draft depth K=1..7 tuned to MoE verify + draft relative cost curve.
  float ema_hits = 2.0f, ema_trials = 3.0f;
  int k_cur = 3;
  static constexpr float kCostRatio[8] = {1.00f, 1.33f, 1.44f, 1.55f, 1.70f, 1.76f, 1.91f, 2.04f};
  auto choose_k = [&](float hits, float trials) {
    const float alpha = std::clamp(hits / std::max(trials, 1e-3f), 0.05f, 0.96f);
    int best_k = 1;
    float best_tp = 0.0f;
    float ap = alpha;
    for (int k = 1; k < kMtpBlock; ++k) {
      ap *= alpha;
      const float exp_toks = (1.0f - ap) / std::max(1.0f - alpha, 1e-4f);
      const float tp = exp_toks / kCostRatio[k];
      if (tp > best_tp) {
        best_tp = tp;
        best_k = k;
      }
    }
    return best_k;
  };

  int pend_p0 = 0, pend_m = 0;
  bool pend_restore_conv = true;
  auto run_commit = [&](int p0, int m, bool restore_conv) {
    if (use_graph_ && m >= 1 && m <= kMtpBlock) {
      const int r_idx = restore_conv ? 1 : 0;
      const int32_t cp = int32_t(p0);
      cuda_ok(cudaMemcpyAsync(d_commit_pos_, &cp, sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "upload commit pos");
      if (!commit_graph_exec_[r_idx][m - 1]) {
        cuda_ok(cudaStreamSynchronize(stream_), "sync before commit graph");
        cuda_ok(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin commit graph");
        commit_mtp_context(0, m, d_commit_pos_, restore_conv);
        cuda_ok(cudaStreamEndCapture(stream_, &commit_graph_[r_idx][m - 1]), "end commit graph");
        cuda_ok(cudaGraphInstantiate(&commit_graph_exec_[r_idx][m - 1], commit_graph_[r_idx][m - 1], nullptr, nullptr, 0),
                "instantiate commit graph");
      }
      cuda_ok(cudaGraphLaunch(commit_graph_exec_[r_idx][m - 1], stream_), "launch commit graph");
    } else {
      commit_mtp_context(p0, m, nullptr, restore_conv);
    }
  };
  auto flush_pending = [&]() {
    if (pend_m <= 0) return;
    run_commit(pend_p0, pend_m, pend_restore_conv);
    cuda_ok(cudaStreamSynchronize(stream_), "sync commit flush");
    pend_m = 0;
  };

  int64_t step_ctr = pos_;
  while (st.produced < n_gen) {
    if (cancel && cancel->load(std::memory_order_relaxed)) break;
    const int rem_gen = int(n_gen - st.produced);
    const int room = (max_ctx_ - 2) - pos_;
    if (room < 0) break;
    const bool greedy = (temp_ <= 0.0f || top_k_ == 1);
    const bool can_mtp = (mtp_enabled_ && rem_gen >= 2 && room >= 1);

    if (!can_mtp) {
      if (pend_m > 0) {
        run_commit(pend_p0, pend_m, pend_restore_conv);
        pend_m = 0;
      }
      const bool use_scr = greedy && use_screen_head_;
      forward_tokens(&cur, 1, pos_, true, mtp_enabled_, false, false, nullptr, use_scr);
      int32_t nxt = 0;
      if (use_scr) {
        cuda_ok(cudaMemcpyAsync(h_tok_, d_tok_out_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_), "copy tok");
        cuda_ok(cudaStreamSynchronize(stream_), "sync screen argmax");
        nxt = *h_tok_;
        if (mtp_enabled_) {
          pend_p0 = pos_;
          pend_m = 1;
          pend_restore_conv = false;
        }
        ++pos_;
        live_tokens_.push_back(cur);
        ++passes;
      } else {
        if (mtp_enabled_) commit_mtp_context(pos_, 1, nullptr, false);
        ++pos_;
        live_tokens_.push_back(cur);
        ++passes;
        nxt = sample_logits(pos_);
      }
      last_tok_ = nxt;
      __atomic_store_n(&out_ids[1 + st.produced], nxt, __ATOMIC_RELEASE);
      ++st.produced;
      cur = nxt;
      if (is_eos(nxt)) {
        st.eos = 1;
        break;
      }
      continue;
    }

    if (pend_m > 0) {
      run_commit(pend_p0, pend_m, pend_restore_conv);
      pend_m = 0;
    }

    if (use_tree_ && greedy && tree_max_ >= 2 && room >= 1 && rem_gen >= 2) {
      static constexpr int kBuckets[kNumTreeBuckets] = {2, 3, 4, 5, 6, 8, 10, 12};
      TreeSpec tr{};
      const int depth_cap = std::min(room, rem_gen - 1);
      const int max_nodes = std::min(tree_max_, room + 1);
      if (use_suffix_draft_ && !is_eos(cur) && live_tokens_.size() >= 8) {
        const int L = int(live_tokens_.size());
        int best_p = -1;
        int best_len = 3;
        for (int p = L - 2; p >= 3; --p) {
          if (live_tokens_[size_t(p)] != cur) continue;
          int mlen = 1;
          const int kmax = std::min(16, p);
          for (int k = 1; k <= kmax; ++k) {
            if (live_tokens_[size_t(p - k)] != live_tokens_[size_t(L - k)]) break;
            ++mlen;
          }
          if (mlen > best_len) {
            best_len = mlen;
            best_p = p;
            if (best_len >= 17 && (L - 1 - p) >= kMaxTree - 1) break;
          }
        }
        if (best_p >= 0) {
          const int per = L - best_p;
          const int suf_max = (best_len >= 6) ? kMaxTree : tree_max_;
          const int chain_cap = std::min({suf_max, kMaxTree, room + 1, depth_cap + 1});
          const int max_j = (best_len >= per) ? (chain_cap - 1) : std::min(chain_cap - 1, per - 1);
          int32_t chain[kMaxTree];
          chain[0] = cur;
          int raw_m = 1;
          for (int j = 1; j <= max_j; ++j) {
            const int off = (best_len >= per) ? (j % per) : j;
            const int32_t nt = (off == 0) ? cur : live_tokens_[size_t(best_p + off)];
            if (nt < 0 || nt >= cfg_.vocab || is_eos(nt)) break;
            if (suppress_active_ && size_t(nt) < h_suppress_.size() && h_suppress_[size_t(nt)]) break;
            chain[raw_m++] = nt;
          }
          int snap_m = 0;
          for (int mb : kBuckets) {
            if (mb <= raw_m) snap_m = mb;
          }
          if (snap_m >= 2) {
            tr.M = snap_m;
            tr.max_depth = snap_m - 1;
            for (int nid = 0; nid < snap_m; ++nid) {
              tr.tok[nid] = chain[nid];
              const int8_t par = int8_t(nid - 1);
              tr.p1[nid] = par;
              tr.p2[nid] = (nid >= 2) ? int8_t(nid - 2) : int8_t(-1);
              tr.p3[nid] = (nid >= 3) ? int8_t(nid - 3) : int8_t(-1);
              tr.depth[nid] = uint8_t(nid);
              tr.anc_mask[nid] = (nid == 0) ? 1u : (tr.anc_mask[nid - 1] | (1u << nid));
            }
          }
        }
      }
      if (tr.M < 2) draft_beam_tree(cur, pos_, max_nodes, depth_cap, tr);
      if (tr.M >= 2) {
        int bidx = -1;
        for (int b = 0; b < kNumTreeBuckets; ++b) {
          if (kBuckets[b] == tr.M) {
            bidx = b;
            break;
          }
        }
        if (bidx >= 0) {
          cuda_ok(cudaMemcpyAsync(d_tokens_, tr.tok, size_t(tr.M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
                  "upload tree toks");
          cuda_ok(cudaMemcpyAsync(d_tree_p1_, tr.p1, size_t(tr.M) * sizeof(int8_t), cudaMemcpyHostToDevice, stream_),
                  "upload tree p1");
          cuda_ok(cudaMemcpyAsync(d_tree_p2_, tr.p2, size_t(tr.M) * sizeof(int8_t), cudaMemcpyHostToDevice, stream_),
                  "upload tree p2");
          cuda_ok(cudaMemcpyAsync(d_tree_p3_, tr.p3, size_t(tr.M) * sizeof(int8_t), cudaMemcpyHostToDevice, stream_),
                  "upload tree p3");
          cuda_ok(cudaMemcpyAsync(d_tree_depth_, tr.depth, size_t(tr.M) * sizeof(uint8_t), cudaMemcpyHostToDevice,
                                  stream_),
                  "upload tree depth");
          cuda_ok(cudaMemcpyAsync(d_tree_anc_mask_, tr.anc_mask, size_t(tr.M) * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream_),
                  "upload tree mask");

          const int sup_idx = suppress_active_ ? 1 : 0;
          if (use_graph_) {
            const int32_t sp = int32_t(pos_);
            cuda_ok(cudaMemcpyAsync(d_step_pos_, &sp, sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
                    "upload tree step pos");
            if (!tree_verify_graph_exec_[bidx][sup_idx]) {
              cuda_ok(cudaStreamSynchronize(stream_), "sync before tree verify graph");
              cuda_ok(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin tree verify graph");
              forward_tree_verify(tr.M, 0, d_step_pos_);
              cuda_ok(cudaStreamEndCapture(stream_, &tree_verify_graph_[bidx][sup_idx]), "end tree verify graph");
              cuda_ok(cudaGraphInstantiate(&tree_verify_graph_exec_[bidx][sup_idx], tree_verify_graph_[bidx][sup_idx],
                                           nullptr, nullptr, 0),
                      "instantiate tree verify graph");
            }
            cuda_ok(cudaGraphLaunch(tree_verify_graph_exec_[bidx][sup_idx], stream_), "launch tree verify graph");
          } else {
            forward_tree_verify(tr.M, pos_, nullptr);
          }

          cuda_ok(cudaMemcpyAsync(h_tree_next_, d_verify_samples_, size_t(tr.M) * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream_),
                  "copy tree next");
          cuda_ok(cudaStreamSynchronize(stream_), "sync tree verify");

          int32_t acc_path[kMaxTree];
          int32_t acc_toks[kMaxTree];
          int node = 0;
          int a = 0;
          acc_path[0] = 0;
          int32_t target = h_tree_next_[0];
          while (true) {
            acc_toks[a] = target;
            if (is_eos(target)) break;
            int match = -1;
            for (int c = node + 1; c < tr.M; ++c) {
              if (int(tr.p1[c]) == node && tr.tok[c] == target) {
                match = c;
                break;
              }
            }
            if (match < 0) break;
            ++a;
            acc_path[a] = match;
            node = match;
            target = h_tree_next_[match];
          }

          const int m = a + 1;
          const int step_p0 = pos_;
          int commit_len = m;
          int32_t next_cur = acc_toks[a];
          live_tokens_.push_back(cur);
          for (int j = 0; j < m; ++j) {
            const int32_t tok = acc_toks[j];
            __atomic_store_n(&out_ids[1 + st.produced], tok, __ATOMIC_RELEASE);
            ++st.produced;
            if (is_eos(tok)) {
              st.eos = 1;
              commit_len = j + 1;
              next_cur = tok;
              break;
            }
            if (j < a) live_tokens_.push_back(tok);
          }

          for (int j = 0; j < commit_len; ++j) h_pair_toks_[j] = acc_path[j];
          cuda_ok(cudaMemcpyAsync(d_acc_path_, h_pair_toks_, size_t(commit_len) * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream_),
                  "upload acc path");
          const int KVD_m = cfg_.hkv * cfg_.head_dim;
          ink2::launch_commit_sconv_all48(d_conv_base_, d_stage_sconv_, d_acc_path_, commit_len,
                                         int(cfg_.schedule.size()), kMaxTree, KVD_m, cfg_.hidden, stream_);
          ink2::launch_commit_kv_all48(d_kc_ptrs_, d_vc_ptrs_, d_kv_cap_, d_kv_is_swa_, d_stage_k_, d_stage_v_,
                                       d_acc_path_, step_p0, commit_len, int(cfg_.schedule.size()), kMaxTree,
                                       cfg_.hkv, cfg_.head_dim, kv_bf16_, stream_);
          ink2::launch_compact_taps(d_tap_[0], d_tap_[1], d_tap_[2], d_h_norm_, d_tap_scratch_, d_acc_path_,
                                    commit_len, cfg_.hidden, stream_);

          st.drafted += (tr.M - 1);
          st.accepted += a;
          ++passes;
          step_ctr += a + 1;

          pend_p0 = step_p0;
          pend_m = commit_len;
          pend_restore_conv = false;
          pos_ += commit_len;
          last_tok_ = next_cur;
          cur = next_cur;
          if (st.eos) break;
          continue;
        }
      }
    }

    const int Kd = std::min({k_cur, kMtpBlock - 1, rem_gen - 1, room});
    const int R = Kd + 1;
    const int V = cfg_.vocab;
    h_pair_toks_[0] = cur;
    cuda_ok(cudaMemcpyAsync(d_tokens_, &h_pair_toks_[0], sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
            "upload anchor tok");

    if (greedy && use_graph_) {
      const int32_t sp = int32_t(pos_);
      cuda_ok(cudaMemcpyAsync(d_step_pos_, &sp, sizeof(int32_t), cudaMemcpyHostToDevice, stream_), "upload step pos");
      const int gidx = Kd - 1;
      const int sup_idx = suppress_active_ ? 1 : 0;
      if (!dv_graph_exec_[gidx][sup_idx]) {
        cuda_ok(cudaStreamSynchronize(stream_), "sync before dv graph");
        cuda_ok(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin dv graph");
        mtp_draft_step(0, Kd, d_step_pos_, true, step_ctr);
        mtp_verify_step(0, R, d_step_pos_, true);
        cuda_ok(cudaStreamEndCapture(stream_, &dv_graph_[gidx][sup_idx]), "end dv graph");
        cuda_ok(cudaGraphInstantiate(&dv_graph_exec_[gidx][sup_idx], dv_graph_[gidx][sup_idx], nullptr, nullptr, 0),
                "instantiate dv graph");
      }
      cuda_ok(cudaGraphLaunch(dv_graph_exec_[gidx][sup_idx], stream_), "launch dv graph");
      cuda_ok(cudaMemcpyAsync(h_pair_toks_ + 1, d_tokens_ + 1, size_t(Kd) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                              stream_),
              "copy drafts");
      cuda_ok(cudaMemcpyAsync(h_pair_toks_ + kMtpBlock, d_verify_samples_, size_t(R) * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream_),
              "copy verify samples");
      cuda_ok(cudaStreamSynchronize(stream_), "sync dv graph");
    } else {
      mtp_draft_step(pos_, Kd, nullptr, greedy, step_ctr);
      mtp_verify_step(pos_, R, nullptr, greedy);
      if (greedy) {
        cuda_ok(cudaMemcpyAsync(h_pair_toks_ + 1, d_tokens_ + 1, size_t(Kd) * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream_),
                "copy drafts");
        cuda_ok(cudaMemcpyAsync(h_pair_toks_ + kMtpBlock, d_verify_samples_, size_t(R) * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream_),
                "copy verify samples");
      } else {
        cuda_ok(cudaMemcpyAsync(h_verify_logits_, d_verify_logits_, size_t(R) * size_t(V) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_),
                "copy verify logits");
      }
      cuda_ok(cudaStreamSynchronize(stream_), "sync verify");
    }

    int32_t accepted_out[kMtpBlock];
    const int a = ink2_accept_tokens(accepted_out, h_pair_toks_ + 1, Kd, V, h_verify_logits_, h_mtp_draft_logits_,
                                     h_pair_toks_ + kMtpBlock, temp_, top_k_, top_p_, min_p_, seed_, step_ctr);
    st.drafted += Kd;
    st.accepted += a;
    ++passes;
    step_ctr += a + 1;

    const int m = a + 1;
    int commit_len = m;
    int32_t next_cur = accepted_out[a];
    live_tokens_.push_back(cur);
    for (int j = 0; j < m; ++j) {
      const int32_t tok = accepted_out[j];
      __atomic_store_n(&out_ids[1 + st.produced], tok, __ATOMIC_RELEASE);
      ++st.produced;
      if (is_eos(tok)) {
        st.eos = 1;
        commit_len = j + 1;
        next_cur = tok;
        break;
      }
      if (j < a) live_tokens_.push_back(tok);
    }

    pend_p0 = pos_;
    pend_m = commit_len;
    pend_restore_conv = true;
    pos_ += commit_len;
    last_tok_ = next_cur;
    cur = next_cur;

    if (st.eos) break;
    ema_hits = ema_hits * 0.85f + float(a);
    ema_trials = ema_trials * 0.85f + float(std::min(Kd, a + 1));
    k_cur = choose_k(ema_hits, ema_trials);
  }
  flush_pending();
  const auto t2 = clk::now();
  st.gen_ms = ms(t2 - t1).count();
  st.passes = passes;
}

}  // namespace yskn
