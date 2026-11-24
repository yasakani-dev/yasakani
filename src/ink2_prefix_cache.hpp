// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yskn {

constexpr int kInk2MaxSnapshots = 6;
constexpr int32_t kInk2EndMessageId = 200010;  // <|end_message|> in Inkling-2-Mini tokenizer.json

inline void ink2_cache_cuda_ok(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

struct Ink2PinnedBuf {
  void* ptr = nullptr;
  size_t cap = 0;

  Ink2PinnedBuf() = default;
  ~Ink2PinnedBuf() { release(nullptr); }

  Ink2PinnedBuf(const Ink2PinnedBuf&) = delete;
  Ink2PinnedBuf& operator=(const Ink2PinnedBuf&) = delete;

  Ink2PinnedBuf(Ink2PinnedBuf&& o) noexcept : ptr(o.ptr), cap(o.cap) {
    o.ptr = nullptr;
    o.cap = 0;
  }

  Ink2PinnedBuf& operator=(Ink2PinnedBuf&& o) noexcept {
    if (this != &o) {
      release(nullptr);
      ptr = o.ptr;
      cap = o.cap;
      o.ptr = nullptr;
      o.cap = 0;
    }
    return *this;
  }

  void ensure(size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return;
    if (ptr && bytes <= cap) return;
    if (ptr) {
      if (stream) ink2_cache_cuda_ok(cudaStreamSynchronize(stream), "sync before pinned realloc");
      cudaFreeHost(ptr);
      ptr = nullptr;
      cap = 0;
    }
    ink2_cache_cuda_ok(cudaMallocHost(&ptr, bytes), "cudaMallocHost Ink2PinnedBuf");
    cap = bytes;
  }

  void release(cudaStream_t stream = nullptr) {
    if (!ptr) return;
    if (stream) cudaStreamSynchronize(stream);
    cudaFreeHost(ptr);
    ptr = nullptr;
    cap = 0;
  }
};

enum class Ink2SnapKind : int {
  kPromptEnd = 0,
  kEndMessage = 1,
};

struct Ink2Snapshot {
  bool valid = false;
  int pos = 0;
  Ink2SnapKind kind = Ink2SnapKind::kPromptEnd;
  bool gpu_kv_live = false;
  bool swa_backed_up = false;
  int swa_base_rows = 0;
  int swa_mtp_rows = 0;
  uint64_t lru_seq = 0;
  uint32_t hit_count = 0;
  bool has_visual = false;
  std::vector<int32_t> tokens;
  std::vector<uint64_t> vis_hashes;
  Ink2PinnedBuf h_conv;
  Ink2PinnedBuf h_mtp_rec_feat;
  Ink2PinnedBuf h_swa_kv;
};

class Ink2PrefixCache {
 public:
  void init(size_t conv_bytes, size_t mtp_rec_bytes, bool mtp_enabled, int base_swa, int mtp_swa, int max_swa,
            int ring_rows, size_t base_kv_row_bytes, size_t mtp_kv_row_bytes, std::vector<int> base_local_visits,
            std::vector<int> mtp_local_layers, bool debug_log) {
    conv_bytes_ = conv_bytes;
    mtp_rec_bytes_ = mtp_rec_bytes;
    mtp_enabled_ = mtp_enabled;
    base_swa_ = base_swa;
    mtp_swa_ = mtp_swa;
    max_swa_ = max_swa;
    ring_rows_ = ring_rows;
    base_kv_row_bytes_ = base_kv_row_bytes;
    mtp_kv_row_bytes_ = mtp_kv_row_bytes;
    base_local_visits_ = std::move(base_local_visits);
    mtp_local_layers_ = std::move(mtp_local_layers);
    debug_log_ = debug_log;
    clock_ = 0;
    for (int i = 0; i < kInk2MaxSnapshots; ++i) invalidate_slot(slots_[i], false);
  }

  void release(cudaStream_t stream) {
    if (stream) cudaStreamSynchronize(stream);
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      invalidate_slot(slots_[i], false);
      slots_[i].h_conv.release(nullptr);
      slots_[i].h_mtp_rec_feat.release(nullptr);
      slots_[i].h_swa_kv.release(nullptr);
    }
  }

  void clear(cudaStream_t stream) {
    if (stream) ink2_cache_cuda_ok(cudaStreamSynchronize(stream), "sync prefix_cache clear");
    clock_ = 0;
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      invalidate_slot(slots_[i], false);
      slots_[i].h_swa_kv.release(nullptr);
    }
  }

  void invalidate_all() {
    for (int i = 0; i < kInk2MaxSnapshots; ++i) invalidate_slot(slots_[i], false);
  }

  void invalidate_after(int rewind_pos) {
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      if (slots_[i].valid && slots_[i].pos > rewind_pos) {
        invalidate_slot(slots_[i], false);
      }
    }
  }

  int valid_count() const {
    int n = 0;
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      if (slots_[i].valid) ++n;
    }
    return n;
  }

  int latest_pos() const {
    int best = 0;
    uint64_t best_seq = 0;
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      if (slots_[i].valid && slots_[i].lru_seq >= best_seq) {
        best_seq = slots_[i].lru_seq;
        best = slots_[i].pos;
      }
    }
    return best;
  }

  int find_best_match(const int32_t* prompt, int n_prompt, const std::vector<uint64_t>& active_vis_hash,
                      bool vis_active) const {
    if (!prompt || n_prompt <= 1) return -1;
    int best_idx = -1;
    int best_pos = 0;
    uint64_t best_seq = 0;
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      const Ink2Snapshot& s = slots_[i];
      if (!s.valid || s.pos <= 0 || s.pos >= n_prompt) continue;
      if (s.pos < best_pos) continue;
      if (s.pos == best_pos && s.lru_seq <= best_seq) continue;
      if (!std::equal(s.tokens.begin(), s.tokens.end(), prompt)) continue;
      bool vis_ok = true;
      for (int p = 0; p < s.pos; ++p) {
        const uint64_t ah = (vis_active && p < int(active_vis_hash.size())) ? active_vis_hash[size_t(p)] : 0ULL;
        if (s.vis_hashes[size_t(p)] != ah) {
          vis_ok = false;
          break;
        }
      }
      if (!vis_ok) continue;
      best_idx = i;
      best_pos = s.pos;
      best_seq = s.lru_seq;
    }
    return best_idx;
  }

  bool capture(int pos, Ink2SnapKind kind, const std::vector<int32_t>& live_tokens,
               const std::vector<uint64_t>& live_vis_hash, const float* d_conv_base, const float* d_mtp_rec_feat,
               int max_written_pos, cudaStream_t stream) {
    if (pos <= 0 || pos > int(live_tokens.size()) || pos > int(live_vis_hash.size())) return false;

    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      Ink2Snapshot& s = slots_[i];
      if (!s.valid || s.pos != pos) continue;
      if (std::equal(s.tokens.begin(), s.tokens.end(), live_tokens.begin()) &&
          std::equal(s.vis_hashes.begin(), s.vis_hashes.end(), live_vis_hash.begin())) {
        s.lru_seq = ++clock_;
        if (kind == Ink2SnapKind::kPromptEnd) s.kind = Ink2SnapKind::kPromptEnd;
        return false;
      }
    }

    const int slot_idx = pick_slot_for_insert(live_tokens, live_vis_hash, pos);
    Ink2Snapshot& s = slots_[slot_idx];
    s.h_conv.ensure(conv_bytes_, stream);
    if (mtp_enabled_ && d_mtp_rec_feat && mtp_rec_bytes_ > 0) {
      s.h_mtp_rec_feat.ensure(mtp_rec_bytes_, stream);
    }

    ink2_cache_cuda_ok(
        cudaMemcpyAsync(s.h_conv.ptr, d_conv_base, conv_bytes_, cudaMemcpyDeviceToHost, stream),
        "capture snapshot conv d2h");
    if (mtp_enabled_ && d_mtp_rec_feat && mtp_rec_bytes_ > 0) {
      ink2_cache_cuda_ok(
          cudaMemcpyAsync(s.h_mtp_rec_feat.ptr, d_mtp_rec_feat, mtp_rec_bytes_, cudaMemcpyDeviceToHost, stream),
          "capture snapshot mtp_rec d2h");
    }

    s.valid = true;
    s.pos = pos;
    s.kind = kind;
    s.gpu_kv_live = (max_written_pos - pos <= ring_rows_ - max_swa_);
    s.swa_backed_up = false;
    s.swa_base_rows = 0;
    s.swa_mtp_rows = 0;
    s.lru_seq = ++clock_;
    s.hit_count = 0;
    s.tokens.assign(live_tokens.begin(), live_tokens.begin() + pos);
    s.vis_hashes.assign(live_vis_hash.begin(), live_vis_hash.begin() + pos);
    s.has_visual = false;
    for (uint64_t h : s.vis_hashes) {
      if (h != 0ULL) {
        s.has_visual = true;
        break;
      }
    }

    if (debug_log_) {
      std::fprintf(stderr, "[ink2_snap] capture slot=%d pos=%d kind=%d vis=%d gpu_live=%d\n", slot_idx, s.pos,
                   int(s.kind), s.has_visual ? 1 : 0, s.gpu_kv_live ? 1 : 0);
    }
    return true;
  }

  void backup_before_write(int new_max_written_pos, const std::vector<void*>& d_kc, const std::vector<void*>& d_vc,
                           void* const d_mtp_kc[3], void* const d_mtp_vc[3], cudaStream_t stream) {
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      Ink2Snapshot& s = slots_[i];
      if (!s.valid || !s.gpu_kv_live) continue;
      if (new_max_written_pos - s.pos <= ring_rows_ - max_swa_) continue;

      if (!s.swa_backed_up) {
        const int n_base = std::min(s.pos, base_swa_);
        const int n_mtp = mtp_enabled_ ? std::min(s.pos, mtp_swa_) : 0;
        const size_t base_slice_bytes = size_t(n_base) * base_kv_row_bytes_;
        const size_t mtp_slice_bytes = size_t(n_mtp) * mtp_kv_row_bytes_;
        const size_t need = base_local_visits_.size() * 2 * base_slice_bytes +
                            mtp_local_layers_.size() * 2 * mtp_slice_bytes;
        if (need > 0) {
          s.h_swa_kv.ensure(need, stream);
          uint8_t* h_dst = static_cast<uint8_t*>(s.h_swa_kv.ptr);
          const int base_start = s.pos - n_base;
          for (int v : base_local_visits_) {
            copy_ring_slice(h_dst, d_kc[size_t(v)], base_start, n_base, ring_rows_, base_kv_row_bytes_,
                            cudaMemcpyDeviceToHost, stream);
            h_dst += base_slice_bytes;
            copy_ring_slice(h_dst, d_vc[size_t(v)], base_start, n_base, ring_rows_, base_kv_row_bytes_,
                            cudaMemcpyDeviceToHost, stream);
            h_dst += base_slice_bytes;
          }
          if (mtp_enabled_ && n_mtp > 0) {
            const int mtp_start = s.pos - n_mtp;
            for (int l : mtp_local_layers_) {
              copy_ring_slice(h_dst, d_mtp_kc[l], mtp_start, n_mtp, ring_rows_, mtp_kv_row_bytes_,
                              cudaMemcpyDeviceToHost, stream);
              h_dst += mtp_slice_bytes;
              copy_ring_slice(h_dst, d_mtp_vc[l], mtp_start, n_mtp, ring_rows_, mtp_kv_row_bytes_,
                              cudaMemcpyDeviceToHost, stream);
              h_dst += mtp_slice_bytes;
            }
          }
        }
        s.swa_backed_up = true;
        s.swa_base_rows = n_base;
        s.swa_mtp_rows = n_mtp;
        if (debug_log_) {
          std::fprintf(stderr, "[ink2_snap] backup pos=%d base_rows=%d mtp_rows=%d bytes=%zu trigger_write_end=%d\n",
                       s.pos, n_base, n_mtp, need, new_max_written_pos);
        }
      }
      s.gpu_kv_live = false;
    }
  }

  bool restore(int slot_idx, float* d_conv_base, float* d_mtp_rec_feat, const std::vector<void*>& d_kc,
               const std::vector<void*>& d_vc, void* const d_mtp_kc[3], void* const d_mtp_vc[3],
               int& max_written_pos, std::vector<int32_t>& live_tokens, std::vector<uint64_t>& live_vis_hash,
               cudaStream_t stream) {
    if (slot_idx < 0 || slot_idx >= kInk2MaxSnapshots || !slots_[slot_idx].valid) {
      throw std::runtime_error("ink2 prefix cache: invalid snapshot slot");
    }
    Ink2Snapshot& s = slots_[slot_idx];
    const int target_pos = s.pos;

    ink2_cache_cuda_ok(
        cudaMemcpyAsync(d_conv_base, s.h_conv.ptr, conv_bytes_, cudaMemcpyHostToDevice, stream),
        "restore snapshot conv h2d");
    if (mtp_enabled_ && d_mtp_rec_feat && mtp_rec_bytes_ > 0) {
      ink2_cache_cuda_ok(
          cudaMemcpyAsync(d_mtp_rec_feat, s.h_mtp_rec_feat.ptr, mtp_rec_bytes_, cudaMemcpyHostToDevice, stream),
          "restore snapshot mtp_rec h2d");
    }

    const bool zero_copy = s.gpu_kv_live && (max_written_pos - target_pos <= ring_rows_ - max_swa_);
    bool restored_swa = false;
    invalidate_after(target_pos);

    if (!zero_copy) {
      if (!s.swa_backed_up) {
        throw std::runtime_error("ink2 prefix cache: missing SWA backup for snapshot restore");
      }
      const int n_base = s.swa_base_rows;
      const int n_mtp = s.swa_mtp_rows;
      const size_t base_slice_bytes = size_t(n_base) * base_kv_row_bytes_;
      const size_t mtp_slice_bytes = size_t(n_mtp) * mtp_kv_row_bytes_;
      const uint8_t* h_src = static_cast<const uint8_t*>(s.h_swa_kv.ptr);
      if (n_base > 0) {
        const int base_start = target_pos - n_base;
        for (int v : base_local_visits_) {
          copy_ring_slice(const_cast<uint8_t*>(h_src), d_kc[size_t(v)], base_start, n_base, ring_rows_,
                          base_kv_row_bytes_, cudaMemcpyHostToDevice, stream);
          h_src += base_slice_bytes;
          copy_ring_slice(const_cast<uint8_t*>(h_src), d_vc[size_t(v)], base_start, n_base, ring_rows_,
                          base_kv_row_bytes_, cudaMemcpyHostToDevice, stream);
          h_src += base_slice_bytes;
        }
      }
      if (mtp_enabled_ && n_mtp > 0) {
        const int mtp_start = target_pos - n_mtp;
        for (int l : mtp_local_layers_) {
          copy_ring_slice(const_cast<uint8_t*>(h_src), d_mtp_kc[l], mtp_start, n_mtp, ring_rows_, mtp_kv_row_bytes_,
                          cudaMemcpyHostToDevice, stream);
          h_src += mtp_slice_bytes;
          copy_ring_slice(const_cast<uint8_t*>(h_src), d_mtp_vc[l], mtp_start, n_mtp, ring_rows_, mtp_kv_row_bytes_,
                          cudaMemcpyHostToDevice, stream);
          h_src += mtp_slice_bytes;
        }
      }
      s.gpu_kv_live = true;
      max_written_pos = target_pos;
      restored_swa = true;
    }

    s.lru_seq = ++clock_;
    ++s.hit_count;
    live_tokens = s.tokens;
    live_vis_hash = s.vis_hashes;

    if (debug_log_) {
      std::fprintf(stderr, "[ink2_snap] restore slot=%d pos=%d mode=%s base_rows=%d\n", slot_idx, s.pos,
                   restored_swa ? "swa_host_restore" : "zero_kv_copy",
                   restored_swa ? s.swa_base_rows : 0);
    }
    return restored_swa;
  }

 private:
  static void invalidate_slot(Ink2Snapshot& s, bool release_swa_buf) {
    s.valid = false;
    s.pos = 0;
    s.kind = Ink2SnapKind::kPromptEnd;
    s.gpu_kv_live = false;
    s.swa_backed_up = false;
    s.swa_base_rows = 0;
    s.swa_mtp_rows = 0;
    s.lru_seq = 0;
    s.hit_count = 0;
    s.has_visual = false;
    s.tokens.clear();
    s.vis_hashes.clear();
    if (release_swa_buf) s.h_swa_kv.release(nullptr);
  }

  static void copy_ring_slice(void* host_ptr, void* dev_base, int start_pos, int n_rows, int ring_rows,
                              size_t row_bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    if (n_rows <= 0) return;
    const int slot0 = start_pos % ring_rows;
    const int first = std::min(n_rows, ring_rows - slot0);
    const int second = n_rows - first;
    uint8_t* h = static_cast<uint8_t*>(host_ptr);
    uint8_t* d = static_cast<uint8_t*>(dev_base);
    if (kind == cudaMemcpyDeviceToHost) {
      ink2_cache_cuda_ok(
          cudaMemcpyAsync(h, d + size_t(slot0) * row_bytes, size_t(first) * row_bytes, kind, stream),
          "copy_ring_slice d2h 1");
      if (second > 0) {
        ink2_cache_cuda_ok(
            cudaMemcpyAsync(h + size_t(first) * row_bytes, d, size_t(second) * row_bytes, kind, stream),
            "copy_ring_slice d2h 2");
      }
    } else {
      ink2_cache_cuda_ok(
          cudaMemcpyAsync(d + size_t(slot0) * row_bytes, h, size_t(first) * row_bytes, kind, stream),
          "copy_ring_slice h2d 1");
      if (second > 0) {
        ink2_cache_cuda_ok(
            cudaMemcpyAsync(d, h + size_t(first) * row_bytes, size_t(second) * row_bytes, kind, stream),
            "copy_ring_slice h2d 2");
      }
    }
  }

  static bool is_prefix_of(const Ink2Snapshot& s, const std::vector<int32_t>& live_tokens,
                           const std::vector<uint64_t>& live_vis_hash, int pos) {
    if (!s.valid || s.pos <= 0 || s.pos > pos) return false;
    return std::equal(s.tokens.begin(), s.tokens.end(), live_tokens.begin()) &&
           std::equal(s.vis_hashes.begin(), s.vis_hashes.end(), live_vis_hash.begin());
  }

  int pick_slot_for_insert(const std::vector<int32_t>& live_tokens, const std::vector<uint64_t>& live_vis_hash,
                           int pos) const {
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      if (!slots_[i].valid) return i;
    }

    // Protect one shared Turn-1 root prefix (preferring visual or reused kPromptEnd prefix, else shallowest root).
    int protected_idx = -1;
    int best_root_pos = 0x7fffffff;
    bool best_root_pref = false;
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      const Ink2Snapshot& s = slots_[i];
      if (s.kind != Ink2SnapKind::kPromptEnd) continue;
      if (!is_prefix_of(s, live_tokens, live_vis_hash, pos)) continue;
      const bool pref = s.has_visual || (s.hit_count > 0);
      if (protected_idx < 0 || (pref && !best_root_pref) || (pref == best_root_pref && s.pos < best_root_pos)) {
        protected_idx = i;
        best_root_pref = pref;
        best_root_pos = s.pos;
      }
    }

    int victim = -1;
    int best_tier = 99;
    uint64_t oldest_seq = UINT64_MAX;
    for (int i = 0; i < kInk2MaxSnapshots; ++i) {
      if (i == protected_idx) continue;
      const Ink2Snapshot& s = slots_[i];
      int tier = 2;
      if (s.kind == Ink2SnapKind::kEndMessage) {
        tier = (!s.gpu_kv_live && s.hit_count == 0) ? 0 : 1;
      }
      if (victim < 0 || tier < best_tier || (tier == best_tier && s.lru_seq < oldest_seq)) {
        victim = i;
        best_tier = tier;
        oldest_seq = s.lru_seq;
      }
    }
    return victim >= 0 ? victim : 0;
  }

  Ink2Snapshot slots_[kInk2MaxSnapshots];
  uint64_t clock_ = 0;
  size_t conv_bytes_ = 0;
  size_t mtp_rec_bytes_ = 0;
  bool mtp_enabled_ = false;
  int base_swa_ = 1024;
  int mtp_swa_ = 1024;
  int max_swa_ = 1024;
  int ring_rows_ = 2048;
  size_t base_kv_row_bytes_ = 0;
  size_t mtp_kv_row_bytes_ = 0;
  std::vector<int> base_local_visits_;
  std::vector<int> mtp_local_layers_;
  bool debug_log_ = false;
};

}  // namespace yskn
