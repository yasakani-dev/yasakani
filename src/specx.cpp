// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include <cstdlib>
#include "specx.hpp"
#include "edit_copy.hpp"
#include "prefill_mode.hpp"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <atomic>
#include <unordered_set>
namespace yskn {


std::atomic<int> g_specx_cancel{0};
static SpecxStatic g_static;
const SpecxStatic& specx_static() { return g_static; }
void specx_load_static(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb"); if (!f) throw std::runtime_error("specx: cannot open static table " + path);
  char magic[4]; int64_t hdr[2];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "YD1S", 4) != 0 || std::fread(hdr, 8, 2, f) != 2) { std::fclose(f); throw std::runtime_error("specx: bad static table header"); }
  SpecxStatic s; s.uni.resize((size_t)hdr[0]);
  if (std::fread(s.uni.data(), 4, s.uni.size(), f) != s.uni.size()) { std::fclose(f); throw std::runtime_error("specx: static table truncated (uni)"); }
  s.bi.reserve((size_t)hdr[1] * 2);
  for (int64_t i = 0; i < hdr[1]; ++i) {
    unsigned char rec[12]; if (std::fread(rec, 1, 12, f) != 12) { std::fclose(f); throw std::runtime_error("specx: static table truncated (bi)"); }
    uint64_t k; int32_t v; std::memcpy(&k, rec, 8); std::memcpy(&v, rec + 8, 4); s.bi.emplace(k, v);
  }
  std::fclose(f); s.loaded = true; g_static = std::move(s);
}
template <class Eng> SpecxT<Eng>::SpecxT(Eng& e, const SpecxConfig& c, int vocab) : e_(e), c_(c), V_(vocab) {
  if (c_.gamma < 0 || c_.gamma > 15) throw std::runtime_error("specx: gamma must be 0..15");
  if (c_.d1_static && (!g_static.loaded || (int)g_static.uni.size() != V_)) throw std::runtime_error("specx: d1_static needs specx_load_static with the model's vocabulary");
  std::array<int32_t, 8> empty; empty.fill(-1); T_.assign(V_, empty);
  ng_.assign(c_.d0_nmax + 1, {});
}
template <class Eng> uint64_t SpecxT<Eng>::gram_hash(int end, int n) const {        
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t)n;
  for (int i = end - n + 1; i <= end; ++i) { h ^= (uint64_t)(uint32_t)S_[i] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); h *= 0xBF58476D1CE4E5B9ull; }
  return h;
}
template <class Eng> void SpecxT<Eng>::index_upto(int end) {
  for (; indexed_ < end; ++indexed_)
    for (int n = c_.d0_nmin; n <= c_.d0_nmax; ++n) if (indexed_ - n + 1 >= 0) ng_[n][gram_hash(indexed_, n)] = indexed_;
}
template <class Eng> int SpecxT<Eng>::d0_lookup(int F, std::vector<int32_t>& chain, int gmax) {
  chain.clear(); lk_src_ = -1; if (!c_.use_d0 || gmax <= 0) return 0;
  index_upto(F);                                         
  for (int n = c_.d0_nmax; n >= c_.d0_nmin; --n) {
    if (F - n + 1 < 0) continue;
    auto it = ng_[n].find(gram_hash(F, n)); if (it == ng_[n].end()) continue;
    const int p = it->second; bool same = true;           
    for (int k = 0; k < n && same; ++k) same = S_[p - k] == S_[F - k];
    if (!same) continue;
    for (int q = p + 1; q <= F && (int)chain.size() < gmax; ++q) chain.push_back(S_[q]);
    if (!chain.empty()) lk_src_ = p + 1;
    return n;
  }
  return 0;
}
// Copy lookup with the copy cursor (copy_index = 1): continue where the last copy left off when the frontier still follows it, else
// ask the 4-token chain index (nearest place after the point where the copy stopped, or the longest backward match).  Returns the
// match length capped at 8 (the scale of d0_lookup) and fills up to gmax tokens; lk_src_ = the S_ index of the first one.
template <class Eng> int SpecxT<Eng>::copy_lookup(int F, std::vector<int32_t>& chain, int gmax) {
  chain.clear(); lk_src_ = -1; if (!c_.use_d0 || gmax <= 0) return 0;
  int p = -1, len = 0;
  if (cur_ > 0 && cur_ <= F && S_[cur_ - 1] == S_[F]) {
    const int q = cur_ - 1; int l = 0; while (l < 64 && q - l >= 0 && S_[q - l] == S_[F - l]) ++l;
    if (l >= CopyIndex::N) { p = q; len = l; stats.cursor_hits++; }
  }
  if (p < 0) {
    ci_.index_upto(S_.data(), F);
    const int after = (last_src_ >= 0 && F - last_src_at_ <= 256) ? last_src_ : -1;
    p = ci_.find(S_.data(), F, after, 256, 64, &len);
  }
  if (p < 0 || len < CopyIndex::N) return 0;
  for (int q = p + 1; q <= F && (int)chain.size() < gmax; ++q) chain.push_back(S_[q]);
  if (chain.empty()) return 0;
  lk_src_ = p + 1;
  return std::min(len, 8);
}
template <class Eng> int32_t SpecxT<Eng>::next_tok(int32_t p2, int32_t prev, int32_t cur) const {
  if (c_.d1_trigram && p2 >= 0 && prev >= 0) { auto it = T3_.find(tkey(p2, prev, cur)); if (it != T3_.end()) return it->second; }
  if (c_.d1_bigram && prev >= 0) { auto it = T2_.find(bkey(prev, cur)); if (it != T2_.end()) return it->second; }
  if (T_[cur][0] >= 0) return T_[cur][0];
  if (c_.d1_static) {
    if (prev >= 0) { auto it = g_static.bi.find(bkey(prev, cur)); if (it != g_static.bi.end()) return it->second; }
    if (g_static.uni[cur] >= 0) return g_static.uni[cur];
  }
  return -1;
}
template <class Eng> void SpecxT<Eng>::d1_chain(std::vector<int32_t>& chain, int32_t p2, int32_t prev, int32_t cur, int gmax) {
  while ((int)chain.size() < gmax) { const int32_t t = next_tok(p2, prev, cur); if (t < 0) break; chain.push_back(t); p2 = prev; prev = cur; cur = t; }
}
template <class Eng> double SpecxT<Eng>::p_acc(int s, int j) const {   
  static const double prior[3] = {0.6, 0.3, 0.6};
  const double pr = (c_.prmode == 1 && j > 0) ? std::min(prior[s], 0.9 * p_acc(s, j - 1)) : prior[s];   
  return (acc_[s][j] + 4.0 * pr) / (reach_[s][j] + 4.0);
}
template <class Eng> int SpecxT<Eng>::best_len(int s, int lmax, double fixed_us, double per_tok_us, int rows0) const {
  double reach = 1.0, gain = 0.0, best = 0.0, cost = fixed_us; int bl = 0;   
  const double val = tok_val();
  for (int L = 1; L <= lmax && L <= 15; ++L) {
    reach *= p_acc(s, L - 1); gain += reach;
    cost += per_tok_us + row_us(rows0 + L);                       
    const double v = gain * val - cost;
    if (v > best) { best = v; bl = L; }
  }
  return bl;
}
template <class Eng> void SpecxT<Eng>::record(int32_t p2, int32_t prev, int32_t cur, const int32_t* top8, const float* val8) {
  std::copy(top8, top8 + 8, T_[cur].begin());
  if (c_.d1_bigram && prev >= 0) { T2_[bkey(prev, cur)] = top8[0]; if (val8) T2m_[bkey(prev, cur)] = val8[0] - val8[1]; }
  if (c_.d1_trigram && p2 >= 0 && prev >= 0) T3_[tkey(p2, prev, cur)] = top8[0];
}
template <class Eng> int SpecxT<Eng>::prefill(const int32_t* ids, int n) {
  // With snapshots (snap = 1): continue the live state when it is a prefix of the prompt (mode 1); otherwise restore the saved slot
  // with the largest position inside the common prefix with what the engine holds (mode 2); otherwise read from the start (mode 0).
  // Slots are taken right after the last chat start of the prompt and, for long prompts, on the absolute prefill grid near its end
  // and every snap_sparse ids.  The prompt is read in passes cut at absolute multiples of the prefill chunk, so a read resumed from a
  // slot on that grid runs the same passes as a fresh read.
  using clk = std::chrono::steady_clock;
  int start = 0, mode = 0, lcp = 0, saved = 0, nslots = 0; double ms_snap = 0;
  std::vector<int> cuts, extra_cuts;
  int chunk = 0;
  if (c_.frontier) {
    // The frontier form: the engine holds ids[0 .. n-2] (its live sequence, e.g. set by yskn_ms_unload) and ids[n-1] was emitted but
    // not read.  Nothing is read here; generate() starts with ids[n-1] as the frontier, as the next pass of a stream that was never
    // moved would.  A continued read (mode 1) that starts at n - 1; returns -1 (no token).
    if constexpr (std::is_same_v<Eng, Engine35>) {
      const std::vector<int32_t>& live = e_.snap_live();
      if (!c_.snap || e_.has_draft() || n < 2 || !e_.snap_live_ok() || (int)live.size() != n - 1 || !std::equal(live.begin(), live.end(), ids))
        throw std::runtime_error("specx: the frontier form needs snap = 1, no built-in draft head, and the engine holding exactly the prompt "
                                 "without its last token as its live sequence");
      // the last prompt token must be the frontier itself: the token the stream emitted last (kept with the live sequence).  Another
      // token would be checked by the first pass as if the stream had emitted it, and the ids would differ without any error.
      if (ids[n - 1] != e_.snap_front())
        throw std::runtime_error("specx: the frontier form: the last prompt token (" + std::to_string(ids[n - 1]) + ") is not the token the "
                                 "engine's stream emitted last (" + std::to_string(e_.snap_front()) + "); give that token as the frontier");
      S_.assign(ids, ids + n);
      T2_.clear(); T2m_.clear(); T3_.clear(); draft_done_ = n - 1;
      indexed_ = 0; for (auto& m : ng_) m.clear(); recent_.clear(); stats = SpecxStats{};
      stats.sh_mode = 1; stats.sh_start = n - 1; stats.sh_lcp = n - 1; stats.sh_slots = e_.snap_count();
      return -1;
    } else throw std::runtime_error("specx: the frontier form is for the qwen3_5 engine");
  }
  if constexpr (std::is_same_v<Eng, Engine35>) {
    if (c_.snap) {
      chunk = e_.prefill_chunk();
      const std::vector<int32_t>& live = e_.snap_live(); const bool live_ok = e_.snap_live_ok();
      if (live_ok && !live.empty() && (int)live.size() < n && std::equal(live.begin(), live.end(), ids)) { start = (int)live.size(); mode = 1; lcp = start; }
      else {
        const std::vector<int32_t>& ref = live_ok ? live : e_.snap_seq();
        const int m = std::min((int)ref.size(), n - 1);
        int L = 0; while (L < m && ref[(size_t)L] == ids[L]) ++L;
        lcp = L;
        e_.snap_trim(L);
        const int i = e_.snap_best(L);
        if (i >= 0) { const auto t0 = clk::now(); e_.snap_load_slot(i); ms_snap += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                      start = e_.snap_slot_pos(i); mode = 2; }
        else e_.reset();
      }
      e_.snap_set_seq(ids, n);
      int q = -1;
      if (c_.snap_tok >= 0) for (int p = n - 1; p >= start; --p) if (ids[p] == c_.snap_tok) { q = p + 1; break; }
      if (q < 0 && mode == 0) q = n - 1;                           // a continued read without a new chat start keeps the slots it has
      if (!(q > start && q < n)) q = -1;
      // keep the restored slot when there are others to use; with a single slot, overwrite it (as 0.1.3 did)
      const int cap = e_.snap_slots_max() > 1 ? e_.snap_slots_max() - (mode == 2 ? 1 : 0) : std::max(0, e_.snap_slots_max());
      std::vector<int> pick;
      if (q > 0 && cap > 0) pick.push_back(q);
      if (n >= e_.snap_min() && cap > 1) {
        const int st = e_.snap_step(), U = q > 0 ? q : n;
        int nt = 0;
        for (int c = ((U - 1) / st) * st; c > start && c >= n - e_.snap_tail() && nt < 4 && (int)pick.size() < cap; c -= st, ++nt) pick.push_back(c);
        if (e_.snap_sparse() > 0) {
          std::vector<int> sp; for (int c = e_.snap_sparse(); c < n; c += e_.snap_sparse()) if (c > start) sp.push_back(c);
          for (auto it = sp.rbegin(); it != sp.rend() && (int)pick.size() < cap; ++it) if (std::find(pick.begin(), pick.end(), *it) == pick.end()) pick.push_back(*it);
        }
      }
      std::sort(pick.begin(), pick.end()); pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
      cuts = pick;
      if (mode == 0) e_.snap_add_gap_cuts(n, cuts, extra_cuts);
    }
  }
  std::vector<uint8_t> keep;
  if constexpr (std::is_same_v<Eng, Engine35>) {
    if (c_.prefill_head && n - start >= 2048 && q35_prefill_v2() && !std::getenv("YSKN_Q35_PREFILL_DENSE_HEAD")) {
      keep.resize(n);
      std::vector<uint8_t> seen(V_);
      std::unordered_set<uint64_t> bigrams, trigrams;
      if (c_.d1_bigram) bigrams.reserve(n - start);
      if (c_.d1_trigram) trigrams.reserve(n - start);
      // record() overwrites each table entry. Its final value comes from
      // the last position with that key, even across prefill passes.
      for (int p = n - 1; p >= start; --p) {
        const int32_t cur = ids[p], prev = p > 0 ? ids[p - 1] : -1, p2 = p > 1 ? ids[p - 2] : -1;
        const bool valid = cur >= 0 && cur < V_;
        bool needed = !valid || !seen[cur];
        if (valid) seen[cur] = 1;
        if (c_.d1_bigram && prev >= 0 && bigrams.insert(bkey(prev, cur)).second) needed = true;
        if (c_.d1_trigram && p2 >= 0 && prev >= 0 && trigrams.insert(tkey(p2, prev, cur)).second) needed = true;
        keep[p] = needed;
      }
    }
  }
  std::vector<int32_t> am(n), ti((size_t)n * 8); std::vector<float> tv((size_t)n * 8);
  auto run = [&](int a, int b) -> int {
    if (!c_.prefill_head) return e_.prefill(ids + a, b - a);
    if constexpr (std::is_same_v<Eng, Engine35>) {
      if (q35_prefill_v2())
        return e_.prefill_prompt_rows(ids + a, b - a, chunk, am.data() + a, ti.data() + (size_t)a * 8, tv.data() + (size_t)a * 8,
                                      keep.empty() ? nullptr : keep.data() + a);
    }
    if (chunk <= 0) return e_.prefill_rows(ids + a, b - a, am.data() + a, ti.data() + (size_t)a * 8, tv.data() + (size_t)a * 8);
    int r = 0;
    for (int x = a; x < b;) {                                    // one pass per piece of the absolute grid
      const int y = std::min(b, (x / chunk + 1) * chunk);
      r = e_.prefill_rows(ids + x, y - x, am.data() + x, ti.data() + (size_t)x * 8, tv.data() + (size_t)x * 8); x = y;
    }
    return r;
  };
  int a = start;
  for (int c : cuts) {
    run(a, c); a = c;
    if constexpr (std::is_same_v<Eng, Engine35>) {
      const bool extra = std::find(extra_cuts.begin(), extra_cuts.end(), c) != extra_cuts.end();
      const auto t0 = clk::now(); saved += e_.snap_save_at(c, extra) ? 1 : 0; ms_snap += std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
  }
  const int first = run(a, n);
  if constexpr (std::is_same_v<Eng, Engine35>) nslots = e_.snap_count();
  S_.assign(ids, ids + n); S_.push_back(first);
  T2_.clear(); T2m_.clear(); T3_.clear(); draft_done_ = n;
  if (c_.prefill_head)
    for (int p = start; p < n; ++p) {
      if (keep.empty() || keep[p])
        record(p > 1 ? ids[p - 2] : -1, p > 0 ? ids[p - 1] : -1, ids[p], ti.data() + (size_t)p * 8, tv.data() + (size_t)p * 8);
    }
  indexed_ = 0; for (auto& m : ng_) m.clear(); recent_.clear(); stats = SpecxStats{};
  stats.sh_mode = mode; stats.sh_start = start; stats.sh_lcp = lcp; stats.sh_slots = nslots; stats.sh_saved = saved; stats.ms_snap = ms_snap;
  return first;
}
template <class Eng> void SpecxT<Eng>::generate(int n_tokens, int32_t* out) {
  constexpr int VR = Engine35::kVerifyMax;                         // rows of the widest pass (copy passes; others stay <= 16)
  std::vector<int32_t> chain, ids, am(VR), ti(VR * 8), dev_ids(VR); std::vector<int8_t> src; std::vector<float> tv(VR * 8);
  int produced = 0, gamma = c_.gamma, prev_a = -1;                 
  bool full_prev = false;                                          
  const bool d3 = c_.use_d3 && e_.has_draft() && !c_.d3_lazy;     
  const bool d3l = c_.use_d3 && e_.has_draft() && c_.d3_lazy;     
  
  
  
  bool fused = false;
  static const bool vr_on = [] { const char* e = std::getenv("YSKN_SPECX_VR"); return e && e[0] == '1'; }();
  
  
  
  static const int blk_mode = [] { const char* e = std::getenv("YSKN_BLK16"); return e && e[0] ? std::atoi(e) : 2; }();
  static const double b16_ratio = [] { const char* e = std::getenv("YSKN_B16_RATIO"); return e && e[0] ? std::atoi(e) / 100.0 : 0.22; }();
  static const int b16_probe = [] { const char* e = std::getenv("YSKN_B16_PROBE"); return e && e[0] ? std::atoi(e) : 256; }();
  static const double b16_decay = [] { const char* e = std::getenv("YSKN_B16_DECAY"); return e && e[0] ? std::atoi(e) / 100.0 : 0.9; }();
  double b16_ema = 2.0, b8_ema = 4.0; int b16_since = 0; long long b16_passes = 0, b16_all = 0;
  static const bool b16_backoff = [] { const char* e = std::getenv("YSKN_B16_BACKOFF"); return !(e && e[0] == '0'); }();   
  int b16_probe_cur = b16_probe;
  static const int b16_max = [] { const char* e = std::getenv("YSKN_B16_MAX"); return e && e[0] ? std::atoi(e) : 256; }();   
  static const double b16_full = [] { const char* e = std::getenv("YSKN_B16_FULL"); return e && e[0] ? std::atoi(e) / 100.0 : 0.45; }();   
  double full8_ema = 0.0;
  static const double b16_floor = [] { const char* e = std::getenv("YSKN_B16_FLOOR"); return e && e[0] ? std::atoi(e) / 100.0 : 0.6; }();   
  static const double b16_decay8 = [] { const char* e = std::getenv("YSKN_B16_DECAY8"); return e && e[0] ? std::atoi(e) / 100.0 : 0.97; }();   
  static const int b16_full_since = [] { const char* e = std::getenv("YSKN_B16_FULL_SINCE"); return e && e[0] ? std::atoi(e) : 4; }();   
  int b16_n = 0;                                            
  auto b16_observe = [&](int a_obs) {                              
    const double v = std::max(0, a_obs - 7);
    b16_ema = (b16_n++ == 0) ? v : b16_decay * b16_ema + (1.0 - b16_decay) * v; b16_since = 0;
    if (b16_backoff) b16_probe_cur = b16_ema >= std::max(b16_floor, b16_ratio * b8_ema) ? b16_probe : std::min(b16_max, b16_probe_cur * 2);
  };
  if constexpr (Eng::kVerifyRows) fused = vr_on && d3l && c_.use_graph;
  const bool want_vals = c_.d3_gate > 0;
  using clk = std::chrono::steady_clock; auto ms_since = [](clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); };
  if constexpr (std::is_same_v<Eng, Engine35>) e_.set_specx_scope(true);     
  struct ScopeOff { Eng& e; ~ScopeOff() { e.set_specx_scope(false); } } scope_off{e_};
  auto is_eos = [&](int32_t t) { return t >= 0 && (t == c_.eos1 || t == c_.eos2); };
  bool stop = false;
  if (!S_.empty() && is_eos(S_.back())) stop = true;             
  const bool sel = c_.sel && c_.cost_tok_us > 0, mauto = c_.mtp_auto && c_.cost_tok_us > 0;
  bool bdr = false;                                                
  if constexpr (std::is_same_v<Eng, Engine35>) {
    if (c_.bdraft) {
      if (!e_.has_bdraft()) throw std::runtime_error("specx: bdraft needs Engine35::load_bdraft");
      if (c_.use_d3) throw std::runtime_error("specx: bdraft replaces the MTP (use_d3 = 0)");
      bdr = true;
    }
  }
  dema_[0] = dema_[1] = 3.0; dpend_.clear(); dprobe_ = 0;
  bool wide_on = false;                                            // copy passes wider than 16 rows
  int wmask = 0;                                                   // 0.1.7: the widths that may be used (bit 0 = 32, 1 = 48, 2 = 64)
  if constexpr (std::is_same_v<Eng, Engine35>) {
    wmask = e_.wide_mask();
    wide_on = bdr && c_.copy_rows > 16 && e_.wide_ok() && !e_.has_draft() && wmask != 0;   // not with the built-in draft head loaded (16 rows)
    double* t[4] = {&t16_, &t32_, &t48_, &t64_};                    // this GPU's times of copy cycles when the caller measured them
    for (int i = 0; i < 4; ++i) if (e_.wide_t_init_us(i) > 0) *t[i] = e_.wide_t_init_us(i) / 1000.0;
  }
  stats.wide_mask = wide_on ? wmask : 0;
  if (c_.copy_index && bdr) ci_.reset((int)S_.size() + std::max(0, n_tokens) + 64);
  cur_ = -1; last_src_ = -1; last_src_at_ = -1; wide_need_ = std::max(0, c_.wide_need0); wide_run_ = 0; wst_tok_ = 0; wst_ms_ = 0; wide_k_ = 0; wpv_tok_ = 0; wpv_ms_ = 0; wpv_k_ = 0;
  bool xw32 = false;                                               // 32-row copy passes: only with the exact-sum wide GEMV (cheap)
  if constexpr (std::is_same_v<Eng, Engine35>) xw32 = e_.xw_rows() >= 32;
  // drafter context of the rows committed by the last pass (more than 16 after a wide pass)
  auto feed_ctx = [&](int& n_ctx, int pos_ctx) {
    if constexpr (std::is_same_v<Eng, Engine35>) { if (n_ctx > 16) { e_.bdraft_ctx(0, n_ctx, pos_ctx); n_ctx = 0; } }
    else { (void)n_ctx; (void)pos_ctx; } };
  b16_ema = 0.0; b8_ema = 4.0; b16_since = 0; b16_probe_cur = b16_probe; full8_ema = 0.0; b16_n = 0;   
  int dctx_n = 0, dctx_pos = 0;                                    
  for (auto& r : reach_) for (double& x : r) x = 0; for (auto& r : acc_) for (double& x : r) x = 0; for (int& x : skipped_) x = 0;
  const int probe_every = 16;                                  
  tok_us_ = c_.cost_tok_us; ema_us_ = 0; ema_tok_ = 0; ema_n_ = 0;   
  
  FILE* dump = nullptr;
  if constexpr (std::is_same_v<Eng, Engine35>) {
    static const char* dp = std::getenv("YSKN_SPECX_DUMP");
    if (dp && dp[0] && d3l) { dump = std::fopen(dp, "a"); if (dump) e_.set_opt(4, 1); }
  }
  struct DumpOff { Eng& e; FILE*& f; ~DumpOff() { if (f) { std::fclose(f); if constexpr (std::is_same_v<Eng, Engine35>) e.set_opt(4, 0); } } } dump_off{e_, dump};
  std::vector<int32_t> lg_ids(16 * 8); std::vector<float> lg_vals(16 * 8);
  
  bool wk_on = false;
  bool hb = false;                                                  
  if constexpr (std::is_same_v<Eng, Engine35>) { hb = bdr && e_.hcap_on(); if (hb) e_.set_opt(6, 1); }
  struct HbOff { Eng& e; bool on; ~HbOff() { if constexpr (std::is_same_v<Eng, Engine35>) { if (on) { e.set_opt(6, 0); e.hcap_pass(0); } } } } hb_off{e_, hb};
  if constexpr (std::is_same_v<Eng, Engine35>) { if (c_.lfrow > 0 && d3l) { wk_on = true; e_.set_opt(6, 1); e_.set_opt(4, 1); } }
  struct WkOff { Eng& e; bool on; FILE*& f; ~WkOff() { if constexpr (std::is_same_v<Eng, Engine35>) { if (on) { e.set_opt(6, 0); if (!f) e.set_opt(4, 0); } } } } wk_off{e_, wk_on, dump};
  for (double& x : wq_n_) x = 0; for (double& x : wq_h_) x = 0;
  const double wq0 = c_.wk_q_pm / 1000.0;
  int nleaf = 0, kch = 0, leaf_row[16] = {}, leaf_depth[16] = {}, leaf_parent[16] = {};
  bool edit_enabled = false;
  if constexpr (std::is_same_v<Eng, Engine35>)
    edit_enabled = bdr && c_.use_d0 && c_.copy_index && e_.edit_copy_enabled();
  EditCopy edits(static_cast<int>(S_.size()) - 1);
  std::vector<int32_t> edited;
  int edit_pause = 0;
  double edit_p = 0;      // share of edit passes whose first edit was taken (running mean over about 8)
  bool edit_last = false; // the last edit pass took its first edit
  const auto t_loop = clk::now();
  while (produced < n_tokens && !stop && !(c_.cancel && c_.cancel->load(std::memory_order_relaxed))) {
    dk_ = e_.position() / 1024.0;
    const auto t_iter = clk::now();
    const int F = (int)S_.size() - 1; const int32_t xF = S_[F], xP = F > 0 ? S_[F - 1] : -1, xQ = F > 1 ? S_[F - 2] : -1;
    if (c_.auto_gamma && recent_.size() == 64) {          
      double m = 0; for (int t : recent_) m += t; m /= 64.0;
      gamma = m < 1.05 ? 0 : (m < 1.15 ? std::min(1, c_.gamma) : c_.gamma);
    }
    
    const int room = std::min(n_tokens - produced - 1, e_.max_ctx() - e_.position() - 1);
    const int g_short = std::max(0, std::min(gamma, room)), g_all = std::max(0, std::min(std::max(gamma, c_.gamma_long), room));
    
    chain.clear(); src.clear(); bool on_device = false; nleaf = 0;
    std::vector<int32_t> c0; int n0 = 0; lk_src_ = -1;
    if (g_all > 0) { const auto t_ix = clk::now(); n0 = (c_.copy_index && bdr) ? copy_lookup(F, c0, g_all) : d0_lookup(F, c0, g_all); stats.ms_index += ms_since(t_ix); }
    const int exp_src = lk_src_;                                    // the copy source the frontier follows (checked after the pass)
    int edit_first = -1, edit_rows = 0;
    if (edit_enabled && g_all > 0) {
      edits.advance(S_, F + 1, n0 >= 6 ? exp_src : -1);
      if (edit_pause > 0) --edit_pause;
      else {
        const int cap = std::min(room, wide_on ? c_.copy_rows - 1 : std::min(15, c_.bdr_d0_len));
        edit_first = edits.propose(S_, F + 1, cap, edited);
        if (edit_first >= 0) {
          const int n = static_cast<int>(edited.size());
          // a wide edit pass only when it is expected to give more tokens per ms than 16 rows, with the measured pass times of this GPU
          // and the share of edit passes that took their edit (a miss gives the tokens before the edit either way); never right after a miss
          auto wide_ok = [&](int W, double tw) {
            const double e1 = edit_first + 1;
            return edit_last && (edit_p * W + (1.0 - edit_p) * e1) * t16_ >= (edit_p * 16.0 + (1.0 - edit_p) * std::min(16.0, e1)) * tw; };
          if (wide_on && (wmask & 4) && n >= 63 && wide_ok(64, t64_)) edit_rows = 64;    // only the widths opened on this GPU (their graphs exist)
          else if (wide_on && (wmask & 2) && n >= 47 && wide_ok(48, t48_)) edit_rows = 48;
          else if (wide_on && xw32 && (wmask & 1) && n >= 31 && wide_ok(32, t32_)) edit_rows = 32;
          else edit_rows = std::min({n, 15, c_.bdr_d0_len}) + 1;
          if (edit_first >= edit_rows - 1) edit_rows = 0;
        }
      }
    }
    int wideW = 0;                                                  // rows of this pass when it is a wide copy pass
    bool hb_pass = false; int hb_M = 0;
    
    const int g_eff0 = (c_.gamma_long > gamma && ((full_prev && n0 >= c_.long_n) || (c_.long_d0 > 0 && n0 >= c_.long_d0))) ? g_all : g_short;
    const int g_eff = bdr ? std::min(g_eff0, std::max(BDR_J, c_.bdr_d0_len)) : g_eff0;   
    if ((int)c0.size() > g_eff) c0.resize(g_eff);
    auto ctx3 = [&](int k) -> int32_t {                            
      const int nc = (int)chain.size(); if (k < nc) return chain[nc - 1 - k];
      const int q = F - (k - nc); return q >= 0 ? S_[q] : -1; };
    auto extend = [&]() {                                          
      int cap = g_eff;
      if (sel) {                                                   
        int l1 = best_len(1, g_eff - (int)chain.size(), 0.0, 0.0, (int)chain.size() + 1);
        if (l1 == 0 && ++skipped_[1] >= probe_every) { l1 = std::min(1, g_eff - (int)chain.size()); skipped_[1] = 0; }
        else if (l1 > 0) skipped_[1] = 0;
        cap = (int)chain.size() + l1;
      }
      while ((int)chain.size() < cap) { const int32_t t = next_tok(ctx3(2), ctx3(1), ctx3(0)); if (t < 0) break; chain.push_back(t); src.push_back(1); }
    };
    auto take_d0 = [&]() {                                         
      chain = c0; src.assign(chain.size(), 0);
      if (c_.extend && c_.use_d1 && !chain.empty() && (int)chain.size() < g_eff) extend();
    };
    bool lazy_dev = false;                                         
    bool bdr_dev = false, bdr_longm = false, bdr_d0 = false;       
    int kdfl = std::min(7, g_eff); bool b16_pass = false;          
    const int room16 = std::min(BDR_J, room);                      
    if (bdr && blk_mode == 1 && room16 > 7) { kdfl = room16; b16_pass = true; }
    else if (bdr && blk_mode == 2 && room16 > 7 && (b16_ema >= std::max(b16_floor, b16_ratio * b8_ema) || b16_since >= b16_probe_cur || (full8_ema >= b16_full && b16_since >= b16_full_since))) { kdfl = room16; b16_pass = true; }
    if constexpr (std::is_same_v<Eng, Engine35>) {
      if (bdr && g_eff > 0) {
        bdr_longm = n0 >= c_.bdr_long_n && !c0.empty();
        bdr_d0 = edit_rows > 0 || (bdr_longm && (c_.bdr_adapt == 2 || (c_.bdr_adapt == 1 && dema_[1] > dema_[0])));
        const bool probe = bdr_d0 && c_.bdr_adapt == 1 && c_.bdr_probe > 0 && ++dprobe_ % c_.bdr_probe == 0;
        const bool run = !bdr_d0 || probe;                         
        if (probe && blk_mode && room16 > 7) { kdfl = room16; b16_pass = true; }   
                                                                                  
        static const int omega_mode = [] { const char* e = std::getenv("YSKN_OMEGA"); return (e && e[0]) ? std::atoi(e) : 2; }();
        static const double omega_t12 = [] { const char* e = std::getenv("YSKN_OMEGA_T12"); return (e && e[0]) ? std::atof(e) : 3.35; }();
        static const double omega_t16 = [] { const char* e = std::getenv("YSKN_OMEGA_T16"); return (e && e[0]) ? std::atof(e) : 2.55; }();
        const bool grd = e_.greedy_on();
        hb_pass = hb && (omega_mode > 0 || !grd) && !bdr_d0 && !b16_pass && kdfl == 7 && e_.position() + 8 <= e_.max_ctx();
        if (hb_pass) {
          hb_M = 8;
          if (false && grd) {
            if (omega_mode >= 3 && b8_ema < omega_t16 && room >= 15 && e_.position() + 16 <= e_.max_ctx()) hb_M = 16;
            else if (omega_mode >= 2 && b8_ema < omega_t12 && room >= 11 && e_.position() + 12 <= e_.max_ctx()) hb_M = 12;
          }
        }
        if (hb) e_.hcap_pass(hb_pass ? hb_M : 0);
        const auto t_d = clk::now();
        if (run) e_.set_verify_tokens(&xF, 0, 1);
        feed_ctx(dctx_n, dctx_pos);
        e_.bdraft_step(dctx_n, dctx_pos, run ? (b16_pass ? 2 : 1) : 0, c_.use_graph); dctx_n = 0;   
        stats.ms_draft += ms_since(t_d);
        if (bdr_d0) {
          if (wide_on && wide_run_ >= wide_need_ && exp_src >= 0) {   // a long run of fully accepted copy passes: copy W - 1 at once
            const int avail = F + 1 - exp_src;
            const int lim = std::min({avail, room, e_.max_ctx() - e_.position() - 1, c_.copy_rows - 1});
            wideW = (lim >= 63 && c_.copy_rows >= 64 && (wmask & 4)) ? 64
                  : ((lim >= 47 && c_.copy_rows >= 48 && (wmask & 2)) ? 48 : ((lim >= 31 && c_.copy_rows >= 32 && xw32 && (wmask & 1)) ? 32 : 0));
          }
          if (edit_rows > 0) {
            wideW = edit_rows > 16 ? edit_rows : 0;
            chain.assign(edited.begin(), edited.begin() + (edit_rows - 1));
            src.assign(chain.size(), 0);
          }
          else if (wideW) { chain.assign(S_.begin() + exp_src, S_.begin() + exp_src + (wideW - 1)); src.assign(chain.size(), 0); }
          else { if ((int)c0.size() > c_.bdr_d0_len) c0.resize(c_.bdr_d0_len); take_d0(); }
          stats.bdr_d0++;
          if (probe) { int32_t dr[BDR_J]; e_.bdraft_out(dr); dpend_.push_back(BdrPend{F, 0, std::vector<int32_t>(dr, dr + kdfl)}); stats.bdr_probes++; }
        } else {
          const int k = kdfl; chain.assign(k, 0); src.assign(k, 2); on_device = true; bdr_dev = true; stats.bdr_rounds++;
          if (bdr_longm) dpend_.push_back(BdrPend{F, 1, std::vector<int32_t>(c0.begin(), c0.begin() + std::min<int>((int)c0.size(), c_.bdr_d0_len))});
        }
      }
    }
    if constexpr (std::is_same_v<Eng, Engine35>) {                
      if (bdr && dctx_n) { const auto t_x = clk::now(); feed_ctx(dctx_n, dctx_pos); e_.bdraft_step(dctx_n, dctx_pos, 0, c_.use_graph); dctx_n = 0; stats.ms_ctx += ms_since(t_x); } }
    const bool mtp_spot = !bdr && d3l && g_eff > 0 && n0 < c_.d0_priority_n;   
    if (mtp_spot) {
      bool call = true;
      if (c_.d3_gate > 0 && c_.d1_bigram && xP >= 0) {             
        (void)xQ;
        auto it = T2m_.find(bkey(xP, xF)); call = !(it != T2m_.end() && it->second >= (float)c_.d3_gate);
      }
      int steps_sel = c_.d3_steps > 0 ? std::min(c_.d3_steps, g_eff) : g_eff;
      if (call && (sel || mauto)) {                                
        const double fixed = mtp_us() * (1.0 + (F - draft_done_ + 1) / 32.0);   
        int k = best_len(2, steps_sel, fixed - mtp_us(), mtp_us(), 1);
        if (k == 0 && sel && ++skipped_[2] >= probe_every) { k = steps_sel; skipped_[2] = 0; }
        else if (k > 0) skipped_[2] = 0;
        if (!mauto && k > 0) k = steps_sel;                        
        if (!sel && k == 0) k = 1;                                  
        if (k == 0) call = false; else steps_sel = k;
      }
      if (call) {
        const auto t_d = clk::now();
        if constexpr (std::is_same_v<Eng, Engine35>) { if (dump || wk_on) e_.mtp_log_reset(); }
        e_.set_verify_tokens(&xF, 0, 1);
        e_.draft_catchup(draft_done_, F - draft_done_ + 1, S_.data() + draft_done_, c_.use_graph); draft_done_ = F + 1;
        if (c_.d3_extend) {
          const int32_t d1 = e_.get_verify_token(1); chain.assign(1, d1); src.assign(1, 2);
          if (c_.use_d1) extend();
          if (chain.size() > 1 && !fused) e_.set_verify_tokens(chain.data() + 1, 2, (int)chain.size() - 1);
        } else {
          const int steps = steps_sel;
          e_.draft_steps(steps, c_.use_graph); on_device = true; chain.assign(steps, 0); src.assign(steps, 2);
          if constexpr (std::is_same_v<Eng, Engine35>) {
            if (wk_on) {                                           
              const double val = tok_val();
              double reach = 1.0, gv[16]; int gd[16], nc = 0;
              for (int j = 1; j <= steps && j <= 15; ++j) {
                const double p = p_acc(2, j - 1), q = (wq_h_[j - 1] + 4.0 * wq0) / (wq_n_[j - 1] + 4.0);
                gv[nc] = reach * (1.0 - p) * q * val; gd[nc] = j; ++nc; reach *= p; }
              for (int x = 0; x < nc; ++x) for (int y = x + 1; y < nc; ++y) if (gv[y] > gv[x]) { std::swap(gv[x], gv[y]); std::swap(gd[x], gd[y]); }
              int rows = steps + 1;
              for (int x = 0; x < nc && nleaf < c_.lfrow && rows < 16 && e_.position() + rows + 1 <= e_.max_ctx(); ++x) {
                if (gv[x] <= row_us(rows + 1)) break;
                leaf_depth[nleaf] = gd[x]; leaf_parent[nleaf] = gd[x] - 1; leaf_row[nleaf] = rows; ++nleaf; ++rows; }
              if (nleaf) {                                         
                uint8_t tab[48] = {}; for (int t = 0; t <= steps; ++t) tab[t] = (uint8_t)t;
                int o = 16; for (int r = 0; r <= steps; ++r) { tab[o++] = (uint8_t)r;
                  for (int i = 0; i < nleaf; ++i) if (leaf_depth[i] == r + 1) tab[o++] = (uint8_t)(leaf_row[i] | 0x80); }
                for (int i = 0; i < nleaf; ++i) { tab[leaf_row[i]] = (uint8_t)leaf_depth[i]; tab[32 + leaf_row[i]] = 1; }
                e_.lfrow_table(tab); e_.lfrow_fill(steps, steps + 1 + nleaf);
              }
            }
          }
        }
        lazy_dev = true; stats.ms_draft += ms_since(t_d);
      }
    }
    if (lazy_dev || bdr) {                                         
    } else if (d3) {                                                      
      if (prev_a < 0) e_.set_verify_tokens(&xF, 0, 1);            
      const bool use_d0 = n0 >= c_.d0_priority_n;
      const int steps = use_d0 || g_eff == 0 || c_.d3_extend ? 1 : (c_.d3_steps > 0 ? std::min(c_.d3_steps, g_eff) : g_eff);
      e_.draft_round(prev_a < 0 ? 0 : 1, std::max(0, prev_a), steps, c_.use_graph);
      if (g_eff > 0 && use_d0) { take_d0(); e_.set_verify_tokens(chain.data(), 1, (int)chain.size()); }
      else if (g_eff > 0 && c_.d3_extend) {                        
        const int32_t d1 = e_.get_verify_token(1); chain.assign(1, d1); src.assign(1, 2);
        if (c_.use_d1) extend();
        if (chain.size() > 1) e_.set_verify_tokens(chain.data() + 1, 2, (int)chain.size() - 1);
      }
      else if (g_eff > 0) { on_device = true; chain.assign(steps, 0); src.assign(steps, 2); }
    } else if (g_eff > 0) {
      if (n0 >= c_.d0_priority_n) take_d0();
      else if (c_.use_d1 && next_tok(xQ, xP, xF) >= 0) extend();
      else if (n0 > 0) take_d0();
    }
    
    kch = (int)chain.size(); const int M = hb_pass ? hb_M : (kch + 1 + nleaf);
    const auto t_v = clk::now();
    bool hist_done = false;
    if constexpr (Eng::kVerifyRows) {
      if (fused && !on_device) {                                   
        ids.assign(1, xF); ids.insert(ids.end(), chain.begin(), chain.end());
        e_.verify_rows(M, ids.data(), M, c_.use_graph, am.data(), dev_ids.data(), ti.data(), want_vals ? tv.data() : nullptr, true);
        for (int j = 0; j < M; ++j) if (dev_ids[j] != ids[j]) throw std::runtime_error("specx: verify_rows rows differ");
        hist_done = true;
      }
    }
    if (hist_done) {
    } else if (d3 || lazy_dev || bdr_dev) {
      if constexpr (std::is_same_v<Eng, Engine35>) { if (hb_pass) e_.lfrow_device_table(); }
      e_.verify_device(M, c_.use_graph, am.data(), dev_ids.data(), ti.data(), want_vals ? tv.data() : nullptr);
      if constexpr (std::is_same_v<Eng, Engine35>) {
        if (hb_pass) {                                             
          int cf[16]; e_.hcap_cfg(cf); const int L = cf[0];
          if (L < 3 || L > 7 || cf[1] != (M - 1) - L) throw std::runtime_error("specx: bad HCAP cfg");
          chain.resize(L); src.resize(L); kch = L; nleaf = cf[1];
          for (int i = 0; i < nleaf; ++i) { leaf_row[i] = L + 1 + i; leaf_depth[i] = cf[2 + i] & 0xff; leaf_parent[i] = cf[2 + i] >> 8; }
        }
      }
      if (dev_ids[0] != xF) throw std::runtime_error("specx: verify row 0 is not the frontier token");
      if (on_device) for (int j = 1; j <= kch; ++j) chain[j - 1] = dev_ids[j];
      else for (int j = 1; j <= kch; ++j) if (chain[j - 1] != dev_ids[j]) throw std::runtime_error("specx: verify rows differ from the host chain");
      ids.assign(dev_ids.begin(), dev_ids.begin() + M);
    } else {
      ids.assign(1, xF); ids.insert(ids.end(), chain.begin(), chain.end());
      e_.verify(ids.data(), M, c_.use_graph, am.data(), ti.data(), want_vals ? tv.data() : nullptr);
    }
    stats.ms_verify += ms_since(t_v);
    int a = 0; while (a < (int)chain.size() && chain[a] == am[a]) ++a;
    if (blk_mode == 2 && bdr_dev) {                              
      b8_ema = b16_decay * b8_ema + (1.0 - b16_decay) * (std::min(a, 7) + 1);
      if (b16_pass) b16_observe(a);
      else { ++b16_since; full8_ema = 0.75 * full8_ema + 0.25 * (a >= 7 ? 1.0 : 0.0); b16_ema *= b16_decay8; }
    }
    if (bdr_dev) { ++b16_all; if (b16_pass) ++b16_passes; }
    const int a_all = a;                                          
    if (bdr_longm) { const int w = bdr_d0 ? 1 : 0; dema_[w] += (std::min(a_all, BDR_J) - dema_[w]) / 16.0; }   
    if constexpr (std::is_same_v<Eng, Engine35>) {
      if (dump && lazy_dev && on_device) {                         
        const int k = (int)chain.size(), n = e_.mtp_log_read(lg_ids.data(), lg_vals.data(), 16);
        std::fprintf(dump, "{\"F\":%d,\"k\":%d,\"a\":%d,\"n\":%d,\"chain\":[", F, k, a_all, n);
        for (int j = 0; j < k; ++j) std::fprintf(dump, "%s%d", j ? "," : "", chain[j]);
        std::fprintf(dump, "],\"am\":[");
        for (int j = 0; j < M; ++j) std::fprintf(dump, "%s%d", j ? "," : "", am[j]);
        std::fprintf(dump, "],\"ti\":[");
        for (int j = 0; j < n; ++j) { std::fprintf(dump, "%s[", j ? "," : ""); for (int t = 0; t < 8; ++t) std::fprintf(dump, "%s%d", t ? "," : "", lg_ids[j * 8 + t]); std::fprintf(dump, "]"); }
        std::fprintf(dump, "],\"tv\":[");
        for (int j = 0; j < n; ++j) { std::fprintf(dump, "%s[", j ? "," : ""); for (int t = 0; t < 8; ++t) std::fprintf(dump, "%s%.4f", t ? "," : "", lg_vals[j * 8 + t]); std::fprintf(dump, "]"); }
        if (want_vals) {                                           
          std::fprintf(dump, "],\"rt\":[");
          for (int j = 0; j < M; ++j) { std::fprintf(dump, "%s[", j ? "," : ""); for (int t = 0; t < 8; ++t) std::fprintf(dump, "%s%d", t ? "," : "", ti[j * 8 + t]); std::fprintf(dump, "]"); }
          std::fprintf(dump, "],\"rv\":[");
          for (int j = 0; j < M; ++j) { std::fprintf(dump, "%s[", j ? "," : ""); for (int t = 0; t < 8; ++t) std::fprintf(dump, "%s%.4f", t ? "," : "", tv[j * 8 + t]); std::fprintf(dump, "]"); }
        }
        std::fprintf(dump, "]}\n");
      }
    }
    if (sel || mauto || wk_on) {                                  
      int pos_in[3] = {0, 0, 0};
      for (int k = 0; k < (int)chain.size() && k <= a_all; ++k) {
        const int s = src[k], j = std::min(pos_in[s]++, 15);
        reach_[s][j] = 0.97 * reach_[s][j] + 1.0; acc_[s][j] = 0.97 * acc_[s][j] + (k < a_all ? 1.0 : 0.0);
      }
    }
    for (int j = 0; j < a_all; ++j) if (is_eos(chain[j])) { a = j; stop = true; break; }
    if (!stop && is_eos(am[a])) stop = true;
    if (edit_rows > 0 && bdr_d0) {
      ++stats.edit_passes;
      stats.edit_accepted += a;
      edit_p += ((a > edit_first ? 1.0 : 0.0) - edit_p) / 8.0; edit_last = a > edit_first;
      if (a > edit_first) { ++stats.edit_hits; edits.hit(); }
      else {
        edit_pause = 3;
        if (a == edit_first) edits.missed();                      // the target rejected the edit itself: count it against the rule
      }
    }
    int acc_rows[4] = {-1, -1, -1, -1}; int n_off_acc = 0;
    if (nleaf && !stop && a == a_all && a < kch) {
      bool any = false;
      for (int i = 0; i < nleaf; ++i) if (leaf_parent[i] == a) { any = true; if (acc_rows[0] < 0 && dev_ids[leaf_row[i]] == am[a]) acc_rows[0] = leaf_row[i]; }
      if (any) { wq_n_[a] = 0.97 * wq_n_[a] + 1.0; wq_h_[a] = 0.97 * wq_h_[a] + (acc_rows[0] >= 0 ? 1.0 : 0.0); stats.leaf_breaks++; }
      if (acc_rows[0] >= 0) {
        n_off_acc = 1;
        stats.leaf_hits++;
        int cur_r = acc_rows[0];
        if (is_eos(am[cur_r])) stop = true;
        while (!stop && n_off_acc < 4) {
          int nxt_r = -1;
          for (int i = 0; i < nleaf; ++i) {
            if (leaf_parent[i] == cur_r && dev_ids[leaf_row[i]] == am[cur_r]) { nxt_r = leaf_row[i]; break; }
          }
          if (nxt_r < 0) break;
          acc_rows[n_off_acc++] = nxt_r;
          stats.leaf_hits++;
          cur_r = nxt_r;
          if (is_eos(am[cur_r])) stop = true;
        }
      }
    }
    stats.leaves += nleaf;
    if constexpr (std::is_same_v<Eng, Engine35>) {
      for (int m = 0; m < n_off_acc; ++m) e_.lfrow_promote(acc_rows[m], a + 1 + m);
    }
    const int n_commit = a + n_off_acc;
    if (d3l && !hist_done) e_.hist_store(F, n_commit + 1);
    const float* tvp = want_vals ? tv.data() : nullptr;

    const auto t_r = clk::now();
    auto in_tok = [&](int j) -> int32_t { return j >= 0 ? ids[j] : (j == -1 ? xP : xQ); };
    if (c_.d1_rejected) for (int j = a + 1; j <= kch; ++j) record(in_tok(j - 2), in_tok(j - 1), ids[j], ti.data() + j * 8, tvp ? tvp + j * 8 : nullptr);
    for (int j = 0; j <= a; ++j) record(in_tok(j - 2), in_tok(j - 1), ids[j], ti.data() + j * 8, tvp ? tvp + j * 8 : nullptr);
    for (int m = 0; m < n_off_acc; ++m) {
      const int32_t t_m2 = m == 0 ? in_tok(a - 1) : (m == 1 ? in_tok(a) : dev_ids[acc_rows[m - 2]]);
      const int32_t t_m1 = m == 0 ? in_tok(a)     : dev_ids[acc_rows[m - 1]];
      const int r_cur = acc_rows[m];
      record(t_m2, t_m1, dev_ids[r_cur], ti.data() + r_cur * 8, tvp ? tvp + r_cur * 8 : nullptr);
    }
    stats.ms_record += ms_since(t_r);
    full_prev = !chain.empty() && a == (int)chain.size();

    const auto t_c = clk::now(); e_.commit(n_commit); stats.ms_commit += ms_since(t_c); prev_a = n_commit;
    if constexpr (std::is_same_v<Eng, Engine35>) {
      if (bdr) { dctx_n = n_commit + 1; dctx_pos = F; } }
    for (int j = 0; j < a; ++j) { S_.push_back(chain[j]); if (produced < n_tokens) out[produced] = chain[j]; ++produced; }
    if (n_off_acc == 0) {
      S_.push_back(am[a]); if (produced < n_tokens) out[produced] = am[a]; ++produced;
    } else {
      for (int m = 0; m < n_off_acc; ++m) {
        const int32_t tk_m = dev_ids[acc_rows[m]];
        S_.push_back(tk_m); if (produced < n_tokens) out[produced] = tk_m; ++produced;
      }
      const int32_t tk_last = am[acc_rows[n_off_acc - 1]];
      S_.push_back(tk_last); if (produced < n_tokens) out[produced] = tk_last; ++produced;
      if (is_eos(tk_last)) stop = true;
    }
    {                                                               // the copy cursor follows the source while the output does
      const int ne = (int)S_.size() - (F + 1);
      if (exp_src >= 0) {
        int k = 0; while (k < ne && exp_src + k < (int)S_.size() && S_[(size_t)(exp_src + k)] == S_[(size_t)(F + 1 + k)]) ++k;
        if (k == ne) cur_ = exp_src + ne;
        else { cur_ = -1; last_src_ = exp_src + k; last_src_at_ = F + 1 + k; }
      } else cur_ = -1;
      if (bdr_d0) {                                                 // copy passes: when to go wide, when to come back
        stats.copy_passes++; stats.copy_tok += a;
        const bool full = a == (int)chain.size();
        const double t_pass = ms_since(t_iter);
        if (wideW) {
          double& tw = wideW == 64 ? t64_ : (wideW == 48 ? t48_ : t32_);
          const double tp = std::min(t_pass, 2.0 * tw);              // a stall (e.g. of the OS) counts as at most twice the usual time
          if (!c_.det) tw += (tp - tw) / 8.0;      // det: the width that was measured does not steer the next one
          stats.wide_passes++; stats.wide_tok += a + 1; if (full) stats.wide_full++;
          // an edit pass (edit copy) is not part of a copy run: only its time counts (above), not the wide statistics or the run
          if (edit_rows == 0) { wst_tok_ += a + 1; wst_ms_ += tp; ++wide_k_; }   // 0.1.7: the measured time of each pass (was the running mean)
          if (edit_rows == 0 && (!full || wide_k_ >= 4)) {      // the run ended, or 4 more wide passes: were they faster than 16 rows?
            // 0.1.7: checked inside a run too, not only at its end.  A run that ends with fewer than 4 wide passes since the last check
            // is judged together with the window checked before it (so a partly accepted pass at the end of a copy is not judged alone)
            const bool merge = !full && wide_k_ < 4 && wpv_k_ > 0;
            const double tok = wst_tok_ + (merge ? wpv_tok_ : 0.0), ms = wst_ms_ + (merge ? wpv_ms_ : 0.0);
            const int nk = wide_k_ + (merge ? wpv_k_ : 0);
            // det: the same question without the clock - did these passes carry at least 16 tokens each (the rate of a 16-row pass)?
            const bool won = c_.det ? (tok >= 16.0 * nk) : (tok * std::max(1.0, t16_) >= 16.0 * ms);
            if (won) wide_need_ = std::max(0, c_.wide_need0); else { wide_need_ = std::min(16, wide_need_ * 2); stats.wide_backoffs++; }
            if (full && won) { wpv_tok_ = wst_tok_; wpv_ms_ = wst_ms_; wpv_k_ = wide_k_; } else { wpv_tok_ = 0; wpv_ms_ = 0; wpv_k_ = 0; }
            wst_tok_ = 0; wst_ms_ = 0; wide_k_ = 0;
            if (!full || !won) wide_run_ = 0;                       // back to 16 rows (a slower run stops at once)
          }
        } else {
          if (M == 16 && !c_.det) t16_ += (std::min(t_pass, 2.0 * t16_) - t16_) / 8.0;
          if (edit_rows > 0) {} else if (full && (int)chain.size() >= BDR_J) ++wide_run_; else wide_run_ = 0;   // edit passes leave the run
        }
      } else wide_run_ = 0;
      stats.max_rows = std::max<int64_t>(stats.max_rows, M);
    }
    for (size_t i = 0; i < dpend_.size();) {                      
      const BdrPend& p = dpend_[i]; int k = 0; bool done = false;
      for (; k < (int)p.ch.size(); ++k) { const int q = p.F + 1 + k; if (q >= (int)S_.size()) break; if (S_[q] != p.ch[k]) { done = true; break; } }
      if (k == (int)p.ch.size()) done = true;
      if (done) { dema_[p.which] += (k - dema_[p.which]) / 16.0;
                  if (blk_mode == 2 && p.which == 0 && (int)p.ch.size() > 7) { b8_ema = b16_decay * b8_ema + (1.0 - b16_decay) * (std::min(k, 7) + 1); b16_observe(k); }   
                  dpend_.erase(dpend_.begin() + (std::ptrdiff_t)i); } else ++i;
    }
    if (c_.mark_tok >= 0 && stats.mark_at < 0)                    
      for (int64_t q = std::max<int64_t>(0, produced - (a + 1 + n_off_acc)); q < std::min<int64_t>(produced, n_tokens); ++q)
        if (out[q] == c_.mark_tok) { stats.mark_at = q + 1; stats.ms_mark = ms_since(t_loop); break; }   
    
    const int emitted = a + 1 + n_off_acc;
    stats.passes++; stats.tokens += emitted; stats.rows_hist[M]++; if (chain.empty()) stats.chain_none++;
    for (int k = 0; k < (int)chain.size(); ++k) { stats.drafted[src[k]]++; if (k < a) stats.accepted[src[k]]++; }
    recent_.push_back(emitted); if (recent_.size() > 64) recent_.erase(recent_.begin());
    
    
    if (c_.prmode == 1 || (c_.prmode == 2 && mtp_spot)) {
      ema_us_ = 0.97 * ema_us_ + ms_since(t_iter) * 1000.0; ema_tok_ = 0.97 * ema_tok_ + emitted; ++ema_n_;
      if (ema_n_ >= 8) tok_us_ = ema_us_ / ema_tok_;
    }
  }
  if constexpr (std::is_same_v<Eng, Engine35>) { if (bdr && dctx_n) { feed_ctx(dctx_n, dctx_pos); if (dctx_n) e_.bdraft_step(dctx_n, dctx_pos, 0, c_.use_graph); } }   
  stats.ms_total = ms_since(t_loop); stats.produced = std::min(produced, n_tokens); stats.eos = stop ? 1 : 0;
  stats.t_copy[0] = t16_; stats.t_copy[1] = t32_; stats.t_copy[2] = t48_; stats.t_copy[3] = t64_;
  if (blk_mode) { static const bool lg = std::getenv("YSKN_BLK16_LOG") != nullptr;
    if (lg) std::fprintf(stderr, "[blk16] mode %d: passes %lld, BDR_J-row passes %lld, ema16 %.2f ema8 %.2f probe %d full8 %.2f\n", blk_mode, b16_all, b16_passes, b16_ema, b8_ema, b16_probe_cur, full8_ema); }
}
template class SpecxT<Engine>;
template class SpecxT<Engine35>;
}
