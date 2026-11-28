// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once










#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "engine.hpp"
#include "engine35.hpp"
#include "copy_index.hpp"
namespace yskn {
struct SpecxConfig {
  std::atomic<int>* cancel = nullptr;
  int gamma = 7;            
  int use_d0 = 1, use_d1 = 1;
  int d0_nmin = 2, d0_nmax = 8;
  int d0_priority_n = 4;    
  int extend = 1;           
  int d1_rejected = 1;      
  int auto_gamma = 0;       
  int use_graph = 1;
  int use_d3 = 0;           
  int d3_steps = 0;         
  int d1_bigram = 0;        
  int d1_static = 0;        
  int d3_extend = 0;        
  int d3_lazy = 0;          
  int d3_gate = 0;          
  int gamma_long = 0;       
  int long_n = 6;
  int d1_trigram = 0;       
  int eos1 = -1, eos2 = -1; 
  int long_d0 = 0;          
  int sel = 0;              
  int cost_tok_us = 0;      
  int cost_row_us = 0;      
  int cost_mtp_us = 0;      
  int mtp_auto = 0;         
  int prmode = 0;          
                            
  int cost_row2_us = 0;     
  int lfrow = 0;           
  int wk_q_pm = 300;        
  int cost_tok_k = 0, cost_row2_k = 0, cost_mtp_k = 0;   
  int mark_tok = -1;        
  
  
  
  int bdraft = 0;           
  int bdr_long_n = 4;       
  int bdr_adapt = 1;        
  int bdr_probe = 4;        
  int bdr_d0_len = 15;      
  
  int prefill_head = 1;     
                            
  int snap = 0;           
  int copy_rows = 16;       // most rows of a copy-from-input pass (16 = as before; up to 64 where wide passes are on)
  int copy_index = 0;       // 0 = the n-gram tables, 1 = the 4-token chain index with the copy cursor (with the block drafter)
  int snap_tok = -1;      
  // experimental (a stream handed back from a slot, yskn_ms.h): the engine holds the prompt without its last token as its live
  // sequence, and that last token is the one the stream emitted last.  Nothing is read: the first pass checks it with its drafts.
  int frontier = 0;
  // det: the copy loop decides by what the passes carried and never by their measured times, so that the same prompt
  // gives the same widths, the same passes and the same ids on every run (speed no longer steers the width).
  int det = 0;
  // wide_need0: how many fully accepted copy passes a run must have before a wide pass is tried (2 = as before, 1 or 0 = sooner)
  int wide_need0 = 2;
};
struct SpecxStats {
  int64_t passes = 0, tokens = 0, rows_hist[65] = {};
  int64_t drafted[3] = {}, accepted[3] = {};   
  int64_t chain_none = 0;                      
  double ms_verify = 0, ms_draft = 0, ms_total = 0;   
  double ms_commit = 0, ms_record = 0;   
  int64_t produced = 0; int eos = 0;     
  int64_t leaves = 0, leaf_breaks = 0, leaf_hits = 0;   
  double ms_mark = -1; int64_t mark_at = -1;   
  int64_t bdr_rounds = 0, bdr_d0 = 0, bdr_probes = 0;   
  double ms_ctx = 0;                                     
  int64_t sh_lcp = 0, sh_slots = 0, sh_saved = 0; double ms_snap = 0;   // prompt snapshots
  int64_t copy_passes = 0, wide_passes = 0, copy_tok = 0, wide_tok = 0, max_rows = 0, cursor_hits = 0, wide_full = 0; double ms_index = 0;
  int64_t wide_backoffs = 0; double t_copy[4] = {}; int wide_mask = 0;   // 0.1.7: back-offs, final times of copy cycles (ms), widths
  int64_t edit_passes = 0, edit_hits = 0, edit_accepted = 0;
  int sh_mode = 0; int64_t sh_start = 0;                 
};

struct SpecxStatic { bool loaded = false; std::vector<int32_t> uni; std::unordered_map<uint64_t, int32_t> bi; };
const SpecxStatic& specx_static();
void specx_load_static(const std::string& path);

template <class Eng>
class SpecxT {
public:
  SpecxT(Eng& e, const SpecxConfig& c, int vocab);
  int  prefill(const int32_t* ids, int n);     
  void generate(int n_tokens, int32_t* out);   
  SpecxStats stats;
  const std::vector<int32_t>& seq() const { return S_; }   
private:
  Eng& e_; SpecxConfig c_; int V_;
  std::vector<int32_t> S_;                     
  std::vector<std::array<int32_t, 8>> T_;      
  std::unordered_map<uint64_t, int32_t> T2_;   
  std::unordered_map<uint64_t, float> T2m_;    
  std::unordered_map<uint64_t, int32_t> T3_;   
  int draft_done_ = 0;                         
  std::vector<std::unordered_map<uint64_t, int>> ng_;   
  int indexed_ = 0;                            
  std::vector<int> recent_;                    
  
  double reach_[3][16] = {}, acc_[3][16] = {}; int skipped_[3] = {};
  double p_acc(int s, int j) const;            
  int best_len(int s, int lmax, double fixed_us, double per_tok_us, int rows0 = 1) const;   
  double tok_us_ = 0, ema_us_ = 0, ema_tok_ = 0; int ema_n_ = 0;   
  double wq_n_[16] = {}, wq_h_[16] = {};          
  double dema_[2] = {3.0, 3.0};                 
  struct BdrPend { int F; int which; std::vector<int32_t> ch; };
  std::vector<BdrPend> dpend_;                    
  int dprobe_ = 0;
  // copy-from-input drafting (copy_index = 1) and wide copy passes (copy_rows > 16)
  CopyIndex ci_;
  int lk_src_ = -1;                 // S_ index of the first token of the copy chain found by the last lookup
  int cur_ = -1;                    // copy cursor: S_ index of the token expected next if copying goes on
  int last_src_ = -1, last_src_at_ = -1;   // where the last copy stopped (S_ index) and the frontier then
  int copy_lookup(int F, std::vector<int32_t>& chain, int gmax);
  int wide_need_ = 2, wide_run_ = 0;   // full 16-row copy passes needed before going wide, and the current run of them
  double wst_tok_ = 0, wst_ms_ = 0;    // tokens and time of the wide passes since the last check
  int wide_k_ = 0;                     // wide passes since the last check (0.1.7: checked every 4 passes, not only when a run ends)
  double wpv_tok_ = 0, wpv_ms_ = 0;    // 0.1.7: the window checked last in the current run (it won), and its number of passes: a run
  int wpv_k_ = 0;                      // that ends with fewer than 4 passes since then is judged together with it
  double t16_ = 10.12, t32_ = 15.5, t48_ = 21.0, t64_ = 23.5;   // ms per copy pass by rows (built-in initial values measured on the RTX
                                                               // 5090: the exact-sum wide passes of 0.1.4, fastest model, 4K context;
                                                               // 0.1.7: the caller can give this GPU's own, set_opt 39..42)
  double row_us(int rows) const { return rows > 8 ? (c_.cost_row2_us > 0 ? c_.cost_row2_us : c_.cost_row_us) + c_.cost_row2_k * dk_ : c_.cost_row_us; }   
  double dk_ = 0;            
  double tok_val() const { return (c_.prmode && tok_us_ > 0) ? tok_us_ : c_.cost_tok_us + c_.cost_tok_k * dk_; }   
  double mtp_us() const { return c_.cost_mtp_us + c_.cost_mtp_k * dk_; }
  uint64_t gram_hash(int end, int n) const;
  void index_upto(int end);                    
  int  d0_lookup(int F, std::vector<int32_t>& chain, int gmax);   
  int32_t next_tok(int32_t p2, int32_t prev, int32_t cur) const;  
  void d1_chain(std::vector<int32_t>& chain, int32_t p2, int32_t prev, int32_t cur, int gmax);
  void record(int32_t p2, int32_t prev, int32_t cur, const int32_t* top8, const float* val8 = nullptr);   
  uint64_t tkey(int32_t a, int32_t b, int32_t c) const { return ((uint64_t)(uint32_t)a * (uint64_t)V_ + (uint64_t)(uint32_t)b) * (uint64_t)V_ + (uint64_t)(uint32_t)c; }
  uint64_t bkey(int32_t a, int32_t b) const { return (uint64_t)(uint32_t)a * (uint64_t)V_ + (uint64_t)(uint32_t)b; }
};
using Specx = SpecxT<Engine>;
using Specx35 = SpecxT<Engine35>;
}
