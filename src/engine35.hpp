// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once






#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "model.hpp"
#include "q35.hpp"
#include "args.hpp"
#include "q35_g6.hpp"
namespace yskn {
struct Shared35Weights;
struct Q35Layer {
  bool full = false;
  DevTensor qkv, z, out, q, k, v, o, gate, up, down;
  uint16_t *ln1 = nullptr, *ln2 = nullptr;
  uint16_t *wa = nullptr, *wb = nullptr, *conv = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gnorm = nullptr;   
  uint16_t *qn = nullptr, *kn = nullptr;                                                                         
  float* S = nullptr; uint16_t* hist = nullptr; uint16_t* pmix = nullptr; float* pab = nullptr;   
  float *tq = nullptr, *tk = nullptr, *tv = nullptr, *tbeta = nullptr, *tg = nullptr;           
  uint16_t *kc = nullptr, *vc = nullptr;                                                           
  int8_t *k8 = nullptr, *v8 = nullptr; float *ks = nullptr, *vs = nullptr;                        
  uint8_t *kp = nullptr, *vp = nullptr; float *kps = nullptr, *vps = nullptr;                      
};





struct Q35Mtp {
  bool on = false; Container c; DevTensor fc, head; Q35Layer L; int nsub = 0;
  uint16_t *en = nullptr, *hn = nullptr, *norm = nullptr; int* ids = nullptr;   
  uint16_t *hin = nullptr, *hout = nullptr, *pend = nullptr;                    
  int *toks = nullptr, *d_pos = nullptr, *idx = nullptr, *out = nullptr, *vtoks = nullptr, *iota = nullptr;
  uint16_t* hist = nullptr; int* ctoks = nullptr; int32_t* h_ctoks = nullptr; int last_rows = 1;   
  std::map<int, cudaGraphExec_t> graphs;
  float *tkv = nullptr, *top_v = nullptr, *lg_val = nullptr; int *tki = nullptr, *top_i = nullptr, *lg_idx = nullptr, *lg_cnt = nullptr; unsigned* tkc = nullptr;   
};
class Engine35 {
public:
  Engine35(const std::string& path, int max_ctx, int flags);
  ~Engine35();
  void shared_info(uint64_t* out) const;
  void reset();
  int  prefill(const int32_t* ids, int n);                 
  
  
  
  
  
  
  
  
  void set_visual(const uint16_t* features, const int32_t* slots, int n_image,
                  const int32_t* positions, int n_prompt, int delta);
  void set_suppress(const int32_t* ids, int n);
  void decode(int32_t* out, int steps);                    
  void nll(const int32_t* ids, int n, float* out);         
  void logits(float* out);                                 
  int  position() const { return pos_; }
  int  vocab() const { return V_; }
  int  max_ctx() const { return max_ctx_; }
  void verify(const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals,
              const int32_t* targets = nullptr, float* out_nll = nullptr);
  void commit(int n_accept);
  void hidden_rows(int n, float* out);
  void logits_rows(int row0, int n, float* out);
  
  
  float time_verify(int M, int mask, int iters);
  int  prefill_rows(const int32_t* ids, int n, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
  int  prefill_prompt_rows(const int32_t* ids, int n, int grid, int32_t* out_argmax, int32_t* top_ids, float* top_vals,
                           const uint8_t* keep = nullptr);
  
  void load_draft(const std::string& path, const std::string& head_ids_path);   
  bool has_draft() const { return mtp_.on; }
  void set_verify_tokens(const int32_t* ids, int first_row, int n);
  int32_t get_verify_token(int row);
  void verify_device(int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids, float* top_vals = nullptr);
  static constexpr bool kVerifyRows = false;                 
  
  
  
  
  
  void set_opt(int key, int value);
  void set_specx_scope(bool on) { specx_scope_ = on; }
  
  void mtp_log_reset();
  
  
  
  void lfrow_table(const uint8_t* tab48);
  void lfrow_fill(int k, int M);
  void lfrow_promote(int src, int dst);
  int  mtp_log_read(int32_t* ids, float* vals, int max_steps);   
  
  
  void draft_round(int mode, int a, int steps, int use_graph);
  
  
  void hist_store(int pos0, int n);
  void draft_catchup(int p0, int n, const int32_t* toks, int use_graph);
  void draft_steps(int steps, int use_graph);
  
  
  void debug_hidden(const int32_t* ids, int n, int upto, int mode, float* out);
  
  
  
  
  int state_hash(uint64_t* out, int max_out);
  // hist_dump: the conv history of one layer (3 x Q35_CONV uint16) into host memory, for the width tests.  Read only.
  int hist_dump(int layer, uint16_t* out, int max_words);
  // layer_hash: FNV of one layer's conv history (which = 0) or recurrent state S (which = 1), for the width tests.  Read only;
  // call it only after a read is finished (a dump in the middle of a read disturbs the next pass).
  int layer_hash(int layer, int which, uint64_t* out);
  
  
  
  
  
  
  
  // Prompt snapshots: up to YSKN_SNAP_SLOTS saved states (pinned host memory), all prefixes of snap_seq(), whose KV rows stay on the
  // GPU.  A slot holds the linear-attention state, the conv history and the drafter ring at one position.
  void snap_set_seq(const int32_t* ids, int n) { sh_seq_.assign(ids, ids + n); }
  const std::vector<int32_t>& snap_seq() const { return sh_seq_; }
  bool snap_save_at(int n, bool extra = false);
  void snap_add_gap_cuts(int n, std::vector<int>& cuts, std::vector<int>& extra);
  void snap_load_slot(int i);                        // restore slot i (position = its position)
  int  snap_best(int L) const;                       // the slot with the largest position <= L, or -1
  int  snap_slot_pos(int i) const { return sh_slots_[(size_t)i].pos; }
  void snap_trim(int L);                             // forget the slots after position L
  int  snap_count() const { int c = 0; for (const SnapSlot& s : sh_slots_) c += s.pos > 0; return c; }
  int  snap_pos() const { int p = -1; for (const SnapSlot& s : sh_slots_) p = std::max(p, s.pos); return p; }
  // the live sequence: ids = the n tokens this engine holds in its KV, front = the token it emitted last (not in its KV yet; -1 = not
  // known).  Both hold while snap_live_ok(); the frontier form of yskn_spec_generate checks the last prompt token against front.
  void snap_note(const int32_t* ids, int n, int32_t front) { sh_cur_.assign(ids, ids + n); sh_front_ = front; sh_note_ = sh_epoch_; }
  // forget the live sequence: the next read cannot continue it (mode 1); the saved states stay (mode 2)
  void live_drop() { sh_cur_.clear(); sh_front_ = -1; sh_note_ = 0; }
  bool snap_live_ok() const { return sh_note_ == sh_epoch_ && (int)sh_cur_.size() == pos_ && pos_ > 0; }
  const std::vector<int32_t>& snap_live() const { return sh_cur_; }
  int32_t snap_front() const { return snap_live_ok() ? sh_front_ : -1; }   // the frontier of the live sequence, -1 without one
  void snap_drop() { for (SnapSlot& s : sh_slots_) s.pos = -1; sh_seq_.clear(); sh_last_ = -1; }
  int  snap_slots_max() const { return sh_max_; }
  int  snap_step() const { return sh_step_; }
  int  snap_tail() const { return sh_tail_; }
  int  snap_sparse() const { return sh_sparse_; }
  int  snap_min() const { return sh_min_; }
  int  prefill_chunk() const { return PM_; }
  // Copy passes wider than 16 rows (verify of host tokens only).
  static constexpr int kVerifyMax = 64;
  bool wide_ok() const { return wide_ok_; }
  bool edit_copy_enabled() const { return edit_copy_; }   // also when sampling (the drafts are checked as copy drafts are)
  int xw_rows() const { return xw_rows_; }           // passes up to this many rows use the exact-sum GEMV (16 = none wider)
  void prepare_wide(int max_rows);                   // build the graphs of the wide passes and their commits (nothing runs)
  // 0.1.7: which widths of copy passes may be used (bit 0 = 32 rows, 1 = 48, 2 = 64; set_opt 38) and the times of one copy cycle
  // that the copy loop starts from (16, 32, 48 and 64 rows, microseconds, 0 = the RTX 5090's values; set_opt 39..42).
  int wide_mask() const { return wide_ok_ ? wide_mask_ : 0; }
  int wide_t_init_us(int i) const { return (i >= 0 && i < 4) ? wide_t_us_[i] : 0; }
  // One copy cycle of M rows at the current position, as the copy loop runs it: the verify pass (graph, every row through the output
  // layer, top-8), the commit of all M rows and the drafter's context of these rows; then the position goes back. ms[i] = wall time
  // of cycle i. The values in the state are not kept meaningful: reset afterwards.
  void time_cycle(const int32_t* ids, int M, int iters, float* ms);
  int wide_state(int32_t* out, int n) const;         // see yskn_wide_state
  
  
  
  
  
  
  int chk_set(const std::string& name, const long long* c, int Kc, unsigned long long key, int w_max);
  void chk_stats(unsigned long long* out, int reset);
  void chk_fault(int kind, const std::string& name, int a, int b);
  void chk_enable(int on);
  
  
  void set_capture(const int* layers, int n);
  void capture_rows(int k, int n, float* out);
  
  void set_kz_debug(int layer);
  void kz_debug_read(int n, float* o, uint16_t* q);
  int kv_format() const { return kv8_ ? 8 : (kv_fp16_ ? 16 : 116); }
  void kv16_read(int layer, int pos0, int n, uint16_t* k, uint16_t* v);
  void kv8_read(int layer, int pos0, int n, int8_t* k8, float* ks, int8_t* v8, float* vs);
  
  
  
  
  
  
  void load_bdraft(const std::string& path);
  bool has_bdraft() const { return bdr_.on; }
  void bdraft_ctx(int row0, int n, int pos0);
  void bdraft_draft(int use_graph);
  
  
  void bdraft_step(int n_ctx, int pos_ctx, int draft, int use_graph);
  
  int prefill_rows_k20(const int32_t* ids, int n, int32_t* out_argmax, int32_t* k20_ids, float* k20_vals);
  void bdraft_debug(const float* feats, int n, int pos0, int32_t anchor, float* hidden, int32_t* top_ids, float* top_vals, int32_t* path);
  int  bdraft_out(int32_t* drafts);
  int  bdraft_cands(int32_t* ids, float* vals);   // J x 16 drafter candidates (host out)            
  void bdraft_ring(int layer, int slot0, int n, uint16_t* k_out, uint16_t* v_out);   
  // multi-stream pass (experimental): the rows of several streams (requests) ride in one pass of this engine (the carrier).  A pass
  // has at most 16 rows.  ms_bind sets the stream slots: n_light light slots (the state containers of a stream: recurrent states,
  // their histories, the KV rows; numbers n .. n + n_light - 1) and, for tests only, n engines that share the weights (numbers 0 ..
  // n - 1).  ms_load copies the state of an engine after its prompt pass into a light slot.  ms_step runs one pass: the rows of a
  // stream are next to each other, toks of its first row = the stream's last token, the following rows = draft tokens.  The first row
  // is committed inside the pass (position + 1).  out_argmax[r] = the token the model gives after row r (the sampled one when the
  // slot of row r samples).  ms_commit (once, right after ms_step) says how many drafts of each stream were right: the positions
  // advance by that many rows.  Sampling is set per slot (ms_set_sampling: the values and checks of set_opt 10, 11, 12, 29, 16 and 13;
  // ms_load takes the source engine's, a new slot starts with this engine's); a row whose slot has temperature 0 takes the argmax,
  // so greedy and sampled streams share a pass.  ms_seeds sets only the seed of every slot.  ms_flush: finish the pending work of the
  // streams; the single-request entries of this engine (the C API) call it first when anything is pending (ms_gate).
  // This engine's live sequence (what a read may continue, mode 1): ms_bind, ms_load, ms_draft and ms_step drop it (the stream it held
  // may live on in a slot); ms_commit, ms_flush, the settings and the queries keep it; ms_unload sets it to the stream it takes.
  // The frontier of a light slot (the token its stream emitted last, not in its KV yet): ms_load takes the source's (known when the
  // source has a live sequence), ms_step sets it to out_argmax of the stream's first row, ms_commit to the one after its accepted
  // drafts; ms_unload hands it to this engine with the live sequence.  While it is known, ms_step / ms_draft refuse a first row with
  // another token.  A light slot that no ms_load has filled takes no rows.
  void ms_bind(Engine35* const* eng, int n, int n_light = 0);
  void ms_load(int slot, Engine35* src);
  // The inverse of ms_load: this engine takes the stream of a light slot (its states, histories, KV rows, position, drafter memory and
  // sampling); ids = the n tokens the slot holds (n = its position; checked: the tokens of its passes, and the ones it was loaded with
  // when the source had a live sequence).  Its live sequence becomes ids, so that a read of ids + more tokens continues it (mode 1);
  // when the source had none (the head of ids cannot be checked) it stays dropped and no saved state stays.  Its saved states stay
  // only where they still match the KV rows it now holds (see the definition).  The slot is marked as moved out: passes refuse it
  // until ms_load fills it again.
  void ms_unload(int slot, const int32_t* ids, int n);
  // what a caller needs to plan the slots (see yskn_ms_info in yskn_ms.h); returns the number of values written
  int ms_info(int64_t* out, int max_out) const;
  int  ms_slot_hash(int slot, uint64_t* out, int max_out);               // the values state_hash gives for an engine in that state
  int  ms_slot_pos(int slot);
  // The drafter for the streams (needs the drafter loaded before ms_bind, and ms_opt(1, 1)): ms_draft takes the rows of the NEXT pass
  // (toks: the last token of every stream at its first row; for a stream with dev == 0 at its first row also the caller's own drafts)
  // and fills the other rows of the streams with dev != 0 on the device.  ms_step with toks == nullptr then runs that pass; out_toks
  // gets the tokens of all rows.  While ms_opt(1) is on, the rows a stream accepted go into its drafter memory in ms_commit: call it
  // right after every ms_step (n may be 0), before the engine is used for anything else (the rows are read from the capture of the pass).
  void ms_step(int M, const int32_t* slots, const int32_t* toks, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals, int32_t* out_toks = nullptr);
  void ms_draft(int M, const int32_t* slots, const int32_t* toks, const int32_t* dev, int use_graph);
  void ms_opt(int key, int value);
  void ms_commit(int n, const int32_t* slots, const int32_t* n_accept);
  void ms_seeds(const uint32_t* seeds, int n);
  void ms_set_sampling(int slot, int temp_x1000, int top_p_x1000, int top_k, int min_p_x1000, int sampler, uint32_t seed);
  void ms_flush();
  void ms_gate() { if (ms_ns_ > 0 && (ms_open_ || ms_dtoks_ || ms_ctx_pending_ || ms_pending())) ms_flush(); }
private:
  struct MsTab { float** S = nullptr; uint16_t** hist = nullptr; int8_t** k8 = nullptr; float** ks = nullptr; int8_t** v8 = nullptr; float** vs = nullptr;
                 uint8_t** kp = nullptr; float** kps = nullptr; uint8_t** vp = nullptr; float** vps = nullptr; };
  // kept conv outputs of a pass (16 rows per recurrent layer): the accepted drafts of a pass go into the state at the start of the next
  // pass (or in ms_settle), so a pass writes one set while the rows of the pass before are read from the other.  The single-request
  // path keeps its own rows in Q35Layer::tq.. for its commit replay; the two never share a buffer.
  struct MsKeep { float *tq = nullptr, *tk = nullptr, *tv = nullptr, *tbeta = nullptr, *tg = nullptr; };
  struct MsPend { int row0 = 0, n = 0; };                                // accepted rows of the last pass that the state does not hold yet
  struct MsLay { float* S = nullptr; uint16_t* hist = nullptr; uint8_t* kp = nullptr; float* kps = nullptr; uint8_t* vp = nullptr; float* vps = nullptr;
                 int8_t* k8 = nullptr; float* ks = nullptr; int8_t* v8 = nullptr; float* vs = nullptr; };   // the state containers of one layer of a stream
  // the sampling of a stream slot, as set_opt keeps it for this engine (smp_, smp_p_): on = temperature > 0 (else the argmax)
  struct MsSmp { int on = 0; Q35SmpParams p{0.6, 0.95, 20, 0, 0ull}; };
  // a stream slot without an engine.  out: moved to this engine by ms_unload (no pass takes it until ms_load).  What ms_load knew
  // of the source: src_self = it was this engine, src_tick = sh_tick_ then (the saved states older than that were taken on the KV
  // rows the slot copied), src_ids = the tokens the source held (its live sequence; src_known = false when it had none).
  // ring = the drafter memory came from the source (false: cleared, the source had no drafter).  load_pos = the position ms_load
  // copied; tail = the tokens of the rows the slot took since (each pass: the first row, then the accepted drafts), so that
  // load_pos + tail.size() = pos and ms_unload can check every token of ids it did not get from the source's live sequence.
  // front = the stream's frontier (-1 = not known: loaded from a source without a live sequence and no pass since); filled = an
  // ms_load has put a stream into the slot (a slot made by ms_bind holds none).
  struct MsLight { std::vector<MsLay> L; int pos = 0; MsSmp smp; bool out = false, src_self = false, src_known = false, ring = false;
                   unsigned long long src_tick = 0; std::vector<int32_t> src_ids; int load_pos = 0; std::vector<int32_t> tail;
                   int32_t front = -1; bool filled = false; };
  std::vector<MsLight> ms_light_; int ms_ns_ = 0;                        // light slots that exist; slots of the current binding (engines + light)
  int& ms_posr(int s) { const int ne = (int)ms_eng_.size(); return s < ne ? ms_eng_[(size_t)s]->pos_ : ms_light_[(size_t)(s - ne)].pos; }
  std::vector<MsSmp> ms_esmp_;                                           // the sampling of the engine slots (test-only form) of the binding
  MsSmp& ms_smpr(int s) { const int ne = (int)ms_eng_.size(); return s < ne ? ms_esmp_[(size_t)s] : ms_light_[(size_t)(s - ne)].smp; }
  std::vector<MsTab> ms_tab_; std::vector<Engine35*> ms_eng_; int ms_cap_ = 0; int *ms_pos_ = nullptr, *ms_slot_ = nullptr; bool ms_now_ = false;
  int *ms_rows_ = nullptr, *ms_tl_ = nullptr, *ms_len_ = nullptr, *ms_pend0_ = nullptr, *ms_pendn_ = nullptr, *ms_grp_ = nullptr; int32_t* ms_hgrp_ = nullptr;
  // per row of a pass: the sampling of its slot (temp 0 = argmax), device table and pinned staging; ms_smp_now_ = a row of the
  // current pass samples (the head runs the sampling kernel)
  Q35SmpParams *ms_prm_ = nullptr, *ms_hprm_ = nullptr; bool ms_smp_now_ = false;
  std::vector<MsKeep> ms_keep_[2]; std::vector<MsPend> ms_pend_; int ms_par_ = 0; bool ms_open_ = false;   // ms_keep_[parity of the pass][layer]
  bool ms_pending() const { for (const MsPend& p : ms_pend_) if (p.n > 0) return true; return false; }
  std::vector<int> ms_last_r0_, ms_last_len_; std::vector<int32_t> ms_last_toks_, ms_last_am_;
  const uint16_t** ms_mixed_tab_ = nullptr; uint16_t*** ms_hist_tab_ = nullptr; int ms_ngdn_ = 0;
  Q35MsRows ms_rows() const { return Q35MsRows{ms_pos_, ms_slot_, ms_tl_, ms_len_, ms_pend0_, ms_pendn_}; }
  void ms_settle(const std::vector<int>& slots);
  void ms_carrier_check(const char* what) const;
  int ms_unsupported(std::string* why, bool profile = true) const;      // 0, or the code of a setting the pass is not built for (yskn_ms_info)
  void smp_assign(int on, const Q35SmpParams& p);                        // this engine's sampling, as set_opt 10..13, 16 and 29 set it
  void ms_drop_graphs();                                                 // the graphs of the pass and of its drafter
  void ms_attn_post(Q35Layer& Y, const Q35AttnPostArgs& X);              // the kernels of a multi-stream pass (q35.hpp *_ms)
  void ms_attention(Q35Layer& Y, const Q35AttnArgs& AT, int cap_ord, bool kz);
  // drafter of the multi-stream pass
  struct MsRing { uint16_t* rk[5] = {}; uint16_t* rv[5] = {}; };          // the drafter rings of a light slot
  std::vector<MsRing> ms_dring_;
  uint16_t** msd_rk_[5] = {}; uint16_t** msd_rv_[5] = {}; int msd_cap_ = 0;   // per drafter layer: the ring of every slot (device tables)
  int* msd_tab_ = nullptr; unsigned long long* msd_seed_ = nullptr; int32_t* msd_h_ = nullptr; unsigned long long* msd_hseed_ = nullptr;
  int ms_drafter_ = 0, ms_last_graph_ = 0; bool ms_ctx_pending_ = false, ms_dtoks_ = false, msd_ok_ = false;
  std::vector<int> ms_last_acc_, ms_dlay_;
  std::map<int, cudaGraphExec_t> ms_dgraph_, ms_cgraph_;
  void ms_ctx_push(int use_graph);
  void bdr_block_ms(int R, int ng, int J);
  void bdr_ctx_ms(int m);
  std::map<int, cudaGraphExec_t> ms_graph_;
  std::shared_ptr<Shared35Weights> shared_weights_;
  struct Bdr {
    bool on = false; Container c; DevTensor fc; DevTensor L[5][9];        
    uint16_t *hnorm = nullptr, *fnorm = nullptr, *ln1[5] = {}, *ln2[5] = {}, *qn[5] = {}, *kn[5] = {}, *abase[5] = {}, *mbase[5] = {};
    uint16_t *hp = nullptr, *pc = nullptr, *sc = nullptr, *rk[5] = {}, *rv[5] = {};
    float *x = nullptr, *x1 = nullptr, *x2 = nullptr, *r = nullptr, *alpha = nullptr, *us = nullptr, *h = nullptr, *dyn = nullptr, *att = nullptr;
    float *yq = nullptr, *yk = nullptr, *yv = nullptr, *yo = nullptr, *yg = nullptr, *yu = nullptr, *ys = nullptr, *yd = nullptr, *yfc = nullptr;
    float *hf = nullptr, *lg = nullptr, *tval = nullptr, *hpv = nullptr; int8_t* qx = nullptr; long long* hacc = nullptr;
    int *tidx = nullptr, *ids = nullptr, *pos = nullptr, *out = nullptr; const int8_t* signs = nullptr;
    float *apart = nullptr, *tcv = nullptr; int *tci = nullptr, *cpos = nullptr;   
    int* cpl = nullptr; int cpl_pm = 850;                                 
    cudaGraphExec_t sgraph[3][17] = {};   
    
    
    Container csub; DevTensor hsub; int* sub_ids = nullptr; int nsub = 0; int sub_on = 0;
    
    int* alt = nullptr; int* alt3 = nullptr; float* marg = nullptr; float* hb_tab = nullptr; int* hb_cfg = nullptr; int* hb_go = nullptr; int hb_on = 0; int hb_pass = 0;
    
    
    
    float *sk_bt = nullptr, *sk_as = nullptr, *sk_ws = nullptr, *sk_q = nullptr, *sk_s = nullptr, *sk_cv = nullptr; int8_t *sk_a = nullptr, *sk_w = nullptr;
    unsigned* sk_hist = nullptr; int *sk_thr = nullptr, *sk_ci = nullptr; int sk_m = 0, sk_grid = 0;
  } bdr_;
  void load_shead(const char* path);                                      
  void free_shead();                                                      
  void bdr_gemv(std::initializer_list<DevTensor*> ts, int M);          
  void bdr_ctx_rows(const float* x_dev, int row0, int m);                
  void bdr_block(int rows);                                                       
  void bdr_block_impl(int* toks, bool hcap, int rows);   
  void chk_after(std::initializer_list<DevTensor*> ts, int M, const int8_t* qx, const float* alpha);
  float* cap_ = nullptr; std::vector<int> cap_layers_; int cap_stage_ = 0;   // 0: h_ after the layer, 1: h_ after the SSM/attn block, 2: core_
  float* d_k20v_ = nullptr; int* d_k20i_ = nullptr; bool k20_on_ = false;   
  struct ChkEntry { long long* c = nullptr; unsigned long long key = 0; int w_max = 127; int idx = 0; };
  std::map<const DevTensor*, ChkEntry> chk_; Q35ChkStats* d_mstats_ = nullptr; int chk_on_ = 1;
  int mm_kind_ = 0, mm_row_ = 0, mm_bit_ = 0; const DevTensor* mm_t_ = nullptr;
  DevTensor* tensor_by_name(const std::string& name);
  void drop_pass_graphs();
public:
  bool hcap_on() const { return bdr_.on && bdr_.hb_on && tbcap_ == 0; }
  bool greedy_on() const { return smp_ == 0; }
  void hcap_pass(int on) { bdr_.hb_pass = on; }
  void hcap_cfg(int* out16) const;
  void lfrow_device_table() { wk_set_ = true; }
private:
  void apply_kz();   
  [[noreturn]] static void no_d3() { throw std::runtime_error("qwen3_5 engine: lazy D3 mode is not supported (use the eager MTP draft)"); }
  enum { HEAD_NONE = 0, HEAD_LAST = 1, HEAD_ALL = 2 };
  DevTensor up(const std::string& name, int force_workers = 0, const Container* src = nullptr);
  uint16_t* vec(const std::string& name, size_t n, const Container* src = nullptr, bool host = false);
  void gemv(std::initializer_list<DevTensor*> ts, int M);
  void res_norm(const DevTensor* src, int M, const uint16_t* w, const int8_t* sg_in, uint16_t* xn_out);
  void emb_src(const int* ids, int M, const uint16_t*& emb, const int*& gids);   
  void attn_block(Q35Layer& Y, int M, const int* dpos, int* slices);
  void mlp_block(Q35Layer& Y, int M, bool down = true);   
  void layer(int l, int M, int n_commit);
  void mtp_pass(int M, bool head);                          
  void run_mtp(int M, bool head, bool use_graph);
  void mtp_catchup(int M);                                  
  void ensure_part(size_t floats);
  void head_slice(int row0, int rows, bool advance, bool topk, bool nll);
  void pass_body(int M, bool commit, int upto, int head_mode, bool topk, bool nll, bool advance);
  void replay_commit(int rows);
  void run_pass(int M, bool commit, int head_mode, bool topk, bool nll, bool use_graph);
  void upload_rows(const int32_t* ids, int M, const int32_t* targets);
  Model m_; std::vector<Q35Layer> L_; DevTensor head_; uint16_t *fnorm_ = nullptr, *emb_ = nullptr; Q35Mtp mtp_; size_t part_floats_ = 0; int vt_slot_ = 0;
  int V_ = 0, H_ = 0, F_ = 0, nlayers_ = 0, max_ctx_ = 0, splits_ = 32, want_ = 0, pos_ = 0, last_M_ = 0, flags_ = 0;
  int Mmax_ = 64;        
  int Pmax_ = 64;        
  uint16_t* pmix_big_ = nullptr; float* pab_big_ = nullptr;   
  int *pos_slices_ = nullptr, *mtp_pos_slices_ = nullptr;     
  int kskip_ = 0;        
                         
                         
  bool ks(int bit) const { return (kskip_ & bit) != 0; }
  int pfa_ = 1, pfa_wpc_ = 0;   
  int nwk(const DevTensor& t, int M) const { return (pfa_ && M > Mmax_) ? 1 : t.dev.n_workers; }   
  int xsm_ = 1;          
  Q35G6Workspace g6_; 
  int mlp_stripe_rows_ = 0;
  bool prefill_mlp_stripes(Q35Layer& layer, int rows);
  bool xsmc_ = false;    
  float *us_ = nullptr, *S_ = nullptr, *mtp_part_ = nullptr;   
  float* ybuf_ = nullptr;  
  float* pt(const DevTensor& t, int M) const { return (xsmc_ && M > xw_rows_) ? ybuf_ + (t.part - part_) : t.part; }
  int xsm_part_ = 0;     
  int nmode(int M, bool normed = false) const { return M > xw_rows_ ? (normed ? 3 : 1) : (xsm_part_ ? 2 : 0); }   
  // Exact-sum passes of 17..xw_rows_ rows (compute capability 12.x): the same int64 sums as the 16-row path, read by the same consumer
  // kernels (mode 0).  16 = off (17..64 rows use the prefill GEMM and float outputs, as before).
  int xw_rows_ = 16, xw_arch_ = 0;
  int xw_v32_ = 1, xw_v64_ = 0;      // kernel for 17..32 / 33..64 rows: 0 = staged-input GEMM with a K split, 1 = 32-row GEMV
  int xw_wpc_ = 8, xw_nws_ = 0;      // warps per CTA of the staged-input kernel; K split: 0 = chosen per launch shape (xw_pick),
                                     // > 0 = the 16-row split x xw_nws_ / rows
  int xw_gamma_ = 200;               // xw_pick: cost of one more wave of CTAs, in hundredths of the time of one 512-column group
  std::map<std::vector<int>, int> xw_pick_cache_;
  int xw_pick(const GemvLaunch& P, int cols_per_cta, int slots);
  int xw_head_ = 1;                  // the output layer in one exact-sum GEMV for passes of up to xw_rows_ rows
  int xw_nw(const DevTensor& t, int M) const;
  void gemv_xw(GemvLaunch& P);
  void head_wide(int M, bool topk, bool nll, int first = 0, bool prompt = false);
  void head_all_rows(int M, bool topk, bool nll);
public:
  int test_xsum(const std::string& name, int M, unsigned seed, int variant, long long* out4);   // compare with 16-row launches
private:
  void gemv(std::initializer_list<DevTensor*> ts, int M, bool normed, const GemvLaunch* fin = nullptr);   
  
  
  void gemv_res(DevTensor& src, int M, const uint16_t* w, const int8_t* sg_in, uint16_t* xn_out);
  
  
  
  
  
  
  
  int kv8_ = 1;
  int tbcap_ = 0;        
  int tbcap_kb_ = 4;     
  int stg_rows_ = 0; int8_t *stg_k8_ = nullptr, *stg_v8_ = nullptr; float *stg_ks_ = nullptr, *stg_vs_ = nullptr; unsigned char* tb_rec_ = nullptr; unsigned char* tb_scr_ = nullptr;   
  int kv_fp16_ = 0; 
  int emb_host_ = 0;     
  
  
  
  bool emb_coded_ = false; int emb_K_ = 0; uint32_t* emb_codes_ = nullptr; uint16_t *emb_lut_ = nullptr, *emb_sc_ = nullptr, *emb_rows_ = nullptr;
  int8_t* emb_sg_ = nullptr; int* emb_iota_ = nullptr;
  uint32_t* emb2_codes_ = nullptr; uint16_t *emb2_lut_ = nullptr, *emb2_sc_ = nullptr; int emb2_K_ = 0;   
  uint32_t* emb3_codes_ = nullptr; uint16_t *emb3_lut_ = nullptr, *emb3_sc_ = nullptr; int emb3_K_ = 0;   
  int fa_ = 1, fa_splits_ = 31;
  int atx_ = 1, atx_splits_ = 42;   
  
  
  int atx_short_ = 31, atx_long_ = 4096;
  int atx_vsp_ = 3;   
                      
  
  
  int kz_ = 0, kz_dbg_layer_ = -1, kz_dbg_rows_ = 0; float* kz_dbg_ = nullptr; uint16_t* kz_dbgq_ = nullptr;
  
  unsigned long long* flow_trace_ = nullptr;
  int opt_flow_ = 0; int8_t* qxd_ = nullptr; float* ald_ = nullptr; float* usd_ = nullptr; unsigned* flow_cnt_ = nullptr;
  bool flow_ok(int M) const { return opt_flow_ && xsmc_ && !xsm_part_ && M <= 8 && !kskip_ && !opt_fin_ && !in2_; }
  void flow_mlp(Q35Layer& Y, int M, int l, bool prepare_only = false);
  void flow_trace_read(float* out, int n);
  void flow_prepare() { if (opt_flow_ && xsm_ && !xsm_part_) for (int M = 1; M <= 8; ++M) flow_mlp(L_[0], M, 0, true); }
  int opt_fin_ = 0; bool in2_ = false; int8_t* qx2_ = nullptr; float* alpha2_ = nullptr; unsigned* fin_cnt_ = nullptr;
  
  
  
  int l2pf_ = 0; double l2pf_mb_[3] = {4.0, 6.0, 6.0}; Q35Pf next_pf_;
  
  
  double l2pf2_mb_ = 0.0; cudaStream_t st3_{}; cudaEvent_t ev_g_{}, ev_pf_{}; bool pf2_live_ = false;
  
  
  int opt_lo_ = 1; bool lo_now_ = false;                                   
  Q35Pf pf_of(std::initializer_list<const DevTensor*> ts, double mb) const;
  bool pf_on(int bit, int M) const { return (l2pf_ & bit) && xsmc_ && M <= 16; }
  
  
  int smp_ = 0; Q35SmpParams smp_p_{0.6, 0.95, 20, 0, 0ull}; Q35SmpParams* d_smp_ = nullptr; float* kj_val_ = nullptr; int* kj_idx_ = nullptr; unsigned* kj_cnt_ = nullptr;
  int HM_ = 16;          
  int PM_ = 64;          
  float eps_ = 1e-6f; size_t lno_bytes_ = 0;
  cudaStream_t st_{};
  
  
  
  cudaStream_t st2_{}; cudaEvent_t ev_v_{}, ev_rp_{}; bool rp_pending_ = false; int opt_rp_async_ = 1;   
  
  
  int opt_rp_defer_ = 0; cudaGraphExec_t rp_graph_ = nullptr; bool rp_deferred_ = false;   
  void rp_launch() { if (rp_deferred_) { cudaStreamWaitEvent(st2_, ev_v_, 0); cudaGraphLaunch(rp_graph_, st2_); cudaEventRecord(ev_rp_, st2_); rp_deferred_ = false; } }
  void replay_fence() { if (rp_pending_) { rp_launch(); cudaStreamWaitEvent(st_, ev_rp_, 0); rp_pending_ = false; } }
  float *h_ = nullptr, *alpha_ = nullptr, *z_ = nullptr, *gq_ = nullptr, *gk_ = nullptr, *gv_ = nullptr, *gbeta_ = nullptr, *gg_ = nullptr, *core_ = nullptr;
  uint16_t *xn_ = nullptr, *q16_ = nullptr, *gate16_ = nullptr; int8_t* qx_ = nullptr;
  float *pm_ = nullptr, *pl_ = nullptr, *pacc_ = nullptr; float2* rope_ = nullptr; float* part_ = nullptr;
  uint8_t* suppressed_ = nullptr; bool suppress_on_ = false;
  bool yarn2_ = false;
  float *tc_h_ = nullptr, *tc_al_ = nullptr, *tc_S_ = nullptr; uint16_t* tc_xn_ = nullptr; int8_t* tc_qx_ = nullptr; int* tc_ids_ = nullptr; int tc_rows_ = 0;
  int8_t* tq_qx_ = nullptr; float* tq_al_ = nullptr; int8_t* scr_qx_ = nullptr; float* scr_al_ = nullptr;   
  
  
  int uat2_ = 0;    
  int ovl_ = 0; cudaStream_t nb_st_ = nullptr; cudaEvent_t nb_e1_[2] = {}, nb_e2_[2] = {}; int nb_rows_ = 0;
  uint16_t* nb_pmx_[2] = {}; float *nb_z_[2] = {}, *nb_cq_[2] = {}, *nb_ck_[2] = {}, *nb_cv_[2] = {}, *nb_cb_[2] = {}, *nb_cg_[2] = {}, *nb_core_[2] = {};
  void gdn_a(int l, int M, int b, const uint16_t* prev); void gdn_s(int l, int M, int b); void gdn_b(int l, int M, int b);
  float* tc_cap_ = nullptr; const float* dctx_cap_ = nullptr; int dctx_rows_ = 0;   
  int8_t* tc_dq_=nullptr; float* tc_da_=nullptr; const int8_t* dctx_q_=nullptr; const float* dctx_a_=nullptr;
  int layer_rows_=0; bool layer_ready_=false, layer_unavailable_=false;
  uint4* layer_weights_=nullptr; size_t layer_weight_bytes_=0;
  bool layer_prepare(); void layer_decode(int l);
  int prefill_vsp(const int32_t* ids, int n, int32_t* prediction = nullptr,
                  int32_t* top_ids = nullptr, float* top_vals = nullptr, const uint8_t* keep = nullptr);
  int prefill_chunks(const int32_t* ids,int n);
public:
  void nll_layer(const int32_t* ids,int n,float* out);
  void layer_info(uint64_t* out) const { out[0]=layer_ready_;out[1]=layer_unavailable_;out[2]=tc_rows_;out[3]=ovl_;out[4]=layer_weight_bytes_;out[5]=layer_rows_; }
private:
  bool visual_on_ = false;
  int visual_cap_ = 0, visual_prompt_ = 0, visual_delta_ = 0;
  uint16_t* visual_features_ = nullptr;
  int *visual_map_ = nullptr, *visual_positions_ = nullptr;
  float *logits_ = nullptr, *blk_val_ = nullptr, *blk_max_ = nullptr, *blk_sum_ = nullptr, *d_nll8_ = nullptr, *tk_val_ = nullptr, *top_val_ = nullptr;
  int *blk_idx_ = nullptr, *d_pos_ = nullptr, *d_toks_ = nullptr, *d_next8_ = nullptr, *d_target8_ = nullptr, *tk_idx_ = nullptr, *top_idx_ = nullptr;
  unsigned *cnt_ = nullptr, *am_cnt_ = nullptr, *tk_cnt_ = nullptr;
  
  int32_t* h_pin_ = nullptr; float* h_pinf_ = nullptr;
  int32_t* h_dec_ = nullptr;   
  cudaGraphExec_t gexec_ = nullptr; std::map<int, cudaGraphExec_t> pgraph_;
  // the graphs of read passes (commit, more than HM_ rows) are kept for kReadGraphs keys at most, the least recently used dropped
  // first (read_graph_room): reads come in every length, and each kept graph holds GPU memory
  static constexpr int kReadGraphs = 32; std::map<int, uint64_t> pg_use_; uint64_t pg_tick_ = 0; void read_graph_room();
  cudaGraphExec_t gexec_alt_ = nullptr;   
  std::map<int, cudaGraphExec_t> rgraph_;                     
  int opt_replay_graph_ = 1, opt_single_commit_ = 1, opt_catchup_nosync_ = 1, opt_mtp_log_ = 0;
  int decl_nc_ = 0; int run_nc_ = -1;   // in-pass state write: rows the caller will commit
  int opt_lfrow_ = 0; bool wk_now_ = false, wk_set_ = false; uint8_t* d_wk_ = nullptr; uint8_t* h_wk_ = nullptr; int wk_slot_ = 0;
  WkCopy* d_wkcopy_ = nullptr; int n_wkcopy_ = 0;
  bool wk_pass(int M, bool commit, int head_mode) const { return opt_lfrow_ && wk_set_ && !commit && M <= HM_ && head_mode == HEAD_ALL; }
  void wk_plain_table();
  int opt_kconv_ = 1;      
  bool specx_scope_ = false;  
  bool pre_committed_ = false;   
  
  struct SnapSlot { float* S = nullptr; uint16_t* hist = nullptr; uint16_t* ring = nullptr; uint16_t* pend = nullptr; int pos = -1; unsigned long long tick = 0;
                    unsigned long long born = 0; bool extra = false; };   // born: the tick of the save (a restore moves tick, not born)
  std::vector<SnapSlot> sh_slots_; int sh_max_ = 1, sh_step_ = 1024, sh_tail_ = 4096, sh_sparse_ = 65536, sh_min_ = 8192, sh_last_ = -1;
  unsigned long long sh_tick_ = 0; bool sh_warned_ = false;
  std::vector<int32_t> sh_seq_, sh_cur_; unsigned long long sh_epoch_ = 1, sh_note_ = 0; int32_t sh_front_ = -1;
  bool snap_alloc(SnapSlot& s);
  bool snap_grow();
  bool sh_fill_gaps_ = true;
  void snap_free();
  int last_keep_ = 1;          // the last verify pass kept the conv outputs for the commit replay (rows <= 16)
  bool wide_ok_ = false;       // wide copy passes allowed on this GPU
  // 0.1.7: the code the GPU runs (PTX version of a probe kernel, e.g. 86, 89, 120) decides the default; YSKN_WIDE overrides it
  int code_arch_ = 0, sass_arch_ = 0, cc_ = 0, code_arch_test_ = 0; bool xw_env_ = true;
  int wide_pref_ = -1;         // YSKN_WIDE: -1 not set (12.x on, 8.x off), 0 off, 1 on (8.x: closed until the caller opens widths), 2 all
  int wide_mask_ = 0;          // widths of copy passes allowed (bit 0 = 32, 1 = 48, 2 = 64)
  int wide_t_us_[4] = {};      // initial times of copy cycles of 16 / 32 / 48 / 64 rows (us; 0 = built-in)
  int pci_[3] = {}; unsigned char uuid_[16] = {};
  bool wide_native() const { return code_arch_ >= 120; }
  void set_wide_mask(int mask);
  bool edit_copy_ = false;
  
  
  
  int vis_n_ = 0, vis_delta_ = 0; std::vector<int32_t> vis_pos_; std::vector<int> vis_map_; std::vector<unsigned long long> vis_hash_;
  bool visual_prefix_agrees(const std::vector<int>& map, const int32_t* positions, int n_prompt, int delta,
                            const std::vector<unsigned long long>& hash) const;
};
} 
