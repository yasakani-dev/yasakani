// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once






#include <cuda_runtime.h>
#include <cstdint>
namespace yskn {
constexpr int Q35_H = 5120, Q35_NH = 24, Q35_KVH = 4, Q35_HD = 256, Q35_RD = 64;
constexpr int Q35_GK = 16, Q35_GV = 48, Q35_GD = 128, Q35_CONV = 10240, Q35_Z = Q35_GV * Q35_GD;





constexpr int Q35_SST = 16;



struct Q35Xsm { const float* us = nullptr; const float* S = nullptr; float eps = 0.f; int mode = 0; };


struct Q35Pf { const void* p[2] = {nullptr, nullptr}; unsigned long long n[2] = {0ull, 0ull}; };
// multi-stream pass (experimental): rows of several streams in one pass.  The rows of a stream are next to each other: its last token
// first, draft tokens after it.  Per row r: pos[r] = absolute position of the row in its stream, slot[r] = index of that stream in
// the per-layer pointer tables, tl[r] = index of the row among the rows of its stream in this pass (0 = first), len[r] = number of
// rows of the stream in this pass (set on its first row, 0 on the others), pend0[r] / pendn[r] (first row only) = the rows of the
// previous pass that the stream accepted and that its recurrent state does not hold yet (first row index in that pass, count).
// A multi-stream pass runs its own kernels (the *_ms launchers); the argument structs of the single-request kernels carry none of this.
struct Q35MsRows { const int* pos = nullptr; const int* slot = nullptr; const int* tl = nullptr; const int* len = nullptr;
                   const int* pend0 = nullptr; const int* pendn = nullptr; };


void launch_l2pf(const Q35Pf& f, cudaStream_t st);






struct Q35ResNormArgs { const float* part; const float* tau; int nw; const int8_t* sg_out; const uint16_t* emb; const int* ids;
                        int M; int N; float* h; const uint16_t* w; float eps; const int8_t* sg_in; int8_t* qx; float* alpha;
                        uint16_t* xn; unsigned* cnt;
                        int keep_h = 0;      
                        int qx_stride = 0;   
                        int al_stride = 0; };
void launch_q35_res_norm(const Q35ResNormArgs&, cudaStream_t);


void launch_q35_residual(const Q35ResNormArgs&, cudaStream_t);
void launch_q35_res_norm_row(const Q35ResNormArgs&, cudaStream_t);   



struct Q35AbArgs { const uint16_t* xn; const uint16_t* wb; const uint16_t* wa; int M; float* ab; const float* S = nullptr; float eps = 0.f; };
void launch_q35_ab(const Q35AbArgs&, cudaStream_t);



struct Q35MixArgs { const float *part_qkv, *part_z; const float *tau_qkv, *tau_z; int nw_qkv, nw_z; const int8_t *sg_qkv, *sg_z;
                    int M; uint16_t* mixed; float* z; Q35AbArgs ab; int with_ab; Q35Xsm xsm; Q35Pf pf; };
void launch_q35_mix(const Q35MixArgs&, cudaStream_t);



struct Q35ConvArgs { const uint16_t* mixed; const uint16_t* conv_w; uint16_t* hist; const float* ab; const uint16_t* A_log;
                     const uint16_t* dt_bias; int M; int n_commit; float* q; float* k; float* v; float* beta; float* g; };
void launch_q35_conv(const Q35ConvArgs&, cudaStream_t);
// multi-stream pass (at most 16 rows): hist[slot] = the conv history of this layer for each stream
struct Q35ConvMs { Q35MsRows ms; uint16_t* const* hist = nullptr; };
void launch_q35_conv_ms(const Q35ConvArgs&, const Q35ConvMs&, cudaStream_t);



struct Q35StateArgs { const float *q, *k, *v, *beta, *g; float* S; int M; int n_commit; float* core; const uint16_t* mixed; uint16_t* hist; };
void launch_q35_capture_a8(const float*,int,int,int,const int8_t*,int8_t*,float*,int,cudaStream_t);
void launch_q35_state_nb(const Q35StateArgs&, cudaStream_t);
void launch_q35_state(const Q35StateArgs&, cudaStream_t);
// multi-stream pass (at most 16 rows): S[slot] / hist[slot] = the recurrent state and the conv history of this layer for each stream;
// pk / pv / pbeta / pg = the kept conv outputs of the previous pass (the rows a stream accepted there go into its state first)
struct Q35StateMs { Q35MsRows ms; float* const* S = nullptr; uint16_t* const* hist = nullptr;
                    const float *pk = nullptr, *pv = nullptr, *pbeta = nullptr, *pg = nullptr; };
void launch_q35_state_ms(const Q35StateArgs&, const Q35StateMs&, cudaStream_t);
// multi-stream pass: store the history of rows a stream accepted after the pass.  mixed[l] = the mixed rows of the last pass of
// recurrent layer l, hist[l][slot] = history of that layer for the stream, grp = (slot, first row, rows) per group.
struct Q35MsHistArgs { const uint16_t* const* mixed; uint16_t* const* const* hist; const int* grp; int layers; int groups; };
void launch_q35_ms_hist(const Q35MsHistArgs&, cudaStream_t);

struct Q35GdnOutArgs { const float* core; const float* z; const uint16_t* gw; float eps; int M; const int8_t* sg_in; int8_t* qx; float* alpha; int lo = 0; };
void launch_q35_gdn_out(const Q35GdnOutArgs&, cudaStream_t);


struct Q35AttnPostArgs { const float *part_q, *part_k, *part_v; const float *tau_q, *tau_k, *tau_v; int nw_q, nw_k, nw_v;
                         const int8_t *sg_q, *sg_k, *sg_v; const uint16_t *qn_w, *kn_w; float eps; int M; const int* d_pos;
                         const float2* rope; uint16_t* q16; uint16_t* gate16; uint16_t* kc; uint16_t* vc; int max_ctx; Q35Xsm xsm;
                         int8_t* k8 = nullptr; int8_t* v8 = nullptr; float* ks = nullptr; float* vs = nullptr; Q35Pf pf; int kv_fp16 = 0;
                         const unsigned char* tb = nullptr; int tb_layer = 0; unsigned char* tb_scr = nullptr;
                         
                         
                         int kstg = 0, vstg = 0; int pks = 128; uint8_t* kp = nullptr; uint8_t* vp = nullptr; float* kps = nullptr; float* vps = nullptr; };   
void launch_q35_attn_post(const Q35AttnPostArgs&, cudaStream_t);
void launch_tb_requant(const Q35AttnPostArgs&, cudaStream_t);   
// multi-stream pass (at most 16 rows): the KV containers of each stream (by slot); the rows of a staged (packed) layer go through
// the shared staging rows as in a single-request pass
struct Q35AttnPostMs { Q35MsRows ms; int8_t* const* k8 = nullptr; float* const* ks = nullptr; int8_t* const* v8 = nullptr; float* const* vs = nullptr;
                       uint8_t* const* kp = nullptr; float* const* kps = nullptr; uint8_t* const* vp = nullptr; float* const* vps = nullptr; };
void launch_q35_attn_post_ms(const Q35AttnPostArgs&, const Q35AttnPostMs&, cudaStream_t);
void launch_tb_requant_ms(const Q35AttnPostArgs&, const Q35AttnPostMs&, cudaStream_t);
void launch_q35_rope_table(float2* out, int max_ctx, cudaStream_t, int pm = 0, bool yarn2 = false);
void launch_q35_visual_rope(float2* out, int max_ctx, const int* positions, int n, int delta, cudaStream_t, int pm, bool yarn2 = false);
void launch_q35_visual_embed(uint16_t* rows, const uint16_t* table, const int* ids,
                            const uint16_t* features, const int* map, const int* pos,
                            int M, int max_ctx, cudaStream_t);


void q35_set_pm(int on, cudaStream_t st);
void q35_flow_set_pm(int on, cudaStream_t st);
void pfa_set_pm(int on, cudaStream_t st);



struct Q35AttnArgs { const uint16_t* q16; const uint16_t* kc; const uint16_t* vc; int max_ctx; int M; const int* d_pos; int splits;
                     float* pm; float* pl; float* pacc; int early;
                     const int8_t* k8 = nullptr; const int8_t* v8 = nullptr; const float* ks = nullptr; const float* vs = nullptr;   
                     int splits_short = 0, long_len = 0;    
                     int vsp = 0; int kv_fp16 = 0; };                       
void launch_q35_attention(const Q35AttnArgs&, cudaStream_t);

void launch_q35_fa(const Q35AttnArgs&, cudaStream_t);


void launch_q35_atx(const Q35AttnArgs&, const uint8_t* wk, cudaStream_t);


constexpr int Q35_SMP_K = 20;
struct Q35SmpParams { double temp; double top_p; int top_k; int version; unsigned long long seed;
                       int pm = 0;   
                       double min_p = 0.0; };   
struct Q35SmpArgs { const float* logits; int V; int rows; int row0; int* d_tok_out; const int* d_pos; const uint8_t* wk; const Q35SmpParams* prm;
                     float* cand_val; int* cand_idx; unsigned* cnt; int* adv_pos = nullptr; int* adv_tok = nullptr;   
                     float* top_val = nullptr; int* top_idx = nullptr;    
                     float* topk_val = nullptr; int* topk_idx = nullptr; };   
void launch_q35_smp(const Q35SmpArgs&, cudaStream_t);
bool q35_smp2_on();   
// multi-stream pass: pos[r] = the position of row r in its stream, prm[r] = the sampling of that stream (temp 0: the row keeps the
// argmax that d_tok_out holds); Q35SmpArgs::prm is not read
void launch_q35_smp_ms(const Q35SmpArgs&, const int* pos, const Q35SmpParams* prm, cudaStream_t);

struct Q35CombineArgs { const float* pm; const float* pl; const float* pacc; int splits; int M; const uint16_t* gate16;
                        const int8_t* sg_in; int8_t* qx; float* alpha;
                        const int* d_pos = nullptr; const uint8_t* wk = nullptr; int splits_short = 0, long_len = 0; int lo = 0; };   
void launch_q35_combine(const Q35CombineArgs&, cudaStream_t);
void launch_q35_combine_ms(const Q35CombineArgs&, const int* pos, cudaStream_t);   // multi-stream pass: pos[r] = the position of row r


void launch_q35_kz(const Q35AttnArgs&, const uint8_t* wk, cudaStream_t);
void launch_q35_kz_combine(const Q35CombineArgs&, float* dbg, cudaStream_t);

struct Q35MlpArgs { const float *part_g, *part_u; const float *tau_g, *tau_u; int nw_g, nw_u; const int8_t *sg_g, *sg_u; int M; int F;
                    const int8_t* sg_in; int8_t* qx; float* alpha; Q35Xsm xsm; Q35Pf pf; int lo = 0; };
void launch_q35_mlp(const Q35MlpArgs&, cudaStream_t);


struct Q35XsmResArgs { const float* part; const float* us; const float* tau; const int8_t* sg_out; const uint16_t* emb; const int* ids;
                       int M; float* h; const uint16_t* w; const int8_t* sg_in; int8_t* qx; float* alpha; float* S; int mode = 0; int nw = 0;
                       uint16_t* xu = nullptr; Q35Pf pf; int lo = 0; };   
void launch_q35_res_norm_xsm(const Q35XsmResArgs&, cudaStream_t);

void launch_q35_xsm_xn(const float* h, const float* S, const uint16_t* w, float eps, int M, uint16_t* xn, cudaStream_t);

void launch_q35_pick(const int* idx, const int* ids, int* out, int* out2, cudaStream_t);
void launch_q35_mtp_log(const int* top_i, const float* top_v, const int* ids, int* lg_idx, float* lg_val, int* cnt, cudaStream_t);   


void launch_wk_conv(const Q35ConvArgs&, const uint8_t* wk, cudaStream_t);
void launch_wk_state(const Q35StateArgs&, const uint8_t* wk, cudaStream_t);
void launch_wk_attn_post(const Q35AttnPostArgs&, const uint8_t* wk, cudaStream_t);
void launch_wk_attention(const Q35AttnArgs&, const uint8_t* wk, cudaStream_t);
struct WkCopy { unsigned long long ptr; int row_bytes; int kind; };   
                                                                   
void launch_wk_promote(const WkCopy* tab, int n, int src, int dst, int base, int max_ctx, cudaStream_t);
void launch_wk_fill(int* toks, const uint8_t* wk, const int* lg_idx, int k, int M, cudaStream_t);
void launch_q35_iota(int* out, int n, cudaStream_t);






struct Q35EmbArgs { const uint32_t* codes; int K; const uint16_t* lut16; const uint16_t* scale16; const int8_t* signs; const int* ids; uint16_t* out;
                    const uint32_t* codes2 = nullptr; int K2 = 0; const uint16_t* lut2 = nullptr; const uint16_t* scale2 = nullptr;
                    const uint32_t* codes3 = nullptr; int K3 = 0; const uint16_t* lut3 = nullptr; const uint16_t* scale3 = nullptr; };   
void launch_q35_embed(const Q35EmbArgs&, int M, cudaStream_t);

void launch_q35_pos_slices(const int* base, int* out, int n, int step, cudaStream_t);

struct Q35ChkStats { unsigned long long checks, eq_fail, range_fail; unsigned first_set; int first_tensor, first_row, pad;
                        unsigned long long first_lhs, first_rhs, first_viol; };
struct Q35ChkArgs { const unsigned long long* acc; const int8_t* qx; const float* alpha; const long long* c; int N, Kc; unsigned long long key;
                       int w_max; int tensor; Q35ChkStats* st; };
void launch_q35_chk(const Q35ChkArgs&, int M, cudaStream_t);
void launch_q35_chk_fault(unsigned long long* acc, int kind, int row, int bit, cudaStream_t);

// drafter rows of several streams in one block (multi-stream pass)
struct BdrMs {
  const int* tl = nullptr;      // per block row: its index in its stream's block (0 = the stream's last token)
  const int* pos = nullptr;     // per block row: its position in its stream
  const int* grp = nullptr;     // per block row: its group (a stream of this block)
  const int* g_b0 = nullptr;    // per group: its first block row
  const int* g_len = nullptr;   // per group: its rows
  const int* g_slot = nullptr;  // per group: its stream slot (index into the ring tables)
  const int* g_tok = nullptr;   // per group: the index of its first row in the token array of the pass
  const int* g_d0 = nullptr;    // per group: its first row in the draft-row arrays
  const int* d_row = nullptr;   // per draft row: its block row
  const unsigned long long* g_seed = nullptr;   // per group: the sampling seed of the stream
  int ng = 0;
  uint16_t* const* rk = nullptr; uint16_t* const* rv = nullptr;   // per stream slot: the ring of the layer
};
struct BdrMsCtx {                // context rows of several streams
  const int* row = nullptr;     // per context row: its row in the capture of the last pass
  const int* pos = nullptr;     // per context row: its position in its stream
  const int* slot = nullptr;    // per context row: its stream slot
  uint16_t* const* rk = nullptr; uint16_t* const* rv = nullptr;
};
void launch_bdr_rownorm(const float* x, int n, int ld, float eps, float* r, int M, cudaStream_t);
void launch_bdr_a8(const float* x, int ld, int n, const float* r, const uint16_t* w, const int8_t* signs, float* xn, int8_t* qx, float* alpha, int M, cudaStream_t);
void launch_bdr_post(long long* acc, const float* us, const float* tau, int N, float* y, int ldy, int M, cudaStream_t);
void launch_bdr_gconv(const float* x, const float* dyn, int which, const uint16_t* base, float* out, int resid, int M, cudaStream_t);
void launch_bdr_qk_rope(float* q, float* k, const uint16_t* qn, const uint16_t* kn, const int* pos, int pos_off, float eps, int M, int with_q, cudaStream_t);
void launch_bdr_ring_store(const float* k, const float* v, uint16_t* rk, uint16_t* rv, const int* pos, int pos_off, int M, cudaStream_t);
void launch_bdr_attn(const float* q, const float* kb, const float* vb, const uint16_t* rk, const uint16_t* rv, const int* pos, int M, float* part, float* out, cudaStream_t);
size_t bdr_attn_part_bytes();                                           
void launch_bdr_silu_mul(const float* g, const float* u, int n, float* s, int M, cudaStream_t);
void launch_bdr_embed(const uint16_t* emb, const int* gids, float* h, int M, cudaStream_t);
void launch_bdr_block_ids(const int* toks, int mask, int* ids, int M, cudaStream_t);
void launch_bdr_gather_feats(const float* cap, int pmax, int row0, float* x, int M, cudaStream_t);
void launch_bdr_topk16(float* lg, int V, float* val, int* idx, int rows, float* cand_v, int* cand_i, cudaStream_t);
void launch_bdr_map_ids(int* idx, const int* ids, int n, cudaStream_t);   
size_t bdr_topk_cand();

void launch_bdr_a8n(const float* x, int ld, int n, float eps, const uint16_t* w, const int8_t* signs, float* xn, int8_t* qx, float* alpha, int M, cudaStream_t);
void launch_bdr_a8_cap(const float* cap, int pmax, int row0, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t);
void launch_bdr_dynconv_a8(long long* acc, const float* us, const float* tau, float* dyn, const float* x, const uint16_t* base, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t);
void launch_bdr_qkv_post(long long* aq, long long* ak, long long* av, const float* us, const float* tq, const float* tk, const float* tv, const uint16_t* qn,
                         const uint16_t* kn, const int* pos, float eps, float* yq, float* yk, float* yv, int M, cudaStream_t);
void launch_bdr_kv_ring(long long* ak, long long* av, const float* us, const float* tk, const float* tv, const uint16_t* kn, const int* pos, float eps,
                        uint16_t* rk, uint16_t* rv, int M, cudaStream_t);
void launch_bdr_attn_part(const float* q, const float* kb, const float* vb, const uint16_t* rk, const uint16_t* rv, const int* pos, int M, float* part, cudaStream_t);
void launch_bdr_attn_comb_a8(const float* part, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t);
void launch_bdr_post_conv_res(long long* acc, const float* us, const float* tau, const float* dyn, const uint16_t* base, int M, float* h, cudaStream_t);
void launch_bdr_glu_a8(long long* ag, long long* au, const float* us, const float* tg, const float* tu, int FF, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t);                                                 
void launch_bdr_hproj(const float* hf, const uint16_t* w, float* hp, int rows, cudaStream_t);
void launch_bdr_select(const float* hp, const float* val, const int* idx, const uint16_t* pc, const uint16_t* sc, int* toks, int* out,
                       const Q35SmpParams* prm, const int* cpl, const int* pos, int nj, cudaStream_t, int* alt = nullptr, float* marg = nullptr, int* alt3 = nullptr);   
void launch_bdr_hcap(const float* hp, const float* val, const int* idx, const uint16_t* pc, const uint16_t* sc,
                     int* toks, const int* alt, const int* alt3, const float* marg, const float* tab, uint8_t* wk, int* cfg, const int* cpl, const int* go, int nj, cudaStream_t);
// the drafter kernels of a multi-stream block (BdrMs / BdrMsCtx above); the single-stream launchers above are not changed by them
void launch_bdr_block_ids_ms(const int* toks, int mask, int* ids, int M, const BdrMs& ms, cudaStream_t);
void launch_bdr_a8n_rows(const float* x, int ld, int n, float eps, const uint16_t* w, const int8_t* signs, float* xn, int8_t* qx, float* alpha, int M,
                         const int* rowmap, cudaStream_t);   // output row m is made from row rowmap[m] of x
void launch_bdr_a8_cap_rows(const float* cap, int pmax, const int* rows, const int8_t* signs, int8_t* qx, float* alpha, int M, cudaStream_t);   // capture rows rows[m]
void launch_bdr_dynconv_a8_ms(long long* acc, const float* us, const float* tau, float* dyn, const float* x, const uint16_t* base, const int8_t* signs, int8_t* qx,
                              float* alpha, int M, const int* tl, cudaStream_t);
void launch_bdr_qkv_post_ms(long long* aq, long long* ak, long long* av, const float* us, const float* tq, const float* tk, const float* tv, const uint16_t* qn,
                            const uint16_t* kn, const int* rpos, float eps, float* yq, float* yk, float* yv, int M, cudaStream_t);   // rpos[m] = position of row m
void launch_bdr_kv_ring_ms(long long* ak, long long* av, const float* us, const float* tk, const float* tv, const uint16_t* kn, float eps, int M,
                           const BdrMsCtx& ms, cudaStream_t);
void launch_bdr_attn_part_ms(const float* q, const float* kb, const float* vb, float* part, const BdrMs& ms, cudaStream_t);
void launch_bdr_post_conv_res_ms(long long* acc, const float* us, const float* tau, const float* dyn, const uint16_t* base, int M, float* h, const int* tl, cudaStream_t);
void launch_bdr_select_ms(const float* hp, const float* val, const int* idx, const uint16_t* pc, const uint16_t* sc, int* toks, const Q35SmpParams* prm, const int* gcpl,
                          const BdrMs& ms, cudaStream_t);   // gcpl[g] = the draft temperature of group g (-1: greedy drafts); prm: pm only





constexpr int BDR_B = 16, BDR_J = BDR_B - 1;   
constexpr int BDR_SHEAD_R = 512, BDR_SHEAD_CAP = 2048, BDR_SHEAD_HIST = BDR_J * (1024 + 2048);
void launch_bdr_shead(const float* hf, const float* bt, const int8_t* a, const float* as, const int8_t* w8, const float* ws, int V, int M, int grid,
                      float* q, float* s, unsigned* hist, int* thr, int* ci, float* cv, float* val, int* idx, int rows, cudaStream_t);
int bdr_shead_grid();                                                     
} 
