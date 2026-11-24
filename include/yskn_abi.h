// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
#  if defined(YSKN_BUILD)
#    define YSKN_API __declspec(dllexport)
#  else
#    define YSKN_API __declspec(dllimport)
#  endif
#else
#  define YSKN_API __attribute__((visibility("default")))
#endif
#define YSKNCUDA_API
#ifdef __cplusplus
extern "C" {
#endif
YSKNCUDA_API int yskn_repack_warp_to_frag(const uint32_t* d_words, uint32_t* d_frag, int rows, int seqs, int wps, void* cuda_stream);
YSKNCUDA_API int yskn_repack_warp_to_frag_host(const uint32_t* h_words, uint32_t* h_frag, int rows, int seqs, int wps);
typedef struct yskn_engine yskn_engine;
 
YSKN_API yskn_engine* yskn_open(const char* yskn_path, int max_ctx, int device, int flags);
YSKN_API void          yskn_close(yskn_engine* e);
YSKN_API int           yskn_reset(yskn_engine* e);                                        
YSKN_API int           yskn_prefill(yskn_engine* e, const int32_t* ids, int n, int32_t* first_pred);  


YSKN_API int           yskn_set_visual(yskn_engine* e, const uint16_t* features, const int32_t* slots, int n_image,
                                       const int32_t* positions, int n_prompt, int delta);
/* Encode one or more RGB uint8 frames [H_i, W_i, 3] on GPU into BF16 Inkling-2 visual embeddings [n_tokens, hidden]. */
YSKN_API int           yskn_ink2_encode_rgb(yskn_engine* e, const uint8_t* const* frames,
                                            const int32_t* widths, const int32_t* heights, int n_frames,
                                            uint16_t* out_bf16, int32_t* out_grids, int32_t* out_n_tokens);
/* Suppress output token IDs (logits -> -inf). Prompt IDs are unaffected. Pass token_ids=NULL or n_tokens=0 to clear. */
YSKN_API int           yskn_set_suppress(yskn_engine* e, const int32_t* token_ids, int n_tokens);
YSKN_API int           yskn_decode(yskn_engine* e, int32_t* out_ids, int steps);           
YSKN_API int           yskn_nll(yskn_engine* e, const int32_t* ids, int n, float* out_nll);  
YSKN_API int           yskn_logits(yskn_engine* e, float* out_vocab);                      
YSKN_API int           yskn_position(yskn_engine* e);
YSKN_API const char*   yskn_last_error(void);



YSKN_API int           yskn_verify(yskn_engine* e, const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
YSKN_API int           yskn_verify_nll(yskn_engine* e, const int32_t* ids, const int32_t* targets, int M, int use_graph, int32_t* out_argmax, float* out_nll);
YSKN_API int           yskn_commit(yskn_engine* e, int n_accept);
YSKN_API int           yskn_prefill_rows(yskn_engine* e, const int32_t* ids, int n, int32_t* first_pred, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
YSKN_API int           yskn_hidden_rows(yskn_engine* e, int n, float* out);               
YSKN_API int           yskn_logits_rows(yskn_engine* e, int row0, int n, float* out);     
 
YSKN_API int           yskn_set_capture(yskn_engine* e, const int32_t* layers, int n);
YSKN_API int           yskn_capture_rows(yskn_engine* e, int k, int n, float* out);




 
YSKN_API int           yskn_load_draft(yskn_engine* e, const char* yskn_path, const char* head_ids_path);
YSKN_API int           yskn_draft_round(yskn_engine* e, int mode, int a, int steps, int use_graph);    
YSKN_API int           yskn_verify_device(yskn_engine* e, int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids);
YSKN_API int           yskn_set_verify_tokens(yskn_engine* e, const int32_t* ids, int first_row, int n);
YSKN_API int           yskn_draft_catchup(yskn_engine* e, int p0, int n, const int32_t* toks, int use_graph);
YSKN_API int           yskn_spec_generate(yskn_engine* e, const int32_t* prompt, int n_prompt, int n_gen, const int32_t* cfg, int n_cfg, int32_t* out_ids, double* stats, double* times_ms);




YSKN_API int           yskn_specx_load_static(const char* path);
/* 0.1.4: build the graphs of the wide copy passes (up to max_rows rows) ahead of the first request; nothing runs.
   0.1.7: only the widths that are open (set_opt 38). */
YSKN_API int           yskn_prepare_wide(yskn_engine* e, int max_rows);
/* 0.1.7: wide copy passes per GPU.
   - YSKN_WIDE (read when the model is opened): not set = by the code the GPU runs (sm_120: allowed, all widths open; sm_86 / sm_89:
     off, as in 0.1.6), 0 = off, 1 = allowed (sm_86 / sm_89: no width open until set_opt 38 opens one), 2 = allowed, all widths open.
   - set_opt 38: the widths of copy passes that may be used (bit 0 = 32 rows, bit 1 = 48, bit 2 = 64); 0 = none.
     set_opt 39..42: the time of one copy cycle of 16 / 32 / 48 / 64 rows the copy loop starts from (us; 0 = the RTX 5090's values).
   - yskn_time_cycle: run `iters` copy cycles of M rows (ids[0..M)) at the current position (verify pass with every row through the
     output layer, commit of all M rows, the drafter reading them) and go back to the same position after each; ms[i] = wall time
     of cycle i. Read a prompt first and reset afterwards.
   - yskn_wide_state: out[0] layout (1), [1] code architecture (PTX version the GPU runs, e.g. 86, 89, 120), [2] compute capability
     x10, [3] SMs, [4] wide passes allowed, [5] exact-sum wide GEMV possible, [6] rows up to which it is used, [7] open widths,
     [8] YSKN_WIDE (-1 not set), [9] 1 for 12.x code, [10..13] set_opt 39..42, [14] free MiB, [15] total MiB, [16..18] PCI domain,
     bus, device, [19..22] the GPU's UUID (16 bytes, big-endian words), [23] code architecture set by a test build, [24] machine-code
     architecture, [25] built-in draft head loaded, [26] drafter loaded. *n_out = the number of values written (at most n). */
YSKN_API int           yskn_time_cycle(yskn_engine* e, const int32_t* ids, int M, int iters, float* ms);
YSKN_API int           yskn_wide_state(yskn_engine* e, int32_t* out, int n, int* n_out);
/* "cuda=13020;wide=1": the CUDA version this library was built with, and wide=1 for 0.1.7's switches of the wide copy passes.
   Libraries before 0.1.7 have no yskn_build_info. */
YSKN_API const char*   yskn_build_info(void);


YSKN_API int           yskn_set_opt(yskn_engine* e, int key, int value);


YSKN_API int           yskn_lfrow_table(yskn_engine* e, const uint8_t* tab48);
YSKN_API int           yskn_lfrow_promote(yskn_engine* e, int src, int dst);



#ifdef __cplusplus
}
#endif
