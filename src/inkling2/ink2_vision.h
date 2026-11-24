// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Image encoder of Inkling-2 Mini on the GPU: the internal C++ interface of ink2_vision.cu. Nothing here is
// exported from the library; the exported C calls are the ink2_vision_* wrappers of the engine
// (ink2_vision_api.h), which check their arguments, own the feature buffer and call these functions.
//
// One still image in, one BF16 vector of 2304 values per image token out:
//   RGB bytes -> pad to multiples of 40 px with -1, v = (p - 127.5) / 127.5 -> 20 x 20 native patches -> patch
//   projection + bilinear position grid + class token -> LayerNorm -> 23 pre-norm blocks (attention with a 2-D
//   rotary embedding, exact-GELU MLP, layer scale) -> 2 x 2 merge -> proj_in, exact GELU, proj_out -> BF16.
// Weights stay BF16 as published; every matrix product takes BF16 inputs and sums in FP32; the residual stream,
// LayerNorm, rotary, softmax and GELU are FP32. The output is rounded to BF16 once.
//
// Rules of every call: it returns 0 (or a valid pointer) on success and -1 (or null) on refusal, with a message in
// err (at least INK2V_ERR_LEN bytes; err may be null). Every check runs before anything is written; a refused call
// leaves the encoder and the caller's buffers unchanged. No call aborts the process.
#pragma once

#include "ink2_vision_api.h"  // Ink2VisionConfig

struct CUstream_st;  // cudaStream_t without including the CUDA headers
struct Ink2Vis;

#define INK2V_ERR_LEN 512
#define INK2V_TENSORS 422          // tensors model.visual.* of the checkpoint
#define INK2V_OUT_DIM 2304         // values per image token (text hidden size)
#define INK2V_TILE 40              // pixels per image token side (2 x 2 native patches of 20 px)

// Stages kept from the last encode, for tests (ink2v_stage). Shapes with N image tokens, S = 1 + 4 N:
//   0 pixel_values [N, 40, 40, 3]  1 patch_proj [4 N, 1024]  2 embeddings [S, 1024]  3 block1 [S, 1024]
//   4 block23 [S, 1024]            5 merged [N, 4096]         6 output [N, 2304] (FP32, before the BF16 rounding)
// Stages 1..3 need ink2v_set_capture(v, 1) before finalize (three extra [S, 1024] FP32 buffers).
enum { INK2V_STAGES = 7 };

// Checks every value of c (the only supported model: see ink2_vision.cu) and the device (compute capability 8.0
// or newer, and the library holds code for it). Every later call runs its copies and kernels on `stream`; the
// engine passes its own stream, so the image rows are written before any later pass of the engine reads them.
Ink2Vis* ink2v_create(const Ink2VisionConfig* c, CUstream_st* stream, char* err);

// One tensor of the checkpoint by its full name (model.visual....). nbytes must equal the BF16 size of its shape.
// The first call reserves the device memory of all weights (636,512,256 bytes). Unknown names, duplicates and wrong
// sizes are refused. The bytes are copied before the call returns.
int ink2v_set_tensor(Ink2Vis* v, const char* name, const void* bf16, long long nbytes, char* err);
// The same with the dtype size and shape of the file header checked as well (dtype_bytes must be 2).
int ink2v_set_tensor_shaped(Ink2Vis* v, const char* name, const void* bf16, int dtype_bytes, const long long* shape,
                            int ndim, long long nbytes, char* err);
// Keep stages 1..3 of each encode (tests only). Before finalize.
int ink2v_set_capture(Ink2Vis* v, int on, char* err);
// Refuses unless all 422 tensors are present; allocates all scratch for max_image_tokens.
int ink2v_finalize(Ink2Vis* v, char* err);

// Encodes one image. rgb: host bytes, rows of width x 3 (R, G, B), rgb_bytes = width * height * 3. rows and cols
// must equal ceil(height / 40) and ceil(width / 40); rows * cols must not exceed max_image_tokens nor out_rows.
// Writes rows * cols BF16 rows of 2304 values to dev_out (device memory, row-major), in raster order of the token
// grid. Allocates nothing. Runs on the stream given at create (returns when the work is queued). The RGB bytes are
// copied with cudaMemcpyAsync: pageable memory (what the engine's callers pass) is read before the call returns;
// page-locked memory must stay unchanged until the stream has run the copy.
int ink2v_encode(Ink2Vis* v, const unsigned char* rgb, int width, int height, long long rgb_bytes, int rows, int cols,
                 unsigned short* dev_out, long long out_rows, char* err);

// Number of floats of a stage of the last encode (0 when there is none or the stage is not kept).
long long ink2v_stage_floats(const Ink2Vis* v, int stage);
// Copies one stage of the last encode to host memory (waits for the stream). out_floats >= ink2v_stage_floats.
int ink2v_stage(Ink2Vis* v, int stage, float* out, long long out_floats, char* err);

// The engine's feature buffer: `bytes` of device memory, zero filled. A failed allocation returns null with the
// message in err and clears the CUDA error it caused, so the engine's own error state is not touched.
unsigned short* ink2v_alloc_rows(long long bytes, char* err);
void ink2v_free_rows(unsigned short* p);

// Device bytes held: weights and scratch (stage buffers included).
int ink2v_memory(const Ink2Vis* v, long long* weights, long long* scratch, char* err);

void ink2v_destroy(Ink2Vis* v);
