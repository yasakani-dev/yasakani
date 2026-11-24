// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Image input of the Inkling-2 Mini engine (ABI version 4, capability bit INK2_CAP_VISION).
//
// The image tower turns one RGB image into N = ceil(height / 40) * ceil(width / 40) image vectors of the text
// hidden size (2304). The engine keeps them in a feature buffer on the device, BF16 [max_feature_rows, hidden].
// A prompt position whose token is the image placeholder (INK2_IMAGE_TOKEN) names one row of that buffer
// (ink2_feed_many_img); that position then takes the image vector as its embedding, in place of the embedding
// table row and its RMS norm (the official model scatters the image vectors after the embedding norm). Every
// other part of the forward pass (one-dimensional positions, causal and sliding attention, caches, rollback)
// treats image positions as ordinary positions.
//
// Order of the calls: ink2_finalize (text model), ink2_vision_create, ink2_vision_set_tensor (or
// ink2_vision_set_tensor_shaped) for every tensor of the image tower, ink2_vision_finalize; then per request ink2_vision_encode (or ink2_vision_set_rows with
// vectors kept from an earlier encode) and ink2_feed_many_img. Every call returns 0, or -1 with a message in
// ink2_last_error(h); no call aborts the process. Every check happens before anything is written on the device,
// so a refused call leaves the position, the caches and the feature buffer as they were. (An encode that fails on
// the GPU after its checks may leave its target rows partly written; those rows are then marked unwritten.)
#pragma once
#include "ink2_api.h"

#define INK2_IMAGE_TOKEN 200054  // the placeholder id of one image position

typedef struct Ink2VisionConfig {
  // Values of the image tower (config.json "vision_config"); only the published Inkling-2 Mini values are taken:
  // hidden 1024, layers 23, heads 16, head_dim 64, inter 4096, patch 20, merge 2, temporal 2, grid 32,
  // proj_hidden 4096, text_hidden = hidden size of the text model, projector_bias 0, eps 1e-5, rope_theta 10000.
  int hidden, layers, heads, head_dim, inter, patch, merge, temporal, grid, proj_hidden, text_hidden, projector_bias;
  float eps, rope_theta;
  int max_image_tokens;  // image vectors of one image at most (64 .. 4096): sizes the encoder scratch
  int max_feature_rows;  // image vectors of one request at most (max_image_tokens .. 65536): sizes the feature buffer
} Ink2VisionConfig;

// After ink2_finalize, CUDA backend only. Checks the config values.
INK2_API int ink2_vision_create(void* h, const Ink2VisionConfig* c);
// One tensor of the image tower (name as in the checkpoint, "model.visual.*"), BF16, exactly its size in bytes.
// Unknown names, a tensor given twice and a wrong byte count are refused; the tensor is then not taken.
INK2_API int ink2_vision_set_tensor(void* h, const char* name, const void* bf16, long long nbytes);
// The same, with the dtype ("BF16") and the shape of the file header checked too (a transposed matrix has the
// right byte count but the wrong shape).
INK2_API int ink2_vision_set_tensor_shaped(void* h, const char* name, const char* dtype, const long long* shape, int ndim,
                                           const void* bf16, long long nbytes);
// All tensors present: allocates the feature buffer and the scratch of one image (the weights are on the
// device from the first ink2_vision_set_tensor on). The recorded kernel sequences of the text model are dropped
// once here and recorded again on their next use. ink2_set_option(h, "vision_capture", 1) before this call keeps
// stages 1 .. 3 of every encode for ink2_vision_stage (tests; three extra float32 buffers of [S, 1024]).
INK2_API int ink2_vision_finalize(void* h);
// Encodes one image (RGB bytes, row-major, width * height * 3 of them) into the feature rows
// [row0, row0 + rows * cols); rows = ceil(height / 40), cols = ceil(width / 40). Allocates nothing. Pageable rgb
// memory is read before the call returns; page-locked memory must stay unchanged until the next synchronisation.
INK2_API int ink2_vision_encode(void* h, const unsigned char* rgb, int width, int height, long long rgb_bytes, int rows,
                                int cols, int row0);
// Uploads n image vectors (BF16 [n, hidden], nbytes = n * hidden * 2) into the feature rows [row0, row0 + n).
INK2_API int ink2_vision_set_rows(void* h, const unsigned short* bf16, int row0, int n, long long nbytes);
// Reads n feature rows as float32 [n, hidden] (out_floats >= n * hidden). Only rows written since the last reset.
INK2_API int ink2_vision_get_rows(void* h, float* out, int row0, int n, long long out_floats);
// One intermediate result of the last encode, float32 (for tests): 0 pixel values, 1 patch projection,
// 2 embeddings, 3 after block 1, 4 after the last block, 5 merged, 6 output (before the BF16 rounding).
// Stages 1 .. 3 only with the option "vision_capture" (see ink2_vision_finalize).
// Sizes (N image vectors, S = 1 + 4 N encoder rows): 0 [4 N, 1200], 1 [4 N, 1024], 2 .. 4 [S, 1024], 5 [N, 4096],
// 6 [N, text_hidden]; out_floats must hold the whole stage.
INK2_API int ink2_vision_stage(void* h, int stage, float* out, long long out_floats);
// Bytes held on the device: weights of the tower, scratch of one image, feature buffer.
INK2_API int ink2_vision_memory(void* h, long long* weights, long long* scratch, long long* features);
// Frees the image tower and the feature buffer (the text model stays loaded). ink2_vision_create may follow.
INK2_API int ink2_vision_unload(void* h);
// As ink2_feed_many, with img_row[i] = -1 for a text position, else the feature row of an image position
// (ids[i] must then be INK2_IMAGE_TOKEN, the row must have been written by ink2_vision_encode or
// ink2_vision_set_rows since the last ink2_reset, and the image rows of one call must increase).
// img_row may be NULL (all text: the same as ink2_feed_many).
INK2_API int ink2_feed_many_img(void* h, const int* ids, const int* img_row, int n);
