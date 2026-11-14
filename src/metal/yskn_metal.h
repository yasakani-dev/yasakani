// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// Handles are owned by the caller. Operations on one handle are serialized.
// The model file must not be modified or removed while its handle is open.
void* yskn_metal_open(const char* path);
void yskn_metal_close(void* handle);
int yskn_metal_linear(void* handle, const char* name, const float* x, int n, float* y, int m);
int yskn_metal_row(void* handle, const char* name, int row, float* out, int n);
int yskn_metal_embedding(void* handle, int token, float* out, int n);
const char* yskn_metal_last_error(void);
#ifdef __cplusplus
}
#endif
