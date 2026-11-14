// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
// Embedded so a prebuilt dylib does not need a compiler installation or shader files.
static const char* yskn_metal_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params {
    uint rows, cols, bits, warp, parts, row;
    uint stream_offset, lut_offset, scale_offset;
};
uint word(device const uchar* stream, constant Params& p, uint r, uint seq, uint w) {
    if (w >= 4 * p.bits) return 0;
    uint q = seq * p.rows + r;
    ulong i = p.warp ? (ulong(q / 32) * p.bits + w / 4) * 128 + (q % 32) * 4 + w % 4
                     : (ulong(r) * (p.cols / 256) + seq) * (4 * p.bits) + w;
    ulong b = p.stream_offset + 4 * i;
    return uint(stream[b]) | (uint(stream[b+1]) << 8) | (uint(stream[b+2]) << 16) | (uint(stream[b+3]) << 24);
}
float fp16(device const uchar* data, ulong at) {
    ushort b = ushort(data[at]) | (ushort(data[at+1]) << 8);
    return float(as_type<half>(b));
}
uint state(device const uchar* stream, constant Params& p, uint r, uint seq, uint j) {
    int start = int(p.bits * (j + 1)) - 12;
    if (start < 0) return (word(stream, p, r, seq, 0) << uint(-start)) & 4095;
    uint w = uint(start) / 32, sh = uint(start) % 32;
    uint lo = word(stream, p, r, seq, w) >> sh;
    if (sh > 20) lo |= word(stream, p, r, seq, w + 1) << (32 - sh);
    return lo & 4095;
}
kernel void compressed_mv(device const uchar* stream [[buffer(0)]],
                          device const uchar* lut [[buffer(1)]],
                          device const uchar* scale [[buffer(2)]],
                          device const float* x [[buffer(3)]],
                          device float* partial [[buffer(4)]],
                          constant Params& p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    uint r = i % p.rows, part = i / p.rows;
    if (part >= p.parts) return;
    float sum = 0;
    for (uint seq = part; seq < p.cols / 256; seq += p.parts) {
        for (uint j = 0; j < 128; ++j) {
            uint st = state(stream, p, r, seq, j);
            sum += fp16(lut, p.lut_offset + 4 * st) * x[256 * seq + 2 * j];
            sum += fp16(lut, p.lut_offset + 4 * st + 2) * x[256 * seq + 2 * j + 1];
        }
    }
    partial[i] = sum * fp16(scale, p.scale_offset + 2 * r);
}
kernel void reduce_mv(device const float* partial [[buffer(0)]], device float* out [[buffer(1)]],
                      constant Params& p [[buffer(2)]], uint r [[thread_position_in_grid]]) {
    if (r >= p.rows) return;
    float sum = 0;
    for (uint k = 0; k < p.parts; ++k) sum += partial[k * p.rows + r];
    out[r] = sum;
}
kernel void compressed_row(device const uchar* stream [[buffer(0)]], device const uchar* lut [[buffer(1)]],
                           device const uchar* scale [[buffer(2)]], device float* out [[buffer(3)]],
                           constant Params& p [[buffer(4)]], uint col [[thread_position_in_grid]]) {
    if (col >= p.cols) return;
    uint st = state(stream, p, p.row, col / 256, (col % 256) / 2);
    out[col] = fp16(lut, p.lut_offset + 4 * st + 2 * (col % 2)) * fp16(scale, p.scale_offset + 2 * p.row);
}
)MSL";
