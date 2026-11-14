// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
// Embedded so a prebuilt dylib does not need a compiler installation or shader files.
static const char* yskn_metal_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params {
    uint rows, cols, bits, warp, parts, row;
    uint stream_offset, lut_offset, scale_offset, batch;
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
float fp16(constant uchar* data, ulong at) {
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
                          constant uchar* lut [[buffer(1)]],
                          device const uchar* scale [[buffer(2)]],
                          constant float* x [[buffer(3)]],
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
// Released models use 6/8-bit warp streams and aligned sections. Read each
// packed word once per 16 symbols instead of reconstructing overlapping words
// from bytes at every step. Keep the generic kernel for other valid encodings.
template<uint Bits>
uint warp_word(device const uint* stream, uint rows, uint r, uint seq, uint w) {
    uint q = seq * rows + r;
    ulong index = (ulong(q / 32) * Bits + w / 4) * 128 + (q % 32) * 4 + w % 4;
    return stream[index];
}
// Uniform input indices permit constant-address broadcasts within a SIMD group.
// The read-only LUT address space was benchmarked; arithmetic order is unchanged.
template<uint Bits, uint Batch>
void fast_mv(device const uchar* stream_bytes, constant uchar* lut_bytes,
             device const uchar* scale_bytes, constant float2* x,
             device float* partial, constant Params& p, uint i) {
    uint r = i % p.rows, part = i / p.rows;
    if (part >= p.parts) return;
    device const uint* stream = reinterpret_cast<device const uint*>(stream_bytes + p.stream_offset);
    constant half2* lut = reinterpret_cast<constant half2*>(lut_bytes + p.lut_offset);
    device const half* scale = reinterpret_cast<device const half*>(scale_bytes + p.scale_offset);
    // Four independent accumulation chains hide dependent-add latency. For
    // prefill, reuse each decoded pair across several token vectors.
    float4 sums[Batch];
    #pragma unroll
    for (uint b = 0; b < Batch; ++b) sums[b] = float4(0);
    for (uint seq = part; seq < p.cols / 256; seq += p.parts) {
        uint previous = 0;
        for (uint chunk = 0; chunk < 8; ++chunk) {
            uint w = chunk * (Bits / 2);
            uint a = warp_word<Bits>(stream, p.rows, r, seq, w);
            uint b = warp_word<Bits>(stream, p.rows, r, seq, w + 1);
            uint c = warp_word<Bits>(stream, p.rows, r, seq, w + 2);
            uint d = Bits == 8 ? warp_word<Bits>(stream, p.rows, r, seq, w + 3) : 0;
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                int start = int(Bits * (j + 1)) - 12;
                uint state;
                if (start < 0) {
                    state = (previous >> uint(32 + start)) | (a << uint(-start));
                } else {
                    uint wi = uint(start) / 32, sh = uint(start) % 32;
                    uint lo = wi == 0 ? a : (wi == 1 ? b : (wi == 2 ? c : d));
                    uint hi = wi == 0 ? b : (wi == 1 ? c : d);
                    state = lo >> sh;
                    if (sh > 20) state |= hi << (32 - sh);
                }
                float2 value = float2(lut[state & 4095]);
                #pragma unroll
                for (uint b = 0; b < Batch; ++b) {
                    if (b < p.batch) {
                        float2 input = x[b * (p.cols / 2) + seq * 128 + chunk * 16 + j];
                        if (j & 1) {
                            sums[b].z += value.x * input.x;
                            sums[b].w += value.y * input.y;
                        } else {
                            sums[b].x += value.x * input.x;
                            sums[b].y += value.y * input.y;
                        }
                    }
                }
            }
            previous = Bits == 8 ? d : c;
        }
    }
    #pragma unroll
    for (uint b = 0; b < Batch; ++b) {
        if (b < p.batch) {
            float sum = (sums[b].x + sums[b].y) + (sums[b].z + sums[b].w);
            partial[b * (p.rows * p.parts) + i] = sum * float(scale[r]);
        }
    }
}
// Scalar LUT cache variant: arithmetic and accumulation order match fast_mv.
template<uint Bits, uint Batch>
void cached_mv(device const uchar* stream_bytes, constant uchar* lut_bytes,
             device const uchar* scale_bytes, constant float2* x,
             device float* partial, constant Params& p, uint i, threadgroup half2* cached_lut) {
    uint r = i % p.rows, part = i / p.rows;
    if (part >= p.parts) return;
    device const uint* stream = reinterpret_cast<device const uint*>(stream_bytes + p.stream_offset);
    threadgroup const half2* lut = cached_lut;
    device const half* scale = reinterpret_cast<device const half*>(scale_bytes + p.scale_offset);
    // Four independent accumulation chains hide dependent-add latency. For
    // prefill, reuse each decoded pair across several token vectors.
    float4 sums[Batch];
    #pragma unroll
    for (uint b = 0; b < Batch; ++b) sums[b] = float4(0);
    for (uint seq = part; seq < p.cols / 256; seq += p.parts) {
        uint previous = 0;
        for (uint chunk = 0; chunk < 8; ++chunk) {
            uint w = chunk * (Bits / 2);
            uint a = warp_word<Bits>(stream, p.rows, r, seq, w);
            uint b = warp_word<Bits>(stream, p.rows, r, seq, w + 1);
            uint c = warp_word<Bits>(stream, p.rows, r, seq, w + 2);
            uint d = Bits == 8 ? warp_word<Bits>(stream, p.rows, r, seq, w + 3) : 0;
            #pragma unroll
            for (uint j = 0; j < 16; ++j) {
                int start = int(Bits * (j + 1)) - 12;
                uint state;
                if (start < 0) {
                    state = (previous >> uint(32 + start)) | (a << uint(-start));
                } else {
                    uint wi = uint(start) / 32, sh = uint(start) % 32;
                    uint lo = wi == 0 ? a : (wi == 1 ? b : (wi == 2 ? c : d));
                    uint hi = wi == 0 ? b : (wi == 1 ? c : d);
                    state = lo >> sh;
                    if (sh > 20) state |= hi << (32 - sh);
                }
                float2 value = float2(lut[state & 4095]);
                #pragma unroll
                for (uint b = 0; b < Batch; ++b) {
                    if (b < p.batch) {
                        float2 input = x[b * (p.cols / 2) + seq * 128 + chunk * 16 + j];
                        if (j & 1) {
                            sums[b].z += value.x * input.x;
                            sums[b].w += value.y * input.y;
                        } else {
                            sums[b].x += value.x * input.x;
                            sums[b].y += value.y * input.y;
                        }
                    }
                }
            }
            previous = Bits == 8 ? d : c;
        }
    }
    #pragma unroll
    for (uint b = 0; b < Batch; ++b) {
        if (b < p.batch) {
            float sum = (sums[b].x + sums[b].y) + (sums[b].z + sums[b].w);
            partial[b * (p.rows * p.parts) + i] = sum * float(scale[r]);
        }
    }
}
kernel void compressed_tg6(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                           device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                           device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                           uint i [[thread_position_in_grid]], uint local [[thread_index_in_threadgroup]]) {
    threadgroup half2 cached_lut[4096];
    constant half2* table = reinterpret_cast<constant half2*>(lut + p.lut_offset);
    // Dispatch full groups; padded lanes load and synchronize before fast_mv returns.
    for (uint entry = local; entry < 4096; entry += 128) cached_lut[entry] = table[entry];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cached_mv<6, 1>(stream, lut, scale, x, partial, p, i, cached_lut);
}
kernel void compressed_tg8(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                           device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                           device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                           uint i [[thread_position_in_grid]], uint local [[thread_index_in_threadgroup]]) {
    threadgroup half2 cached_lut[4096];
    constant half2* table = reinterpret_cast<constant half2*>(lut + p.lut_offset);
    // Dispatch full groups; padded lanes load and synchronize before fast_mv returns.
    for (uint entry = local; entry < 4096; entry += 128) cached_lut[entry] = table[entry];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cached_mv<8, 1>(stream, lut, scale, x, partial, p, i, cached_lut);
}

kernel void compressed_mv6(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                           device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                           device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    fast_mv<6, 1>(stream, lut, scale, x, partial, p, i);
}
kernel void compressed_mv8(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                           device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                           device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    fast_mv<8, 1>(stream, lut, scale, x, partial, p, i);
}
kernel void compressed_mv6_b2(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                              device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                              device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    fast_mv<6, 2>(stream, lut, scale, x, partial, p, i);
}
kernel void compressed_mv6_b4(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                              device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                              device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    fast_mv<6, 4>(stream, lut, scale, x, partial, p, i);
}
kernel void compressed_mv6_b8(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                              device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                              device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    fast_mv<6, 8>(stream, lut, scale, x, partial, p, i);
}
kernel void compressed_mv8_b2(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                              device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                              device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    fast_mv<8, 2>(stream, lut, scale, x, partial, p, i);
}
kernel void compressed_mv8_b4(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                              device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                              device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    fast_mv<8, 4>(stream, lut, scale, x, partial, p, i);
}
kernel void compressed_mv8_b8(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                              device const uchar* scale [[buffer(2)]], constant float2* x [[buffer(3)]],
                              device float* partial [[buffer(4)]], constant Params& p [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    fast_mv<8, 8>(stream, lut, scale, x, partial, p, i);
}
kernel void reduce_mv(device const float* partial [[buffer(0)]], device float* out [[buffer(1)]],
                      constant Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint r = i % p.rows, b = i / p.rows;
    if (b >= p.batch) return;
    float sum = 0;
    for (uint k = 0; k < p.parts; ++k) sum += partial[(b * p.parts + k) * p.rows + r];
    out[i] = sum;
}
kernel void compressed_row(device const uchar* stream [[buffer(0)]], constant uchar* lut [[buffer(1)]],
                           device const uchar* scale [[buffer(2)]], device float* out [[buffer(3)]],
                           constant Params& p [[buffer(4)]], uint col [[thread_position_in_grid]]) {
    if (col >= p.cols) return;
    uint st = state(stream, p, p.row, col / 256, (col % 256) / 2);
    out[col] = fp16(lut, p.lut_offset + 4 * st + 2 * (col % 2)) * fp16(scale, p.scale_offset + 2 * p.row);
}
)MSL";
