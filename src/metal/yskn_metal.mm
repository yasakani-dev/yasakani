// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#include "yskn_metal.h"
#include "kernels.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using json = nlohmann::json;
thread_local std::string error;
struct Params {
    uint32_t rows, cols, bits, warp, parts, row;
    uint32_t stream_offset, lut_offset, scale_offset, batch;
};
struct Rotation { uint64_t seed, offset; bool splitmix; };
struct Region { id<MTLBuffer> buffer = nil; uint32_t offset = 0; size_t base_offset = 0; };
struct Tensor {
    std::string name;
    uint32_t rows = 0, physical_rows = 0, cols = 0, bits = 0;
    bool warp = false, output_rotation = false;
    uint64_t stream_off = 0, stream_bytes = 0, lut_off = 0, scale_off = 0;
    Rotation in{}, out{};
    std::vector<float> in_signs, out_signs;
    Region stream, lut, scale;
};
uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, sizeof v); return v; }
uint32_t positive(const json& j, const char* key) {
    int64_t n = j.at(key).get<int64_t>();
    if (n <= 0 || n > std::numeric_limits<int>::max()) throw std::runtime_error(std::string("invalid dimension: ") + key);
    return static_cast<uint32_t>(n);
}
Rotation rotation(const json& j, uint64_t offset) {
    json r = j.is_null() ? json{{"kind", "H512"}, {"sign_seed", 7}, {"sign_offset", offset}, {"rule", "mulberry32_7919_13"}} : j;
    if (r.at("kind") != "H512" || r.value("qm_off", uint64_t(0))) throw std::runtime_error("Metal supports H512 rotations only");
    std::string rule = r.value("rule", std::string("splitmix64"));
    if (rule != "splitmix64" && rule != "mulberry32_7919_13") throw std::runtime_error("unknown sign rule");
    return {r.at("sign_seed").get<uint64_t>(), r.at("sign_offset").get<uint64_t>(), rule == "splitmix64"};
}
std::vector<float> signs(Rotation r, uint32_t n) {
    std::vector<float> result(n);
    uint32_t s0 = uint32_t(r.seed) * 7919u + 13u;
    if (!s0) s0 = 0x9e3779b9u;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t k = r.offset + i + 1;
        if (r.splitmix) {
            uint64_t z = r.seed + k * 0x9E3779B97F4A7C15ull;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            result[i] = (z >> 63) ? 1.f : -1.f;
        } else {
            uint32_t t = s0 + uint32_t(k) * 0x6D2B79F5u;
            t = (t ^ (t >> 15)) * (t | 1u);
            t ^= t + (t ^ (t >> 7)) * (t | 61u);
            t ^= t >> 14;
            result[i] = t < 0x80000000u ? -1.f : 1.f;
        }
    }
    return result;
}
void hadamard(float* data, uint32_t n) {
    constexpr float norm = 0.044194173824159216f;
    for (uint32_t base = 0; base < n; base += 512) {
        for (uint32_t width = 1; width < 512; width *= 2)
            for (uint32_t block = 0; block < 512; block += 2 * width)
                for (uint32_t j = 0; j < width; ++j) {
                    float a = data[base + block + j], b = data[base + block + j + width];
                    data[base + block + j] = a + b;
                    data[base + block + j + width] = a - b;
                }
        for (uint32_t j = 0; j < 512; ++j) data[base + j] *= norm;
    }
}
std::string ns_error(NSError* e) { return e ? e.localizedDescription.UTF8String : "unknown Metal error"; }
struct Model {
    // Destroy buffers before unmapping their backing storage (explicitly done below).
    uint8_t* bytes = nullptr;
    size_t length = 0, mapped_length = 0, page = 0;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> mv = nil, mv6 = nil, mv8 = nil, reduce = nil, row = nil;
    id<MTLComputePipelineState> batch6[3] = {}, batch8[3] = {};
    id<MTLBuffer> input = nil, partial = nil, output = nil, shared_map = nil;
    std::unordered_map<std::string, Tensor> tensors;
    uint64_t embedding_offset = 0, embedding_count = 0;
    uint32_t embedding_width = 0;
    bool embedding_bf16 = false;
    std::mutex mutex;
    ~Model() { tensors.clear(); input = nil; partial = nil; output = nil; shared_map = nil; if (bytes) munmap(bytes, mapped_length); }
    void bounds(uint64_t off, uint64_t count) const {
        if (off > length || count > length - off) throw std::runtime_error("YSKN data runs past end of file");
    }
    Region region(uint64_t off, uint64_t count) {
        bounds(off, count);
        size_t start = size_t(off) / page * page;
        if (shared_map) return {shared_map, uint32_t(off - start), start};
        size_t end = (size_t(off + count) + page - 1) / page * page;
        size_t size = end - start;
        if (!size || size > device.maxBufferLength) throw std::runtime_error("tensor exceeds this Metal device's maxBufferLength");
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:bytes + start length:size options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffer) throw std::runtime_error("Metal could not map the compressed model data");
        return {buffer, uint32_t(off - start)};
    }
    void load(const char* path) {
        if (!path || !*path) throw std::runtime_error("model path is empty");
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open model file");
        struct stat st{};
        if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 32) { ::close(fd); throw std::runtime_error("invalid model file"); }
        length = size_t(st.st_size);
        page = size_t(sysconf(_SC_PAGESIZE));
        if (!page || length > std::numeric_limits<size_t>::max() - page) { ::close(fd); throw std::runtime_error("invalid file size"); }
        mapped_length = (length + page - 1) / page * page;
        // Private writable virtual memory meets Metal's storage requirements. Kernels only read
        // these buffers; MAP_PRIVATE ensures the model file is never written.
        void* mapping = mmap(nullptr, mapped_length, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mapping == MAP_FAILED) throw std::runtime_error("cannot map model file");
        bytes = static_cast<uint8_t*>(mapping);
        uint32_t version; std::memcpy(&version, bytes + 4, 4);
        if (std::memcmp(bytes, "YSKN", 4) || version != 1) throw std::runtime_error("not a YSKN v1 file");
        uint64_t json_length = u64(bytes + 8);
        bounds(32, json_length);
        if (json_length > 64 * 1024 * 1024) throw std::runtime_error("YSKN header exceeds 64 MiB");
        if (u64(bytes + 24) != length) throw std::runtime_error("YSKN declared file length differs from file");
        json head = json::parse(bytes + 32, bytes + 32 + json_length);
        for (const auto& t : head.at("tensors")) {
            Tensor d;
            d.name = t.at("name").get<std::string>();
            d.rows = positive(t, "rows"); d.cols = positive(t, "cols");
            d.physical_rows = t.contains("physical_rows") ? positive(t, "physical_rows") : d.rows;
            d.bits = positive(t, "bits");
            if (d.rows > d.physical_rows || d.cols % 512 || d.bits < 4 || d.bits > 8 ||
                t.value("l", 12) != 12 || t.value("d", 2) != 2 || t.value("t", 256) != 256 ||
                t.value("start", std::string("zero")) != "zero" || (t.contains("rar") && !t["rar"].is_null()))
                throw std::runtime_error("unsupported Metal tensor: " + d.name);
            std::string layout = t.value("layout", std::string("warp"));
            if (layout != "row" && layout != "warp") throw std::runtime_error("unknown stream layout");
            d.warp = layout == "warp";
            uint64_t sequences = uint64_t(d.physical_rows) * (d.cols / 256);
            if (sequences > UINT32_MAX || (d.warp && sequences % 32)) throw std::runtime_error("unsupported sequence count");
            d.stream_off = t.at("stream_off").get<uint64_t>();
            d.stream_bytes = t.at("stream_bytes").get<uint64_t>();
            d.lut_off = t.at("lut_off").get<uint64_t>(); d.scale_off = t.at("scale_off").get<uint64_t>();
            if (d.stream_bytes != sequences * 16 * d.bits) throw std::runtime_error("incorrect stream size: " + d.name);
            bounds(d.stream_off, d.stream_bytes); bounds(d.lut_off, 16384); bounds(d.scale_off, uint64_t(d.physical_rows) * 2);
            d.in = rotation(t.value("rot_in", json()), 0);
            std::string role = t.at("role").get<std::string>();
            d.output_rotation = t.contains("rot_out") ? (!t["rot_out"].is_null() && t["rot_out"].value("kind", std::string()) != "none")
                                                       : (role != "head" && role != "embedding");
            if (d.output_rotation) {
                if (d.physical_rows % 512) throw std::runtime_error("output rotation requires 512-row blocks");
                d.out = rotation(t.value("rot_out", json()), d.cols);
            }
            if (!tensors.emplace(d.name, std::move(d)).second) throw std::runtime_error("duplicate tensor name");
        }
        if (!tensors.count("embed_tokens")) {
            for (const auto& v : head.at("vectors")) {
                if (v.at("name") != "embed_tokens") continue;
                if (embedding_count) throw std::runtime_error("duplicate embedding vector");
                embedding_width = positive(head.at("model"), "hidden");
                embedding_count = v.at("n").get<uint64_t>();
                embedding_offset = v.at("off").get<uint64_t>();
                const std::string dtype = v.value("dtype", std::string("f16"));
                if (!embedding_count || embedding_count % embedding_width ||
                    embedding_count > UINT64_MAX / 2 || (dtype != "f16" && dtype != "bf16"))
                    throw std::runtime_error("invalid embedding vector");
                embedding_bf16 = dtype == "bf16";
                bounds(embedding_offset, embedding_count * 2);
            }
        }
        device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) throw std::runtime_error("an Apple Silicon Mac with a Metal device is required");
        // Reuse one no-copy resource when the complete file fits. Bind a page-aligned
        // base so shader-relative offsets stay uint32 even for files above 4 GiB.
        // A failed allocation is reported; only the size limit selects per-region maps.
        if (mapped_length <= device.maxBufferLength) {
            shared_map = [device newBufferWithBytesNoCopy:bytes length:mapped_length options:MTLResourceStorageModeShared deallocator:nil];
            if (!shared_map) throw std::runtime_error("Metal could not map the complete compressed model");
        }
        queue = [device newCommandQueue];
        NSError* failure = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:yskn_metal_source] options:options error:&failure];
        if (!library) throw std::runtime_error("Metal shader compilation failed: " + ns_error(failure));
        auto pipeline = [&](NSString* name) {
            id<MTLFunction> fn = [library newFunctionWithName:name];
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&failure];
            if (!p) throw std::runtime_error("Metal pipeline creation failed: " + ns_error(failure));
            return p;
        };
        mv = pipeline(@"compressed_mv"); mv6 = pipeline(@"compressed_mv6"); mv8 = pipeline(@"compressed_mv8");
        batch6[0] = pipeline(@"compressed_mv6_b2"); batch6[1] = pipeline(@"compressed_mv6_b4"); batch6[2] = pipeline(@"compressed_mv6_b8");
        batch8[0] = pipeline(@"compressed_mv8_b2"); batch8[1] = pipeline(@"compressed_mv8_b4"); batch8[2] = pipeline(@"compressed_mv8_b8");
        reduce = pipeline(@"reduce_mv"); row = pipeline(@"compressed_row");
        if (!queue) throw std::runtime_error("Metal command queue creation failed");
    }
    Tensor& get(const char* name) {
        if (!name) throw std::runtime_error("tensor name is null");
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error(std::string("tensor not found: ") + name);
        Tensor& t = it->second;
        if (!t.stream.buffer) {
            // Prepare everything locally: any allocation failure must leave this tensor
            // uninitialized and safe to retry. stream is the readiness marker, set last.
            Region stream = region(t.stream_off, t.stream_bytes);
            Region lut = region(t.lut_off, 16384);
            Region scale = region(t.scale_off, uint64_t(t.physical_rows) * 2);
            std::vector<float> in_signs = signs(t.in, t.cols);
            std::vector<float> out_signs;
            if (t.output_rotation) out_signs = signs(t.out, t.physical_rows);
            t.in_signs = std::move(in_signs); t.out_signs = std::move(out_signs);
            t.lut = std::move(lut); t.scale = std::move(scale);
            t.stream = std::move(stream);
        }
        return t;
    }
    void buffer(id<MTLBuffer> __strong& b, size_t size) {
        if (size > device.maxBufferLength) throw std::runtime_error("Metal scratch exceeds device maxBufferLength");
        if (!b || b.length < size) {
            id<MTLBuffer> replacement = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
            if (!replacement) throw std::runtime_error("Metal scratch allocation failed");
            b = replacement;
        }
    }
    Params params(const Tensor& t) {
        return {t.physical_rows, t.cols, t.bits, uint32_t(t.warp), std::min(uint32_t(8), t.cols / 256), 0,
                t.stream.offset, t.lut.offset, t.scale.offset, 1};
    }
    id<MTLCommandBuffer> command() {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) throw std::runtime_error("Metal command buffer allocation failed");
        return cmd;
    }
    id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> cmd) {
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        if (!e) throw std::runtime_error("Metal command encoder allocation failed");
        return e;
    }
    void dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, size_t count) {
        [e setComputePipelineState:p];
        NSUInteger width = std::min(NSUInteger(128), p.maxTotalThreadsPerThreadgroup);
        [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [e endEncoding];
    }
    void wait(id<MTLCommandBuffer> cmd) {
        [cmd commit]; [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("Metal execution failed: " + ns_error(cmd.error));
    }
    bool fast_tensor(const Tensor& t) const {
        return t.warp && (t.bits == 6 || t.bits == 8) && !(t.stream.offset % 4) && !(t.lut.offset % 4) && !(t.scale.offset % 2);
    }
    void encode_linear(id<MTLCommandBuffer> cmd, const Tensor& t, const Params& p,
                       size_t input_offset, size_t partial_offset, size_t output_offset) {
        id<MTLComputeCommandEncoder> e = encoder(cmd);
        [e setBuffer:t.stream.buffer offset:t.stream.base_offset atIndex:0]; [e setBuffer:t.lut.buffer offset:t.lut.base_offset atIndex:1];
        [e setBuffer:t.scale.buffer offset:t.scale.base_offset atIndex:2]; [e setBuffer:input offset:input_offset atIndex:3];
        [e setBuffer:partial offset:partial_offset atIndex:4]; [e setBytes:&p length:sizeof(p) atIndex:5];
        id<MTLComputePipelineState> kernel = mv;
        if (fast_tensor(t)) {
            if (p.batch == 1) kernel = t.bits == 6 ? mv6 : mv8;
            else {
                int tile = p.batch <= 2 ? 0 : (p.batch <= 4 ? 1 : 2);
                kernel = t.bits == 6 ? batch6[tile] : batch8[tile];
            }
        }
        dispatch(e, kernel, size_t(t.physical_rows) * p.parts);
        e = encoder(cmd);
        [e setBuffer:partial offset:partial_offset atIndex:0]; [e setBuffer:output offset:output_offset atIndex:1]; [e setBytes:&p length:sizeof(p) atIndex:2];
        dispatch(e, reduce, size_t(p.batch) * t.physical_rows);
    }
    void linear_tile(Tensor& t, const float* x, int batch, int n, float* y, int m) {
        Params p = params(t); p.batch = uint32_t(batch);
        if (size_t(batch) * t.physical_rows * p.parts > UINT32_MAX)
            throw std::runtime_error("Metal batch scratch exceeds shader indexing range");
        buffer(input, size_t(batch) * n * 4); buffer(output, size_t(batch) * t.physical_rows * 4);
        buffer(partial, size_t(batch) * t.physical_rows * p.parts * 4);
        float* in = static_cast<float*>(input.contents);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) in[size_t(b) * n + i] = x[size_t(b) * n + i] * t.in_signs[i];
            hadamard(in + size_t(b) * n, t.cols);
        }
        id<MTLCommandBuffer> cmd = command();
        encode_linear(cmd, t, p, 0, 0, 0);
        wait(cmd);
        float* out = static_cast<float*>(output.contents);
        for (int b = 0; b < batch; ++b) {
            float* row_out = out + size_t(b) * t.physical_rows;
            if (t.output_rotation) {
                hadamard(row_out, t.physical_rows);
                for (uint32_t i = 0; i < t.physical_rows; ++i) row_out[i] *= t.out_signs[i];
            }
            std::memcpy(y + size_t(b) * m, row_out, size_t(m) * 4);
        }
    }
    void linear(const char* name, const float* x, int n, float* y, int m) {
        Tensor& t = get(name);
        if (!x || !y || n != int(t.cols) || m != int(t.rows)) throw std::runtime_error("linear input/output shape mismatch");
        linear_tile(t, x, 1, n, y, m);
    }
    void linear_batch(const char* name, const float* x, int batch, int n, float* y, int m) {
        Tensor& t = get(name);
        if (!x || !y || batch < 1 || n != int(t.cols) || m != int(t.rows))
            throw std::runtime_error("linear_batch input/output shape mismatch");
        const int tile = fast_tensor(t) ? 8 : 1;
        for (int first = 0; first < batch;) {
            int count = std::min(tile, batch - first);
            linear_tile(t, x + size_t(first) * n, count, n, y + size_t(first) * m, m);
            first += count;
        }
    }
    void linear_multi(const char* const* names, int count, const float* x, int n, float* y, int m) {
        if (!names || !x || !y || count < 1 || count > 3 || n < 1 || m < 1)
            throw std::runtime_error("linear_multi expects 1 to 3 names and nonempty input/output");
        struct Projection {
            Tensor* tensor = nullptr;
            Params parameters{};
            size_t input_offset = 0, partial_offset = 0, output_offset = 0, result_offset = 0;
        };
        std::array<Projection, 3> projections;
        size_t input_bytes = 0, partial_bytes = 0, output_bytes = 0;
        int logical_rows = 0;
        auto reserve = [](size_t& total, size_t elements) {
            if (elements > (std::numeric_limits<size_t>::max() - total) / sizeof(float))
                throw std::runtime_error("linear_multi scratch size overflow");
            size_t offset = total;
            total += elements * sizeof(float);
            return offset;
        };
        // Validate every projection and reserve disjoint ranges before encoding or
        // submitting any GPU work. A later invalid name/shape cannot partly run.
        for (int i = 0; i < count; ++i) {
            Tensor& t = get(names[i]);
            if (n != int(t.cols) || t.rows > uint32_t(std::numeric_limits<int>::max() - logical_rows))
                throw std::runtime_error("linear_multi input shape or combined output size mismatch");
            Params p = params(t);
            const size_t partial_elements = size_t(t.physical_rows) * p.parts;
            if (partial_elements > UINT32_MAX)
                throw std::runtime_error("linear_multi scratch exceeds shader indexing range");
            Projection& projection = projections[i];
            projection.tensor = &t; projection.parameters = p;
            projection.input_offset = reserve(input_bytes, t.cols);
            projection.partial_offset = reserve(partial_bytes, partial_elements);
            projection.output_offset = reserve(output_bytes, t.physical_rows);
            projection.result_offset = size_t(logical_rows);
            logical_rows += int(t.rows);
        }
        if (m != logical_rows) throw std::runtime_error("linear_multi combined output shape mismatch");
        if (input_bytes > device.maxBufferLength || partial_bytes > device.maxBufferLength || output_bytes > device.maxBufferLength)
            throw std::runtime_error("linear_multi scratch exceeds device maxBufferLength");
        buffer(input, input_bytes); buffer(partial, partial_bytes); buffer(output, output_bytes);
        for (int i = 0; i < count; ++i) {
            const auto& projection = projections[i];
            const Tensor& t = *projection.tensor;
            float* in = reinterpret_cast<float*>(static_cast<uint8_t*>(input.contents) + projection.input_offset);
            // Rotations may have different signs even when the input is shared.
            for (int j = 0; j < n; ++j) in[j] = x[j] * t.in_signs[j];
            hadamard(in, t.cols);
        }
        id<MTLCommandBuffer> cmd = command();
        for (int i = 0; i < count; ++i) {
            const auto& projection = projections[i];
            encode_linear(cmd, *projection.tensor, projection.parameters, projection.input_offset,
                          projection.partial_offset, projection.output_offset);
        }
        wait(cmd);
        for (int i = 0; i < count; ++i) {
            const auto& projection = projections[i];
            const Tensor& t = *projection.tensor;
            float* out = reinterpret_cast<float*>(static_cast<uint8_t*>(output.contents) + projection.output_offset);
            if (t.output_rotation) {
                hadamard(out, t.physical_rows);
                for (uint32_t j = 0; j < t.physical_rows; ++j) out[j] *= t.out_signs[j];
            }
            std::memcpy(y + projection.result_offset, out, size_t(t.rows) * sizeof(float));
        }
    }
    void embedding(int token, float* out, int n) {
        if (!embedding_count) {
            decode_row("embed_tokens", token, out, n);
            if (tensors.count("embed_tokens.res2") && !tensors.count("embed_tokens.res"))
                throw std::runtime_error("embed_tokens.res2 requires embed_tokens.res");
            for (const char* name : {"embed_tokens.res", "embed_tokens.res2"}) {
                if (tensors.count(name)) {
                    std::vector<float> extra(n); decode_row(name, token, extra.data(), n);
                    for (int i = 0; i < n; ++i) out[i] += extra[i];
                }
            }
            return;
        }
        if (!out || n != int(embedding_width) || token < 0 || uint64_t(token) >= embedding_count / embedding_width)
            throw std::runtime_error("embedding vector row/shape mismatch");
        const uint8_t* data = bytes + embedding_offset + uint64_t(token) * embedding_width * 2;
        for (int i = 0; i < n; ++i) {
            uint16_t bits; std::memcpy(&bits, data + 2 * i, 2);
            if (embedding_bf16) {
                uint32_t f = uint32_t(bits) << 16; std::memcpy(out + i, &f, 4);
            } else {
                _Float16 f; std::memcpy(&f, &bits, 2); out[i] = float(f);
            }
        }
    }
    void decode_row(const char* name, int r, float* out, int n) {
        Tensor& t = get(name);
        if (!out || n != int(t.cols) || r < 0 || r >= int(t.rows) || t.output_rotation)
            throw std::runtime_error("row requires a valid row, column count, and no output rotation");
        Params p = params(t); p.row = uint32_t(r);
        buffer(output, size_t(n) * 4);
        id<MTLCommandBuffer> cmd = command();
        id<MTLComputeCommandEncoder> e = encoder(cmd);
        [e setBuffer:t.stream.buffer offset:t.stream.base_offset atIndex:0]; [e setBuffer:t.lut.buffer offset:t.lut.base_offset atIndex:1];
        [e setBuffer:t.scale.buffer offset:t.scale.base_offset atIndex:2]; [e setBuffer:output offset:0 atIndex:3]; [e setBytes:&p length:sizeof(p) atIndex:4];
        dispatch(e, row, t.cols); wait(cmd);
        float* values = static_cast<float*>(output.contents);
        hadamard(values, t.cols);
        for (int i = 0; i < n; ++i) out[i] = values[i] * t.in_signs[i];
    }
};
Model& checked(void* handle) { if (!handle) throw std::runtime_error("Metal handle is null"); return *static_cast<Model*>(handle); }
template<class F> int call(F fn) {
    @autoreleasepool {
        try { error.clear(); fn(); return 0; }
        catch (const std::exception& ex) { error = ex.what(); }
        catch (...) { error = "unexpected Metal backend error"; }
        return -1;
    }
}
} // namespace
extern "C" void* yskn_metal_open(const char* path) {
    @autoreleasepool {
        try { error.clear(); auto model = std::make_unique<Model>(); model->load(path); return model.release(); }
        catch (const std::exception& ex) { error = ex.what(); }
        catch (...) { error = "unexpected Metal backend error"; }
        return nullptr;
    }
}
extern "C" void yskn_metal_close(void* handle) { @autoreleasepool { delete static_cast<Model*>(handle); } }
extern "C" const char* yskn_metal_last_error(void) { return error.c_str(); }
extern "C" int yskn_metal_linear(void* h, const char* name, const float* x, int n, float* y, int m) {
    return call([&] { Model& model = checked(h); std::lock_guard<std::mutex> lock(model.mutex); model.linear(name, x, n, y, m); });
}
extern "C" int yskn_metal_linear_batch(void* h, const char* name, const float* x, int batch, int n, float* y, int m) {
    return call([&] { Model& model = checked(h); std::lock_guard<std::mutex> lock(model.mutex); model.linear_batch(name, x, batch, n, y, m); });
}
extern "C" int yskn_metal_linear_multi(void* h, const char** names, int count, const float* x, int n, float* y, int m) {
    return call([&] { Model& model = checked(h); std::lock_guard<std::mutex> lock(model.mutex); model.linear_multi(names, count, x, n, y, m); });
}
extern "C" int yskn_metal_row(void* h, const char* name, int r, float* out, int n) {
    return call([&] { Model& model = checked(h); std::lock_guard<std::mutex> lock(model.mutex); model.decode_row(name, r, out, n); });
}
extern "C" int yskn_metal_embedding(void* h, int token, float* out, int n) {
    return call([&] {
        Model& model = checked(h); std::lock_guard<std::mutex> lock(model.mutex);
        model.embedding(token, out, n);
    });
}
