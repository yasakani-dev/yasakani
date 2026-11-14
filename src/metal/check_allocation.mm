// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Standalone fault-injection regression. Compile this translation unit alone:
// clang++ -std=c++17 -fobjc-arc -Ithird_party src/metal/check_allocation.mm \
//   -framework Foundation -framework Metal -o build/check_metal_allocation
#import <Metal/Metal.h>
static id<MTLDevice> selected_device = nil;
static id<MTLDevice> allocation_test_device() {
    return selected_device ? selected_device : MTLCreateSystemDefaultDevice();
}
// Override device selection only in this standalone test translation unit.
#define MTLCreateSystemDefaultDevice allocation_test_device
#include "yskn_metal.mm"
#undef MTLCreateSystemDefaultDevice
#include <iostream>

// Wrap the real device to inject memory pressure and a smaller buffer limit.
// No fault injection hooks are compiled into the distributed library.
@interface FailingDevice : NSObject
@property(strong) id<MTLDevice> realDevice;
@property NSUInteger calls;
@property NSUInteger failAt;
@property NSUInteger forcedMaxBufferLength;
@property NSUInteger forcedThreadgroupMemoryLength;
@end
@implementation FailingDevice
- (id)forwardingTargetForSelector:(SEL)selector { return self.realDevice; }
- (NSUInteger)maxBufferLength {
    return self.forcedMaxBufferLength ? self.forcedMaxBufferLength : self.realDevice.maxBufferLength;
}
- (NSUInteger)maxThreadgroupMemoryLength {
    return self.forcedThreadgroupMemoryLength ? self.forcedThreadgroupMemoryLength : self.realDevice.maxThreadgroupMemoryLength;
}
- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void*)pointer length:(NSUInteger)length
                                options:(MTLResourceOptions)options
                            deallocator:(void (^)(void*, NSUInteger))deallocator {
    self.calls += 1;
    if (self.calls == self.failAt) return nil;
    return [self.realDevice newBufferWithBytesNoCopy:pointer length:length options:options deallocator:deallocator];
}
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {
    self.calls += 1;
    if (self.calls == self.failAt) return nil;
    return [self.realDevice newBufferWithLength:length options:options];
}
@end

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 2) throw std::runtime_error("usage: check_metal_allocation tools/testdata/tiny.yskn");
            id<MTLDevice> real = MTLCreateSystemDefaultDevice();
            require(real && real.hasUnifiedMemory, "an accessible Apple Silicon Metal device is required");
            FailingDevice* failing = [FailingDevice new];
            failing.realDevice = real; failing.failAt = 1;
            selected_device = (id<MTLDevice>)failing;
            std::unique_ptr<Model> shared(static_cast<Model*>(yskn_metal_open(argv[1])));
            require(!shared && failing.calls == 1, "shared mapping failure silently fell back or succeeded");
            require(std::string(yskn_metal_last_error()).find("complete compressed model") != std::string::npos,
                    "shared mapping failure did not cross the C ABI as an error");
            // Retrying open creates a fresh handle; failed handles never escape the ABI.
            shared.reset(static_cast<Model*>(yskn_metal_open(argv[1])));
            require(shared && shared->shared_map && failing.calls == 2, "shared mapping open was not retryable");
            std::vector<float> x(512, 1.f), expected(512), y(512);
            require(yskn_metal_linear(shared.get(), "tiny.body6", x.data(), 512, expected.data(), 512) == 0,
                    "shared mapping retry could not execute linear");
            for (float v : expected) require(std::isfinite(v), "shared mapping retry produced nonfinite output");
            const Tensor& shared_tensor = shared->tensors.at("tiny.body6");
            for (const auto& pair : {std::make_pair(&shared_tensor.stream, shared_tensor.stream_off),
                                     std::make_pair(&shared_tensor.lut, shared_tensor.lut_off),
                                     std::make_pair(&shared_tensor.scale, shared_tensor.scale_off)}) {
                require(pair.first->buffer == shared->shared_map && pair.first->base_offset % shared->page == 0 &&
                        pair.first->offset < shared->page && pair.first->base_offset + pair.first->offset == pair.second,
                        "shared mapping region has an invalid buffer or offset");
            }
            require(shared->mapped_length > shared->page, "fixture is too small to force per-region fallback");
            const size_t forced_limit = shared->mapped_length - shared->page;
            shared.reset();

            failing.forcedMaxBufferLength = forced_limit;
            failing.forcedThreadgroupMemoryLength = 1;
            failing.calls = 0; failing.failAt = 2;
            std::unique_ptr<Model> fallback(static_cast<Model*>(yskn_metal_open(argv[1])));
            require(fallback && !fallback->shared_map && failing.calls == 0,
                    "device buffer limit did not select lazy per-region fallback");
            require(!fallback->tg6 && !fallback->tg8, "limited threadgroup memory did not select original fast kernels");
            Model& model = *fallback;
            bool rejected = false;
            try { model.get("tiny.body6"); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected && failing.calls == 2, "second tensor allocation did not fail");
            const Tensor& t = model.tensors.at("tiny.body6");
            require(!t.stream.buffer && !t.lut.buffer && !t.scale.buffer && t.in_signs.empty() && t.out_signs.empty(),
                    "failed initialization left a partially ready tensor");
            failing.failAt = 0;
            require(yskn_metal_linear(fallback.get(), "tiny.body6", x.data(), 512, y.data(), 512) == 0,
                    "per-region initialization retry could not execute linear");
            require(!model.shared_map && failing.calls >= 5, "fallback retry bypassed per-region allocations");
            require(std::memcmp(expected.data(), y.data(), y.size() * sizeof(float)) == 0,
                    "fallback retry differs from shared mapping output");
            NSUInteger old_length = model.input.length;
            failing.calls = 0; failing.failAt = 1;
            rejected = false;
            try { model.buffer(model.input, old_length * 2); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected && failing.calls == 1 && model.input.length == old_length,
                    "failed scratch growth lost the previous buffer");
            failing.failAt = 0;
            require(yskn_metal_linear(fallback.get(), "tiny.body6", x.data(), 512, y.data(), 512) == 0,
                    "scratch allocation failure prevented a later linear call");
            require(std::memcmp(expected.data(), y.data(), y.size() * sizeof(float)) == 0,
                    "scratch growth failure changed a later result");
            selected_device = nil;
            std::cout << "PASS: shared mapping open, per-region initialization, and scratch growth failures remain retryable\n";
            return 0;
        } catch (const std::exception& ex) {
            std::cerr << "FAIL: " << ex.what() << '\n';
            return 1;
        }
    }
}
