// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Standalone fault-injection regression. Compile this translation unit alone:
// clang++ -std=c++17 -fobjc-arc -Ithird_party src/metal/check_allocation.mm \
//   -framework Foundation -framework Metal -o build/check_metal_allocation
#include "yskn_metal.mm"
#include <iostream>

// Wrap the real device and fail the second buffer allocation, as memory pressure
// can do. No fault injection hooks are compiled into the distributed library.
@interface FailingDevice : NSObject
@property(strong) id<MTLDevice> realDevice;
@property NSUInteger calls;
@property NSUInteger failAt;
@end
@implementation FailingDevice
- (NSUInteger)maxBufferLength { return self.realDevice.maxBufferLength; }
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
            Model model; model.load(argv[1]);
            id<MTLDevice> real = model.device;
            FailingDevice* failing = [FailingDevice new];
            failing.realDevice = real; failing.calls = 0; failing.failAt = 2;
            model.device = (id<MTLDevice>)failing;
            bool rejected = false;
            try { model.get("tiny.body6"); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "second tensor allocation did not fail");
            const Tensor& t = model.tensors.at("tiny.body6");
            require(!t.stream.buffer && !t.lut.buffer && !t.scale.buffer && t.in_signs.empty() && t.out_signs.empty(),
                    "failed initialization left a partially ready tensor");
            model.device = real;
            std::vector<float> x(512, 1.f), y(512);
            model.linear("tiny.body6", x.data(), 512, y.data(), 512);
            for (float v : y) require(std::isfinite(v), "retry did not produce finite output");
            NSUInteger old_length = model.input.length;
            failing.calls = 0; failing.failAt = 1; model.device = (id<MTLDevice>)failing;
            rejected = false;
            try { model.buffer(model.input, old_length * 2); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected && model.input.length == old_length, "failed scratch growth lost the previous buffer");
            model.device = real;
            model.linear("tiny.body6", x.data(), 512, y.data(), 512);
            std::cout << "PASS: failed tensor initialization and scratch growth both remain retryable\n";
            return 0;
        } catch (const std::exception& ex) {
            std::cerr << "FAIL: " << ex.what() << '\n';
            return 1;
        }
    }
}
