// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Linux build without the CPU emulation (ink2_emu.cpp uses Windows fibers).
// The "host" backend and ink2_selftest are not available; the "cuda" backend,
// which runs the model, does not use them.
#include <stdexcept>
#include "ink2_kargs.h"

Backend* ink2_make_host_backend() {
    throw std::runtime_error("host backend (CPU emulation) is Windows-only; "
                             "this Linux build has the cuda backend only");
}
