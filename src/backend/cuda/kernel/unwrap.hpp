// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <kernel/config.hpp>
#include <nvrtc_kernel_headers/unwrap_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

template<typename T>
void unwrap(Param<T> out, CParam<T> in, const int wx, const int wy,
            const int sx, const int sy, const int px, const int py,
            const int dx, const int dy, const int nx, const bool is_column) {
    auto unwrap = common::getKernel(
        "flare::cuda::unwrap", {{unwrap_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(is_column)));

    dim3 threads, blocks;
    int reps;

    if (is_column) {
        int TX = std::min(THREADS_PER_BLOCK, nextpow2(out.dims[0]));

        threads = dim3(TX, THREADS_PER_BLOCK / TX);
        blocks = dim3(divup(out.dims[1], threads.y), out.dims[2] * out.dims[3]);
        reps   = divup((wx * wy),
                       threads.x);  // is > 1 only when TX == 256 && wx * wy > 256
    } else {
        threads = dim3(THREADS_X, THREADS_Y);
        blocks = dim3(divup(out.dims[0], threads.x), out.dims[2] * out.dims[3]);

        reps = divup((wx * wy), threads.y);
    }

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    unwrap(qArgs, out, in, wx, wy, sx, sy, px, py, dx, dy, nx, reps);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
