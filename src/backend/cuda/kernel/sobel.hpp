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
#include <nvrtc_kernel_headers/sobel_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename Ti, typename To>
void sobel(Param<To> dx, Param<To> dy, CParam<Ti> in,
           const unsigned& ker_size) {
    UNUSED(ker_size);

    auto sobel3x3 = common::getKernel(
        "flare::cuda::sobel3x3", {{sobel_cuh_src}},
        TemplateArgs(TemplateTypename<Ti>(), TemplateTypename<To>()),
        {{DefineValue(THREADS_X), DefineValue(THREADS_Y)}});

    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(in.dims[0], threads.x);
    int blk_y = divup(in.dims[1], threads.y);

    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    // TODO: call other cases when support for 5x5 & 7x7 is added
    // Note: This is checked at sobel API entry point
    sobel3x3(qArgs, dx, dy, in, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
