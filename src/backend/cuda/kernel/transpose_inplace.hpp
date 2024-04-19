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
#include <nvrtc_kernel_headers/transpose_inplace_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

static const int TILE_DIM  = 32;
static const int THREADS_X = TILE_DIM;
static const int THREADS_Y = 256 / TILE_DIM;

template<typename T>
void transpose_inplace(Param<T> in, const bool conjugate,
                       const bool is32multiple) {
    auto transposeIP = common::getKernel(
        "flare::cuda::transposeIP", {{transpose_inplace_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(conjugate),
                     TemplateArg(is32multiple)),
        {{DefineValue(TILE_DIM), DefineValue(THREADS_Y)}});

    // dimensions passed to this function should be input dimensions
    // any necessary transformations and dimension related calculations are
    // carried out here and inside the kernel
    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x = divup(in.dims[0], TILE_DIM);
    int blk_y = divup(in.dims[1], TILE_DIM);
    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    transposeIP(qArgs, in, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
