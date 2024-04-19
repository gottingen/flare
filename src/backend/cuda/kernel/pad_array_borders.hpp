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
#include <nvrtc_kernel_headers/pad_array_borders_cuh.hpp>
#include <fly/defines.h>

#include <array>

namespace flare {
namespace cuda {
namespace kernel {

static const int PADB_THREADS_X = 32;
static const int PADB_THREADS_Y = 8;

template<typename T>
void padBorders(Param<T> out, CParam<T> in, dim4 const lBoundPadding,
                const fly::borderType btype) {
    auto padBorders = common::getKernel(
        "flare::cuda::padBorders", {{pad_array_borders_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(btype)));

    dim3 threads(kernel::PADB_THREADS_X, kernel::PADB_THREADS_Y);

    int blk_x = divup(out.dims[0], PADB_THREADS_X);
    int blk_y = divup(out.dims[1], PADB_THREADS_Y);

    dim3 blocks(blk_x * out.dims[2], blk_y * out.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    padBorders(qArgs, out, in, lBoundPadding[0], lBoundPadding[1],
               lBoundPadding[2], lBoundPadding[3], blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
