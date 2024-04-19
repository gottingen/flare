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
#include <nvrtc_kernel_headers/diff_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

template<typename T>
void diff(Param<T> out, CParam<T> in, const int indims, const unsigned dim,
          const bool isDiff2) {
    constexpr unsigned TX = 16;
    constexpr unsigned TY = 16;

    auto diff =
        common::getKernel("flare::cuda::diff", {{diff_cuh_src}},
                          TemplateArgs(TemplateTypename<T>(), TemplateArg(dim),
                                       TemplateArg(isDiff2)));

    dim3 threads(TX, TY, 1);

    if (dim == 0 && indims == 1) { threads = dim3(TX * TY, 1, 1); }

    int blocksPerMatX = divup(out.dims[0], TX);
    int blocksPerMatY = divup(out.dims[1], TY);
    dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3], 1);

    const int oElem = out.dims[0] * out.dims[1] * out.dims[2] * out.dims[3];

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    diff(qArgs, out, in, oElem, blocksPerMatX, blocksPerMatY);

    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
}  // namespace flare
