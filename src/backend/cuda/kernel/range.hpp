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
#include <nvrtc_kernel_headers/range_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

template<typename T>
void range(Param<T> out, const int dim) {
    constexpr unsigned RANGE_TX    = 32;
    constexpr unsigned RANGE_TY    = 8;
    constexpr unsigned RANGE_TILEX = 512;
    constexpr unsigned RANGE_TILEY = 32;

    auto range = common::getKernel("flare::cuda::range", {{range_cuh_src}},
                                   TemplateArgs(TemplateTypename<T>()));

    dim3 threads(RANGE_TX, RANGE_TY, 1);

    int blocksPerMatX = divup(out.dims[0], RANGE_TILEX);
    int blocksPerMatY = divup(out.dims[1], RANGE_TILEY);
    dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3], 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    range(qArgs, out, dim, blocksPerMatX, blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
