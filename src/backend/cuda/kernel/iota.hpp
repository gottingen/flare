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
#include <nvrtc_kernel_headers/iota_cuh.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace cuda {
namespace kernel {

template<typename T>
void iota(Param<T> out, const fly::dim4 &sdims) {
    constexpr unsigned IOTA_TX = 32;
    constexpr unsigned IOTA_TY = 8;
    constexpr unsigned TILEX   = 512;
    constexpr unsigned TILEY   = 32;

    auto iota = common::getKernel("flare::cuda::iota", {{iota_cuh_src}},
                                  TemplateArgs(TemplateTypename<T>()));

    dim3 threads(IOTA_TX, IOTA_TY, 1);

    int blocksPerMatX = divup(out.dims[0], TILEX);
    int blocksPerMatY = divup(out.dims[1], TILEY);

    dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3], 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    iota(qArgs, out, sdims[0], sdims[1], sdims[2], sdims[3], blocksPerMatX,
         blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
