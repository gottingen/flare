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
#include <nvrtc_kernel_headers/gradient_cuh.hpp>

#include <array>

namespace flare {
namespace cuda {
namespace kernel {

template<typename T>
void gradient(Param<T> grad0, Param<T> grad1, CParam<T> in) {
    constexpr unsigned TX = 32;
    constexpr unsigned TY = 8;

    auto gradient =
        common::getKernel("flare::cuda::gradient", {{gradient_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()),
                          {{DefineValue(TX), DefineValue(TY)}});

    dim3 threads(TX, TY, 1);

    int blocksPerMatX = divup(in.dims[0], TX);
    int blocksPerMatY = divup(in.dims[1], TY);
    dim3 blocks(blocksPerMatX * in.dims[2], blocksPerMatY * in.dims[3], 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    gradient(qArgs, grad0, grad1, in, blocksPerMatX, blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
