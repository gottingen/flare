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

#include <Param.hpp>
#include <assign_kernel_param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/assign_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

template<typename T>
void assign(Param<T> out, CParam<T> in, const AssignKernelParam& p) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    auto assignKer =
        common::getKernel("flare::cuda::assign", {{assign_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()));

    const dim3 threads(THREADS_X, THREADS_Y);

    int blks_x = divup(in.dims[0], threads.x);
    int blks_y = divup(in.dims[1], threads.y);

    dim3 blocks(blks_x * in.dims[2], blks_y * in.dims[3]);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    assignKer(qArgs, out, in, p, blks_x, blks_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
