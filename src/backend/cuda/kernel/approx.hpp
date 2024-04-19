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
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/approx1_cuh.hpp>
#include <nvrtc_kernel_headers/approx2_cuh.hpp>
#include <fly/defines.h>

namespace flare {
namespace cuda {
namespace kernel {

// Kernel Launch Config Values
static const int TX      = 16;
static const int TY      = 16;
static const int THREADS = 256;

template<typename Ty, typename Tp>
void approx1(Param<Ty> yo, CParam<Ty> yi, CParam<Tp> xo, const int xdim,
             const Tp &xi_beg, const Tp &xi_step, const float offGrid,
             const fly::interpType method, const int order) {
    auto approx1 = common::getKernel(
        "flare::cuda::approx1", {{approx1_cuh_src}},
        TemplateArgs(TemplateTypename<Ty>(), TemplateTypename<Tp>(),
                     TemplateArg(xdim), TemplateArg(order)));

    dim3 threads(THREADS, 1, 1);
    int blocksPerMat = divup(yo.dims[0], threads.x);
    dim3 blocks(blocksPerMat * yo.dims[1], yo.dims[2] * yo.dims[3]);

    bool batch = !(xo.dims[1] == 1 && xo.dims[2] == 1 && xo.dims[3] == 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    approx1(qArgs, yo, yi, xo, xi_beg, Tp(1) / xi_step, offGrid, blocksPerMat,
            batch, method);

    POST_LAUNCH_CHECK();
}

template<typename Ty, typename Tp>
void approx2(Param<Ty> zo, CParam<Ty> zi, CParam<Tp> xo, const int xdim,
             const Tp &xi_beg, const Tp &xi_step, CParam<Tp> yo, const int ydim,
             const Tp &yi_beg, const Tp &yi_step, const float offGrid,
             const fly::interpType method, const int order) {
    auto approx2 = common::getKernel(
        "flare::cuda::approx2", {{approx2_cuh_src}},
        TemplateArgs(TemplateTypename<Ty>(), TemplateTypename<Tp>(),
                     TemplateArg(xdim), TemplateArg(ydim), TemplateArg(order)));

    dim3 threads(TX, TY, 1);
    int blocksPerMatX = divup(zo.dims[0], threads.x);
    int blocksPerMatY = divup(zo.dims[1], threads.y);
    dim3 blocks(blocksPerMatX * zo.dims[2], blocksPerMatY * zo.dims[3]);

    bool batch = !(xo.dims[2] == 1 && xo.dims[3] == 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    approx2(qArgs, zo, zi, xo, xi_beg, Tp(1) / xi_step, yo, yi_beg,
            Tp(1) / yi_step, offGrid, blocksPerMatX, blocksPerMatY, batch,
            method);

    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
}  // namespace flare
