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
#include <nvrtc_kernel_headers/anisotropic_diffusion_cuh.hpp>
#include <fly/defines.h>

namespace flare {
namespace cuda {
namespace kernel {

constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int YDIM_LOAD = 2 * THREADS_X / THREADS_Y;

template<typename T>
void anisotropicDiffusion(Param<T> inout, const float dt, const float mct,
                          const fly::fluxFunction fftype, bool isMCDE) {
    auto diffUpdate = common::getKernel(
        "flare::cuda::diffUpdate", {{anisotropic_diffusion_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(fftype),
                     TemplateArg(isMCDE)),
        {{DefineValue(THREADS_X), DefineValue(THREADS_Y),
          DefineValue(YDIM_LOAD)}});

    dim3 threads(THREADS_X, THREADS_Y, 1);

    int blkX = divup(inout.dims[0], threads.x);
    int blkY = divup(inout.dims[1], threads.y * YDIM_LOAD);

    dim3 blocks(blkX * inout.dims[2], blkY * inout.dims[3], 1);

    const int maxBlkY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    const int blkZ    = divup(blocks.y, maxBlkY);

    if (blkZ > 1) {
        blocks.y = maxBlkY;
        blocks.z = blkZ;
    }

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    diffUpdate(qArgs, inout, dt, mct, blkX, blkY);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
