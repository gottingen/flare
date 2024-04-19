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
#include <common/defines.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/flood_fill_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

constexpr int THREADS   = 256;
constexpr int TILE_DIM  = 32;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = THREADS / TILE_DIM;

// Shared memory per block required by floodFill kernel
template<typename T>
constexpr size_t sharedMemRequiredByFloodFill() {
    // 1-pixel border neighborhood
    return sizeof(T) * ((THREADS_X + 2) * (THREADS_Y + 2));
}

template<typename T>
void floodFill(Param<T> out, CParam<T> image, CParam<uint> seedsx,
               CParam<uint> seedsy, const T newValue, const T lowValue,
               const T highValue, const fly::connectivity nlookup) {
    UNUSED(nlookup);
    if (sharedMemRequiredByFloodFill<T>() >
        getDeviceProp(getActiveDeviceId()).sharedMemPerBlock) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nCurrent thread's CUDA device doesn't have sufficient "
                 "shared memory required by FloodFill\n");
        CUDA_NOT_SUPPORTED(errMessage);
    }

    auto initSeeds =
        common::getKernel("flare::cuda::initSeeds", {{flood_fill_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()));
    auto floodStep =
        common::getKernel("flare::cuda::floodStep", {{flood_fill_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()),
                          {{DefineValue(THREADS_X), DefineValue(THREADS_Y)}});
    auto finalizeOutput = common::getKernel(
        "flare::cuda::finalizeOutput", {{flood_fill_cuh_src}},
        TemplateArgs(TemplateTypename<T>()));

    EnqueueArgs qArgs(dim3(divup(seedsx.elements(), THREADS)), dim3(THREADS),
                      getActiveStream());
    initSeeds(qArgs, out, seedsx, seedsy);
    POST_LAUNCH_CHECK();

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks(divup(image.dims[0], threads.x),
                divup(image.dims[1], threads.y));
    EnqueueArgs fQArgs(blocks, threads, getActiveStream());

    auto continueFlagPtr = floodStep.getDevPtr("doAnotherLaunch");

    for (int doAnotherLaunch = 1; doAnotherLaunch > 0;) {
        doAnotherLaunch = 0;
        floodStep.setFlag(continueFlagPtr, &doAnotherLaunch);
        floodStep(fQArgs, out, image, lowValue, highValue);
        POST_LAUNCH_CHECK();
        doAnotherLaunch = floodStep.getFlag(continueFlagPtr);
    }
    finalizeOutput(fQArgs, out, newValue);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
