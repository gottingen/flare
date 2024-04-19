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
#include <nvrtc_kernel_headers/rotate_cuh.hpp>
#include <fly/defines.h>

namespace flare {
namespace cuda {
namespace kernel {

// Kernel Launch Config Values
constexpr unsigned TX = 16;
constexpr unsigned TY = 16;
// Used for batching images
constexpr int TI = 4;

typedef struct {
    float tmat[6];
} tmat_t;

template<typename T>
void rotate(Param<T> out, CParam<T> in, const float theta,
            const fly::interpType method, const int order) {
    auto rotate = common::getKernel(
        "flare::cuda::rotate", {{rotate_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(order)));

    const float c = cos(-theta), s = sin(-theta);
    float tx, ty;
    {
        const float nx = 0.5 * (in.dims[0] - 1);
        const float ny = 0.5 * (in.dims[1] - 1);
        const float mx = 0.5 * (out.dims[0] - 1);
        const float my = 0.5 * (out.dims[1] - 1);
        const float sx = (mx * c + my * -s);
        const float sy = (mx * s + my * c);
        tx             = -(sx - nx);
        ty             = -(sy - ny);
    }

    // Rounding error. Anything more than 3 decimal points wont make a diff
    tmat_t t;
    t.tmat[0] = round(c * 1000) / 1000.0f;
    t.tmat[1] = round(-s * 1000) / 1000.0f;
    t.tmat[2] = round(tx * 1000) / 1000.0f;
    t.tmat[3] = round(s * 1000) / 1000.0f;
    t.tmat[4] = round(c * 1000) / 1000.0f;
    t.tmat[5] = round(ty * 1000) / 1000.0f;

    int nimages  = in.dims[2];
    int nbatches = in.dims[3];

    dim3 threads(TX, TY, 1);
    dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

    const int blocksXPerImage = blocks.x;
    const int blocksYPerImage = blocks.y;

    if (nimages > TI) {
        int tile_images = divup(nimages, TI);
        nimages         = TI;
        blocks.x        = blocks.x * tile_images;
    }

    blocks.y = blocks.y * nbatches;

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    rotate(qArgs, out, in, t, nimages, nbatches, blocksXPerImage,
           blocksYPerImage, method);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
