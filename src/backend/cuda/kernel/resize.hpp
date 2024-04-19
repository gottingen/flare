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
#include <nvrtc_kernel_headers/resize_cuh.hpp>
#include <fly/defines.h>

namespace flare {
namespace cuda {
namespace kernel {

// Kernel Launch Config Values
static const unsigned TX = 16;
static const unsigned TY = 16;

template<typename T>
void resize(Param<T> out, CParam<T> in, fly_interp_type method) {
    auto resize = common::getKernel(
        "flare::cuda::resize", {{resize_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(method)));

    dim3 threads(TX, TY, 1);
    dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));
    int blocksPerMatX = blocks.x;
    int blocksPerMatY = blocks.y;

    if (in.dims[2] > 1) { blocks.x *= in.dims[2]; }
    if (in.dims[3] > 1) { blocks.y *= in.dims[3]; }
    float xf = (float)in.dims[0] / (float)out.dims[0];
    float yf = (float)in.dims[1] / (float)out.dims[1];

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    resize(qArgs, out, in, blocksPerMatX, blocksPerMatY, xf, yf);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
