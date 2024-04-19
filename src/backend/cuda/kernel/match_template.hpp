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
#include <nvrtc_kernel_headers/match_template_cuh.hpp>
#include <fly/defines.h>

namespace flare {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename inType, typename outType>
void matchTemplate(Param<outType> out, CParam<inType> srch,
                   CParam<inType> tmplt, const fly::matchType mType,
                   bool needMean) {
    auto matchTemplate = common::getKernel(
        "flare::cuda::matchTemplate", {{match_template_cuh_src}},
        TemplateArgs(TemplateTypename<inType>(), TemplateTypename<outType>(),
                     TemplateArg(mType), TemplateArg(needMean)));

    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(srch.dims[0], threads.x);
    int blk_y = divup(srch.dims[1], threads.y);

    dim3 blocks(blk_x * srch.dims[2], blk_y * srch.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    matchTemplate(qArgs, out, srch, tmplt, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
