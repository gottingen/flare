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
#include <nvrtc_kernel_headers/meanshift_cuh.hpp>

#include <array>
#include <type_traits>

namespace flare {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T>
void meanshift(Param<T> out, CParam<T> in, const float spatialSigma,
               const float chromaticSigma, const uint numIters, bool IsColor) {
    typedef typename std::conditional<std::is_same<T, double>::value, double,
                                      float>::type AccType;
    auto meanshift = common::getKernel(
        "flare::cuda::meanshift", {{meanshift_cuh_src}},
        TemplateArgs(TemplateTypename<AccType>(), TemplateTypename<T>(),
                     TemplateArg((IsColor ? 3 : 1))  // channels
                     ));

    static dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x        = divup(in.dims[0], THREADS_X);
    int blk_y        = divup(in.dims[1], THREADS_Y);
    const int bCount = (IsColor ? 1 : in.dims[2]);

    dim3 blocks(blk_x * bCount, blk_y * in.dims[3]);

    // clamp spatical and chromatic sigma's
    int radius       = std::max((int)(spatialSigma * 1.5f), 1);
    const float cvar = chromaticSigma * chromaticSigma;

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    meanshift(qArgs, out, in, radius, cvar, numIters, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
