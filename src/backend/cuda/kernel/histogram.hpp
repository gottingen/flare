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
#include <nvrtc_kernel_headers/histogram_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

constexpr int MAX_BINS  = 4000;
constexpr int THREADS_X = 256;
constexpr int THRD_LOAD = 16;

template<typename T>
void histogram(Param<uint> out, CParam<T> in, int nbins, float minval,
               float maxval, bool isLinear) {
    auto histogram = common::getKernel(
        "flare::cuda::histogram", {{histogram_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(isLinear)),
        {{DefineValue(MAX_BINS), DefineValue(THRD_LOAD)}});

    dim3 threads(kernel::THREADS_X, 1);

    int nElems = in.dims[0] * in.dims[1];
    int blk_x  = divup(nElems, THRD_LOAD * THREADS_X);

    dim3 blocks(blk_x * in.dims[2], in.dims[3]);

    // If nbins > MAX_BINS, we are using global memory so smem_size can be 0;
    int smem_size = nbins <= MAX_BINS ? (nbins * sizeof(uint)) : 0;

    EnqueueArgs qArgs(blocks, threads, getActiveStream(), smem_size);
    histogram(qArgs, out, in, nElems, nbins, minval, maxval, blk_x);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
