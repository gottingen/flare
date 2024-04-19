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
#include <nvrtc_kernel_headers/iir_cuh.hpp>

namespace flare {
namespace cuda {
namespace kernel {

template<typename T, bool batch_a>
void iir(Param<T> y, CParam<T> c, CParam<T> a) {
    constexpr int MAX_A_SIZE = 1024;

    auto iir = common::getKernel(
        "flare::cuda::iir", {{iir_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(batch_a)),
        {{DefineValue(MAX_A_SIZE)}});

    const int blocks_y = y.dims[1];
    const int blocks_x = y.dims[2];

    dim3 blocks(blocks_x, blocks_y * y.dims[3]);

    int threads = 256;
    while (threads > y.dims[0] && threads > 32) threads /= 2;

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    iir(qArgs, y, c, a, blocks_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
