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
#include <nvrtc_kernel_headers/moments_cuh.hpp>
#include <fly/defines.h>

namespace flare {
namespace cuda {
namespace kernel {

static const int THREADS = 128;

template<typename T>
void moments(Param<float> out, CParam<T> in, const fly::momentType moment) {
    auto moments =
        common::getKernel("flare::cuda::moments", {{moments_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()));

    dim3 threads(THREADS, 1, 1);
    dim3 blocks(in.dims[1], in.dims[2] * in.dims[3]);

    bool pBatch = !(in.dims[2] == 1 && in.dims[3] == 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream(),
                      sizeof(float) * out.dims[0]);

    moments(qArgs, out, in, moment, pBatch);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
