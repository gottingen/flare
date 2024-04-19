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
#include <nvrtc_kernel_headers/morph_cuh.hpp>

#include <limits>

namespace flare {
namespace cuda {
namespace kernel {

static const int MAX_MORPH_FILTER_LEN = 19;
static const int THREADS_X            = 16;
static const int THREADS_Y            = 16;
static const int CUBE_X               = 8;
static const int CUBE_Y               = 8;
static const int CUBE_Z               = 8;

template<typename T>
void morph(Param<T> out, CParam<T> in, CParam<T> mask, bool isDilation) {
    const int windLen  = mask.dims[0];
    const int SeLength = (windLen <= 10 ? windLen : 0);

    auto morph = common::getKernel(
        "flare::cuda::morph", {{morph_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(isDilation),
                     TemplateArg(SeLength)),
        {{DefineValue(MAX_MORPH_FILTER_LEN)}});

    morph.copyToReadOnly(morph.getDevPtr("cFilter"),
                         reinterpret_cast<CUdeviceptr>(mask.ptr),
                         mask.dims[0] * mask.dims[1] * sizeof(T));

    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x = divup(in.dims[0], THREADS_X);
    int blk_y = divup(in.dims[1], THREADS_Y);
    // launch batch * blk_x blocks along x dimension
    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    // calculate shared memory size
    int padding = (windLen % 2 == 0 ? (windLen - 1) : (2 * (windLen / 2)));
    int shrdLen =
        kernel::THREADS_X + padding + 1;  // +1 for to avoid bank conflicts
    int shrdSize = shrdLen * (kernel::THREADS_Y + padding) * sizeof(T);

    EnqueueArgs qArgs(blocks, threads, getActiveStream(), shrdSize);
    morph(qArgs, out, in, blk_x, blk_y, windLen);
    POST_LAUNCH_CHECK();
}

template<typename T>
void morph3d(Param<T> out, CParam<T> in, CParam<T> mask, bool isDilation) {
    const int windLen = mask.dims[0];

    if (windLen > 7) {
        CUDA_NOT_SUPPORTED("Morph 3D does not support kernels larger than 7.");
    }

    auto morph3D = common::getKernel(
        "flare::cuda::morph3D", {{morph_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(isDilation),
                     TemplateArg(windLen)),
        {{DefineValue(MAX_MORPH_FILTER_LEN)}});

    morph3D.copyToReadOnly(
        morph3D.getDevPtr("cFilter"), reinterpret_cast<CUdeviceptr>(mask.ptr),
        mask.dims[0] * mask.dims[1] * mask.dims[2] * sizeof(T));

    dim3 threads(kernel::CUBE_X, kernel::CUBE_Y, kernel::CUBE_Z);

    int blk_x = divup(in.dims[0], CUBE_X);
    int blk_y = divup(in.dims[1], CUBE_Y);
    int blk_z = divup(in.dims[2], CUBE_Z);
    dim3 blocks(blk_x * in.dims[3], blk_y, blk_z);

    // calculate shared memory size
    int padding = (windLen % 2 == 0 ? (windLen - 1) : (2 * (windLen / 2)));
    int shrdLen =
        kernel::CUBE_X + padding + 1;  // +1 for to avoid bank conflicts
    int shrdSize = shrdLen * (kernel::CUBE_Y + padding) *
                   (kernel::CUBE_Z + padding) * sizeof(T);

    EnqueueArgs qArgs(blocks, threads, getActiveStream(), shrdSize);
    morph3D(qArgs, out, in, blk_x);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
