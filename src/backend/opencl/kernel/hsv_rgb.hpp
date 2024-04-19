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
#include <debug_opencl.hpp>
#include <kernel_headers/hsv_rgb.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T>
void hsv2rgb_convert(Param out, const Param in, bool isHSV2RGB) {
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    std::array<TemplateArg, 2> targs = {
        TemplateTypename<T>(),
        TemplateArg(isHSV2RGB),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        getTypeBuildDefinition<T>()};
    if (isHSV2RGB) { options.emplace_back(DefineKey(isHSV2RGB)); }

    auto convert =
        common::getKernel("hsvrgbConvert", {{hsv_rgb_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    // all images are three channels, so batch
    // parameter would be along 4th dimension
    cl::NDRange global(blk_x * in.info.dims[3] * THREADS_X, blk_y * THREADS_Y);

    convert(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
            *in.data, in.info, blk_x);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
