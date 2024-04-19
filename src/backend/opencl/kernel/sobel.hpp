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
#include <kernel_headers/sobel.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {
template<typename Ti, typename To, unsigned ker_size>
void sobel(Param dx, Param dy, const Param in) {
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    std::vector<TemplateArg> targs = {
        TemplateTypename<Ti>(),
        TemplateTypename<To>(),
        TemplateArg(ker_size),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(KER_SIZE, ker_size),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    auto sobel =
        common::getKernel("sobel3x3", {{sobel_cl_src}}, targs, compileOpts);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                       blk_y * in.info.dims[3] * THREADS_Y);
    size_t loc_size =
        (THREADS_X + ker_size - 1) * (THREADS_Y + ker_size - 1) * sizeof(Ti);

    sobel(cl::EnqueueArgs(getQueue(), global, local), *dx.data, dx.info,
          *dy.data, dy.info, *in.data, in.info, cl::Local(loc_size), blk_x,
          blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
