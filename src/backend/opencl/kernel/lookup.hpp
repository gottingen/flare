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
#include <common/half.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/lookup.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename in_t, typename idx_t>
void lookup(Param out, const Param in, const Param indices,
            const unsigned dim) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    std::array<TemplateArg, 3> targs = {
        TemplateTypename<in_t>(),
        TemplateTypename<idx_t>(),
        TemplateArg(dim),
    };
    std::array<std::string, 4> options = {
        DefineKeyValue(in_t, dtype_traits<in_t>::getName()),
        DefineKeyValue(idx_t, dtype_traits<idx_t>::getName()),
        DefineKeyValue(DIM, dim), getTypeBuildDefinition<in_t, idx_t>()};

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * out.info.dims[2] * THREADS_X,
                       blk_y * out.info.dims[3] * THREADS_Y);

    auto arrIdxOp =
        common::getKernel("lookupND", {{lookup_cl_src}}, targs, options);

    arrIdxOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *in.data, in.info, *indices.data, indices.info, blk_x, blk_y);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace flare
