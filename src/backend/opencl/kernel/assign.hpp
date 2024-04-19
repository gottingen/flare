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
#include <kernel_headers/assign.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

typedef struct {
    int offs[4];
    int strds[4];
    char isSeq[4];
} AssignKernelParam_t;

template<typename T>
void assign(Param out, const Param in, const AssignKernelParam_t& p,
            cl::Buffer* bPtr[4]) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    std::array<TemplateArg, 1> targs = {
        TemplateTypename<T>(),
    };
    std::array<std::string, 2> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        getTypeBuildDefinition<T>()};

    auto assign =
        common::getKernel("assignKernel", {{assign_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                       blk_y * in.info.dims[3] * THREADS_Y);

    assign(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *in.data, in.info, p, *bPtr[0], *bPtr[1], *bPtr[2], *bPtr[3], blk_x,
           blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
