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
#include <kernel/config.hpp>
#include <kernel_headers/identity.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T>
static void identity(Param out) {
    std::array<TemplateArg, 1> targs = {
        TemplateTypename<T>(),
    };
    std::array<std::string, 4> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ONE, scalar_to_option(scalar<T>(1))),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        getTypeBuildDefinition<T>()};

    auto identityOp = common::getKernel("identity_kernel", {{identity_cl_src}},
                                        targs, options);

    cl::NDRange local(32, 8);
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_y = divup(out.info.dims[1], local[1]);
    cl::NDRange global(groups_x * out.info.dims[2] * local[0],
                       groups_y * out.info.dims[3] * local[1]);

    identityOp(cl::EnqueueArgs(getQueue(), global, local), *(out.data),
               out.info, groups_x, groups_y);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace flare
