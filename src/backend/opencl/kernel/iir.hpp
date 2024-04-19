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
#include <kernel_headers/iir.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T, bool batch_a>
void iir(Param y, Param c, Param a) {
    // FIXME: This is a temporary fix. Ideally the local memory should be
    // allocted outside
    constexpr int MAX_A_SIZE = (1024 * sizeof(double)) / sizeof(T);

    std::array<TemplateArg, 2> targs = {
        TemplateTypename<T>(),
        TemplateArg(batch_a),
    };
    std::array<std::string, 5> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()), DefineValue(MAX_A_SIZE),
        DefineKeyValue(BATCH_A, batch_a),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        getTypeBuildDefinition<T>()};

    auto iir = common::getKernel("iir_kernel", {{iir_cl_src}}, targs, options);

    const int groups_y = y.info.dims[1];
    const int groups_x = y.info.dims[2];

    int threads = 256;
    while (threads > (int)y.info.dims[0] && threads > 32) threads /= 2;

    cl::NDRange local(threads, 1);
    cl::NDRange global(groups_x * local[0],
                       groups_y * y.info.dims[3] * local[1]);

    try {
        iir(cl::EnqueueArgs(getQueue(), global, local), *y.data, y.info,
            *c.data, c.info, *a.data, a.info, groups_y);
    } catch (cl::Error& clerr) {
        FLY_ERROR("Size of a too big for this datatype", FLY_ERR_SIZE);
    }
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace flare
