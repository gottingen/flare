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
#include <kernel_headers/range.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T>
void range(Param out, const int dim) {
    constexpr int RANGE_TX    = 32;
    constexpr int RANGE_TY    = 8;
    constexpr int RANGE_TILEX = 512;
    constexpr int RANGE_TILEY = 32;

    std::array<TemplateArg, 1> targs   = {TemplateTypename<T>()};
    std::array<std::string, 2> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        getTypeBuildDefinition<T>()};

    auto rangeOp =
        common::getKernel("range_kernel", {{range_cl_src}}, targs, options);

    cl::NDRange local(RANGE_TX, RANGE_TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], RANGE_TILEX);
    int blocksPerMatY = divup(out.info.dims[1], RANGE_TILEY);
    cl::NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                       local[1] * blocksPerMatY * out.info.dims[3], 1);

    rangeOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
            dim, blocksPerMatX, blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
