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
#include <kernel_headers/diff.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T>
void diff(Param out, const Param in, const unsigned indims, const unsigned dim,
          const bool isDiff2) {
    constexpr int TX = 16;
    constexpr int TY = 16;

    std::array<TemplateArg, 3> targs = {
        TemplateTypename<T>(),
        TemplateArg(dim),
        TemplateArg(isDiff2),
    };
    std::array<std::string, 4> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()), DefineKeyValue(DIM, dim),
        DefineKeyValue(isDiff2, (isDiff2 ? 1 : 0)),
        getTypeBuildDefinition<T>()};

    auto diffOp =
        common::getKernel("diff_kernel", {{diff_cl_src}}, targs, options);

    cl::NDRange local(TX, TY, 1);
    if (dim == 0 && indims == 1) { local = cl::NDRange(TX * TY, 1, 1); }

    int blocksPerMatX = divup(out.info.dims[0], local[0]);
    int blocksPerMatY = divup(out.info.dims[1], local[1]);
    cl::NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                       local[1] * blocksPerMatY * out.info.dims[3], 1);

    const int oElem = out.info.dims[0] * out.info.dims[1] * out.info.dims[2] *
                      out.info.dims[3];

    diffOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, *in.data,
           out.info, in.info, oElem, blocksPerMatX, blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
