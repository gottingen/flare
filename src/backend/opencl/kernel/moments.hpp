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
#include <kernel/config.hpp>
#include <kernel_headers/moments.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T>
void moments(Param out, const Param in, fly_moment_type moment) {
    constexpr int THREADS = 128;

    std::array<TemplateArg, 2> targs = {
        TemplateTypename<T>(),
        TemplateArg(out.info.dims[0]),
    };
    std::array<std::string, 3> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(MOMENTS_SZ, out.info.dims[0]),
        getTypeBuildDefinition<T>()};

    auto momentsOp =
        common::getKernel("moments", {{moments_cl_src}}, targs, options);

    cl::NDRange local(THREADS, 1, 1);
    cl::NDRange global(in.info.dims[1] * local[0],
                       in.info.dims[2] * in.info.dims[3] * local[1]);

    bool pBatch = !(in.info.dims[2] == 1 && in.info.dims[3] == 1);

    momentsOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, (int)moment, (int)pBatch);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace flare
