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
#include <kernel_headers/histogram.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T>
void histogram(Param out, const Param in, int nbins, float minval, float maxval,
               bool isLinear) {
    constexpr int MAX_BINS  = 4000;
    constexpr int THREADS_X = 256;
    constexpr int THRD_LOAD = 16;

    std::array<TemplateArg, 2> targs = {
        TemplateTypename<T>(),
        TemplateArg(isLinear),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(THRD_LOAD),
        DefineValue(MAX_BINS),
    };
    options.emplace_back(getTypeBuildDefinition<T>());
    if (isLinear) { options.emplace_back(DefineKey(IS_LINEAR)); }

    auto histogram =
        common::getKernel("histogram", {{histogram_cl_src}}, targs, options);

    int nElems  = in.info.dims[0] * in.info.dims[1];
    int blk_x   = divup(nElems, THRD_LOAD * THREADS_X);
    int locSize = nbins <= MAX_BINS ? (nbins * sizeof(uint)) : 1;

    cl::NDRange local(THREADS_X, 1);
    cl::NDRange global(blk_x * in.info.dims[2] * THREADS_X, in.info.dims[3]);

    histogram(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, cl::Local(locSize), nElems, nbins, minval,
              maxval, blk_x);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
