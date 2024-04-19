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
#include <kernel_headers/select.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <array>
#include <string>

namespace flare {
namespace opencl {
namespace kernel {
constexpr uint DIMX  = 32;
constexpr uint DIMY  = 8;
constexpr int REPEAT = 64;

template<typename T>
void selectLauncher(Param out, Param cond, Param a, Param b, const int ndims,
                    const bool is_same) {
    std::array<TemplateArg, 2> targs = {
        TemplateTypename<T>(),
        TemplateArg(is_same),
    };
    std::array<std::string, 3> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()), DefineValue(is_same),
        getTypeBuildDefinition<T>()};

    auto selectOp =
        common::getKernel("select_kernel", {{select_cl_src}}, targs, options);

    int threads[] = {DIMX, DIMY};

    if (ndims == 1) {
        threads[0] *= threads[1];
        threads[1] = 1;
    }

    cl::NDRange local(threads[0], threads[1]);

    int groups_0 = divup(out.info.dims[0], REPEAT * local[0]);
    int groups_1 = divup(out.info.dims[1], local[1]);

    cl::NDRange global(groups_0 * out.info.dims[2] * local[0],
                       groups_1 * out.info.dims[3] * local[1]);

    selectOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *cond.data, cond.info, *a.data, a.info, *b.data, b.info, groups_0,
             groups_1);
}

template<typename T>
void select(Param out, Param cond, Param a, Param b, int ndims) {
    bool is_same = true;
    for (int i = 0; i < 4; i++) {
        is_same &= (a.info.dims[i] == b.info.dims[i]);
    }
    selectLauncher<T>(out, cond, a, b, ndims, is_same);
}

template<typename T>
void select_scalar(Param out, Param cond, Param a, const T b, const int ndims,
                   const bool flip) {
    std::array<TemplateArg, 2> targs = {
        TemplateTypename<T>(),
        TemplateArg(flip),
    };
    std::array<std::string, 3> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()), DefineValue(flip),
        getTypeBuildDefinition<T>()};

    auto selectOp = common::getKernel("select_scalar_kernel", {{select_cl_src}},
                                      targs, options);

    int threads[] = {DIMX, DIMY};

    if (ndims == 1) {
        threads[0] *= threads[1];
        threads[1] = 1;
    }

    cl::NDRange local(threads[0], threads[1]);

    int groups_0 = divup(out.info.dims[0], REPEAT * local[0]);
    int groups_1 = divup(out.info.dims[1], local[1]);

    cl::NDRange global(groups_0 * out.info.dims[2] * local[0],
                       groups_1 * out.info.dims[3] * local[1]);

    selectOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *cond.data, cond.info, *a.data, a.info, b, groups_0, groups_1);
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
