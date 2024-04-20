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
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/sparse_arith_cuh.hpp>
#include <optypes.hpp>

namespace flare {
namespace cuda {
namespace kernel {

constexpr unsigned TX      = 32;
constexpr unsigned TY      = 8;
constexpr unsigned THREADS = TX * TY;

template<typename T, fly_op_t op>
void sparseArithOpCSR(Param<T> out, CParam<T> values, CParam<int> rowIdx,
                      CParam<int> colIdx, CParam<T> rhs, const bool reverse) {
    auto csrArithDSD = common::getKernel(
        "flare::cuda::csrArithDSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(TX), DefineValue(TY)}});

    // Each Y for threads does one row
    dim3 threads(TX, TY, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(out.dims[0], TY), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    csrArithDSD(qArgs, out, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

template<typename T, fly_op_t op>
void sparseArithOpCOO(Param<T> out, CParam<T> values, CParam<int> rowIdx,
                      CParam<int> colIdx, CParam<T> rhs, const bool reverse) {
    auto cooArithDSD = common::getKernel(
        "flare::cuda::cooArithDSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(THREADS)}});

    // Linear indexing with one elements per thread
    dim3 threads(THREADS, 1, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(values.dims[0], THREADS), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    cooArithDSD(qArgs, out, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

template<typename T, fly_op_t op>
void sparseArithOpCSR(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      CParam<T> rhs, const bool reverse) {
    auto csrArithSSD = common::getKernel(
        "flare::cuda::csrArithSSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(TX), DefineValue(TY)}});

    // Each Y for threads does one row
    dim3 threads(TX, TY, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(rhs.dims[0], TY), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    csrArithSSD(qArgs, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

template<typename T, fly_op_t op>
void sparseArithOpCOO(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      CParam<T> rhs, const bool reverse) {
    auto cooArithSSD = common::getKernel(
        "flare::cuda::cooArithSSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(THREADS)}});

    // Linear indexing with one elements per thread
    dim3 threads(THREADS, 1, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(values.dims[0], THREADS), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    cooArithSSD(qArgs, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace flare
