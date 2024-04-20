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

namespace flare {
namespace cuda {

template<typename T>
__global__ void coo2Dense(Param<T> output, CParam<T> values, CParam<int> rowIdx,
                          CParam<int> colIdx) {
    int id = blockIdx.x * blockDim.x * reps + threadIdx.x;
    if (id >= values.dims[0]) return;

    for (int i = threadIdx.x; i <= reps * blockDim.x; i += blockDim.x) {
        if (i >= values.dims[0]) return;

        T v   = values.ptr[i];
        int r = rowIdx.ptr[i];
        int c = colIdx.ptr[i];

        int offset = r + c * output.strides[1];

        output.ptr[offset] = v;
    }
}

}  // namespace cuda
}  // namespace flare
