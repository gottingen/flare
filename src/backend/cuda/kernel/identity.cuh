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
#include <math.hpp>

namespace flare {
namespace cuda {

template<typename T>
__global__ void identity(Param<T> out, int blocks_x, int blocks_y) {
    const dim_t idz = blockIdx.x / blocks_x;
    const dim_t idw = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;

    const dim_t blockIdx_x = blockIdx.x - idz * blocks_x;
    const dim_t blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - idw * blocks_y;

    const dim_t idx = threadIdx.x + blockIdx_x * blockDim.x;
    const dim_t idy = threadIdx.y + blockIdx_y * blockDim.y;

    if (idx >= out.dims[0] || idy >= out.dims[1] || idz >= out.dims[2] ||
        idw >= out.dims[3])
        return;

    const T one  = scalar<T>(1);
    const T zero = scalar<T>(0);

    T *ptr = out.ptr + idz * out.strides[2] + idw * out.strides[3];
    T val  = (idx == idy) ? one : zero;
    ptr[idx + idy * out.strides[1]] = val;
}

}  // namespace cuda
}  // namespace flare
