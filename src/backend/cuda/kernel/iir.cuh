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

template<typename T, bool batch_a>
__global__ void iir(Param<T> y, CParam<T> c, CParam<T> a, const int blocks_y) {
    __shared__ T s_z[MAX_A_SIZE];
    __shared__ T s_a[MAX_A_SIZE];
    __shared__ T s_y;

    const int idz = blockIdx.x;
    const int idw = blockIdx.y / blocks_y;
    const int idy = blockIdx.y - idw * blocks_y;

    const int tx    = threadIdx.x;
    const int num_a = a.dims[0];

    int y_off = idw * y.strides[3] + idz * y.strides[2] + idy * y.strides[1];
    int c_off = idw * c.strides[3] + idz * c.strides[2] + idy * c.strides[1];
    int a_off = 0;

    if (batch_a)
        a_off = idw * a.strides[3] + idz * a.strides[2] + idy * a.strides[1];

    T *d_y           = y.ptr + y_off;
    const T *d_c     = c.ptr + c_off;
    const T *d_a     = a.ptr + a_off;
    const int repeat = (num_a + blockDim.x - 1) / blockDim.x;

    for (int ii = 0; ii < MAX_A_SIZE / blockDim.x; ii++) {
        int id  = ii * blockDim.x + tx;
        s_z[id] = scalar<T>(0);
        s_a[id] = (id < num_a) ? d_a[id] : scalar<T>(0);
    }
    __syncthreads();

    for (int i = 0; i < y.dims[0]; i++) {
        if (tx == 0) {
            s_y    = (d_c[i] + s_z[0]) / s_a[0];
            d_y[i] = s_y;
        }
        __syncthreads();

#pragma unroll
        for (int ii = 0; ii < repeat; ii++) {
            int id = ii * blockDim.x + tx + 1;

            T z = s_z[id] - s_a[id] * s_y;
            __syncthreads();

            s_z[id - 1] = z;
            __syncthreads();
        }
    }
}

}  // namespace cuda
}  // namespace flare
