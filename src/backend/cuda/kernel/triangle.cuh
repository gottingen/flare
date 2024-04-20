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

template<typename T, bool is_upper, bool is_unit_diag>
__global__ void triangle(Param<T> r, CParam<T> in, const int blocksPerMatX,
                         const int blocksPerMatY) {
    const int oz = blockIdx.x / blocksPerMatX;
    const int ow = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;

    const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - ow * blocksPerMatY;

    const int xx = threadIdx.x + blockIdx_x * blockDim.x;
    const int yy = threadIdx.y + blockIdx_y * blockDim.y;

    const int incy = blocksPerMatY * blockDim.y;
    const int incx = blocksPerMatX * blockDim.x;

    T *d_r       = r.ptr;
    const T *d_i = in.ptr;

    const T one  = scalar<T>(1);
    const T zero = scalar<T>(0);

    if (oz < r.dims[2] && ow < r.dims[3]) {
        d_i = d_i + oz * in.strides[2] + ow * in.strides[3];
        d_r = d_r + oz * r.strides[2] + ow * r.strides[3];

        for (int oy = yy; oy < r.dims[1]; oy += incy) {
            const T *Yd_i = d_i + oy * in.strides[1];
            T *Yd_r       = d_r + oy * r.strides[1];

            for (int ox = xx; ox < r.dims[0]; ox += incx) {
                bool cond         = is_upper ? (oy >= ox) : (oy <= ox);
                bool do_unit_diag = is_unit_diag && (ox == oy);
                if (cond) {
                    // Change made because of compute 53 failing tests
                    Yd_r[ox] = do_unit_diag ? one : Yd_i[ox];
                } else {
                    Yd_r[ox] = zero;
                }
            }
        }
    }
}

}  // namespace cuda
}  // namespace flare
