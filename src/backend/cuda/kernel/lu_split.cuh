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

template<typename T, bool same_dims>
__global__ void luSplit(Param<T> lower, Param<T> upper, Param<T> in,
                        const int blocksPerMatX, const int blocksPerMatY) {
    const int oz = blockIdx.x / blocksPerMatX;
    const int ow = blockIdx.y / blocksPerMatY;

    const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
    const int blockIdx_y = blockIdx.y - ow * blocksPerMatY;

    const int xx = threadIdx.x + blockIdx_x * blockDim.x;
    const int yy = threadIdx.y + blockIdx_y * blockDim.y;

    const int incy = blocksPerMatY * blockDim.y;
    const int incx = blocksPerMatX * blockDim.x;

    T *d_l = lower.ptr;
    T *d_u = upper.ptr;
    T *d_i = in.ptr;

    if (oz < in.dims[2] && ow < in.dims[3]) {
        d_i = d_i + oz * in.strides[2] + ow * in.strides[3];
        d_l = d_l + oz * lower.strides[2] + ow * lower.strides[3];
        d_u = d_u + oz * upper.strides[2] + ow * upper.strides[3];

        for (int oy = yy; oy < in.dims[1]; oy += incy) {
            T *Yd_i = d_i + oy * in.strides[1];
            T *Yd_l = d_l + oy * lower.strides[1];
            T *Yd_u = d_u + oy * upper.strides[1];
            for (int ox = xx; ox < in.dims[0]; ox += incx) {
                if (ox > oy) {
                    if (same_dims || oy < lower.dims[1]) Yd_l[ox] = Yd_i[ox];
                    if (!same_dims || ox < upper.dims[0])
                        Yd_u[ox] = scalar<T>(0);
                } else if (oy > ox) {
                    if (same_dims || oy < lower.dims[1])
                        Yd_l[ox] = scalar<T>(0);
                    if (!same_dims || ox < upper.dims[0]) Yd_u[ox] = Yd_i[ox];
                } else if (ox == oy) {
                    if (same_dims || oy < lower.dims[1])
                        Yd_l[ox] = scalar<T>(1.0);
                    if (!same_dims || ox < upper.dims[0]) Yd_u[ox] = Yd_i[ox];
                }
            }
        }
    }
}

}  // namespace cuda
}  // namespace flare
