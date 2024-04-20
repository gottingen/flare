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
__global__ void range(Param<T> out, const int dim, const int blocksPerMatX,
                      const int blocksPerMatY) {
    const int mul0 = (dim == 0);
    const int mul1 = (dim == 1);
    const int mul2 = (dim == 2);
    const int mul3 = (dim == 3);

    const int oz = blockIdx.x / blocksPerMatX;
    const int ow = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;

    const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - ow * blocksPerMatY;

    const int xx = threadIdx.x + blockIdx_x * blockDim.x;
    const int yy = threadIdx.y + blockIdx_y * blockDim.y;

    if (xx >= out.dims[0] || yy >= out.dims[1] || oz >= out.dims[2] ||
        ow >= out.dims[3])
        return;

    const int ozw = ow * out.strides[3] + oz * out.strides[2];

    int valZW = (mul3 * ow) + (mul2 * oz);

    const int incy = blocksPerMatY * blockDim.y;
    const int incx = blocksPerMatX * blockDim.x;

    for (int oy = yy; oy < out.dims[1]; oy += incy) {
        compute_t<T> valYZW = valZW + (mul1 * oy);
        int oyzw            = ozw + oy * out.strides[1];
        for (int ox = xx; ox < out.dims[0]; ox += incx) {
            int oidx         = oyzw + ox;
            compute_t<T> val = valYZW + static_cast<compute_t<T>>(ox * mul0);

            out.ptr[oidx] = val;
        }
    }
}

}  // namespace cuda
}  // namespace flare
