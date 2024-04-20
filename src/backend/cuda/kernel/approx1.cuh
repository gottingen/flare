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
#include <interp.hpp>

namespace flare {
namespace cuda {

template<typename Ty, typename Tp, int xdim, int order>
__global__ void approx1(Param<Ty> yo, CParam<Ty> yi, CParam<Tp> xo,
                        const Tp xi_beg, const Tp xi_step_reproc,
                        const float offGrid, const int blocksMatX,
                        const bool batch, fly::interpType method) {
    const int idy        = blockIdx.x / blocksMatX;
    const int blockIdx_x = blockIdx.x - idy * blocksMatX;
    const int idx        = blockIdx_x * blockDim.x + threadIdx.x;

    const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / yo.dims[2];
    const int idz = (blockIdx.y + blockIdx.z * gridDim.y) - idw * yo.dims[2];

    if (idx >= yo.dims[0] || idy >= yo.dims[1] || idz >= yo.dims[2] ||
        idw >= yo.dims[3])
        return;

    // FIXME: Only cubic interpolation is doing clamping
    // We need to make it consistent across all methods
    // Not changing the behavior because tests will fail
    const bool clamp = order == 3;

    bool is_off[] = {xo.dims[0] > 1, xo.dims[1] > 1, xo.dims[2] > 1,
                     xo.dims[3] > 1};

    int xo_idx = idx * is_off[0];
    if (batch) {
        xo_idx += idw * xo.strides[3] * is_off[3];
        xo_idx += idz * xo.strides[2] * is_off[2];
        xo_idx += idy * xo.strides[1] * is_off[1];
    }

    const Tp x = (xo.ptr[xo_idx] - xi_beg) * xi_step_reproc;

    const int yo_idx =
        idw * yo.strides[3] + idz * yo.strides[2] + idy * yo.strides[1] + idx;

#pragma unroll
    for (int flagIdx = 0; flagIdx < 4; ++flagIdx) { is_off[flagIdx] = true; }
    is_off[xdim] = false;

    if (x < 0 || yi.dims[xdim] < x + 1) {
        yo.ptr[yo_idx] = scalar<Ty>(offGrid);
        return;
    }

    int yi_idx = idx * is_off[0];
    yi_idx += idw * yi.strides[3] * is_off[3];
    yi_idx += idz * yi.strides[2] * is_off[2];
    yi_idx += idy * yi.strides[1] * is_off[1];

    Interp1<Ty, Tp, xdim, order> interp;
    interp(yo, yo_idx, yi, yi_idx, x, method, 1, clamp);
}

}  // namespace cuda
}  // namespace flare
