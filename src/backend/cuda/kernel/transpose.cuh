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

#include <Param.hpp>
#include <math.hpp>

namespace flare {
namespace cuda {

template<typename T, bool conjugate>
__device__ T doOp(T in) {
    if (conjugate)
        return conj(in);
    else
        return in;
}

template<typename T, bool conjugate, bool is32Multiple>
__global__ void transpose(Param<T> out, CParam<T> in, const int blocksPerMatX,
                          const int blocksPerMatY) {
    __shared__ T shrdMem[TILE_DIM][TILE_DIM + 1];

    const int oDim0 = out.dims[0];
    const int oDim1 = out.dims[1];
    const int iDim0 = in.dims[0];
    const int iDim1 = in.dims[1];

    const int oStride1 = out.strides[1];
    const int iStride1 = in.strides[1];

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    const int batchId_x  = blockIdx.x / blocksPerMatX;
    const int blockIdx_x = (blockIdx.x - batchId_x * blocksPerMatX);

    const int batchId_y = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;
    const int blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (batchId_y * blocksPerMatY);

    if (batchId_x >= in.dims[2] || batchId_y >= in.dims[3]) return;

    const int x0 = TILE_DIM * blockIdx_x;
    const int y0 = TILE_DIM * blockIdx_y;

    int gx = lx + x0;
    int gy = ly + y0;

    in.ptr += batchId_x * in.strides[2] + batchId_y * in.strides[3];
    out.ptr += batchId_x * out.strides[2] + batchId_y * out.strides[3];

#pragma unroll
    for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int gy_ = gy + repeat;
        if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
            shrdMem[ly + repeat][lx] = in.ptr[gy_ * iStride1 + gx];
    }
    __syncthreads();

    gx = lx + y0;
    gy = ly + x0;

#pragma unroll
    for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int gy_ = gy + repeat;
        if (is32Multiple || (gx < oDim0 && gy_ < oDim1))
            out.ptr[gy_ * oStride1 + gx] =
                doOp<T, conjugate>(shrdMem[lx][ly + repeat]);
    }
}

}  // namespace cuda
}  // namespace flare
