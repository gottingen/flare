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

namespace flare {
namespace cuda {

template<typename T>
__global__ void moments(Param<float> out, CParam<T> in, fly::momentType moment,
                        const bool pBatch) {
    const dim_t idw = blockIdx.y / in.dims[2];
    const dim_t idz = blockIdx.y - idw * in.dims[2];

    const dim_t idy = blockIdx.x;
    dim_t idx       = threadIdx.x;

    if (idy >= in.dims[1] || idz >= in.dims[2] || idw >= in.dims[3]) return;

    extern __shared__ float blk_moment_sum[];
    if (threadIdx.x < out.dims[0]) { blk_moment_sum[threadIdx.x] = 0.f; }
    __syncthreads();

    dim_t mId = idy * in.strides[1] + idx;
    if (pBatch) { mId += idw * in.strides[3] + idz * in.strides[2]; }

    for (; idx < in.dims[0]; idx += blockDim.x) {
        dim_t m_off = 0;
        float val   = (float)in.ptr[mId];
        mId += blockDim.x;

        if ((moment & FLY_MOMENT_M00) > 0) {
            atomicAdd(blk_moment_sum + m_off++, val);
        }
        if ((moment & FLY_MOMENT_M01) > 0) {
            atomicAdd(blk_moment_sum + m_off++, idx * val);
        }
        if ((moment & FLY_MOMENT_M10) > 0) {
            atomicAdd(blk_moment_sum + m_off++, idy * val);
        }
        if ((moment & FLY_MOMENT_M11) > 0) {
            atomicAdd(blk_moment_sum + m_off, idx * idy * val);
        }
    }

    __syncthreads();

    float *offset = const_cast<float *>(
        out.ptr + (idw * out.strides[3] + idz * out.strides[2]) + threadIdx.x);
    if (threadIdx.x < out.dims[0])
        atomicAdd(offset, blk_moment_sum[threadIdx.x]);
}

}  // namespace cuda
}  // namespace flare
