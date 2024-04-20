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
#include <shared.hpp>
#include <types.hpp>

namespace flare {
namespace cuda {

template<typename T, bool isLinear>
__global__ void histogram(Param<uint> out, CParam<T> in, int len, int nbins,
                          float minval, float maxval, int nBBS) {
    SharedMemory<uint> shared;
    uint *shrdMem = shared.getPointer();

    // offset input and output to account for batch ops
    unsigned b2 = blockIdx.x / nBBS;
    const data_t<T> *iptr =
        in.ptr + b2 * in.strides[2] + blockIdx.y * in.strides[3];
    uint *optr = out.ptr + b2 * out.strides[2] + blockIdx.y * out.strides[3];

    int start = (blockIdx.x - b2 * nBBS) * THRD_LOAD * blockDim.x + threadIdx.x;
    int end   = min((start + THRD_LOAD * blockDim.x), len);
    float step = (maxval - minval) / (float)nbins;
    compute_t<T> minvalT(minval);

    // If nbins > max shared memory allocated, then just use atomicAdd on global
    // memory
    bool use_global = nbins > MAX_BINS;

    // Skip initializing shared memory
    if (!use_global) {
        for (int i = threadIdx.x; i < nbins; i += blockDim.x) shrdMem[i] = 0;
        __syncthreads();
    }

    for (int row = start; row < end; row += blockDim.x) {
        int idx =
            isLinear
                ? row
                : ((row % in.dims[0]) + (row / in.dims[0]) * in.strides[1]);
        int bin =
            (int)(static_cast<float>(compute_t<T>(iptr[idx]) - minvalT) / step);
        bin = (bin < 0) ? 0 : bin;
        bin = (bin >= nbins) ? (nbins - 1) : bin;

        if (use_global) {
            atomicAdd((optr + bin), 1);
        } else {
            atomicAdd((shrdMem + bin), 1);
        }
    }

    // No need to write to global if use_global is true
    if (!use_global) {
        __syncthreads();
        for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
            atomicAdd((optr + i), shrdMem[i]);
        }
    }
}

}  // namespace cuda
}  // namespace flare
