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

template<typename T>
__global__ void exampleFunc(Param<T> c, CParam<T> a, CParam<T> b,
                            const fly_someenum_t p) {
    // get current thread global identifiers along required dimensions
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < a.dims[0] && j < a.dims[1]) {
        // if needed use strides array to compute linear index of arrays
        int src1Idx = i + j * a.strides[1];
        int src2Idx = i + j * b.strides[1];
        int dstIdx  = i + j * c.strides[1];

        T* dst        = c.ptr;
        const T* src1 = a.ptr;
        const T* src2 = b.ptr;

        // kernel algorithm goes here
        dst[dstIdx] = src1[src1Idx] + src2[src2Idx];
    }
}

}  // namespace cuda
}  // namespace flare
