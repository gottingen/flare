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
#include <utility.hpp>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void exampleFunction(Param<T> out, CParam<T> a, CParam<T> b,
                     const fly_someenum_t method) {
    UNUSED(method);
    dim4 oDims = out.dims();

    dim4 aStrides = a.strides();  // you can retrieve strides
    dim4 bStrides = b.strides();
    dim4 oStrides = out.strides();

    const T* src1 =
        a.get();  // cpu::Param<T>::get returns the pointer to the
                  // memory allocated for that Param (with proper offsets)
    const T* src2 =
        b.get();  // cpu::Param<T>::get returns the pointer to the
                  // memory allocated for that Param (with proper offsets)
    T* dst = out.get();

    // Implement your algorithm and write results to dst
    for (int j = 0; j < oDims[1]; ++j) {
        for (int i = 0; i < oDims[0]; ++i) {
            int src1Idx = i + j * aStrides[1];
            int src2Idx = i + j * bStrides[1];
            int dstIdx  = i + j * oStrides[1];

            // kernel algorithm goes here
            dst[dstIdx] = src1[src1Idx] + src2[src2Idx];
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
