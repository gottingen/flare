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
#include <vector>

namespace flare {
namespace cpu {
namespace kernel {

template<typename InT, typename IndexT>
void lookup(Param<InT> out, CParam<InT> input, CParam<IndexT> indices,
            unsigned const dim) {
    const fly::dim4 iDims    = input.dims();
    const fly::dim4 oDims    = out.dims();
    const fly::dim4 iStrides = input.strides();
    const fly::dim4 oStrides = out.strides();
    const InT *inPtr        = input.get();
    const IndexT *idxPtr    = indices.get();

    InT *outPtr = out.get();

    for (dim_t l = 0; l < oDims[3]; ++l) {
        dim_t iLOff = iStrides[3] *
                      (dim == 3 ? trimIndex((dim_t)idxPtr[l], iDims[3]) : l);
        dim_t oLOff = l * oStrides[3];

        for (dim_t k = 0; k < oDims[2]; ++k) {
            dim_t iKOff =
                iStrides[2] *
                (dim == 2 ? trimIndex((dim_t)idxPtr[k], iDims[2]) : k);
            dim_t oKOff = k * oStrides[2];

            for (dim_t j = 0; j < oDims[1]; ++j) {
                dim_t iJOff =
                    iStrides[1] *
                    (dim == 1 ? trimIndex((dim_t)idxPtr[j], iDims[1]) : j);
                dim_t oJOff = j * oStrides[1];

                for (dim_t i = 0; i < oDims[0]; ++i) {
                    dim_t iIOff =
                        iStrides[0] *
                        (dim == 0 ? trimIndex((dim_t)idxPtr[i], iDims[0]) : i);
                    dim_t oIOff = i * oStrides[0];

                    outPtr[oLOff + oKOff + oJOff + oIOff] =
                        inPtr[iLOff + iKOff + iJOff + iIOff];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
