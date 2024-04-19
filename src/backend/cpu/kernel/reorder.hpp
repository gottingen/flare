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

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void reorder(Param<T> out, CParam<T> in, const fly::dim4 oDims,
             const fly::dim4 rdims) {
    T* outPtr      = out.get();
    const T* inPtr = in.get();

    const fly::dim4 ist = in.strides();
    const fly::dim4 ost = out.strides();

    dim_t ids[4] = {0};
    for (dim_t ow = 0; ow < oDims[3]; ow++) {
        const dim_t oW = ow * ost[3];
        ids[rdims[3]]  = ow;
        for (dim_t oz = 0; oz < oDims[2]; oz++) {
            const dim_t oZW = oW + oz * ost[2];
            ids[rdims[2]]   = oz;
            for (dim_t oy = 0; oy < oDims[1]; oy++) {
                const dim_t oYZW = oZW + oy * ost[1];
                ids[rdims[1]]    = oy;
                for (dim_t ox = 0; ox < oDims[0]; ox++) {
                    const dim_t oIdx = oYZW + ox;

                    ids[rdims[0]]    = ox;
                    const dim_t iIdx = ids[3] * ist[3] + ids[2] * ist[2] +
                                       ids[1] * ist[1] + ids[0];

                    outPtr[oIdx] = inPtr[iIdx];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
