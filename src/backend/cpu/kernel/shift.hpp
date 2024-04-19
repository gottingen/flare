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
#include <cassert>

namespace flare {
namespace cpu {
namespace kernel {

static inline dim_t simple_mod(const dim_t i, const dim_t dim) {
    return (i < dim) ? i : (i - dim);
}

template<typename T>
void shift(Param<T> out, CParam<T> in, const fly::dim4 sdims) {
    T* outPtr      = out.get();
    const T* inPtr = in.get();

    const fly::dim4 oDims = out.dims();
    const fly::dim4 ist   = in.strides();
    const fly::dim4 ost   = out.strides();

    int sdims_[4];
    // Need to do this because we are mapping output to input in the kernel
    for (int i = 0; i < 4; i++) {
        // sdims_[i] will always be positive and always [0, oDims[i]].
        // Negative shifts are converted to position by going the other way
        // round
        sdims_[i] = -(sdims[i] % (int)oDims[i]) + oDims[i] * (sdims[i] > 0);
        assert(sdims_[i] >= 0 && sdims_[i] <= oDims[i]);
    }

    for (dim_t ow = 0; ow < oDims[3]; ow++) {
        const int oW = ow * ost[3];
        const int iw = simple_mod((ow + sdims_[3]), oDims[3]);
        const int iW = iw * ist[3];
        for (dim_t oz = 0; oz < oDims[2]; oz++) {
            const int oZW = oW + oz * ost[2];
            const int iz  = simple_mod((oz + sdims_[2]), oDims[2]);
            const int iZW = iW + iz * ist[2];
            for (dim_t oy = 0; oy < oDims[1]; oy++) {
                const int oYZW = oZW + oy * ost[1];
                const int iy   = simple_mod((oy + sdims_[1]), oDims[1]);
                const int iYZW = iZW + iy * ist[1];
                for (dim_t ox = 0; ox < oDims[0]; ox++) {
                    const int oIdx = oYZW + ox;
                    const int ix   = simple_mod((ox + sdims_[0]), oDims[0]);
                    const int iIdx = iYZW + ix;

                    outPtr[oIdx] = inPtr[iIdx];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
