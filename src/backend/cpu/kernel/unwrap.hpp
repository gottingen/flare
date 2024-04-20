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
#include <err_cpu.hpp>
#include <math.hpp>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void unwrap_dim(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
                const dim_t dx, const dim_t dy, const int d) {
    const T *inPtr = in.get();
    T *outPtr      = out.get();

    fly::dim4 idims    = in.dims();
    fly::dim4 odims    = out.dims();
    fly::dim4 istrides = in.strides();
    fly::dim4 ostrides = out.strides();

    dim_t nx = 1 + (idims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;

    for (dim_t w = 0; w < odims[3]; w++) {
        for (dim_t z = 0; z < odims[2]; z++) {
            dim_t cOut    = w * ostrides[3] + z * ostrides[2];
            dim_t cIn     = w * istrides[3] + z * istrides[2];
            const T *iptr = inPtr + cIn;
            T *optr_      = outPtr + cOut;

            for (dim_t col = 0; col < odims[d]; col++) {
                // Offset output ptr
                T *optr = optr_ + col * ostrides[d];

                // Calculate input window index
                dim_t winy = (col / nx);
                dim_t winx = (col % nx);

                dim_t startx = winx * sx;
                dim_t starty = winy * sy;

                dim_t spx = startx - px;
                dim_t spy = starty - py;

                // Short cut condition ensuring all values within input
                // dimensions
                bool cond = (spx >= 0 && spx + (wx * dx) < idims[0] &&
                             spy >= 0 && spy + (wy * dy) < idims[1]);

                for (dim_t y = 0; y < wy; y++) {
                    dim_t ypad = spy + y * dy;
                    for (dim_t x = 0; x < wx; x++) {
                        dim_t xpad = spx + x * dx;

                        dim_t oloc = (y * wx + x);
                        if (d == 0) oloc *= ostrides[1];

                        if (cond || (xpad >= 0 && xpad < idims[0] &&
                                     ypad >= 0 && ypad < idims[1])) {
                            dim_t iloc =
                                (ypad * istrides[1] + xpad * istrides[0]);
                            optr[oloc] = iptr[iloc];
                        } else {
                            optr[oloc] = scalar<T>(0.0);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare