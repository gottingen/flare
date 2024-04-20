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

fly::dim4 calcOffset(const fly::dim4 dims, int dim) {
    fly::dim4 offset;
    offset[0] = (dim == 0) ? dims[0] : 0;
    offset[1] = (dim == 1) ? dims[1] : 0;
    offset[2] = (dim == 2) ? dims[2] : 0;
    offset[3] = (dim == 3) ? dims[3] : 0;
    return offset;
}

template<typename T>
void join_append(T *out, const T *X, const fly::dim4 &offset,
                 const fly::dim4 &xdims, const fly::dim4 &ost,
                 const fly::dim4 &xst) {
    for (dim_t ow = 0; ow < xdims[3]; ow++) {
        const dim_t xW = ow * xst[3];
        const dim_t oW = (ow + offset[3]) * ost[3];

        for (dim_t oz = 0; oz < xdims[2]; oz++) {
            const dim_t xZW = xW + oz * xst[2];
            const dim_t oZW = oW + (oz + offset[2]) * ost[2];

            for (dim_t oy = 0; oy < xdims[1]; oy++) {
                const dim_t xYZW = xZW + oy * xst[1];
                const dim_t oYZW = oZW + (oy + offset[1]) * ost[1];

                memcpy(out + oYZW + offset[0], X + xYZW, xdims[0] * sizeof(T));
            }
        }
    }
}

template<typename T>
void join(const int dim, Param<T> out, const std::vector<CParam<T>> inputs,
          int n_arrays) {
    fly::dim4 zero(0, 0, 0, 0);
    fly::dim4 d = zero;
    join_append<T>(out.get(), inputs[0].get(), zero, inputs[0].dims(),
                   out.strides(), inputs[0].strides());
    for (int i = 1; i < n_arrays; i++) {
        d += inputs[i - 1].dims();
        join_append<T>(out.get(), inputs[i].get(), calcOffset(d, dim),
                       inputs[i].dims(), out.strides(), inputs[i].strides());
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
