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
void iir(Param<T> y, Param<T> c, CParam<T> a) {
    dim4 ydims = c.dims();
    int num_a  = a.dims(0);

    for (int l = 0; l < (int)ydims[3]; l++) {
        dim_t yidx3 = l * y.strides(3);
        dim_t cidx3 = l * c.strides(3);
        dim_t aidx3 = l * a.strides(3);

        for (int k = 0; k < (int)ydims[2]; k++) {
            dim_t yidx2 = k * y.strides(2) + yidx3;
            dim_t cidx2 = k * c.strides(2) + cidx3;
            dim_t aidx2 = k * a.strides(2) + aidx3;

            for (int j = 0; j < (int)ydims[1]; j++) {
                dim_t yidx1 = j * y.strides(1) + yidx2;
                dim_t cidx1 = j * c.strides(1) + cidx2;
                dim_t aidx1 = j * a.strides(1) + aidx2;

                std::vector<T> h_z(num_a);

                const T *h_a = a.get() + (a.dims().ndims() > 1 ? aidx1 : 0);
                T *h_c       = c.get() + cidx1;
                T *h_y       = y.get() + yidx1;

                for (int i = 0; i < (int)ydims[0]; i++) {
                    T y = h_y[i] = (h_c[i] + h_z[0]) / h_a[0];
                    for (int ii = 1; ii < num_a; ii++) {
                        h_z[ii - 1] = h_z[ii] - h_a[ii] * y;
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
