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
#include <math.hpp>

#include <fly/dim4.hpp>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void diagCreate(Param<T> out, CParam<T> in, int const num) {
    int batch = in.dims(1);
    int size  = out.dims(0);

    T const *iptr = in.get();
    T *optr       = out.get();

    for (int k = 0; k < batch; k++) {
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                T val = scalar<T>(0);
                if (i == j - num) { val = (num > 0) ? iptr[i] : iptr[j]; }
                optr[i + j * out.strides(1)] = val;
            }
        }
        optr += out.strides(2);
        iptr += in.strides(1);
    }
}

template<typename T>
void diagExtract(Param<T> out, CParam<T> in, int const num) {
    fly::dim4 const odims = out.dims();
    fly::dim4 const idims = in.dims();

    int const i_off = (num > 0) ? (num * in.strides(1)) : (-num);

    for (int l = 0; l < (int)odims[3]; l++) {
        for (int k = 0; k < (int)odims[2]; k++) {
            const T *iptr =
                in.get() + l * in.strides(3) + k * in.strides(2) + i_off;
            T *optr = out.get() + l * out.strides(3) + k * out.strides(2);

            for (int i = 0; i < (int)odims[0]; i++) {
                T val = scalar<T>(0);
                if (i < idims[0] && i < idims[1])
                    val = iptr[i * in.strides(1) + i];
                optr[i] = val;
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
