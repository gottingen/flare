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

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void gradient(Param<T> grad0, Param<T> grad1, CParam<T> in) {
    const fly::dim4 dims = in.dims();

    T *d_grad0    = grad0.get();
    T *d_grad1    = grad1.get();
    const T *d_in = in.get();

    const fly::dim4 inst = in.strides();
    const fly::dim4 g0st = grad0.strides();
    const fly::dim4 g1st = grad1.strides();

    T v5 = scalar<T>(0.5);
    T v1 = scalar<T>(1.0);

    for (dim_t idw = 0; idw < dims[3]; idw++) {
        const dim_t inW = idw * inst[3];
        const dim_t g0W = idw * g0st[3];
        const dim_t g1W = idw * g1st[3];
        for (dim_t idz = 0; idz < dims[2]; idz++) {
            const dim_t inZW = inW + idz * inst[2];
            const dim_t g0ZW = g0W + idz * g0st[2];
            const dim_t g1ZW = g1W + idz * g1st[2];
            dim_t xl, xr, yl, yr;
            T f0, f1;
            for (dim_t idy = 0; idy < dims[1]; idy++) {
                const dim_t inYZW = inZW + idy * inst[1];
                const dim_t g0YZW = g0ZW + idy * g0st[1];
                const dim_t g1YZW = g1ZW + idy * g1st[1];
                if (idy == 0) {
                    yl = inYZW + inst[1];
                    yr = inYZW;
                    f1 = v1;
                } else if (idy == dims[1] - 1) {
                    yl = inYZW;
                    yr = inYZW - inst[1];
                    f1 = v1;
                } else {
                    yl = inYZW + inst[1];
                    yr = inYZW - inst[1];
                    f1 = v5;
                }
                for (dim_t idx = 0; idx < dims[0]; idx++) {
                    const dim_t inMem = inYZW + idx;
                    const dim_t g0Mem = g0YZW + idx;
                    const dim_t g1Mem = g1YZW + idx;
                    if (idx == 0) {
                        xl = inMem + 1;
                        xr = inMem;
                        f0 = v1;
                    } else if (idx == dims[0] - 1) {
                        xl = inMem;
                        xr = inMem - 1;
                        f0 = v1;
                    } else {
                        xl = inMem + 1;
                        xr = inMem - 1;
                        f0 = v5;
                    }

                    d_grad0[g0Mem] = f0 * (d_in[xl] - d_in[xr]);
                    d_grad1[g1Mem] = f1 * (d_in[yl + idx] - d_in[yr + idx]);
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
