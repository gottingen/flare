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

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <diagonal.hpp>
#include <handle.hpp>
#include <lu.hpp>
#include <math.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/lapack.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::imag;
using detail::real;
using detail::scalar;

template<typename T>
T det(const fly_array a) {
    using namespace detail;
    const Array<T> A = getArray<T>(a);

    const int num = A.dims()[0];

    if (num == 0) {
        T res = scalar<T>(1.0);
        return res;
    }

    std::vector<T> hD(num);
    std::vector<int> hP(num);

    Array<T> D       = createEmptyArray<T>(dim4());
    Array<int> pivot = createEmptyArray<int>(dim4());

    // Free memory as soon as possible
    {
        Array<T> A_copy = copyArray<T>(A);

        Array<int> pivot = lu_inplace(A_copy, false);
        copyData(&hP[0], pivot);

        Array<T> D = diagExtract(A_copy, 0);
        copyData(&hD[0], D);
    }

    bool is_neg = false;
    T res       = scalar<T>(is_neg ? -1 : 1);
    for (int i = 0; i < num; i++) {
        res = res * hD[i];
        is_neg ^= (hP[i] != (i + 1));
    }

    if (is_neg) { res = res * scalar<T>(-1); }

    return res;
}

fly_err fly_det(double *real_val, double *imag_val, const fly_array in) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("solve can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();

        if (i_info.dims()[0]) {
            DIM_ASSERT(1, i_info.dims()[0] ==
                              i_info.dims()[1]);  // Only square matrices
        }
        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types

        *real_val = 0;
        *imag_val = 0;

        cfloat cfval;
        cdouble cdval;

        switch (type) {
            case f32: *real_val = det<float>(in); break;
            case f64: *real_val = det<double>(in); break;
            case c32:
                cfval     = det<cfloat>(in);
                *real_val = real(cfval);
                *imag_val = imag(cfval);
                break;
            case c64:
                cdval     = det<cdouble>(in);
                *real_val = real(cdval);
                *imag_val = imag(cdval);
                break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
