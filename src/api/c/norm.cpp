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

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <lu.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <fly/array.h>
#include <fly/constants.h>
#include <fly/defines.h>
#include <fly/lapack.h>
#include <fly/traits.hpp>

using fly::dim4;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::getScalar;
using detail::reduce;
using detail::reduce_all;
using detail::scalar;

template<typename T>
double matrixNorm(const Array<T> &A, double p) {
    if (p == 1) {
        Array<T> colSum = reduce<fly_add_t, T, T>(A, 0);
        return getScalar<T>(reduce_all<fly_max_t, T, T>(colSum));
    }
    if (p == fly::Inf) {
        Array<T> rowSum = reduce<fly_add_t, T, T>(A, 1);
        return getScalar<T>(reduce_all<fly_max_t, T, T>(rowSum));
    }

    FLY_ERROR("This type of norm is not supported in Flare\n",
             FLY_ERR_NOT_SUPPORTED);
}

template<typename T>
double vectorNorm(const Array<T> &A, double p) {
    if (p == 1) { return getScalar<T>(reduce_all<fly_add_t, T, T>(A)); }
    if (p == fly::Inf) {
        return getScalar<T>(reduce_all<fly_max_t, T, T>(A));
    } else if (p == 2) {
        Array<T> A_sq = arithOp<T, fly_mul_t>(A, A, A.dims());
        return std::sqrt(getScalar<T>(reduce_all<fly_add_t, T, T>(A_sq)));
    }

    Array<T> P   = createValueArray<T>(A.dims(), scalar<T>(p));
    Array<T> A_p = arithOp<T, fly_pow_t>(A, P, A.dims());
    return std::pow(getScalar<T>(reduce_all<fly_add_t, T, T>(A_p)), T(1.0 / p));
}

template<typename T>
double LPQNorm(const Array<T> &A, double p, double q) {
    Array<T> A_p_norm = createEmptyArray<T>(dim4());

    if (p == 1) {
        A_p_norm = reduce<fly_add_t, T, T>(A, 0);
    } else {
        Array<T> P    = createValueArray<T>(A.dims(), scalar<T>(p));
        Array<T> invP = createValueArray<T>(A.dims(), scalar<T>(1.0 / p));

        Array<T> A_p     = arithOp<T, fly_pow_t>(A, P, A.dims());
        Array<T> A_p_sum = reduce<fly_add_t, T, T>(A_p, 0);
        A_p_norm         = arithOp<T, fly_pow_t>(A_p_sum, invP, invP.dims());
    }

    if (q == 1) { return getScalar<T>(reduce_all<fly_add_t, T, T>(A_p_norm)); }

    Array<T> Q          = createValueArray<T>(A_p_norm.dims(), scalar<T>(q));
    Array<T> A_p_norm_q = arithOp<T, fly_pow_t>(A_p_norm, Q, Q.dims());

    return std::pow(getScalar<T>(reduce_all<fly_add_t, T, T>(A_p_norm_q)),
                    T(1.0 / q));
}

template<typename T>
double norm(const fly_array a, const fly_norm_type type, const double p,
            const double q) {
    using BT = typename fly::dtype_traits<T>::base_type;

    const Array<BT> A = detail::abs<BT, T>(getArray<T>(a));

    switch (type) {
        case FLY_NORM_EUCLID: return vectorNorm(A, 2);

        case FLY_NORM_VECTOR_1: return vectorNorm(A, 1);

        case FLY_NORM_VECTOR_INF: return vectorNorm(A, fly::Inf);

        case FLY_NORM_VECTOR_P: return vectorNorm(A, p);

        case FLY_NORM_MATRIX_1: return matrixNorm(A, 1);

        case FLY_NORM_MATRIX_INF: return matrixNorm(A, fly::Inf);

        case FLY_NORM_MATRIX_2: return matrixNorm(A, 2);

        case FLY_NORM_MATRIX_L_PQ: return LPQNorm(A, p, q);

        default:
            FLY_ERROR("This type of norm is not supported in Flare\n",
                     FLY_ERR_NOT_SUPPORTED);
    }
}

fly_err fly_norm(double *out, const fly_array in, const fly_norm_type type,
               const double p, const double q) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("solve can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype i_type = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types

        *out = 0;

        if (i_info.ndims() == 0) { return FLY_SUCCESS; }

        switch (i_type) {
            case f32: *out = norm<float>(in, type, p, q); break;
            case f64: *out = norm<double>(in, type, p, q); break;
            case c32: *out = norm<cfloat>(in, type, p, q); break;
            case c64: *out = norm<cdouble>(in, type, p, q); break;
            default: TYPE_ERROR(1, i_type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
