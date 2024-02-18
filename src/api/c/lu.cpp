/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <lu.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/lapack.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::isLAPACKAvailable;

template<typename T>
static inline void lu(fly_array *lower, fly_array *upper, fly_array *pivot,
                      const fly_array in) {
    Array<T> lowerArray   = createEmptyArray<T>(fly::dim4());
    Array<T> upperArray   = createEmptyArray<T>(fly::dim4());
    Array<int> pivotArray = createEmptyArray<int>(fly::dim4());

    lu<T>(lowerArray, upperArray, pivotArray, getArray<T>(in));

    *lower = getHandle(lowerArray);
    *upper = getHandle(upperArray);
    *pivot = getHandle(pivotArray);
}

template<typename T>
static inline fly_array lu_inplace(fly_array in, bool is_lapack_piv) {
    return getHandle(lu_inplace<T>(getArray<T>(in), !is_lapack_piv));
}

fly_err fly_lu(fly_array *lower, fly_array *upper, fly_array *pivot,
             const fly_array in) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("lu can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();

        ARG_ASSERT(0, lower != nullptr);
        ARG_ASSERT(1, upper != nullptr);
        ARG_ASSERT(2, pivot != nullptr);
        ARG_ASSERT(3, i_info.isFloating());  // Only floating and complex types

        if (i_info.ndims() == 0) {
            FLY_CHECK(fly_create_handle(lower, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(upper, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(pivot, 0, nullptr, type));
            return FLY_SUCCESS;
        }

        switch (type) {
            case f32: lu<float>(lower, upper, pivot, in); break;
            case f64: lu<double>(lower, upper, pivot, in); break;
            case c32: lu<cfloat>(lower, upper, pivot, in); break;
            case c64: lu<cdouble>(lower, upper, pivot, in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_lu_inplace(fly_array *pivot, fly_array in, const bool is_lapack_piv) {
    try {
        const ArrayInfo &i_info = getInfo(in);
        fly_dtype type           = i_info.getType();

        if (i_info.ndims() > 2) {
            FLY_ERROR("lu can not be used in batch mode", FLY_ERR_BATCH);
        }

        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(0, pivot != nullptr);

        if (i_info.ndims() == 0) {
            return fly_create_handle(pivot, 0, nullptr, type);
        }

        fly_array out;
        switch (type) {
            case f32: out = lu_inplace<float>(in, is_lapack_piv); break;
            case f64: out = lu_inplace<double>(in, is_lapack_piv); break;
            case c32: out = lu_inplace<cfloat>(in, is_lapack_piv); break;
            case c64: out = lu_inplace<cdouble>(in, is_lapack_piv); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*pivot, out);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_is_lapack_available(bool *out) {
    try {
        *out = isLAPACKAvailable();
    }
    CATCHALL;

    return FLY_SUCCESS;
}
