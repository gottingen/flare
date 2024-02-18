/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cholesky.hpp>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/lapack.h>

using flare::getArray;
using detail::cdouble;
using detail::cfloat;

template<typename T>
static inline fly_array cholesky(int *info, const fly_array in,
                                const bool is_upper) {
    return getHandle(cholesky<T>(info, getArray<T>(in), is_upper));
}

template<typename T>
static inline int cholesky_inplace(fly_array in, const bool is_upper) {
    return cholesky_inplace<T>(getArray<T>(in), is_upper);
}

fly_err fly_cholesky(fly_array *out, int *info, const fly_array in,
                   const bool is_upper) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("cholesky can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();

        if (i_info.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }
        DIM_ASSERT(
            1, i_info.dims()[0] == i_info.dims()[1]);  // Only square matrices
        ARG_ASSERT(2, i_info.isFloating());  // Only floating and complex types

        fly_array output;
        switch (type) {
            case f32: output = cholesky<float>(info, in, is_upper); break;
            case f64: output = cholesky<double>(info, in, is_upper); break;
            case c32: output = cholesky<cfloat>(info, in, is_upper); break;
            case c64: output = cholesky<cdouble>(info, in, is_upper); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_cholesky_inplace(int *info, fly_array in, const bool is_upper) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("cholesky can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();
        if (i_info.ndims() == 0) { return FLY_SUCCESS; }
        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types
        DIM_ASSERT(
            1, i_info.dims()[0] == i_info.dims()[1]);  // Only square matrices

        int out;

        switch (type) {
            case f32: out = cholesky_inplace<float>(in, is_upper); break;
            case f64: out = cholesky_inplace<double>(in, is_upper); break;
            case c32: out = cholesky_inplace<cfloat>(in, is_upper); break;
            case c64: out = cholesky_inplace<cdouble>(in, is_upper); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*info, out);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
