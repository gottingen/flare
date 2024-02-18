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
#include <diff.hpp>
#include <handle.hpp>
#include <fly/algorithm.h>
#include <fly/defines.h>

using fly::dim4;
using flare::getArray;
using flare::getHandle;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array diff1(const fly_array in, const int dim) {
    return getHandle(diff1<T>(getArray<T>(in), dim));
}

template<typename T>
static inline fly_array diff2(const fly_array in, const int dim) {
    return getHandle(diff2<T>(getArray<T>(in), dim));
}

fly_err fly_diff1(fly_array* out, const fly_array in, const int dim) {
    try {
        ARG_ASSERT(2, ((dim >= 0) && (dim < 4)));

        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();

        fly::dim4 in_dims = info.dims();
        if (in_dims[dim] < 2) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        DIM_ASSERT(1, in_dims[dim] >= 2);

        fly_array output;

        switch (type) {
            case f32: output = diff1<float>(in, dim); break;
            case c32: output = diff1<cfloat>(in, dim); break;
            case f64: output = diff1<double>(in, dim); break;
            case c64: output = diff1<cdouble>(in, dim); break;
            case b8: output = diff1<char>(in, dim); break;
            case s32: output = diff1<int>(in, dim); break;
            case u32: output = diff1<uint>(in, dim); break;
            case s64: output = diff1<intl>(in, dim); break;
            case u64: output = diff1<uintl>(in, dim); break;
            case s16: output = diff1<short>(in, dim); break;
            case u16: output = diff1<ushort>(in, dim); break;
            case u8: output = diff1<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_diff2(fly_array* out, const fly_array in, const int dim) {
    try {
        ARG_ASSERT(2, ((dim >= 0) && (dim < 4)));

        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();

        fly::dim4 in_dims = info.dims();
        if (in_dims[dim] < 3) {
            return fly_create_handle(out, 0, nullptr, type);
        }
        DIM_ASSERT(1, in_dims[dim] >= 3);

        fly_array output;

        switch (type) {
            case f32: output = diff2<float>(in, dim); break;
            case c32: output = diff2<cfloat>(in, dim); break;
            case f64: output = diff2<double>(in, dim); break;
            case c64: output = diff2<cdouble>(in, dim); break;
            case b8: output = diff2<char>(in, dim); break;
            case s32: output = diff2<int>(in, dim); break;
            case u32: output = diff2<uint>(in, dim); break;
            case s64: output = diff2<intl>(in, dim); break;
            case u64: output = diff2<uintl>(in, dim); break;
            case s16: output = diff2<short>(in, dim); break;
            case u16: output = diff2<ushort>(in, dim); break;
            case u8: output = diff2<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
