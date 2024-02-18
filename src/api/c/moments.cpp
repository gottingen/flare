/*******************************************************
 * Copyright (c) 2016, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/data.h>
#include <fly/image.h>
#include <fly/index.h>

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <moments.hpp>
#include <reorder.hpp>
#include <tile.hpp>

#include <limits>
#include <vector>

using fly::dim4;

using detail::Array;
using std::vector;

template<typename T>
static inline void moments(fly_array* out, const fly_array in,
                           fly_moment_type moment) {
    Array<float> temp = moments<T>(getArray<T>(in), moment);
    *out              = getHandle<float>(temp);
}

fly_err fly_moments(fly_array* out, const fly_array in,
                  const fly_moment_type moment) {
    try {
        const ArrayInfo& in_info = getInfo(in);
        fly_dtype type            = in_info.getType();

        switch (type) {
            case f32: moments<float>(out, in, moment); break;
            case f64: moments<double>(out, in, moment); break;
            case u32: moments<unsigned>(out, in, moment); break;
            case s32: moments<int>(out, in, moment); break;
            case u16: moments<unsigned short>(out, in, moment); break;
            case s16: moments<short>(out, in, moment); break;
            case b8: moments<char>(out, in, moment); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline void moment_copy(double* out, const fly_array moments) {
    const auto& info = getInfo(moments);
    vector<T> h_moments(info.elements());
    copyData(h_moments.data(), moments);

    // convert to double
    copy(begin(h_moments), end(h_moments), out);
}

fly_err fly_moments_all(double* out, const fly_array in,
                      const fly_moment_type moment) {
    try {
        const ArrayInfo& in_info = getInfo(in);
        dim4 idims               = in_info.dims();
        DIM_ASSERT(1, idims[2] == 1 && idims[3] == 1);

        fly_array moments_arr;
        FLY_CHECK(fly_moments(&moments_arr, in, moment));
        moment_copy<float>(out, moments_arr);
        FLY_CHECK(fly_release_array(moments_arr));
    }
    CATCHALL;

    return FLY_SUCCESS;
}
