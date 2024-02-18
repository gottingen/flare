/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <histogram.hpp>
#include <fly/dim4.hpp>
#include <fly/image.h>

using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
inline fly_array histogram(const fly_array in, const unsigned &nbins,
                          const double &minval, const double &maxval,
                          const bool islinear) {
    return getHandle(
        histogram<T>(getArray<T>(in), nbins, minval, maxval, islinear));
}

fly_err fly_histogram(fly_array *out, const fly_array in, const unsigned nbins,
                    const double minval, const double maxval) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (info.ndims() == 0) { return fly_retain_array(out, in); }

        fly_array output;
        switch (type) {
            case f32:
                output = histogram<float>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case f64:
                output = histogram<double>(in, nbins, minval, maxval,
                                           info.isLinear());
                break;
            case b8:
                output =
                    histogram<char>(in, nbins, minval, maxval, info.isLinear());
                break;
            case s32:
                output =
                    histogram<int>(in, nbins, minval, maxval, info.isLinear());
                break;
            case u32:
                output =
                    histogram<uint>(in, nbins, minval, maxval, info.isLinear());
                break;
            case s16:
                output = histogram<short>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case u16:
                output = histogram<ushort>(in, nbins, minval, maxval,
                                           info.isLinear());
                break;
            case s64:
                output =
                    histogram<intl>(in, nbins, minval, maxval, info.isLinear());
                break;
            case u64:
                output = histogram<uintl>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case u8:
                output = histogram<uchar>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case f16:
                output = histogram<flare::common::half>(
                    in, nbins, minval, maxval, info.isLinear());
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
