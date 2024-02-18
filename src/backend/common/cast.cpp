/*******************************************************
 * Copyright (c) 2021, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/cast.hpp>
#include <handle.hpp>

using flare::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

namespace flare {
namespace common {

template<typename To>
detail::Array<To> castArray(const fly_array &in) {
    const ArrayInfo &info = getInfo(in);

    if (static_cast<fly::dtype>(fly::dtype_traits<To>::fly_type) ==
        info.getType()) {
        return getArray<To>(in);
    }

    switch (info.getType()) {
        case f32: return common::cast<To, float>(getArray<float>(in));
        case f64: return common::cast<To, double>(getArray<double>(in));
        case c32: return common::cast<To, cfloat>(getArray<cfloat>(in));
        case c64: return common::cast<To, cdouble>(getArray<cdouble>(in));
        case s32: return common::cast<To, int>(getArray<int>(in));
        case u32: return common::cast<To, uint>(getArray<uint>(in));
        case u8: return common::cast<To, uchar>(getArray<uchar>(in));
        case b8: return common::cast<To, char>(getArray<char>(in));
        case s64: return common::cast<To, intl>(getArray<intl>(in));
        case u64: return common::cast<To, uintl>(getArray<uintl>(in));
        case s16: return common::cast<To, short>(getArray<short>(in));
        case u16: return common::cast<To, ushort>(getArray<ushort>(in));
        case f16:
            return common::cast<To, common::half>(getArray<common::half>(in));
        default: TYPE_ERROR(1, info.getType());
    }
}

template detail::Array<float> castArray(const fly_array &in);
template detail::Array<double> castArray(const fly_array &in);
template detail::Array<cfloat> castArray(const fly_array &in);
template detail::Array<cdouble> castArray(const fly_array &in);
template detail::Array<int> castArray(const fly_array &in);
template detail::Array<uint> castArray(const fly_array &in);
template detail::Array<uchar> castArray(const fly_array &in);
template detail::Array<char> castArray(const fly_array &in);
template detail::Array<intl> castArray(const fly_array &in);
template detail::Array<uintl> castArray(const fly_array &in);
template detail::Array<short> castArray(const fly_array &in);
template detail::Array<ushort> castArray(const fly_array &in);
template detail::Array<half> castArray(const fly_array &in);

}  // namespace common
}  // namespace flare
