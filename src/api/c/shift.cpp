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
#include <handle.hpp>
#include <shift.hpp>
#include <fly/data.h>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array shift(const fly_array in, const int sdims[4]) {
    return getHandle(shift<T>(getArray<T>(in), sdims));
}

fly_err fly_shift(fly_array *out, const fly_array in, const int sdims[4]) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (info.ndims() == 0) { return fly_retain_array(out, in); }
        DIM_ASSERT(1, info.elements() > 0);

        fly_array output;

        switch (type) {
            case f32: output = shift<float>(in, sdims); break;
            case c32: output = shift<cfloat>(in, sdims); break;
            case f64: output = shift<double>(in, sdims); break;
            case c64: output = shift<cdouble>(in, sdims); break;
            case b8: output = shift<char>(in, sdims); break;
            case s32: output = shift<int>(in, sdims); break;
            case u32: output = shift<uint>(in, sdims); break;
            case s64: output = shift<intl>(in, sdims); break;
            case u64: output = shift<uintl>(in, sdims); break;
            case s16: output = shift<short>(in, sdims); break;
            case u16: output = shift<ushort>(in, sdims); break;
            case u8: output = shift<uchar>(in, sdims); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_shift(fly_array *out, const fly_array in, const int x, const int y,
                const int z, const int w) {
    const int sdims[] = {x, y, z, w};
    return fly_shift(out, in, sdims);
}
