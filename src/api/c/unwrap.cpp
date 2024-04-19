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
#include <unwrap.hpp>
#include <fly/defines.h>
#include <fly/image.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array unwrap(const fly_array in, const dim_t wx, const dim_t wy,
                              const dim_t sx, const dim_t sy, const dim_t px,
                              const dim_t py, const bool is_column) {
    return getHandle(
        unwrap<T>(getArray<T>(in), wx, wy, sx, sy, px, py, 1, 1, is_column));
}

fly_err fly_unwrap(fly_array* out, const fly_array in, const dim_t wx,
                 const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px,
                 const dim_t py, const bool is_column) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 idims        = info.dims();

        ARG_ASSERT(2, wx > 0 && wx <= idims[0] + 2 * px);
        ARG_ASSERT(3, wy > 0 && wy <= idims[1] + 2 * py);
        ARG_ASSERT(4, sx > 0);
        ARG_ASSERT(5, sy > 0);
        ARG_ASSERT(6, px >= 0 && px < wx);
        ARG_ASSERT(7, py >= 0 && py < wy);

        fly_array output;

        switch (type) {
            case f32:
                output = unwrap<float>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case f64:
                output = unwrap<double>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case c32:
                output = unwrap<cfloat>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case c64:
                output = unwrap<cdouble>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case s32:
                output = unwrap<int>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case u32:
                output = unwrap<uint>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case s64:
                output = unwrap<intl>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case u64:
                output = unwrap<uintl>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case s16:
                output = unwrap<short>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case u16:
                output = unwrap<ushort>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case u8:
                output = unwrap<uchar>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            case b8:
                output = unwrap<char>(in, wx, wy, sx, sy, px, py, is_column);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
