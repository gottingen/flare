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
#include <wrap.hpp>
#include <fly/defines.h>
#include <fly/image.h>

using fly::dim4;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline void wrap(fly_array* out, const fly_array in, const dim_t wx,
                        const dim_t wy, const dim_t sx, const dim_t sy,
                        const dim_t px, const dim_t py, const bool is_column) {
    wrap<T>(getArray<T>(*out), getArray<T>(in), wx, wy, sx, sy, px, py,
            is_column);
}

void fly_wrap_common(fly_array* out, const fly_array in, const dim_t ox,
                    const dim_t oy, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy, const dim_t px,
                    const dim_t py, const bool is_column, bool allocate_out) {
    ARG_ASSERT(0, out != 0);  // *out (the fly_array) can be null, but not out
    ARG_ASSERT(1, in != 0);

    const ArrayInfo& info  = getInfo(in);
    const fly_dtype in_type = info.getType();
    const dim4& in_dims    = info.dims();
    const dim4 out_dims(ox, oy, in_dims[2], in_dims[3]);

    ARG_ASSERT(4, wx > 0);
    ARG_ASSERT(5, wy > 0);
    ARG_ASSERT(6, sx > 0);
    ARG_ASSERT(7, sy > 0);

    const dim_t nx = (ox + 2 * px - wx) / sx + 1;
    const dim_t ny = (oy + 2 * py - wy) / sy + 1;

    const dim_t patch_size  = is_column ? in_dims[0] : in_dims[1];
    const dim_t num_patches = is_column ? in_dims[1] : in_dims[0];

    DIM_ASSERT(1, patch_size == wx * wy);
    DIM_ASSERT(1, num_patches == nx * ny);

    if (allocate_out) { *out = createHandleFromValue(out_dims, 0.0, in_type); }

    // The out pointer can be passed in to the function by the user
    DIM_ASSERT(0, getInfo(*out).dims() == out_dims);

    // clang-format off
    switch(in_type) {
        case f32: wrap<float  >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case f64: wrap<double >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case c32: wrap<cfloat >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case c64: wrap<cdouble>(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case s32: wrap<int    >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case u32: wrap<uint   >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case s64: wrap<intl   >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case u64: wrap<uintl  >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case s16: wrap<short  >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case u16: wrap<ushort >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case u8:  wrap<uchar  >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        case b8:  wrap<char   >(out, in, wx, wy, sx, sy, px, py, is_column);  break;
        default:  TYPE_ERROR(1, in_type);
    }
    // clang-format on
}

fly_err fly_wrap(fly_array* out, const fly_array in, const dim_t ox, const dim_t oy,
               const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
               const dim_t px, const dim_t py, const bool is_column) {
    try {
        fly_wrap_common(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column,
                       true);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_wrap_v2(fly_array* out, const fly_array in, const dim_t ox,
                  const dim_t oy, const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy, const dim_t px,
                  const dim_t py, const bool is_column) {
    try {
        ARG_ASSERT(0, out != 0);  // need to dereference out in next call
        fly_wrap_common(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column,
                       *out == 0);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
