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
#include <resize.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/image.h>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array resize(const fly_array in, const dim_t odim0,
                              const dim_t odim1, const fly_interp_type method) {
    return getHandle(resize<T>(getArray<T>(in), odim0, odim1, method));
}

fly_err fly_resize(fly_array* out, const fly_array in, const dim_t odim0,
                 const dim_t odim1, const fly_interp_type method) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();

        ARG_ASSERT(4, method == FLY_INTERP_NEAREST ||
                          method == FLY_INTERP_BILINEAR ||
                          method == FLY_INTERP_BILINEAR_COSINE ||
                          method == FLY_INTERP_BICUBIC ||
                          method == FLY_INTERP_BICUBIC_SPLINE ||
                          method == FLY_INTERP_LOWER);

        DIM_ASSERT(2, odim0 > 0);
        DIM_ASSERT(3, odim1 > 0);

        bool is_resize_supported =
            (method == FLY_INTERP_LOWER || method == FLY_INTERP_NEAREST ||
             method == FLY_INTERP_BILINEAR);

        if (!is_resize_supported) {
            // Fall back to scale for additional methods
            return fly_scale(out, in, 0, 0, odim0, odim1, method);
        }

        fly_array output;

        switch (type) {
            case f32: output = resize<float>(in, odim0, odim1, method); break;
            case f64: output = resize<double>(in, odim0, odim1, method); break;
            case c32: output = resize<cfloat>(in, odim0, odim1, method); break;
            case c64: output = resize<cdouble>(in, odim0, odim1, method); break;
            case s32: output = resize<int>(in, odim0, odim1, method); break;
            case u32: output = resize<uint>(in, odim0, odim1, method); break;
            case s64: output = resize<intl>(in, odim0, odim1, method); break;
            case u64: output = resize<uintl>(in, odim0, odim1, method); break;
            case s16: output = resize<short>(in, odim0, odim1, method); break;
            case u16: output = resize<ushort>(in, odim0, odim1, method); break;
            case u8: output = resize<uchar>(in, odim0, odim1, method); break;
            case b8: output = resize<char>(in, odim0, odim1, method); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
