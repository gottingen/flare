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
#include <common/err_common.hpp>
#include <handle.hpp>
#include <medfilt.hpp>
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include <fly/signal.h>

using fly::dim4;
using detail::uchar;
using detail::uint;
using detail::ushort;

fly_err fly_medfilt(fly_array *out, const fly_array in, const dim_t wind_length,
                  const dim_t wind_width, const fly_border_type edge_pad) {
    return fly_medfilt2(out, in, wind_length, wind_width, edge_pad);
}

template<typename T>
static fly_array medfilt1(fly_array const &in, dim_t w_wid,
                         fly_border_type edge_pad) {
    return getHandle<T>(
        medfilt1<T>(getArray<T>(in), static_cast<int>(w_wid), edge_pad));
}

fly_err fly_medfilt1(fly_array *out, const fly_array in, const dim_t wind_width,
                   const fly_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad >= FLY_PAD_ZERO && edge_pad <= FLY_PAD_SYM));

        const ArrayInfo &info = getInfo(in);
        fly::dim4 dims         = info.dims();

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 1));

        if (wind_width == 1) {
            *out = retain(in);
            return FLY_SUCCESS;
        }
        fly_array output = nullptr;
        fly_dtype type   = info.getType();
        switch (type) {
            case f32: output = medfilt1<float>(in, wind_width, edge_pad); break;
            case f64:
                output = medfilt1<double>(in, wind_width, edge_pad);
                break;
            case b8: output = medfilt1<char>(in, wind_width, edge_pad); break;
            case s32: output = medfilt1<int>(in, wind_width, edge_pad); break;
            case u32: output = medfilt1<uint>(in, wind_width, edge_pad); break;
            case s16: output = medfilt1<short>(in, wind_width, edge_pad); break;
            case u16:
                output = medfilt1<ushort>(in, wind_width, edge_pad);
                break;
            case u8: output = medfilt1<uchar>(in, wind_width, edge_pad); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
inline fly_array medfilt2(fly_array const &in, dim_t w_len, dim_t w_wid,
                         fly_border_type edge_pad) {
    return getHandle(medfilt2<T>(getArray<T>(in), static_cast<int>(w_len),
                                 static_cast<int>(w_wid), edge_pad));
}

fly_err fly_medfilt2(fly_array *out, const fly_array in, const dim_t wind_length,
                   const dim_t wind_width, const fly_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_length == wind_width));
        ARG_ASSERT(2, (wind_length > 0));
        ARG_ASSERT(3, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad >= FLY_PAD_ZERO && edge_pad <= FLY_PAD_SYM));

        const ArrayInfo &info = getInfo(in);
        fly::dim4 dims         = info.dims();

        if (info.isColumn()) {
            return fly_medfilt1(out, in, wind_width, edge_pad);
        }

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 2));

        if (wind_length == 1) {
            *out = retain(in);
            return FLY_SUCCESS;
        }
        fly_array output = nullptr;
        fly_dtype type   = info.getType();
        switch (type) {
            case f32:
                output = medfilt2<float>(in, wind_length, wind_width, edge_pad);
                break;
            case f64:
                output =
                    medfilt2<double>(in, wind_length, wind_width, edge_pad);
                break;
            case b8:
                output = medfilt2<char>(in, wind_length, wind_width, edge_pad);
                break;
            case s32:
                output = medfilt2<int>(in, wind_length, wind_width, edge_pad);
                break;
            case u32:
                output = medfilt2<uint>(in, wind_length, wind_width, edge_pad);
                break;
            case s16:
                output = medfilt2<short>(in, wind_length, wind_width, edge_pad);
                break;
            case u16:
                output =
                    medfilt2<ushort>(in, wind_length, wind_width, edge_pad);
                break;
            case u8:
                output = medfilt2<uchar>(in, wind_length, wind_width, edge_pad);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_minfilt(fly_array *out, const fly_array in, const dim_t wind_length,
                  const dim_t wind_width, const fly_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_length == wind_width));
        ARG_ASSERT(2, (wind_length > 0));
        ARG_ASSERT(3, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad == FLY_PAD_ZERO));

        const ArrayInfo &info = getInfo(in);
        fly::dim4 dims         = info.dims();

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 2));

        fly_array mask;
        dim_t wdims[] = {wind_length, wind_width};
        FLY_CHECK(fly_constant(&mask, 1, 2, wdims, info.getType()));

        FLY_CHECK(fly_erode(out, in, mask));

        FLY_CHECK(fly_release_array(mask));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_maxfilt(fly_array *out, const fly_array in, const dim_t wind_length,
                  const dim_t wind_width, const fly_border_type edge_pad) {
    try {
        ARG_ASSERT(2, (wind_length == wind_width));
        ARG_ASSERT(2, (wind_length > 0));
        ARG_ASSERT(3, (wind_width > 0));
        ARG_ASSERT(4, (edge_pad == FLY_PAD_ZERO));

        const ArrayInfo &info = getInfo(in);
        fly::dim4 dims         = info.dims();

        dim_t input_ndims = dims.ndims();
        DIM_ASSERT(1, (input_ndims >= 2));

        fly_array mask;
        dim_t wdims[] = {wind_length, wind_width};
        FLY_CHECK(fly_constant(&mask, 1, 2, wdims, info.getType()));

        FLY_CHECK(fly_dilate(out, in, mask));

        FLY_CHECK(fly_release_array(mask));
    }
    CATCHALL;

    return FLY_SUCCESS;
}
