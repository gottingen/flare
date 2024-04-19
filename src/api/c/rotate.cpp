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
#include <rotate.hpp>
#include <fly/image.h>
#include <cmath>

using fly::dim4;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::cos;
using std::fabs;
using std::sin;

template<typename T>
static inline fly_array rotate(const fly_array in, const float theta,
                              const fly::dim4 &odims,
                              const fly_interp_type method) {
    return getHandle(rotate<T>(castArray<T>(in), theta, odims, method));
}

fly_err fly_rotate(fly_array *out, const fly_array in, const float theta,
                 const bool crop, const fly_interp_type method) {
    try {
        dim_t odims0 = 0, odims1 = 0;

        const ArrayInfo &info = getInfo(in);
        fly::dim4 idims        = info.dims();

        if (!crop) {
            odims0 = idims[0] * fabs(cos(theta)) + idims[1] * fabs(sin(theta));
            odims1 = idims[1] * fabs(cos(theta)) + idims[0] * fabs(sin(theta));
        } else {
            odims0 = idims[0];
            odims1 = idims[1];
        }

        fly_dtype itype = info.getType();

        ARG_ASSERT(4, method == FLY_INTERP_NEAREST ||
                          method == FLY_INTERP_BILINEAR ||
                          method == FLY_INTERP_BILINEAR_COSINE ||
                          method == FLY_INTERP_BICUBIC ||
                          method == FLY_INTERP_BICUBIC_SPLINE ||
                          method == FLY_INTERP_LOWER);

        if (idims.elements() == 0) { return fly_retain_array(out, in); }
        DIM_ASSERT(1, idims.elements() > 0);

        fly::dim4 odims(odims0, odims1, idims[2], idims[3]);

        fly_array output = 0;
        switch (itype) {
            case f32: output = rotate<float>(in, theta, odims, method); break;
            case f64: output = rotate<double>(in, theta, odims, method); break;
            case c32: output = rotate<cfloat>(in, theta, odims, method); break;
            case c64: output = rotate<cdouble>(in, theta, odims, method); break;
            case s32: output = rotate<int>(in, theta, odims, method); break;
            case u32: output = rotate<uint>(in, theta, odims, method); break;
            case s64: output = rotate<intl>(in, theta, odims, method); break;
            case u64: output = rotate<uintl>(in, theta, odims, method); break;
            case s16: output = rotate<short>(in, theta, odims, method); break;
            case u16: output = rotate<ushort>(in, theta, odims, method); break;
            case u8:
            case b8: output = rotate<uchar>(in, theta, odims, method); break;
            default: TYPE_ERROR(1, itype);
        }
        std::swap(*out, output);
    }
    CATCHALL

    return FLY_SUCCESS;
}
