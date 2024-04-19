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

#include <common/err_common.hpp>
#include <handle.hpp>
#include <imgproc_common.hpp>
#include <fly/defines.h>
#include <fly/image.h>

using fly::dim4;
using flare::common::integralImage;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename To, typename Ti>
inline fly_array sat(const fly_array& in) {
    return getHandle<To>(integralImage<To, Ti>(getArray<Ti>(in)));
}

fly_err fly_sat(fly_array* out, const fly_array in) {
    try {
        const ArrayInfo& info = getInfo(in);
        const dim4& dims      = info.dims();

        ARG_ASSERT(1, (dims.ndims() >= 2));

        fly_dtype inputType = info.getType();

        fly_array output = 0;
        switch (inputType) {
            case f64: output = sat<double, double>(in); break;
            case f32: output = sat<float, float>(in); break;
            case s32: output = sat<int, int>(in); break;
            case u32: output = sat<uint, uint>(in); break;
            case b8: output = sat<int, char>(in); break;
            case u8: output = sat<uint, uchar>(in); break;
            case s64: output = sat<intl, intl>(in); break;
            case u64: output = sat<uintl, uintl>(in); break;
            case s16: output = sat<int, short>(in); break;
            case u16: output = sat<uint, ushort>(in); break;
            default: TYPE_ERROR(1, inputType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
