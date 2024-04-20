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
#include <hsv_rgb.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>

using fly::dim4;
using detail::Array;
using detail::hsv2rgb;
using detail::rgb2hsv;

template<typename T, bool isHSV2RGB>
static fly_array convert(const fly_array& in) {
    const Array<T> input = getArray<T>(in);
    if (isHSV2RGB) {
        return getHandle<T>(hsv2rgb<T>(input));
    } else {
        return getHandle<T>(rgb2hsv<T>(input));
    }
}

template<bool isHSV2RGB>
fly_err convert(fly_array* out, const fly_array& in) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype iType        = info.getType();
        fly::dim4 inputDims    = info.dims();

        if (info.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, iType);
        }

        ARG_ASSERT(1, (inputDims.ndims() >= 3));

        fly_array output = 0;
        switch (iType) {
            case f64: output = convert<double, isHSV2RGB>(in); break;
            case f32: output = convert<float, isHSV2RGB>(in); break;
            default: TYPE_ERROR(1, iType); break;
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_hsv2rgb(fly_array* out, const fly_array in) {
    return convert<true>(out, in);
}

fly_err fly_rgb2hsv(fly_array* out, const fly_array in) {
    return convert<false>(out, in);
}
