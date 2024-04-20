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
#include <sobel.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include <utility>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

using ArrayPair = std::pair<fly_array, fly_array>;
template<typename Ti, typename To>
ArrayPair sobelDerivatives(const fly_array &in, const unsigned &ker_size) {
    using BAPair = std::pair<Array<To>, Array<To>>;
    BAPair out   = sobelDerivatives<Ti, To>(getArray<Ti>(in), ker_size);
    return std::make_pair(getHandle<To>(out.first), getHandle<To>(out.second));
}

fly_err fly_sobel_operator(fly_array *dx, fly_array *dy, const fly_array img,
                         const unsigned ker_size) {
    try {
        // FIXME: ADD SUPPORT FOR OTHER KERNEL SIZES
        // ARG_ASSERT(4, (ker_size==3 || ker_size==5 || ker_size==7));
        ARG_ASSERT(4, (ker_size == 3));

        const ArrayInfo &info = getInfo(img);
        fly::dim4 dims         = info.dims();

        DIM_ASSERT(3, (dims.ndims() >= 2));

        ArrayPair output;
        fly_dtype type = info.getType();
        switch (type) {
            case f32:
                output = sobelDerivatives<float, float>(img, ker_size);
                break;
            case f64:
                output = sobelDerivatives<double, double>(img, ker_size);
                break;
            case s32: output = sobelDerivatives<int, int>(img, ker_size); break;
            case u32:
                output = sobelDerivatives<uint, int>(img, ker_size);
                break;
            case s16:
                output = sobelDerivatives<short, int>(img, ker_size);
                break;
            case u16:
                output = sobelDerivatives<ushort, int>(img, ker_size);
                break;
            case b8: output = sobelDerivatives<char, int>(img, ker_size); break;
            case u8:
                output = sobelDerivatives<uchar, int>(img, ker_size);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*dx, output.first);
        std::swap(*dy, output.second);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
