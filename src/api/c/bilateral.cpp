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
#include <bilateral.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>

#include <type_traits>

using fly::dim4;
using detail::bilateral;
using detail::uchar;
using detail::uint;
using detail::ushort;
using std::conditional;
using std::is_same;

template<typename T>
inline fly_array bilateral(const fly_array &in, const float &sp_sig,
                          const float &chr_sig) {
    using OutType =
        typename conditional<is_same<T, double>::value, double, float>::type;
    return getHandle(bilateral<T, OutType>(getArray<T>(in), sp_sig, chr_sig));
}

fly_err fly_bilateral(fly_array *out, const fly_array in, const float ssigma,
                    const float csigma, const bool iscolor) {
    UNUSED(iscolor);
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 dims         = info.dims();

        DIM_ASSERT(1, (dims.ndims() >= 2));

        fly_array output = nullptr;
        switch (type) {
            case f64: output = bilateral<double>(in, ssigma, csigma); break;
            case f32: output = bilateral<float>(in, ssigma, csigma); break;
            case b8: output = bilateral<char>(in, ssigma, csigma); break;
            case s32: output = bilateral<int>(in, ssigma, csigma); break;
            case u32: output = bilateral<uint>(in, ssigma, csigma); break;
            case u8: output = bilateral<uchar>(in, ssigma, csigma); break;
            case s16: output = bilateral<short>(in, ssigma, csigma); break;
            case u16: output = bilateral<ushort>(in, ssigma, csigma); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
