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
#include <histogram.hpp>
#include <fly/dim4.hpp>
#include <fly/image.h>

using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
inline fly_array histogram(const fly_array in, const unsigned &nbins,
                          const double &minval, const double &maxval,
                          const bool islinear) {
    return getHandle(
        histogram<T>(getArray<T>(in), nbins, minval, maxval, islinear));
}

fly_err fly_histogram(fly_array *out, const fly_array in, const unsigned nbins,
                    const double minval, const double maxval) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (info.ndims() == 0) { return fly_retain_array(out, in); }

        fly_array output;
        switch (type) {
            case f32:
                output = histogram<float>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case f64:
                output = histogram<double>(in, nbins, minval, maxval,
                                           info.isLinear());
                break;
            case b8:
                output =
                    histogram<char>(in, nbins, minval, maxval, info.isLinear());
                break;
            case s32:
                output =
                    histogram<int>(in, nbins, minval, maxval, info.isLinear());
                break;
            case u32:
                output =
                    histogram<uint>(in, nbins, minval, maxval, info.isLinear());
                break;
            case s16:
                output = histogram<short>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case u16:
                output = histogram<ushort>(in, nbins, minval, maxval,
                                           info.isLinear());
                break;
            case s64:
                output =
                    histogram<intl>(in, nbins, minval, maxval, info.isLinear());
                break;
            case u64:
                output = histogram<uintl>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case u8:
                output = histogram<uchar>(in, nbins, minval, maxval,
                                          info.isLinear());
                break;
            case f16:
                output = histogram<flare::common::half>(
                    in, nbins, minval, maxval, info.isLinear());
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
