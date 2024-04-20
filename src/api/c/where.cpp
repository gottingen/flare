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
#include <where.hpp>
#include <fly/algorithm.h>
#include <fly/dim4.hpp>
#include <complex>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::swap;

template<typename T>
static inline fly_array where(const fly_array in) {
    // Making it more explicit that the output is uint
    return getHandle<uint>(where<T>(getArray<T>(in)));
}

fly_err fly_where(fly_array* idx, const fly_array in) {
    try {
        const ArrayInfo& i_info = getInfo(in);
        fly_dtype type           = i_info.getType();

        if (i_info.ndims() == 0) {
            return fly_create_handle(idx, 0, nullptr, u32);
        }

        fly_array res;
        switch (type) {
            case f32: res = where<float>(in); break;
            case f64: res = where<double>(in); break;
            case c32: res = where<cfloat>(in); break;
            case c64: res = where<cdouble>(in); break;
            case s32: res = where<int>(in); break;
            case u32: res = where<uint>(in); break;
            case s64: res = where<intl>(in); break;
            case u64: res = where<uintl>(in); break;
            case s16: res = where<short>(in); break;
            case u16: res = where<ushort>(in); break;
            case u8: res = where<uchar>(in); break;
            case b8: res = where<char>(in); break;
            default: TYPE_ERROR(1, type);
        }
        swap(*idx, res);
    }
    CATCHALL

    return FLY_SUCCESS;
}
