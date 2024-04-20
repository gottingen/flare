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

#include <Array.hpp>
#include <backend.hpp>
#include <common/half.hpp>
#include <common/indexing_helpers.hpp>
#include <handle.hpp>
#include <fly/array.h>
#include <fly/data.h>

#include <cassert>

using fly::dim4;
using flare::getArray;
using flare::common::flip;
using flare::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uintl;
using detail::ushort;
using std::swap;

template<typename T>
static inline fly_array flip(const fly_array in, const unsigned dim) {
    return getHandle(
        flip(getArray<T>(in), {dim == 0, dim == 1, dim == 2, dim == 3}));
}

fly_err fly_flip(fly_array *result, const fly_array in, const unsigned dim) {
    fly_array out;
    try {
        const ArrayInfo &in_info = getInfo(in);

        if (in_info.ndims() <= dim) {
            *result = retain(in);
            return FLY_SUCCESS;
        }

        fly_dtype in_type = in_info.getType();

        switch (in_type) {
            case f16: out = flip<half>(in, dim); break;
            case f32: out = flip<float>(in, dim); break;
            case c32: out = flip<cfloat>(in, dim); break;
            case f64: out = flip<double>(in, dim); break;
            case c64: out = flip<cdouble>(in, dim); break;
            case b8: out = flip<char>(in, dim); break;
            case s32: out = flip<int>(in, dim); break;
            case u32: out = flip<unsigned>(in, dim); break;
            case s64: out = flip<intl>(in, dim); break;
            case u64: out = flip<uintl>(in, dim); break;
            case s16: out = flip<short>(in, dim); break;
            case u16: out = flip<ushort>(in, dim); break;
            case u8: out = flip<uchar>(in, dim); break;
            default: TYPE_ERROR(1, in_type);
        }
        swap(*result, out);
    }
    CATCHALL

    return FLY_SUCCESS;
}
