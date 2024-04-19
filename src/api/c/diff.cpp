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
#include <diff.hpp>
#include <handle.hpp>
#include <fly/algorithm.h>
#include <fly/defines.h>

using fly::dim4;
using flare::getArray;
using flare::getHandle;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array diff1(const fly_array in, const int dim) {
    return getHandle(diff1<T>(getArray<T>(in), dim));
}

template<typename T>
static inline fly_array diff2(const fly_array in, const int dim) {
    return getHandle(diff2<T>(getArray<T>(in), dim));
}

fly_err fly_diff1(fly_array* out, const fly_array in, const int dim) {
    try {
        ARG_ASSERT(2, ((dim >= 0) && (dim < 4)));

        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();

        fly::dim4 in_dims = info.dims();
        if (in_dims[dim] < 2) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        DIM_ASSERT(1, in_dims[dim] >= 2);

        fly_array output;

        switch (type) {
            case f32: output = diff1<float>(in, dim); break;
            case c32: output = diff1<cfloat>(in, dim); break;
            case f64: output = diff1<double>(in, dim); break;
            case c64: output = diff1<cdouble>(in, dim); break;
            case b8: output = diff1<char>(in, dim); break;
            case s32: output = diff1<int>(in, dim); break;
            case u32: output = diff1<uint>(in, dim); break;
            case s64: output = diff1<intl>(in, dim); break;
            case u64: output = diff1<uintl>(in, dim); break;
            case s16: output = diff1<short>(in, dim); break;
            case u16: output = diff1<ushort>(in, dim); break;
            case u8: output = diff1<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_diff2(fly_array* out, const fly_array in, const int dim) {
    try {
        ARG_ASSERT(2, ((dim >= 0) && (dim < 4)));

        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();

        fly::dim4 in_dims = info.dims();
        if (in_dims[dim] < 3) {
            return fly_create_handle(out, 0, nullptr, type);
        }
        DIM_ASSERT(1, in_dims[dim] >= 3);

        fly_array output;

        switch (type) {
            case f32: output = diff2<float>(in, dim); break;
            case c32: output = diff2<cfloat>(in, dim); break;
            case f64: output = diff2<double>(in, dim); break;
            case c64: output = diff2<cdouble>(in, dim); break;
            case b8: output = diff2<char>(in, dim); break;
            case s32: output = diff2<int>(in, dim); break;
            case u32: output = diff2<uint>(in, dim); break;
            case s64: output = diff2<intl>(in, dim); break;
            case u64: output = diff2<uintl>(in, dim); break;
            case s16: output = diff2<short>(in, dim); break;
            case u16: output = diff2<ushort>(in, dim); break;
            case u8: output = diff2<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
