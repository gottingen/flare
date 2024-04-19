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
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <optypes.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>

using fly::dim4;
using flare::castSparse;
using flare::getHandle;
using flare::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

static fly_array cast(const fly_array in, const fly_dtype type) {
    const ArrayInfo& info = getInfo(in, false);

    if (info.getType() == type) { return retain(in); }

    if (info.isSparse()) {
        switch (type) {
            case f32: return getHandle(castSparse<float>(in));
            case f64: return getHandle(castSparse<double>(in));
            case c32: return getHandle(castSparse<cfloat>(in));
            case c64: return getHandle(castSparse<cdouble>(in));
            default: TYPE_ERROR(2, type);
        }
    } else {
        switch (type) {
            case f32: return getHandle(castArray<float>(in));
            case f64: return getHandle(castArray<double>(in));
            case c32: return getHandle(castArray<cfloat>(in));
            case c64: return getHandle(castArray<cdouble>(in));
            case s32: return getHandle(castArray<int>(in));
            case u32: return getHandle(castArray<uint>(in));
            case u8: return getHandle(castArray<uchar>(in));
            case b8: return getHandle(castArray<char>(in));
            case s64: return getHandle(castArray<intl>(in));
            case u64: return getHandle(castArray<uintl>(in));
            case s16: return getHandle(castArray<short>(in));
            case u16: return getHandle(castArray<ushort>(in));
            case f16: return getHandle(castArray<half>(in));
            default: TYPE_ERROR(2, type);
        }
    }
}

fly_err fly_cast(fly_array* out, const fly_array in, const fly_dtype type) {
    try {
        const ArrayInfo& info = getInfo(in, false);

        fly_dtype inType = info.getType();
        if ((inType == c32 || inType == c64) &&
            (type == f32 || type == f64 || type == f16)) {
            FLY_ERROR(
                "Casting is not allowed from complex (c32/c64) to real "
                "(f16/f32/f64) types.\n"
                "Use abs, real, imag etc to convert complex to floating type.",
                FLY_ERR_TYPE);
        }

        dim4 idims = info.dims();
        if (idims.elements() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        fly_array res = cast(in, type);

        std::swap(*out, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
