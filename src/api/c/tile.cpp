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

#include <common/tile.hpp>

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <unary.hpp>
#include <fly/arith.h>
#include <fly/data.h>

using fly::dim4;
using flare::common::half;
using flare::common::tile;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::unaryOp;
using detail::ushort;

template<typename T>
static inline fly_array tile(const fly_array in, const fly::dim4 &tileDims) {
    return getHandle(flare::common::tile<T>(getArray<T>(in), tileDims));
}

fly_err fly_tile(fly_array *out, const fly_array in, const fly::dim4 &tileDims) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (info.ndims() == 0) { return fly_retain_array(out, in); }
        DIM_ASSERT(1, info.dims().elements() > 0);
        DIM_ASSERT(2, tileDims.elements() > 0);

        fly_array output;

        switch (type) {
            case f32: output = tile<float>(in, tileDims); break;
            case c32: output = tile<cfloat>(in, tileDims); break;
            case f64: output = tile<double>(in, tileDims); break;
            case c64: output = tile<cdouble>(in, tileDims); break;
            case b8: output = tile<char>(in, tileDims); break;
            case s32: output = tile<int>(in, tileDims); break;
            case u32: output = tile<uint>(in, tileDims); break;
            case s64: output = tile<intl>(in, tileDims); break;
            case u64: output = tile<uintl>(in, tileDims); break;
            case s16: output = tile<short>(in, tileDims); break;
            case u16: output = tile<ushort>(in, tileDims); break;
            case u8: output = tile<uchar>(in, tileDims); break;
            case f16: output = tile<half>(in, tileDims); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_tile(fly_array *out, const fly_array in, const unsigned x,
               const unsigned y, const unsigned z, const unsigned w) {
    fly::dim4 tileDims(x, y, z, w);
    return fly_tile(out, in, tileDims);
}
