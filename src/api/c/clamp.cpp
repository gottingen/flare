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

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <logic.hpp>
#include <optypes.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>

using fly::dim4;
using flare::common::half;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array clampOp(const fly_array in, const fly_array lo,
                               const fly_array hi, const dim4& odims) {
    const Array<T> L = castArray<T>(lo);
    const Array<T> H = castArray<T>(hi);
    const Array<T> I = castArray<T>(in);
    return getHandle(
        arithOp<T, fly_min_t>(arithOp<T, fly_max_t>(I, L, odims), H, odims));
}

fly_err fly_clamp(fly_array* out, const fly_array in, const fly_array lo,
                const fly_array hi, const bool batch) {
    try {
        const ArrayInfo& linfo = getInfo(lo);
        const ArrayInfo& hinfo = getInfo(hi);
        const ArrayInfo& iinfo = getInfo(in);

        DIM_ASSERT(2, linfo.dims() == hinfo.dims());
        TYPE_ASSERT(linfo.getType() == hinfo.getType());

        dim4 odims           = getOutDims(iinfo.dims(), linfo.dims(), batch);
        const fly_dtype otype = implicit(iinfo.getType(), linfo.getType());

        fly_array res;
        switch (otype) {
            case f32: res = clampOp<float>(in, lo, hi, odims); break;
            case f64: res = clampOp<double>(in, lo, hi, odims); break;
            case c32: res = clampOp<cfloat>(in, lo, hi, odims); break;
            case c64: res = clampOp<cdouble>(in, lo, hi, odims); break;
            case s32: res = clampOp<int>(in, lo, hi, odims); break;
            case u32: res = clampOp<uint>(in, lo, hi, odims); break;
            case u8: res = clampOp<uchar>(in, lo, hi, odims); break;
            case b8: res = clampOp<char>(in, lo, hi, odims); break;
            case s64: res = clampOp<intl>(in, lo, hi, odims); break;
            case u64: res = clampOp<uintl>(in, lo, hi, odims); break;
            case s16: res = clampOp<short>(in, lo, hi, odims); break;
            case u16: res = clampOp<ushort>(in, lo, hi, odims); break;
            case f16: res = clampOp<half>(in, lo, hi, odims); break;
            default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
