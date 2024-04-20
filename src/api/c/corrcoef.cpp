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
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <stats.h>
#include <types.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/statistics.h>

#include <cmath>

using fly::dim4;
using flare::common::cast;
using detail::arithOp;
using detail::Array;
using detail::getScalar;
using detail::intl;
using detail::reduce_all;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename Ti, typename To>
static To corrcoef(const fly_array& X, const fly_array& Y) {
    Array<To> xIn = cast<To>(getArray<Ti>(X));
    Array<To> yIn = cast<To>(getArray<Ti>(Y));

    const dim4& dims = xIn.dims();
    dim_t n          = xIn.elements();

    To xSum = getScalar<To>(reduce_all<fly_add_t, To, To>(xIn));
    To ySum = getScalar<To>(reduce_all<fly_add_t, To, To>(yIn));

    Array<To> xSq = arithOp<To, fly_mul_t>(xIn, xIn, dims);
    Array<To> ySq = arithOp<To, fly_mul_t>(yIn, yIn, dims);
    Array<To> xy  = arithOp<To, fly_mul_t>(xIn, yIn, dims);

    To xSqSum = getScalar<To>(reduce_all<fly_add_t, To, To>(xSq));
    To ySqSum = getScalar<To>(reduce_all<fly_add_t, To, To>(ySq));
    To xySum  = getScalar<To>(reduce_all<fly_add_t, To, To>(xy));

    To result =
        (n * xySum - xSum * ySum) / (std::sqrt(n * xSqSum - xSum * xSum) *
                                     std::sqrt(n * ySqSum - ySum * ySum));

    return result;
}

// NOLINTNEXTLINE
fly_err fly_corrcoef(double* realVal, double* imagVal, const fly_array X,
                   const fly_array Y) {
    UNUSED(imagVal);  // TODO(umar): implement for complex types
    try {
        const ArrayInfo& xInfo = getInfo(X);
        const ArrayInfo& yInfo = getInfo(Y);
        dim4 xDims             = xInfo.dims();
        dim4 yDims             = yInfo.dims();
        fly_dtype xType         = xInfo.getType();
        fly_dtype yType         = yInfo.getType();

        ARG_ASSERT(2, (xType == yType));
        ARG_ASSERT(2, (xDims.ndims() == yDims.ndims()));

        for (dim_t i = 0; i < xDims.ndims(); ++i) {
            ARG_ASSERT(2, (xDims[i] == yDims[i]));
        }

        switch (xType) {
            case f64: *realVal = corrcoef<double, double>(X, Y); break;
            case f32: *realVal = corrcoef<float, float>(X, Y); break;
            case s32: *realVal = corrcoef<int, float>(X, Y); break;
            case u32: *realVal = corrcoef<uint, float>(X, Y); break;
            case s64: *realVal = corrcoef<intl, double>(X, Y); break;
            case u64: *realVal = corrcoef<uintl, double>(X, Y); break;
            case s16: *realVal = corrcoef<short, float>(X, Y); break;
            case u16: *realVal = corrcoef<ushort, float>(X, Y); break;
            case u8: *realVal = corrcoef<uchar, float>(X, Y); break;
            case b8: *realVal = corrcoef<char, float>(X, Y); break;
            default: TYPE_ERROR(1, xType);
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}
