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
#include <handle.hpp>
#include <math.hpp>
#include <mean.hpp>
#include <reduce.hpp>
#include <tile.hpp>
#include <unary.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/statistics.h>

#include "stats.h"

using fly::dim4;
using flare::common::cast;
using detail::arithOp;
using detail::Array;
using detail::createValueArray;
using detail::intl;
using detail::mean;
using detail::reduce;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T, typename cType>
static fly_array cov(const fly_array& X, const fly_array& Y,
                    const fly_var_bias bias) {
    using weightType  = typename baseOutType<cType>::type;
    const Array<T> _x = getArray<T>(X);
    const Array<T> _y = getArray<T>(Y);
    Array<cType> xArr = cast<cType>(_x);
    Array<cType> yArr = cast<cType>(_y);

    dim4 xDims = xArr.dims();
    dim_t N    = (bias == FLY_VARIANCE_SAMPLE ? xDims[0] - 1 : xDims[0]);

    Array<cType> xmArr =
        createValueArray<cType>(xDims, mean<T, weightType, cType>(_x));
    Array<cType> ymArr =
        createValueArray<cType>(xDims, mean<T, weightType, cType>(_y));
    Array<cType> nArr = createValueArray<cType>(xDims, scalar<cType>(N));

    Array<cType> diffX  = arithOp<cType, fly_sub_t>(xArr, xmArr, xDims);
    Array<cType> diffY  = arithOp<cType, fly_sub_t>(yArr, ymArr, xDims);
    Array<cType> mulXY  = arithOp<cType, fly_mul_t>(diffX, diffY, xDims);
    Array<cType> redArr = reduce<fly_add_t, cType, cType>(mulXY, 0);
    xDims[0]            = 1;
    Array<cType> result = arithOp<cType, fly_div_t>(redArr, nArr, xDims);

    return getHandle<cType>(result);
}

fly_err fly_cov(fly_array* out, const fly_array X, const fly_array Y,
              const bool isbiased) {
    const fly_var_bias bias =
        (isbiased ? FLY_VARIANCE_SAMPLE : FLY_VARIANCE_POPULATION);
    return fly_cov_v2(out, X, Y, bias);
}

fly_err fly_cov_v2(fly_array* out, const fly_array X, const fly_array Y,
                 const fly_var_bias bias) {
    try {
        const ArrayInfo& xInfo = getInfo(X);
        const ArrayInfo& yInfo = getInfo(Y);
        dim4 xDims             = xInfo.dims();
        dim4 yDims             = yInfo.dims();
        fly_dtype xType         = xInfo.getType();
        fly_dtype yType         = yInfo.getType();

        ARG_ASSERT(1, (xDims.ndims() <= 2));
        ARG_ASSERT(2, (xDims.ndims() == yDims.ndims()));
        ARG_ASSERT(2, (xDims[0] == yDims[0]));
        ARG_ASSERT(2, (xDims[1] == yDims[1]));
        ARG_ASSERT(2, (xType == yType));

        fly_array output = 0;
        switch (xType) {
            case f64: output = cov<double, double>(X, Y, bias); break;
            case f32: output = cov<float, float>(X, Y, bias); break;
            case s32: output = cov<int, float>(X, Y, bias); break;
            case u32: output = cov<uint, float>(X, Y, bias); break;
            case s64: output = cov<intl, double>(X, Y, bias); break;
            case u64: output = cov<uintl, double>(X, Y, bias); break;
            case s16: output = cov<short, float>(X, Y, bias); break;
            case u16: output = cov<ushort, float>(X, Y, bias); break;
            case u8: output = cov<uchar, float>(X, Y, bias); break;
            default: TYPE_ERROR(1, xType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
