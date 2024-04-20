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
#include <copy.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <mean.hpp>
#include <reduce.hpp>
#include <tile.hpp>
#include <unary.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/statistics.h>
#include <cmath>
#include <complex>

#include "stats.h"

using fly::dim4;
using flare::common::cast;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createValueArray;
using detail::division;
using detail::getScalar;
using detail::intl;
using detail::mean;
using detail::reduce;
using detail::reduce_all;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename inType, typename outType>
static outType stdev(const fly_array& in, const fly_var_bias bias) {
    using weightType        = typename baseOutType<outType>::type;
    const Array<inType> _in = getArray<inType>(in);
    Array<outType> input    = cast<outType>(_in);
    Array<outType> meanCnst = createValueArray<outType>(
        input.dims(), mean<inType, weightType, outType>(_in));
    Array<outType> diff =
        detail::arithOp<outType, fly_sub_t>(input, meanCnst, input.dims());
    Array<outType> diffSq =
        detail::arithOp<outType, fly_mul_t>(diff, diff, diff.dims());
    outType result = division(
        getScalar<outType>(reduce_all<fly_add_t, outType, outType>(diffSq)),
        (input.elements() - (bias == FLY_VARIANCE_SAMPLE)));
    return sqrt(result);
}

template<typename inType, typename outType>
static fly_array stdev(const fly_array& in, int dim, const fly_var_bias bias) {
    using weightType        = typename baseOutType<outType>::type;
    const Array<inType> _in = getArray<inType>(in);
    Array<outType> input    = cast<outType>(_in);
    dim4 iDims              = input.dims();

    Array<outType> meanArr = mean<inType, weightType, outType>(_in, dim);

    /* now tile meanArr along dim and use it for variance computation */
    dim4 tileDims(1);
    tileDims[dim]           = iDims[dim];
    Array<outType> tMeanArr = detail::tile<outType>(meanArr, tileDims);
    /* now mean array is ready */

    Array<outType> diff =
        detail::arithOp<outType, fly_sub_t>(input, tMeanArr, tMeanArr.dims());
    Array<outType> diffSq =
        detail::arithOp<outType, fly_mul_t>(diff, diff, diff.dims());
    Array<outType> redDiff = reduce<fly_add_t, outType, outType>(diffSq, dim);
    const dim4& oDims      = redDiff.dims();

    Array<outType> divArr = createValueArray<outType>(
        oDims, scalar<outType>((iDims[dim] - (bias == FLY_VARIANCE_SAMPLE))));
    Array<outType> varArr =
        detail::arithOp<outType, fly_div_t>(redDiff, divArr, redDiff.dims());
    Array<outType> result = detail::unaryOp<outType, fly_sqrt_t>(varArr);

    return getHandle<outType>(result);
}

// NOLINTNEXTLINE(readability-non-const-parameter)
fly_err fly_stdev_all(double* realVal, double* imagVal, const fly_array in) {
    return fly_stdev_all_v2(realVal, imagVal, in, FLY_VARIANCE_POPULATION);
}

fly_err fly_stdev_all_v2(double* realVal, double* imagVal, const fly_array in,
                       const fly_var_bias bias) {
    UNUSED(imagVal);  // TODO implement for complex values
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();
        switch (type) {
            case f64: *realVal = stdev<double, double>(in, bias); break;
            case f32: *realVal = stdev<float, float>(in, bias); break;
            case s32: *realVal = stdev<int, float>(in, bias); break;
            case u32: *realVal = stdev<uint, float>(in, bias); break;
            case s16: *realVal = stdev<short, float>(in, bias); break;
            case u16: *realVal = stdev<ushort, float>(in, bias); break;
            case s64: *realVal = stdev<intl, double>(in, bias); break;
            case u64: *realVal = stdev<uintl, double>(in, bias); break;
            case u8: *realVal = stdev<uchar, float>(in, bias); break;
            case b8: *realVal = stdev<char, float>(in, bias); break;
            // TODO(umar): FIXME: sqrt(complex) is not present in cuda
            // backend case c32: {
            //    cfloat tmp = stdev<cfloat,cfloat>(in);
            //    *realVal = real(tmp);
            //    *imagVal = imag(tmp);
            //    } break;
            // case c64: {
            //    cdouble tmp = stdev<cdouble,cdouble>(in);
            //    *realVal = real(tmp);
            //    *imagVal = imag(tmp);
            //    } break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_stdev(fly_array* out, const fly_array in, const dim_t dim) {
    return fly_stdev_v2(out, in, FLY_VARIANCE_POPULATION, dim);
}

fly_err fly_stdev_v2(fly_array* out, const fly_array in, const fly_var_bias bias,
                   const dim_t dim) {
    try {
        ARG_ASSERT(2, (dim >= 0 && dim <= 3));

        fly_array output       = 0;
        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();
        switch (type) {
            case f64: output = stdev<double, double>(in, dim, bias); break;
            case f32: output = stdev<float, float>(in, dim, bias); break;
            case s32: output = stdev<int, float>(in, dim, bias); break;
            case u32: output = stdev<uint, float>(in, dim, bias); break;
            case s16: output = stdev<short, float>(in, dim, bias); break;
            case u16: output = stdev<ushort, float>(in, dim, bias); break;
            case s64: output = stdev<intl, double>(in, dim, bias); break;
            case u64: output = stdev<uintl, double>(in, dim, bias); break;
            case u8: output = stdev<uchar, float>(in, dim, bias); break;
            case b8: output = stdev<char, float>(in, dim, bias); break;
            // TODO(umar): FIXME: sqrt(complex) is not present in cuda
            // backend case c32: output = stdev<cfloat,  cfloat>(in, dim);
            // break; case c64: output = stdev<cdouble,cdouble>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
