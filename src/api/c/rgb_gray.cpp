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

#include <fly/data.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include <fly/index.h>

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <common/tile.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <math.hpp>

using fly::dim4;
using flare::common::cast;
using detail::arithOp;
using detail::Array;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::join;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T, typename cType>
static fly_array rgb2gray(const fly_array& in, const float r, const float g,
                         const float b) {
    Array<cType> input = cast<cType>(getArray<T>(in));
    dim4 inputDims     = input.dims();
    dim4 matDims(inputDims[0], inputDims[1], 1, inputDims[3]);

    Array<cType> rCnst = createValueArray<cType>(matDims, scalar<cType>(r));
    Array<cType> gCnst = createValueArray<cType>(matDims, scalar<cType>(g));
    Array<cType> bCnst = createValueArray<cType>(matDims, scalar<cType>(b));

    std::vector<fly_seq> slice1(4, fly_span), slice2(4, fly_span),
        slice3(4, fly_span);
    // extract three channels as three slices
    slice1[2] = {0, 0, 1};
    slice2[2] = {1, 1, 1};
    slice3[2] = {2, 2, 1};

    Array<cType> ch1Temp = createSubArray(input, slice1);
    Array<cType> ch2Temp = createSubArray(input, slice2);
    Array<cType> ch3Temp = createSubArray(input, slice3);

    // r*Slice0
    Array<cType> expr1 = arithOp<cType, fly_mul_t>(ch1Temp, rCnst, matDims);
    // g*Slice1
    Array<cType> expr2 = arithOp<cType, fly_mul_t>(ch2Temp, gCnst, matDims);
    // b*Slice2
    Array<cType> expr3 = arithOp<cType, fly_mul_t>(ch3Temp, bCnst, matDims);
    // r*Slice0 + g*Slice1
    Array<cType> expr4 = arithOp<cType, fly_add_t>(expr1, expr2, matDims);
    // r*Slice0 + g*Slice1 + b*Slice2
    Array<cType> result = arithOp<cType, fly_add_t>(expr3, expr4, matDims);

    return getHandle<cType>(result);
}

template<typename T, typename cType>
static fly_array gray2rgb(const fly_array& in, const float r, const float g,
                         const float b) {
    if (r == 1.0 && g == 1.0 && b == 1.0) {
        dim4 tileDims(1, 1, 3, 1);
        return getHandle(flare::common::tile(getArray<T>(in), tileDims));
    }

    fly_array mod_input = 0;
    dim4 inputDims     = getInfo(in).dims();

    dim4 matDims(inputDims[0], inputDims[1], 1, inputDims[2] * inputDims[3]);

    FLY_CHECK(fly_moddims(&mod_input, in, matDims.ndims(), matDims.get()));
    Array<cType> mod_in = cast<cType>(getArray<cType>(mod_input));

    Array<cType> rCnst = createValueArray<cType>(matDims, scalar<cType>(r));
    Array<cType> gCnst = createValueArray<cType>(matDims, scalar<cType>(g));
    Array<cType> bCnst = createValueArray<cType>(matDims, scalar<cType>(b));

    Array<cType> expr1 = arithOp<cType, fly_mul_t>(mod_in, rCnst, matDims);
    Array<cType> expr2 = arithOp<cType, fly_mul_t>(mod_in, gCnst, matDims);
    Array<cType> expr3 = arithOp<cType, fly_mul_t>(mod_in, bCnst, matDims);

    FLY_CHECK(fly_release_array(mod_input));

    // join channels
    dim4 odims(expr1.dims()[0], expr1.dims()[1], 3);
    Array<cType> out = createEmptyArray<cType>(odims);
    join<cType>(out, 2, {expr3, expr1, expr2});
    return getHandle(out);
}

template<typename T, typename cType, bool isRGB2GRAY>
static fly_array convert(const fly_array& in, const float r, const float g,
                        const float b) {
    if (isRGB2GRAY) {
        return rgb2gray<T, cType>(in, r, g, b);
    } else {
        return gray2rgb<T, cType>(in, r, g, b);
    }
}

template<bool isRGB2GRAY>
fly_err convert(fly_array* out, const fly_array in, const float r, const float g,
               const float b) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype iType        = info.getType();
        fly::dim4 inputDims    = info.dims();

        // 2D is not required.
        if (info.elements() == 0) {
            return fly_create_handle(out, 0, nullptr, iType);
        }

        // If RGB is input, then assert 3 channels
        // else 1 channel
        if (isRGB2GRAY) {
            ARG_ASSERT(1, (inputDims[2] == 3));
        } else {
            ARG_ASSERT(1, (inputDims[2] == 1));
        }

        fly_array output = 0;
        switch (iType) {
            case f64:
                output = convert<double, double, isRGB2GRAY>(in, r, g, b);
                break;
            case f32:
                output = convert<float, float, isRGB2GRAY>(in, r, g, b);
                break;
            case u32:
                output = convert<uint, float, isRGB2GRAY>(in, r, g, b);
                break;
            case s32:
                output = convert<int, float, isRGB2GRAY>(in, r, g, b);
                break;
            case u16:
                output = convert<ushort, float, isRGB2GRAY>(in, r, g, b);
                break;
            case s16:
                output = convert<short, float, isRGB2GRAY>(in, r, g, b);
                break;
            case u8:
                output = convert<uchar, float, isRGB2GRAY>(in, r, g, b);
                break;
            default: TYPE_ERROR(1, iType); break;
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_rgb2gray(fly_array* out, const fly_array in, const float rPercent,
                   const float gPercent, const float bPercent) {
    return convert<true>(out, in, rPercent, gPercent, bPercent);
}

fly_err fly_gray2rgb(fly_array* out, const fly_array in, const float rFactor,
                   const float gFactor, const float bFactor) {
    return convert<false>(out, in, rFactor, gFactor, bFactor);
}
