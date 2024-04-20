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
#include <common/err_common.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <math.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>

using fly::dim4;
using detail::arithOp;
using detail::Array;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::join;
using detail::scalar;

template<typename T>
static Array<T> mix(const Array<T>& X, const Array<T>& Y, double xf,
                    double yf) {
    const dim4& dims = X.dims();
    Array<T> xf_cnst = createValueArray<T>(dims, xf);
    Array<T> yf_cnst = createValueArray<T>(dims, yf);

    Array<T> fX = arithOp<T, fly_mul_t>(xf_cnst, X, dims);
    Array<T> fY = arithOp<T, fly_mul_t>(yf_cnst, Y, dims);

    return arithOp<T, fly_add_t>(fX, fY, dims);
}

template<typename T>
static Array<T> mix(const Array<T>& X, const Array<T>& Y, const Array<T>& Z,
                    double xf, double yf, double zf) {
    const dim4& dims = X.dims();
    Array<T> xf_cnst = createValueArray<T>(dims, xf);
    Array<T> yf_cnst = createValueArray<T>(dims, yf);
    Array<T> zf_cnst = createValueArray<T>(dims, zf);

    Array<T> fX = arithOp<T, fly_mul_t>(xf_cnst, X, dims);
    Array<T> fY = arithOp<T, fly_mul_t>(yf_cnst, Y, dims);
    Array<T> fZ = arithOp<T, fly_mul_t>(zf_cnst, Z, dims);

    Array<T> fx_fy = arithOp<T, fly_add_t>(fX, fY, dims);
    return arithOp<T, fly_add_t>(fx_fy, fZ, dims);
}

template<typename T>
static Array<T> digitize(const Array<T> ch, const double scale,
                         const double offset) {
    const dim4& dims = ch.dims();
    Array<T> base    = createValueArray<T>(dims, scalar<T>(offset));
    Array<T> cnst    = createValueArray<T>(dims, scalar<T>(scale));
    Array<T> scl     = arithOp<T, fly_mul_t>(ch, cnst, dims);
    return arithOp<T, fly_add_t>(scl, base, dims);
}

template<typename T, bool isYCbCr2RGB>
static fly_array convert(const fly_array& in, const fly_ycc_std standard) {
    static const float INV_219 = 0.004566210;
    static const float INV_112 = 0.008928571;
    const static float k[6]    = {0.1140f, 0.2990f, 0.0722f,
                                  0.2126f, 0.0593f, 0.2627f};
    unsigned stdIdx            = 0;  // Default standard is FLY_YCC_601
    switch (standard) {
        case FLY_YCC_709: stdIdx = 2; break;
        case FLY_YCC_2020: stdIdx = 4; break;
        default: stdIdx = 0; break;
    }
    float kb    = k[stdIdx];
    float kr    = k[stdIdx + 1];
    float kl    = 1.0f - kb - kr;
    float invKl = 1 / kl;

    // extract three channels as three slices
    // prepare sequence objects
    // get Array objects for corresponding channel views
    const Array<T> input = getArray<T>(in);
    std::vector<fly_seq> indices(4, fly_span);

    indices[2] = {0, 0, 1};
    Array<T> X = createSubArray(input, indices, false);

    indices[2] = {1, 1, 1};
    Array<T> Y = createSubArray(input, indices, false);

    indices[2] = {2, 2, 1};
    Array<T> Z = createSubArray(input, indices, false);

    if (isYCbCr2RGB) {
        const dim4& dims = X.dims();
        Array<T> yc      = createValueArray<T>(dims, 16);
        Array<T> cc      = createValueArray<T>(dims, 128);
        Array<T> Y_      = arithOp<T, fly_sub_t>(X, yc, dims);
        Array<T> Cb_     = arithOp<T, fly_sub_t>(Y, cc, dims);
        Array<T> Cr_     = arithOp<T, fly_sub_t>(Z, cc, dims);
        Array<T> R       = mix<T>(Y_, Cr_, INV_219, INV_112 * (1 - kr));
        Array<T> G =
            mix<T>(Y_, Cr_, Cb_, INV_219, INV_112 * (kr - 1) * kr * invKl,
                   INV_112 * (kb - 1) * kb * invKl);
        Array<T> B = mix<T>(Y_, Cb_, INV_219, INV_112 * (1 - kb));
        // join channels
        dim4 odims(R.dims()[0], R.dims()[1], 3);
        Array<T> rgbout = createEmptyArray<T>(odims);
        join<T>(rgbout, 2, {R, G, B});
        return getHandle(rgbout);
    }
    Array<T> Ey = mix<T>(X, Y, Z, kr, kl, kb);
    Array<T> Ecr =
        mix<T>(X, Y, Z, 0.5, 0.5 * kl / (kr - 1), 0.5 * kb / (kr - 1));
    Array<T> Ecb =
        mix<T>(X, Y, Z, 0.5 * kr / (kb - 1), 0.5 * kl / (kb - 1), 0.5);
    Array<T> Y_ = digitize<T>(Ey, 219.0, 16.0);
    Array<T> Cr = digitize<T>(Ecr, 224.0, 128.0);
    Array<T> Cb = digitize<T>(Ecb, 224.0, 128.0);
    // join channels
    dim4 odims(Y_.dims()[0], Y_.dims()[1], 3);
    Array<T> ycbcrout = createEmptyArray<T>(odims);
    join<T>(ycbcrout, 2, {Y_, Cb, Cr});
    return getHandle(ycbcrout);
}

template<bool isYCbCr2RGB>
fly_err convert(fly_array* out, const fly_array& in, const fly_ycc_std standard) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype iType        = info.getType();
        fly::dim4 inputDims    = info.dims();

        ARG_ASSERT(1, (inputDims.ndims() >= 3));

        fly_array output = 0;
        switch (iType) {
            case f64:
                output = convert<double, isYCbCr2RGB>(in, standard);
                break;
            case f32: output = convert<float, isYCbCr2RGB>(in, standard); break;
            default: TYPE_ERROR(1, iType); break;
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_ycbcr2rgb(fly_array* out, const fly_array in,
                    const fly_ycc_std standard) {
    return convert<true>(out, in, standard);
}

fly_err fly_rgb2ycbcr(fly_array* out, const fly_array in,
                    const fly_ycc_std standard) {
    return convert<false>(out, in, standard);
}
