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
#include <copy.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <range.hpp>
#include <reduce.hpp>
#include <transpose.hpp>
#include <unary.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>

using fly::dim4;
using detail::arithOp;
using detail::Array;
using detail::createValueArray;
using detail::getScalar;
using detail::range;
using detail::reduce_all;
using detail::scalar;
using detail::transpose;
using detail::unaryOp;

template<typename T>
Array<T> gaussianKernel(const int rows, const int cols, const double sigma_r,
                        const double sigma_c) {
    const dim4 odims = dim4(rows, cols);
    double sigma     = 0;

    Array<T> tmp  = createValueArray<T>(odims, scalar<T>(0));
    Array<T> half = createValueArray<T>(odims, 0.5);
    Array<T> zero = createValueArray<T>(odims, scalar<T>(0));

    if (cols > 1) {
        Array<T> wt = range<T>(dim4(cols, rows), 0);
        Array<T> w  = transpose<T>(wt, false);

        Array<T> c = createValueArray<T>(
            odims, scalar<T>(static_cast<double>(cols - 1) / 2.0));
        w = arithOp<T, fly_sub_t>(w, c, odims);

        sigma        = sigma_c > 0 ? sigma_c : 0.25 * cols;
        Array<T> sig = createValueArray<T>(odims, sigma);
        w            = arithOp<T, fly_div_t>(w, sig, odims);

        w   = arithOp<T, fly_mul_t>(w, w, odims);
        tmp = arithOp<T, fly_add_t>(w, tmp, odims);
    }

    if (rows > 1) {
        Array<T> w = range<T>(dim4(rows, cols), 0);

        Array<T> r = createValueArray<T>(
            odims, scalar<T>(static_cast<double>(rows - 1) / 2.0));
        w = arithOp<T, fly_sub_t>(w, r, odims);

        sigma        = sigma_r > 0 ? sigma_r : 0.25 * rows;
        Array<T> sig = createValueArray<T>(odims, sigma);

        w   = arithOp<T, fly_div_t>(w, sig, odims);
        w   = arithOp<T, fly_mul_t>(w, w, odims);
        tmp = arithOp<T, fly_add_t>(w, tmp, odims);
    }

    tmp = arithOp<T, fly_mul_t>(half, tmp, odims);
    tmp = arithOp<T, fly_sub_t>(zero, tmp, odims);
    tmp = unaryOp<T, fly_exp_t>(tmp);

    // Use this instead of (2 * pi * sig^2);
    // This ensures the window adds up to 1
    T norm_factor = getScalar<T>(reduce_all<fly_add_t, T, T>(tmp));

    Array<T> norm = createValueArray(odims, norm_factor);
    Array<T> res  = arithOp<T, fly_div_t>(tmp, norm, odims);

    return res;
}

fly_err fly_gaussian_kernel(fly_array *out, const int rows, const int cols,
                          const double sigma_r, const double sigma_c) {
    try {
        fly_array res;
        res = getHandle<float>(
            gaussianKernel<float>(rows, cols, sigma_r, sigma_c));
        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
