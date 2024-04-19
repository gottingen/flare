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
#include <blas.hpp>
#include <common/err_common.hpp>
#include <convolve.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include <vector>

using fly::dim4;
using detail::arithOp;
using detail::Array;
using detail::createEmptyArray;
using detail::createHostDataArray;
using detail::createSubArray;
using detail::scalar;

template<typename T>
Array<T> multiplyIndexed(const Array<T> &lhs, const Array<T> &rhs,
                         std::vector<fly_seq> idx) {
    Array<T> rhs_sub = createSubArray(rhs, idx);
    Array<T> out     = createEmptyArray<T>(
        dim4(lhs.dims()[0], rhs_sub.dims()[1], lhs.dims()[2], lhs.dims()[3]));
    T alpha = scalar<T>(1.0);
    T beta  = scalar<T>(0.0);
    gemm(out, FLY_MAT_NONE, FLY_MAT_NONE, &alpha, lhs, rhs_sub, &beta);
    return out;
}

template<typename T>
static fly_array transform_coordinates(const fly_array &tf_, const float d0_,
                                      const float d1_) {
    fly::dim4 h_dims(4, 3);
    T zero = 0;
    T one  = 1;
    T d0   = static_cast<T>(d0_);
    T d1   = static_cast<T>(d1_);
    // clang-format off
    T h_in[4 * 3] = {zero, zero,  d1,   d1,
                     zero,   d0,  d0, zero,
                      one,  one, one,  one};
    // clang-format on

    const Array<T> tf = getArray<T>(tf_);
    Array<T> in       = createHostDataArray<T>(h_dims, h_in);

    std::vector<fly_seq> idx(2);
    idx[0] = fly_make_seq(0, 2, 1);

    // w = 1.0 / matmul(tf, in(span, 2));
    // iw = matmul(tf, in(span, 2));
    idx[1]      = fly_make_seq(2, 2, 1);
    Array<T> iw = multiplyIndexed(in, tf, idx);

    // xt = w * matmul(tf, in(span, 0));
    // xt = matmul(tf, in(span, 0)) / iw;
    idx[1] = fly_make_seq(0, 0, 1);
    Array<T> xt =
        arithOp<T, fly_div_t>(multiplyIndexed(in, tf, idx), iw, iw.dims());

    // yt = w * matmul(tf, in(span, 1));
    // yt = matmul(tf, in(span, 1)) / iw;
    idx[1] = fly_make_seq(1, 1, 1);
    Array<T> yw =
        arithOp<T, fly_div_t>(multiplyIndexed(in, tf, idx), iw, iw.dims());

    // return join(1, xt, yt)
    Array<T> r = join(1, xt, yw);
    return getHandle(r);
}

fly_err fly_transform_coordinates(fly_array *out, const fly_array tf,
                                const float d0_, const float d1_) {
    try {
        const ArrayInfo &tfInfo = getInfo(tf);
        dim4 tfDims             = tfInfo.dims();
        ARG_ASSERT(1,
                   (tfDims[0] == 3 && tfDims[1] == 3 && tfDims.ndims() == 2));

        fly_array output;
        fly_dtype type = tfInfo.getType();
        switch (type) {
            case f32:
                output = transform_coordinates<float>(tf, d0_, d1_);
                break;
            case f64:
                output = transform_coordinates<double>(tf, d0_, d1_);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
