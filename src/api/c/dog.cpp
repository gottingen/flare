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

#include <Array.hpp>
#include <arith.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <convolve.hpp>
#include <handle.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include <fly/vision.h>

using fly::dim4;
using detail::arithOp;
using detail::Array;
using detail::convolve;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T, typename accT>
static fly_array dog(const fly_array& in, const int radius1, const int radius2) {
    fly_array g1, g2;
    g1 = g2 = 0;
    FLY_CHECK(
        fly_gaussian_kernel(&g1, 2 * radius1 + 1, 2 * radius1 + 1, 0.0, 0.0));
    FLY_CHECK(
        fly_gaussian_kernel(&g2, 2 * radius2 + 1, 2 * radius2 + 1, 0.0, 0.0));

    Array<accT> input = castArray<accT>(in);
    dim4 iDims        = input.dims();

    FLY_BATCH_KIND bkind = iDims[2] > 1 ? FLY_BATCH_LHS : FLY_BATCH_NONE;

    Array<accT> smth1 =
        convolve<accT, accT>(input, castArray<accT>(g1), bkind, 2, false);
    Array<accT> smth2 =
        convolve<accT, accT>(input, castArray<accT>(g2), bkind, 2, false);
    Array<accT> retVal = arithOp<accT, fly_sub_t>(smth1, smth2, iDims);

    FLY_CHECK(fly_release_array(g1));
    FLY_CHECK(fly_release_array(g2));

    return getHandle<accT>(retVal);
}

fly_err fly_dog(fly_array* out, const fly_array in, const int radius1,
              const int radius2) {
    try {
        const ArrayInfo& info = getInfo(in);
        dim4 inDims           = info.dims();
        ARG_ASSERT(1, (inDims.ndims() >= 2));
        ARG_ASSERT(1, (inDims.ndims() <= 3));

        fly_array output;
        fly_dtype type = info.getType();
        switch (type) {
            case f32: output = dog<float, float>(in, radius1, radius2); break;
            case f64: output = dog<double, double>(in, radius1, radius2); break;
            case b8: output = dog<char, float>(in, radius1, radius2); break;
            case s32: output = dog<int, float>(in, radius1, radius2); break;
            case u32: output = dog<uint, float>(in, radius1, radius2); break;
            case s16: output = dog<short, float>(in, radius1, radius2); break;
            case u16: output = dog<ushort, float>(in, radius1, radius2); break;
            case u8: output = dog<uchar, float>(in, radius1, radius2); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
