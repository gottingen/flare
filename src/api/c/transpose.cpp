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

#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <transpose.hpp>
#include <fly/arith.h>
#include <fly/blas.h>
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>

using fly::dim4;
using flare::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array trs(const fly_array in, const bool conjugate) {
    return getHandle<T>(detail::transpose<T>(getArray<T>(in), conjugate));
}

fly_err fly_transpose(fly_array* out, fly_array in, const bool conjugate) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 dims         = info.dims();

        if (dims.elements() == 0) { return fly_retain_array(out, in); }

        if (dims[0] == 1 || dims[1] == 1) {
            fly::dim4 outDims(dims[1], dims[0], dims[2], dims[3]);
            if (conjugate) {
                fly_array temp = 0;
                FLY_CHECK(fly_conjg(&temp, in));
                FLY_CHECK(fly_moddims(out, temp, outDims.ndims(), outDims.get()));
                FLY_CHECK(fly_release_array(temp));
                return FLY_SUCCESS;
            } else {
                // for a vector OR a batch of vectors
                // we can use modDims to transpose
                FLY_CHECK(fly_moddims(out, in, outDims.ndims(), outDims.get()));
                return FLY_SUCCESS;
            }
        }

        fly_array output;
        switch (type) {
            case f32: output = trs<float>(in, conjugate); break;
            case c32: output = trs<cfloat>(in, conjugate); break;
            case f64: output = trs<double>(in, conjugate); break;
            case c64: output = trs<cdouble>(in, conjugate); break;
            case b8: output = trs<char>(in, conjugate); break;
            case s32: output = trs<int>(in, conjugate); break;
            case u32: output = trs<uint>(in, conjugate); break;
            case u8: output = trs<uchar>(in, conjugate); break;
            case s64: output = trs<intl>(in, conjugate); break;
            case u64: output = trs<uintl>(in, conjugate); break;
            case s16: output = trs<short>(in, conjugate); break;
            case u16: output = trs<ushort>(in, conjugate); break;
            case f16: output = trs<half>(in, conjugate); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline void transpose_inplace(fly_array in, const bool conjugate) {
    return detail::transpose_inplace<T>(getArray<T>(in), conjugate);
}

fly_err fly_transpose_inplace(fly_array in, const bool conjugate) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 dims         = info.dims();

        // InPlace only works on square matrices
        DIM_ASSERT(0, dims[0] == dims[1]);

        // If singleton element
        if (dims[0] == 1) { return FLY_SUCCESS; }

        switch (type) {
            case f32: transpose_inplace<float>(in, conjugate); break;
            case c32: transpose_inplace<cfloat>(in, conjugate); break;
            case f64: transpose_inplace<double>(in, conjugate); break;
            case c64: transpose_inplace<cdouble>(in, conjugate); break;
            case b8: transpose_inplace<char>(in, conjugate); break;
            case s32: transpose_inplace<int>(in, conjugate); break;
            case u32: transpose_inplace<uint>(in, conjugate); break;
            case u8: transpose_inplace<uchar>(in, conjugate); break;
            case s64: transpose_inplace<intl>(in, conjugate); break;
            case u64: transpose_inplace<uintl>(in, conjugate); break;
            case s16: transpose_inplace<short>(in, conjugate); break;
            case u16: transpose_inplace<ushort>(in, conjugate); break;
            case f16: transpose_inplace<half>(in, conjugate); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
