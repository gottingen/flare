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
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <transform.hpp>
#include <fly/defines.h>
#include <fly/image.h>

using fly::dim4;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline void transform(fly_array *out, const fly_array in,
                             const fly_array tf, const fly_interp_type method,
                             const bool inverse, const bool perspective) {
    transform<T>(getArray<T>(*out), getArray<T>(in), getArray<float>(tf),
                 method, inverse, perspective);
}

FLY_BATCH_KIND getTransformBatchKind(const dim4 &iDims, const dim4 &tDims) {
    static const int baseDim = 2;

    dim_t iNd = iDims.ndims();
    dim_t tNd = tDims.ndims();

    if (iNd == baseDim && tNd == baseDim) { return FLY_BATCH_NONE; }
    if (iNd == baseDim && tNd <= 4) {
        return FLY_BATCH_RHS;
    } else if (iNd <= 4 && tNd == baseDim) {
        return FLY_BATCH_LHS;
    } else if (iNd <= 4 && tNd <= 4) {
        bool dimsMatch     = true;
        bool isInterleaved = true;
        for (dim_t i = baseDim; i < 4; i++) {
            dimsMatch &= (iDims[i] == tDims[i]);
            isInterleaved &=
                (iDims[i] == 1 || tDims[i] == 1 || iDims[i] == tDims[i]);
        }
        if (dimsMatch) { return FLY_BATCH_SAME; }
        return (isInterleaved ? FLY_BATCH_DIFF : FLY_BATCH_UNSUPPORTED);
    } else {
        return FLY_BATCH_UNSUPPORTED;
    }
}

void fly_transform_common(fly_array *out, const fly_array in, const fly_array tf,
                         const dim_t odim0, const dim_t odim1,
                         const fly_interp_type method, const bool inverse,
                         bool allocate_out) {
    ARG_ASSERT(0, out != 0);  // *out (the fly_array) can be null, but not out
    ARG_ASSERT(1, in != 0);
    ARG_ASSERT(2, tf != 0);

    const ArrayInfo &t_info = getInfo(tf);
    const ArrayInfo &i_info = getInfo(in);

    const dim4 &idims    = i_info.dims();
    const dim4 &tdims    = t_info.dims();
    const fly_dtype itype = i_info.getType();

    // Assert type and interpolation
    ARG_ASSERT(2, t_info.getType() == f32);
    ARG_ASSERT(5, method == FLY_INTERP_NEAREST || method == FLY_INTERP_BILINEAR ||
                      method == FLY_INTERP_BILINEAR_COSINE ||
                      method == FLY_INTERP_BICUBIC ||
                      method == FLY_INTERP_BICUBIC_SPLINE ||
                      method == FLY_INTERP_LOWER);

    // Assert dimesions
    // Image can be 2D or higher
    DIM_ASSERT(1, idims.elements() > 0);
    DIM_ASSERT(1, idims.ndims() >= 2);

    // Transform can be 3x2 for affine transform or 3x3 for perspective
    // transform
    DIM_ASSERT(2, (tdims[0] == 3 && (tdims[1] == 2 || tdims[1] == 3)));

    // If transform is batched, the output dimensions must be specified
    if (tdims[2] * tdims[3] > 1) {
        ARG_ASSERT(3, odim0 > 0);
        ARG_ASSERT(4, odim1 > 0);
    }

    // If idims[2] > 1 and tdims[2] > 1, then both must be equal
    // else at least one of them must be 1
    if (tdims[2] != 1 && idims[2] != 1) {
        DIM_ASSERT(2, idims[2] == tdims[2]);
    } else {
        DIM_ASSERT(2, idims[2] == 1 || tdims[2] == 1);
    }

    // If idims[3] > 1 and tdims[3] > 1, then both must be equal
    // else at least one of them must be 1
    if (tdims[3] != 1 && idims[3] != 1) {
        DIM_ASSERT(2, idims[3] == tdims[3]);
    } else {
        DIM_ASSERT(2, idims[3] == 1 || tdims[3] == 1);
    }

    const bool perspective = (tdims[1] == 3);
    dim_t o0 = odim0, o1 = odim1, o2 = 0, o3 = 0;
    if (odim0 * odim1 == 0) {
        o0 = idims[0];
        o1 = idims[1];
    }

    switch (getTransformBatchKind(idims, tdims)) {
        case FLY_BATCH_NONE:  // Both are exactly 2D
        case FLY_BATCH_LHS:   // Image is 3/4D, transform is 2D
        case FLY_BATCH_SAME:  // Both are 3/4D and have the same dims
            o2 = idims[2];
            o3 = idims[3];
            break;
        case FLY_BATCH_RHS:  // Image is 2D, transform is 3/4D
            o2 = tdims[2];
            o3 = tdims[3];
            break;
        case FLY_BATCH_DIFF:  // Both are 3/4D, but have different dims
            o2 = idims[2] == 1 ? tdims[2] : idims[2];
            o3 = idims[3] == 1 ? tdims[3] : idims[3];
            break;
        case FLY_BATCH_UNSUPPORTED:
        default:
            FLY_ERROR(
                "Unsupported combination of batching parameters in "
                "transform",
                FLY_ERR_NOT_SUPPORTED);
            break;
    }

    const dim4 odims(o0, o1, o2, o3);
    if (allocate_out) { *out = createHandle(odims, itype); }

    // clang-format off
    switch(itype) {
    case f32: transform<float  >(out, in, tf, method, inverse, perspective);  break;
    case f64: transform<double >(out, in, tf, method, inverse, perspective);  break;
    case c32: transform<cfloat >(out, in, tf, method, inverse, perspective);  break;
    case c64: transform<cdouble>(out, in, tf, method, inverse, perspective);  break;
    case s32: transform<int    >(out, in, tf, method, inverse, perspective);  break;
    case u32: transform<uint   >(out, in, tf, method, inverse, perspective);  break;
    case s64: transform<intl   >(out, in, tf, method, inverse, perspective);  break;
    case u64: transform<uintl  >(out, in, tf, method, inverse, perspective);  break;
    case s16: transform<short  >(out, in, tf, method, inverse, perspective);  break;
    case u16: transform<ushort >(out, in, tf, method, inverse, perspective);  break;
    case u8:  transform<uchar  >(out, in, tf, method, inverse, perspective);  break;
    case b8:  transform<char   >(out, in, tf, method, inverse, perspective);  break;
    default:  TYPE_ERROR(1, itype);
    }
    // clang-format on
}

fly_err fly_transform(fly_array *out, const fly_array in, const fly_array tf,
                    const dim_t odim0, const dim_t odim1,
                    const fly_interp_type method, const bool inverse) {
    try {
        fly_transform_common(out, in, tf, odim0, odim1, method, inverse, true);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_transform_v2(fly_array *out, const fly_array in, const fly_array tf,
                       const dim_t odim0, const dim_t odim1,
                       const fly_interp_type method, const bool inverse) {
    try {
        ARG_ASSERT(0, out != 0);  // need to dereference out in next call
        fly_transform_common(out, in, tf, odim0, odim1, method, inverse,
                            *out == 0);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_translate(fly_array *out, const fly_array in, const float trans0,
                    const float trans1, const dim_t odim0, const dim_t odim1,
                    const fly_interp_type method) {
    try {
        float trans_mat[6] = {1, 0, 0, 0, 1, 0};
        trans_mat[2]       = trans0;
        trans_mat[5]       = trans1;

        const dim4 tdims(3, 2, 1, 1);
        fly_array t = 0;

        FLY_CHECK(
            fly_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32));
        FLY_CHECK(fly_transform(out, in, t, odim0, odim1, method, true));
        FLY_CHECK(fly_release_array(t));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_scale(fly_array *out, const fly_array in, const float scale0,
                const float scale1, const dim_t odim0, const dim_t odim1,
                const fly_interp_type method) {
    try {
        const ArrayInfo &i_info = getInfo(in);
        dim4 idims              = i_info.dims();

        dim_t _odim0 = odim0, _odim1 = odim1;
        float sx, sy;

        if (_odim0 == 0 || _odim1 == 0) {
            DIM_ASSERT(2, scale0 != 0);
            DIM_ASSERT(3, scale1 != 0);

            sx = 1.f / scale0, sy = 1.f / scale1;
            _odim0 = idims[0] / sx;
            _odim1 = idims[1] / sy;

        } else if (scale0 == 0 || scale1 == 0) {
            DIM_ASSERT(4, odim0 != 0);
            DIM_ASSERT(5, odim1 != 0);

            sx = idims[0] / static_cast<float>(_odim0);
            sy = idims[1] / static_cast<float>(_odim1);

        } else {
            sx = 1.f / scale0, sy = 1.f / scale1;
        }

        float trans_mat[6] = {1, 0, 0, 0, 1, 0};
        trans_mat[0]       = sx;
        trans_mat[4]       = sy;

        const dim4 tdims(3, 2, 1, 1);
        fly_array t = 0;
        FLY_CHECK(
            fly_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32));
        FLY_CHECK(fly_transform(out, in, t, _odim0, _odim1, method, true));
        FLY_CHECK(fly_release_array(t));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_skew(fly_array *out, const fly_array in, const float skew0,
               const float skew1, const dim_t odim0, const dim_t odim1,
               const fly_interp_type method, const bool inverse) {
    try {
        float tx = std::tan(skew0);
        float ty = std::tan(skew1);

        float trans_mat[6] = {1, 0, 0, 0, 1, 0};
        trans_mat[1]       = ty;
        trans_mat[3]       = tx;

        if (inverse) {
            if (tx == 0 || ty == 0) {
                trans_mat[1] = tx;
                trans_mat[3] = ty;
            } else {
                // calc_tranform_inverse(trans_mat);
                // short cut of calc_transform_inverse
                float d      = 1.0f / (1.0f - tx * ty);
                trans_mat[0] = d;
                trans_mat[1] = ty * d;
                trans_mat[3] = tx * d;
                trans_mat[4] = d;
            }
        }
        const dim4 tdims(3, 2, 1, 1);
        fly_array t = 0;
        FLY_CHECK(
            fly_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32));
        FLY_CHECK(fly_transform(out, in, t, odim0, odim1, method, true));
        FLY_CHECK(fly_release_array(t));
    }
    CATCHALL;

    return FLY_SUCCESS;
}
