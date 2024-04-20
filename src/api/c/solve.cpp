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
#include <solve.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/lapack.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::solveLU;

template<typename T>
static inline fly_array solve(const fly_array a, const fly_array b,
                             const fly_mat_prop options) {
    return getHandle(solve<T>(getArray<T>(a), getArray<T>(b), options));
}

fly_err fly_solve(fly_array* out, const fly_array a, const fly_array b,
                const fly_mat_prop options) {
    try {
        const ArrayInfo& a_info = getInfo(a);
        const ArrayInfo& b_info = getInfo(b);

        fly_dtype a_type = a_info.getType();
        fly_dtype b_type = b_info.getType();

        dim4 adims = a_info.dims();
        dim4 bdims = b_info.dims();

        ARG_ASSERT(1, a_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(2, b_info.isFloating());  // Only floating and complex types

        TYPE_ASSERT(a_type == b_type);

        DIM_ASSERT(1, bdims[0] == adims[0]);
        DIM_ASSERT(1, bdims[2] == adims[2]);
        DIM_ASSERT(1, bdims[3] == adims[3]);

        if (a_info.ndims() == 0 || b_info.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, a_type);
        }

        bool is_triangle_solve =
            (options & FLY_MAT_LOWER) || (options & FLY_MAT_UPPER);

        if (options != FLY_MAT_NONE && !is_triangle_solve) {
            FLY_ERROR("Using this property is not yet supported in solve",
                     FLY_ERR_NOT_SUPPORTED);
        }

        if (is_triangle_solve) {
            DIM_ASSERT(1, adims[0] == adims[1]);
            if ((options & FLY_MAT_TRANS || options & FLY_MAT_CTRANS)) {
                FLY_ERROR("Using FLY_MAT_TRANS is not yet supported in solve",
                         FLY_ERR_NOT_SUPPORTED);
            }
        }

        fly_array output;

        switch (a_type) {
            case f32: output = solve<float>(a, b, options); break;
            case f64: output = solve<double>(a, b, options); break;
            case c32: output = solve<cfloat>(a, b, options); break;
            case c64: output = solve<cdouble>(a, b, options); break;
            default: TYPE_ERROR(1, a_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline fly_array solve_lu(const fly_array a, const fly_array pivot,
                                const fly_array b, const fly_mat_prop options) {
    return getHandle(solveLU<T>(getArray<T>(a), getArray<int>(pivot),
                                getArray<T>(b), options));
}

fly_err fly_solve_lu(fly_array* out, const fly_array a, const fly_array piv,
                   const fly_array b, const fly_mat_prop options) {
    try {
        const ArrayInfo& a_info   = getInfo(a);
        const ArrayInfo& b_info   = getInfo(b);
        const ArrayInfo& piv_info = getInfo(piv);

        if (a_info.ndims() > 2 || b_info.ndims() > 2) {
            FLY_ERROR("solveLU can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype a_type = a_info.getType();
        fly_dtype b_type = b_info.getType();

        dim4 adims = a_info.dims();
        dim4 bdims = b_info.dims();
        if (a_info.ndims() == 0 || b_info.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, a_type);
        }

        ARG_ASSERT(1, a_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(2, b_info.isFloating());  // Only floating and complex types

        TYPE_ASSERT(a_type == b_type);

        fly_dtype piv_type = piv_info.getType();
        TYPE_ASSERT(piv_type == s32);  // TODO: add support for 64 bit types

        DIM_ASSERT(1, adims[0] == adims[1]);
        DIM_ASSERT(1, bdims[0] == adims[0]);
        DIM_ASSERT(1, bdims[2] == adims[2]);
        DIM_ASSERT(1, bdims[3] == adims[3]);

        if (options != FLY_MAT_NONE) {
            FLY_ERROR("Using this property is not yet supported in solveLU",
                     FLY_ERR_NOT_SUPPORTED);
        }

        fly_array output;

        switch (a_type) {
            case f32: output = solve_lu<float>(a, piv, b, options); break;
            case f64: output = solve_lu<double>(a, piv, b, options); break;
            case c32: output = solve_lu<cfloat>(a, piv, b, options); break;
            case c64: output = solve_lu<cdouble>(a, piv, b, options); break;
            default: TYPE_ERROR(1, a_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
