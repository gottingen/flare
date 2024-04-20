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
#include <common/moddims.hpp>
#include <common/tile.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <optypes.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>

#include <arith.hpp>
#include <logic.hpp>
#include <sparse_arith.hpp>

#include <common/half.hpp>

using fly::dim4;
using fly::dtype;
using flare::castSparse;
using flare::getSparseArray;
using flare::getSparseArrayBase;
using flare::common::half;
using flare::common::modDims;
using flare::common::SparseArrayBase;
using flare::common::tile;
using detail::arithOp;
using detail::arithOpD;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T, fly_op_t op>
static inline fly_array arithOp(const fly_array lhs, const fly_array rhs,
                               const dim4 &odims) {
    const ArrayInfo &linfo = getInfo(lhs);
    const ArrayInfo &rinfo = getInfo(rhs);

    dtype type = static_cast<fly::dtype>(fly::dtype_traits<T>::fly_type);

    const detail::Array<T> &l =
        linfo.getType() == type ? getArray<T>(lhs) : castArray<T>(lhs);
    const detail::Array<T> &r =
        rinfo.getType() == type ? getArray<T>(rhs) : castArray<T>(rhs);

    return getHandle(arithOp<T, op>(l, r, odims));
}

template<typename T, fly_op_t op>
static inline fly_array arithOpBroadcast(const fly_array lhs,
                                        const fly_array rhs) {
    const ArrayInfo &linfo = getInfo(lhs);
    const ArrayInfo &rinfo = getInfo(rhs);

    dim4 odims(1), ltile(1), rtile(1);
    dim4 lshape = linfo.dims();
    dim4 rshape = rinfo.dims();

    for (int d = 0; d < FLY_MAX_DIMS; ++d) {
        DIM_ASSERT(
            1, ((lshape[d] == rshape[d]) || (lshape[d] == 1 && rshape[d] > 1) ||
                (lshape[d] > 1 && rshape[d] == 1)));
        odims[d] = std::max(lshape[d], rshape[d]);
        if (lshape[d] == rshape[d]) {
            ltile[d] = rtile[d] = 1;
        } else if (lshape[d] == 1 && rshape[d] > 1) {
            ltile[d] = odims[d];
        } else if (lshape[d] > 1 && rshape[d] == 1) {
            rtile[d] = odims[d];
        }
    }

    Array<T> lhst =
        flare::common::tile<T>(modDims(getArray<T>(lhs), lshape), ltile);
    Array<T> rhst =
        flare::common::tile<T>(modDims(getArray<T>(rhs), rshape), rtile);

    return getHandle(arithOp<T, op>(lhst, rhst, odims));
}

template<typename T, fly_op_t op>
static inline fly_array sparseArithOp(const fly_array lhs, const fly_array rhs) {
    auto res = arithOp<T, op>(getSparseArray<T>(lhs), getSparseArray<T>(rhs));
    return getHandle(res);
}

template<typename T, fly_op_t op>
static inline fly_array arithSparseDenseOp(const fly_array lhs,
                                          const fly_array rhs,
                                          const bool reverse) {
    if (op == fly_add_t || op == fly_sub_t) {
        return getHandle(
            arithOpD<T, op>(castSparse<T>(lhs), castArray<T>(rhs), reverse));
    }
    if (op == fly_mul_t || op == fly_div_t) {
        return getHandle(
            arithOp<T, op>(castSparse<T>(lhs), castArray<T>(rhs), reverse));
    }
}

template<fly_op_t op>
static fly_err fly_arith(fly_array *out, const fly_array lhs, const fly_array rhs,
                       const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        const fly_dtype otype = implicit(linfo.getType(), rinfo.getType());
        fly_array res;

        if (batchMode || linfo.dims() == rinfo.dims()) {
            dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);
            if (odims.ndims() == 0) {
                return fly_create_handle(out, 0, nullptr, otype);
            }

            switch (otype) {
                case f32: res = arithOp<float, op>(lhs, rhs, odims); break;
                case f64: res = arithOp<double, op>(lhs, rhs, odims); break;
                case c32: res = arithOp<cfloat, op>(lhs, rhs, odims); break;
                case c64: res = arithOp<cdouble, op>(lhs, rhs, odims); break;
                case s32: res = arithOp<int, op>(lhs, rhs, odims); break;
                case u32: res = arithOp<uint, op>(lhs, rhs, odims); break;
                case u8: res = arithOp<uchar, op>(lhs, rhs, odims); break;
                case b8: res = arithOp<char, op>(lhs, rhs, odims); break;
                case s64: res = arithOp<intl, op>(lhs, rhs, odims); break;
                case u64: res = arithOp<uintl, op>(lhs, rhs, odims); break;
                case s16: res = arithOp<short, op>(lhs, rhs, odims); break;
                case u16: res = arithOp<ushort, op>(lhs, rhs, odims); break;
                case f16: res = arithOp<half, op>(lhs, rhs, odims); break;
                default: TYPE_ERROR(0, otype);
            }
        } else {
            if (linfo.ndims() == 0 && rinfo.ndims() == 0) {
                return fly_create_handle(out, 0, nullptr, otype);
            }
            switch (otype) {
                case f32: res = arithOpBroadcast<float, op>(lhs, rhs); break;
                case f64: res = arithOpBroadcast<double, op>(lhs, rhs); break;
                case c32: res = arithOpBroadcast<cfloat, op>(lhs, rhs); break;
                case c64: res = arithOpBroadcast<cdouble, op>(lhs, rhs); break;
                case s32: res = arithOpBroadcast<int, op>(lhs, rhs); break;
                case u32: res = arithOpBroadcast<uint, op>(lhs, rhs); break;
                case u8: res = arithOpBroadcast<uchar, op>(lhs, rhs); break;
                case b8: res = arithOpBroadcast<char, op>(lhs, rhs); break;
                case s64: res = arithOpBroadcast<intl, op>(lhs, rhs); break;
                case u64: res = arithOpBroadcast<uintl, op>(lhs, rhs); break;
                case s16: res = arithOpBroadcast<short, op>(lhs, rhs); break;
                case u16: res = arithOpBroadcast<ushort, op>(lhs, rhs); break;
                case f16: res = arithOpBroadcast<half, op>(lhs, rhs); break;
                default: TYPE_ERROR(0, otype);
            }
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<fly_op_t op>
static fly_err fly_arith_real(fly_array *out, const fly_array lhs,
                            const fly_array rhs, const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);
        const fly_dtype otype = implicit(linfo.getType(), rinfo.getType());
        if (odims.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, otype);
        }

        fly_array res;
        switch (otype) {
            case f32: res = arithOp<float, op>(lhs, rhs, odims); break;
            case f64: res = arithOp<double, op>(lhs, rhs, odims); break;
            case s32: res = arithOp<int, op>(lhs, rhs, odims); break;
            case u32: res = arithOp<uint, op>(lhs, rhs, odims); break;
            case u8: res = arithOp<uchar, op>(lhs, rhs, odims); break;
            case b8: res = arithOp<char, op>(lhs, rhs, odims); break;
            case s64: res = arithOp<intl, op>(lhs, rhs, odims); break;
            case u64: res = arithOp<uintl, op>(lhs, rhs, odims); break;
            case s16: res = arithOp<short, op>(lhs, rhs, odims); break;
            case u16: res = arithOp<ushort, op>(lhs, rhs, odims); break;
            case f16: res = arithOp<half, op>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, otype);
        }
        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<fly_op_t op>
static fly_err fly_arith_sparse(fly_array *out, const fly_array lhs,
                              const fly_array rhs) {
    try {
        const SparseArrayBase linfo = getSparseArrayBase(lhs);
        const SparseArrayBase rinfo = getSparseArrayBase(rhs);

        ARG_ASSERT(1, (linfo.getStorage() == rinfo.getStorage()));
        ARG_ASSERT(1, (linfo.dims() == rinfo.dims()));
        ARG_ASSERT(1, (linfo.getStorage() == FLY_STORAGE_CSR));

        const fly_dtype otype = implicit(linfo.getType(), rinfo.getType());
        fly_array res;
        switch (otype) {
            case f32: res = sparseArithOp<float, op>(lhs, rhs); break;
            case f64: res = sparseArithOp<double, op>(lhs, rhs); break;
            case c32: res = sparseArithOp<cfloat, op>(lhs, rhs); break;
            case c64: res = sparseArithOp<cdouble, op>(lhs, rhs); break;
            default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<fly_op_t op>
static fly_err fly_arith_sparse_dense(fly_array *out, const fly_array lhs,
                                    const fly_array rhs,
                                    const bool reverse = false) {
    try {
        const SparseArrayBase linfo = getSparseArrayBase(lhs);
        if (linfo.ndims() > 2) {
            FLY_ERROR(
                "Sparse-Dense arithmetic operations cannot be used in batch "
                "mode",
                FLY_ERR_BATCH);
        }
        const ArrayInfo &rinfo = getInfo(rhs);

        const fly_dtype otype = implicit(linfo.getType(), rinfo.getType());
        fly_array res;
        switch (otype) {
            case f32:
                res = arithSparseDenseOp<float, op>(lhs, rhs, reverse);
                break;
            case f64:
                res = arithSparseDenseOp<double, op>(lhs, rhs, reverse);
                break;
            case c32:
                res = arithSparseDenseOp<cfloat, op>(lhs, rhs, reverse);
                break;
            case c64:
                res = arithSparseDenseOp<cdouble, op>(lhs, rhs, reverse);
                break;
            default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_add(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            return fly_arith_sparse<fly_add_t>(out, lhs, rhs);
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return fly_arith_sparse_dense<fly_add_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            // second operand(Array) of fly_arith call should be dense
            return fly_arith_sparse_dense<fly_add_t>(out, rhs, lhs, true);
        }
        return fly_arith<fly_add_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

fly_err fly_mul(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            // return fly_arith_sparse<fly_mul_t>(out, lhs, rhs);
            // MKL doesn't have mul or div support yet, hence
            // this is commented out although alternative cpu code exists
            return FLY_ERR_NOT_SUPPORTED;
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return fly_arith_sparse_dense<fly_mul_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            return fly_arith_sparse_dense<fly_mul_t>(
                out, rhs, lhs,
                true);  // dense should be rhs
        }
        return fly_arith<fly_mul_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

fly_err fly_sub(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            return fly_arith_sparse<fly_sub_t>(out, lhs, rhs);
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return fly_arith_sparse_dense<fly_sub_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            return fly_arith_sparse_dense<fly_sub_t>(
                out, rhs, lhs,
                true);  // dense should be rhs
        }
        return fly_arith<fly_sub_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

fly_err fly_div(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            // return fly_arith_sparse<fly_div_t>(out, lhs, rhs);
            // MKL doesn't have mul or div support yet, hence
            // this is commented out although alternative cpu code exists
            return FLY_ERR_NOT_SUPPORTED;
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return fly_arith_sparse_dense<fly_div_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            // Division by sparse is currently not allowed - for convinence of
            // dealing with division by 0
            // return fly_arith_sparse_dense<fly_div_t>(out, rhs, lhs, true); //
            // dense should be rhs
            return FLY_ERR_NOT_SUPPORTED;
        }
        return fly_arith<fly_div_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

fly_err fly_maxof(fly_array *out, const fly_array lhs, const fly_array rhs,
                const bool batchMode) {
    return fly_arith<fly_max_t>(out, lhs, rhs, batchMode);
}

fly_err fly_minof(fly_array *out, const fly_array lhs, const fly_array rhs,
                const bool batchMode) {
    return fly_arith<fly_min_t>(out, lhs, rhs, batchMode);
}

fly_err fly_rem(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    return fly_arith_real<fly_rem_t>(out, lhs, rhs, batchMode);
}

fly_err fly_mod(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    return fly_arith_real<fly_mod_t>(out, lhs, rhs, batchMode);
}

fly_err fly_pow(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);
        if (rinfo.isComplex()) {
            fly_array log_lhs, log_res;
            fly_array res;
            FLY_CHECK(fly_log(&log_lhs, lhs));
            FLY_CHECK(fly_mul(&log_res, log_lhs, rhs, batchMode));
            FLY_CHECK(fly_exp(&res, log_res));
            FLY_CHECK(fly_release_array(log_lhs));
            FLY_CHECK(fly_release_array(log_res));
            std::swap(*out, res);
            return FLY_SUCCESS;
        }
        if (linfo.isComplex()) {
            fly_array mag, angle;
            fly_array mag_res, angle_res;
            fly_array real_res, imag_res, cplx_res;
            fly_array res;
            FLY_CHECK(fly_abs(&mag, lhs));
            FLY_CHECK(fly_arg(&angle, lhs));
            FLY_CHECK(fly_pow(&mag_res, mag, rhs, batchMode));
            FLY_CHECK(fly_mul(&angle_res, angle, rhs, batchMode));
            FLY_CHECK(fly_cos(&real_res, angle_res));
            FLY_CHECK(fly_sin(&imag_res, angle_res));
            FLY_CHECK(fly_cplx2(&cplx_res, real_res, imag_res, batchMode));
            FLY_CHECK(fly_mul(&res, mag_res, cplx_res, batchMode));
            FLY_CHECK(fly_release_array(mag));
            FLY_CHECK(fly_release_array(angle));
            FLY_CHECK(fly_release_array(mag_res));
            FLY_CHECK(fly_release_array(angle_res));
            FLY_CHECK(fly_release_array(real_res));
            FLY_CHECK(fly_release_array(imag_res));
            FLY_CHECK(fly_release_array(cplx_res));
            std::swap(*out, res);
            return FLY_SUCCESS;
        }
    }
    CATCHALL;

    return fly_arith_real<fly_pow_t>(out, lhs, rhs, batchMode);
}

fly_err fly_root(fly_array *out, const fly_array lhs, const fly_array rhs,
               const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);
        if (linfo.isComplex() || rinfo.isComplex()) {
            fly_array log_lhs, log_res;
            fly_array res;
            FLY_CHECK(fly_log(&log_lhs, lhs));
            FLY_CHECK(fly_div(&log_res, log_lhs, rhs, batchMode));
            FLY_CHECK(fly_exp(&res, log_res));
            std::swap(*out, res);
            return FLY_SUCCESS;
        }

        fly_array one;
        FLY_CHECK(fly_constant(&one, 1, linfo.ndims(), linfo.dims().get(),
                             linfo.getType()));

        fly_array inv_lhs;
        FLY_CHECK(fly_div(&inv_lhs, one, lhs, batchMode));

        FLY_CHECK(fly_arith_real<fly_pow_t>(out, rhs, inv_lhs, batchMode));

        FLY_CHECK(fly_release_array(one));
        FLY_CHECK(fly_release_array(inv_lhs));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_atan2(fly_array *out, const fly_array lhs, const fly_array rhs,
                const bool batchMode) {
    try {
        const fly_dtype type = implicit(lhs, rhs);

        if (type != f32 && type != f64) {
            FLY_ERROR("Only floating point arrays are supported for atan2 ",
                     FLY_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);
        if (odims.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        fly_array res;
        switch (type) {
            case f32: res = arithOp<float, fly_atan2_t>(lhs, rhs, odims); break;
            case f64: res = arithOp<double, fly_atan2_t>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_hypot(fly_array *out, const fly_array lhs, const fly_array rhs,
                const bool batchMode) {
    try {
        const fly_dtype type = implicit(lhs, rhs);

        if (type != f32 && type != f64) {
            FLY_ERROR("Only floating point arrays are supported for hypot ",
                     FLY_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        if (odims.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        fly_array res;
        switch (type) {
            case f32: res = arithOp<float, fly_hypot_t>(lhs, rhs, odims); break;
            case f64: res = arithOp<double, fly_hypot_t>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<typename T, fly_op_t op>
static inline fly_array logicOp(const fly_array lhs, const fly_array rhs,
                               const dim4 &odims) {
    fly_array res =
        getHandle(logicOp<T, op>(castArray<T>(lhs), castArray<T>(rhs), odims));
    return res;
}

template<fly_op_t op>
static fly_err fly_logic(fly_array *out, const fly_array lhs, const fly_array rhs,
                       const bool batchMode) {
    try {
        const fly_dtype type = implicit(lhs, rhs);

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        if (odims.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        fly_array res;
        switch (type) {
            case f32: res = logicOp<float, op>(lhs, rhs, odims); break;
            case f64: res = logicOp<double, op>(lhs, rhs, odims); break;
            case c32: res = logicOp<cfloat, op>(lhs, rhs, odims); break;
            case c64: res = logicOp<cdouble, op>(lhs, rhs, odims); break;
            case s32: res = logicOp<int, op>(lhs, rhs, odims); break;
            case u32: res = logicOp<uint, op>(lhs, rhs, odims); break;
            case u8: res = logicOp<uchar, op>(lhs, rhs, odims); break;
            case b8: res = logicOp<char, op>(lhs, rhs, odims); break;
            case s64: res = logicOp<intl, op>(lhs, rhs, odims); break;
            case u64: res = logicOp<uintl, op>(lhs, rhs, odims); break;
            case s16: res = logicOp<short, op>(lhs, rhs, odims); break;
            case u16: res = logicOp<ushort, op>(lhs, rhs, odims); break;
            case f16: res = logicOp<half, op>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_eq(fly_array *out, const fly_array lhs, const fly_array rhs,
             const bool batchMode) {
    return fly_logic<fly_eq_t>(out, lhs, rhs, batchMode);
}

fly_err fly_neq(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    return fly_logic<fly_neq_t>(out, lhs, rhs, batchMode);
}

fly_err fly_gt(fly_array *out, const fly_array lhs, const fly_array rhs,
             const bool batchMode) {
    return fly_logic<fly_gt_t>(out, lhs, rhs, batchMode);
}

fly_err fly_ge(fly_array *out, const fly_array lhs, const fly_array rhs,
             const bool batchMode) {
    return fly_logic<fly_ge_t>(out, lhs, rhs, batchMode);
}

fly_err fly_lt(fly_array *out, const fly_array lhs, const fly_array rhs,
             const bool batchMode) {
    return fly_logic<fly_lt_t>(out, lhs, rhs, batchMode);
}

fly_err fly_le(fly_array *out, const fly_array lhs, const fly_array rhs,
             const bool batchMode) {
    return fly_logic<fly_le_t>(out, lhs, rhs, batchMode);
}

fly_err fly_and(fly_array *out, const fly_array lhs, const fly_array rhs,
              const bool batchMode) {
    return fly_logic<fly_and_t>(out, lhs, rhs, batchMode);
}

fly_err fly_or(fly_array *out, const fly_array lhs, const fly_array rhs,
             const bool batchMode) {
    return fly_logic<fly_or_t>(out, lhs, rhs, batchMode);
}

template<typename T, fly_op_t op>
static inline fly_array bitOp(const fly_array lhs, const fly_array rhs,
                             const dim4 &odims) {
    fly_array res =
        getHandle(bitOp<T, op>(castArray<T>(lhs), castArray<T>(rhs), odims));
    return res;
}

template<fly_op_t op>
static fly_err fly_bitwise(fly_array *out, const fly_array lhs, const fly_array rhs,
                         const bool batchMode) {
    try {
        const fly_dtype type = implicit(lhs, rhs);

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        if (odims.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        fly_array res;
        switch (type) {
            case s32: res = bitOp<int, op>(lhs, rhs, odims); break;
            case u32: res = bitOp<uint, op>(lhs, rhs, odims); break;
            case u8: res = bitOp<uchar, op>(lhs, rhs, odims); break;
            case b8: res = bitOp<char, op>(lhs, rhs, odims); break;
            case s64: res = bitOp<intl, op>(lhs, rhs, odims); break;
            case u64: res = bitOp<uintl, op>(lhs, rhs, odims); break;
            case s16: res = bitOp<short, op>(lhs, rhs, odims); break;
            case u16: res = bitOp<ushort, op>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_bitand(fly_array *out, const fly_array lhs, const fly_array rhs,
                 const bool batchMode) {
    return fly_bitwise<fly_bitand_t>(out, lhs, rhs, batchMode);
}

fly_err fly_bitor(fly_array *out, const fly_array lhs, const fly_array rhs,
                const bool batchMode) {
    return fly_bitwise<fly_bitor_t>(out, lhs, rhs, batchMode);
}

fly_err fly_bitxor(fly_array *out, const fly_array lhs, const fly_array rhs,
                 const bool batchMode) {
    return fly_bitwise<fly_bitxor_t>(out, lhs, rhs, batchMode);
}

fly_err fly_bitshiftl(fly_array *out, const fly_array lhs, const fly_array rhs,
                    const bool batchMode) {
    return fly_bitwise<fly_bitshiftl_t>(out, lhs, rhs, batchMode);
}

fly_err fly_bitshiftr(fly_array *out, const fly_array lhs, const fly_array rhs,
                    const bool batchMode) {
    return fly_bitwise<fly_bitshiftr_t>(out, lhs, rhs, batchMode);
}
