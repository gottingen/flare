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

#include <fly/blas.h>

#include <Array.hpp>
#include <backend.hpp>
#include <blas.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <sparse_blas.hpp>
#include <sparse_handle.hpp>

#include <type_util.hpp>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>

using flare::getSparseArray;
using flare::getSparseArrayBase;
using flare::common::half;
using flare::common::SparseArrayBase;
using detail::cdouble;
using detail::cfloat;
using detail::gemm;
using detail::matmul;

namespace {
template<typename T>
static inline fly_array sparseMatmul(const fly_array lhs, const fly_array rhs,
                                    fly_mat_prop optLhs, fly_mat_prop optRhs) {
    return getHandle(
        matmul<T>(getSparseArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline void gemm(fly_array *out, fly_mat_prop optLhs, fly_mat_prop optRhs,
                        const T *alpha, const fly_array lhs, const fly_array rhs,
                        const T *betas) {
    gemm<T>(getArray<T>(*out), optLhs, optRhs, alpha, getArray<T>(lhs),
            getArray<T>(rhs), betas);
}

template<typename T>
static inline fly_array dot(const fly_array lhs, const fly_array rhs,
                           fly_mat_prop optLhs, fly_mat_prop optRhs) {
    return getHandle(
        dot<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline T dotAll(fly_array out) {
    T res{};
    FLY_CHECK(fly_eval(out));
    FLY_CHECK(fly_get_data_ptr((void *)&res, out));
    return res;
}

}  // namespace

fly_err fly_sparse_matmul(fly_array *out, const fly_array lhs, const fly_array rhs,
                        const fly_mat_prop optLhs, const fly_mat_prop optRhs) {
    try {
        const SparseArrayBase lhsBase = getSparseArrayBase(lhs);
        const ArrayInfo &rhsInfo      = getInfo(rhs);

        ARG_ASSERT(2,
                   lhsBase.isSparse() == true && rhsInfo.isSparse() == false);

        fly_dtype lhs_type = lhsBase.getType();
        fly_dtype rhs_type = rhsInfo.getType();

        ARG_ASSERT(1, lhsBase.getStorage() == FLY_STORAGE_CSR);

        if (!(optLhs == FLY_MAT_NONE || optLhs == FLY_MAT_TRANS ||
              optLhs == FLY_MAT_CTRANS)) {  // Note the ! operator.
            FLY_ERROR(
                "Using this property is not yet supported in sparse matmul",
                FLY_ERR_NOT_SUPPORTED);
        }

        // No transpose options for RHS
        if (optRhs != FLY_MAT_NONE) {
            FLY_ERROR("Using this property is not yet supported in matmul",
                     FLY_ERR_NOT_SUPPORTED);
        }

        if (rhsInfo.ndims() > 2) {
            FLY_ERROR("Sparse matmul can not be used in batch mode",
                     FLY_ERR_BATCH);
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        fly::dim4 ldims = lhsBase.dims();
        int lColDim    = (optLhs == FLY_MAT_NONE) ? 1 : 0;
        int rRowDim    = (optRhs == FLY_MAT_NONE) ? 0 : 1;

        DIM_ASSERT(1, ldims[lColDim] == rhsInfo.dims()[rRowDim]);

        fly_array output = 0;
        switch (lhs_type) {
            case f32:
                output = sparseMatmul<float>(lhs, rhs, optLhs, optRhs);
                break;
            case c32:
                output = sparseMatmul<cfloat>(lhs, rhs, optLhs, optRhs);
                break;
            case f64:
                output = sparseMatmul<double>(lhs, rhs, optLhs, optRhs);
                break;
            case c64:
                output = sparseMatmul<cdouble>(lhs, rhs, optLhs, optRhs);
                break;
            default: TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_gemm(fly_array *out, const fly_mat_prop optLhs,
               const fly_mat_prop optRhs, const void *alpha, const fly_array lhs,
               const fly_array rhs, const void *beta) {
    try {
        const ArrayInfo &lhsInfo = getInfo(lhs, false);
        const ArrayInfo &rhsInfo = getInfo(rhs, true);

        fly_dtype lhs_type = lhsInfo.getType();
        fly_dtype rhs_type = rhsInfo.getType();

        if (!(optLhs == FLY_MAT_NONE || optLhs == FLY_MAT_TRANS ||
              optLhs == FLY_MAT_CTRANS)) {
            FLY_ERROR("Using this property is not yet supported in matmul",
                     FLY_ERR_NOT_SUPPORTED);
        }

        if (!(optRhs == FLY_MAT_NONE || optRhs == FLY_MAT_TRANS ||
              optRhs == FLY_MAT_CTRANS)) {
            FLY_ERROR("Using this property is not yet supported in matmul",
                     FLY_ERR_NOT_SUPPORTED);
        }

        fly::dim4 lDims = lhsInfo.dims();
        fly::dim4 rDims = rhsInfo.dims();

        if (lDims.ndims() > 2 && rDims.ndims() > 2) {
            DIM_ASSERT(3, lDims.ndims() == rDims.ndims());
            if (lDims[2] != rDims[2] && lDims[2] != 1 && rDims[2] != 1) {
                FLY_ERROR("Batch size mismatch along dimension 2", FLY_ERR_BATCH);
            }
            if (lDims[3] != rDims[3] && lDims[3] != 1 && rDims[3] != 1) {
                FLY_ERROR("Batch size mismatch along dimension 3", FLY_ERR_BATCH);
            }
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        int aColDim = (optLhs == FLY_MAT_NONE) ? 1 : 0;
        int bRowDim = (optRhs == FLY_MAT_NONE) ? 0 : 1;

        DIM_ASSERT(1, lhsInfo.dims()[aColDim] == rhsInfo.dims()[bRowDim]);

        // Assume that *out is either initialized to null or an actual fly_array
        // Otherwise, this function has undefined behavior
        fly_array output = 0;
        if (*out) {
            output = *out;
        } else {
            const int aRowDim    = (optLhs == FLY_MAT_NONE) ? 0 : 1;
            const int bColDim    = (optRhs == FLY_MAT_NONE) ? 1 : 0;
            const int M          = lDims[aRowDim];
            const int N          = rDims[bColDim];
            const dim_t d2       = std::max(lDims[2], rDims[2]);
            const dim_t d3       = std::max(lDims[3], rDims[3]);
            const fly::dim4 oDims = fly::dim4(M, N, d2, d3);
            FLY_CHECK(fly_create_handle(&output, lhsInfo.ndims(), oDims.get(),
                                      lhs_type));
        }

        switch (lhs_type) {
            case f32:
                gemm<float>(&output, optLhs, optRhs,
                            static_cast<const float *>(alpha), lhs, rhs,
                            static_cast<const float *>(beta));
                break;
            case c32:
                gemm<cfloat>(&output, optLhs, optRhs,
                             static_cast<const cfloat *>(alpha), lhs, rhs,
                             static_cast<const cfloat *>(beta));
                break;
            case f64:
                gemm<double>(&output, optLhs, optRhs,
                             static_cast<const double *>(alpha), lhs, rhs,
                             static_cast<const double *>(beta));
                break;
            case c64:
                gemm<cdouble>(&output, optLhs, optRhs,
                              static_cast<const cdouble *>(alpha), lhs, rhs,
                              static_cast<const cdouble *>(beta));
                break;
            case f16:
                gemm<half>(&output, optLhs, optRhs,
                           static_cast<const half *>(alpha), lhs, rhs,
                           static_cast<const half *>(beta));
                break;
            default: TYPE_ERROR(3, lhs_type);
        }

        std::swap(*out, output);
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_matmul(fly_array *out, const fly_array lhs, const fly_array rhs,
                 const fly_mat_prop optLhs, const fly_mat_prop optRhs) {
    try {
        const ArrayInfo &lhsInfo = getInfo(lhs, false);
        const ArrayInfo &rhsInfo = getInfo(rhs, true);

        if (lhsInfo.isSparse()) {
            return fly_sparse_matmul(out, lhs, rhs, optLhs, optRhs);
        }

        const int aRowDim = (optLhs == FLY_MAT_NONE) ? 0 : 1;
        const int bColDim = (optRhs == FLY_MAT_NONE) ? 1 : 0;

        const fly::dim4 &lDims = lhsInfo.dims();
        const fly::dim4 &rDims = rhsInfo.dims();
        const int M           = lDims[aRowDim];
        const int N           = rDims[bColDim];

        const dim_t d2       = std::max(lDims[2], rDims[2]);
        const dim_t d3       = std::max(lDims[3], rDims[3]);
        const fly::dim4 oDims = fly::dim4(M, N, d2, d3);

        fly_array gemm_out = 0;
        FLY_CHECK(fly_create_handle(&gemm_out, oDims.ndims(), oDims.get(),
                                  lhsInfo.getType()));

        fly_dtype lhs_type = lhsInfo.getType();
        switch (lhs_type) {
            case f16: {
                static const half alpha(1.0f);
                static const half beta(0.0f);
                FLY_CHECK(fly_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case f32: {
                float alpha = 1.f;
                float beta  = 0.f;
                FLY_CHECK(fly_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case c32: {
                cfloat alpha{1.f, 0.f};
                cfloat beta{0.f, 0.f};

                FLY_CHECK(fly_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case f64: {
                double alpha = 1.0;
                double beta  = 0.0;
                FLY_CHECK(fly_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case c64: {
                cdouble alpha{1.0, 0.0};
                cdouble beta{0.0, 0.0};
                FLY_CHECK(fly_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            default: TYPE_ERROR(1, lhs_type);
        }

        std::swap(*out, gemm_out);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_dot(fly_array *out, const fly_array lhs, const fly_array rhs,
              const fly_mat_prop optLhs, const fly_mat_prop optRhs) {
    try {
        const ArrayInfo &lhsInfo = getInfo(lhs);
        const ArrayInfo &rhsInfo = getInfo(rhs);

        if (optLhs != FLY_MAT_NONE && optLhs != FLY_MAT_CONJ) {
            FLY_ERROR("Using this property is not yet supported in dot",
                     FLY_ERR_NOT_SUPPORTED);
        }

        if (optRhs != FLY_MAT_NONE && optRhs != FLY_MAT_CONJ) {
            FLY_ERROR("Using this property is not yet supported in dot",
                     FLY_ERR_NOT_SUPPORTED);
        }

        DIM_ASSERT(1, lhsInfo.dims()[0] == rhsInfo.dims()[0]);
        fly_dtype lhs_type = lhsInfo.getType();
        fly_dtype rhs_type = rhsInfo.getType();

        if (lhsInfo.ndims() == 0) { return fly_retain_array(out, lhs); }
        if (lhsInfo.ndims() > 1 || rhsInfo.ndims() > 1) {
            FLY_ERROR("dot can not be used in batch mode", FLY_ERR_BATCH);
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        fly_array output = 0;

        switch (lhs_type) {
            case f16: output = dot<half>(lhs, rhs, optLhs, optRhs); break;
            case f32: output = dot<float>(lhs, rhs, optLhs, optRhs); break;
            case c32: output = dot<cfloat>(lhs, rhs, optLhs, optRhs); break;
            case f64: output = dot<double>(lhs, rhs, optLhs, optRhs); break;
            case c64: output = dot<cdouble>(lhs, rhs, optLhs, optRhs); break;
            default: TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_dot_all(double *rval, double *ival, const fly_array lhs,
                  const fly_array rhs, const fly_mat_prop optLhs,
                  const fly_mat_prop optRhs) {
    using namespace detail;  // NOLINT needed for imag and real functions
                             // name resolution

    try {
        *rval = 0;
        if (ival) { *ival = 0; }

        fly_array out = 0;
        FLY_CHECK(fly_dot(&out, lhs, rhs, optLhs, optRhs));

        const ArrayInfo &lhsInfo = getInfo(lhs);
        fly_dtype lhs_type        = lhsInfo.getType();

        switch (lhs_type) {
            case f16: *rval = static_cast<double>(dotAll<half>(out)); break;
            case f32: *rval = dotAll<float>(out); break;
            case f64: *rval = dotAll<double>(out); break;
            case c32: {
                cfloat temp = dotAll<cfloat>(out);
                *rval       = real(temp);
                if (ival) { *ival = imag(temp); }
            } break;
            case c64: {
                cdouble temp = dotAll<cdouble>(out);
                *rval        = real(temp);
                if (ival) { *ival = imag(temp); }
            } break;
            default: TYPE_ERROR(1, lhs_type);
        }

        if (out != 0) { FLY_CHECK(fly_release_array(out)); }
    }
    CATCHALL
    return FLY_SUCCESS;
}
