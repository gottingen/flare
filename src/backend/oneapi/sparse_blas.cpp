/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>

#include <common/err_common.hpp>
#include <complex.hpp>
#include <err_oneapi.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <transpose.hpp>
#include <fly/dim4.hpp>

#include <oneapi/mkl/spblas.hpp>

#include <sycl/sycl.hpp>

#include <cassert>
#include <stdexcept>
#include <string>

namespace flare {
namespace oneapi {

using namespace common;

// Converts an fly_mat_prop options to a transpose type for mkl
static ::oneapi::mkl::transpose toBlasTranspose(fly_mat_prop opt) {
    switch (opt) {
        case FLY_MAT_NONE: return ::oneapi::mkl::transpose::nontrans;
        case FLY_MAT_TRANS: return ::oneapi::mkl::transpose::trans;
        case FLY_MAT_CTRANS: return ::oneapi::mkl::transpose::conjtrans;
        default: FLY_ERROR("INVALID fly_mat_prop", FLY_ERR_ARG);
    }
}

template<typename T>
Array<T> matmul(const common::SparseArray<T>& lhs, const Array<T>& rhsIn,
                fly_mat_prop optLhs, fly_mat_prop optRhs) {
    int lRowDim = (optLhs == FLY_MAT_NONE) ? 0 : 1;
    static const int rColDim =
        1;  // Unsupported : (optRhs == FLY_MAT_NONE) ? 1 : 0;

    dim4 lDims    = lhs.dims();
    dim4 rDims    = rhsIn.dims();
    dim4 rStrides = rhsIn.strides();
    int M         = lDims[lRowDim];
    int N         = rDims[rColDim];

    Array<T> out  = createEmptyArray<T>(fly::dim4(M, N, 1, 1));
    dim4 oStrides = out.strides();

    static const T alpha = scalar<T>(1.0);
    static const T beta  = scalar<T>(0.0);

    const Array<T>& values      = lhs.getValues();
    const Array<int>& rowIdx    = lhs.getRowIdx();
    const Array<int>& colIdx    = lhs.getColIdx();
    sycl::buffer<T, 1> valBuf   = values.template getBufferWithOffset<T>();
    sycl::buffer<int, 1> rowBuf = rowIdx.template getBufferWithOffset<int>();
    sycl::buffer<int, 1> colBuf = colIdx.template getBufferWithOffset<int>();

    const auto lOpts = toBlasTranspose(optLhs);
    const auto rOpts = toBlasTranspose(optRhs);

    sycl::buffer<T, 1> rhsBuf = rhsIn.template getBufferWithOffset<T>();
    sycl::buffer<T, 1> outBuf = out.template getBufferWithOffset<T>();

    ::oneapi::mkl::sparse::matrix_handle_t CSRHandle = nullptr;
    ::oneapi::mkl::sparse::init_matrix_handle(&CSRHandle);
    ::oneapi::mkl::sparse::set_csr_data(
        getQueue(), CSRHandle, lDims[0], lDims[1],
        ::oneapi::mkl::index_base::zero, rowBuf, colBuf, valBuf);

    if (N == 1) {
        ::oneapi::mkl::sparse::gemv(getQueue(), lOpts, alpha, CSRHandle, rhsBuf,
                                    beta, outBuf);
    } else {
        ::oneapi::mkl::sparse::gemm(
            getQueue(), ::oneapi::mkl::layout::col_major, lOpts, rOpts, alpha,
            CSRHandle, rhsBuf, N, rStrides[1], beta, outBuf, oStrides[1]);
    }
    ::oneapi::mkl::sparse::release_matrix_handle(getQueue(), &CSRHandle);
    return out;
}

#define INSTANTIATE_SPARSE(T)                                            \
    template Array<T> matmul<T>(const common::SparseArray<T>& lhs,       \
                                const Array<T>& rhs, fly_mat_prop optLhs, \
                                fly_mat_prop optRhs);

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}  // namespace oneapi
}  // namespace flare
