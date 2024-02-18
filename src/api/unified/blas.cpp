/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/blas.h>
#include "symbol_manager.hpp"

FLY_API fly_err fly_gemm(fly_array *out, const fly_mat_prop optLhs,
                     const fly_mat_prop optRhs, const void *alpha,
                     const fly_array lhs, const fly_array rhs, const void *beta) {
    CHECK_ARRAYS(out, lhs, rhs);
    CALL(fly_gemm, out, optLhs, optRhs, alpha, lhs, rhs, beta);
}

fly_err fly_matmul(fly_array *out, const fly_array lhs, const fly_array rhs,
                 const fly_mat_prop optLhs, const fly_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_matmul, out, lhs, rhs, optLhs, optRhs);
}

fly_err fly_dot(fly_array *out, const fly_array lhs, const fly_array rhs,
              const fly_mat_prop optLhs, const fly_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_dot, out, lhs, rhs, optLhs, optRhs);
}

fly_err fly_dot_all(double *rval, double *ival, const fly_array lhs,
                  const fly_array rhs, const fly_mat_prop optLhs,
                  const fly_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_dot_all, rval, ival, lhs, rhs, optLhs, optRhs);
}

fly_err fly_transpose(fly_array *out, fly_array in, const bool conjugate) {
    CHECK_ARRAYS(in);
    CALL(fly_transpose, out, in, conjugate);
}

fly_err fly_transpose_inplace(fly_array in, const bool conjugate) {
    CHECK_ARRAYS(in);
    CALL(fly_transpose_inplace, in, conjugate);
}
