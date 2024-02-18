/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/lapack.h>
#include "symbol_manager.hpp"

fly_err fly_svd(fly_array *u, fly_array *s, fly_array *vt, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_svd, u, s, vt, in);
}

fly_err fly_svd_inplace(fly_array *u, fly_array *s, fly_array *vt, fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_svd_inplace, u, s, vt, in);
}

fly_err fly_lu(fly_array *lower, fly_array *upper, fly_array *pivot,
             const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_lu, lower, upper, pivot, in);
}

fly_err fly_lu_inplace(fly_array *pivot, fly_array in, const bool is_lapack_piv) {
    CHECK_ARRAYS(in);
    CALL(fly_lu_inplace, pivot, in, is_lapack_piv);
}

fly_err fly_qr(fly_array *q, fly_array *r, fly_array *tau, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_qr, q, r, tau, in);
}

fly_err fly_qr_inplace(fly_array *tau, fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_qr_inplace, tau, in);
}

fly_err fly_cholesky(fly_array *out, int *info, const fly_array in,
                   const bool is_upper) {
    CHECK_ARRAYS(in);
    CALL(fly_cholesky, out, info, in, is_upper);
}

fly_err fly_cholesky_inplace(int *info, fly_array in, const bool is_upper) {
    CHECK_ARRAYS(in);
    CALL(fly_cholesky_inplace, info, in, is_upper);
}

fly_err fly_solve(fly_array *x, const fly_array a, const fly_array b,
                const fly_mat_prop options) {
    CHECK_ARRAYS(a, b);
    CALL(fly_solve, x, a, b, options);
}

fly_err fly_solve_lu(fly_array *x, const fly_array a, const fly_array piv,
                   const fly_array b, const fly_mat_prop options) {
    CHECK_ARRAYS(a, piv, b);
    CALL(fly_solve_lu, x, a, piv, b, options);
}

fly_err fly_inverse(fly_array *out, const fly_array in, const fly_mat_prop options) {
    CHECK_ARRAYS(in);
    CALL(fly_inverse, out, in, options);
}

fly_err fly_pinverse(fly_array *out, const fly_array in, const double tol,
                   const fly_mat_prop options) {
    CHECK_ARRAYS(in);
    CALL(fly_pinverse, out, in, tol, options);
}

fly_err fly_rank(unsigned *rank, const fly_array in, const double tol) {
    CHECK_ARRAYS(in);
    CALL(fly_rank, rank, in, tol);
}

fly_err fly_det(double *det_real, double *det_imag, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_det, det_real, det_imag, in);
}

fly_err fly_norm(double *out, const fly_array in, const fly_norm_type type,
               const double p, const double q) {
    CHECK_ARRAYS(in);
    CALL(fly_norm, out, in, type, p, q);
}

fly_err fly_is_lapack_available(bool *out) { CALL(fly_is_lapack_available, out); }
