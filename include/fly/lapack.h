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

#pragma once
#include <fly/array.h>
#include <fly/defines.h>

#ifdef __cplusplus
namespace fly
{
    /**
       C++ Interface to perform singular value decomposition.

       \param[out] u  U
       \param[out] s  diagonal values of sigma (singular values of the input 
                      matrix)
       \param[out] vt V^H
       \param[in]  in input array

       \ingroup lapack_factor_func_svd
    */
    FLY_API void svd(array &u, array &s, array &vt, const array &in);

    /**
       C++ Interface to perform in-place singular value decomposition.

       This function minimizes memory usage if `in` is dispensable. Input array
       `in` is limited to arrays where `dim0` \f$\geq\f$ `dim1`.

       \param[out]   u  U
       \param[out]   s  diagonal values of sigma (singular values of the input
                        matrix)
       \param[out]   vt V^H
       \param[inout] in input array; contains random data after the operation                       this operation

       \ingroup lapack_factor_func_svd
    */
    FLY_API void svdInPlace(array &u, array &s, array &vt, array &in);

    /**
       C++ Interface to perform LU decomposition in packed format.

       This function is not supported in GFOR.

       \param[out] out           packed LU decomposition
       \param[out] pivot         permutation indices mapping the input to the
                                 decomposition
       \param[in]  in            input array
       \param[in]  is_lapack_piv specifies if the pivot is returned in original
                                 LAPACK compliant format

       \ingroup lapack_factor_func_lu
    */
    FLY_API void lu(array &out, array &pivot, const array &in, const bool is_lapack_piv=true);

    /**
       C++ Interface to perform LU decomposition.

       This function is not supported in GFOR.

       \param[out] lower lower triangular matrix of the LU decomposition
       \param[out] upper upper triangular matrix of the LU decomposition
       \param[out] pivot permutation indices mapping the input to the
                         decomposition
       \param[in]  in    input array

       \ingroup lapack_factor_func_lu
    */
    FLY_API void lu(array &lower, array &upper, array &pivot, const array &in);

    /**
       C++ Interface to perform in-place LU decomposition.

       This function is not supported in GFOR.

       \param[out]   pivot         permutation indices mapping the input to the
                                   decomposition
       \param[inout] in            input array on entry; packed LU
                                   decomposition on exit
       \param[in]    is_lapack_piv specifies if the pivot is returned in
                                   original LAPACK-compliant format

       \ingroup lapack_factor_func_lu
    */
    FLY_API void luInPlace(array &pivot, array &in, const bool is_lapack_piv=true);

    /**
       C++ Interface to perform QR decomposition in packed format.

       This function is not supported in GFOR.

       \param[out] out packed QR decomposition
       \param[out] tau additional information needed for unpacking the data
       \param[in]  in  input array

       \ingroup lapack_factor_func_qr
    */
    FLY_API void qr(array &out, array &tau, const array &in);

    /**
       C++ Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out] q   orthogonal matrix from QR decomposition
       \param[out] r   upper triangular matrix from QR decomposition
       \param[out] tau additional information needed for solving a
                       least-squares problem using `q` and `r`
       \param[in]  in  input array

       \ingroup lapack_factor_func_qr
    */
    FLY_API void qr(array &q, array &r, array &tau, const array &in);

    /**
       C++ Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out]   tau additional information needed for unpacking the data
       \param[inout] in  input array on entry; packed QR decomposition on exit

       \ingroup lapack_factor_func_qr
    */
    FLY_API void qrInPlace(array &tau, array &in);

    /**
       C++ Interface to perform Cholesky decomposition.

       Multiplying `out` with its conjugate transpose reproduces the input
       `in`.
       
       The input must be positive definite.
       
       This function is not supported in GFOR.

       \param[out] out      triangular matrix; 
       \param[in]  in       input matrix
       \param[in]  is_upper boolean determining if `out` is upper or lower
                            triangular
       \returns    `0` if cholesky decomposition passes; if not, it returns the
                   rank at which the decomposition fails

       \ingroup lapack_factor_func_cholesky
    */
    FLY_API int cholesky(array &out, const array &in, const bool is_upper = true);

    /**
       C++ Interface to perform in-place Cholesky decomposition.

       The input must be positive definite.

       This function is not supported in GFOR.

       \param[inout] in       input matrix on entry; triangular matrix on exit
       \param[in]    is_upper boolean determining if `in` is upper or lower
                              triangular
       \returns      `0` if cholesky decomposition passes; if not, it returns
                     the rank at which the decomposition fails

       \ingroup lapack_factor_func_cholesky
    */
    FLY_API int choleskyInPlace(array &in, const bool is_upper = true);

    /**
       C++ Interface to solve a system of equations.

       The `options` parameter must be one of \ref FLY_MAT_NONE,
       \ref FLY_MAT_LOWER or \ref FLY_MAT_UPPER.

       This function is not supported in GFOR.

       \param[in] a       coefficient matrix
       \param[in] b       measured values
       \param[in] options determines various properties of matrix `a`
       \returns   `x`, the matrix of unknown variables

       \ingroup lapack_solve_func_gen
    */
    FLY_API array solve(const array &a, const array &b, const matProp options = FLY_MAT_NONE);

    /**
       C++ Interface to solve a system of equations.

       The `options` parameter currently must be \ref FLY_MAT_NONE.

       This function is not supported in GFOR.

       \param[in] a       packed LU decomposition of the coefficient matrix
       \param[in] piv     pivot array from the packed LU decomposition of the
                          coefficient matrix
       \param[in] b       measured values
       \param[in] options determines various properties of matrix `a`
       \returns   `x`, the matrix of unknown variables

       \ingroup lapack_solve_lu_func_gen
    */
    FLY_API array solveLU(const array &a, const array &piv,
                        const array &b, const matProp options = FLY_MAT_NONE);

    /**
       C++ Interface to invert a matrix.

       The `options` parameter currently must be \ref FLY_MAT_NONE.

       This function is not supported in GFOR.

       \param[in] in      input matrix
       \param[in] options determines various properties of matrix `in`
       \returns   inverse matrix

       \ingroup lapack_ops_func_inv
    */
    FLY_API array inverse(const array &in, const matProp options = FLY_MAT_NONE);

    /**
       C++ Interface to pseudo-invert (Moore-Penrose) a matrix.

       Currently uses the SVD-based approach.

       Parameter `tol` is not the actual lower threshold, but it is passed in
       as a parameter to the calculation of the actual threshold relative to
       the shape and contents of `in`.
       
       This function is not supported in GFOR.

       \param[in] in      input matrix
       \param[in] tol     defines the lower threshold for singular values from
                          SVD
       \param[in] options must be FLY_MAT_NONE (more options might be supported
                          in the future)
       \returns   pseudo-inverse matrix

       \ingroup lapack_ops_func_pinv
    */
    FLY_API array pinverse(const array &in, const double tol=1E-6,
                         const matProp options = FLY_MAT_NONE);

    /**
       C++ Interface to find the rank of a matrix.

       \param[in] in  input matrix
       \param[in] tol tolerance value
       \returns   rank

       \ingroup lapack_ops_func_rank
    */
    FLY_API unsigned rank(const array &in, const double tol=1E-5);

    /**
       C++ Interface to find the determinant of a matrix.

       \param[in] in input matrix
       \returns   determinant

       \ingroup lapack_ops_func_det
    */
    template<typename T> T det(const array &in);

    /**
       C++ Interface to find the norm of a matrix.

       \param[in] in   input matrix
       \param[in] type \ref fly::normType. Default: \ref FLY_NORM_VECTOR_1
       \param[in] p    value of P when `type` is \ref FLY_NORM_VECTOR_P or
                       \ref FLY_NORM_MATRIX_L_PQ, else ignored
       \param[in] q    value of Q when `type` is \ref FLY_NORM_MATRIX_L_PQ, else
                       ignored
       \returns   norm

       \ingroup lapack_ops_func_norm
    */
    FLY_API double norm(const array &in, const normType type=FLY_NORM_EUCLID,
                      const double p=1, const double q=1);

    /**
       Returns true if Flare is compiled with LAPACK support.

       \returns true if LAPACK support is available; false otherwise

       \ingroup lapack_helper_func_available
    */
    FLY_API bool isLAPACKAvailable();

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface to perform singular value decomposition.

       \param[out] u  U
       \param[out] s  diagonal values of sigma (singular values of the input
                      matrix)
       \param[out] vt V^H
       \param[in]  in input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_factor_func_svd
    */
    FLY_API fly_err fly_svd(fly_array *u, fly_array *s, fly_array *vt, const fly_array in);

    /**
       C Interface to perform in-place singular value decomposition.

       This function minimizes memory usage if `in` is dispensable. Input array
       `in` is limited to arrays where `dim0` \f$\geq\f$ `dim1`.

       \param[out]   u  U
       \param[out]   s  diagonal values of sigma (singular values of the input
                        matrix)
       \param[out]   vt V^H
       \param[inout] in input array; contains random data after the operation                       this operation
       \return       \ref FLY_SUCCESS, if function returns successfully, else
                     an \ref fly_err code is given

       \ingroup lapack_factor_func_svd
    */
    FLY_API fly_err fly_svd_inplace(fly_array *u, fly_array *s, fly_array *vt, fly_array in);

    /**
       C Interface to perform LU decomposition.

       \param[out] lower lower triangular matrix of the LU decomposition
       \param[out] upper upper triangular matrix of the LU decomposition
       \param[out] pivot permutation indices mapping the input to the
                         decomposition
       \param[in]  in    input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_factor_func_lu
    */
    FLY_API fly_err fly_lu(fly_array *lower, fly_array *upper, fly_array *pivot, const fly_array in);

    /**
       C Interface to perform in-place LU decomposition.

       This function is not supported in GFOR.

       \param[out]   pivot         permutation indices mapping the input to the
                                   decomposition
       \param[inout] in            input array on entry; packed LU
                                   decomposition on exit
       \param[in]    is_lapack_piv specifies if the pivot is returned in
                                   original LAPACK-compliant format
       \return       \ref FLY_SUCCESS, if function returns successfully, else
                     an \ref fly_err code is given

       \ingroup lapack_factor_func_lu
    */
    FLY_API fly_err fly_lu_inplace(fly_array *pivot, fly_array in, const bool is_lapack_piv);

    /**
       C Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out] q   orthogonal matrix from QR decomposition
       \param[out] r   upper triangular matrix from QR decomposition
       \param[out] tau additional information needed for solving a
                       least-squares problem using `q` and `r`
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_factor_func_qr
    */
    FLY_API fly_err fly_qr(fly_array *q, fly_array *r, fly_array *tau, const fly_array in);

    /**
       C Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out]   tau additional information needed for unpacking the data
       \param[inout] in  input array on entry; packed QR decomposition on exit
       \return       \ref FLY_SUCCESS, if function returns successfully, else
                     an \ref fly_err code is given

       \ingroup lapack_factor_func_qr
    */
    FLY_API fly_err fly_qr_inplace(fly_array *tau, fly_array in);

    /**
       C Interface to perform Cholesky decomposition.

       Multiplying `out` with its conjugate transpose reproduces the input
       `in`.

       The input must be positive definite.

       \param[out] out      triangular matrix;
       \param[out] info     `0` if cholesky decomposition passes; if not, it
                            returns the rank at which the decomposition fails
       \param[in]  in       input matrix
       \param[in]  is_upper boolean determining if `out` is upper or lower
                            triangular
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_factor_func_cholesky
    */
    FLY_API fly_err fly_cholesky(fly_array *out, int *info, const fly_array in, const bool is_upper);

    /**
       C Interface to perform in-place Cholesky decomposition.

       The input must be positive definite.

       \param[out]   info     `0` if cholesky decomposition passes; if not, it
                              returns the rank at which the decomposition fails
       \param[inout] in       input matrix on entry; triangular matrix on exit
       \param[in]    is_upper boolean determining if `in` is upper or lower
                              triangular
       \return       \ref FLY_SUCCESS, if function returns successfully, else
                     an \ref fly_err code is given

       \ingroup lapack_factor_func_cholesky
    */
    FLY_API fly_err fly_cholesky_inplace(int *info, fly_array in, const bool is_upper);

    /**
       C Interface to solve a system of equations.

       The `options` parameter must be one of \ref FLY_MAT_NONE,
       \ref FLY_MAT_LOWER or \ref FLY_MAT_UPPER.

       This function is not supported in GFOR.

       \param[out] x       matrix of unknown variables
       \param[in]  a       coefficient matrix
       \param[in]  b       measured values
       \param[in]  options determines various properties of matrix `a`
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_solve_func_gen
    */
    FLY_API fly_err fly_solve(fly_array *x, const fly_array a, const fly_array b,
                          const fly_mat_prop options);

    /**
       C Interface to solve a system of equations.

       The `options` parameter currently must be \ref FLY_MAT_NONE.

       \param[out] x       matrix of unknown variables
       \param[in]  a       packed LU decomposition of the coefficient matrix
       \param[in]  piv     pivot array from the packed LU decomposition of the
                           coefficient matrix
       \param[in]  b       measured values
       \param[in]  options determines various properties of matrix `a`
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_solve_lu_func_gen
    */
    FLY_API fly_err fly_solve_lu(fly_array *x, const fly_array a, const fly_array piv,
                             const fly_array b, const fly_mat_prop options);

    /**
       C Interface to invert a matrix.

       The `options` parameter currently must be \ref FLY_MAT_NONE.

       \param[out] out     inverse matrix
       \param[in]  in      input matrix
       \param[in]  options determines various properties of matrix `in`
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_ops_func_inv
    */
    FLY_API fly_err fly_inverse(fly_array *out, const fly_array in, const fly_mat_prop options);

    /**
       C Interface to pseudo-invert (Moore-Penrose) a matrix.

       Currently uses the SVD-based approach.

       Parameter `tol` is not the actual lower threshold, but it is passed in
       as a parameter to the calculation of the actual threshold relative to
       the shape and contents of `in`.

       Suggested parameters for `tol`:  1e-6 for single precision and 1e-12 for
       double precision.

       \param[out] out     pseudo-inverse matrix
       \param[in]  in      input matrix
       \param[in]  tol     defines the lower threshold for singular values from
                           SVD
       \param[in]  options must be FLY_MAT_NONE (more options might be supported
                           in the future)
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_ops_func_pinv
    */
    FLY_API fly_err fly_pinverse(fly_array *out, const fly_array in, const double tol,
                             const fly_mat_prop options);

    /**
       C Interface to find the rank of a matrix.

       \param[out] rank rank
       \param[in]  in   input matrix
       \param[in]  tol  tolerance value
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_ops_func_rank
    */
    FLY_API fly_err fly_rank(unsigned *rank, const fly_array in, const double tol);

    /**
       C Interface to find the determinant of a matrix.

       \param[out] det_real real part of the determinant
       \param[out] det_imag imaginary part of the determinant
       \param[in]  in       input matrix
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_ops_func_det
    */
    FLY_API fly_err fly_det(double *det_real, double *det_imag, const fly_array in);

    /**
       C Interface to find the norm of a matrix.

       \param[out] out  norm
       \param[in]  in   input matrix
       \param[in]  type \ref fly::normType. Default: \ref FLY_NORM_VECTOR_1
       \param[in]  p    value of P when `type` is \ref FLY_NORM_VECTOR_P or
                        \ref FLY_NORM_MATRIX_L_PQ, else ignored
       \param[in]  q    value of Q when `type` is \ref FLY_NORM_MATRIX_L_PQ, else
                        ignored
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup lapack_ops_func_norm
    */
    FLY_API fly_err fly_norm(double *out, const fly_array in, const fly_norm_type type, const double p, const double q);

    /**
       Returns true if Flare is compiled with LAPACK support.

       \param[out] out true if LAPACK support is available; false otherwise
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given; does not depend on the value
                   of `out`

       \ingroup lapack_helper_func_available
    */
    FLY_API fly_err fly_is_lapack_available(bool *out);


#ifdef __cplusplus
}
#endif
