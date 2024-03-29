/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace flare {
namespace cuda {
template<typename T>
void gemm(Array<T> &out, fly_mat_prop optLhs, fly_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta);

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs, fly_mat_prop optLhs,
                fly_mat_prop optRhs) {
    int Mdim     = optLhs == FLY_MAT_NONE ? 0 : 1;
    int Ndim     = optRhs == FLY_MAT_NONE ? 1 : 0;
    Array<T> res = createEmptyArray<T>(
        dim4(lhs.dims()[Mdim], rhs.dims()[Ndim], lhs.dims()[2], lhs.dims()[3]));
    constexpr T alpha = 1.0;
    constexpr T beta  = 0.0;
    gemm(res, optLhs, optRhs, &alpha, lhs, rhs, &beta);
    return res;
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, fly_mat_prop optLhs,
             fly_mat_prop optRhs);

template<typename T>
void trsm(const Array<T> &lhs, Array<T> &rhs, fly_mat_prop trans = FLY_MAT_NONE,
          bool is_upper = false, bool is_left = true, bool is_unit = false);

}  // namespace cuda
}  // namespace flare
