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
namespace opencl {
namespace cpu {

template<typename T>
void gemm(Array<T> &out, fly_mat_prop optLhs, fly_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta);
}  // namespace cpu
}  // namespace opencl
}  // namespace flare
