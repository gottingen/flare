/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/SparseArray.hpp>

namespace flare {
namespace cuda {

template<typename T>
Array<T> matmul(const common::SparseArray<T>& lhs, const Array<T>& rhs,
                fly_mat_prop optLhs, fly_mat_prop optRhs);

}  // namespace cuda
}  // namespace flare
