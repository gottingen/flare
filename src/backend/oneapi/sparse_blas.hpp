/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/SparseArray.hpp>
#include <sparse.hpp>

namespace flare {
namespace oneapi {

template<typename T>
Array<T> matmul(const common::SparseArray<T>& lhs, const Array<T>& rhs,
                fly_mat_prop optLhs, fly_mat_prop optRhs);

}  // namespace oneapi
}  // namespace flare
