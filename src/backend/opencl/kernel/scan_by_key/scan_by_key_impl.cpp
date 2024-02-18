/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <kernel/scan_dim_by_key_impl.hpp>
#include <kernel/scan_first_by_key_impl.hpp>

// This file instantiates scan_dim_by_key as separate object files from CMake
// The line below is read by CMake to determenine the instantiations
// SBK_BINARY_OPS:fly_add_t fly_mul_t fly_max_t fly_min_t

namespace flare {
namespace opencl {
namespace kernel {
INSTANTIATE_SCAN_FIRST_BY_KEY_OP(TYPE)
INSTANTIATE_SCAN_DIM_BY_KEY_OP(TYPE)
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
