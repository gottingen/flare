/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <optypes.hpp>

namespace flare {
namespace opencl {
template<fly_op_t op, typename Ti, typename To>
Array<To> scan(const Array<Ti>& in, const int dim, bool inclusive_scan = true);
}  // namespace opencl
}  // namespace flare
