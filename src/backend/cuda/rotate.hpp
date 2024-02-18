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
Array<T> rotate(const Array<T> &in, const float theta, const fly::dim4 &odims,
                const fly_interp_type method);
}  // namespace cuda
}  // namespace flare
