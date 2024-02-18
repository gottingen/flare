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
Array<T> reorder(const Array<T> &in, const fly::dim4 &rdims);
}  // namespace cuda
}  // namespace flare
