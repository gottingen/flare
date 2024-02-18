/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace flare {
namespace oneapi {
template<typename T>
void transform(Array<T> &out, const Array<T> &in, const Array<float> &tf,
               const fly_interp_type method, const bool inverse,
               const bool perspective);
}  // namespace oneapi
}  // namespace flare
