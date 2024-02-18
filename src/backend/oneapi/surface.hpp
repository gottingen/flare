/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/graphics_common.hpp>

namespace flare {
namespace oneapi {

template<typename T>
void copy_surface(const Array<T> &P, fg_surface surface);

}  // namespace oneapi
}  // namespace flare
