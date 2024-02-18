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
Array<T> exampleFunction(const Array<T> &a, const Array<T> &b,
                         const fly_someenum_t method);
}  // namespace oneapi
}  // namespace flare
