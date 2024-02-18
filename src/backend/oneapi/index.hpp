/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <fly/index.h>

namespace flare {
namespace oneapi {

template<typename T>
Array<T> index(const Array<T>& in, const fly_index_t idxrs[]);

}  // namespace oneapi
}  // namespace flare
