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
void assign(Array<T>& out, const fly_index_t idxrs[], const Array<T>& rhs);

}  // namespace oneapi
}  // namespace flare
