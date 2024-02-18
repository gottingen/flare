/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/index.h>

namespace flare {
namespace cpu {
template<typename T>
class Array;

template<typename T>
void assign(Array<T>& out, const fly_index_t idxrs[], const Array<T>& rhs);

}  // namespace cpu
}  // namespace flare
