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
namespace cpu {
template<typename T>
Array<T> morph(const Array<T> &in, const Array<T> &mask, bool isDilation);

template<typename T>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask, bool isDilation);
}  // namespace cpu
}  // namespace flare
