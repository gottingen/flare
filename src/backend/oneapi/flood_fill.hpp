/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <fly/defines.h>

namespace flare {
namespace oneapi {
template<typename T>
Array<T> floodFill(const Array<T>& image, const Array<uint>& seedsX,
                   const Array<uint>& seedsY, const T newValue,
                   const T lowValue, const T highValue,
                   const fly::connectivity nlookup = FLY_CONNECTIVITY_8_4);
}
}  // namespace flare
