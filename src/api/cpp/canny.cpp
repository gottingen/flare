/*******************************************************
 * Copyright (c) 2017, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {
array canny(const array& in, const cannyThreshold ctType, const float ltr,
            const float htr, const unsigned sW, const bool isFast) {
    fly_array temp = 0;
    FLY_THROW(fly_canny(&temp, in.get(), ctType, ltr, htr, sW, isFast));
    return array(temp);
}
}  // namespace fly
