/*******************************************************
 * Copyright (c) 2014, Flare
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

array rgb2gray(const array& in, const float rPercent, const float gPercent,
               const float bPercent) {
    fly_array temp = 0;
    FLY_THROW(fly_rgb2gray(&temp, in.get(), rPercent, gPercent, bPercent));
    return array(temp);
}

array gray2rgb(const array& in, const float rFactor, const float gFactor,
               const float bFactor) {
    fly_array temp = 0;
    FLY_THROW(fly_gray2rgb(&temp, in.get(), rFactor, gFactor, bFactor));
    return array(temp);
}

}  // namespace fly
