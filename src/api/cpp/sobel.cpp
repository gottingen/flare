/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/arith.h>
#include <fly/array.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {

void sobel(array &dx, array &dy, const array &img, const unsigned ker_size) {
    fly_array fly_dx = 0;
    fly_array fly_dy = 0;
    FLY_THROW(fly_sobel_operator(&fly_dx, &fly_dy, img.get(), ker_size));
    dx = array(fly_dx);
    dy = array(fly_dy);
}

array sobel(const array &img, const unsigned ker_size, const bool isFast) {
    array dx;
    array dy;
    sobel(dx, dy, img, ker_size);
    if (isFast) {
        return abs(dx) + abs(dy);
    } else {
        return sqrt(dx * dx + dy * dy);
    }
}

}  // namespace fly
