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

array hsv2rgb(const array& in) {
    fly_array temp = 0;
    FLY_THROW(fly_hsv2rgb(&temp, in.get()));
    return array(temp);
}

array rgb2hsv(const array& in) {
    fly_array temp = 0;
    FLY_THROW(fly_rgb2hsv(&temp, in.get()));
    return array(temp);
}

}  // namespace fly
