/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/vision.h>
#include "error.hpp"

namespace fly {

array dog(const array& in, const int radius1, const int radius2) {
    fly_array temp = 0;
    FLY_THROW(fly_dog(&temp, in.get(), radius1, radius2));
    return array(temp);
}

}  // namespace fly
