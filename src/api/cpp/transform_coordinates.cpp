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

array transformCoordinates(const array& tf, const float d0, const float d1) {
    fly_array out = 0;
    FLY_THROW(fly_transform_coordinates(&out, tf.get(), d0, d1));
    return array(out);
}

}  // namespace fly
