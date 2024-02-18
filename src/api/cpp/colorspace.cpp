/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/compatible.h>
#include <fly/defines.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {

array colorspace(const array& image, const CSpace to, const CSpace from) {
    return colorSpace(image, to, from);
}

array colorSpace(const array& image, const CSpace to, const CSpace from) {
    fly_array temp = 0;
    FLY_THROW(fly_color_space(&temp, image.get(), to, from));
    return array(temp);
}

}  // namespace fly
