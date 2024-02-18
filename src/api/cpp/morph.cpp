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

array dilate(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_dilate(&out, in.get(), mask.get()));
    return array(out);
}

array dilate3(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_dilate3(&out, in.get(), mask.get()));
    return array(out);
}

array erode(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_erode(&out, in.get(), mask.get()));
    return array(out);
}

array erode3(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_erode3(&out, in.get(), mask.get()));
    return array(out);
}

}  // namespace fly
