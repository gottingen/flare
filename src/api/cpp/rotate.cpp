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

array rotate(const array& in, const float theta, const bool crop,
             const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_rotate(&out, in.get(), theta, crop, method));
    return array(out);
}

}  // namespace fly
