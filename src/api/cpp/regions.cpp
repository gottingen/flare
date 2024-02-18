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

array regions(const array& in, const fly::connectivity connectivity,
              const fly::dtype type) {
    fly_array temp = 0;
    FLY_THROW(fly_regions(&temp, in.get(), connectivity, type));
    return array(temp);
}

}  // namespace fly
