/*******************************************************
 * Copyright (c) 2016, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/defines.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {

array moments(const array& in, const fly_moment_type moment) {
    fly_array out = 0;
    FLY_THROW(fly_moments(&out, in.get(), moment));
    return array(out);
}

void moments(double* out, const array& in, const fly_moment_type moment) {
    FLY_THROW(fly_moments_all(out, in.get(), moment));
}

}  // namespace fly
