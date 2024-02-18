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

array bilateral(const array &in, const float spatial_sigma,
                const float chromatic_sigma, const bool is_color) {
    fly_array out = 0;
    FLY_THROW(
        fly_bilateral(&out, in.get(), spatial_sigma, chromatic_sigma, is_color));
    return array(out);
}

}  // namespace fly
