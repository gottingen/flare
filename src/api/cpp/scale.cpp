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

array scale(const array& in, const float scale0, const float scale1,
            const dim_t odim0, const dim_t odim1, const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_scale(&out, in.get(), scale0, scale1, odim0, odim1, method));
    return array(out);
}

}  // namespace fly
