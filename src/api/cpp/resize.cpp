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

array resize(const array &in, const dim_t odim0, const dim_t odim1,
             const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_resize(&out, in.get(), odim0, odim1, method));
    return array(out);
}

array resize(const float scale0, const float scale1, const array &in,
             const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_resize(&out, in.get(), in.dims(0) * scale0, in.dims(1) * scale1,
                       method));
    return array(out);
}

array resize(const float scale, const array &in, const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_resize(&out, in.get(), in.dims(0) * scale, in.dims(1) * scale,
                       method));
    return array(out);
}

}  // namespace fly
