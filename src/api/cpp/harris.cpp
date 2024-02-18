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

features harris(const array& in, const unsigned max_corners,
                const float min_response, const float sigma,
                const unsigned block_size, const float k_thr) {
    fly_features temp;
    FLY_THROW(fly_harris(&temp, in.get(), max_corners, min_response, sigma,
                       block_size, k_thr));
    return features(temp);
}

}  // namespace fly
