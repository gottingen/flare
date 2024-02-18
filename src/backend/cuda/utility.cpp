/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <utility.hpp>

#include <err_cuda.hpp>

namespace flare {
namespace cuda {

int interpOrder(const fly_interp_type p) noexcept {
    int order = 1;
    switch (p) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER: order = 1; break;
        case FLY_INTERP_LINEAR:
        case FLY_INTERP_BILINEAR:
        case FLY_INTERP_LINEAR_COSINE:
        case FLY_INTERP_BILINEAR_COSINE: order = 2; break;
        case FLY_INTERP_CUBIC:
        case FLY_INTERP_BICUBIC:
        case FLY_INTERP_CUBIC_SPLINE:
        case FLY_INTERP_BICUBIC_SPLINE: order = 3; break;
    }
    return order;
}

}  // namespace cuda
}  // namespace flare
