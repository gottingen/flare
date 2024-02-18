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

void homography(array &H, int &inliers, const array &x_src, const array &y_src,
                const array &x_dst, const array &y_dst,
                const fly_homography_type htype, const float inlier_thr,
                const unsigned iterations, const fly::dtype otype) {
    fly_array outH;
    FLY_THROW(fly_homography(&outH, &inliers, x_src.get(), y_src.get(),
                           x_dst.get(), y_dst.get(), htype, inlier_thr,
                           iterations, otype));

    H = array(outH);
}

}  // namespace fly
