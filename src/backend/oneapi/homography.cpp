/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <homography.hpp>

#include <arith.hpp>
#include <err_oneapi.hpp>
#include <fly/dim4.hpp>

#include <algorithm>
#include <limits>

using fly::dim4;
using std::numeric_limits;

namespace flare {
namespace oneapi {

template<typename T>
int homography(Array<T> &bestH, const Array<float> &x_src,
               const Array<float> &y_src, const Array<float> &x_dst,
               const Array<float> &y_dst, const Array<float> &initial,
               const fly_homography_type htype, const float inlier_thr,
               const unsigned iterations) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

#define INSTANTIATE(T)                                                     \
    template int homography(                                               \
        Array<T> &H, const Array<float> &x_src, const Array<float> &y_src, \
        const Array<float> &x_dst, const Array<float> &y_dst,              \
        const Array<float> &initial, const fly_homography_type htype,       \
        const float inlier_thr, const unsigned iterations);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace oneapi
}  // namespace flare
