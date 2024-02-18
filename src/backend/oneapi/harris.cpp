/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_oneapi.hpp>
#include <fly/dim4.hpp>
#include <fly/features.h>

using fly::dim4;
using fly::features;

namespace flare {
namespace oneapi {

template<typename T, typename convAccT>
unsigned harris(Array<float> &x_out, Array<float> &y_out,
                Array<float> &score_out, const Array<T> &in,
                const unsigned max_corners, const float min_response,
                const float sigma, const unsigned filter_len,
                const float k_thr) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned harris<T, convAccT>(                                    \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const unsigned max_corners,                       \
        const float min_response, const float sigma,                          \
        const unsigned filter_len, const float k_thr);

INSTANTIATE(double, double)
INSTANTIATE(float, float)

}  // namespace oneapi
}  // namespace flare
