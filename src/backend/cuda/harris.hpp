/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <fly/features.h>

using fly::features;

namespace flare {
namespace cuda {

template<typename T, typename convAccT>
unsigned harris(Array<float> &x_out, Array<float> &y_out,
                Array<float> &score_out, const Array<T> &in,
                const unsigned max_corners, const float min_response,
                const float sigma, const unsigned filter_len,
                const float k_thr);

}  // namespace cuda
}  // namespace flare
