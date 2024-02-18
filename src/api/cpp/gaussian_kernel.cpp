/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <fly/algorithm.h>
#include <fly/array.h>
#include <fly/compatible.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include "error.hpp"

namespace fly {
array gaussianKernel(const int rows, const int cols, const double sig_r,
                     const double sig_c) {
    fly_array res;
    FLY_THROW(fly_gaussian_kernel(&res, rows, cols, sig_r, sig_c));
    return array(res);
}

// Compatible function
array gaussiankernel(const int rows, const int cols, const double sig_r,
                     const double sig_c) {
    return gaussianKernel(rows, cols, sig_r, sig_c);
}

}  // namespace fly
