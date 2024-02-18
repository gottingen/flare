/*******************************************************
 * Copyright (c) 2016, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/image.h>
#include "symbol_manager.hpp"

fly_err fly_moments(fly_array* out, const fly_array in,
                  const fly_moment_type moment) {
    CHECK_ARRAYS(in);
    CALL(fly_moments, out, in, moment);
}

fly_err fly_moments_all(double* out, const fly_array in,
                      const fly_moment_type moment) {
    CHECK_ARRAYS(in);
    CALL(fly_moments_all, out, in, moment);
}
