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
#include "error.hpp"

namespace fly {
array diff1(const array &in, const int dim) {
    fly_array out = 0;
    FLY_THROW(fly_diff1(&out, in.get(), dim));
    return array(out);
}

array diff2(const array &in, const int dim) {
    fly_array out = 0;
    FLY_THROW(fly_diff2(&out, in.get(), dim));
    return array(out);
}
}  // namespace fly
