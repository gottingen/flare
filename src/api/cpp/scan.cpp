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
array accum(const array& in, const int dim) {
    fly_array out = 0;
    FLY_THROW(fly_accum(&out, in.get(), dim));
    return array(out);
}

array scan(const array& in, const int dim, binaryOp op, bool inclusive_scan) {
    fly_array out = 0;
    FLY_THROW(fly_scan(&out, in.get(), dim, op, inclusive_scan));
    return array(out);
}

array scanByKey(const array& key, const array& in, const int dim, binaryOp op,
                bool inclusive_scan) {
    fly_array out = 0;
    FLY_THROW(
        fly_scan_by_key(&out, key.get(), in.get(), dim, op, inclusive_scan));
    return array(out);
}
}  // namespace fly
