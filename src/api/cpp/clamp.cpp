/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/gfor.h>
#include "error.hpp"

namespace fly {
array clamp(const array &in, const array &lo, const array &hi) {
    fly_array out;
    FLY_THROW(fly_clamp(&out, in.get(), lo.get(), hi.get(), gforGet()));
    return array(out);
}

array clamp(const array &in, const array &lo, const double hi) {
    return clamp(in, lo, constant(hi, lo.dims(), lo.type()));
}

array clamp(const array &in, const double lo, const array &hi) {
    return clamp(in, constant(lo, hi.dims(), hi.type()), hi);
}

array clamp(const array &in, const double lo, const double hi) {
    return clamp(in, constant(lo, in.dims(), in.type()),
                 constant(hi, in.dims(), in.type()));
}
}  // namespace fly
