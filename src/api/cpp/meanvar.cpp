/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/statistics.h>
#include "error.hpp"

using fly::array;

namespace fly {
void meanvar(array& mean, array& var, const array& in, const array& weights,
             const fly_var_bias bias, const dim_t dim) {
    fly_array mean_ = mean.get();
    fly_array var_  = var.get();
    FLY_THROW(fly_meanvar(&mean_, &var_, in.get(), weights.get(), bias, dim));
    mean.set(mean_);
    var.set(var_);
}
}  // namespace fly
