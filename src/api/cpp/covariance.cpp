/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/statistics.h>
#include "error.hpp"

namespace fly {

array cov(const array& X, const array& Y, const bool isbiased) {
    const fly_var_bias bias =
        (isbiased ? FLY_VARIANCE_SAMPLE : FLY_VARIANCE_POPULATION);
    return cov(X, Y, bias);
}

array cov(const array& X, const array& Y, const fly_var_bias bias) {
    fly_array temp = 0;
    FLY_THROW(fly_cov_v2(&temp, X.get(), Y.get(), bias));
    return array(temp);
}

}  // namespace fly
