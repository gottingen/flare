/*******************************************************
 * Copyright (c) 2018, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/statistics.h>
#include "common.hpp"
#include "error.hpp"

namespace fly {
void topk(array &values, array &indices, const array &in, const int k,
          const int dim, const topkFunction order) {
    fly_array fly_vals = 0;
    fly_array fly_idxs = 0;

    FLY_THROW(fly_topk(&fly_vals, &fly_idxs, in.get(), k, dim, order));

    values  = array(fly_vals);
    indices = array(fly_idxs);
}
}  // namespace fly
