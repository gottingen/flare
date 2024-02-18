/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/vision.h>
#include "error.hpp"

namespace fly {

void nearestNeighbour(array& idx, array& dist, const array& query,
                      const array& train, const dim_t dist_dim,
                      const unsigned n_dist, const fly_match_type dist_type) {
    fly_array temp_idx  = 0;
    fly_array temp_dist = 0;
    FLY_THROW(fly_nearest_neighbour(&temp_idx, &temp_dist, query.get(),
                                  train.get(), dist_dim, n_dist, dist_type));
    idx  = array(temp_idx);
    dist = array(temp_dist);
}

}  // namespace fly
