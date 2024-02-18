/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/defines.h>
#include <fly/vision.h>

fly_err fly_hamming_matcher(fly_array* idx, fly_array* dist, const fly_array query,
                          const fly_array train, const dim_t dist_dim,
                          const unsigned n_dist) {
    return fly_nearest_neighbour(idx, dist, query, train, dist_dim, n_dist,
                                FLY_SHD);
}
