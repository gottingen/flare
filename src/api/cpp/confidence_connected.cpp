/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include "error.hpp"

namespace fly {

array confidenceCC(const array &in, const size_t num_seeds,
                   const unsigned *seedx, const unsigned *seedy,
                   const unsigned radius, const unsigned multiplier,
                   const int iter, const double segmentedValue) {
    fly::array xs(dim4(num_seeds), seedx);
    fly::array ys(dim4(num_seeds), seedy);
    fly_array temp = 0;
    FLY_THROW(fly_confidence_cc(&temp, in.get(), xs.get(), ys.get(), radius,
                              multiplier, iter, segmentedValue));
    return array(temp);
}

array confidenceCC(const array &in, const array &seeds, const unsigned radius,
                   const unsigned multiplier, const int iter,
                   const double segmentedValue) {
    fly::array xcoords = seeds.col(0);
    fly::array ycoords = seeds.col(1);
    fly_array temp     = 0;
    FLY_THROW(fly_confidence_cc(&temp, in.get(), xcoords.get(), ycoords.get(),
                              radius, multiplier, iter, segmentedValue));
    return array(temp);
}

array confidenceCC(const array &in, const array &seedx, const array &seedy,
                   const unsigned radius, const unsigned multiplier,
                   const int iter, const double segmentedValue) {
    fly_array temp = 0;
    FLY_THROW(fly_confidence_cc(&temp, in.get(), seedx.get(), seedy.get(), radius,
                              multiplier, iter, segmentedValue));
    return array(temp);
}

}  // namespace fly
