// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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
