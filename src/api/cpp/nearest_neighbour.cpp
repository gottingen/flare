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
