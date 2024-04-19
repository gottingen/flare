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

#include <fly/defines.h>
#include <fly/vision.h>

fly_err fly_hamming_matcher(fly_array* idx, fly_array* dist, const fly_array query,
                          const fly_array train, const dim_t dist_dim,
                          const unsigned n_dist) {
    return fly_nearest_neighbour(idx, dist, query, train, dist_dim, n_dist,
                                FLY_SHD);
}
