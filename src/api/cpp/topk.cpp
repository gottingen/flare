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
