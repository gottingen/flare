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
