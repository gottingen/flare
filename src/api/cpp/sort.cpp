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

#include <fly/algorithm.h>
#include <fly/array.h>
#include "error.hpp"

namespace fly {
array sort(const array &in, const unsigned dim, const bool isAscending) {
    fly_array out = 0;
    FLY_THROW(fly_sort(&out, in.get(), dim, isAscending));
    return array(out);
}

void sort(array &out, array &indices, const array &in, const unsigned dim,
          const bool isAscending) {
    fly_array out_, indices_;
    FLY_THROW(fly_sort_index(&out_, &indices_, in.get(), dim, isAscending));
    out     = array(out_);
    indices = array(indices_);
}

void sort(array &out_keys, array &out_values, const array &keys,
          const array &values, const unsigned dim, const bool isAscending) {
    fly_array okeys, ovalues;
    FLY_THROW(fly_sort_by_key(&okeys, &ovalues, keys.get(), values.get(), dim,
                            isAscending));
    out_keys   = array(okeys);
    out_values = array(ovalues);
}
}  // namespace fly
