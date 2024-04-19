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
#include <fly/compatible.h>
#include "error.hpp"

namespace fly {

array setunique(const array &in, const bool is_sorted) {
    return setUnique(in, is_sorted);
}

array setUnique(const array &in, const bool is_sorted) {
    fly_array out = 0;
    FLY_THROW(fly_set_unique(&out, in.get(), is_sorted));
    return array(out);
}

array setunion(const array &first, const array &second, const bool is_unique) {
    return setUnion(first, second, is_unique);
}

array setUnion(const array &first, const array &second, const bool is_unique) {
    fly_array out = 0;
    FLY_THROW(fly_set_union(&out, first.get(), second.get(), is_unique));
    return array(out);
}

array setintersect(const array &first, const array &second,
                   const bool is_unique) {
    return setIntersect(first, second, is_unique);
}

array setIntersect(const array &first, const array &second,
                   const bool is_unique) {
    fly_array out = 0;
    FLY_THROW(fly_set_intersect(&out, first.get(), second.get(), is_unique));
    return array(out);
}

}  // namespace fly
