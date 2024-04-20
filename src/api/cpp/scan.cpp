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
array accum(const array& in, const int dim) {
    fly_array out = 0;
    FLY_THROW(fly_accum(&out, in.get(), dim));
    return array(out);
}

array scan(const array& in, const int dim, binaryOp op, bool inclusive_scan) {
    fly_array out = 0;
    FLY_THROW(fly_scan(&out, in.get(), dim, op, inclusive_scan));
    return array(out);
}

array scanByKey(const array& key, const array& in, const int dim, binaryOp op,
                bool inclusive_scan) {
    fly_array out = 0;
    FLY_THROW(
        fly_scan_by_key(&out, key.get(), in.get(), dim, op, inclusive_scan));
    return array(out);
}
}  // namespace fly
