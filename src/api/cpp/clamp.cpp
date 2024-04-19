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

#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/gfor.h>
#include "error.hpp"

namespace fly {
array clamp(const array &in, const array &lo, const array &hi) {
    fly_array out;
    FLY_THROW(fly_clamp(&out, in.get(), lo.get(), hi.get(), gforGet()));
    return array(out);
}

array clamp(const array &in, const array &lo, const double hi) {
    return clamp(in, lo, constant(hi, lo.dims(), lo.type()));
}

array clamp(const array &in, const double lo, const array &hi) {
    return clamp(in, constant(lo, hi.dims(), hi.type()), hi);
}

array clamp(const array &in, const double lo, const double hi) {
    return clamp(in, constant(lo, in.dims(), in.type()),
                 constant(hi, in.dims(), in.type()));
}
}  // namespace fly
