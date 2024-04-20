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
#include <fly/defines.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {

array moments(const array& in, const fly_moment_type moment) {
    fly_array out = 0;
    FLY_THROW(fly_moments(&out, in.get(), moment));
    return array(out);
}

void moments(double* out, const array& in, const fly_moment_type moment) {
    FLY_THROW(fly_moments_all(out, in.get(), moment));
}

}  // namespace fly
