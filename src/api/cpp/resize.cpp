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
#include <fly/image.h>
#include "error.hpp"

namespace fly {

array resize(const array &in, const dim_t odim0, const dim_t odim1,
             const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_resize(&out, in.get(), odim0, odim1, method));
    return array(out);
}

array resize(const float scale0, const float scale1, const array &in,
             const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_resize(&out, in.get(), in.dims(0) * scale0, in.dims(1) * scale1,
                       method));
    return array(out);
}

array resize(const float scale, const array &in, const interpType method) {
    fly_array out = 0;
    FLY_THROW(fly_resize(&out, in.get(), in.dims(0) * scale, in.dims(1) * scale,
                       method));
    return array(out);
}

}  // namespace fly
