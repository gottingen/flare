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

array dilate(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_dilate(&out, in.get(), mask.get()));
    return array(out);
}

array dilate3(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_dilate3(&out, in.get(), mask.get()));
    return array(out);
}

array erode(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_erode(&out, in.get(), mask.get()));
    return array(out);
}

array erode3(const array& in, const array& mask) {
    fly_array out = 0;
    FLY_THROW(fly_erode3(&out, in.get(), mask.get()));
    return array(out);
}

}  // namespace fly
