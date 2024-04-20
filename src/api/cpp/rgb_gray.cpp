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

array rgb2gray(const array& in, const float rPercent, const float gPercent,
               const float bPercent) {
    fly_array temp = 0;
    FLY_THROW(fly_rgb2gray(&temp, in.get(), rPercent, gPercent, bPercent));
    return array(temp);
}

array gray2rgb(const array& in, const float rFactor, const float gFactor,
               const float bFactor) {
    fly_array temp = 0;
    FLY_THROW(fly_gray2rgb(&temp, in.get(), rFactor, gFactor, bFactor));
    return array(temp);
}

}  // namespace fly
