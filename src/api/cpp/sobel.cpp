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
#include <fly/image.h>
#include "error.hpp"

namespace fly {

void sobel(array &dx, array &dy, const array &img, const unsigned ker_size) {
    fly_array fly_dx = 0;
    fly_array fly_dy = 0;
    FLY_THROW(fly_sobel_operator(&fly_dx, &fly_dy, img.get(), ker_size));
    dx = array(fly_dx);
    dy = array(fly_dy);
}

array sobel(const array &img, const unsigned ker_size, const bool isFast) {
    array dx;
    array dy;
    sobel(dx, dy, img, ker_size);
    if (isFast) {
        return abs(dx) + abs(dy);
    } else {
        return sqrt(dx * dx + dy * dy);
    }
}

}  // namespace fly
