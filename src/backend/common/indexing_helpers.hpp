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

#pragma once

#include <Array.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>

#include <array>

namespace flare {
namespace common {

// will generate indexes to flip input array
// of size original dims according to axes specified in flip
template<typename T>
static detail::Array<T> flip(const detail::Array<T>& in,
                             const std::array<bool, FLY_MAX_DIMS> flip) {
    std::vector<fly_seq> index(4, fly_span);
    const fly::dim4& dims = in.dims();

    for (int i = 0; i < FLY_MAX_DIMS; ++i) {
        if (flip[i]) {
            index[i] = {static_cast<double>(dims[i] - 1), 0.0, -1.0};
        }
    }
    return createSubArray(in, index);
}

}  // namespace common
}  // namespace flare
