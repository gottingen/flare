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

#include <utility.hpp>

#include <err_cuda.hpp>

namespace flare {
namespace cuda {

int interpOrder(const fly_interp_type p) noexcept {
    int order = 1;
    switch (p) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER: order = 1; break;
        case FLY_INTERP_LINEAR:
        case FLY_INTERP_BILINEAR:
        case FLY_INTERP_LINEAR_COSINE:
        case FLY_INTERP_BILINEAR_COSINE: order = 2; break;
        case FLY_INTERP_CUBIC:
        case FLY_INTERP_BICUBIC:
        case FLY_INTERP_CUBIC_SPLINE:
        case FLY_INTERP_BICUBIC_SPLINE: order = 3; break;
    }
    return order;
}

}  // namespace cuda
}  // namespace flare
