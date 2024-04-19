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

#include <Array.hpp>
#include <fly/features.h>

using fly::features;

namespace flare {
namespace cuda {

template<typename T, typename convAccT>
unsigned harris(Array<float> &x_out, Array<float> &y_out,
                Array<float> &score_out, const Array<T> &in,
                const unsigned max_corners, const float min_response,
                const float sigma, const unsigned filter_len,
                const float k_thr);

}  // namespace cuda
}  // namespace flare
