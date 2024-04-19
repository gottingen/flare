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

#include <homography.hpp>

#include <arith.hpp>
#include <err_oneapi.hpp>
#include <fly/dim4.hpp>

#include <algorithm>
#include <limits>

using fly::dim4;
using std::numeric_limits;

namespace flare {
namespace oneapi {

template<typename T>
int homography(Array<T> &bestH, const Array<float> &x_src,
               const Array<float> &y_src, const Array<float> &x_dst,
               const Array<float> &y_dst, const Array<float> &initial,
               const fly_homography_type htype, const float inlier_thr,
               const unsigned iterations) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

#define INSTANTIATE(T)                                                     \
    template int homography(                                               \
        Array<T> &H, const Array<float> &x_src, const Array<float> &y_src, \
        const Array<float> &x_dst, const Array<float> &y_dst,              \
        const Array<float> &initial, const fly_homography_type htype,       \
        const float inlier_thr, const unsigned iterations);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace oneapi
}  // namespace flare
