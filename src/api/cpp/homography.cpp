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
#include <fly/vision.h>
#include "error.hpp"

namespace fly {

void homography(array &H, int &inliers, const array &x_src, const array &y_src,
                const array &x_dst, const array &y_dst,
                const fly_homography_type htype, const float inlier_thr,
                const unsigned iterations, const fly::dtype otype) {
    fly_array outH;
    FLY_THROW(fly_homography(&outH, &inliers, x_src.get(), y_src.get(),
                           x_dst.get(), y_dst.get(), htype, inlier_thr,
                           iterations, otype));

    H = array(outH);
}

}  // namespace fly
