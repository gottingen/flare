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

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <homography.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/random.h>
#include <fly/vision.h>

#include <utility>

using fly::dim4;
using detail::Array;
using detail::createEmptyArray;
using std::swap;

template<typename T>
static inline void homography(fly_array& H, int& inliers, const fly_array x_src,
                              const fly_array y_src, const fly_array x_dst,
                              const fly_array y_dst,
                              const fly_homography_type htype,
                              const float inlier_thr,
                              const unsigned iterations) {
    Array<T> bestH = createEmptyArray<T>(fly::dim4(3, 3));
    fly_array initial;
    unsigned d    = (iterations + 256 - 1) / 256;
    dim_t rdims[] = {4, d * 256};
    FLY_CHECK(fly_randu(&initial, 2, rdims, f32));
    inliers =
        homography<T>(bestH, getArray<float>(x_src), getArray<float>(y_src),
                      getArray<float>(x_dst), getArray<float>(y_dst),
                      getArray<float>(initial), htype, inlier_thr, iterations);
    FLY_CHECK(fly_release_array(initial));

    H = getHandle<T>(bestH);
}

fly_err fly_homography(fly_array* H, int* inliers, const fly_array x_src,
                     const fly_array y_src, const fly_array x_dst,
                     const fly_array y_dst, const fly_homography_type htype,
                     const float inlier_thr, const unsigned iterations,
                     const fly_dtype otype) {
    try {
        const ArrayInfo& xsinfo = getInfo(x_src);
        const ArrayInfo& ysinfo = getInfo(y_src);
        const ArrayInfo& xdinfo = getInfo(x_dst);
        const ArrayInfo& ydinfo = getInfo(y_dst);

        fly::dim4 xsdims = xsinfo.dims();
        fly::dim4 ysdims = ysinfo.dims();
        fly::dim4 xddims = xdinfo.dims();
        fly::dim4 yddims = ydinfo.dims();

        fly_dtype xstype = xsinfo.getType();
        fly_dtype ystype = ysinfo.getType();
        fly_dtype xdtype = xdinfo.getType();
        fly_dtype ydtype = ydinfo.getType();

        if (xstype != f32) { TYPE_ERROR(1, xstype); }
        if (ystype != f32) { TYPE_ERROR(2, ystype); }
        if (xdtype != f32) { TYPE_ERROR(3, xdtype); }
        if (ydtype != f32) { TYPE_ERROR(4, ydtype); }

        ARG_ASSERT(1, (xsdims[0] > 0));
        ARG_ASSERT(2, (ysdims[0] == xsdims[0]));
        ARG_ASSERT(3, (xddims[0] > 0));
        ARG_ASSERT(4, (yddims[0] == xddims[0]));

        ARG_ASSERT(5, (inlier_thr >= 0.1f));
        ARG_ASSERT(6, (iterations > 0));
        ARG_ASSERT(
            7, (htype == FLY_HOMOGRAPHY_RANSAC || htype == FLY_HOMOGRAPHY_LMEDS));

        fly_array outH;
        int outInl;

        switch (otype) {
            case f32:
                homography<float>(outH, outInl, x_src, y_src, x_dst, y_dst,
                                  htype, inlier_thr, iterations);
                break;
            case f64:
                homography<double>(outH, outInl, x_src, y_src, x_dst, y_dst,
                                   htype, inlier_thr, iterations);
                break;
            default: TYPE_ERROR(1, otype);
        }
        swap(*H, outH);
        swap(*inliers, outInl);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
