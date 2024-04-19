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
#include <err_oneapi.hpp>
// #include <kernel/orb.hpp>
#include <math.hpp>
#include <fly/dim4.hpp>
#include <fly/features.h>

using fly::dim4;
using fly::features;

namespace flare {
namespace oneapi {

template<typename T, typename convAccT>
unsigned orb(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
             Array<float> &ori_out, Array<float> &size_out,
             Array<uint> &desc_out, const Array<T> &image, const float fast_thr,
             const unsigned max_feat, const float scl_fctr,
             const unsigned levels, const bool blur_img) {
    ONEAPI_NOT_SUPPORTED("orb Not supported");
    return 0;

    // unsigned nfeat;

    // Param x;
    // Param y;
    // Param score;
    // Param ori;
    // Param size;
    // Param desc;

    // kernel::orb<T, convAccT>(&nfeat, x, y, score, ori, size, desc, image,
    //                          fast_thr, max_feat, scl_fctr, levels, blur_img);

    // if (nfeat > 0) {
    //     const dim4 out_dims(nfeat);
    //     const dim4 desc_dims(8, nfeat);

    //     x_out     = createParamArray<float>(x, true);
    //     y_out     = createParamArray<float>(y, true);
    //     score_out = createParamArray<float>(score, true);
    //     ori_out   = createParamArray<float>(ori, true);
    //     size_out  = createParamArray<float>(size, true);
    //     desc_out  = createParamArray<unsigned>(desc, true);
    // }

    // return nfeat;
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned orb<T, convAccT>(                                       \
        Array<float> & x, Array<float> & y, Array<float> & score,             \
        Array<float> & ori, Array<float> & size, Array<uint> & desc,          \
        const Array<T> &image, const float fast_thr, const unsigned max_feat, \
        const float scl_fctr, const unsigned levels, const bool blur_img);

INSTANTIATE(float, float)
INSTANTIATE(double, double)

}  // namespace oneapi
}  // namespace flare
