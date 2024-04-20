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
#include <common/err_common.hpp>
#include <features.hpp>
#include <handle.hpp>
#include <orb.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/features.h>
#include <fly/vision.h>

using fly::dim4;

using detail::Array;
using detail::createEmptyArray;
using detail::uint;

template<typename T, typename convAccT>
static void orb(fly_features& feat_, fly_array& descriptor, const fly_array& in,
                const float fast_thr, const unsigned max_feat,
                const float scl_fctr, const unsigned levels,
                const bool blur_img) {
    Array<float> x     = createEmptyArray<float>(dim4());
    Array<float> y     = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());
    Array<float> ori   = createEmptyArray<float>(dim4());
    Array<float> size  = createEmptyArray<float>(dim4());
    Array<uint> desc   = createEmptyArray<uint>(dim4());

    fly_features_t feat;

    feat.n = orb<T, convAccT>(x, y, score, ori, size, desc, getArray<T>(in),
                              fast_thr, max_feat, scl_fctr, levels, blur_img);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(ori);
    feat.size        = getHandle(size);

    feat_      = getFeaturesHandle(feat);
    descriptor = getHandle<unsigned>(desc);
}

fly_err fly_orb(fly_features* feat, fly_array* desc, const fly_array in,
              const float fast_thr, const unsigned max_feat,
              const float scl_fctr, const unsigned levels,
              const bool blur_img) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly::dim4 dims         = info.dims();

        ARG_ASSERT(
            2, (dims[0] >= 7 && dims[1] >= 7 && dims[2] == 1 && dims[3] == 1));
        ARG_ASSERT(3, fast_thr > 0.0f);
        ARG_ASSERT(4, max_feat > 0);
        ARG_ASSERT(5, scl_fctr > 1.0f);
        ARG_ASSERT(6, levels > 0);

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims == 2));

        fly_array tmp_desc;
        fly_dtype type = info.getType();
        switch (type) {
            case f32:
                orb<float, float>(*feat, tmp_desc, in, fast_thr, max_feat,
                                  scl_fctr, levels, blur_img);
                break;
            case f64:
                orb<double, double>(*feat, tmp_desc, in, fast_thr, max_feat,
                                    scl_fctr, levels, blur_img);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*desc, tmp_desc);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
