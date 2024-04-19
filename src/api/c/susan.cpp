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
#include <susan.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/features.h>
#include <fly/vision.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T>
static fly_features susan(fly_array const& in, const unsigned radius,
                         const float diff_thr, const float geom_thr,
                         const float feature_ratio, const unsigned edge) {
    Array<float> x     = createEmptyArray<float>(dim4());
    Array<float> y     = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());

    fly_features_t feat;
    feat.n = susan<T>(x, y, score, getArray<T>(in), radius, diff_thr, geom_thr,
                      feature_ratio, edge);

    feat.x     = getHandle(x);
    feat.y     = getHandle(y);
    feat.score = getHandle(score);
    feat.orientation =
        getHandle(feat.n > 0 ? createValueArray<float>(feat.n, 0.0)
                             : createEmptyArray<float>(dim4()));
    feat.size = getHandle(feat.n > 0 ? createValueArray<float>(feat.n, 1.0)
                                     : createEmptyArray<float>(dim4()));

    return getFeaturesHandle(feat);
}

fly_err fly_susan(fly_features* out, const fly_array in, const unsigned radius,
                const float diff_thr, const float geom_thr,
                const float feature_ratio, const unsigned edge) {
    try {
        const ArrayInfo& info = getInfo(in);
        fly::dim4 dims         = info.dims();

        ARG_ASSERT(1, dims.ndims() == 2);
        ARG_ASSERT(2, radius < 10);
        ARG_ASSERT(2, radius <= edge);
        ARG_ASSERT(3, diff_thr > 0.0f);
        ARG_ASSERT(4, geom_thr > 0.0f);
        ARG_ASSERT(5, (feature_ratio > 0.0f && feature_ratio <= 1.0f));
        ARG_ASSERT(6, (dims[0] >= (dim_t)(2 * edge + 1) ||
                       dims[1] >= (dim_t)(2 * edge + 1)));

        fly_dtype type = info.getType();
        switch (type) {
            case f32:
                *out = susan<float>(in, radius, diff_thr, geom_thr,
                                    feature_ratio, edge);
                break;
            case f64:
                *out = susan<double>(in, radius, diff_thr, geom_thr,
                                     feature_ratio, edge);
                break;
            case b8:
                *out = susan<char>(in, radius, diff_thr, geom_thr,
                                   feature_ratio, edge);
                break;
            case s32:
                *out = susan<int>(in, radius, diff_thr, geom_thr, feature_ratio,
                                  edge);
                break;
            case u32:
                *out = susan<uint>(in, radius, diff_thr, geom_thr,
                                   feature_ratio, edge);
                break;
            case s16:
                *out = susan<short>(in, radius, diff_thr, geom_thr,
                                    feature_ratio, edge);
                break;
            case u16:
                *out = susan<ushort>(in, radius, diff_thr, geom_thr,
                                     feature_ratio, edge);
                break;
            case u8:
                *out = susan<uchar>(in, radius, diff_thr, geom_thr,
                                    feature_ratio, edge);
                break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
