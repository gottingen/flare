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
#include <backend.hpp>
#include <common/err_common.hpp>
#include <fast.hpp>
#include <features.hpp>
#include <handle.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/features.h>
#include <fly/vision.h>

using fly::dim4;
using detail::Array;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T>
static fly_features fast(fly_array const &in, const float thr,
                        const unsigned arc_length, const bool non_max,
                        const float feature_ratio, const unsigned edge) {
    Array<float> x     = createEmptyArray<float>(dim4());
    Array<float> y     = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());

    fly_features_t feat;
    feat.n = fast<T>(x, y, score, getArray<T>(in), thr, arc_length, non_max,
                     feature_ratio, edge);

    Array<float> orientation = createValueArray<float>(feat.n, 0.0);
    Array<float> size        = createValueArray<float>(feat.n, 1.0);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(orientation);
    feat.size        = getHandle(size);

    return getFeaturesHandle(feat);
}

fly_err fly_fast(fly_features *out, const fly_array in, const float thr,
               const unsigned arc_length, const bool non_max,
               const float feature_ratio, const unsigned edge) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly::dim4 dims         = info.dims();

        ARG_ASSERT(2, (dims[0] >= (dim_t)(2 * edge + 1) ||
                       dims[1] >= (dim_t)(2 * edge + 1)));
        ARG_ASSERT(3, thr > 0.0f);
        ARG_ASSERT(4, (arc_length >= 9 && arc_length <= 16));
        ARG_ASSERT(6, (feature_ratio > 0.0f && feature_ratio <= 1.0f));

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims == 2));

        fly_dtype type = info.getType();
        switch (type) {
            case f32:
                *out = fast<float>(in, thr, arc_length, non_max, feature_ratio,
                                   edge);
                break;
            case f64:
                *out = fast<double>(in, thr, arc_length, non_max, feature_ratio,
                                    edge);
                break;
            case b8:
                *out = fast<char>(in, thr, arc_length, non_max, feature_ratio,
                                  edge);
                break;
            case s32:
                *out = fast<int>(in, thr, arc_length, non_max, feature_ratio,
                                 edge);
                break;
            case u32:
                *out = fast<uint>(in, thr, arc_length, non_max, feature_ratio,
                                  edge);
                break;
            case s16:
                *out = fast<short>(in, thr, arc_length, non_max, feature_ratio,
                                   edge);
                break;
            case u16:
                *out = fast<ushort>(in, thr, arc_length, non_max, feature_ratio,
                                    edge);
                break;
            case u8:
                *out = fast<uchar>(in, thr, arc_length, non_max, feature_ratio,
                                   edge);
                break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
