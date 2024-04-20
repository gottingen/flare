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
#include <harris.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/features.h>
#include <fly/vision.h>

#include <cmath>

using fly::dim4;
using detail::Array;
using detail::createEmptyArray;
using detail::createValueArray;
using std::floor;

template<typename T, typename convAccT>
static fly_features harris(fly_array const &in, const unsigned max_corners,
                          const float min_response, const float sigma,
                          const unsigned filter_len, const float k_thr) {
    Array<float> x     = createEmptyArray<float>(dim4());
    Array<float> y     = createEmptyArray<float>(dim4());
    Array<float> score = createEmptyArray<float>(dim4());

    fly_features_t feat;
    feat.n = harris<T, convAccT>(x, y, score, getArray<T>(in), max_corners,
                                 min_response, sigma, filter_len, k_thr);

    Array<float> orientation = createValueArray<float>(feat.n, 0.0);
    Array<float> size        = createValueArray<float>(feat.n, 1.0);

    feat.x           = getHandle(x);
    feat.y           = getHandle(y);
    feat.score       = getHandle(score);
    feat.orientation = getHandle(orientation);
    feat.size        = getHandle(size);

    return getFeaturesHandle(feat);
}

fly_err fly_harris(fly_features *out, const fly_array in,
                 const unsigned max_corners, const float min_response,
                 const float sigma, const unsigned block_size,
                 const float k_thr) {
    try {
        const ArrayInfo &info = getInfo(in);
        dim4 dims             = info.dims();
        dim_t in_ndims        = dims.ndims();

        unsigned filter_len = (block_size == 0)
                                  ? static_cast<unsigned>(floor(6.f * sigma))
                                  : block_size;
        if (block_size == 0 && filter_len % 2 == 0) { filter_len--; }

        const unsigned edge =
            (block_size > 0) ? block_size / 2 : filter_len / 2;

        DIM_ASSERT(1, (in_ndims == 2));
        ARG_ASSERT(1, (dims[0] >= (dim_t)(2 * edge + 1) ||
                       dims[1] >= (dim_t)(2 * edge + 1)));
        ARG_ASSERT(3, (max_corners > 0) || (min_response > 0.0f));
        ARG_ASSERT(7, (k_thr >= 0.01f));
        ARG_ASSERT(4, (block_size > 2) || (sigma >= 0.5f && sigma <= 5.f));
        ARG_ASSERT(5, (block_size <= 32));

        fly_dtype type = info.getType();
        switch (type) {
            case f64:
                *out = harris<double, double>(in, max_corners, min_response,
                                              sigma, filter_len, k_thr);
                break;
            case f32:
                *out = harris<float, float>(in, max_corners, min_response,
                                            sigma, filter_len, k_thr);
                break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
