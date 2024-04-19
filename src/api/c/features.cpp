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

#include <features.hpp>
#include <handle.hpp>
#include <fly/array.h>
#include <fly/features.h>

fly_err fly_release_features(fly_features featHandle) {
    try {
        fly_features_t feat = *static_cast<fly_features_t *>(featHandle);
        if (feat.n > 0) {
            if (feat.x != 0) { FLY_CHECK(fly_release_array(feat.x)); }
            if (feat.y != 0) { FLY_CHECK(fly_release_array(feat.y)); }
            if (feat.score != 0) { FLY_CHECK(fly_release_array(feat.score)); }
            if (feat.orientation != 0) {
                FLY_CHECK(fly_release_array(feat.orientation));
            }
            if (feat.size != 0) { FLY_CHECK(fly_release_array(feat.size)); }
            feat.n = 0;
        }
        delete static_cast<fly_features_t *>(featHandle);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_features getFeaturesHandle(const fly_features_t feat) {
    auto *featHandle = new fly_features_t;
    *featHandle      = feat;
    return static_cast<fly_features>(featHandle);
}

fly_err fly_create_features(fly_features *featHandle, dim_t num) {
    try {
        fly_features_t feat;
        feat.n = num;

        if (num > 0) {
            dim_t out_dims[4] = {dim_t(num), 1, 1, 1};
            FLY_CHECK(fly_create_handle(&feat.x, 4, out_dims, f32));
            FLY_CHECK(fly_create_handle(&feat.y, 4, out_dims, f32));
            FLY_CHECK(fly_create_handle(&feat.score, 4, out_dims, f32));
            FLY_CHECK(fly_create_handle(&feat.orientation, 4, out_dims, f32));
            FLY_CHECK(fly_create_handle(&feat.size, 4, out_dims, f32));
        }

        *featHandle = getFeaturesHandle(feat);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_features_t getFeatures(const fly_features featHandle) {
    return *static_cast<fly_features_t *>(featHandle);
}

fly_err fly_retain_features(fly_features *outHandle,
                          const fly_features featHandle) {
    try {
        fly_features_t feat = getFeatures(featHandle);
        fly_features_t out;

        out.n = feat.n;
        FLY_CHECK(fly_retain_array(&out.x, feat.x));
        FLY_CHECK(fly_retain_array(&out.y, feat.y));
        FLY_CHECK(fly_retain_array(&out.score, feat.score));
        FLY_CHECK(fly_retain_array(&out.orientation, feat.orientation));
        FLY_CHECK(fly_retain_array(&out.size, feat.size));

        *outHandle = getFeaturesHandle(out);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_features_num(dim_t *num, const fly_features featHandle) {
    try {
        fly_features_t feat = getFeatures(featHandle);
        *num               = feat.n;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_features_xpos(fly_array *out, const fly_features featHandle) {
    try {
        fly_features_t feat = getFeatures(featHandle);
        *out               = feat.x;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_features_ypos(fly_array *out, const fly_features featHandle) {
    try {
        fly_features_t feat = getFeatures(featHandle);
        *out               = feat.y;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_features_score(fly_array *out, const fly_features featHandle) {
    try {
        fly_features_t feat = getFeatures(featHandle);
        *out               = feat.score;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_features_orientation(fly_array *out,
                                   const fly_features featHandle) {
    try {
        fly_features_t feat = getFeatures(featHandle);
        *out               = feat.orientation;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_features_size(fly_array *out, const fly_features featHandle) {
    try {
        fly_features_t feat = getFeatures(featHandle);
        *out               = feat.size;
    }
    CATCHALL;
    return FLY_SUCCESS;
}
