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
#include <fly/features.h>
#include "symbol_manager.hpp"

fly_err fly_create_features(fly_features *feat, dim_t num) {
    CALL(fly_create_features, feat, num);
}

fly_err fly_retain_features(fly_features *out, const fly_features feat) {
    CALL(fly_retain_features, out, feat);
}

fly_err fly_get_features_num(dim_t *num, const fly_features feat) {
    CALL(fly_get_features_num, num, feat);
}

#define FEAT_HAPI_DEF(fly_func)                              \
    fly_err fly_func(fly_array *out, const fly_features feat) { \
        CALL(fly_func, out, feat);                           \
    }

FEAT_HAPI_DEF(fly_get_features_xpos)
FEAT_HAPI_DEF(fly_get_features_ypos)
FEAT_HAPI_DEF(fly_get_features_score)
FEAT_HAPI_DEF(fly_get_features_orientation)
FEAT_HAPI_DEF(fly_get_features_size)

fly_err fly_release_features(fly_features feat) {
    CALL(fly_release_features, feat);
}
