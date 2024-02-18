/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
