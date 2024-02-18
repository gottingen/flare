/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/features.h>
#include "error.hpp"

#include <utility>

namespace fly {

features::features() : feat{} { FLY_THROW(fly_create_features(&feat, 0)); }

features::features(const size_t n) : feat{} {
    FLY_THROW(fly_create_features(&feat, (int)n));
}

features::features(fly_features f) : feat(f) {}

features::features(const features& other) {
    if (this != &other) { FLY_THROW(fly_retain_features(&feat, other.get())); }
}

features& features::operator=(const features& other) {
    if (this != &other) {
        FLY_THROW(fly_release_features(feat));
        FLY_THROW(fly_retain_features(&feat, other.get()));
    }
    return *this;
}

features::features(features&& other)
    : feat(std::exchange(other.feat, nullptr)) {}

features& features::operator=(features&& other) {
    std::swap(feat, other.feat);
    return *this;
}

features::~features() {
    // THOU SHALL NOT THROW IN DESTRUCTORS
    if (feat) { fly_release_features(feat); }
}

size_t features::getNumFeatures() const {
    dim_t n = 0;
    FLY_THROW(fly_get_features_num(&n, feat));
    return n;
}

array features::getX() const {
    fly_array x = 0;
    FLY_THROW(fly_get_features_xpos(&x, feat));
    fly_array tmp = 0;
    FLY_THROW(fly_retain_array(&tmp, x));
    return array(tmp);
}

array features::getY() const {
    fly_array y = 0;
    FLY_THROW(fly_get_features_ypos(&y, feat));
    fly_array tmp = 0;
    FLY_THROW(fly_retain_array(&tmp, y));
    return array(tmp);
}

array features::getScore() const {
    fly_array s = 0;
    FLY_THROW(fly_get_features_score(&s, feat));
    fly_array tmp = 0;
    FLY_THROW(fly_retain_array(&tmp, s));
    return array(tmp);
}

array features::getOrientation() const {
    fly_array ori = 0;
    FLY_THROW(fly_get_features_orientation(&ori, feat));
    fly_array tmp = 0;
    FLY_THROW(fly_retain_array(&tmp, ori));
    return array(tmp);
}

array features::getSize() const {
    fly_array s = 0;
    FLY_THROW(fly_get_features_size(&s, feat));
    fly_array tmp = 0;
    FLY_THROW(fly_retain_array(&tmp, s));
    return array(tmp);
}

fly_features features::get() const { return feat; }

};  // namespace fly
