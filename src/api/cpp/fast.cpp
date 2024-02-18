/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/vision.h>
#include "error.hpp"

namespace fly {

features fast(const array& in, const float thr, const unsigned arc_length,
              const bool non_max, const float feature_ratio,
              const unsigned edge) {
    fly_features temp;
    FLY_THROW(fly_fast(&temp, in.get(), thr, arc_length, non_max, feature_ratio,
                     edge));
    return features(temp);
}

}  // namespace fly
