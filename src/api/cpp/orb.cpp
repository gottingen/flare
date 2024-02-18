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

void orb(features& feat, array& desc, const array& in, const float fast_thr,
         const unsigned max_feat, const float scl_fctr, const unsigned levels,
         const bool blur_img) {
    fly_features temp_feat;
    fly_array temp_desc = 0;
    FLY_THROW(fly_orb(&temp_feat, &temp_desc, in.get(), fast_thr, max_feat,
                    scl_fctr, levels, blur_img));

    dim_t num = 0;
    FLY_THROW(fly_get_features_num(&num, temp_feat));
    feat = features(temp_feat);
    desc = array(temp_desc);
}

}  // namespace fly
