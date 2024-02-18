/*******************************************************
 * Copyright (c) 2015, Flare
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

void sift(features& feat, array& desc, const array& in, const unsigned n_layers,
          const float contrast_thr, const float edge_thr,
          const float init_sigma, const bool double_input,
          const float img_scale, const float feature_ratio) {
    fly_features temp_feat;
    fly_array temp_desc = 0;
    FLY_THROW(fly_sift(&temp_feat, &temp_desc, in.get(), n_layers, contrast_thr,
                     edge_thr, init_sigma, double_input, img_scale,
                     feature_ratio));

    dim_t num = 0;
    FLY_THROW(fly_get_features_num(&num, temp_feat));
    feat = features(temp_feat);
    desc = array(temp_desc);
}

void gloh(features& feat, array& desc, const array& in, const unsigned n_layers,
          const float contrast_thr, const float edge_thr,
          const float init_sigma, const bool double_input,
          const float img_scale, const float feature_ratio) {
    fly_features temp_feat;
    fly_array temp_desc = 0;
    FLY_THROW(fly_gloh(&temp_feat, &temp_desc, in.get(), n_layers, contrast_thr,
                     edge_thr, init_sigma, double_input, img_scale,
                     feature_ratio));

    dim_t num = 0;
    FLY_THROW(fly_get_features_num(&num, temp_feat));
    feat = features(temp_feat);
    desc = array(temp_desc);
}

}  // namespace fly
