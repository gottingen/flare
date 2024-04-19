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
