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
#include "symbol_manager.hpp"

fly_err fly_fast(fly_features *out, const fly_array in, const float thr,
               const unsigned arc_length, const bool non_max,
               const float feature_ratio, const unsigned edge) {
    CHECK_ARRAYS(in);
    CALL(fly_fast, out, in, thr, arc_length, non_max, feature_ratio, edge);
}

fly_err fly_harris(fly_features *out, const fly_array in,
                 const unsigned max_corners, const float min_response,
                 const float sigma, const unsigned block_size,
                 const float k_thr) {
    CHECK_ARRAYS(in);
    CALL(fly_harris, out, in, max_corners, min_response, sigma, block_size,
         k_thr);
}

fly_err fly_orb(fly_features *feat, fly_array *desc, const fly_array in,
              const float fast_thr, const unsigned max_feat,
              const float scl_fctr, const unsigned levels,
              const bool blur_img) {
    CHECK_ARRAYS(in);
    CALL(fly_orb, feat, desc, in, fast_thr, max_feat, scl_fctr, levels,
         blur_img);
}

fly_err fly_sift(fly_features *feat, fly_array *desc, const fly_array in,
               const unsigned n_layers, const float contrast_thr,
               const float edge_thr, const float init_sigma,
               const bool double_input, const float intensity_scale,
               const float feature_ratio) {
    CHECK_ARRAYS(in);
    CALL(fly_sift, feat, desc, in, n_layers, contrast_thr, edge_thr, init_sigma,
         double_input, intensity_scale, feature_ratio);
}

fly_err fly_gloh(fly_features *feat, fly_array *desc, const fly_array in,
               const unsigned n_layers, const float contrast_thr,
               const float edge_thr, const float init_sigma,
               const bool double_input, const float intensity_scale,
               const float feature_ratio) {
    CHECK_ARRAYS(in);
    CALL(fly_gloh, feat, desc, in, n_layers, contrast_thr, edge_thr, init_sigma,
         double_input, intensity_scale, feature_ratio);
}

fly_err fly_hamming_matcher(fly_array *idx, fly_array *dist, const fly_array query,
                          const fly_array train, const dim_t dist_dim,
                          const unsigned n_dist) {
    CHECK_ARRAYS(query, train);
    CALL(fly_hamming_matcher, idx, dist, query, train, dist_dim, n_dist);
}

fly_err fly_nearest_neighbour(fly_array *idx, fly_array *dist, const fly_array query,
                            const fly_array train, const dim_t dist_dim,
                            const unsigned n_dist,
                            const fly_match_type dist_type) {
    CHECK_ARRAYS(query, train);
    CALL(fly_nearest_neighbour, idx, dist, query, train, dist_dim, n_dist,
         dist_type);
}

fly_err fly_match_template(fly_array *out, const fly_array search_img,
                         const fly_array template_img,
                         const fly_match_type m_type) {
    CHECK_ARRAYS(search_img, template_img);
    CALL(fly_match_template, out, search_img, template_img, m_type);
}

fly_err fly_susan(fly_features *out, const fly_array in, const unsigned radius,
                const float diff_thr, const float geom_thr,
                const float feature_ratio, const unsigned edge) {
    CHECK_ARRAYS(in);
    CALL(fly_susan, out, in, radius, diff_thr, geom_thr, feature_ratio, edge);
}

fly_err fly_dog(fly_array *out, const fly_array in, const int radius1,
              const int radius2) {
    CHECK_ARRAYS(in);
    CALL(fly_dog, out, in, radius1, radius2);
}

fly_err fly_homography(fly_array *H, int *inliers, const fly_array x_src,
                     const fly_array y_src, const fly_array x_dst,
                     const fly_array y_dst, const fly_homography_type htype,
                     const float inlier_thr, const unsigned iterations,
                     const fly_dtype type) {
    CHECK_ARRAYS(x_src, y_src, x_dst, y_dst);
    CALL(fly_homography, H, inliers, x_src, y_src, x_dst, y_dst, htype,
         inlier_thr, iterations, type);
}
