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

#include <sift.hpp>

#include <kernel/sift.hpp>

using fly::dim4;
using fly::features;

namespace flare {
namespace cuda {

template<typename T, typename convAccT>
unsigned sift(Array<float>& x, Array<float>& y, Array<float>& score,
              Array<float>& ori, Array<float>& size, Array<float>& desc,
              const Array<T>& in, const unsigned n_layers,
              const float contrast_thr, const float edge_thr,
              const float init_sigma, const bool double_input,
              const float img_scale, const float feature_ratio,
              const bool compute_GLOH) {
    unsigned nfeat_out;
    unsigned desc_len;
    float* x_out;
    float* y_out;
    float* score_out;
    float* orientation_out;
    float* size_out;
    float* desc_out;

    kernel::sift<T, convAccT>(
        &nfeat_out, &desc_len, &x_out, &y_out, &score_out, &orientation_out,
        &size_out, &desc_out, in, n_layers, contrast_thr, edge_thr, init_sigma,
        double_input, img_scale, feature_ratio, compute_GLOH);

    if (nfeat_out > 0) {
        if (x_out == NULL || y_out == NULL || score_out == NULL ||
            orientation_out == NULL || size_out == NULL || desc_out == NULL) {
            FLY_ERROR("sift: feature array is null.", FLY_ERR_SIZE);
        }

        const dim4 feat_dims(nfeat_out);
        const dim4 desc_dims(desc_len, nfeat_out);

        x     = createDeviceDataArray<float>(feat_dims, x_out);
        y     = createDeviceDataArray<float>(feat_dims, y_out);
        score = createDeviceDataArray<float>(feat_dims, score_out);
        ori   = createDeviceDataArray<float>(feat_dims, orientation_out);
        size  = createDeviceDataArray<float>(feat_dims, size_out);
        desc  = createDeviceDataArray<float>(desc_dims, desc_out);
    }

    return nfeat_out;
}

#define INSTANTIATE(T, convAccT)                                               \
    template unsigned sift<T, convAccT>(                                       \
        Array<float> & x, Array<float> & y, Array<float> & score,              \
        Array<float> & ori, Array<float> & size, Array<float> & desc,          \
        const Array<T>& in, const unsigned n_layers, const float contrast_thr, \
        const float edge_thr, const float init_sigma, const bool double_input, \
        const float img_scale, const float feature_ratio,                      \
        const bool compute_GLOH);

INSTANTIATE(float, float)
INSTANTIATE(double, double)

}  // namespace cuda
}  // namespace flare
