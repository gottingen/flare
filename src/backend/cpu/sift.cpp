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

namespace flare {
namespace cpu {

template<typename T, typename convAccT>
unsigned sift(Array<float>& x, Array<float>& y, Array<float>& score,
              Array<float>& ori, Array<float>& size, Array<float>& desc,
              const Array<T>& in, const unsigned n_layers,
              const float contrast_thr, const float edge_thr,
              const float init_sigma, const bool double_input,
              const float img_scale, const float feature_ratio,
              const bool compute_GLOH) {
    return sift_impl<T, convAccT>(
        x, y, score, ori, size, desc, in, n_layers, contrast_thr, edge_thr,
        init_sigma, double_input, img_scale, feature_ratio, compute_GLOH);
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

}  // namespace cpu
}  // namespace flare
