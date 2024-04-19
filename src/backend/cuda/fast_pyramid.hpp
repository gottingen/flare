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

#pragma once

#include <Array.hpp>

#include <vector>

namespace flare {
namespace cuda {
template<typename T>
void fast_pyramid(std::vector<unsigned> &feat_pyr,
                  std::vector<Array<float>> &d_x_pyr,
                  std::vector<Array<float>> &d_y_pyr,
                  std::vector<unsigned> &lvl_best, std::vector<float> &lvl_scl,
                  std::vector<Array<T>> &img_pyr, const Array<T> &in,
                  const float fast_thr, const unsigned max_feat,
                  const float scl_fctr, const unsigned levels,
                  const unsigned patch_size);
}  // namespace cuda
}  // namespace flare
