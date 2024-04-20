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

features fast(const array& in, const float thr, const unsigned arc_length,
              const bool non_max, const float feature_ratio,
              const unsigned edge) {
    fly_features temp;
    FLY_THROW(fly_fast(&temp, in.get(), thr, arc_length, non_max, feature_ratio,
                     edge));
    return features(temp);
}

}  // namespace fly
