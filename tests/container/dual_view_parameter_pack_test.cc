// Copyright 2023 The Elastic-AI Authors.
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

#include <flare/core.h>
#include <flare/dual_tensor.h>

namespace {

template <class DataType, class Arg1Type = void, class Arg2Type = void,
          class Arg3Type = void>
void not_supported_anymore(
    flare::DualTensor<DataType, Arg1Type, Arg2Type, Arg2Type> x) {
  static_assert(flare::is_dual_tensor_v<decltype(x)>);
}

template <class DataType, class... Properties>
void prefer_instead(flare::DualTensor<DataType, Properties...> x) {
  static_assert(flare::is_dual_tensor_v<decltype(x)>);
}

using KDV = flare::DualTensor<int*>;


static_assert(std::is_void_v<decltype(prefer_instead(std::declval<KDV>()))>);

}  // namespace
