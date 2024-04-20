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

#include <common/half.hpp>
#include "reduce_impl.hpp"

using flare::common::half;

namespace flare {
namespace cuda {
// alltrue
INSTANTIATE(fly_and_t, float, char)
INSTANTIATE(fly_and_t, double, char)
INSTANTIATE(fly_and_t, cfloat, char)
INSTANTIATE(fly_and_t, cdouble, char)
INSTANTIATE(fly_and_t, int, char)
INSTANTIATE(fly_and_t, uint, char)
INSTANTIATE(fly_and_t, intl, char)
INSTANTIATE(fly_and_t, uintl, char)
INSTANTIATE(fly_and_t, char, char)
INSTANTIATE(fly_and_t, uchar, char)
INSTANTIATE(fly_and_t, short, char)
INSTANTIATE(fly_and_t, ushort, char)
INSTANTIATE(fly_and_t, half, char)
}  // namespace cuda
}  // namespace flare
