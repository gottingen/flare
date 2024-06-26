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
// max
INSTANTIATE(fly_max_t, float, float)
INSTANTIATE(fly_max_t, double, double)
INSTANTIATE(fly_max_t, cfloat, cfloat)
INSTANTIATE(fly_max_t, cdouble, cdouble)
INSTANTIATE(fly_max_t, int, int)
INSTANTIATE(fly_max_t, uint, uint)
INSTANTIATE(fly_max_t, intl, intl)
INSTANTIATE(fly_max_t, uintl, uintl)
INSTANTIATE(fly_max_t, char, char)
INSTANTIATE(fly_max_t, uchar, uchar)
INSTANTIATE(fly_max_t, short, short)
INSTANTIATE(fly_max_t, ushort, ushort)
INSTANTIATE(fly_max_t, half, half)
}  // namespace cuda
}  // namespace flare
