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
// count
INSTANTIATE(fly_notzero_t, float, uint)
INSTANTIATE(fly_notzero_t, double, uint)
INSTANTIATE(fly_notzero_t, cfloat, uint)
INSTANTIATE(fly_notzero_t, cdouble, uint)
INSTANTIATE(fly_notzero_t, int, uint)
INSTANTIATE(fly_notzero_t, uint, uint)
INSTANTIATE(fly_notzero_t, intl, uint)
INSTANTIATE(fly_notzero_t, uintl, uint)
INSTANTIATE(fly_notzero_t, short, uint)
INSTANTIATE(fly_notzero_t, ushort, uint)
INSTANTIATE(fly_notzero_t, char, uint)
INSTANTIATE(fly_notzero_t, uchar, uint)
INSTANTIATE(fly_notzero_t, half, uint)
}  // namespace cuda
}  // namespace flare
