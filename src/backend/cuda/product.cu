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
// mul
INSTANTIATE(fly_mul_t, float, float)
INSTANTIATE(fly_mul_t, double, double)
INSTANTIATE(fly_mul_t, cfloat, cfloat)
INSTANTIATE(fly_mul_t, cdouble, cdouble)
INSTANTIATE(fly_mul_t, int, int)
INSTANTIATE(fly_mul_t, uint, uint)
INSTANTIATE(fly_mul_t, intl, intl)
INSTANTIATE(fly_mul_t, uintl, uintl)
INSTANTIATE(fly_mul_t, char, int)
INSTANTIATE(fly_mul_t, uchar, uint)
INSTANTIATE(fly_mul_t, short, int)
INSTANTIATE(fly_mul_t, ushort, uint)
INSTANTIATE(fly_mul_t, half, float)
}  // namespace cuda
}  // namespace flare
