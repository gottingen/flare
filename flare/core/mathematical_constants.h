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
#ifndef FLARE_CORE_MATHEMATICAL_CONSTANTS_H_
#define FLARE_CORE_MATHEMATICAL_CONSTANTS_H_

#include <flare/core/defines.h>
#include <type_traits>

namespace flare::numbers {

#define FLARE_IMPL_MATH_CONSTANT(TRAIT, VALUE)                \
  template <class T>                                           \
  inline constexpr auto TRAIT##_v =                            \
      std::enable_if_t<std::is_floating_point_v<T>, T>(VALUE); \
  inline constexpr auto TRAIT = TRAIT##_v<double>

// clang-format off
FLARE_IMPL_MATH_CONSTANT(e,          2.718281828459045235360287471352662498L);
FLARE_IMPL_MATH_CONSTANT(log2e,      1.442695040888963407359924681001892137L);
FLARE_IMPL_MATH_CONSTANT(log10e,     0.434294481903251827651128918916605082L);
FLARE_IMPL_MATH_CONSTANT(pi,         3.141592653589793238462643383279502884L);
FLARE_IMPL_MATH_CONSTANT(inv_pi,     0.318309886183790671537767526745028724L);
FLARE_IMPL_MATH_CONSTANT(inv_sqrtpi, 0.564189583547756286948079451560772586L);
FLARE_IMPL_MATH_CONSTANT(ln2,        0.693147180559945309417232121458176568L);
FLARE_IMPL_MATH_CONSTANT(ln10,       2.302585092994045684017991454684364208L);
FLARE_IMPL_MATH_CONSTANT(sqrt2,      1.414213562373095048801688724209698079L);
FLARE_IMPL_MATH_CONSTANT(sqrt3,      1.732050807568877293527446341505872367L);
FLARE_IMPL_MATH_CONSTANT(inv_sqrt3,  0.577350269189625764509148780501957456L);
FLARE_IMPL_MATH_CONSTANT(egamma,     0.577215664901532860606512090082402431L);
FLARE_IMPL_MATH_CONSTANT(phi,        1.618033988749894848204586834365638118L);
// clang-format on

#undef FLARE_IMPL_MATH_CONSTANT

}  // namespace flare::numbers


#endif  // FLARE_CORE_MATHEMATICAL_CONSTANTS_H_
