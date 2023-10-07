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

#ifndef FLARE_CORE_HALF_H_
#define FLARE_CORE_HALF_H_

#include <flare/core/common/half_floating_point_wrapper.h>
#include <flare/core/common/half_numeric_traits.h>
#include <flare/core/common/half_mathematical_functions.h>

namespace flare {

////////////// BEGIN FP16/binary16 limits //////////////
#define FLARE_IMPL_FP16_MAX 65504.0F  // Maximum normalized number
#define FLARE_IMPL_FP16_MIN \
  0.000000059604645F  // Minimum normalized positive half precision number
#define FLARE_IMPL_FP16_RADIX \
  2  // Value of the base of the exponent representation. TODO: on all archs?
#define FLARE_IMPL_FP16_MANT_DIG \
  15  // Number of digits in the matissa that can be represented without losing
    // precision. TODO: Confirm this
#define FLARE_IMPL_FP16_MIN_EXP \
  -14  // This is the smallest possible exponent value
#define FLARE_IMPL_FP16_MAX_EXP \
  15  // This is the largest possible exponent value
#define FLARE_IMPL_FP16_SIGNIFICAND_BITS 10
#define FLARE_IMPL_FP16_EPSILON 0.0009765625F  // 1/2^10
#define FLARE_IMPL_HUGE_VALH 0x7c00            // bits [10,14] set.
////////////// END FP16/binary16 limits //////////////

////////////// BEGIN BF16/float16 limits //////////////
#define FLARE_IMPL_BF16_MAX 3.38953139e38  // Maximum normalized number
#define FLARE_IMPL_BF16_MIN \
  1.1754494351e-38  // Minimum normalized positive bhalf number
#define FLARE_IMPL_BF16_RADIX \
  2  // Value of the base of the exponent representation. TODO: on all archs?
#define FLARE_IMPL_BF16_MANT_DIG_MIN 2
#define FLARE_IMPL_BF16_MANT_DIG_MAX 3
#define FLARE_IMPL_BF16_MANT_DIG \
  FLARE_IMPL_BF16_MANT_DIG_MIN  // Number of digits in the matissa that
    // can be represented without losing
    // precision.
#define FLARE_IMPL_BF16_MIN_EXP \
  -126  // This is the smallest possible exponent value
#define FLARE_IMPL_BF16_MAX_EXP \
  127  // This is the largest possible exponent value
#define FLARE_IMPL_BF16_EPSILON 0.0078125F  // 1/2^7
////////////// END BF16/bfloat16 limits //////////////

}  // namespace flare

#endif  // FLARE_CORE_HALF_H_
