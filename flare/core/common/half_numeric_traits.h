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

#ifndef FLARE_CORE_COMMON_HALF_NUMERIC_TRAITS_H_
#define FLARE_CORE_COMMON_HALF_NUMERIC_TRAITS_H_

#include <flare/core/numeric_traits.h>

////////////// BEGIN HALF_T (binary16) limits //////////////
// clang-format off
// '\brief:' below are from the libc definitions for float and double:
// https://www.gnu.org/software/libc/manual/html_node/Floating-Point-Parameters.html
//
// The arithmetic encoding and equations below are derived from:
// Ref1: https://en.wikipedia.org/wiki/Single-precision_floating-point_format
// Ref2: https://en.wikipedia.org/wiki/Exponent_bias
// Ref3; https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
//
// Some background on the magic numbers 2**10=1024 and 2**15=32768 used below:
//
// IMPORTANT: For IEEE754 encodings, see Ref1.
//
// For binary16, we have B = 2 and p = 16 with 2**16 possible significands.
// The binary16 format is: [s  e  e  e  e  e  f f f f f f f f f f]
//              bit index:  15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
// s: signed bit (1 bit)
// e: exponent bits (5 bits)
// f: fractional bits (10 bits)
//
// E_bias      = 2**(n_exponent_bits - 1) - 1 = 2**(5 - 1) - 1 = 15
// E_subnormal = 00000 (base2)
// E_infinity  = 11111 (base2)
// E_min       = 1 - E_bias = 1 - 15
// E_max       = 2**5 - 1 - E_bias = 2**5 - 1 - 15 = 16
//
// 2**10=1024 is the smallest denominator that is representable in binary16:
// [s  e  e  e  e  e  f f f f f f f f f f]
// [0  0  0  0  0  0  0 0 0 0 0 0 0 0 0 1]
// which is: 1 / 2**-10
//
//
// 2**15 is the largest exponent factor representable in binary16, for example the
// largest integer value representable in binary16 is:
// [s  e  e  e  e  e  f f f f f f f f f f]
// [0  1  1  1  1  0  1 1 1 1 1 1 1 1 1 1]
// which is: 2**(2**4 + 2**3 + 2**2 + 2**1 - 15) * (1 + 2**-10 + 2**-9 + 2**-8 + 2**-7 + 2**-6 + 2**-5 + 2**-4 + 2**-3 + 2**-2 + 2**-1)) =
//           2**15 * (1 + 0.9990234375) =
//           65504.0
//
#if defined(FLARE_HALF_T_IS_FLOAT) && !FLARE_HALF_T_IS_FLOAT
/// \brief: Infinity
///
/// Binary16 encoding:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [0  1  1  1  1  1  0 0 0 0 0 0 0 0 0 0]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
///
template <>
struct flare::experimental::detail::infinity_helper<flare::experimental::half_t> {
  static constexpr int value = 0x7C00;
};

/// \brief: Minimum normalized number
///
/// Stdc defines this as the smallest number (representable in binary16).
///
/// Binary16 encoding:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [1  1  1  1  1  0  1 1 1 1 1 1 1 1 1 1]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
///
/// and in base10: -1 * 2**(2**4 + 2**3 + 2**2 + 2**1 - 15) * (1 + 2**-10 + 2**-9 + 2**-8 + 2**-7 + 2**-6 + 2**-5 + 2**-4 + 2**-3 + 2**-2 + 2**-1)
///              = -2**15 * (1 + (2**10 - 1) / 2**10)
template <>
struct flare::experimental::detail::finite_min_helper<
    flare::experimental::half_t> {
  static constexpr float value = -65504.0F;
};

/// \brief: Maximum normalized number
///
/// Stdc defines this as the maximum number (representable in binary16).
///
/// Binary16 encoding:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [0  1  1  1  1  0  1 1 1 1 1 1 1 1 1 1]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
///
/// and in base10: 1 * 2**(2**4 + 2**3 + 2**2 + 2**1 - 15) * (1 + 2**-10 + 2**-9 + 2**-8 + 2**-7 + 2**-6 + 2**-5 + 2**-4 + 2**-3 + 2**-2 + 2**-1)
///              = 2**15 * (1 + (2**10 - 1) / 2**10)
template <>
struct flare::experimental::detail::finite_max_helper<
    flare::experimental::half_t> {
  static constexpr float value = 65504.0F;
};

/// \brief: This is the difference between 1 and the smallest floating point
///         number of type binary16 that is greater than 1
///
/// Smallest number in binary16 that is greater than 1 encoding:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [0  0  1  1  1  1  0 0 0 0 0 0 0 0 0 1]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
///
/// and in base10: 1 * 2**(2**3 + 2**2 + 2**1 + 2**0 - 15) * (1 + 2**-10)
///                = 2**0 * (1 + 2**-10)
///                = 1.0009765625
///
/// Lastly, 1 - 1.0009765625 = 0.0009765625.
template <>
struct flare::experimental::detail::epsilon_helper<
    flare::experimental::half_t> {
  static constexpr float value = 0.0009765625F;
};

/// @brief: The largest possible rounding error in ULPs
///
/// This simply uses the maximum rounding error.
///
/// Reference: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#689
template <>
struct flare::experimental::detail::round_error_helper<
    flare::experimental::half_t> {
  static constexpr float value = 0.5F;
};

/// \brief: Minimum normalized positive half precision number
///
/// Stdc defines this as the minimum normalized positive floating
/// point number that is representable in type binary16
///
/// Smallest number in binary16 that is greater than 1 encoding:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [0  0  0  0  0  1  0 0 0 0 0 0 0 0 0 0]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
///
/// and in base10: 1 * 2**(2**0 - 15) * (1)
///                = 2**-14
template <>
struct flare::experimental::detail::norm_min_helper<
    flare::experimental::half_t> {
  static constexpr float value = 0.00006103515625F;
};

/// \brief: Quiet not a half precision number
///
/// IEEE 754 defines this as all exponent bits high.
///
/// Quiet NaN in binary16:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [1  1  1  1  1  1  0 0 0 0 0 0 0 0 0 0]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
template <>
struct flare::experimental::detail::quiet_NaN_helper<
    flare::experimental::half_t> {
  static constexpr float value = 0xfc000;
};

/// \brief: Signaling not a half precision number
///
/// IEEE 754 defines this as all exponent bits and the first fraction bit high.
///
/// Quiet NaN in binary16:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [1  1  1  1  1  1  1 0 0 0 0 0 0 0 0 0]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
template <>
struct flare::experimental::detail::signaling_NaN_helper<
    flare::experimental::half_t> {
  static constexpr float value = 0xfe000;
};

/// \brief: Number of digits in the matissa that can be represented
///         without losing precision.
///
/// Stdc defines this as the number of base-RADIX digits in the floating point mantissa for the binary16 data type.
///
/// In binary16, we have 10 fractional bits plus the implicit leading 1.
template <>
struct flare::experimental::detail::digits_helper<flare::experimental::half_t> {
  static constexpr int value = 11;
};

/// \brief: "The number of base-10 digits that can be represented by the type T without change"
/// Reference: https://en.cppreference.com/w/cpp/types/numeric_limits/digits10.
///
/// "For base-radix types, it is the value of digits() (digits - 1 for floating-point types) multiplied by log10(radix) and rounded down."
/// Reference: https://en.cppreference.com/w/cpp/types/numeric_limits/digits10.
///
/// This is: floor(11 - 1 * log10(2))
template <>
struct flare::experimental::detail::digits10_helper<
    flare::experimental::half_t> {
  static constexpr int value = 3;
};

/// \brief: Value of the base of the exponent representation.
///
/// Stdc defined this as the value of the base, or radix, of the exponent representation.
template <>
struct flare::experimental::detail::radix_helper<flare::experimental::half_t> {
  static constexpr int value = 2;
};

/// \brief: This is the smallest possible exponent value
///
/// Stdc defines this as the smallest possible exponent value for type binary16. 
/// More precisely, it is the minimum negative integer such that the value min_exponent_helper
/// raised to this power minus 1 can be represented as a normalized floating point number of type float.
///
/// In binary16:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [0  0  0  0  0  1  0 0 0 0 0 0 0 0 0 0]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
/// 
/// and in base10: 1 * 2**(2**0 - 15) * (1 + 0)
///                = 2**-14
/// 
/// with a bias of one from (C11 5.2.4.2.2), gives -13;
template <>
struct flare::experimental::detail::min_exponent_helper<
    flare::experimental::half_t> {
  static constexpr int value = -13;
};

/// \brief: This is the largest possible exponent value
///
/// In binary16:
///             [s  e  e  e  e  e  f f f f f f f f f f]
///             [0  1  1  1  1  0  0 0 0 0 0 0 0 0 0 0]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
/// 
/// and in base10: 1 * 2**(2**4 + 2**3 + 2**2 + 2**1 - 15) * (1 + 0)
///                = 2**(30 - 15)
///                = 2**15
/// 
/// with a bias of one from (C11 5.2.4.2.2), gives 16;
template <>
struct flare::experimental::detail::max_exponent_helper<
    flare::experimental::half_t> {
  static constexpr int value = 16;
};
#endif
////////////// END HALF_T (binary16) limits //////////////

////////////// BEGIN BHALF_T (bfloat16) limits //////////////
#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
/// \brief: Infinity
///
/// Bfloat16 encoding:
///             [s  e  e  e  e  e  e e e f f f f f f f]
///             [0  1  1  1  1  1  1 1 1 0 0 0 0 0 0 0]
/// bit index:   15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
///
template <>
struct flare::experimental::detail::infinity_helper<flare::experimental::bhalf_t> {
  static constexpr int value = 0x7F80;
};

// Minimum normalized number
template <>
struct flare::experimental::detail::finite_min_helper<
    flare::experimental::bhalf_t> {
  static constexpr float value = -3.38953139e38;
};
// Maximum normalized number
template <>
struct flare::experimental::detail::finite_max_helper<
    flare::experimental::bhalf_t> {
  static constexpr float value = 3.38953139e38;
};
// 1/2^7
template <>
struct flare::experimental::detail::epsilon_helper<
    flare::experimental::bhalf_t> {
  static constexpr float value = 0.0078125F;
};
template <>
struct flare::experimental::detail::round_error_helper<
    flare::experimental::bhalf_t> {
  static constexpr float value = 0.5F;
};
// Minimum normalized positive bhalf number
template <>
struct flare::experimental::detail::norm_min_helper<
    flare::experimental::bhalf_t> {
  static constexpr float value = 1.1754494351e-38;
};
// Quiet not a bhalf number
template <>
struct flare::experimental::detail::quiet_NaN_helper<
    flare::experimental::bhalf_t> {
  static constexpr float value = 0x7fc000;
};
// Signaling not a bhalf number
template <>
struct flare::experimental::detail::signaling_NaN_helper<
    flare::experimental::bhalf_t> {
  static constexpr float value = 0x7fe000;
};
// Number of digits in the matissa that can be represented
// without losing precision.
template <>
struct flare::experimental::detail::digits_helper<
    flare::experimental::bhalf_t> {
  static constexpr int value = 2;
};
// 7 - 1 * log10(2)
template <>
struct flare::experimental::detail::digits10_helper<
    flare::experimental::bhalf_t> {
  static constexpr int value = 1;
};
// Value of the base of the exponent representation.
template <>
struct flare::experimental::detail::radix_helper<flare::experimental::bhalf_t> {
  static constexpr int value = 2;
};
// This is the smallest possible exponent value
// with a bias of one (C11 5.2.4.2.2).
template <>
struct flare::experimental::detail::min_exponent_helper<
    flare::experimental::bhalf_t> {
  static constexpr int value = -125;
};
// This is the largest possible exponent value
// with a bias of one (C11 5.2.4.2.2).
template <>
struct flare::experimental::detail::max_exponent_helper<
    flare::experimental::bhalf_t> {
  static constexpr int value = 128;
};
#endif
////////////// END BHALF_T (bfloat16) limits //////////

#endif  // FLARE_CORE_COMMON_HALF_NUMERIC_TRAITS_H_
