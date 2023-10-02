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

#ifndef FLARE_CORE_COMMON_QUAD_PRECISION_MATH_H_
#define FLARE_CORE_COMMON_QUAD_PRECISION_MATH_H_

#include <flare/core/defines.h>

#if defined(FLARE_ENABLE_LIBQUADMATH)

#include <flare/core/numeric_traits.h>
#include <flare/core/reduction_identity.h>
#include <flare/core/mathematical_constants.h>
#include <flare/core/mathematical_functions.h>

#include <quadmath.h>

#if !(defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__))
#error __float128 not supported on this host
#endif

namespace flare {
namespace experimental {
#define FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(TRAIT, TYPE, VALUE_TYPE, VALUE) \
  template <>                                                                \
  struct TRAIT<TYPE> {                                                       \
    static constexpr VALUE_TYPE value = VALUE;                               \
  };                                                                         \
  template <>                                                                \
  inline constexpr auto TRAIT##_v<TYPE> = TRAIT<TYPE>::value;

// clang-format off
// Numeric distinguished value traits
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(infinity,       __float128, __float128, HUGE_VALQ)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(finite_min,     __float128, __float128, -FLT128_MAX)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(finite_max,     __float128, __float128, FLT128_MAX)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(epsilon,        __float128, __float128, FLT128_EPSILON)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(round_error,    __float128, __float128, static_cast<__float128>(0.5))
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(norm_min,       __float128, __float128, FLT128_MIN)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(denorm_min,     __float128, __float128, FLT128_DENORM_MIN)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(quiet_NaN,      __float128, __float128, __builtin_nanq(""))
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(signaling_NaN,  __float128, __float128, __builtin_nansq(""))

// Numeric characteristics traits
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(digits,         __float128,        int, FLT128_MANT_DIG)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(digits10,       __float128,        int, FLT128_DIG)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(max_digits10,   __float128,        int, 36)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(radix,          __float128,        int, 2)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(min_exponent,   __float128,        int, FLT128_MIN_EXP)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(max_exponent,   __float128,        int, FLT128_MAX_EXP)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(min_exponent10, __float128,        int, FLT128_MIN_10_EXP)
FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT(max_exponent10, __float128,        int, FLT128_MAX_10_EXP)
// clang-format on

#undef FLARE_IMPL_SPECIALIZE_NUMERIC_TRAIT
}  // namespace experimental
}  // namespace flare

namespace flare {
template <>
struct reduction_identity<__float128> {
  FLARE_FORCEINLINE_FUNCTION constexpr static __float128 sum() {
    return static_cast<__float128>(0.0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static __float128 prod() {
    return static_cast<__float128>(1.0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static __float128 max() {
    return -FLT128_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static __float128 min() {
    return FLT128_MAX;
  }
};
}  // namespace flare

namespace flare {
// clang-format off
namespace detail {
template <> struct promote<__float128> { using type = __float128; };
}
// Basic operations
inline __float128 abs(__float128 x) { return ::fabsq(x); }
inline __float128 fabs(__float128 x) { return ::fabsq(x); }
inline __float128 fmod(__float128 x, __float128 y) { return ::fmodq(x, y); }
inline __float128 remainder(__float128 x, __float128 y) { return ::remainderq(x, y); }
// remquo
inline __float128 fma(__float128 x, __float128 y, __float128 z) { return ::fmaq(x, y, z); }
inline __float128 fmax(__float128 x, __float128 y) { return ::fmaxq(x, y); }
inline __float128 fmin(__float128 x, __float128 y) { return ::fminq(x, y); }
inline __float128 fdim(__float128 x, __float128 y) { return ::fdimq(x, y); }
inline __float128 nanq(char const* arg) { return ::nanq(arg); }
// Exponential functions
inline __float128 exp(__float128 x) { return ::expq(x); }
#if defined(FLARE_COMPILER_GNU) && (FLARE_COMPILER_GNU >= 910)
inline __float128 exp2(__float128 x) { return ::exp2q(x); }
#endif
inline __float128 expm1(__float128 x) { return ::expm1q(x); }
inline __float128 log(__float128 x) { return ::logq(x); }
inline __float128 log10(__float128 x) { return ::log10q(x); }
inline __float128 log2(__float128 x) { return ::log2q(x); }
inline __float128 log1p(__float128 x) { return ::log1pq(x); }
// Power functions
inline __float128 pow(__float128 x, __float128 y) { return ::powq(x, y); }
inline __float128 sqrt(__float128 x) { return ::sqrtq(x); }
inline __float128 cbrt(__float128 x) { return ::cbrtq(x); }
inline __float128 hypot(__float128 x, __float128 y) { return ::hypotq(x, y); }
// Trigonometric functions
inline __float128 sin(__float128 x) { return ::sinq(x); }
inline __float128 cos(__float128 x) { return ::cosq(x); }
inline __float128 tan(__float128 x) { return ::tanq(x); }
inline __float128 asin(__float128 x) { return ::asinq(x); }
inline __float128 acos(__float128 x) { return ::acosq(x); }
inline __float128 atan(__float128 x) { return ::atanq(x); }
inline __float128 atan2(__float128 x, __float128 y) { return ::atan2q(x, y); }
// Hyperbolic functions
inline __float128 sinh(__float128 x) { return ::sinhq(x); }
inline __float128 cosh(__float128 x) { return ::coshq(x); }
inline __float128 tanh(__float128 x) { return ::tanhq(x); }
inline __float128 asinh(__float128 x) { return ::asinhq(x); }
inline __float128 acosh(__float128 x) { return ::acoshq(x); }
inline __float128 atanh(__float128 x) { return ::atanhq(x); }
// Error and gamma functions
inline __float128 erf(__float128 x) { return ::erfq(x); }
inline __float128 erfc(__float128 x) { return ::erfcq(x); }
inline __float128 tgamma(__float128 x) { return ::tgammaq(x); }
inline __float128 lgamma(__float128 x) { return ::lgammaq(x); }
// Nearest integer floating point operations
inline __float128 ceil(__float128 x) { return ::ceilq(x); }
inline __float128 floor(__float128 x) { return ::floorq(x); }
inline __float128 trunc(__float128 x) { return ::truncq(x); }
inline __float128 round(__float128 x) { return ::roundq(x); }
// lround
// llround
inline __float128 nearbyint(__float128 x) { return ::nearbyintq(x); }
// rint
// lrint
// llrint
// Floating point manipulation functions
// frexp
// ldexp
// modf
// scalbn
// scalbln
// ilog
inline __float128 logb(__float128 x) { return ::logbq(x); }
inline __float128 nextafter(__float128 x, __float128 y) { return ::nextafterq(x, y); }
// nexttoward
inline __float128 copysign(__float128 x, __float128 y) { return ::copysignq(x, y); }
// Classification and comparison
// fpclassify
inline bool isfinite(__float128 x) { return !::isinfq(x); }  // isfiniteq not provided
inline bool isinf(__float128 x) { return ::isinfq(x); }
inline bool isnan(__float128 x) { return ::isnanq(x); }
// isnormal
inline bool signbit(__float128 x) { return ::signbitq(x); }
// isgreater
// isgreaterequal
// isless
// islessequal
// islessgreater
// isunordered
// clang-format on
}  // namespace flare

namespace flare::numbers {
// clang-format off
template <> constexpr __float128 e_v         <__float128> = 2.718281828459045235360287471352662498Q;
template <> constexpr __float128 log2e_v     <__float128> = 1.442695040888963407359924681001892137Q;
template <> constexpr __float128 log10e_v    <__float128> = 0.434294481903251827651128918916605082Q;
template <> constexpr __float128 pi_v        <__float128> = 3.141592653589793238462643383279502884Q;
template <> constexpr __float128 inv_pi_v    <__float128> = 0.318309886183790671537767526745028724Q;
template <> constexpr __float128 inv_sqrtpi_v<__float128> = 0.564189583547756286948079451560772586Q;
template <> constexpr __float128 ln2_v       <__float128> = 0.693147180559945309417232121458176568Q;
template <> constexpr __float128 ln10_v      <__float128> = 2.302585092994045684017991454684364208Q;
template <> constexpr __float128 sqrt2_v     <__float128> = 1.414213562373095048801688724209698079Q;
template <> constexpr __float128 sqrt3_v     <__float128> = 1.732050807568877293527446341505872367Q;
template <> constexpr __float128 inv_sqrt3_v <__float128> = 0.577350269189625764509148780501957456Q;
template <> constexpr __float128 egamma_v    <__float128> = 0.577215664901532860606512090082402431Q;
template <> constexpr __float128 phi_v       <__float128> = 1.618033988749894848204586834365638118Q;
// clang-format on
}  // namespace flare::numbers

#endif

#endif  // FLARE_CORE_COMMON_QUAD_PRECISION_MATH_H_
