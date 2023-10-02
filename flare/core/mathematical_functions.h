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

#ifndef FLARE_MATHEMATICAL_FUNCTIONS_H_
#define FLARE_MATHEMATICAL_FUNCTIONS_H_

#include <flare/core/defines.h>
#include <cmath>
#include <cstdlib>
#include <type_traits>

namespace flare {

namespace detail {
template <class T, bool = std::is_integral_v<T>>
struct promote {
  using type = double;
};
template <class T>
struct promote<T, false> {};
template <>
struct promote<long double> {
  using type = long double;
};
template <>
struct promote<double> {
  using type = double;
};
template <>
struct promote<float> {
  using type = float;
};
template <class T>
using promote_t = typename promote<T>::type;
template <class T, class U,
          bool = std::is_arithmetic_v<T>&& std::is_arithmetic_v<U>>
struct promote_2 {
  using type = decltype(promote_t<T>() + promote_t<U>());
};
template <class T, class U>
struct promote_2<T, U, false> {};
template <class T, class U>
using promote_2_t = typename promote_2<T, U>::type;
template <class T, class U, class V,
          bool = std::is_arithmetic_v<T>&& std::is_arithmetic_v<U>&&
              std::is_arithmetic_v<V>>
struct promote_3 {
  using type = decltype(promote_t<T>() + promote_t<U>() + promote_t<V>());
};
template <class T, class U, class V>
struct promote_3<T, U, V, false> {};
template <class T, class U, class V>
using promote_3_t = typename promote_3<T, U, V>::type;
}  // namespace detail

// NOTE long double overloads are not available on the device


#if (defined(FLARE_COMPILER_NVCC) || defined(FLARE_COMPILER_NVHPC)) && \
    defined(__GNUC__) && (__GNUC__ < 6) && !defined(__clang__)
#define FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE
#else
#define FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE std
#endif

#define FLARE_IMPL_MATH_UNARY_FUNCTION(FUNC)                                  \
  FLARE_INLINE_FUNCTION float FUNC(float x) {                                 \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(x);                                                            \
  }                                                                            \
  FLARE_INLINE_FUNCTION double FUNC(double x) {                               \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(x);                                                            \
  }                                                                            \
  inline long double FUNC(long double x) {                                     \
    using std::FUNC;                                                           \
    return FUNC(x);                                                            \
  }                                                                            \
  FLARE_INLINE_FUNCTION float FUNC##f(float x) {                              \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(x);                                                            \
  }                                                                            \
  inline long double FUNC##l(long double x) {                                  \
    using std::FUNC;                                                           \
    return FUNC(x);                                                            \
  }                                                                            \
  template <class T>                                                           \
  FLARE_INLINE_FUNCTION std::enable_if_t<std::is_integral_v<T>, double> FUNC( \
      T x) {                                                                   \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(static_cast<double>(x));                                       \
  }

// isinf, isnan, and isinfinite do not work on Windows with CUDA with std::
// getting warnings about calling host function in device function then
// runtime test fails
#if defined(_WIN32) && defined(FLARE_ON_CUDA_DEVICE)
#define FLARE_IMPL_MATH_UNARY_PREDICATE(FUNC)                               \
  FLARE_INLINE_FUNCTION bool FUNC(float x) { return ::FUNC(x); }            \
  FLARE_INLINE_FUNCTION bool FUNC(double x) { return ::FUNC(x); }           \
  inline bool FUNC(long double x) {                                          \
    using std::FUNC;                                                         \
    return FUNC(x);                                                          \
  }                                                                          \
  template <class T>                                                         \
  FLARE_INLINE_FUNCTION std::enable_if_t<std::is_integral_v<T>, bool> FUNC( \
      T x) {                                                                 \
    return ::FUNC(static_cast<double>(x));                                   \
  }
#else
#define FLARE_IMPL_MATH_UNARY_PREDICATE(FUNC)                               \
  FLARE_INLINE_FUNCTION bool FUNC(float x) {                                \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                        \
    return FUNC(x);                                                          \
  }                                                                          \
  FLARE_INLINE_FUNCTION bool FUNC(double x) {                               \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                        \
    return FUNC(x);                                                          \
  }                                                                          \
  inline bool FUNC(long double x) {                                          \
    using std::FUNC;                                                         \
    return FUNC(x);                                                          \
  }                                                                          \
  template <class T>                                                         \
  FLARE_INLINE_FUNCTION std::enable_if_t<std::is_integral_v<T>, bool> FUNC( \
      T x) {                                                                 \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                        \
    return FUNC(static_cast<double>(x));                                     \
  }
#endif

#define FLARE_IMPL_MATH_BINARY_FUNCTION(FUNC)                                 \
  FLARE_INLINE_FUNCTION float FUNC(float x, float y) {                        \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(x, y);                                                         \
  }                                                                            \
  FLARE_INLINE_FUNCTION double FUNC(double x, double y) {                     \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(x, y);                                                         \
  }                                                                            \
  inline long double FUNC(long double x, long double y) {                      \
    using std::FUNC;                                                           \
    return FUNC(x, y);                                                         \
  }                                                                            \
  FLARE_INLINE_FUNCTION float FUNC##f(float x, float y) {                     \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(x, y);                                                         \
  }                                                                            \
  inline long double FUNC##l(long double x, long double y) {                   \
    using std::FUNC;                                                           \
    return FUNC(x, y);                                                         \
  }                                                                            \
  template <class T1, class T2>                                                \
  FLARE_INLINE_FUNCTION                                                       \
      std::enable_if_t<std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2> && \
                           !std::is_same_v<T1, long double> &&                 \
                           !std::is_same_v<T2, long double>,                   \
                       flare::detail::promote_2_t<T1, T2>>                      \
      FUNC(T1 x, T2 y) {                                                       \
    using Promoted = flare::detail::promote_2_t<T1, T2>;                        \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                          \
    return FUNC(static_cast<Promoted>(x), static_cast<Promoted>(y));           \
  }                                                                            \
  template <class T1, class T2>                                                \
  inline std::enable_if_t<std::is_arithmetic_v<T1> &&                          \
                              std::is_arithmetic_v<T2> &&                      \
                              (std::is_same_v<T1, long double> ||              \
                               std::is_same_v<T2, long double>),               \
                          long double>                                         \
  FUNC(T1 x, T2 y) {                                                           \
    using Promoted = flare::detail::promote_2_t<T1, T2>;                        \
    static_assert(std::is_same_v<Promoted, long double>, "");                  \
    using std::FUNC;                                                           \
    return FUNC(static_cast<Promoted>(x), static_cast<Promoted>(y));           \
  }

#define FLARE_IMPL_MATH_TERNARY_FUNCTION(FUNC)                             \
  FLARE_INLINE_FUNCTION float FUNC(float x, float y, float z) {            \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                       \
    return FUNC(x, y, z);                                                   \
  }                                                                         \
  FLARE_INLINE_FUNCTION double FUNC(double x, double y, double z) {        \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                       \
    return FUNC(x, y, z);                                                   \
  }                                                                         \
  inline long double FUNC(long double x, long double y, long double z) {    \
    using std::FUNC;                                                        \
    return FUNC(x, y, z);                                                   \
  }                                                                         \
  FLARE_INLINE_FUNCTION float FUNC##f(float x, float y, float z) {         \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                       \
    return FUNC(x, y, z);                                                   \
  }                                                                         \
  inline long double FUNC##l(long double x, long double y, long double z) { \
    using std::FUNC;                                                        \
    return FUNC(x, y, z);                                                   \
  }                                                                         \
  template <class T1, class T2, class T3>                                   \
  FLARE_INLINE_FUNCTION std::enable_if_t<                                  \
      std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2> &&               \
          std::is_arithmetic_v<T3> && !std::is_same_v<T1, long double> &&   \
          !std::is_same_v<T2, long double> &&                               \
          !std::is_same_v<T3, long double>,                                 \
      flare::detail::promote_3_t<T1, T2, T3>>                                \
  FUNC(T1 x, T2 y, T3 z) {                                                  \
    using Promoted = flare::detail::promote_3_t<T1, T2, T3>;                 \
    using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::FUNC;                       \
    return FUNC(static_cast<Promoted>(x), static_cast<Promoted>(y),         \
                static_cast<Promoted>(z));                                  \
  }                                                                         \
  template <class T1, class T2, class T3>                                   \
  inline std::enable_if_t<std::is_arithmetic_v<T1> &&                       \
                              std::is_arithmetic_v<T2> &&                   \
                              std::is_arithmetic_v<T3> &&                   \
                              (std::is_same_v<T1, long double> ||           \
                               std::is_same_v<T2, long double> ||           \
                               std::is_same_v<T3, long double>),            \
                          long double>                                      \
  FUNC(T1 x, T2 y, T3 z) {                                                  \
    using Promoted = flare::detail::promote_3_t<T1, T2, T3>;                 \
    static_assert(std::is_same_v<Promoted, long double>);                   \
    using std::FUNC;                                                        \
    return FUNC(static_cast<Promoted>(x), static_cast<Promoted>(y),         \
                static_cast<Promoted>(z));                                  \
  }

// Basic operations
FLARE_INLINE_FUNCTION int abs(int n) {
  using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::abs;
  return abs(n);
}
FLARE_INLINE_FUNCTION long abs(long n) {
// FIXME_NVHPC ptxas fatal   : unresolved extern function 'labs'
#if defined(FLARE_COMPILER_NVHPC) && FLARE_COMPILER_NVHPC < 230700
  return n > 0 ? n : -n;
#else
  using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::abs;
  return abs(n);
#endif
}
FLARE_INLINE_FUNCTION long long abs(long long n) {
// FIXME_NVHPC ptxas fatal   : unresolved extern function 'labs'
#if defined(FLARE_COMPILER_NVHPC) && FLARE_COMPILER_NVHPC < 230700
  return n > 0 ? n : -n;
#else
  using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::abs;
  return abs(n);
#endif
}
FLARE_INLINE_FUNCTION float abs(float x) {
  using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::abs;
  return abs(x);
}
FLARE_INLINE_FUNCTION double abs(double x) {
  using FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE::abs;
  return abs(x);
}
inline long double abs(long double x) {
  using std::abs;
  return abs(x);
}

FLARE_IMPL_MATH_UNARY_FUNCTION(fabs)
FLARE_IMPL_MATH_BINARY_FUNCTION(fmod)
FLARE_IMPL_MATH_BINARY_FUNCTION(remainder)
// remquo
FLARE_IMPL_MATH_TERNARY_FUNCTION(fma)
FLARE_IMPL_MATH_BINARY_FUNCTION(fmax)
FLARE_IMPL_MATH_BINARY_FUNCTION(fmin)
FLARE_IMPL_MATH_BINARY_FUNCTION(fdim)
FLARE_INLINE_FUNCTION float nanf(char const* arg) { return ::nanf(arg); }
FLARE_INLINE_FUNCTION double nan(char const* arg) { return ::nan(arg); }
inline long double nanl(char const* arg) { return ::nanl(arg); }

// Exponential functions
FLARE_IMPL_MATH_UNARY_FUNCTION(exp)
// FIXME_NVHPC nvc++ has issues with exp2
#if defined(FLARE_COMPILER_NVHPC) && FLARE_COMPILER_NVHPC < 230700
FLARE_INLINE_FUNCTION float exp2(float val) {
  constexpr float ln2 = 0.693147180559945309417232121458176568L;
  return exp(ln2 * val);
}
FLARE_INLINE_FUNCTION double exp2(double val) {
  constexpr double ln2 = 0.693147180559945309417232121458176568L;
  return exp(ln2 * val);
}
inline long double exp2(long double val) {
  constexpr long double ln2 = 0.693147180559945309417232121458176568L;
  return exp(ln2 * val);
}
template <class T>
FLARE_INLINE_FUNCTION double exp2(T val) {
  constexpr double ln2 = 0.693147180559945309417232121458176568L;
  return exp(ln2 * static_cast<double>(val));
}
#else
FLARE_IMPL_MATH_UNARY_FUNCTION(exp2)
#endif
FLARE_IMPL_MATH_UNARY_FUNCTION(expm1)
FLARE_IMPL_MATH_UNARY_FUNCTION(log)
FLARE_IMPL_MATH_UNARY_FUNCTION(log10)
FLARE_IMPL_MATH_UNARY_FUNCTION(log2)
FLARE_IMPL_MATH_UNARY_FUNCTION(log1p)
// Power functions
FLARE_IMPL_MATH_BINARY_FUNCTION(pow)
FLARE_IMPL_MATH_UNARY_FUNCTION(sqrt)
FLARE_IMPL_MATH_UNARY_FUNCTION(cbrt)
FLARE_IMPL_MATH_BINARY_FUNCTION(hypot)
#if defined(FLARE_ON_CUDA_DEVICE)
FLARE_INLINE_FUNCTION float hypot(float x, float y, float z) {
  return sqrt(x * x + y * y + z * z);
}
FLARE_INLINE_FUNCTION double hypot(double x, double y, double z) {
  return sqrt(x * x + y * y + z * z);
}
inline long double hypot(long double x, long double y, long double z) {
  return sqrt(x * x + y * y + z * z);
}
FLARE_INLINE_FUNCTION float hypotf(float x, float y, float z) {
  return sqrt(x * x + y * y + z * z);
}
inline long double hypotl(long double x, long double y, long double z) {
  return sqrt(x * x + y * y + z * z);
}
template <
    class T1, class T2, class T3,
    class Promoted = std::enable_if_t<
        std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2> &&
            std::is_arithmetic_v<T3> && !std::is_same_v<T1, long double> &&
            !std::is_same_v<T2, long double> &&
            !std::is_same_v<T3, long double>,
        detail::promote_3_t<T1, T2, T3>>>
FLARE_INLINE_FUNCTION Promoted hypot(T1 x, T2 y, T3 z) {
  return hypot(static_cast<Promoted>(x), static_cast<Promoted>(y),
               static_cast<Promoted>(z));
}
template <
    class T1, class T2, class T3,
    class = std::enable_if_t<
        std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2> &&
        std::is_arithmetic_v<T3> &&
        (std::is_same_v<T1, long double> || std::is_same_v<T2, long double> ||
         std::is_same_v<T3, long double>)>>
inline long double hypot(T1 x, T2 y, T3 z) {
  return hypot(static_cast<long double>(x), static_cast<long double>(y),
               static_cast<long double>(z));
}
#else
FLARE_IMPL_MATH_TERNARY_FUNCTION(hypot)
#endif
// Trigonometric functions
FLARE_IMPL_MATH_UNARY_FUNCTION(sin)
FLARE_IMPL_MATH_UNARY_FUNCTION(cos)
FLARE_IMPL_MATH_UNARY_FUNCTION(tan)
FLARE_IMPL_MATH_UNARY_FUNCTION(asin)
FLARE_IMPL_MATH_UNARY_FUNCTION(acos)
FLARE_IMPL_MATH_UNARY_FUNCTION(atan)
FLARE_IMPL_MATH_BINARY_FUNCTION(atan2)
// Hyperbolic functions
FLARE_IMPL_MATH_UNARY_FUNCTION(sinh)
FLARE_IMPL_MATH_UNARY_FUNCTION(cosh)
FLARE_IMPL_MATH_UNARY_FUNCTION(tanh)
FLARE_IMPL_MATH_UNARY_FUNCTION(asinh)
FLARE_IMPL_MATH_UNARY_FUNCTION(acosh)
FLARE_IMPL_MATH_UNARY_FUNCTION(atanh)
// Error and gamma functions
FLARE_IMPL_MATH_UNARY_FUNCTION(erf)
FLARE_IMPL_MATH_UNARY_FUNCTION(erfc)
FLARE_IMPL_MATH_UNARY_FUNCTION(tgamma)
FLARE_IMPL_MATH_UNARY_FUNCTION(lgamma)
// Nearest integer floating point operations
FLARE_IMPL_MATH_UNARY_FUNCTION(ceil)
FLARE_IMPL_MATH_UNARY_FUNCTION(floor)
FLARE_IMPL_MATH_UNARY_FUNCTION(trunc)
FLARE_IMPL_MATH_UNARY_FUNCTION(round)
// lround
// llround

FLARE_IMPL_MATH_UNARY_FUNCTION(nearbyint)
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
FLARE_IMPL_MATH_UNARY_FUNCTION(logb)
FLARE_IMPL_MATH_BINARY_FUNCTION(nextafter)
// nexttoward
FLARE_IMPL_MATH_BINARY_FUNCTION(copysign)
// Classification and comparison
// fpclassify
FLARE_IMPL_MATH_UNARY_PREDICATE(isfinite)
FLARE_IMPL_MATH_UNARY_PREDICATE(isinf)
FLARE_IMPL_MATH_UNARY_PREDICATE(isnan)
// isnormal
FLARE_IMPL_MATH_UNARY_PREDICATE(signbit)
// isgreater
// isgreaterequal
// isless
// islessequal
// islessgreater
// isunordered

#undef FLARE_IMPL_MATH_FUNCTIONS_NAMESPACE
#undef FLARE_IMPL_MATH_UNARY_FUNCTION
#undef FLARE_IMPL_MATH_UNARY_PREDICATE
#undef FLARE_IMPL_MATH_BINARY_FUNCTION
#undef FLARE_IMPL_MATH_TERNARY_FUNCTION

// non-standard math functions provided by CUDA
FLARE_INLINE_FUNCTION float rsqrt(float val) {
#if defined(FLARE_ON_CUDA_DEVICE)
  FLARE_IF_ON_DEVICE(return ::rsqrtf(val);)
  FLARE_IF_ON_HOST(return 1.0f / flare::sqrt(val);)
#else
  return 1.0f / flare::sqrt(val);
#endif
}
FLARE_INLINE_FUNCTION double rsqrt(double val) {
#if defined(FLARE_ON_CUDA_DEVICE)
  FLARE_IF_ON_DEVICE(return ::rsqrt(val);)
  FLARE_IF_ON_HOST(return 1.0 / flare::sqrt(val);)
#else
  return 1.0 / flare::sqrt(val);
#endif
}
inline long double rsqrt(long double val) { return 1.0l / flare::sqrt(val); }
FLARE_INLINE_FUNCTION float rsqrtf(float x) { return flare::rsqrt(x); }
inline long double rsqrtl(long double x) { return flare::rsqrt(x); }
template <class T>
FLARE_INLINE_FUNCTION std::enable_if_t<std::is_integral_v<T>, double> rsqrt(
    T x) {
  return flare::rsqrt(static_cast<double>(x));
}

}  // namespace flare

#endif  // FLARE_MATHEMATICAL_FUNCTIONS_H_
