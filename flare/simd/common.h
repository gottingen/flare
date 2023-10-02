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


#ifndef FLARE_SIMD_COMMON_H_
#define FLARE_SIMD_COMMON_H_

#include <cmath>
#include <cstring>

#include <flare/core.h>

namespace flare {

namespace experimental {

template <class T, class Abi>
class simd;

template <class T, class Abi>
class simd_mask;

struct element_aligned_tag {};

// class template declarations for const_where_expression and where_expression

template <class M, class T>
class const_where_expression {
 protected:
  T& m_value;
  M const& m_mask;

 public:
  const_where_expression(M const& mask_arg, T const& value_arg)
      : m_value(const_cast<T&>(value_arg)), m_mask(mask_arg) {}
  FLARE_FORCEINLINE_FUNCTION T const& value() const { return this->m_value; }
};

template <class M, class T>
class where_expression : public const_where_expression<M, T> {
  using base_type = const_where_expression<M, T>;

 public:
  where_expression(M const& mask_arg, T& value_arg)
      : base_type(mask_arg, value_arg) {}
  FLARE_FORCEINLINE_FUNCTION T& value() { return this->m_value; }
};

// specializations of where expression templates for the case when the
// mask type is bool, to allow generic code to use where() on both
// SIMD types and non-SIMD builtin arithmetic types

template <class T>
class const_where_expression<bool, T> {
 protected:
  T& m_value;
  bool m_mask;

 public:
  FLARE_FORCEINLINE_FUNCTION
  const_where_expression(bool mask_arg, T const& value_arg)
      : m_value(const_cast<T&>(value_arg)), m_mask(mask_arg) {}
  FLARE_FORCEINLINE_FUNCTION T const& value() const { return this->m_value; }
};

template <class T>
class where_expression<bool, T> : public const_where_expression<bool, T> {
  using base_type = const_where_expression<bool, T>;

 public:
  FLARE_FORCEINLINE_FUNCTION
  where_expression(bool mask_arg, T& value_arg)
      : base_type(mask_arg, value_arg) {}
  FLARE_FORCEINLINE_FUNCTION T& value() { return this->m_value; }
  template <class U,
            std::enable_if_t<std::is_convertible_v<U, T>, bool> = false>
  FLARE_FORCEINLINE_FUNCTION void operator=(U const& x) {
    if (this->m_mask) this->m_value = x;
  }
};

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(typename simd<T, Abi>::mask_type const& mask, simd<T, Abi>& value) {
  return where_expression(mask, value);
}

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    const_where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(typename simd<T, Abi>::mask_type const& mask,
          simd<T, Abi> const& value) {
  return const_where_expression(mask, value);
}

template <class T>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION where_expression<bool, T> where(
    bool mask, T& value) {
  return where_expression(mask, value);
}

template <class T>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION const_where_expression<bool, T> where(
    bool mask, T const& value) {
  return const_where_expression(mask, value);
}

// fallback simd multiplication using generator constructor
// At the time of this writing, this fallback is only used
// to multiply vectors of 64-bit signed integers for the AVX2 backend

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd<T, Abi> operator*(
    simd<T, Abi> const& lhs, simd<T, Abi> const& rhs) {
  return simd<T, Abi>([&](std::size_t i) { return lhs[i] * rhs[i]; });
}

// fallback simd shift using generator constructor
// At the time of this edit, only the fallback for shift vectors of
// 64-bit signed integers for the AVX2 backend is used

template <typename T, typename Abi,
          typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd<T, Abi> operator>>(
    simd<T, Abi> const& lhs, int rhs) {
  return simd<T, Abi>([&](std::size_t i) { return lhs[i] >> rhs; });
}

template <typename T, typename Abi,
          typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd<T, Abi> operator<<(
    simd<T, Abi> const& lhs, int rhs) {
  return simd<T, Abi>([&](std::size_t i) { return lhs[i] << rhs; });
}

template <typename T, typename Abi,
          typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd<T, Abi> operator>>(
    simd<T, Abi> const& lhs, simd<T, Abi> const& rhs) {
  return simd<T, Abi>([&](std::size_t i) { return lhs[i] >> rhs[i]; });
}

template <typename T, typename Abi,
          typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd<T, Abi> operator<<(
    simd<T, Abi> const& lhs, simd<T, Abi> const& rhs) {
  return simd<T, Abi>([&](std::size_t i) { return lhs[i] << rhs[i]; });
}

// The code below provides:
// operator@(simd<T, Abi>, Arithmetic)
// operator@(Arithmetic, simd<T, Abi>)
// operator@=(simd<T, Abi>&, U&&)
// operator@=(where_expression<M, T>&, U&&)

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator+(
    experimental::simd<T, Abi> const& lhs, U rhs) {
  using result_member = decltype(lhs[0] + rhs);
  return experimental::simd<result_member, Abi>(lhs) +
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator+(
    U lhs, experimental::simd<T, Abi> const& rhs) {
  using result_member = decltype(lhs + rhs[0]);
  return experimental::simd<result_member, Abi>(lhs) +
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi>
FLARE_FORCEINLINE_FUNCTION simd<T, Abi>& operator+=(simd<T, Abi>& lhs,
                                                     U&& rhs) {
  lhs = lhs + std::forward<U>(rhs);
  return lhs;
}

template <class M, class T, class U>
FLARE_FORCEINLINE_FUNCTION where_expression<M, T>& operator+=(
    where_expression<M, T>& lhs, U&& rhs) {
  lhs = lhs.value() + std::forward<U>(rhs);
  return lhs;
}

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator-(
    experimental::simd<T, Abi> const& lhs, U rhs) {
  using result_member = decltype(lhs[0] - rhs);
  return experimental::simd<result_member, Abi>(lhs) -
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator-(
    U lhs, experimental::simd<T, Abi> const& rhs) {
  using result_member = decltype(lhs - rhs[0]);
  return experimental::simd<result_member, Abi>(lhs) -
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi>
FLARE_FORCEINLINE_FUNCTION simd<T, Abi>& operator-=(simd<T, Abi>& lhs,
                                                     U&& rhs) {
  lhs = lhs - std::forward<U>(rhs);
  return lhs;
}

template <class M, class T, class U>
FLARE_FORCEINLINE_FUNCTION where_expression<M, T>& operator-=(
    where_expression<M, T>& lhs, U&& rhs) {
  lhs = lhs.value() - std::forward<U>(rhs);
  return lhs;
}

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator*(
    experimental::simd<T, Abi> const& lhs, U rhs) {
  using result_member = decltype(lhs[0] * rhs);
  return experimental::simd<result_member, Abi>(lhs) *
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator*(
    U lhs, experimental::simd<T, Abi> const& rhs) {
  using result_member = decltype(lhs * rhs[0]);
  return experimental::simd<result_member, Abi>(lhs) *
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi>
FLARE_FORCEINLINE_FUNCTION simd<T, Abi>& operator*=(simd<T, Abi>& lhs,
                                                     U&& rhs) {
  lhs = lhs * std::forward<U>(rhs);
  return lhs;
}

template <class M, class T, class U>
FLARE_FORCEINLINE_FUNCTION where_expression<M, T>& operator*=(
    where_expression<M, T>& lhs, U&& rhs) {
  lhs = lhs.value() * std::forward<U>(rhs);
  return lhs;
}

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator/(
    experimental::simd<T, Abi> const& lhs, U rhs) {
  using result_member = decltype(lhs[0] / rhs);
  return experimental::simd<result_member, Abi>(lhs) /
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi,
          std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION auto operator/(
    U lhs, experimental::simd<T, Abi> const& rhs) {
  using result_member = decltype(lhs / rhs[0]);
  return experimental::simd<result_member, Abi>(lhs) /
         experimental::simd<result_member, Abi>(rhs);
}

template <class T, class U, class Abi>
FLARE_FORCEINLINE_FUNCTION simd<T, Abi>& operator/=(simd<T, Abi>& lhs,
                                                     U&& rhs) {
  lhs = lhs / std::forward<U>(rhs);
  return lhs;
}

template <class M, class T, class U>
FLARE_FORCEINLINE_FUNCTION where_expression<M, T>& operator/=(
    where_expression<M, T>& lhs, U&& rhs) {
  lhs = lhs.value() / std::forward<U>(rhs);
  return lhs;
}

// implement mask reductions for type bool to allow generic code to accept
// both simd<double, Abi> and just double

[[nodiscard]] FLARE_FORCEINLINE_FUNCTION constexpr bool all_of(bool a) {
  return a;
}

[[nodiscard]] FLARE_FORCEINLINE_FUNCTION constexpr bool any_of(bool a) {
  return a;
}

[[nodiscard]] FLARE_FORCEINLINE_FUNCTION constexpr bool none_of(bool a) {
  return !a;
}

// fallback implementations of reductions across simd_mask:

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool all_of(
    simd_mask<T, Abi> const& a) {
  return a == simd_mask<T, Abi>(true);
}

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool any_of(
    simd_mask<T, Abi> const& a) {
  return a != simd_mask<T, Abi>(false);
}

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool none_of(
    simd_mask<T, Abi> const& a) {
  return a == simd_mask<T, Abi>(false);
}

template <typename T, typename Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION T
hmin(const_where_expression<simd_mask<T, Abi>, simd<T, Abi>> const& x) {
  auto const& v = x.impl_get_value();
  auto const& m = x.impl_get_mask();
  auto result   = flare::reduction_identity<T>::min();
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (m[i]) result = flare::min(result, v[i]);
  }
  return result;
}

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION T
hmax(const_where_expression<simd_mask<T, Abi>, simd<T, Abi>> const& x) {
  auto const& v = x.impl_get_value();
  auto const& m = x.impl_get_mask();
  auto result   = flare::reduction_identity<T>::max();
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (m[i]) result = flare::max(result, v[i]);
  }
  return result;
}

template <class T, class Abi>
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION T
reduce(const_where_expression<simd_mask<T, Abi>, simd<T, Abi>> const& x, T,
       std::plus<>) {
  auto const& v = x.impl_get_value();
  auto const& m = x.impl_get_mask();
  auto result   = flare::reduction_identity<T>::sum();
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (m[i]) result += v[i];
  }
  return result;
}

}  // namespace experimental

template <class T, class Abi>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION experimental::simd<T, Abi> min(
    experimental::simd<T, Abi> const& a, experimental::simd<T, Abi> const& b) {
  experimental::simd<T, Abi> result;
  for (std::size_t i = 0; i < experimental::simd<T, Abi>::size(); ++i) {
    result[i] = flare::min(a[i], b[i]);
  }
  return result;
}

template <class T, class Abi>
[[nodiscard]] FLARE_FORCEINLINE_FUNCTION experimental::simd<T, Abi> max(
    experimental::simd<T, Abi> const& a, experimental::simd<T, Abi> const& b) {
  experimental::simd<T, Abi> result;
  for (std::size_t i = 0; i < experimental::simd<T, Abi>::size(); ++i) {
    result[i] = flare::max(a[i], b[i]);
  }
  return result;
}

// fallback implementations of <cmath> functions.
// individual Abi types may provide overloads with more efficient
// implementations.
// These are not in the Experimental namespace because their double
// overloads are not either

#define FLARE_IMPL_SIMD_UNARY_FUNCTION(FUNC)                               \
  template <class Abi>                                                      \
  [[nodiscard]] FLARE_FORCEINLINE_FUNCTION experimental::simd<double, Abi> \
  FUNC(experimental::simd<double, Abi> const& a) {                          \
    experimental::simd<double, Abi> result;                                 \
    for (std::size_t i = 0; i < experimental::simd<double, Abi>::size();    \
         ++i) {                                                             \
      result[i] = flare::FUNC(a[i]);                                       \
    }                                                                       \
    return result;                                                          \
  }

FLARE_IMPL_SIMD_UNARY_FUNCTION(abs)
FLARE_IMPL_SIMD_UNARY_FUNCTION(exp)
FLARE_IMPL_SIMD_UNARY_FUNCTION(exp2)
FLARE_IMPL_SIMD_UNARY_FUNCTION(log)
FLARE_IMPL_SIMD_UNARY_FUNCTION(log10)
FLARE_IMPL_SIMD_UNARY_FUNCTION(log2)
FLARE_IMPL_SIMD_UNARY_FUNCTION(sqrt)
FLARE_IMPL_SIMD_UNARY_FUNCTION(cbrt)
FLARE_IMPL_SIMD_UNARY_FUNCTION(sin)
FLARE_IMPL_SIMD_UNARY_FUNCTION(cos)
FLARE_IMPL_SIMD_UNARY_FUNCTION(tan)
FLARE_IMPL_SIMD_UNARY_FUNCTION(asin)
FLARE_IMPL_SIMD_UNARY_FUNCTION(acos)
FLARE_IMPL_SIMD_UNARY_FUNCTION(atan)
FLARE_IMPL_SIMD_UNARY_FUNCTION(sinh)
FLARE_IMPL_SIMD_UNARY_FUNCTION(cosh)
FLARE_IMPL_SIMD_UNARY_FUNCTION(tanh)
FLARE_IMPL_SIMD_UNARY_FUNCTION(asinh)
FLARE_IMPL_SIMD_UNARY_FUNCTION(acosh)
FLARE_IMPL_SIMD_UNARY_FUNCTION(atanh)
FLARE_IMPL_SIMD_UNARY_FUNCTION(erf)
FLARE_IMPL_SIMD_UNARY_FUNCTION(erfc)
FLARE_IMPL_SIMD_UNARY_FUNCTION(tgamma)
FLARE_IMPL_SIMD_UNARY_FUNCTION(lgamma)

#define FLARE_IMPL_SIMD_BINARY_FUNCTION(FUNC)                              \
  template <class Abi>                                                      \
  [[nodiscard]] FLARE_FORCEINLINE_FUNCTION experimental::simd<double, Abi> \
  FUNC(experimental::simd<double, Abi> const& a,                            \
       experimental::simd<double, Abi> const& b) {                          \
    experimental::simd<double, Abi> result;                                 \
    for (std::size_t i = 0; i < experimental::simd<double, Abi>::size();    \
         ++i) {                                                             \
      result[i] = flare::FUNC(a[i], b[i]);                                 \
    }                                                                       \
    return result;                                                          \
  }

FLARE_IMPL_SIMD_BINARY_FUNCTION(pow)
FLARE_IMPL_SIMD_BINARY_FUNCTION(hypot)
FLARE_IMPL_SIMD_BINARY_FUNCTION(atan2)
FLARE_IMPL_SIMD_BINARY_FUNCTION(copysign)

#define FLARE_IMPL_SIMD_TERNARY_FUNCTION(FUNC)                             \
  template <class Abi>                                                      \
  [[nodiscard]] FLARE_FORCEINLINE_FUNCTION experimental::simd<double, Abi> \
  FUNC(experimental::simd<double, Abi> const& a,                            \
       experimental::simd<double, Abi> const& b,                            \
       experimental::simd<double, Abi> const& c) {                          \
    experimental::simd<double, Abi> result;                                 \
    for (std::size_t i = 0; i < experimental::simd<double, Abi>::size();    \
         ++i) {                                                             \
      result[i] = flare::FUNC(a[i], b[i], c[i]);                           \
    }                                                                       \
    return result;                                                          \
  }

FLARE_IMPL_SIMD_TERNARY_FUNCTION(fma)
FLARE_IMPL_SIMD_TERNARY_FUNCTION(hypot)

}  // namespace flare

#endif  // FLARE_SIMD_COMMON_H_
