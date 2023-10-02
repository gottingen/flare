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


#ifndef FLARE_SIMD_AVX512_H_
#define FLARE_SIMD_AVX512_H_

#include <functional>
#include <type_traits>

#include <flare/simd/common.h>
#include <flare/core/bit_manipulation.h>  // bit_cast

#include <immintrin.h>

namespace flare {
namespace experimental {

namespace simd_abi {

template <int N>
class avx512_fixed_size {};

}  // namespace simd_abi

template <class T>
class simd_mask<T, simd_abi::avx512_fixed_size<8>> {
  __mmask8 m_value;

 public:
  class reference {
    __mmask8& m_mask;
    int m_lane;
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION __mmask8 bit_mask() const {
      return __mmask8(std::int16_t(1 << m_lane));
    }

   public:
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference(__mmask8& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      if (value) {
        m_mask |= bit_mask();
      } else {
        m_mask &= ~bit_mask();
      }
      return *this;
    }
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return (m_mask & bit_mask()) != 0;
    }
  };
  using value_type                                  = bool;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(-std::int16_t(value)) {}
  template <class U>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<U, simd_abi::avx512_fixed_size<8>> const& other)
      : m_value(static_cast<__mmask8>(other)) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(G&& gen) : m_value(false) {
    reference(m_value, int(0)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 0>()));
    reference(m_value, int(1)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 1>()));
    reference(m_value, int(2)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 2>()));
    reference(m_value, int(3)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 3>()));
    reference(m_value, int(4)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 4>()));
    reference(m_value, int(5)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 5>()));
    reference(m_value, int(6)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 6>()));
    reference(m_value, int(7)) =
        static_cast<bool>(gen(std::integral_constant<std::size_t, 7>()));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      __mmask8 const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __mmask8()
      const {
    return m_value;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    auto const bit_mask = __mmask8(std::int16_t(1 << i));
    return (m_value & bit_mask) != 0;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(_kor_mask8(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(_kand_mask8(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    static const __mmask8 true_value(static_cast<__mmask8>(simd_mask(true)));
    return simd_mask(_kxor_mask8(true_value, m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    return m_value == other.m_value;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return m_value != other.m_value;
  }
};

template <>
class simd<std::int32_t, simd_abi::avx512_fixed_size<8>> {
  __m256i m_value;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_epi32(value_type(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m256i const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::uint64_t, abi_type> const& other);
  template <class G,
            std::enable_if_t<
                // basically, can you do { value_type r =
                // gen(std::integral_constant<std::size_t, i>()); }
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(
            _mm256_setr_epi32(gen(std::integral_constant<std::size_t, 0>()),
                              gen(std::integral_constant<std::size_t, 1>()),
                              gen(std::integral_constant<std::size_t, 2>()),
                              gen(std::integral_constant<std::size_t, 3>()),
                              gen(std::integral_constant<std::size_t, 4>()),
                              gen(std::integral_constant<std::size_t, 5>()),
                              gen(std::integral_constant<std::size_t, 6>()),
                              gen(std::integral_constant<std::size_t, 7>()))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_mask_storeu_epi32(ptr, static_cast<__mmask8>(mask_type(true)),
                             m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_mask_loadu_epi32(
        _mm256_set1_epi32(0), static_cast<__mmask8>(mask_type(true)), ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(_mm256_sub_epi32(_mm256_set1_epi32(0), m_value));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_mullo_epi32(static_cast<__m256i>(lhs),
                                   static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd<std::int32_t, simd_abi::avx512_fixed_size<8>>(
        _mm256_add_epi32(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd<std::int32_t, simd_abi::avx512_fixed_size<8>>(
        _mm256_sub_epi32(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmplt_epi32_mask(static_cast<__m256i>(lhs),
                                             static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmplt_epi32_mask(static_cast<__m256i>(rhs),
                                             static_cast<__m256i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmple_epi32_mask(static_cast<__m256i>(lhs),
                                             static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmple_epi32_mask(static_cast<__m256i>(rhs),
                                             static_cast<__m256i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmpeq_epi32_mask(static_cast<__m256i>(lhs),
                                             static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmpneq_epi32_mask(static_cast<__m256i>(lhs),
                                              static_cast<__m256i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, int rhs) noexcept {
    return simd(_mm256_srai_epi32(static_cast<__m256i>(lhs), rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_srav_epi32(static_cast<__m256i>(lhs),
                                  static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, int rhs) noexcept {
    return simd(_mm256_slli_epi32(static_cast<__m256i>(lhs), rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_sllv_epi32(static_cast<__m256i>(lhs),
                                  static_cast<__m256i>(rhs)));
  }
};

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int32_t, simd_abi::avx512_fixed_size<8>>
    abs(simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& a) {
  __m256i const rhs = static_cast<__m256i>(a);
  return simd<std::int32_t, simd_abi::avx512_fixed_size<8>>(
      _mm256_abs_epi32(rhs));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int32_t, simd_abi::avx512_fixed_size<8>>
    condition(simd_mask<std::int32_t, simd_abi::avx512_fixed_size<8>> const& a,
              simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& b,
              simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<std::int32_t, simd_abi::avx512_fixed_size<8>>(
      _mm256_mask_blend_epi32(static_cast<__mmask8>(a), static_cast<__m256i>(c),
                              static_cast<__m256i>(b)));
}

template <>
class simd<std::uint32_t, simd_abi::avx512_fixed_size<8>> {
  __m256i m_value;

 public:
  using value_type = std::uint32_t;
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_epi32(
            flare::bit_cast<std::int32_t>(value_type(value)))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m256i const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& other)
      : m_value(static_cast<__m256i>(other)) {}
  template <class G,
            std::enable_if_t<
                // basically, can you do { value_type r =
                // gen(std::integral_constant<std::size_t, i>()); }
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(
            _mm256_setr_epi32(gen(std::integral_constant<std::size_t, 0>()),
                              gen(std::integral_constant<std::size_t, 1>()),
                              gen(std::integral_constant<std::size_t, 2>()),
                              gen(std::integral_constant<std::size_t, 3>()),
                              gen(std::integral_constant<std::size_t, 4>()),
                              gen(std::integral_constant<std::size_t, 5>()),
                              gen(std::integral_constant<std::size_t, 6>()),
                              gen(std::integral_constant<std::size_t, 7>()))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_mask_storeu_epi32(ptr, static_cast<__mmask8>(mask_type(true)),
                             m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_mask_loadu_epi32(
        _mm256_set1_epi32(0), static_cast<__mmask8>(mask_type(true)), ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_mullo_epi32(static_cast<__m256i>(lhs),
                                   static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_add_epi32(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_sub_epi32(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmplt_epu32_mask(static_cast<__m256i>(lhs),
                                             static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmplt_epu32_mask(static_cast<__m256i>(rhs),
                                             static_cast<__m256i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmple_epu32_mask(static_cast<__m256i>(lhs),
                                             static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmple_epu32_mask(static_cast<__m256i>(rhs),
                                             static_cast<__m256i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmpeq_epu32_mask(static_cast<__m256i>(lhs),
                                             static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmpneq_epu32_mask(static_cast<__m256i>(lhs),
                                              static_cast<__m256i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, int rhs) noexcept {
    return simd(_mm256_srli_epi32(static_cast<__m256i>(lhs), rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_srlv_epi32(static_cast<__m256i>(lhs),
                                  static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, int rhs) noexcept {
    return simd(_mm256_slli_epi32(static_cast<__m256i>(lhs), rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_sllv_epi32(static_cast<__m256i>(lhs),
                                  static_cast<__m256i>(rhs)));
  }
};

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>
    abs(simd<std::uint32_t, simd_abi::avx512_fixed_size<8>> const& a) {
  return a;
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>
    condition(simd_mask<std::uint32_t, simd_abi::avx512_fixed_size<8>> const& a,
              simd<std::uint32_t, simd_abi::avx512_fixed_size<8>> const& b,
              simd<std::uint32_t, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>(
      _mm256_mask_blend_epi32(static_cast<__mmask8>(a), static_cast<__m256i>(c),
                              static_cast<__m256i>(b)));
}

template <>
class simd<std::int64_t, simd_abi::avx512_fixed_size<8>> {
  __m512i m_value;

 public:
  using value_type = std::int64_t;
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm512_set1_epi64(value_type(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& other)
      : m_value(_mm512_cvtepi32_epi64(static_cast<__m256i>(other))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& other);
  template <class G,
            std::enable_if_t<
                // basically, can you do { value_type r =
                // gen(std::integral_constant<std::size_t, i>()); }
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(
            _mm512_setr_epi64(gen(std::integral_constant<std::size_t, 0>()),
                              gen(std::integral_constant<std::size_t, 1>()),
                              gen(std::integral_constant<std::size_t, 2>()),
                              gen(std::integral_constant<std::size_t, 3>()),
                              gen(std::integral_constant<std::size_t, 4>()),
                              gen(std::integral_constant<std::size_t, 5>()),
                              gen(std::integral_constant<std::size_t, 6>()),
                              gen(std::integral_constant<std::size_t, 7>()))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr simd(__m512i const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm512_loadu_si512(ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm512_storeu_si512(ptr, m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m512i()
      const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(_mm512_sub_epi64(_mm512_set1_epi64(0), m_value));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm512_mullo_epi64(static_cast<__m512i>(lhs),
                                   static_cast<__m512i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_add_epi64(static_cast<__m512i>(lhs), static_cast<__m512i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_sub_epi64(static_cast<__m512i>(lhs), static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmplt_epi64_mask(static_cast<__m512i>(lhs),
                                             static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmplt_epi64_mask(static_cast<__m512i>(rhs),
                                             static_cast<__m512i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmple_epi64_mask(static_cast<__m512i>(lhs),
                                             static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmple_epi64_mask(static_cast<__m512i>(rhs),
                                             static_cast<__m512i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmpeq_epi64_mask(static_cast<__m512i>(lhs),
                                             static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmpneq_epi64_mask(static_cast<__m512i>(lhs),
                                              static_cast<__m512i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, int rhs) {
    return simd(_mm512_srai_epi64(static_cast<__m512i>(lhs), rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, simd const& rhs) {
    return simd(_mm512_srav_epi64(static_cast<__m512i>(lhs),
                                  static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, int rhs) {
    return simd(_mm512_slli_epi64(static_cast<__m512i>(lhs), rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, simd const& rhs) {
    return simd(_mm512_sllv_epi64(static_cast<__m512i>(lhs),
                                  static_cast<__m512i>(rhs)));
  }
};

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int64_t, simd_abi::avx512_fixed_size<8>>
    abs(simd<std::int64_t, simd_abi::avx512_fixed_size<8>> const& a) {
  __m512i const rhs = static_cast<__m512i>(a);
  return simd<std::int64_t, simd_abi::avx512_fixed_size<8>>(
      _mm512_abs_epi64(rhs));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int64_t, simd_abi::avx512_fixed_size<8>>
    condition(simd_mask<std::int64_t, simd_abi::avx512_fixed_size<8>> const& a,
              simd<std::int64_t, simd_abi::avx512_fixed_size<8>> const& b,
              simd<std::int64_t, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<std::int64_t, simd_abi::avx512_fixed_size<8>>(
      _mm512_mask_blend_epi64(static_cast<__mmask8>(a), static_cast<__m512i>(c),
                              static_cast<__m512i>(b)));
}

template <>
class simd<std::uint64_t, simd_abi::avx512_fixed_size<8>> {
  __m512i m_value;

 public:
  using value_type = std::uint64_t;
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm512_set1_epi64(
            flare::bit_cast<std::int64_t>(value_type(value)))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr simd(__m512i const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int32_t, abi_type> const& other)
      : m_value(_mm512_cvtepi32_epi64(static_cast<__m256i>(other))) {}
  template <class G,
            std::enable_if_t<
                // basically, can you do { value_type r =
                // gen(std::integral_constant<std::size_t, i>()); }
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(
            _mm512_setr_epi64(gen(std::integral_constant<std::size_t, 0>()),
                              gen(std::integral_constant<std::size_t, 1>()),
                              gen(std::integral_constant<std::size_t, 2>()),
                              gen(std::integral_constant<std::size_t, 3>()),
                              gen(std::integral_constant<std::size_t, 4>()),
                              gen(std::integral_constant<std::size_t, 5>()),
                              gen(std::integral_constant<std::size_t, 6>()),
                              gen(std::integral_constant<std::size_t, 7>()))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int64_t, abi_type> const& other)
      : m_value(static_cast<__m512i>(other)) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm512_loadu_si512(ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm512_storeu_si512(ptr, m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m512i()
      const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm512_mullo_epi64(static_cast<__m512i>(lhs),
                                   static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_add_epi64(static_cast<__m512i>(lhs), static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_sub_epi64(static_cast<__m512i>(lhs), static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, int rhs) noexcept {
    return _mm512_srli_epi64(static_cast<__m512i>(lhs), rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm512_srlv_epi64(static_cast<__m512i>(lhs),
                             static_cast<__m512i>(rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, int rhs) noexcept {
    return _mm512_slli_epi64(static_cast<__m512i>(lhs), rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm512_sllv_epi64(static_cast<__m512i>(lhs),
                             static_cast<__m512i>(rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator&(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm512_and_epi64(static_cast<__m512i>(lhs),
                            static_cast<__m512i>(rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator|(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm512_or_epi64(static_cast<__m512i>(lhs),
                           static_cast<__m512i>(rhs));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmplt_epu64_mask(static_cast<__m512i>(lhs),
                                             static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmplt_epu64_mask(static_cast<__m512i>(rhs),
                                             static_cast<__m512i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmple_epu64_mask(static_cast<__m512i>(lhs),
                                             static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmple_epu64_mask(static_cast<__m512i>(rhs),
                                             static_cast<__m512i>(lhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmpeq_epu64_mask(static_cast<__m512i>(lhs),
                                             static_cast<__m512i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmpneq_epu64_mask(static_cast<__m512i>(lhs),
                                              static_cast<__m512i>(rhs)));
  }
};

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>
    abs(simd<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& a) {
  return a;
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>
    condition(simd_mask<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& a,
              simd<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& b,
              simd<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>(
      _mm512_mask_blend_epi64(static_cast<__mmask8>(a), static_cast<__m512i>(c),
                              static_cast<__m512i>(b)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int32_t, simd_abi::avx512_fixed_size<8>>::simd(
    simd<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& other)
    : m_value(_mm512_cvtepi64_epi32(static_cast<__m512i>(other))) {}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int64_t, simd_abi::avx512_fixed_size<8>>::simd(
    simd<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& other)
    : m_value(static_cast<__m512i>(other)) {}

template <>
class simd<double, simd_abi::avx512_fixed_size<8>> {
  __m512d m_value;

 public:
  using value_type = double;
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm512_set1_pd(value_type(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m512d const& value_in)
      : m_value(value_in) {}
  template <class G,
            std::enable_if_t<
                // basically, can you do { value_type r =
                // gen(std::integral_constant<std::size_t, i>()); }
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(_mm512_setr_pd(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>()),
                               gen(std::integral_constant<std::size_t, 4>()),
                               gen(std::integral_constant<std::size_t, 5>()),
                               gen(std::integral_constant<std::size_t, 6>()),
                               gen(std::integral_constant<std::size_t, 7>()))) {
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm512_loadu_pd(ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm512_storeu_pd(ptr, m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m512d()
      const {
    return m_value;
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(_mm512_sub_pd(_mm512_set1_pd(0.0), m_value));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_mul_pd(static_cast<__m512d>(lhs), static_cast<__m512d>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_div_pd(static_cast<__m512d>(lhs), static_cast<__m512d>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_add_pd(static_cast<__m512d>(lhs), static_cast<__m512d>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm512_sub_pd(static_cast<__m512d>(lhs), static_cast<__m512d>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmp_pd_mask(static_cast<__m512d>(lhs),
                                        static_cast<__m512d>(rhs), _CMP_LT_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmp_pd_mask(static_cast<__m512d>(rhs),
                                        static_cast<__m512d>(lhs), _CMP_GT_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmp_pd_mask(static_cast<__m512d>(lhs),
                                        static_cast<__m512d>(rhs), _CMP_LE_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmp_pd_mask(static_cast<__m512d>(rhs),
                                        static_cast<__m512d>(lhs), _CMP_GE_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmp_pd_mask(static_cast<__m512d>(lhs),
                                        static_cast<__m512d>(rhs), _CMP_EQ_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm512_cmp_pd_mask(
        static_cast<__m512d>(lhs), static_cast<__m512d>(rhs), _CMP_NEQ_OS));
  }
};

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    copysign(simd<double, simd_abi::avx512_fixed_size<8>> const& a,
             simd<double, simd_abi::avx512_fixed_size<8>> const& b) {
  static const __m512i sign_mask = reinterpret_cast<__m512i>(
      static_cast<__m512d>(simd<double, simd_abi::avx512_fixed_size<8>>(-0.0)));
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      reinterpret_cast<__m512d>(_mm512_xor_epi64(
          _mm512_andnot_epi64(
              sign_mask, reinterpret_cast<__m512i>(static_cast<__m512d>(a))),
          _mm512_and_epi64(
              sign_mask, reinterpret_cast<__m512i>(static_cast<__m512d>(b))))));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    abs(simd<double, simd_abi::avx512_fixed_size<8>> const& a) {
  __m512d const rhs = static_cast<__m512d>(a);
#if defined(FLARE_COMPILER_GNU) && (FLARE_COMPILER_GNU < 830)
  return simd<double, simd_abi::avx512_fixed_size<8>>((__m512d)_mm512_and_epi64(
      (__m512i)rhs, _mm512_set1_epi64(0x7fffffffffffffffLL)));
#else
  return simd<double, simd_abi::avx512_fixed_size<8>>(_mm512_abs_pd(rhs));
#endif
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    sqrt(simd<double, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_sqrt_pd(static_cast<__m512d>(a)));
}

#ifdef __INTEL_COMPILER

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    cbrt(simd<double, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_cbrt_pd(static_cast<__m512d>(a)));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    exp(simd<double, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_exp_pd(static_cast<__m512d>(a)));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    log(simd<double, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_log_pd(static_cast<__m512d>(a)));
}

#endif

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    fma(simd<double, simd_abi::avx512_fixed_size<8>> const& a,
        simd<double, simd_abi::avx512_fixed_size<8>> const& b,
        simd<double, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_fmadd_pd(static_cast<__m512d>(a), static_cast<__m512d>(b),
                      static_cast<__m512d>(c)));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    max(simd<double, simd_abi::avx512_fixed_size<8>> const& a,
        simd<double, simd_abi::avx512_fixed_size<8>> const& b) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_max_pd(static_cast<__m512d>(a), static_cast<__m512d>(b)));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    min(simd<double, simd_abi::avx512_fixed_size<8>> const& a,
        simd<double, simd_abi::avx512_fixed_size<8>> const& b) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_min_pd(static_cast<__m512d>(a), static_cast<__m512d>(b)));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx512_fixed_size<8>>
    condition(simd_mask<double, simd_abi::avx512_fixed_size<8>> const& a,
              simd<double, simd_abi::avx512_fixed_size<8>> const& b,
              simd<double, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<double, simd_abi::avx512_fixed_size<8>>(
      _mm512_mask_blend_pd(static_cast<__mmask8>(a), static_cast<__m512d>(c),
                           static_cast<__m512d>(b)));
}

template <>
class simd<float, simd_abi::avx512_fixed_size<8>> {
  __m256 m_value;

 public:
  using value_type = float;
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_ps(value_type(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m256 const& value_in)
      : m_value(value_in) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_FORCEINLINE_FUNCTION simd(G&& gen)
      : m_value(_mm256_setr_ps(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>()),
                               gen(std::integral_constant<std::size_t, 4>()),
                               gen(std::integral_constant<std::size_t, 5>()),
                               gen(std::integral_constant<std::size_t, 6>()),
                               gen(std::integral_constant<std::size_t, 7>()))) {
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_loadu_ps(ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_storeu_ps(ptr, m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256()
      const {
    return m_value;
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(_mm256_sub_ps(_mm256_set1_ps(0.0), m_value));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_mul_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_div_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_add_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_sub_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps_mask(lhs.m_value, rhs.m_value, _CMP_LT_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps_mask(lhs.m_value, rhs.m_value, _CMP_GT_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps_mask(lhs.m_value, rhs.m_value, _CMP_LE_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps_mask(lhs.m_value, rhs.m_value, _CMP_GE_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps_mask(lhs.m_value, rhs.m_value, _CMP_EQ_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps_mask(lhs.m_value, rhs.m_value, _CMP_NEQ_OS));
  }
};

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> copysign(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a,
    simd<float, simd_abi::avx512_fixed_size<8>> const& b) {
  __m256 const sign_mask = _mm256_set1_ps(-0.0);
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_xor_ps(_mm256_andnot_ps(sign_mask, static_cast<__m256>(a)),
                    _mm256_and_ps(sign_mask, static_cast<__m256>(b))));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> abs(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a) {
  __m256 const sign_mask = _mm256_set1_ps(-0.0);
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_andnot_ps(sign_mask, static_cast<__m256>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> sqrt(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_sqrt_ps(static_cast<__m256>(a)));
}

#ifdef __INTEL_COMPILER

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> cbrt(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_cbrt_ps(static_cast<__m256>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> exp(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_exp_ps(static_cast<__m256>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> log(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_log_ps(static_cast<__m256>(a)));
}

#endif

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> fma(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a,
    simd<float, simd_abi::avx512_fixed_size<8>> const& b,
    simd<float, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(_mm256_fmadd_ps(
      static_cast<__m256>(a), static_cast<__m256>(b), static_cast<__m256>(c)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> max(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a,
    simd<float, simd_abi::avx512_fixed_size<8>> const& b) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_max_ps(static_cast<__m256>(a), static_cast<__m256>(b)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> min(
    simd<float, simd_abi::avx512_fixed_size<8>> const& a,
    simd<float, simd_abi::avx512_fixed_size<8>> const& b) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_min_ps(static_cast<__m256>(a), static_cast<__m256>(b)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx512_fixed_size<8>> condition(
    simd_mask<float, simd_abi::avx512_fixed_size<8>> const& a,
    simd<float, simd_abi::avx512_fixed_size<8>> const& b,
    simd<float, simd_abi::avx512_fixed_size<8>> const& c) {
  return simd<float, simd_abi::avx512_fixed_size<8>>(
      _mm256_mask_blend_ps(static_cast<__mmask8>(a), static_cast<__m256>(c),
                           static_cast<__m256>(b)));
}

template <>
class const_where_expression<simd_mask<double, simd_abi::avx512_fixed_size<8>>,
                             simd<double, simd_abi::avx512_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using value_type = simd<double, abi_type>;
  using mask_type  = simd_mask<double, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(double* mem, element_aligned_tag) const {
    _mm512_mask_storeu_pd(mem, static_cast<__mmask8>(m_mask),
                          static_cast<__m512d>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      double* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) const {
    _mm512_mask_i32scatter_pd(mem, static_cast<__mmask8>(m_mask),
                              static_cast<__m256i>(index),
                              static_cast<__m512d>(m_value), 8);
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<double, simd_abi::avx512_fixed_size<8>>,
                       simd<double, simd_abi::avx512_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<double, simd_abi::avx512_fixed_size<8>>,
          simd<double, simd_abi::avx512_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<double, simd_abi::avx512_fixed_size<8>> const& mask_arg,
      simd<double, simd_abi::avx512_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(double const* mem, element_aligned_tag) {
    m_value = value_type(_mm512_mask_loadu_pd(
        _mm512_set1_pd(0.0), static_cast<__mmask8>(m_mask), mem));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      double const* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) {
    m_value = value_type(_mm512_mask_i32gather_pd(
        static_cast<__m512d>(m_value), static_cast<__mmask8>(m_mask),
        static_cast<__m256i>(index), mem, 8));
  }
  template <class U, std::enable_if_t<
                         std::is_convertible_v<
                             U, simd<double, simd_abi::avx512_fixed_size<8>>>,
                         bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<double, simd_abi::avx512_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = simd<double, simd_abi::avx512_fixed_size<8>>(_mm512_mask_blend_pd(
        static_cast<__mmask8>(m_mask), static_cast<__m512d>(m_value),
        static_cast<__m512d>(x_as_value_type)));
  }
};

template <>
class const_where_expression<simd_mask<float, simd_abi::avx512_fixed_size<8>>,
                             simd<float, simd_abi::avx512_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using value_type = simd<float, abi_type>;
  using mask_type  = simd_mask<float, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(float* mem, element_aligned_tag) const {
    _mm256_mask_storeu_ps(mem, static_cast<__mmask8>(m_mask),
                          static_cast<__m256>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      float* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) const {
    _mm256_mask_i32scatter_ps(mem, static_cast<__mmask8>(m_mask),
                              static_cast<__m256i>(index),
                              static_cast<__m256>(m_value), 4);
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<float, simd_abi::avx512_fixed_size<8>>,
                       simd<float, simd_abi::avx512_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<float, simd_abi::avx512_fixed_size<8>>,
          simd<float, simd_abi::avx512_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<float, simd_abi::avx512_fixed_size<8>> const& mask_arg,
      simd<float, simd_abi::avx512_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(float const* mem, element_aligned_tag) {
    m_value = value_type(_mm256_mask_loadu_ps(
        _mm256_set1_ps(0.0), static_cast<__mmask8>(m_mask), mem));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      float const* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) {
    __m256 on   = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    __m256 mask = _mm256_maskz_mov_ps(static_cast<__mmask8>(m_mask), on);
    m_value     = value_type(
        _mm256_mask_i32gather_ps(static_cast<__m256>(m_value), mem,
                                 static_cast<__m256i>(index), mask, 4));
  }
  template <
      class U,
      std::enable_if_t<
          std::is_convertible_v<U, simd<float, simd_abi::avx512_fixed_size<8>>>,
          bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<float, simd_abi::avx512_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = simd<float, simd_abi::avx512_fixed_size<8>>(_mm256_mask_blend_ps(
        static_cast<__mmask8>(m_mask), static_cast<__m256>(m_value),
        static_cast<__m256>(x_as_value_type)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::int32_t, simd_abi::avx512_fixed_size<8>>,
    simd<std::int32_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using value_type = simd<std::int32_t, abi_type>;
  using mask_type  = simd_mask<std::int32_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, element_aligned_tag) const {
    _mm256_mask_storeu_epi32(mem, static_cast<__mmask8>(m_mask),
                             static_cast<__m256i>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::int32_t* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) const {
    _mm256_mask_i32scatter_epi32(mem, static_cast<__mmask8>(m_mask),
                                 static_cast<__m256i>(index),
                                 static_cast<__m256i>(m_value), 4);
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::int32_t, simd_abi::avx512_fixed_size<8>>,
                       simd<std::int32_t, simd_abi::avx512_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<std::int32_t, simd_abi::avx512_fixed_size<8>>,
          simd<std::int32_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<std::int32_t, simd_abi::avx512_fixed_size<8>> const& mask_arg,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, element_aligned_tag) {
    m_value = value_type(_mm256_mask_loadu_epi32(
        _mm256_set1_epi32(0), static_cast<__mmask8>(m_mask), mem));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int32_t const* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) {
    m_value = value_type(_mm256_mmask_i32gather_epi32(
        static_cast<__m256i>(m_value), static_cast<__mmask8>(m_mask),
        static_cast<__m256i>(index), mem, 4));
  }
  template <class U,
            std::enable_if_t<
                std::is_convertible_v<
                    U, simd<std::int32_t, simd_abi::avx512_fixed_size<8>>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::int32_t, simd_abi::avx512_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = simd<std::int32_t, simd_abi::avx512_fixed_size<8>>(
        _mm256_mask_blend_epi32(static_cast<__mmask8>(m_mask),
                                static_cast<__m256i>(m_value),
                                static_cast<__m256i>(x_as_value_type)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::uint32_t, simd_abi::avx512_fixed_size<8>>,
    simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using value_type = simd<std::uint32_t, abi_type>;
  using mask_type  = simd_mask<std::uint32_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::uint32_t* mem, element_aligned_tag) const {
    _mm256_mask_storeu_epi32(mem, static_cast<__mmask8>(m_mask),
                             static_cast<__m256i>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::uint32_t* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) const {
    _mm256_mask_i32scatter_epi32(mem, static_cast<__mmask8>(m_mask),
                                 static_cast<__m256i>(index),
                                 static_cast<__m256i>(m_value), 4);
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::uint32_t, simd_abi::avx512_fixed_size<8>>,
                       simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<std::uint32_t, simd_abi::avx512_fixed_size<8>>,
          simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<std::uint32_t, simd_abi::avx512_fixed_size<8>> const& mask_arg,
      simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::uint32_t const* mem, element_aligned_tag) {
    m_value = value_type(_mm256_mask_loadu_epi32(
        _mm256_set1_epi32(0), static_cast<__mmask8>(m_mask), mem));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::uint32_t const* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) {
    m_value = value_type(_mm256_mmask_i32gather_epi32(
        static_cast<__m256i>(m_value), static_cast<__mmask8>(m_mask),
        static_cast<__m256i>(index), mem, 4));
  }
  template <class U,
            std::enable_if_t<
                std::is_convertible_v<
                    U, simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = simd<std::uint32_t, simd_abi::avx512_fixed_size<8>>(
        _mm256_mask_blend_epi32(static_cast<__mmask8>(m_mask),
                                static_cast<__m256i>(m_value),
                                static_cast<__m256i>(x_as_value_type)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::int64_t, simd_abi::avx512_fixed_size<8>>,
    simd<std::int64_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using value_type = simd<std::int64_t, abi_type>;
  using mask_type  = simd_mask<std::int64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int64_t* mem, element_aligned_tag) const {
    _mm512_mask_storeu_epi64(mem, static_cast<__mmask8>(m_mask),
                             static_cast<__m512i>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::int64_t* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) const {
    _mm512_mask_i32scatter_epi64(mem, static_cast<__mmask8>(m_mask),
                                 static_cast<__m256i>(index),
                                 static_cast<__m512i>(m_value), 8);
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::int64_t, simd_abi::avx512_fixed_size<8>>,
                       simd<std::int64_t, simd_abi::avx512_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<std::int64_t, simd_abi::avx512_fixed_size<8>>,
          simd<std::int64_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<std::int64_t, simd_abi::avx512_fixed_size<8>> const& mask_arg,
      simd<std::int64_t, simd_abi::avx512_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int64_t const* mem, element_aligned_tag) {
    m_value = value_type(_mm512_mask_loadu_epi64(
        _mm512_set1_epi64(0.0), static_cast<__mmask8>(m_mask), mem));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int64_t const* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) {
    m_value = value_type(_mm512_mask_i32gather_epi64(
        static_cast<__m512i>(m_value), static_cast<__mmask8>(m_mask),
        static_cast<__m256i>(index), mem, 8));
  }
  template <class U,
            std::enable_if_t<
                std::is_convertible_v<
                    U, simd<std::int64_t, simd_abi::avx512_fixed_size<8>>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::int64_t, simd_abi::avx512_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = simd<std::int64_t, simd_abi::avx512_fixed_size<8>>(
        _mm512_mask_blend_epi64(static_cast<__mmask8>(m_mask),
                                static_cast<__m512i>(m_value),
                                static_cast<__m512i>(x_as_value_type)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::uint64_t, simd_abi::avx512_fixed_size<8>>,
    simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx512_fixed_size<8>;
  using value_type = simd<std::uint64_t, abi_type>;
  using mask_type  = simd_mask<std::uint64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::uint64_t* mem, element_aligned_tag) const {
    _mm512_mask_storeu_epi64(mem, static_cast<__mmask8>(m_mask),
                             static_cast<__m512i>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::uint64_t* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) const {
    _mm512_mask_i32scatter_epi64(mem, static_cast<__mmask8>(m_mask),
                                 static_cast<__m256i>(index),
                                 static_cast<__m512i>(m_value), 8);
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::uint64_t, simd_abi::avx512_fixed_size<8>>,
                       simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<std::uint64_t, simd_abi::avx512_fixed_size<8>>,
          simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<std::uint64_t, simd_abi::avx512_fixed_size<8>> const& mask_arg,
      simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::uint64_t const* mem, element_aligned_tag) {
    m_value = value_type(_mm512_mask_loadu_epi64(
        _mm512_set1_epi64(0.0), static_cast<__mmask8>(m_mask), mem));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::uint64_t const* mem,
      simd<std::int32_t, simd_abi::avx512_fixed_size<8>> const& index) {
    m_value = value_type(_mm512_mask_i32gather_epi64(
        static_cast<__m512i>(m_value), static_cast<__mmask8>(m_mask),
        static_cast<__m256i>(index), mem, 8));
  }
  template <class U,
            std::enable_if_t<
                std::is_convertible_v<
                    U, simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = simd<std::uint64_t, simd_abi::avx512_fixed_size<8>>(
        _mm512_mask_blend_epi64(static_cast<__mmask8>(m_mask),
                                static_cast<__m512i>(m_value),
                                static_cast<__m512i>(x_as_value_type)));
  }
};

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION std::int32_t hmax(
    const_where_expression<
        simd_mask<std::int32_t, simd_abi::avx512_fixed_size<8>>,
        simd<std::int32_t, simd_abi::avx512_fixed_size<8>>> const& x) {
  return _mm512_mask_reduce_max_epi32(
      static_cast<__mmask8>(x.impl_get_mask()),
      _mm512_castsi256_si512(static_cast<__m256i>(x.impl_get_value())));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION double hmin(
    const_where_expression<simd_mask<double, simd_abi::avx512_fixed_size<8>>,
                           simd<double, simd_abi::avx512_fixed_size<8>>> const&
        x) {
  return _mm512_mask_reduce_min_pd(static_cast<__mmask8>(x.impl_get_mask()),
                                   static_cast<__m512d>(x.impl_get_value()));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION std::int64_t reduce(
    const_where_expression<
        simd_mask<std::int64_t, simd_abi::avx512_fixed_size<8>>,
        simd<std::int64_t, simd_abi::avx512_fixed_size<8>>> const& x,
    std::int64_t, std::plus<>) {
  return _mm512_mask_reduce_add_epi64(static_cast<__mmask8>(x.impl_get_mask()),
                                      static_cast<__m512i>(x.impl_get_value()));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION double reduce(
    const_where_expression<simd_mask<double, simd_abi::avx512_fixed_size<8>>,
                           simd<double, simd_abi::avx512_fixed_size<8>>> const&
        x,
    double, std::plus<>) {
  return _mm512_mask_reduce_add_pd(static_cast<__mmask8>(x.impl_get_mask()),
                                   static_cast<__m512d>(x.impl_get_value()));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_SIMD_AVX512_H_
