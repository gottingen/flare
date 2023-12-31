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


#ifndef FLARE_SIMD_AVX2_H_
#define FLARE_SIMD_AVX2_H_

#include <functional>
#include <type_traits>

#include <flare/simd/common.h>
#include <flare/core/bit_manipulation.h>  // bit_cast

#include <immintrin.h>

namespace flare {

namespace experimental {

namespace simd_abi {

template <int N>
class avx2_fixed_size {};

}  // namespace simd_abi

template <>
class simd_mask<double, simd_abi::avx2_fixed_size<4>> {
  __m256d m_value;

 public:
  class reference {
    __m256d& m_mask;
    int m_lane;
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION __m256d bit_mask() const {
      return _mm256_castsi256_pd(_mm256_setr_epi64x(
          -std::int64_t(m_lane == 0), -std::int64_t(m_lane == 1),
          -std::int64_t(m_lane == 2), -std::int64_t(m_lane == 3)));
    }

   public:
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference(__m256d& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      if (value) {
        m_mask = _mm256_or_pd(bit_mask(), m_mask);
      } else {
        m_mask = _mm256_andnot_pd(bit_mask(), m_mask);
      }
      return *this;
    }
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return (_mm256_movemask_pd(m_mask) & (1 << m_lane)) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(_mm256_castsi256_pd(_mm256_set1_epi64x(-std::int64_t(value)))) {
  }
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      G&& gen) noexcept
      : m_value(_mm256_castsi256_pd(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 3>()))))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& i32_mask);
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      __m256d const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256d()
      const {
    return m_value;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<__m256d&>(m_value), int(i)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(_mm256_or_pd(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(_mm256_and_pd(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<__m256d>(simd_mask(true));
    return simd_mask(_mm256_andnot_pd(m_value, true_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    return _mm256_movemask_pd(m_value) == _mm256_movemask_pd(other.m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd_mask<float, simd_abi::avx2_fixed_size<4>> {
  __m128 m_value;

 public:
  class reference {
    __m128& m_mask;
    int m_lane;
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION __m128 bit_mask() const {
      return _mm_castsi128_ps(_mm_setr_epi32(
          -std::int32_t(m_lane == 0), -std::int32_t(m_lane == 1),
          -std::int32_t(m_lane == 2), -std::int32_t(m_lane == 3)));
    }

   public:
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference(__m128& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      if (value) {
        m_mask = _mm_or_ps(bit_mask(), m_mask);
      } else {
        m_mask = _mm_andnot_ps(bit_mask(), m_mask);
      }
      return *this;
    }
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return (_mm_movemask_ps(m_mask) & (1 << m_lane)) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(_mm_castsi128_ps(_mm_set1_epi32(-std::int32_t(value)))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      G&& gen) noexcept
      : m_value(_mm_castsi128_ps(_mm_setr_epi32(
            -std::int32_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 3>()))))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      __m128 const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m128()
      const {
    return m_value;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<__m128&>(m_value), int(i)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(_mm_or_ps(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(_mm_and_ps(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<__m128>(simd_mask(true));
    return simd_mask(_mm_andnot_ps(m_value, true_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    return _mm_movemask_ps(m_value) == _mm_movemask_ps(other.m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> {
  __m128i m_value;

 public:
  class reference {
    __m128i& m_mask;
    int m_lane;
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION __m128i bit_mask() const {
      return _mm_setr_epi32(
          -std::int32_t(m_lane == 0), -std::int32_t(m_lane == 1),
          -std::int32_t(m_lane == 2), -std::int32_t(m_lane == 3));
    }

   public:
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference(__m128i& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      if (value) {
        m_mask = _mm_or_si128(bit_mask(), m_mask);
      } else {
        m_mask = _mm_andnot_si128(bit_mask(), m_mask);
      }
      return *this;
    }
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return (_mm_movemask_ps(_mm_castsi128_ps(m_mask)) & (1 << m_lane)) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask()                 = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(simd_mask const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(simd_mask&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(_mm_set1_epi32(-std::int32_t(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      __m128i const& value_in)
      : m_value(value_in) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      G&& gen) noexcept
      : m_value(_mm_setr_epi32(
            -std::int32_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 3>())))) {}
  template <class U>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<U, abi_type> const& other) {
    for (std::size_t i = 0; i < size(); ++i) (*this)[i] = other[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m128i()
      const {
    return m_value;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<__m128i&>(m_value), int(i)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(_mm_or_si128(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(_mm_and_si128(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<__m128i>(simd_mask(true));
    return simd_mask(_mm_andnot_si128(m_value, true_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    return _mm_movemask_ps(_mm_castsi128_ps(m_value)) ==
           _mm_movemask_ps(_mm_castsi128_ps(other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> {
  __m256i m_value;

 public:
  class reference {
    __m256i& m_mask;
    int m_lane;
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION __m256i bit_mask() const {
      return _mm256_setr_epi64x(
          -std::int64_t(m_lane == 0), -std::int64_t(m_lane == 1),
          -std::int64_t(m_lane == 2), -std::int64_t(m_lane == 3));
    }

   public:
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference(__m256i& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      if (value) {
        m_mask = _mm256_or_si256(bit_mask(), m_mask);
      } else {
        m_mask = _mm256_andnot_si256(bit_mask(), m_mask);
      }
      return *this;
    }
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return (_mm256_movemask_pd(_mm256_castsi256_pd(m_mask)) &
              (1 << m_lane)) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask()                 = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(simd_mask const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(simd_mask&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(_mm256_set1_epi64x(-std::int64_t(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      __m256i const& value_in)
      : m_value(value_in) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      G&& gen) noexcept
      : m_value(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 3>())))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<std::int32_t, abi_type> const& other)
      : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<__m256i&>(m_value), int(i)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(_mm256_or_si256(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(_mm256_and_si256(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<__m256i>(simd_mask(true));
    return simd_mask(_mm256_andnot_si256(m_value, true_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    return _mm256_movemask_pd(_mm256_castsi256_pd(m_value)) ==
           _mm256_movemask_pd(_mm256_castsi256_pd(other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> {
  __m256i m_value;

 public:
  class reference {
    __m256i& m_mask;
    int m_lane;
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION __m256i bit_mask() const {
      return _mm256_setr_epi64x(
          -std::int64_t(m_lane == 0), -std::int64_t(m_lane == 1),
          -std::int64_t(m_lane == 2), -std::int64_t(m_lane == 3));
    }

   public:
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference(__m256i& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      if (value) {
        m_mask = _mm256_or_si256(bit_mask(), m_mask);
      } else {
        m_mask = _mm256_andnot_si256(bit_mask(), m_mask);
      }
      return *this;
    }
    FLARE_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return (_mm256_movemask_pd(_mm256_castsi256_pd(m_mask)) &
              (1 << m_lane)) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(_mm256_set1_epi64x(-std::int64_t(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<std::int32_t, abi_type> const& other)
      : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      __m256i const& value_in)
      : m_value(value_in) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      G&& gen) noexcept
      : m_value(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 3>())))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<__m256i&>(m_value), int(i)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(_mm256_or_si256(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(_mm256_and_si256(m_value, other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<__m256i>(simd_mask(true));
    return simd_mask(_mm256_andnot_si256(m_value, true_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    return _mm256_movemask_pd(_mm256_castsi256_pd(m_value)) ==
           _mm256_movemask_pd(_mm256_castsi256_pd(other.m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd_mask<double, simd_abi::avx2_fixed_size<4>>::simd_mask(
    simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& i32_mask)
    : m_value(_mm256_castsi256_pd(
          _mm256_cvtepi32_epi64(static_cast<__m128i>(i32_mask)))) {}

template <>
class simd<double, simd_abi::avx2_fixed_size<4>> {
  __m256d m_value;

 public:
  using value_type = double;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_pd(value_type(value))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m256d const& value_in)
      : m_value(value_in) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(_mm256_setr_pd(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>()))) {
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
    m_value = _mm256_loadu_pd(ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_storeu_pd(ptr, m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256d()
      const {
    return m_value;
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(
        _mm256_sub_pd(_mm256_set1_pd(0.0), static_cast<__m256d>(m_value)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_mul_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_div_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_add_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_sub_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(lhs),
                                   static_cast<__m256d>(rhs), _CMP_LT_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(lhs),
                                   static_cast<__m256d>(rhs), _CMP_GT_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(lhs),
                                   static_cast<__m256d>(rhs), _CMP_LE_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(lhs),
                                   static_cast<__m256d>(rhs), _CMP_GE_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(lhs),
                                   static_cast<__m256d>(rhs), _CMP_EQ_OS));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(lhs),
                                   static_cast<__m256d>(rhs), _CMP_NEQ_OS));
  }
};

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> copysign(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b) {
  __m256d const sign_mask = _mm256_set1_pd(-0.0);
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_xor_pd(_mm256_andnot_pd(sign_mask, static_cast<__m256d>(a)),
                    _mm256_and_pd(sign_mask, static_cast<__m256d>(b))));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> abs(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  __m256d const sign_mask = _mm256_set1_pd(-0.0);
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_andnot_pd(sign_mask, static_cast<__m256d>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> sqrt(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_sqrt_pd(static_cast<__m256d>(a)));
}

#ifdef __INTEL_COMPILER

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> cbrt(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_cbrt_pd(static_cast<__m256d>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> exp(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_exp_pd(static_cast<__m256d>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> log(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_log_pd(static_cast<__m256d>(a)));
}

#endif

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> fma(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b,
    simd<double, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_fmadd_pd(static_cast<__m256d>(a), static_cast<__m256d>(b),
                      static_cast<__m256d>(c)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> max(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_max_pd(static_cast<__m256d>(a), static_cast<__m256d>(b)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> min(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_min_pd(static_cast<__m256d>(a), static_cast<__m256d>(b)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> condition(
    simd_mask<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b,
    simd<double, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_blendv_pd(static_cast<__m256d>(c), static_cast<__m256d>(b),
                       static_cast<__m256d>(a)));
}

template <>
class simd<float, simd_abi::avx2_fixed_size<4>> {
  __m128 m_value;

 public:
  using value_type = float;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                         bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm_set1_ps(value_type(value))) {}
  template <typename G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_FORCEINLINE_FUNCTION simd(G&& gen)
      : m_value(_mm_setr_ps(gen(std::integral_constant<std::size_t, 0>()),
                            gen(std::integral_constant<std::size_t, 1>()),
                            gen(std::integral_constant<std::size_t, 2>()),
                            gen(std::integral_constant<std::size_t, 3>()))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m128 const& value_in)
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
    m_value = _mm_loadu_ps(ptr);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm_storeu_ps(ptr, m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m128()
      const {
    return m_value;
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(_mm_sub_ps(_mm_set1_ps(0.0), m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm_mul_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm_div_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm_add_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm_sub_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm_cmplt_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm_cmpgt_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm_cmple_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm_cmpge_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm_cmpeq_ps(lhs.m_value, rhs.m_value));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm_cmpneq_ps(lhs.m_value, rhs.m_value));
  }
};

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> copysign(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a,
    simd<float, simd_abi::avx2_fixed_size<4>> const& b) {
  __m128 const sign_mask = _mm_set1_ps(-0.0);
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_xor_ps(_mm_andnot_ps(sign_mask, static_cast<__m128>(a)),
                 _mm_and_ps(sign_mask, static_cast<__m128>(b))));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> abs(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a) {
  __m128 const sign_mask = _mm_set1_ps(-0.0);
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_andnot_ps(sign_mask, static_cast<__m128>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> sqrt(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_sqrt_ps(static_cast<__m128>(a)));
}

#ifdef __INTEL_COMPILER

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> cbrt(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_cbrt_ps(static_cast<__m128>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> exp(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_exp_ps(static_cast<__m128>(a)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> log(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_log_ps(static_cast<__m128>(a)));
}

#endif

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> fma(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a,
    simd<float, simd_abi::avx2_fixed_size<4>> const& b,
    simd<float, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(_mm_fmadd_ps(
      static_cast<__m128>(a), static_cast<__m128>(b), static_cast<__m128>(c)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> max(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a,
    simd<float, simd_abi::avx2_fixed_size<4>> const& b) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_max_ps(static_cast<__m128>(a), static_cast<__m128>(b)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> min(
    simd<float, simd_abi::avx2_fixed_size<4>> const& a,
    simd<float, simd_abi::avx2_fixed_size<4>> const& b) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(
      _mm_min_ps(static_cast<__m128>(a), static_cast<__m128>(b)));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<float, simd_abi::avx2_fixed_size<4>> condition(
    simd_mask<float, simd_abi::avx2_fixed_size<4>> const& a,
    simd<float, simd_abi::avx2_fixed_size<4>> const& b,
    simd<float, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<float, simd_abi::avx2_fixed_size<4>>(_mm_blendv_ps(
      static_cast<__m128>(c), static_cast<__m128>(b), static_cast<__m128>(a)));
}

template <>
class simd<std::int32_t, simd_abi::avx2_fixed_size<4>> {
  __m128i m_value;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm_set1_epi32(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(_mm_setr_epi32(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>()))) {
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m128i const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::uint64_t, abi_type> const& other);
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm_maskload_epi32(ptr, static_cast<__m128i>(mask_type(true)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm_maskstore_epi32(ptr, static_cast<__m128i>(mask_type(true)), m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m128i()
      const {
    return m_value;
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(
        _mm_cmpeq_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(
        _mm_cmpgt_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(
        _mm_cmplt_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return (lhs < rhs) || (lhs == rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return (lhs > rhs) || (lhs == rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(lhs == rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm_sub_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm_add_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, int rhs) noexcept {
    return simd(_mm_srai_epi32(static_cast<__m128i>(lhs), rhs));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm_srav_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, int rhs) noexcept {
    return simd(_mm_slli_epi32(static_cast<__m128i>(lhs), rhs));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm_sllv_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
  }
};

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int32_t, simd_abi::avx2_fixed_size<4>> abs(
    simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& a) {
  __m128i const rhs = static_cast<__m128i>(a);
  return simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(_mm_abs_epi32(rhs));
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    condition(simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& a,
              simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& b,
              simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(_mm_castps_si128(
      _mm_blendv_ps(_mm_castsi128_ps(static_cast<__m128i>(c)),
                    _mm_castsi128_ps(static_cast<__m128i>(b)),
                    _mm_castsi128_ps(static_cast<__m128i>(a)))));
}

template <>
class simd<std::int64_t, simd_abi::avx2_fixed_size<4>> {
  __m256i m_value;

  static_assert(sizeof(long long) == 8);

 public:
  using value_type = std::int64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_epi64x(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(_mm256_setr_epi64x(
            gen(std::integral_constant<std::size_t, 0>()),
            gen(std::integral_constant<std::size_t, 1>()),
            gen(std::integral_constant<std::size_t, 2>()),
            gen(std::integral_constant<std::size_t, 3>()))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m256i const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(
      simd<std::uint64_t, abi_type> const& other);
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(
      simd<std::int32_t, abi_type> const& other)
      : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                    static_cast<__m256i>(mask_type(true)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr),
                           static_cast<__m256i>(mask_type(true)), m_value);
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(
        _mm256_sub_epi64(_mm256_set1_epi64x(0), static_cast<__m256i>(m_value)));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_sub_epi64(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_add_epi64(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }

  // AVX2 only has eq and gt comparisons for int64
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmpeq_epi64(static_cast<__m256i>(lhs),
                                        static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmpgt_epi64(static_cast<__m256i>(lhs),
                                        static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return rhs > lhs;
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return (lhs < rhs) || (lhs == rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return (lhs > rhs) || (lhs == rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(lhs == rhs);
  }

  // Shift right arithmetic for 64bit packed ints is not availalbe in AVX2
  // [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd(
  //     simd const& lhs, int rhs) noexcept {
  //   return simd(_mm256_srai_epi64(static_cast<__m256i>(lhs), rhs));
  // }

  // [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd(
  //     simd const& lhs, simd const& rhs) noexcept {
  //   return simd(_mm256_srav_epi64(static_cast<__m256i>(lhs),
  //                                 static_cast<__m256i>(rhs))));
  // }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, int rhs) noexcept {
    return simd(_mm256_slli_epi64(static_cast<__m256i>(lhs), rhs));
  }

  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(_mm256_sllv_epi64(static_cast<__m256i>(lhs),
                                  static_cast<__m256i>(rhs)));
  }
};

// Manually computing absolute values, because _mm256_abs_epi64
// is not in AVX2; it's available in AVX512.
[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    abs(simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(
      [&](std::size_t i) { return (a[i] < 0) ? -a[i] : a[i]; });
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    condition(simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& a,
              simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& b,
              simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(_mm256_castpd_si256(
      _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                       _mm256_castsi256_pd(static_cast<__m256i>(b)),
                       _mm256_castsi256_pd(static_cast<__m256i>(a)))));
}

template <>
class simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> {
  __m256i m_value;

 public:
  using value_type = std::uint64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_epi64x(
            flare::bit_cast<std::int64_t>(value_type(value)))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept
      : m_value(_mm256_setr_epi64x(
            gen(std::integral_constant<std::size_t, 0>()),
            gen(std::integral_constant<std::size_t, 1>()),
            gen(std::integral_constant<std::size_t, 2>()),
            gen(std::integral_constant<std::size_t, 3>()))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr simd(__m256i const& value_in)
      : m_value(value_in) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int32_t, abi_type> const& other)
      : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other))) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int64_t, abi_type> const& other)
      : m_value(static_cast<__m256i>(other)) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                    static_cast<__m256i>(mask_type(true)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_add_epi64(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(
        _mm256_sub_epi64(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, int rhs) noexcept {
    return _mm256_srli_epi64(static_cast<__m256i>(lhs), rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator>>(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm256_srlv_epi64(static_cast<__m256i>(lhs),
                             static_cast<__m256i>(rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, int rhs) noexcept {
    return _mm256_slli_epi64(static_cast<__m256i>(lhs), rhs);
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator<<(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm256_sllv_epi64(static_cast<__m256i>(lhs),
                             static_cast<__m256i>(rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator&(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm256_and_si256(static_cast<__m256i>(lhs),
                            static_cast<__m256i>(rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator|(
      simd const& lhs, simd const& rhs) noexcept {
    return _mm256_or_si256(static_cast<__m256i>(lhs),
                           static_cast<__m256i>(rhs));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(_mm256_cmpeq_epi64(static_cast<__m256i>(lhs),
                                        static_cast<__m256i>(rhs)));
  }
  [[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(lhs == rhs);
  }
};

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::simd(
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& other)
    : m_value(static_cast<__m256i>(other)) {}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> abs(
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& a) {
  return a;
}

[[nodiscard]] FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>
    condition(simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& a,
              simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& b,
              simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(_mm256_castpd_si256(
      _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                       _mm256_castsi256_pd(static_cast<__m256i>(b)),
                       _mm256_castsi256_pd(static_cast<__m256i>(a)))));
}

FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::simd(
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& other) {
  for (std::size_t i = 0; i < 4; ++i) {
    (*this)[i] = std::int32_t(other[i]);
  }
}

template <>
class const_where_expression<simd_mask<double, simd_abi::avx2_fixed_size<4>>,
                             simd<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
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
    _mm256_maskstore_pd(mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask)),
                        static_cast<__m256d>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      double* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
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
class where_expression<simd_mask<double, simd_abi::avx2_fixed_size<4>>,
                       simd<double, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          simd_mask<double, simd_abi::avx2_fixed_size<4>>,
          simd<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      simd<double, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(double const* mem, element_aligned_tag) {
    m_value = value_type(_mm256_maskload_pd(
        mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask))));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      double const* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_pd(
        static_cast<__m256d>(m_value), mem, static_cast<__m128i>(index),
        static_cast<__m256d>(m_mask), 8));
  }
  template <class U,
            std::enable_if_t<std::is_convertible_v<
                                 U, simd<double, simd_abi::avx2_fixed_size<4>>>,
                             bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<double, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = simd<double, simd_abi::avx2_fixed_size<4>>(_mm256_blendv_pd(
        static_cast<__m256d>(m_value), static_cast<__m256d>(x_as_value_type),
        static_cast<__m256d>(m_mask)));
  }
};

template <>
class const_where_expression<simd_mask<float, simd_abi::avx2_fixed_size<4>>,
                             simd<float, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
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
    _mm_maskstore_ps(mem, _mm_castps_si128(static_cast<__m128>(m_mask)),
                     static_cast<__m128>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      float* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
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
class where_expression<simd_mask<float, simd_abi::avx2_fixed_size<4>>,
                       simd<float, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          simd_mask<float, simd_abi::avx2_fixed_size<4>>,
          simd<float, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      simd<float, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(float const* mem, element_aligned_tag) {
    m_value = value_type(
        _mm_maskload_ps(mem, _mm_castps_si128(static_cast<__m128>(m_mask))));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      float const* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm_mask_i32gather_ps(static_cast<__m128>(m_value),
                                               mem, static_cast<__m128i>(index),
                                               static_cast<__m128>(m_mask), 4));
  }
  template <class U,
            std::enable_if_t<std::is_convertible_v<
                                 U, simd<float, simd_abi::avx2_fixed_size<4>>>,
                             bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<float, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = simd<float, simd_abi::avx2_fixed_size<4>>(_mm_blendv_ps(
        static_cast<__m128>(m_value), static_cast<__m128>(x_as_value_type),
        static_cast<__m128>(m_mask)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
    simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
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
    _mm_maskstore_epi32(mem, static_cast<__m128i>(m_mask),
                        static_cast<__m128i>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::int32_t* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
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
class where_expression<simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
                       simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
          simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, element_aligned_tag) {
    m_value = value_type(_mm_maskload_epi32(mem, static_cast<__m128i>(m_mask)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int32_t const* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm_mask_i32gather_epi32(
        static_cast<__m128i>(m_value), mem, static_cast<__m128i>(index),
        static_cast<__m128i>(m_mask), 4));
  }
  template <
      class U,
      std::enable_if_t<std::is_convertible_v<
                           U, simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>,
                       bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(_mm_castps_si128(
        _mm_blendv_ps(_mm_castsi128_ps(static_cast<__m128i>(m_value)),
                      _mm_castsi128_ps(static_cast<__m128i>(x_as_value_type)),
                      _mm_castsi128_ps(static_cast<__m128i>(m_mask)))));
  }
};

template <>
class const_where_expression<
    simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>,
    simd<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = simd<std::int64_t, abi_type>;
  using mask_type  = simd_mask<std::int64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      std::int64_t* mem, element_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem),
                           static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::int64_t* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
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
class where_expression<simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>,
                       simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>,
          simd<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      simd<std::int64_t, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(std::int64_t const* mem,
                                                       element_aligned_tag) {
    m_value = value_type(_mm256_maskload_epi64(
        reinterpret_cast<long long const*>(mem), static_cast<__m256i>(m_mask)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int64_t const* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_epi64(
        static_cast<__m256i>(m_value), reinterpret_cast<long long const*>(mem),
        static_cast<__m128i>(index), static_cast<__m256i>(m_mask), 8));
  }
  template <
      class u,
      std::enable_if_t<std::is_convertible_v<
                           u, simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>,
                       bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(u&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>(
            std::forward<u>(x));
    m_value = simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(
        _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(static_cast<__m256i>(m_value)),
            _mm256_castsi256_pd(static_cast<__m256i>(x_as_value_type)),
            _mm256_castsi256_pd(static_cast<__m256i>(m_mask)))));
  }
};

template <>
class const_where_expression<
    simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>,
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = simd<std::uint64_t, abi_type>;
  using mask_type  = simd_mask<std::uint64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      std::uint64_t* mem, element_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem),
                           static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::uint64_t* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
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
class where_expression<simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>,
                       simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>,
          simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(std::uint64_t const* mem,
                                                       element_aligned_tag) {
    m_value = value_type(_mm256_maskload_epi64(
        reinterpret_cast<long long const*>(mem), static_cast<__m256i>(m_mask)));
  }
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::uint64_t const* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_epi64(
        static_cast<__m256i>(m_value), reinterpret_cast<long long const*>(mem),
        static_cast<__m128i>(index), static_cast<__m256i>(m_mask), 8));
  }
  template <class u,
            std::enable_if_t<
                std::is_convertible_v<
                    u, simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>,
                bool> = false>
  FLARE_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(u&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>(
            std::forward<u>(x));
    m_value = simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(
        _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(static_cast<__m256i>(m_value)),
            _mm256_castsi256_pd(static_cast<__m256i>(x_as_value_type)),
            _mm256_castsi256_pd(static_cast<__m256i>(m_mask)))));
  }
};

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_SIMD_AVX2_H_
