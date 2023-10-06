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

#include <flare/core/defines.h>
#ifdef FLARE_ENABLE_LIBQUADMATH

#include <flare/core/common/quad_precision_math.hpp>
#include <flare/core.h>

#include <doctest.h>

namespace {

// FIXME instantiate only once for default host execution space
TEST_CASE("TEST_CATEGORY, quad_precision_reductions") {
  int const n = 100;
  __float128 r;

  flare::parallel_reduce(
      flare::RangePolicy<flare::DefaultHostExecutionSpace>(0, n),
      FLARE_LAMBDA(int i, __float128 &v) { v += static_cast<__float128>(i); },
      r);
  REQUIRE_EQ(r, n * (n - 1) / 2);

  flare::parallel_reduce(
      flare::RangePolicy<flare::DefaultHostExecutionSpace>(0, n),
      FLARE_LAMBDA(int i, __float128 &v) { v += static_cast<__float128>(i); },
      flare::Sum<__float128>(r));
  REQUIRE_EQ(r, n * (n - 1) / 2);

  flare::parallel_reduce(
      flare::RangePolicy<flare::DefaultHostExecutionSpace>(0, n),
      FLARE_LAMBDA(int i, __float128 &v) {
        if (v > static_cast<__float128>(i)) {
          v = static_cast<__float128>(i);
        }
      },
      flare::Min<__float128>(r));
  REQUIRE_EQ(r, 0);

  flare::parallel_reduce(
      flare::RangePolicy<flare::DefaultHostExecutionSpace>(0, n),
      FLARE_LAMBDA(int i, __float128 &v) {
        if (v < static_cast<__float128>(i)) {
          v = static_cast<__float128>(i);
        }
      },
      flare::Max<__float128>(r));
  REQUIRE_EQ(r, n - 1);

  flare::parallel_reduce(
      flare::RangePolicy<flare::DefaultHostExecutionSpace>(1, n),
      FLARE_LAMBDA(int i, __float128 &v) { v *= static_cast<__float128>(i); },
      flare::Prod<__float128>(r));
  REQUIRE_EQ(r, tgammaq(n + 1));  // factorial(n) = tgamma(n+1)
}

TEST_CASE("TEST_CATEGORY, quad_precision_common_math_functions") {
  flare::parallel_for(
      flare::RangePolicy<flare::DefaultHostExecutionSpace>(0, 1),
      FLARE_LAMBDA(int) {
        (void)flare::fabs((__float128)0);
        (void)flare::sqrt((__float128)1);
        (void)flare::exp((__float128)2);
        (void)flare::sin((__float128)3);
        (void)flare::cosh((__float128)4);
      });
}

constexpr bool test_quad_precision_promotion_traits() {
  static_assert(
      std::is_same<__float128, decltype(flare::pow(__float128(1), 2))>::value);
  static_assert(std::is_same<__float128,
                             decltype(flare::hypot(3, __float128(4)))>::value);
  return true;
}

static_assert(test_quad_precision_promotion_traits());

constexpr bool test_quad_precision_math_constants() {
  // compare to mathematical constants defined in libquadmath when available
  // clang-format off
  static_assert(flare::numbers::e_v     <__float128> == M_Eq);
  static_assert(flare::numbers::log2e_v <__float128> == M_LOG2Eq);
  static_assert(flare::numbers::log10e_v<__float128> == M_LOG10Eq);
  static_assert(flare::numbers::pi_v    <__float128> == M_PIq);
#if defined(FLARE_COMPILER_GNU) && (FLARE_COMPILER_GNU >= 930)
  static_assert(flare::numbers::inv_pi_v<__float128> == M_1_PIq);
#endif
  // inv_sqrtpi_v
  static_assert(flare::numbers::ln2_v   <__float128> == M_LN2q);
  static_assert(flare::numbers::ln10_v  <__float128> == M_LN10q);
#if defined(FLARE_COMPILER_GNU) && (FLARE_COMPILER_GNU >= 930)
  static_assert(flare::numbers::sqrt2_v <__float128> == M_SQRT2q);
#endif
  // sqrt3_v
  // inv_sqrt3_v
  // egamma_v
  // phi_v
  // clang-format on
  return true;
}

static_assert(test_quad_precision_math_constants());

}  // namespace

#endif
