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

#include <doctest.h>
#include <flare/core.h>

template <class T>
FLARE_FUNCTION T *take_address_of(T &arg) {
  return &arg;
}

template <class T>
FLARE_FUNCTION void take_by_value(T) {}

#define DEFINE_MATH_CONSTANT_TRAIT(TRAIT)                     \
  template <class T>                                          \
  struct TRAIT {                                              \
    static constexpr T value = flare::numbers::TRAIT##_v<T>; \
  }

DEFINE_MATH_CONSTANT_TRAIT(e);
DEFINE_MATH_CONSTANT_TRAIT(log2e);
DEFINE_MATH_CONSTANT_TRAIT(log10e);
DEFINE_MATH_CONSTANT_TRAIT(pi);
DEFINE_MATH_CONSTANT_TRAIT(inv_pi);
DEFINE_MATH_CONSTANT_TRAIT(inv_sqrtpi);
DEFINE_MATH_CONSTANT_TRAIT(ln2);
DEFINE_MATH_CONSTANT_TRAIT(ln10);
DEFINE_MATH_CONSTANT_TRAIT(sqrt2);
DEFINE_MATH_CONSTANT_TRAIT(sqrt3);
DEFINE_MATH_CONSTANT_TRAIT(inv_sqrt3);
DEFINE_MATH_CONSTANT_TRAIT(egamma);
DEFINE_MATH_CONSTANT_TRAIT(phi);

template <class Space, class Trait>
struct TestMathematicalConstants {
  using T = std::decay_t<decltype(Trait::value)>;

  TestMathematicalConstants() { run(); }

  void run() const {
    int errors = 0;
    flare::parallel_reduce(flare::RangePolicy<Space, Trait>(0, 1), *this,
                            errors);
    REQUIRE_EQ(errors, 0);
    (void)take_address_of(Trait::value);  // use on host
  }

  FLARE_FUNCTION void operator()(Trait, int, int &) const { use_on_device(); }

  FLARE_FUNCTION void use_on_device() const {
#if defined(FLARE_COMPILER_NVCC) || \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
    take_by_value(Trait::value);
#else
    (void)take_address_of(Trait::value);
#endif
  }
};

#if defined(FLARE_ENABLE_CUDA)
#define TEST_MATH_CONSTANT(TRAIT)                               \
  TEST_CASE("TEST_CATEGORY, mathematical_constants_"#TRAIT) {         \
    TestMathematicalConstants<TEST_EXECSPACE, TRAIT<float>>();  \
    TestMathematicalConstants<TEST_EXECSPACE, TRAIT<double>>(); \
  }
#else
#define TEST_MATH_CONSTANT(TRAIT)                                    \
  TEST(TEST_CATEGORY, mathematical_constants_##TRAIT) {              \
    TestMathematicalConstants<TEST_EXECSPACE, TRAIT<float>>();       \
    TestMathematicalConstants<TEST_EXECSPACE, TRAIT<double>>();      \
    TestMathematicalConstants<TEST_EXECSPACE, TRAIT<long double>>(); \
  }
#endif

TEST_MATH_CONSTANT(e)
TEST_MATH_CONSTANT(log2e)
TEST_MATH_CONSTANT(log10e)
TEST_MATH_CONSTANT(pi)
TEST_MATH_CONSTANT(inv_pi)
TEST_MATH_CONSTANT(inv_sqrtpi)
TEST_MATH_CONSTANT(ln2)
TEST_MATH_CONSTANT(ln10)
TEST_MATH_CONSTANT(sqrt2)
TEST_MATH_CONSTANT(sqrt3)
TEST_MATH_CONSTANT(inv_sqrt3)
TEST_MATH_CONSTANT(egamma)
TEST_MATH_CONSTANT(phi)
