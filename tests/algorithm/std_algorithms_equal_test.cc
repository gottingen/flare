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

#include <std_algorithms_common_test.h>
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace Equal {

namespace KE = flare::experimental;

template <class TensorType>
void test_equal(const TensorType tensor) {
  auto copy = create_deep_copyable_compatible_clone(tensor);

  // pass iterators
  REQUIRE(
      KE::equal(exespace(), KE::begin(tensor), KE::end(tensor), KE::begin(copy)));
  // pass tensors
  REQUIRE(KE::equal(exespace(), tensor, copy));

  // modify copy - make the last element different
  const auto extent = tensor.extent(0);
  if (extent > 0) {
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);

    // pass const iterators
    REQUIRE_FALSE(KE::equal(exespace(), KE::cbegin(tensor), KE::cend(tensor),
                           KE::cbegin(copy)));
    // pass tensors
    REQUIRE_FALSE(KE::equal("label", exespace(), tensor, copy));
  }
}

template <class TensorType>
void test_equal_custom_comparator(const TensorType tensor) {
  using value_t = typename TensorType::value_type;
  const auto p  = CustomEqualityComparator<value_t>();
  auto copy     = create_deep_copyable_compatible_clone(tensor);

  // pass iterators
  REQUIRE(KE::equal(exespace(), KE::begin(tensor), KE::end(tensor),
                        KE::begin(copy), p));
  // pass tensors
  REQUIRE(KE::equal(exespace(), tensor, copy, p));

  // modify copy - make the last element different
  const auto extent = tensor.extent(0);
  if (extent > 0) {
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);

    // pass const iterators
    REQUIRE_FALSE(KE::equal("label", exespace(), KE::cbegin(tensor),
                           KE::cend(tensor), KE::cbegin(copy), p));
    // pass tensors
    REQUIRE_FALSE(KE::equal(exespace(), tensor, copy, p));
  }
}

template <class TensorType>
void test_equal_4_iterators(const TensorType tensor) {
  using value_t = typename TensorType::value_type;
  const auto p  = CustomEqualityComparator<value_t>();
  auto copy     = create_deep_copyable_compatible_clone(tensor);

  // pass iterators
  REQUIRE(KE::equal(exespace(), KE::begin(tensor), KE::end(tensor),
                        KE::begin(copy), KE::end(copy)));
  // pass const and non-const iterators, custom comparator
  REQUIRE(KE::equal("label", exespace(), KE::cbegin(tensor), KE::cend(tensor),
                        KE::begin(copy), KE::end(copy), p));

  const auto extent = tensor.extent(0);
  if (extent > 0) {
    // use different length ranges, pass const iterators
    REQUIRE_FALSE(KE::equal(exespace(), KE::cbegin(tensor), KE::cend(tensor),
                           KE::cbegin(copy), KE::cend(copy) - 1));

    // modify copy - make the last element different
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);
    // pass const iterators
    REQUIRE_FALSE(KE::equal(exespace(), KE::cbegin(tensor), KE::cend(tensor),
                           KE::cbegin(copy), KE::cend(copy)));
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "equal");
    test_equal(tensor);
    test_equal_custom_comparator(tensor);
    test_equal_4_iterators(tensor);
  }
}

TEST_CASE("std_algorithms_equal_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace Equal
}  // namespace stdalgos
}  // namespace Test
