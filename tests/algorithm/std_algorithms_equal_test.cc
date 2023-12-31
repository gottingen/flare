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

template <class ViewType>
void test_equal(const ViewType view) {
  auto copy = create_deep_copyable_compatible_clone(view);

  // pass iterators
  REQUIRE(
      KE::equal(exespace(), KE::begin(view), KE::end(view), KE::begin(copy)));
  // pass views
  REQUIRE(KE::equal(exespace(), view, copy));

  // modify copy - make the last element different
  const auto extent = view.extent(0);
  if (extent > 0) {
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);

    // pass const iterators
    REQUIRE_FALSE(KE::equal(exespace(), KE::cbegin(view), KE::cend(view),
                           KE::cbegin(copy)));
    // pass views
    REQUIRE_FALSE(KE::equal("label", exespace(), view, copy));
  }
}

template <class ViewType>
void test_equal_custom_comparator(const ViewType view) {
  using value_t = typename ViewType::value_type;
  const auto p  = CustomEqualityComparator<value_t>();
  auto copy     = create_deep_copyable_compatible_clone(view);

  // pass iterators
  REQUIRE(KE::equal(exespace(), KE::begin(view), KE::end(view),
                        KE::begin(copy), p));
  // pass views
  REQUIRE(KE::equal(exespace(), view, copy, p));

  // modify copy - make the last element different
  const auto extent = view.extent(0);
  if (extent > 0) {
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);

    // pass const iterators
    REQUIRE_FALSE(KE::equal("label", exespace(), KE::cbegin(view),
                           KE::cend(view), KE::cbegin(copy), p));
    // pass views
    REQUIRE_FALSE(KE::equal(exespace(), view, copy, p));
  }
}

template <class ViewType>
void test_equal_4_iterators(const ViewType view) {
  using value_t = typename ViewType::value_type;
  const auto p  = CustomEqualityComparator<value_t>();
  auto copy     = create_deep_copyable_compatible_clone(view);

  // pass iterators
  REQUIRE(KE::equal(exespace(), KE::begin(view), KE::end(view),
                        KE::begin(copy), KE::end(copy)));
  // pass const and non-const iterators, custom comparator
  REQUIRE(KE::equal("label", exespace(), KE::cbegin(view), KE::cend(view),
                        KE::begin(copy), KE::end(copy), p));

  const auto extent = view.extent(0);
  if (extent > 0) {
    // use different length ranges, pass const iterators
    REQUIRE_FALSE(KE::equal(exespace(), KE::cbegin(view), KE::cend(view),
                           KE::cbegin(copy), KE::cend(copy) - 1));

    // modify copy - make the last element different
    KE::fill(exespace(), KE::end(copy) - 1, KE::end(copy), 1);
    // pass const iterators
    REQUIRE_FALSE(KE::equal(exespace(), KE::cbegin(view), KE::cend(view),
                           KE::cbegin(copy), KE::cend(copy)));
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    auto view = create_view<ValueType>(Tag{}, scenario.second, "equal");
    test_equal(view);
    test_equal_custom_comparator(view);
    test_equal_4_iterators(view);
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
