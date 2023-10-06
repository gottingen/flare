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
#include <flare/algorithm/begin_end.h>
#include <flare/algorithm/all_of.h>
#include <flare/algorithm/any_of.h>
#include <flare/algorithm/none_of.h>
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace AllAnyNoneOf {

namespace KE = flare::experimental;

template <class ViewType>
void test_all_of(const ViewType view) {
  using value_t           = typename ViewType::value_type;
  using view_host_space_t = flare::View<value_t*, flare::HostSpace>;
  const auto equals_zero  = EqualsValFunctor<value_t>(0);

  view_host_space_t expected("all_of_expected", view.extent(0));
  compare_views(expected, view);

  // reference result
  REQUIRE(std::all_of(KE::begin(expected), KE::end(expected), equals_zero));

  // pass iterators
  REQUIRE(
      KE::all_of(exespace(), KE::begin(view), KE::end(view), equals_zero));
  // pass view
  REQUIRE(KE::all_of(exespace(), view, equals_zero));

  fill_views_inc(view, expected);

  if (view.extent(0) > 1) {
    // reference result
    REQUIRE_FALSE(
        std::all_of(KE::begin(expected), KE::end(expected), equals_zero));

    // pass const iterators
    REQUIRE_FALSE(
        KE::all_of(exespace(), KE::cbegin(view), KE::cend(view), equals_zero));
    // pass view
    REQUIRE_FALSE(KE::all_of("label", exespace(), view, equals_zero));
  }
}

template <class ViewType>
void test_any_of(const ViewType view) {
  using value_t              = typename ViewType::value_type;
  using view_host_space_t    = flare::View<value_t*, flare::HostSpace>;
  const auto not_equals_zero = NotEqualsZeroFunctor<value_t>();

  view_host_space_t expected("any_of_expected", view.extent(0));
  compare_views(expected, view);

  // reference result
  REQUIRE_FALSE(
      std::any_of(KE::begin(expected), KE::end(expected), not_equals_zero));

  // pass iterators
  REQUIRE_FALSE(
      KE::any_of(exespace(), KE::begin(view), KE::end(view), not_equals_zero));
  // pass view
  REQUIRE_FALSE(KE::any_of(exespace(), view, not_equals_zero));

  fill_views_inc(view, expected);

  if (view.extent(0) > 1) {
    // reference result
    REQUIRE(
        std::any_of(KE::begin(expected), KE::end(expected), not_equals_zero));

    // pass const iterators
    REQUIRE(KE::any_of(exespace(), KE::cbegin(view), KE::cend(view),
                           not_equals_zero));
    // pass view
    REQUIRE(KE::any_of("label", exespace(), view, not_equals_zero));
  }
}

template <class ViewType>
void test_none_of(const ViewType view) {
  using value_t           = typename ViewType::value_type;
  using view_host_space_t = flare::View<value_t*, flare::HostSpace>;
  const auto is_positive  = IsPositiveFunctor<value_t>();

  view_host_space_t expected("none_of_expected", view.extent(0));
  compare_views(expected, view);

  // reference result
  REQUIRE(
      std::none_of(KE::begin(expected), KE::end(expected), is_positive));

  // pass iterators
  REQUIRE(
      KE::none_of(exespace(), KE::begin(view), KE::end(view), is_positive));
  // pass view
  REQUIRE(KE::none_of(exespace(), view, is_positive));

  fill_views_inc(view, expected);

  if (view.extent(0) > 1) {
    // reference result
    REQUIRE_FALSE(
        std::none_of(KE::begin(expected), KE::end(expected), is_positive));

    // pass const iterators
    REQUIRE_FALSE(
        KE::none_of(exespace(), KE::cbegin(view), KE::cend(view), is_positive));
    // pass view
    REQUIRE_FALSE(KE::none_of("label", exespace(), view, is_positive));
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    {
      auto view = create_view<ValueType>(Tag{}, scenario.second, "all_of");
      test_all_of(view);
    }
    {
      auto view = create_view<ValueType>(Tag{}, scenario.second, "any_of");
      test_any_of(view);
    }
    {
      auto view = create_view<ValueType>(Tag{}, scenario.second, "none_of");
      test_none_of(view);
    }
  }
}

TEST_CASE("std_algorithms_all_any_none_of_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace AllAnyNoneOf
}  // namespace stdalgos
}  // namespace Test
