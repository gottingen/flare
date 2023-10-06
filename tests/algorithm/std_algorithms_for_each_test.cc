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
namespace ForEach {

namespace KE = flare::experimental;

template <class ViewType>
void test_for_each(const ViewType view) {
  using value_t           = typename ViewType::value_type;
  using view_host_space_t = flare::View<value_t*, flare::HostSpace>;

  view_host_space_t expected("for_each_expected", view.extent(0));
  compare_views(expected, view);

  const auto mod_functor = IncrementElementWiseFunctor<value_t>();

  // pass view, functor takes non-const ref
  KE::for_each("label", exespace(), view, mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), mod_functor);
  compare_views(expected, view);

  // pass iterators, functor takes non-const ref
  KE::for_each(exespace(), KE::begin(view), KE::end(view), mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), mod_functor);
  compare_views(expected, view);

  const auto non_mod_functor = NoOpNonMutableFunctor<value_t>();

  // pass view, functor takes const ref
  KE::for_each(exespace(), view, non_mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), non_mod_functor);
  compare_views(expected, view);

  // pass const iterators, functor takes const ref
  KE::for_each(exespace(), KE::cbegin(view), KE::cend(view), non_mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), non_mod_functor);
  compare_views(expected, view);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
  const auto mod_lambda = FLARE_LAMBDA(value_t & i) { ++i; };

  // pass view, lambda takes non-const ref
  KE::for_each(exespace(), view, mod_lambda);
  std::for_each(KE::begin(expected), KE::end(expected), mod_lambda);
  compare_views(expected, view);

  // pass iterators, lambda takes non-const ref
  KE::for_each(exespace(), KE::begin(view), KE::end(view), mod_lambda);
  std::for_each(KE::begin(expected), KE::end(expected), mod_lambda);
  compare_views(expected, view);

  const auto non_mod_lambda = FLARE_LAMBDA(const value_t& i) { (void)i; };

  // pass view, lambda takes const ref
  KE::for_each(exespace(), view, non_mod_lambda);
  std::for_each(KE::cbegin(expected), KE::cend(expected), non_mod_lambda);
  compare_views(expected, view);

  // pass const iterators, lambda takes const ref
  KE::for_each(exespace(), KE::cbegin(view), KE::cend(view), non_mod_lambda);
  std::for_each(KE::cbegin(expected), KE::cend(expected), non_mod_lambda);
  compare_views(expected, view);
#endif
}

// std::for_each_n is C++17, so we cannot compare results directly
template <class ViewType>
void test_for_each_n(const ViewType view) {
  using value_t       = typename ViewType::value_type;
  const std::size_t n = view.extent(0);

  const auto non_mod_functor = NoOpNonMutableFunctor<value_t>();

  // pass const iterators, functor takes const ref
  REQUIRE_EQ(KE::cbegin(view) + n,
            KE::for_each_n(exespace(), KE::cbegin(view), n, non_mod_functor));
  verify_values(value_t{0}, view);

  // pass view, functor takes const ref
  REQUIRE_EQ(KE::begin(view) + n,
            KE::for_each_n(exespace(), view, n, non_mod_functor));
  verify_values(value_t{0}, view);

  // pass iterators, functor takes non-const ref
  const auto mod_functor = IncrementElementWiseFunctor<value_t>();
  REQUIRE_EQ(KE::begin(view) + n,
            KE::for_each_n(exespace(), KE::begin(view), n, mod_functor));
  verify_values(value_t{1}, view);

  // pass view, functor takes non-const ref
  REQUIRE_EQ(KE::begin(view) + n,
            KE::for_each_n("label", exespace(), view, n, mod_functor));
  verify_values(value_t{2}, view);
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    {
      auto view = create_view<ValueType>(Tag{}, scenario.second, "for_each");
      test_for_each(view);
    }
    {
      auto view = create_view<ValueType>(Tag{}, scenario.second, "for_each_n");
      test_for_each_n(view);
    }
  }
}

TEST_CASE("std_algorithms_for_each_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace ForEach
}  // namespace stdalgos
}  // namespace Test
