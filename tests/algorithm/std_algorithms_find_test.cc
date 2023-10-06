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
#include <iterator>
#include <algorithm>

namespace Test {
    namespace stdalgos {
        namespace Find {

            namespace KE = flare::experimental;

            template<class ViewType>
            void test_find(const ViewType view) {
                using value_t = typename ViewType::value_type;
                using view_host_space_t = flare::View<value_t *, flare::HostSpace>;

                view_host_space_t expected("count_expected", view.extent(0));
                compare_views(expected, view);
                constexpr value_t find_value = 13;

                // value not found, return last
                REQUIRE_EQ(KE::end(expected),
                           std::find(KE::begin(expected), KE::end(expected), find_value));

                // pass const iterators, returns const iterator
                REQUIRE_EQ(KE::cend(view),
                           KE::find(exespace(), KE::cbegin(view), KE::cend(view), find_value));
                // pass view, returns iterator
                REQUIRE_EQ(KE::end(view), KE::find(exespace(), view, find_value));

                fill_views_inc(view, expected);

                auto std_result =
                        std::find(KE::begin(expected), KE::end(expected), find_value);
                auto distance = std::distance(KE::begin(expected), std_result);

                // pass iterators, returns iterator
                REQUIRE_EQ(KE::begin(view) + distance,
                           KE::find(exespace(), KE::begin(view), KE::end(view), find_value));
                // pass view, returns iterator
                REQUIRE_EQ(KE::begin(view) + distance, KE::find(exespace(), view, find_value));
            }

            template<class ViewType>
            void test_find_if(const ViewType view) {
                using value_t = typename ViewType::value_type;
                using view_host_space_t = flare::View<value_t *, flare::HostSpace>;

                view_host_space_t expected("count_expected", view.extent(0));
                compare_views(expected, view);

                const auto not_equals_zero = NotEqualsZeroFunctor<value_type>();

                // value not found, return last
                REQUIRE_EQ(
                        KE::end(expected),
                        std::find_if(KE::begin(expected), KE::end(expected), not_equals_zero));

                // pass iterators, returns iterator
                REQUIRE_EQ(KE::end(view), KE::find_if(exespace(), KE::begin(view),
                                                      KE::end(view), not_equals_zero));
                // pass view, returns iterator
                REQUIRE_EQ(KE::end(view), KE::find_if(exespace(), view, not_equals_zero));

                fill_views_inc(view, expected);

                constexpr value_t find_value = 13;
                const auto equals_val = EqualsValFunctor<value_type>(find_value);
                auto std_result =
                        std::find_if(KE::begin(expected), KE::end(expected), equals_val);
                auto distance = std::distance(KE::begin(expected), std_result);

                // pass const iterators, returns const iterator
                REQUIRE_EQ(
                        KE::cbegin(view) + distance,
                        KE::find_if(exespace(), KE::cbegin(view), KE::cend(view), equals_val));
                // pass view, returns iterator
                REQUIRE_EQ(KE::begin(view) + distance,
                           KE::find_if(exespace(), view, equals_val));
            }

            template<class ViewType>
            void test_find_if_not(const ViewType view) {
                using value_t = typename ViewType::value_type;
                using view_host_space_t = flare::View<value_t *, flare::HostSpace>;

                view_host_space_t expected("count_expected", view.extent(0));
                compare_views(expected, view);

                const auto not_equals_zero = NotEqualsZeroFunctor<value_type>();

                // first value matches
                REQUIRE_EQ(KE::begin(expected),
                           std::find_if_not(KE::begin(expected), KE::end(expected),
                                            not_equals_zero));

                // pass iterators, returns iterator
                REQUIRE_EQ(KE::begin(view), KE::find_if_not(exespace(), KE::begin(view),
                                                            KE::end(view), not_equals_zero));
                // pass view, returns iterator
                REQUIRE_EQ(KE::begin(view),
                           KE::find_if_not(exespace(), view, not_equals_zero));

                fill_views_inc(view, expected);

                const auto equals_zero = EqualsValFunctor<value_type>(0);
                auto std_result =
                        std::find_if_not(KE::begin(expected), KE::end(expected), equals_zero);
                auto distance = std::distance(KE::begin(expected), std_result);

                // pass const iterators, returns const iterator
                REQUIRE_EQ(KE::cbegin(view) + distance,
                           KE::find_if_not(exespace(), KE::cbegin(view), KE::cend(view),
                                           equals_zero));
                // pass view, returns const iterator
                REQUIRE_EQ(KE::begin(view) + distance,
                           KE::find_if_not(exespace(), view, equals_zero));
            }

            template<class Tag, class ValueType>
            void run_all_scenarios() {
                for (const auto &scenario: default_scenarios) {
                    {
                        auto view = create_view<ValueType>(Tag{}, scenario.second, "find");
                        test_find(view);
                    }
                    {
                        auto view = create_view<ValueType>(Tag{}, scenario.second, "find_if");
                        test_find_if(view);
                    }
                    {
                        auto view = create_view<ValueType>(Tag{}, scenario.second, "find_if_not");
                        test_find_if_not(view);
                    }
                }
            }

            TEST_CASE("std_algorithms_find_test, test") {
                run_all_scenarios<DynamicTag, double>();

                run_all_scenarios<StridedTwoTag, int>();

                run_all_scenarios<StridedThreeTag, unsigned>();
            }

        }  // namespace Find
    }  // namespace stdalgos
}  // namespace Test
