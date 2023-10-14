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

            template<class TensorType>
            void test_find(const TensorType tensor) {
                using value_t = typename TensorType::value_type;
                using tensor_host_space_t = flare::Tensor<value_t *, flare::HostSpace>;

                tensor_host_space_t expected("count_expected", tensor.extent(0));
                compare_tensors(expected, tensor);
                constexpr value_t find_value = 13;

                // value not found, return last
                REQUIRE_EQ(KE::end(expected),
                           std::find(KE::begin(expected), KE::end(expected), find_value));

                // pass const iterators, returns const iterator
                REQUIRE_EQ(KE::cend(tensor),
                           KE::find(exespace(), KE::cbegin(tensor), KE::cend(tensor), find_value));
                // pass tensor, returns iterator
                REQUIRE_EQ(KE::end(tensor), KE::find(exespace(), tensor, find_value));

                fill_tensors_inc(tensor, expected);

                auto std_result =
                        std::find(KE::begin(expected), KE::end(expected), find_value);
                auto distance = std::distance(KE::begin(expected), std_result);

                // pass iterators, returns iterator
                REQUIRE_EQ(KE::begin(tensor) + distance,
                           KE::find(exespace(), KE::begin(tensor), KE::end(tensor), find_value));
                // pass tensor, returns iterator
                REQUIRE_EQ(KE::begin(tensor) + distance, KE::find(exespace(), tensor, find_value));
            }

            template<class TensorType>
            void test_find_if(const TensorType tensor) {
                using value_t = typename TensorType::value_type;
                using tensor_host_space_t = flare::Tensor<value_t *, flare::HostSpace>;

                tensor_host_space_t expected("count_expected", tensor.extent(0));
                compare_tensors(expected, tensor);

                const auto not_equals_zero = NotEqualsZeroFunctor<value_type>();

                // value not found, return last
                REQUIRE_EQ(
                        KE::end(expected),
                        std::find_if(KE::begin(expected), KE::end(expected), not_equals_zero));

                // pass iterators, returns iterator
                REQUIRE_EQ(KE::end(tensor), KE::find_if(exespace(), KE::begin(tensor),
                                                      KE::end(tensor), not_equals_zero));
                // pass tensor, returns iterator
                REQUIRE_EQ(KE::end(tensor), KE::find_if(exespace(), tensor, not_equals_zero));

                fill_tensors_inc(tensor, expected);

                constexpr value_t find_value = 13;
                const auto equals_val = EqualsValFunctor<value_type>(find_value);
                auto std_result =
                        std::find_if(KE::begin(expected), KE::end(expected), equals_val);
                auto distance = std::distance(KE::begin(expected), std_result);

                // pass const iterators, returns const iterator
                REQUIRE_EQ(
                        KE::cbegin(tensor) + distance,
                        KE::find_if(exespace(), KE::cbegin(tensor), KE::cend(tensor), equals_val));
                // pass tensor, returns iterator
                REQUIRE_EQ(KE::begin(tensor) + distance,
                           KE::find_if(exespace(), tensor, equals_val));
            }

            template<class TensorType>
            void test_find_if_not(const TensorType tensor) {
                using value_t = typename TensorType::value_type;
                using tensor_host_space_t = flare::Tensor<value_t *, flare::HostSpace>;

                tensor_host_space_t expected("count_expected", tensor.extent(0));
                compare_tensors(expected, tensor);

                const auto not_equals_zero = NotEqualsZeroFunctor<value_type>();

                // first value matches
                REQUIRE_EQ(KE::begin(expected),
                           std::find_if_not(KE::begin(expected), KE::end(expected),
                                            not_equals_zero));

                // pass iterators, returns iterator
                REQUIRE_EQ(KE::begin(tensor), KE::find_if_not(exespace(), KE::begin(tensor),
                                                            KE::end(tensor), not_equals_zero));
                // pass tensor, returns iterator
                REQUIRE_EQ(KE::begin(tensor),
                           KE::find_if_not(exespace(), tensor, not_equals_zero));

                fill_tensors_inc(tensor, expected);

                const auto equals_zero = EqualsValFunctor<value_type>(0);
                auto std_result =
                        std::find_if_not(KE::begin(expected), KE::end(expected), equals_zero);
                auto distance = std::distance(KE::begin(expected), std_result);

                // pass const iterators, returns const iterator
                REQUIRE_EQ(KE::cbegin(tensor) + distance,
                           KE::find_if_not(exespace(), KE::cbegin(tensor), KE::cend(tensor),
                                           equals_zero));
                // pass tensor, returns const iterator
                REQUIRE_EQ(KE::begin(tensor) + distance,
                           KE::find_if_not(exespace(), tensor, equals_zero));
            }

            template<class Tag, class ValueType>
            void run_all_scenarios() {
                for (const auto &scenario: default_scenarios) {
                    {
                        auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "find");
                        test_find(tensor);
                    }
                    {
                        auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "find_if");
                        test_find_if(tensor);
                    }
                    {
                        auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "find_if_not");
                        test_find_if_not(tensor);
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
