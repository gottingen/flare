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

namespace KE = flare::experimental;

namespace Test {
    namespace stdalgos {

        struct std_algorithms_min_max_element_test : std_algorithms_test {
            std_algorithms_min_max_element_test() = default;

            ~std_algorithms_min_max_element_test() = default;

            const int m_number_of_filling_cases = 5;

            void fillFixtureTensors(int case_number) {
                static_tensor_t tmpTensor("tmpTensor");
                auto tmp_tensor_h = flare::create_mirror_tensor(flare::HostSpace(), tmpTensor);
                if (case_number == 1) {
                    tmp_tensor_h(0) = 0;
                    tmp_tensor_h(1) = 0;
                    tmp_tensor_h(2) = 0;
                    tmp_tensor_h(3) = 2;
                    tmp_tensor_h(4) = 2;
                    tmp_tensor_h(5) = 1;
                    tmp_tensor_h(6) = 1;
                    tmp_tensor_h(7) = 1;
                    tmp_tensor_h(8) = 1;
                    tmp_tensor_h(9) = 0;
                } else if (case_number == 2) {
                    tmp_tensor_h(0) = 1;
                    tmp_tensor_h(1) = 2;
                    tmp_tensor_h(2) = 3;
                    tmp_tensor_h(3) = 4;
                    tmp_tensor_h(4) = 5;
                    tmp_tensor_h(5) = 6;
                    tmp_tensor_h(6) = 7;
                    tmp_tensor_h(7) = 8;
                    tmp_tensor_h(8) = 9;
                    tmp_tensor_h(9) = 10;
                } else if (case_number == 3) {
                    tmp_tensor_h(0) = 8;
                    tmp_tensor_h(1) = 8;
                    tmp_tensor_h(2) = -1;
                    tmp_tensor_h(3) = -1;
                    tmp_tensor_h(4) = 5;
                    tmp_tensor_h(5) = 5;
                    tmp_tensor_h(6) = 5;
                    tmp_tensor_h(7) = 8;
                    tmp_tensor_h(8) = 2;
                    tmp_tensor_h(9) = 1;
                } else if (case_number == 4) {
                    tmp_tensor_h(0) = 2;
                    tmp_tensor_h(1) = 2;
                    tmp_tensor_h(2) = 2;
                    tmp_tensor_h(3) = 2;
                    tmp_tensor_h(4) = 2;
                    tmp_tensor_h(5) = 2;
                    tmp_tensor_h(6) = 2;
                    tmp_tensor_h(7) = 2;
                    tmp_tensor_h(8) = 2;
                    tmp_tensor_h(9) = 2;
                } else if (case_number == 5) {
                    tmp_tensor_h(0) = 1;
                    tmp_tensor_h(1) = 2;
                    tmp_tensor_h(2) = 3;
                    tmp_tensor_h(3) = 4;
                    tmp_tensor_h(4) = 5;
                    tmp_tensor_h(5) = 12;
                    tmp_tensor_h(6) = 5;
                    tmp_tensor_h(7) = 4;
                    tmp_tensor_h(8) = 3;
                    tmp_tensor_h(9) = 2;
                } else {
                }

                flare::deep_copy(tmpTensor, tmp_tensor_h);
                copyInputTensorToFixtureTensors(tmpTensor);
            }

            flare::pair<int, value_type> goldSolutionMaxElement(int caseNumber) {
                // returns {indexOfMaxElem, maxValue}
                if (caseNumber == 1) {
                    return {3, 2};
                } else if (caseNumber == 2) {
                    return {9, 10};
                } else if (caseNumber == 3) {
                    return {0, 8};
                } else if (caseNumber == 4) {
                    return {0, 2};
                } else if (caseNumber == 5) {
                    return {5, 12};
                } else {
                    return {};
                }
            }

            flare::pair<int, value_type> goldSolutionMinElement(int caseNumber) {
                // returns {indexOfMinElem, minValue}
                if (caseNumber == 1) {
                    return {0, 0};
                } else if (caseNumber == 2) {
                    return {0, 1};
                } else if (caseNumber == 3) {
                    return {2, -1};
                } else if (caseNumber == 4) {
                    return {0, 2};
                } else if (caseNumber == 5) {
                    return {0, 1};
                } else {
                    return {};
                }
            }

            flare::pair<flare::pair<int, value_type>, flare::pair<int, value_type>>
            goldSolutionMinMaxElement(int caseNumber) {
                // returns {{indexOfMinElem, minValue}, {indexOfMaxElem, maxValue}}
                // remember that for min it finds the first smallest element
                // remember that for max it finds the last biggest element
                if (caseNumber == 1) {
                    return {{0, 0},
                            {4, 2}};
                } else if (caseNumber == 2) {
                    return {{0, 1},
                            {9, 10}};
                } else if (caseNumber == 3) {
                    return {{2, -1},
                            {7, 8}};
                } else if (caseNumber == 4) {
                    return {{0, 2},
                            {9, 2}};
                } else if (caseNumber == 5) {
                    return {{0, 1},
                            {5, 12}};
                } else {
                    return {};
                }
            }

            template<class TensorType>
            void test_max_element_non_trivial_data(TensorType tensor);

            template<class TensorType>
            void test_min_element_non_trivial_data(TensorType tensor);

            template<class TensorType>
            void test_minmax_element_non_trivial_data(TensorType tensor);

            template<class TensorType>
            void test_max_element_non_trivial_data_custom_comp(TensorType tensor);

            template<class TensorType>
            void test_min_element_non_trivial_data_custom_comp(TensorType tensor);

            template<class TensorType>
            void test_minmax_element_non_trivial_data_custom_comp(TensorType tensor);
        };

        template<class IndexType, class ValueType, class ItType, class TesteDTensorType>
        void std_algo_min_max_test_verify(flare::pair<IndexType, ValueType> goldPair,
                                          const ItType result,
                                          TesteDTensorType testeDTensor) {
            // check that iterator is pointing to right element
            REQUIRE_EQ(result - KE::begin(testeDTensor), goldPair.first);

            // create a tensor for the result to copy into it the iterator's value
            using result_tensor_t = flare::Tensor<int>;
            result_tensor_t resultTensor("result");
            CopyFromIteratorFunctor<ItType, result_tensor_t> cf(result, resultTensor);
            flare::parallel_for("_std_algo_copy", 1, cf);
            auto result_v_h =
                    flare::create_mirror_tensor_and_copy(flare::HostSpace(), resultTensor);

            // use the host mirror of the result tensor to check that the values match
            REQUIRE_EQ(result_v_h(), goldPair.second);
        }

        template<class GoldSolutionType, class ItType, class TesteDTensorType>
        void std_algo_min_max_test_verify(const GoldSolutionType &goldSolution,
                                          const ItType itMin, const ItType itMax,
                                          TesteDTensorType testeDTensor) {
            std_algo_min_max_test_verify(goldSolution.first, itMin, testeDTensor);
            std_algo_min_max_test_verify(goldSolution.second, itMax, testeDTensor);
        }

        template<class TensorType>
        void test_max_element_trivial_data(TensorType tensor) {
            /* if we pass empty range, should return last */
            auto result = KE::max_element(exespace(), KE::cbegin(tensor), KE::cbegin(tensor));
            REQUIRE_EQ(result, KE::cbegin(tensor));

            /* if we pass empty range, should return last */
            auto it0 = KE::cbegin(tensor) + 3;
            auto it1 = it0;
            auto result2 = KE::max_element(exespace(), it0, it1);
            REQUIRE_EQ(result2, it1);
        }

        template<class TensorType>
        void test_min_element_trivial_data(TensorType tensor) {
            /* if we pass empty range, should return last */
            auto result = KE::min_element(exespace(), KE::cbegin(tensor), KE::cbegin(tensor));
            REQUIRE_EQ(result, KE::cbegin(tensor));

            /* if we pass empty range, should return last */
            auto it0 = KE::cbegin(tensor) + 3;
            auto it1 = it0;
            auto result2 = KE::min_element(exespace(), it0, it1);
            REQUIRE_EQ(result2, it1);
        }

        template<class TensorType>
        void test_minmax_element_empty_range(TensorType tensor) {
            auto result =
                    KE::minmax_element(exespace(), KE::cbegin(tensor), KE::cbegin(tensor));
            REQUIRE_EQ(result.first, KE::cbegin(tensor));
            REQUIRE_EQ(result.second, KE::cbegin(tensor));
            auto it0 = KE::cbegin(tensor) + 3;
            auto it1 = it0;
            auto result2 = KE::minmax_element(exespace(), it0, it1);
            REQUIRE_EQ(result2.first, it1);
            REQUIRE_EQ(result2.second, it1);
        }

        template<class TensorType>
        void std_algorithms_min_max_element_test::test_max_element_non_trivial_data(
                TensorType tensor) {
            for (int id = 1; id <= m_number_of_filling_cases; ++id) {
                fillFixtureTensors(id);
                const auto gold_solution = goldSolutionMaxElement(id);
                // API accepting tensor
                {
                    const auto result = KE::max_element(exespace(), tensor);
                    std_algo_min_max_test_verify(gold_solution, result, tensor);
                    const auto result2 = KE::max_element("MYCUSTOMLABEL1", exespace(), tensor);
                    std_algo_min_max_test_verify(gold_solution, result2, tensor);
                }
                // API accepting iterators
                {
                    const auto result =
                            KE::max_element(exespace(), KE::begin(tensor), KE::end(tensor));
                    std_algo_min_max_test_verify(gold_solution, result, tensor);
                    const auto result2 = KE::max_element("MYCUSTOMLABEL2", exespace(),
                                                         KE::begin(tensor), KE::end(tensor));
                    std_algo_min_max_test_verify(gold_solution, result2, tensor);
                }
            }
        }

        template<class TensorType>
        void std_algorithms_min_max_element_test::test_min_element_non_trivial_data(
                TensorType tensor) {
            for (int id = 1; id <= m_number_of_filling_cases; ++id) {
                fillFixtureTensors(id);
                const auto goldPair = goldSolutionMinElement(id);
                // API accepting tensor
                {
                    const auto result = KE::min_element(exespace(), tensor);
                    std_algo_min_max_test_verify(goldPair, result, tensor);
                    const auto result2 = KE::min_element("MYCUSTOMLABEL1", exespace(), tensor);
                    std_algo_min_max_test_verify(goldPair, result2, tensor);
                }
                // API accepting iterators
                {
                    const auto result =
                            KE::min_element(exespace(), KE::begin(tensor), KE::end(tensor));
                    std_algo_min_max_test_verify(goldPair, result, tensor);
                    const auto result2 = KE::min_element("MYCUSTOMLABEL2", exespace(),
                                                         KE::begin(tensor), KE::end(tensor));
                    std_algo_min_max_test_verify(goldPair, result2, tensor);
                }
            }
        }

        template<class TensorType>
        void std_algorithms_min_max_element_test::test_minmax_element_non_trivial_data(
                TensorType tensor) {
            for (int id = 1; id <= m_number_of_filling_cases; ++id) {
                fillFixtureTensors(id);
                const auto gold = goldSolutionMinMaxElement(id);
                {
                    auto result = KE::minmax_element(exespace(), tensor);
                    std_algo_min_max_test_verify(gold, result.first, result.second, tensor);
                    const auto result2 =
                            KE::minmax_element("MYCUSTOMLABEL1", exespace(), tensor);
                    std_algo_min_max_test_verify(gold, result2.first, result2.second, tensor);
                }
                {
                    const auto result =
                            KE::minmax_element(exespace(), KE::begin(tensor), KE::end(tensor));
                    std_algo_min_max_test_verify(gold, result.first, result.second, tensor);
                    const auto result2 = KE::minmax_element("MYCUSTOMLABEL2", exespace(),
                                                            KE::begin(tensor), KE::end(tensor));
                    std_algo_min_max_test_verify(gold, result2.first, result2.second, tensor);
                }
            }
        }

        template<class TensorType>
        void std_algorithms_min_max_element_test::
        test_max_element_non_trivial_data_custom_comp(TensorType tensor) {
            for (int id = 1; id <= m_number_of_filling_cases; ++id) {
                fillFixtureTensors(id);
                const auto goldPair = goldSolutionMaxElement(id);
                CustomLessThanComparator<value_type, value_type> comp;
                // API accepting tensor
                {
                    const auto result = KE::max_element(exespace(), tensor, comp);
                    std_algo_min_max_test_verify(goldPair, result, tensor);
                    const auto result2 =
                            KE::max_element("MYCUSTOMLABEL3", exespace(), tensor, comp);
                    std_algo_min_max_test_verify(goldPair, result2, tensor);
                }
                // API accepting iterators
                {
                    const auto result =
                            KE::max_element(exespace(), KE::begin(tensor), KE::end(tensor), comp);
                    std_algo_min_max_test_verify(goldPair, result, tensor);
                    const auto result2 = KE::max_element(
                            "MYCUSTOMLABEL4", exespace(), KE::begin(tensor), KE::end(tensor), comp);
                    std_algo_min_max_test_verify(goldPair, result2, tensor);
                }
            }
        }

        template<class TensorType>
        void std_algorithms_min_max_element_test::
        test_min_element_non_trivial_data_custom_comp(TensorType tensor) {
            for (int id = 1; id <= m_number_of_filling_cases; ++id) {
                fillFixtureTensors(id);
                const auto goldPair = goldSolutionMinElement(id);
                CustomLessThanComparator<value_type, value_type> comp;
                // API accepting tensor
                {
                    const auto result = KE::min_element(exespace(), tensor, comp);
                    std_algo_min_max_test_verify(goldPair, result, tensor);
                    const auto result2 =
                            KE::min_element("MYCUSTOMLABEL3", exespace(), tensor, comp);
                    std_algo_min_max_test_verify(goldPair, result2, tensor);
                }
                // API accepting iterators
                {
                    const auto result =
                            KE::min_element(exespace(), KE::begin(tensor), KE::end(tensor), comp);
                    std_algo_min_max_test_verify(goldPair, result, tensor);
                    const auto result2 = KE::min_element(
                            "MYCUSTOMLABEL4", exespace(), KE::begin(tensor), KE::end(tensor), comp);
                    std_algo_min_max_test_verify(goldPair, result2, tensor);
                }
            }
        }

        template<class TensorType>
        void std_algorithms_min_max_element_test::
        test_minmax_element_non_trivial_data_custom_comp(TensorType tensor) {
            for (int id = 1; id <= m_number_of_filling_cases; ++id) {
                fillFixtureTensors(id);
                const auto goldPair = goldSolutionMinMaxElement(id);
                CustomLessThanComparator<value_type, value_type> comp;
                {
                    const auto result = KE::minmax_element(exespace(), tensor, comp);
                    std_algo_min_max_test_verify(goldPair, result.first, result.second, tensor);
                    const auto result2 =
                            KE::minmax_element("MYCUSTOMLABEL3", exespace(), tensor, comp);
                    std_algo_min_max_test_verify(goldPair, result2.first, result2.second,
                                                 tensor);
                }
                {
                    const auto result =
                            KE::minmax_element(exespace(), KE::begin(tensor), KE::end(tensor), comp);
                    std_algo_min_max_test_verify(goldPair, result.first, result.second, tensor);
                    const auto result2 = KE::minmax_element(
                            "MYCUSTOMLABEL4", exespace(), KE::begin(tensor), KE::end(tensor), comp);
                    std_algo_min_max_test_verify(goldPair, result2.first, result2.second,
                                                 tensor);
                }
            }
        }

        // trivial case
        TEST_CASE_FIXTURE(std_algorithms_min_max_element_test, "min_element_empty_range") {
            test_min_element_trivial_data(m_static_tensor);
            test_min_element_trivial_data(m_dynamic_tensor);
            test_min_element_trivial_data(m_strided_tensor);
        }

        TEST_CASE_FIXTURE(std_algorithms_min_max_element_test, "max_element_empty_range") {
            test_max_element_trivial_data(m_static_tensor);
            test_max_element_trivial_data(m_dynamic_tensor);
            test_max_element_trivial_data(m_strided_tensor);
        }

        // non-trivial data
        TEST_CASE_FIXTURE(std_algorithms_min_max_element_test, "min_element_non_trivial_data") {
            test_min_element_non_trivial_data(m_static_tensor);
            test_min_element_non_trivial_data(m_dynamic_tensor);
            test_min_element_non_trivial_data(m_strided_tensor);
        }

        TEST_CASE_FIXTURE(std_algorithms_min_max_element_test, "max_element_non_trivial_data") {
            test_max_element_non_trivial_data(m_static_tensor);
            test_max_element_non_trivial_data(m_dynamic_tensor);
            test_max_element_non_trivial_data(m_strided_tensor);
        }

        // non-trivial data, custom comp
        TEST_CASE_FIXTURE(std_algorithms_min_max_element_test, "min_element_non_trivial_data_custom_comp") {
            test_min_element_non_trivial_data_custom_comp(m_static_tensor);
            test_min_element_non_trivial_data_custom_comp(m_dynamic_tensor);
            test_min_element_non_trivial_data_custom_comp(m_strided_tensor);
        }

        TEST_CASE_FIXTURE(std_algorithms_min_max_element_test, "max_element_non_trivial_data_custom_comp") {
            test_max_element_non_trivial_data_custom_comp(m_static_tensor);
            test_max_element_non_trivial_data_custom_comp(m_dynamic_tensor);
            test_max_element_non_trivial_data_custom_comp(m_strided_tensor);
        }

        TEST_CASE_FIXTURE(std_algorithms_min_max_element_test,"minmax_element_non_trivial_data_custom_comp") {
            test_minmax_element_non_trivial_data_custom_comp(m_static_tensor);
            test_minmax_element_non_trivial_data_custom_comp(m_dynamic_tensor);
            test_minmax_element_non_trivial_data_custom_comp(m_strided_tensor);
        }

    }  // namespace stdalgos
}  // namespace Test
