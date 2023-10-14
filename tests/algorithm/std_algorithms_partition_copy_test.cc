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
#include <utility>
#include <algorithm>

namespace Test {
    namespace stdalgos {
        namespace PartitionCopy {

            namespace KE = flare::experimental;

            template<class ValueType>
            struct UnifDist;

            template<>
            struct UnifDist<int> {
                using dist_type = std::uniform_int_distribution<int>;
                std::mt19937 m_gen;
                dist_type m_dist;

                UnifDist() : m_dist(-100, 100) { m_gen.seed(1034343); }

                int operator()() { return m_dist(m_gen); }
            };

            template<class TensorType>
            void fill_tensor(TensorType dest_tensor, const std::string &name) {
                using value_type = typename TensorType::value_type;
                using exe_space = typename TensorType::execution_space;
                const std::size_t ext = dest_tensor.extent(0);
                using aux_tensor_t = flare::Tensor<value_type *, exe_space>;
                aux_tensor_t aux_tensor("aux_tensor", ext);
                auto v_h = create_mirror_tensor(flare::HostSpace(), aux_tensor);

                if (name == "empty") {
                    // no op
                } else if (name == "one-element-a") {
                    v_h(0) = static_cast<value_type>(1);
                } else if (name == "one-element-b") {
                    v_h(0) = static_cast<value_type>(2);
                } else if (name == "two-elements-a") {
                    v_h(0) = static_cast<value_type>(1);
                    v_h(1) = static_cast<value_type>(2);
                } else if (name == "two-elements-b") {
                    v_h(0) = static_cast<value_type>(2);
                    v_h(1) = static_cast<value_type>(-1);
                } else if (name == "small-a") {
                    for (std::size_t i = 0; i < ext; ++i) {
                        v_h(i) = value_type{-5} + static_cast<value_type>(i + 1);
                    }
                } else if (name == "small-b") {
                    for (std::size_t i = 0; i < ext; ++i) {
                        v_h(i) = static_cast<value_type>(22);
                    }
                } else if (name == "small-c") {
                    for (std::size_t i = 0; i < ext; ++i) {
                        v_h(i) = static_cast<value_type>(-13);
                    }
                } else if (name == "medium" || name == "large") {
                    UnifDist<value_type> randObj;
                    for (std::size_t i = 0; i < ext; ++i) {
                        v_h(i) = randObj();
                    }
                } else {
                    throw std::runtime_error("invalid choice");
                }

                flare::deep_copy(aux_tensor, v_h);
                CopyFunctor<aux_tensor_t, TensorType> F1(aux_tensor, dest_tensor);
                flare::parallel_for("copy", dest_tensor.extent(0), F1);
            }

            template<class TensorTypeFrom, class ResultType, class TensorTypeDestTrue,
                    class TensorTypeDestFalse, class PredType>
            void verify_data(const std::string &name, ResultType my_result,
                             TensorTypeFrom tensor_from, TensorTypeDestTrue tensor_dest_true,
                             TensorTypeDestFalse tensor_dest_false, PredType pred) {
                using value_type = typename TensorTypeFrom::value_type;
                static_assert(
                        std::is_same<value_type, typename TensorTypeDestTrue::value_type>::value,
                        "");
                static_assert(
                        std::is_same<value_type, typename TensorTypeDestFalse::value_type>::value,
                        "");

                const std::size_t ext = tensor_from.extent(0);

                // create host clone of tensor_from and run std::partition_copy on it
                auto trnsor_from_h = create_host_space_copy(tensor_from);
                std::vector<value_type> std_vec_true(ext, 0);
                std::vector<value_type> std_vec_false(ext, 0);
                auto std_result =
                        std::partition_copy(KE::cbegin(trnsor_from_h), KE::cend(trnsor_from_h),
                                            std_vec_true.begin(), std_vec_false.begin(), pred);
                const std::size_t std_diff_true = std_result.first - std_vec_true.begin();
                const std::size_t std_diff_false = std_result.second - std_vec_false.begin();
                const std::size_t my_diff_true = my_result.first - KE::begin(tensor_dest_true);
                const std::size_t my_diff_false =
                        my_result.second - KE::begin(tensor_dest_false);
                REQUIRE_EQ(std_diff_true, my_diff_true);
                REQUIRE_EQ(std_diff_false, my_diff_false);

                auto tensor_dest_true_h = create_host_space_copy(tensor_dest_true);
                for (std::size_t i = 0; i < std_diff_true; ++i) {
                    REQUIRE_EQ(std_vec_true[i], tensor_dest_true_h(i));
                    // std::cout << "i= " << i << " "
                    // 	      << " std_true = " << std_vec_true[i] << " "
                    // 	      << " mine     = " << tensor_dest_true_h(i) << '\n';
                }

                auto tensor_dest_false_h = create_host_space_copy(tensor_dest_false);
                for (std::size_t i = 0; i < std_diff_false; ++i) {
                    REQUIRE_EQ(std_vec_false[i], tensor_dest_false_h(i));
                    // std::cout << "i= " << i << " "
                    // 	      << " std_false = " << std_vec_false[i] << " "
                    // 	      << " mine     = " << tensor_dest_false_h(i) << '\n';
                }

                if (name == "empty") {
                    REQUIRE_EQ(my_diff_true, 0u);
                    REQUIRE_EQ(my_diff_false, 0u);
                } else if (name == "one-element-a") {
                    REQUIRE_EQ(my_diff_true, 0u);
                    REQUIRE_EQ(my_diff_false, 1u);
                } else if (name == "one-element-b") {
                    REQUIRE_EQ(my_diff_true, 1u);
                    REQUIRE_EQ(my_diff_false, 0u);
                } else if (name == "two-elements-a") {
                    REQUIRE_EQ(my_diff_true, 1u);
                    REQUIRE_EQ(my_diff_false, 1u);
                } else if (name == "two-elements-b") {
                    REQUIRE_EQ(my_diff_true, 1u);
                    REQUIRE_EQ(my_diff_false, 1u);
                } else if (name == "small-b") {
                    REQUIRE_EQ(my_diff_true, 13u);
                    REQUIRE_EQ(my_diff_false, 0u);
                } else if (name == "small-c") {
                    REQUIRE_EQ(my_diff_true, 0u);
                    REQUIRE_EQ(my_diff_false, 15u);
                }
            }

            std::string value_type_to_string(int) { return "int"; }

            std::string value_type_to_string(double) { return "double"; }

            template<class Tag, class ValueType, class InfoType>
            void run_single_scenario(const InfoType &scenario_info) {
                const auto name = std::get<0>(scenario_info);
                const std::size_t tensor_ext = std::get<1>(scenario_info);
                // std::cout << "partition_copy: " << name << ", " <<
                // tensor_tag_to_string(Tag{})
                //           << ", " << value_type_to_string(ValueType()) << std::endl;

                auto tensor_from =
                        create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_from");
                IsEvenFunctor<ValueType> pred;

                {
                    auto tensor_dest_true =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_true");
                    auto tensor_dest_false =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_false");
                    fill_tensor(tensor_from, name);
                    auto result = KE::partition_copy(
                            exespace(), KE::cbegin(tensor_from), KE::cend(tensor_from),
                            KE::begin(tensor_dest_true), KE::begin(tensor_dest_false), pred);
                    verify_data(name, result, tensor_from, tensor_dest_true, tensor_dest_false, pred);
                }

                {
                    auto tensor_dest_true =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_true");
                    auto tensor_dest_false =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_false");
                    fill_tensor(tensor_from, name);
                    auto result = KE::partition_copy(
                            "my_label", exespace(), KE::cbegin(tensor_from), KE::cend(tensor_from),
                            KE::begin(tensor_dest_true), KE::begin(tensor_dest_false), pred);
                    verify_data(name, result, tensor_from, tensor_dest_true, tensor_dest_false, pred);
                }

                {
                    auto tensor_dest_true =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_true");
                    auto tensor_dest_false =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_false");
                    fill_tensor(tensor_from, name);
                    auto result = KE::partition_copy(exespace(), tensor_from, tensor_dest_true,
                                                     tensor_dest_false, pred);
                    verify_data(name, result, tensor_from, tensor_dest_true, tensor_dest_false, pred);
                }

                {
                    auto tensor_dest_true =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_true");
                    auto tensor_dest_false =
                            create_tensor<ValueType>(Tag{}, tensor_ext, "partition_copy_dest_false");
                    fill_tensor(tensor_from, name);
                    auto result = KE::partition_copy("my_label", exespace(), tensor_from,
                                                     tensor_dest_true, tensor_dest_false, pred);
                    verify_data(name, result, tensor_from, tensor_dest_true, tensor_dest_false, pred);
                }

                flare::fence();
            }

            template<class Tag, class ValueType>
            void run_all_scenarios() {
                const std::map<std::string, std::size_t> scenarios = {
                        {"empty",          0},
                        {"one-element-a",  1},
                        {"one-element-b",  1},
                        {"two-elements-a", 2},
                        {"two-elements-b", 2},
                        {"small-a",        9},
                        {"small-b",        13},
                        {"small-c",        15},
                        {"medium",         103}};  //      {"large", 101513}};

                for (const auto &it: scenarios) {
                    run_single_scenario<Tag, ValueType>(it);
                }
            }

            TEST_CASE("std_algorithms_partitioning_ops, partition_copy") {
                run_all_scenarios<DynamicTag, int>();
                run_all_scenarios<StridedThreeTag, int>();
            }

        }  // namespace PartitionCopy
    }  // namespace stdalgos
}  // namespace Test
