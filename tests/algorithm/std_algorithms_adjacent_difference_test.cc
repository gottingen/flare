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
#include <flare/algorithm/adjacent_difference.h>
#include <utility>
#include <numeric>

namespace Test {
namespace stdalgos {
namespace AdjacentDifference {

namespace KE = flare::experimental;

template <class DestTensorType>
void fill_tensor(DestTensorType dest_tensor, const std::string& name) {
  // we need to be careful because dest_tensor might not be deep copyable
  // for instance strided layout

  using value_type      = typename DestTensorType::value_type;
  const std::size_t ext = dest_tensor.extent(0);
  auto aux_tensor =
      create_deep_copyable_compatible_tensor_with_same_extent(dest_tensor);
  auto aux_v_h = create_mirror_tensor(flare::HostSpace(), aux_tensor);

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element") {
    aux_v_h(0) = static_cast<value_type>(1);
  }

  else if (name == "two-elements-a") {
    aux_v_h(0) = static_cast<value_type>(1);
    aux_v_h(1) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-b") {
    aux_v_h(0) = static_cast<value_type>(2);
    aux_v_h(1) = static_cast<value_type>(-1);
  }

  else if (name == "small-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      aux_v_h(i) = static_cast<value_type>(i) * 2;
    }
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      aux_v_h(i) = static_cast<value_type>(i) * 3;
    }
    aux_v_h(5) = static_cast<value_type>(-15);
  }

  else if (name == "medium-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      aux_v_h(i) = static_cast<value_type>(i) * 2;
    }
  }

  else if (name == "medium-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      aux_v_h(i) = static_cast<value_type>(i) * 2;
    }
    aux_v_h(4) = static_cast<value_type>(-1);
  }

  else if (name == "large-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      aux_v_h(i) = static_cast<value_type>(-100) + static_cast<value_type>(i);
    }
  }

  else if (name == "large-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      aux_v_h(i) = static_cast<value_type>(-100) + static_cast<value_type>(i);
    }
    aux_v_h(156) = static_cast<value_type>(-250);

  }

  else {
    throw std::runtime_error("invalid choice");
  }

  flare::deep_copy(aux_tensor, aux_v_h);
  CopyFunctor<decltype(aux_tensor), DestTensorType> F1(aux_tensor, dest_tensor);
  flare::parallel_for("copy", dest_tensor.extent(0), F1);
}

template <class TestTensorType, class... Args>
auto compute_gold(TestTensorType test_tensor, const std::string& name,
                  Args... args /* copy on purpose */) {
  // we need to be careful because test_tensor might not be deep copyable
  // for instance strided layout

  const std::size_t ext = test_tensor.extent(0);

  // create a deep copyable clone of test_tensor
  auto test_tensor_dc = create_deep_copyable_compatible_clone(test_tensor);
  auto test_tensor_dc_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), test_tensor_dc);

  // create gold deep copyable tensor
  auto gold_tensor =
      create_deep_copyable_compatible_tensor_with_same_extent(test_tensor);
  auto gold_tensor_h = create_mirror_tensor(flare::HostSpace(), gold_tensor);

  // compute gold solution on host and deep copy to device
  if (name == "empty") {
    return gold_tensor;
  } else {
    using value_type = typename TestTensorType::value_type;
    std::vector<value_type> tmp(ext);
    for (std::size_t i = 0; i < ext; ++i) {
      tmp[i] = test_tensor_dc_h(i);
    }
    // run adj-diff on tmp directly
    std::adjacent_difference(tmp.begin(), tmp.end(), tmp.begin(),
                             std::forward<Args>(args)...);

    // copy from tmp to gold_h
    for (std::size_t i = 0; i < ext; ++i) {
      gold_tensor_h(i) = tmp[i];
    }
    // deep_copy to device
    flare::deep_copy(gold_tensor, gold_tensor_h);
    return gold_tensor;
  }
}

template <class TestTensorType, class GolDTensorType>
void verify_data(TestTensorType test_tensor, GolDTensorType gold) {
  // we need to be careful because test_tensor might not be deep copyable
  // for instance strided layout

  auto test_tensor_dc = create_deep_copyable_compatible_clone(test_tensor);
  auto test_tensor_dc_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), test_tensor_dc);
  // gold is deep_copyable for sure
  const auto gold_h = create_mirror_tensor_and_copy(flare::HostSpace(), gold);

  for (std::size_t i = 0; i < test_tensor.extent(0); ++i) {
    REQUIRE_EQ(gold_h(i), test_tensor_dc_h(i));
  }
}

template <class ValueType1, class ValueType2 = ValueType1,
          class RetType = ValueType2>
struct CustomBinaryOpFunctor {
  FLARE_INLINE_FUNCTION
  RetType operator()(const ValueType1& a, const ValueType2& b) const {
    return a * b;
  }
};

template <class ValueType1, class ValueType2 = ValueType1,
          class RetType = ValueType2>
struct DefaultBinaryOpFunctor {
  FLARE_INLINE_FUNCTION
  RetType operator()(const ValueType1& a, const ValueType2& b) const {
    return a - b;
  }
};

template <class Tag, class ValueType, class InfoType, class... Args>
void run_single_scenario(const InfoType& scenario_info,
                         Args... args /* copy on purpose */) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);

  auto tensor_from =
      create_tensor<ValueType>(Tag{}, tensor_ext, "adj_diff_from_tensor");
  fill_tensor(tensor_from, name);

  const auto gold = compute_gold(tensor_from, name, args...);

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "adj_diff_dest_tensor");
    auto res1 = KE::adjacent_difference(exespace(), KE::cbegin(tensor_from),
                                        KE::cend(tensor_from),
                                        KE::begin(tensor_dest), args...);
    REQUIRE_EQ(res1, KE::end(tensor_dest));
    verify_data(tensor_dest, gold);
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "adj_diff_dest_tensor");
    auto res2 = KE::adjacent_difference(
        "label", exespace(), KE::cbegin(tensor_from), KE::cend(tensor_from),
        KE::begin(tensor_dest), args...);
    REQUIRE_EQ(res2, KE::end(tensor_dest));
    verify_data(tensor_dest, gold);
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "adj_diff_dest_tensor");
    auto res3 =
        KE::adjacent_difference(exespace(), tensor_from, tensor_dest, args...);
    REQUIRE_EQ(res3, KE::end(tensor_dest));
    verify_data(tensor_dest, gold);
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "adj_diff_dest_tensor");
    auto res4 = KE::adjacent_difference("label", exespace(), tensor_from,
                                        tensor_dest, args...);
    REQUIRE_EQ(res4, KE::end(tensor_dest));
    verify_data(tensor_dest, gold);
  }

  flare::fence();
}

template <class Tag, class ValueType, class... Args>
void run_all_scenarios(Args... args /* copy on purpose */) {
  // if (0 < sizeof...(args)) {
  //   std::cout << "adjacent_difference: " << tensor_tag_to_string(Tag{})
  //             << ", custom binary op, all overloads \n";
  // } else {
  //   std::cout << "adjacent_difference: " << tensor_tag_to_string(Tag{})
  //             << ", default binary op, all overloads \n";
  // }

  for (const auto& it : default_scenarios) {
    run_single_scenario<Tag, ValueType>(it, args...);
  }
}

TEST_CASE("std_algorithms_numerics_ops_test, adjecent_difference") {
  using value_type = double;

  run_all_scenarios<DynamicTag, value_type>();
  run_all_scenarios<StridedTwoTag, value_type>();
  run_all_scenarios<StridedThreeTag, value_type>();

  using custom_binary_op = CustomBinaryOpFunctor<value_type>;
  run_all_scenarios<DynamicTag, value_type>(custom_binary_op{});
  run_all_scenarios<StridedTwoTag, value_type>(custom_binary_op{});
  run_all_scenarios<StridedThreeTag, value_type>(custom_binary_op{});
}

}  // namespace AdjacentDifference
}  // namespace stdalgos
}  // namespace Test
