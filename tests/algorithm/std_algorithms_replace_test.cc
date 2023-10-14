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

namespace Test {
namespace stdalgos {
namespace Replace {

namespace KE = flare::experimental;

template <class TensorType>
void fill_tensor(TensorType dest_tensor, const std::string& name) {
  using value_type      = typename TensorType::value_type;
  using exe_space       = typename TensorType::execution_space;
  const std::size_t ext = dest_tensor.extent(0);
  using aux_tensor_t      = flare::Tensor<value_type*, exe_space>;
  aux_tensor_t aux_tensor("aux_tensor", ext);
  auto v_h = create_mirror_tensor(flare::HostSpace(), aux_tensor);

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element-a") {
    v_h(0) = static_cast<value_type>(1);
  }

  else if (name == "one-element-b") {
    v_h(0) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-a") {
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-b") {
    v_h(0) = static_cast<value_type>(2);
    v_h(1) = static_cast<value_type>(-1);
  }

  else if (name == "small-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = value_type{-5} + static_cast<value_type>(i + 1);
    }
    v_h(0) = static_cast<value_type>(2);
    v_h(3) = static_cast<value_type>(2);
    v_h(5) = static_cast<value_type>(2);
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      if (i < 4) {
        v_h(i) = static_cast<value_type>(-1);
      } else {
        v_h(i) = static_cast<value_type>(2);
      }
    }
  }

  else if (name == "medium" || name == "large") {
    for (std::size_t i = 0; i < ext; ++i) {
      if (i % 2 == 0) {
        v_h(i) = static_cast<value_type>(-1);
      } else {
        v_h(i) = static_cast<value_type>(2);
      }
    }
  }

  else {
    throw std::runtime_error("invalid choice");
  }

  flare::deep_copy(aux_tensor, v_h);
  CopyFunctor<aux_tensor_t, TensorType> F1(aux_tensor, dest_tensor);
  flare::parallel_for("copy", dest_tensor.extent(0), F1);
}

template <class TensorType1, class ValueType>
void verify_data(const std::string& name, TensorType1 test_tensor,
                 ValueType new_value) {
  //! always careful because tensors might not be deep copyable
  auto tensor_dc = create_deep_copyable_compatible_clone(test_tensor);
  auto tensor_h  = create_mirror_tensor_and_copy(flare::HostSpace(), tensor_dc);

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element-a") {
    REQUIRE_EQ(tensor_h(0), ValueType{1});
  }

  else if (name == "one-element-b") {
    REQUIRE_EQ(tensor_h(0), new_value);
  }

  else if (name == "two-elements-a") {
    REQUIRE_EQ(tensor_h(0), ValueType{1});
    REQUIRE_EQ(tensor_h(1), new_value);
  }

  else if (name == "two-elements-b") {
    REQUIRE_EQ(tensor_h(0), new_value);
    REQUIRE_EQ(tensor_h(1), ValueType{-1});
  }

  else if (name == "small-a") {
    for (std::size_t i = 0; i < tensor_h.extent(0); ++i) {
      if (i == 0 || i == 3 || i == 5 || i == 6) {
        REQUIRE_EQ(tensor_h(i), new_value);
      } else {
        const auto gold = ValueType{-5} + static_cast<ValueType>(i + 1);
        REQUIRE_EQ(tensor_h(i), gold);
      }
    }
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < tensor_h.extent(0); ++i) {
      if (i < 4) {
        REQUIRE_EQ(tensor_h(i), ValueType{-1});
      } else {
        REQUIRE_EQ(tensor_h(i), new_value);
      }
    }
  }

  else if (name == "medium" || name == "large") {
    for (std::size_t i = 0; i < tensor_h.extent(0); ++i) {
      if (i % 2 == 0) {
        REQUIRE_EQ(tensor_h(i), ValueType{-1});
      } else {
        REQUIRE_EQ(tensor_h(i), new_value);
      }
    }
  }

  else {
    throw std::runtime_error("invalid choice");
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // std::cout << "replace: " << name << ", " << tensor_tag_to_string(Tag{}) << ",
  // "
  //           << value_type_to_string(ValueType()) << std::endl;

  ValueType old_value{2};
  ValueType new_value{43};
  auto tensor = create_tensor<ValueType>(Tag{}, tensor_ext, "replace");

  {
    fill_tensor(tensor, name);
    KE::replace(exespace(), KE::begin(tensor), KE::end(tensor), old_value,
                new_value);
    verify_data(name, tensor, new_value);
  }

  {
    fill_tensor(tensor, name);
    KE::replace("label", exespace(), KE::begin(tensor), KE::end(tensor), old_value,
                new_value);
    verify_data(name, tensor, new_value);
  }

  {
    fill_tensor(tensor, name);
    KE::replace(exespace(), tensor, old_value, new_value);
    verify_data(name, tensor, new_value);
  }

  {
    fill_tensor(tensor, name);
    KE::replace("label", exespace(), tensor, old_value, new_value);
    verify_data(name, tensor, new_value);
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 9},
      {"small-b", 13},       {"medium", 1103},      {"large", 101513}};

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it);
  }
}

TEST_CASE("std_algorithms_replace_ops_test, replace") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

}  // namespace Replace
}  // namespace stdalgos
}  // namespace Test
