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
namespace Unique {

namespace KE = flare::experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(2, 7) { m_gen.seed(1034343); }
  int operator()() { return m_dist(m_gen); }
};

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
    v_h(0)  = static_cast<value_type>(0);
    v_h(1)  = static_cast<value_type>(1);
    v_h(2)  = static_cast<value_type>(3);
    v_h(3)  = static_cast<value_type>(2);
    v_h(4)  = static_cast<value_type>(2);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(4);
    v_h(7)  = static_cast<value_type>(4);
    v_h(8)  = static_cast<value_type>(5);
    v_h(9)  = static_cast<value_type>(6);
    v_h(10) = static_cast<value_type>(6);
  }

  else if (name == "small-b") {
    v_h(0)  = static_cast<value_type>(1);
    v_h(1)  = static_cast<value_type>(1);
    v_h(2)  = static_cast<value_type>(1);
    v_h(3)  = static_cast<value_type>(2);
    v_h(4)  = static_cast<value_type>(3);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(4);
    v_h(7)  = static_cast<value_type>(4);
    v_h(8)  = static_cast<value_type>(5);
    v_h(9)  = static_cast<value_type>(6);
    v_h(10) = static_cast<value_type>(8);
    v_h(11) = static_cast<value_type>(9);
    v_h(12) = static_cast<value_type>(8);
  }

  else if (name == "medium-a") {
    // beginning just contains increasing values
    for (std::size_t i = 0; i < 1000; ++i) {
      v_h(i) = static_cast<value_type>(i);
    }

    // then use random
    UnifDist<value_type> randObj;
    for (std::size_t i = 1000; i < ext; ++i) {
      v_h(i) = randObj();
    }
  }

  else if (name == "medium-b") {
    for (std::size_t i = 0; i < 1000; ++i) {
      v_h(i) = static_cast<value_type>(22);
    }
    for (std::size_t i = 1000; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(44);
    }
  }

  else if (name == "large-a") {
    // put equal elements at the end
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(i);
    }
    v_h(ext - 3) = static_cast<value_type>(44);
    v_h(ext - 2) = static_cast<value_type>(44);
    v_h(ext - 1) = static_cast<value_type>(44);
  }

  else if (name == "large-b") {
    UnifDist<value_type> randObj;
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
  }

  else {
    throw std::runtime_error("invalid choice");
  }

  flare::deep_copy(aux_tensor, v_h);
  CopyFunctor<aux_tensor_t, TensorType> F1(aux_tensor, dest_tensor);
  flare::parallel_for("copy", dest_tensor.extent(0), F1);
}

template <class ResultIt, class TensorType1, class TensorType2, class... Args>
void verify_data(const std::string& name, ResultIt my_result_it,
                 TensorType1 tensor_test, TensorType2 data_v_h, Args... args) {
  // run std unique on host data
  auto std_r = std::unique(KE::begin(data_v_h), KE::end(data_v_h), args...);

  //
  // check the returned iterator is correct
  //
  const auto std_diff = (std::size_t)(std_r - KE::begin(data_v_h));
  const auto my_diff  = (std::size_t)(my_result_it - KE::begin(tensor_test));
  REQUIRE_EQ(my_diff, std_diff);

  //
  // check the data in the tensor
  //
  // Note that we need to stop after std_diff because
  // after that values are unspecified, see std::unique
  auto tensor_test_h = create_host_space_copy(tensor_test);
  for (std::size_t i = 0; i < std_diff; ++i) {
    // std::cout << "i = " << i
    // 		<< " my  = " << tensor_test_h(i) << " "
    // 		<< " std = " << data_v_h(i)
    // 		<< '\n';
    REQUIRE_EQ(tensor_test_h(i), data_v_h(i));
  }

  if (name == "medium-b") {
    using value_type = typename TensorType1::value_type;
    REQUIRE_EQ(my_diff, (std::size_t)2);
    REQUIRE_EQ(tensor_test_h(0), (value_type)22);
    REQUIRE_EQ(tensor_test_h(1), (value_type)44);
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType>
void print_scenario_details(const std::string& name) {
  std::cout << "unique: default predicate: " << name << ", "
            << tensor_tag_to_string(Tag{}) << " "
            << value_type_to_string(ValueType()) << '\n';
}

template <class Tag, class ValueType, class Predicate>
void print_scenario_details(const std::string& name, Predicate pred) {
  (void)pred;
  std::cout << "unique: custom  predicate: " << name << ", "
            << tensor_tag_to_string(Tag{}) << " "
            << value_type_to_string(ValueType()) << '\n';
}

template <class Tag, class ValueType, class InfoType, class... Args>
void run_single_scenario(const InfoType& scenario_info, Args... args) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // print_scenario_details<Tag, ValueType>(name, args...);

  auto test_tensor = create_tensor<ValueType>(Tag{}, tensor_ext, "unique_test_tensor");

  {
    fill_tensor(test_tensor, name);
    // make host copy BEFORE running unique on it since unique modifies it
    auto data_h = create_host_space_copy(test_tensor);

    // run unique
    auto rit = KE::unique(exespace(), KE::begin(test_tensor), KE::end(test_tensor),
                          args...);
    // verify
    verify_data(name, rit, test_tensor, data_h, args...);
  }

  {
    fill_tensor(test_tensor, name);
    // make host copy BEFORE running unique on it since unique modifies it
    auto data_h = create_host_space_copy(test_tensor);

    // run unique
    auto rit = KE::unique("label", exespace(), KE::begin(test_tensor),
                          KE::end(test_tensor), args...);
    // verify
    verify_data(name, rit, test_tensor, data_h, args...);
  }

  {
    fill_tensor(test_tensor, name);
    // make host copy BEFORE running unique on it since unique modifies it
    auto data_h = create_host_space_copy(test_tensor);

    // run unique
    auto rit = KE::unique(exespace(), test_tensor, args...);
    // verify
    verify_data(name, rit, test_tensor, data_h, args...);
  }

  {
    fill_tensor(test_tensor, name);
    // make host copy BEFORE running unique on it since unique modifies it
    auto data_h = create_host_space_copy(test_tensor);

    // run unique
    auto rit = KE::unique("label", exespace(), test_tensor, args...);
    // verify
    verify_data(name, rit, test_tensor, data_h, args...);
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 11},
      {"small-b", 13},       {"medium-a", 11103},   {"medium-b", 21103},
      {"large-a", 101513},   {"large-b", 100111}};

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it);

    using func_t = IsEqualFunctor<ValueType>;
    run_single_scenario<Tag, ValueType>(it, func_t());
  }
}

TEST_CASE("std_algorithms_mod_seq_ops, unique") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

}  // namespace Unique
}  // namespace stdalgos
}  // namespace Test
