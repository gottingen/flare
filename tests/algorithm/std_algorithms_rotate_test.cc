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
namespace Rotate {

namespace KE = flare::experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-50, 50) { m_gen.seed(1034343); }
  int operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-90., 100.) { m_gen.seed(1034343); }

  double operator()() { return m_dist(m_gen); }
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
    v_h(2)  = static_cast<value_type>(1);
    v_h(3)  = static_cast<value_type>(-2);
    v_h(4)  = static_cast<value_type>(3);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(-40);
    v_h(7)  = static_cast<value_type>(4);
    v_h(8)  = static_cast<value_type>(5);
    v_h(9)  = static_cast<value_type>(62);
    v_h(10) = static_cast<value_type>(6);
  }

  else if (name == "small-b") {
    v_h(0)  = static_cast<value_type>(1);
    v_h(1)  = static_cast<value_type>(1);
    v_h(2)  = static_cast<value_type>(-1);
    v_h(3)  = static_cast<value_type>(2);
    v_h(4)  = static_cast<value_type>(-3);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(4);
    v_h(7)  = static_cast<value_type>(24);
    v_h(8)  = static_cast<value_type>(5);
    v_h(9)  = static_cast<value_type>(-46);
    v_h(10) = static_cast<value_type>(8);
    v_h(11) = static_cast<value_type>(9);
    v_h(12) = static_cast<value_type>(8);
  }

  else if (name == "medium" || name == "large") {
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

template <class TensorType, class ResultIt, class TensorHostType>
void verify_data(ResultIt result_it, TensorType tensor, TensorHostType data_tensor_host,
                 std::size_t rotation_point) {
  // run std::rotate
  auto n_it = KE::begin(data_tensor_host) + rotation_point;
  auto std_rit =
      std::rotate(KE::begin(data_tensor_host), n_it, KE::end(data_tensor_host));

  // make sure results match
  const auto my_diff  = result_it - KE::begin(tensor);
  const auto std_diff = std_rit - KE::begin(data_tensor_host);
  REQUIRE_EQ(my_diff, std_diff);

  // check tensors match
  auto tensor_h           = create_host_space_copy(tensor);
  const std::size_t ext = tensor_h.extent(0);
  for (std::size_t i = 0; i < ext; ++i) {
    REQUIRE_EQ(tensor_h(i), data_tensor_host[i]);
    // std::cout << "i= " << i << " "
    // 	      << "mine: " << tensor_h(i) << " "
    // 	      << "std: " << data_tensor_host(i)
    // 	      << '\n';
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType>
void print_scenario_details(const std::string& name,
                            std::size_t rotation_point) {
  std::cout << "rotate: "
            << " at " << rotation_point << ", " << name << ", "
            << tensor_tag_to_string(Tag{}) << ", "
            << value_type_to_string(ValueType()) << std::endl;
}

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info,
                         std::size_t rotation_point) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // print_scenario_details<Tag, ValueType>(name, rotation_point);

  {
    auto tensor = create_tensor<ValueType>(Tag{}, tensor_ext, "rotate_data_tensor");
    fill_tensor(tensor, name);
    // create host copy BEFORE rotate or tensor will be modified
    auto tensor_h = create_host_space_copy(tensor);
    auto n_it   = KE::begin(tensor) + rotation_point;
    auto rit    = KE::rotate(exespace(), KE::begin(tensor), n_it, KE::end(tensor));
    verify_data(rit, tensor, tensor_h, rotation_point);
  }

  {
    auto tensor = create_tensor<ValueType>(Tag{}, tensor_ext, "rotate_data_tensor");
    fill_tensor(tensor, name);
    // create host copy BEFORE rotate or tensor will be modified
    auto tensor_h = create_host_space_copy(tensor);
    auto n_it   = KE::begin(tensor) + rotation_point;
    auto rit =
        KE::rotate("label", exespace(), KE::begin(tensor), n_it, KE::end(tensor));
    verify_data(rit, tensor, tensor_h, rotation_point);
  }

  {
    auto tensor = create_tensor<ValueType>(Tag{}, tensor_ext, "rotate_data_tensor");
    fill_tensor(tensor, name);
    // create host copy BEFORE rotate or tensor will be modified
    auto tensor_h = create_host_space_copy(tensor);
    auto rit    = KE::rotate(exespace(), tensor, rotation_point);
    // verify_data(rit, tensor, tensor_h, rotation_point);
  }

  {
    auto tensor = create_tensor<ValueType>(Tag{}, tensor_ext, "rotate_data_tensor");
    fill_tensor(tensor, name);
    // create host copy BEFORE rotate or tensor will be modified
    auto tensor_h = create_host_space_copy(tensor);
    auto rit    = KE::rotate("label", exespace(), tensor, rotation_point);
    verify_data(rit, tensor, tensor_h, rotation_point);
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 11},
      {"small-b", 13},       {"medium", 21103},     {"large", 101513}};

  std::vector<std::size_t> rotation_points = {0,  1,   2,    3,     8,
                                              56, 101, 1003, 101501};

  for (const auto& it : scenarios) {
    for (const auto& it2 : rotation_points) {
      // for each tensor scenario, we rotate at multiple points
      // but only if the tensor has an extent that is >= rotation point
      const auto tensor_ext = it.second;
      if (tensor_ext >= it2) {
        run_single_scenario<Tag, ValueType>(it, it2);
      }
    }
  }
}

TEST_CASE("std_algorithms_mod_seq_ops, rotate") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
}

}  // namespace Rotate
}  // namespace stdalgos
}  // namespace Test
