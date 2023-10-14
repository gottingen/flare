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
namespace RemoveCopy {

namespace KE = flare::experimental;

constexpr int match_value = 4;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-100, 100) { m_gen.seed(1034343); }

  int operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-100., 100.) { m_gen.seed(341043); }

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
    v_h(0) = static_cast<value_type>(match_value);
  }

  else if (name == "two-elements-a") {
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(match_value);
  }

  else if (name == "two-elements-b") {
    v_h(0) = static_cast<value_type>(match_value);
    v_h(1) = static_cast<value_type>(-1);
  }

  else if (name == "small-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(i + 1);
    }
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      if (i % 2 == 0) {
        v_h(i) = static_cast<value_type>(match_value);
      } else {
        v_h(i) = static_cast<value_type>(-12);
      }
    }
  }

  else if (name == "medium" || name == "large") {
    UnifDist<value_type> randObj;
    for (std::size_t i = 0; i < ext; ++i) {
      if (i % 8 == 0) {
        v_h(i) = static_cast<value_type>(match_value);
      } else {
        v_h(i) = randObj();
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

template <class TensorFromType, class TensorDestType, class MyItResult>
void verify_data(TensorFromType tensor_from, TensorDestType tensor_dest,
                 MyItResult my_result) {
  // make a host copy of the tensor_from
  auto trnsor_from_h      = create_host_space_copy(tensor_from);
  const std::size_t ext = trnsor_from_h.extent(0);
  using value_type      = typename TensorFromType::value_type;

  // run std::remove_copy
  std::vector<value_type> gold_dest_std(ext);
  auto std_result =
      std::remove_copy(KE::cbegin(trnsor_from_h), KE::cend(trnsor_from_h),
                       gold_dest_std.begin(), (value_type)match_value);

  // check that returned iterators are correct
  const std::size_t std_diff = std_result - gold_dest_std.begin();
  const std::size_t my_diff  = my_result - KE::begin(tensor_dest);
  REQUIRE_EQ(std_diff, my_diff);

  // check the actual data after algo has been applied
  auto tensor_dest_h = create_host_space_copy(tensor_dest);
  for (std::size_t i = 0; i < my_diff; ++i) {
    REQUIRE_EQ(tensor_dest_h(i), gold_dest_std[i]);
    // std::cout << "i= " << i << " "
    // 	      << "mine: " << tensor_dest_h(i) << " "
    // 	      << "std: " << gold_dest_std[i]
    // 	      << '\n';
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // std::cout << "remove_copy: " << name << ", " << tensor_tag_to_string(Tag{})
  //           << ", " << value_type_to_string(ValueType()) << std::endl;

  {
    auto tensor_from =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_from");
    fill_tensor(tensor_from, name);

    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_dest");
    auto rit =
        KE::remove_copy(exespace(), KE::cbegin(tensor_from), KE::cend(tensor_from),
                        KE::begin(tensor_dest), (ValueType)match_value);
    verify_data(tensor_from, tensor_dest, rit);
  }

  {
    auto tensor_from =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_from");
    fill_tensor(tensor_from, name);

    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_dest");
    auto rit = KE::remove_copy("label", exespace(), KE::cbegin(tensor_from),
                               KE::cend(tensor_from), KE::begin(tensor_dest),
                               (ValueType)match_value);
    verify_data(tensor_from, tensor_dest, rit);
  }

  {
    auto tensor_from =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_from");
    fill_tensor(tensor_from, name);

    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_dest");
    auto rit = KE::remove_copy(exespace(), tensor_from, tensor_dest,
                               (ValueType)match_value);
    verify_data(tensor_from, tensor_dest, rit);
  }

  {
    auto tensor_from =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_from");
    fill_tensor(tensor_from, name);

    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "remove_copy_tensor_dest");
    auto rit = KE::remove_copy("label", exespace(), tensor_from, tensor_dest,
                               (ValueType)match_value);
    verify_data(tensor_from, tensor_dest, rit);
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 9},
      {"small-b", 13},       {"medium", 13031},     {"large", 101513}};

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it);
  }
}

TEST_CASE("std_algorithms_mod_seq_ops, remove_copy") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
}

}  // namespace RemoveCopy
}  // namespace stdalgos
}  // namespace Test
