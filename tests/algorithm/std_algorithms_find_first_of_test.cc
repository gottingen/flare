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
namespace FindFirstOf {

namespace KE = flare::experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-100, 100) { m_gen.seed(1034343); }
  UnifDist(int a, int b) : m_dist(a, b) { m_gen.seed(514343); }

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

  else if (name == "three-elements-a") {
    v_h(0) = static_cast<value_type>(-1);
    v_h(1) = static_cast<value_type>(2);
    v_h(2) = static_cast<value_type>(3);
  }

  else if (name == "three-elements-b") {
    v_h(0) = static_cast<value_type>(3);
    v_h(1) = static_cast<value_type>(1);
    v_h(2) = static_cast<value_type>(-4);
  }

  else if (name == "four-elements-a") {
    v_h(0) = static_cast<value_type>(-1);
    v_h(1) = static_cast<value_type>(2);
    v_h(2) = static_cast<value_type>(2);
    v_h(3) = static_cast<value_type>(4);
  }

  else if (name == "four-elements-b") {
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(1);
    v_h(2) = static_cast<value_type>(1);
    v_h(3) = static_cast<value_type>(1);
  }

  else if (name == "small-a") {
    v_h(0)  = static_cast<value_type>(0);
    v_h(1)  = static_cast<value_type>(4);
    v_h(2)  = static_cast<value_type>(1);
    v_h(3)  = static_cast<value_type>(2);
    v_h(4)  = static_cast<value_type>(-1);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(1);
    v_h(7)  = static_cast<value_type>(2);
    v_h(8)  = static_cast<value_type>(2);
    v_h(9)  = static_cast<value_type>(4);
    v_h(10) = static_cast<value_type>(1);
  }

  else if (name == "small-b") {
    v_h(0)  = static_cast<value_type>(1);
    v_h(1)  = static_cast<value_type>(2);
    v_h(2)  = static_cast<value_type>(3);
    v_h(3)  = static_cast<value_type>(1);
    v_h(4)  = static_cast<value_type>(-1);
    v_h(5)  = static_cast<value_type>(-2);
    v_h(6)  = static_cast<value_type>(0);
    v_h(7)  = static_cast<value_type>(1);
    v_h(8)  = static_cast<value_type>(2);
    v_h(9)  = static_cast<value_type>(2);
    v_h(10) = static_cast<value_type>(5);
    v_h(11) = static_cast<value_type>(9);
    v_h(12) = static_cast<value_type>(8);
  }

  else {
    UnifDist<value_type> randObj;
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
  }

  flare::deep_copy(aux_tensor, v_h);
  CopyFunctor<aux_tensor_t, TensorType> F1(aux_tensor, dest_tensor);
  flare::parallel_for("copy", dest_tensor.extent(0), F1);
}

template <class TensorType>
auto create_seq_for_find_first_of(TensorType data_tensor, std::size_t seq_extent) {
  (void)data_tensor;
  using value_type = typename TensorType::value_type;
  using exe_space  = typename TensorType::execution_space;
  using seq_tensor_t = flare::Tensor<value_type*, exe_space>;
  seq_tensor_t seq_tensor("seq_tensor", seq_extent);
  auto seq_tensor_h = create_mirror_tensor(flare::HostSpace(), seq_tensor);

  UnifDist<value_type> randObj(-10, -10);
  for (std::size_t i = 0; i < seq_extent; ++i) {
    seq_tensor_h(i) = randObj();
  }

  flare::deep_copy(seq_tensor, seq_tensor_h);
  return seq_tensor;
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType>
void print_scenario_details(const std::string& name, std::size_t seq_ext) {
  std::cout << "find_first_of: default predicate: " << name << ", "
            << "seach_seq_ext = " << seq_ext << ", "
            << tensor_tag_to_string(Tag{}) << " "
            << value_type_to_string(ValueType()) << std::endl;
}

template <class Tag, class ValueType, class Predicate>
void print_scenario_details(const std::string& name, std::size_t seq_ext,
                            Predicate pred) {
  (void)pred;
  std::cout << "find_first_of: custom  predicate: " << name << ", "
            << "seach_seq_ext = " << seq_ext << ", "
            << tensor_tag_to_string(Tag{}) << " "
            << value_type_to_string(ValueType()) << std::endl;
}

template <class Tag, class ValueType, class InfoType, class... Args>
void run_single_scenario(const InfoType& scenario_info, std::size_t seq_ext,
                         Args... args) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // print_scenario_details<Tag, ValueType>(name, seq_ext, args...);

  auto tensor =
      create_tensor<ValueType>(Tag{}, tensor_ext, "find_first_of_test_tensor");
  fill_tensor(tensor, name);
  auto s_tensor = create_seq_for_find_first_of(tensor, seq_ext);

  // run std
  auto tensor_h   = create_host_space_copy(tensor);
  auto s_tensor_h = create_host_space_copy(s_tensor);
  auto stdrit =
      std::find_first_of(KE::cbegin(tensor_h), KE::cend(tensor_h),
                         KE::cbegin(s_tensor_h), KE::cend(s_tensor_h), args...);

  {
    auto myrit =
        KE::find_first_of(exespace(), KE::cbegin(tensor), KE::cend(tensor),
                          KE::cbegin(s_tensor), KE::cend(s_tensor), args...);
    const auto mydiff  = myrit - KE::cbegin(tensor);
    const auto stddiff = stdrit - KE::cbegin(tensor_h);
    REQUIRE_EQ(mydiff, stddiff);
  }

  {
    auto myrit =
        KE::find_first_of("label", exespace(), KE::cbegin(tensor), KE::cend(tensor),
                          KE::cbegin(s_tensor), KE::cend(s_tensor), args...);
    const auto mydiff  = myrit - KE::cbegin(tensor);
    const auto stddiff = stdrit - KE::cbegin(tensor_h);
    REQUIRE_EQ(mydiff, stddiff);
  }

  {
    auto myrit         = KE::find_first_of(exespace(), tensor, s_tensor, args...);
    const auto mydiff  = myrit - KE::begin(tensor);
    const auto stddiff = stdrit - KE::cbegin(tensor_h);
    REQUIRE_EQ(mydiff, stddiff);
  }

  {
    auto myrit = KE::find_first_of("label", exespace(), tensor, s_tensor, args...);
    const auto mydiff  = myrit - KE::begin(tensor);
    const auto stddiff = stdrit - KE::cbegin(tensor_h);
    REQUIRE_EQ(mydiff, stddiff);
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {{"empty", 0},
                                                        {"one-element-a", 1},
                                                        {"one-element-b", 1},
                                                        {"two-elements-a", 2},
                                                        {"two-elements-b", 2},
                                                        {"three-elements-a", 3},
                                                        {"three-elements-b", 3},
                                                        {"four-elements-a", 4},
                                                        {"four-elements-b", 4},
                                                        {"small-a", 11},
                                                        {"small-b", 13},
                                                        {"medium-a", 11103},
                                                        {"medium-b", 21103},
                                                        {"large-a", 101513},
                                                        {"large-b", 100111}};

  const std::vector<std::size_t> seq_extents = {0,  1,  2,  3,   4,   5,   8,
                                                11, 20, 31, 113, 523, 1035};

  // for each scenario we want to run "find_first_of"
  // for a set of sequences of various extents
  for (const auto& it : scenarios) {
    for (const auto& it2 : seq_extents) {
      run_single_scenario<Tag, ValueType>(it, it2);

      using func_t = IsEqualFunctor<ValueType>;
      run_single_scenario<Tag, ValueType>(it, it2, func_t());
    }
  }
}

TEST_CASE("std_algorithms_non_mod_seq_ops, find_first_of") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

}  // namespace FindFirstOf
}  // namespace stdalgos
}  // namespace Test
