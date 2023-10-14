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
namespace UniqueCopy {

namespace KE = flare::experimental;

// impl is here for std because it is only avail from c++>=17
template <class InputIterator, class OutputIterator, class BinaryPredicate>
auto my_unique_copy(InputIterator first, InputIterator last,
                    OutputIterator result, BinaryPredicate pred) {
  if (first != last) {
    typename OutputIterator::value_type t(*first);
    *result = t;
    ++result;
    while (++first != last) {
      if (!pred(t, *first)) {
        t       = *first;
        *result = t;
        ++result;
      }
    }
  }
  return result;
}

template <class InputIterator, class OutputIterator>
auto my_unique_copy(InputIterator first, InputIterator last,
                    OutputIterator result) {
  using value_type = typename OutputIterator::value_type;
  using func_t     = IsEqualFunctor<value_type>;
  return my_unique_copy(first, last, result, func_t());
}

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
std::size_t fill_tensor(TensorType dest_tensor, const std::string& name) {
  using value_type      = typename TensorType::value_type;
  using exe_space       = typename TensorType::execution_space;
  const std::size_t ext = dest_tensor.extent(0);
  using aux_tensor_t      = flare::Tensor<value_type*, exe_space>;
  aux_tensor_t aux_tensor("aux_tensor", ext);
  auto v_h = create_mirror_tensor(flare::HostSpace(), aux_tensor);

  std::size_t count = 0;
  if (name == "empty") {
    // no op
  }

  else if (name == "one-element-a") {
    v_h(0) = static_cast<value_type>(1);
    count  = 1;
  }

  else if (name == "one-element-b") {
    v_h(0) = static_cast<value_type>(2);
    count  = 1;
  }

  else if (name == "two-elements-a") {
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(2);
    count  = 2;
  }

  else if (name == "two-elements-b") {
    v_h(0) = static_cast<value_type>(2);
    v_h(1) = static_cast<value_type>(-1);
    count  = 2;
  }

  else if (name == "small-a") {
    v_h(0)  = static_cast<value_type>(0);
    v_h(1)  = static_cast<value_type>(1);
    v_h(2)  = static_cast<value_type>(1);
    v_h(3)  = static_cast<value_type>(2);
    v_h(4)  = static_cast<value_type>(3);
    v_h(5)  = static_cast<value_type>(4);
    v_h(6)  = static_cast<value_type>(4);
    v_h(7)  = static_cast<value_type>(4);
    v_h(8)  = static_cast<value_type>(5);
    v_h(9)  = static_cast<value_type>(6);
    v_h(10) = static_cast<value_type>(6);
    count   = 7;
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
    count   = 9;
  }

  else if (name == "medium" || name == "large") {
    UnifDist<value_type> randObj;
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
    std::vector<value_type> tmp(v_h.extent(0));
    std::fill(tmp.begin(), tmp.end(), static_cast<value_type>(0));
    using func_t = IsEqualFunctor<value_type>;
    auto std_r =
        my_unique_copy(KE::cbegin(v_h), KE::cend(v_h), tmp.begin(), func_t());
    count = (std::size_t)(std_r - tmp.begin());
  }

  else {
    throw std::runtime_error("invalid choice");
  }

  flare::deep_copy(aux_tensor, v_h);
  CopyFunctor<aux_tensor_t, TensorType> F1(aux_tensor, dest_tensor);
  flare::parallel_for("copy", dest_tensor.extent(0), F1);
  return count;
}

template <class TensorTypeFrom, class TensorTypeTest, class... Args>
void verify_data(const std::string& name, TensorTypeFrom tensor_from,
                 TensorTypeTest tensor_test, Args... args) {
  using value_type = typename TensorTypeTest::value_type;

  //! always careful because tensors might not be deep copyable
  auto tensor_test_dc = create_deep_copyable_compatible_clone(tensor_test);
  auto tensor_test_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), tensor_test_dc);

  auto tensor_from_dc = create_deep_copyable_compatible_clone(tensor_from);
  auto trnsor_from_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), tensor_from_dc);

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element-a") {
    REQUIRE_EQ(tensor_test_h(0), static_cast<value_type>(1));
  }

  else if (name == "one-element-b") {
    REQUIRE_EQ(tensor_test_h(0), static_cast<value_type>(2));
  }

  else if (name == "two-elements-a") {
    REQUIRE_EQ(tensor_test_h(0), static_cast<value_type>(1));
    REQUIRE_EQ(tensor_test_h(1), static_cast<value_type>(2));
  }

  else if (name == "two-elements-b") {
    REQUIRE_EQ(tensor_test_h(0), static_cast<value_type>(2));
    REQUIRE_EQ(tensor_test_h(1), static_cast<value_type>(-1));
  }

  else if (name == "small-a") {
    REQUIRE_EQ(tensor_test_h(0), static_cast<value_type>(0));
    REQUIRE_EQ(tensor_test_h(1), static_cast<value_type>(1));
    REQUIRE_EQ(tensor_test_h(2), static_cast<value_type>(2));
    REQUIRE_EQ(tensor_test_h(3), static_cast<value_type>(3));
    REQUIRE_EQ(tensor_test_h(4), static_cast<value_type>(4));
    REQUIRE_EQ(tensor_test_h(5), static_cast<value_type>(5));
    REQUIRE_EQ(tensor_test_h(6), static_cast<value_type>(6));
    REQUIRE_EQ(tensor_test_h(7), static_cast<value_type>(0));
    REQUIRE_EQ(tensor_test_h(8), static_cast<value_type>(0));
    REQUIRE_EQ(tensor_test_h(9), static_cast<value_type>(0));
    REQUIRE_EQ(tensor_test_h(10), static_cast<value_type>(0));
  }

  else if (name == "small-b") {
    REQUIRE_EQ(tensor_test_h(0), static_cast<value_type>(1));
    REQUIRE_EQ(tensor_test_h(1), static_cast<value_type>(2));
    REQUIRE_EQ(tensor_test_h(2), static_cast<value_type>(3));
    REQUIRE_EQ(tensor_test_h(3), static_cast<value_type>(4));
    REQUIRE_EQ(tensor_test_h(4), static_cast<value_type>(5));
    REQUIRE_EQ(tensor_test_h(5), static_cast<value_type>(6));
    REQUIRE_EQ(tensor_test_h(6), static_cast<value_type>(8));
    REQUIRE_EQ(tensor_test_h(7), static_cast<value_type>(9));
    REQUIRE_EQ(tensor_test_h(8), static_cast<value_type>(8));
    REQUIRE_EQ(tensor_test_h(9), static_cast<value_type>(0));
    REQUIRE_EQ(tensor_test_h(10), static_cast<value_type>(0));
    REQUIRE_EQ(tensor_test_h(11), static_cast<value_type>(0));
    REQUIRE_EQ(tensor_test_h(12), static_cast<value_type>(0));
  }

  else if (name == "medium" || name == "large") {
    std::vector<value_type> tmp(tensor_test_h.extent(0));
    std::fill(tmp.begin(), tmp.end(), static_cast<value_type>(0));

    auto std_r = my_unique_copy(KE::cbegin(trnsor_from_h), KE::cend(trnsor_from_h),
                                tmp.begin(), args...);
    (void)std_r;

    for (std::size_t i = 0; i < trnsor_from_h.extent(0); ++i) {
      REQUIRE_EQ(tensor_test_h(i), tmp[i]);
    }
  }

  else {
    throw std::runtime_error("invalid choice");
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType>
void print_scenario_details(const std::string& name) {
  std::cout << "unique_copy: default predicate: " << name << ", "
            << tensor_tag_to_string(Tag{}) << " "
            << value_type_to_string(ValueType()) << '\n';
}

template <class Tag, class ValueType, class Predicate>
void print_scenario_details(const std::string& name, Predicate pred) {
  (void)pred;
  std::cout << "unique_copy: custom  predicate: " << name << ", "
            << tensor_tag_to_string(Tag{}) << " "
            << value_type_to_string(ValueType()) << '\n';
}

template <class Tag, class ValueType, class InfoType, class... Args>
void run_single_scenario(const InfoType& scenario_info, Args... args) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // print_scenario_details<Tag, ValueType>(name, args...);

  auto tensor_from = create_tensor<ValueType>(Tag{}, tensor_ext, "unique_copy_from");
  auto n         = fill_tensor(tensor_from, name);

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "unique_copy_dest");
    auto rit =
        KE::unique_copy(exespace(), KE::cbegin(tensor_from), KE::cend(tensor_from),
                        KE::begin(tensor_dest), args...);
    verify_data(name, tensor_from, tensor_dest, args...);
    REQUIRE_EQ(rit, (KE::begin(tensor_dest) + n));
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "unique_copy_dest");
    auto rit =
        KE::unique_copy("label", exespace(), KE::cbegin(tensor_from),
                        KE::cend(tensor_from), KE::begin(tensor_dest), args...);
    verify_data(name, tensor_from, tensor_dest, args...);
    REQUIRE_EQ(rit, (KE::begin(tensor_dest) + n));
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "unique_copy_dest");
    auto rit = KE::unique_copy(exespace(), tensor_from, tensor_dest, args...);
    verify_data(name, tensor_from, tensor_dest, args...);
    REQUIRE_EQ(rit, (KE::begin(tensor_dest) + n));
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "unique_copy_dest");
    auto rit =
        KE::unique_copy("label", exespace(), tensor_from, tensor_dest, args...);
    verify_data(name, tensor_from, tensor_dest, args...);
    REQUIRE_EQ(rit, (KE::begin(tensor_dest) + n));
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 11},
      {"small-b", 13},       {"medium", 21103},     {"large", 101513}};

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it);

    using func_t = IsEqualFunctor<ValueType>;
    run_single_scenario<Tag, ValueType>(it, func_t());
  }
}

TEST_CASE("std_algorithms_mod_seq_ops, unique_copy") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

}  // namespace UniqueCopy
}  // namespace stdalgos
}  // namespace Test
