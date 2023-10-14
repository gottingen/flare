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
namespace ReplaceIf {

namespace KE = flare::experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-20., 20.) { m_gen.seed(1034343); }

  double operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(-100, 100) { m_gen.seed(1034343); }

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

  UnifDist<value_type> randObj;
  if (name == "empty") {
    // no op
  }

  else if (name == "one-element") {
    v_h(0) = static_cast<value_type>(1);
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
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
    v_h(5) = static_cast<value_type>(-2);
  }

  else if (name == "medium" || name == "large") {
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

// my own because std::replace_if is ONLY found with std=c++20
template <class ForwardIt, class UnaryPredicate, class T>
void my_host_replace_if(ForwardIt first, ForwardIt last, UnaryPredicate p,
                        const T& new_value) {
  for (; first != last; ++first) {
    if (p(*first)) {
      *first = new_value;
    }
  }
}

template <class TensorType1, class TensorType2, class ValueType,
          class PredicateType>
void verify_data(TensorType1 data_tensor,  // contains data
                 TensorType2 test_tensor,  // the tensor to test
                 ValueType new_value, PredicateType pred) {
  //! always careful because tensors might not be deep copyable

  auto data_tensor_dc = create_deep_copyable_compatible_clone(data_tensor);
  auto data_tensor_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), data_tensor_dc);
  my_host_replace_if(KE::begin(data_tensor_h), KE::end(data_tensor_h), pred,
                     new_value);

  auto test_tensor_dc = create_deep_copyable_compatible_clone(test_tensor);
  auto test_tensor_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), test_tensor_dc);

  if (test_tensor_h.extent(0) > 0) {
    for (std::size_t i = 0; i < test_tensor_h.extent(0); ++i) {
      // std::cout << i << " " << std::setprecision(15)
      // 		<< data_tensor_dc(i) << " "
      // 		<< data_tensor_h(i) << " "
      // 		<< test_tensor_h(i) << std::endl;
      REQUIRE_EQ(data_tensor_h(i), test_tensor_h(i));
    }
  }
}

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType, class InfoType, class PredicateType>
void run_single_scenario(const InfoType& scenario_info, PredicateType pred) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // std::cout << "replace_if: " << name << ", " << tensor_tag_to_string(Tag{})
  //           << ", " << value_type_to_string(ValueType()) << std::endl;

  ValueType new_value{23};
  auto tensor_with_data_1d =
      create_tensor<ValueType>(Tag{}, tensor_ext, "replace_if_v2");
  auto tensor_to_test = create_tensor<ValueType>(Tag{}, tensor_ext, "replace_if_v1");
  fill_tensor(tensor_with_data_1d, name);

  {
    CopyFunctor<decltype(tensor_with_data_1d), decltype(tensor_to_test)> F1(
        tensor_with_data_1d, tensor_to_test);
    flare::parallel_for("copy", tensor_to_test.extent(0), F1);

    KE::replace_if(exespace(), KE::begin(tensor_to_test), KE::end(tensor_to_test),
                   pred, new_value);
    verify_data(tensor_with_data_1d, tensor_to_test, new_value, pred);
  }

  {
    CopyFunctor<decltype(tensor_with_data_1d), decltype(tensor_to_test)> F1(
        tensor_with_data_1d, tensor_to_test);
    flare::parallel_for("copy", tensor_to_test.extent(0), F1);

    KE::replace_if("label", exespace(), KE::begin(tensor_to_test),
                   KE::end(tensor_to_test), pred, new_value);
    verify_data(tensor_with_data_1d, tensor_to_test, new_value, pred);
  }

  {
    CopyFunctor<decltype(tensor_with_data_1d), decltype(tensor_to_test)> F1(
        tensor_with_data_1d, tensor_to_test);
    flare::parallel_for("copy", tensor_to_test.extent(0), F1);

    KE::replace_if(exespace(), tensor_to_test, pred, new_value);
    verify_data(tensor_with_data_1d, tensor_to_test, new_value, pred);
  }

  {
    CopyFunctor<decltype(tensor_with_data_1d), decltype(tensor_to_test)> F1(
        tensor_with_data_1d, tensor_to_test);
    flare::parallel_for("copy", tensor_to_test.extent(0), F1);

    KE::replace_if("label", exespace(), tensor_to_test, pred, new_value);
    verify_data(tensor_with_data_1d, tensor_to_test, new_value, pred);
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element", 1}, {"two-elements-a", 2},
      {"two-elements-b", 2}, {"small-a", 9},     {"small-b", 13},
      {"medium", 1103},      {"large", 101513}};

  for (const auto& it : scenarios) {
    using pred_p_t = IsPositiveFunctor<ValueType>;
    run_single_scenario<Tag, ValueType>(it, pred_p_t{});
    using pred_n_t = IsNegativeFunctor<ValueType>;
    run_single_scenario<Tag, ValueType>(it, pred_n_t{});
  }
}

TEST_CASE("std_algorithms_replace_ops_test, replace_if") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

}  // namespace ReplaceIf
}  // namespace stdalgos
}  // namespace Test
