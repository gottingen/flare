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
#include <iomanip>

namespace Test {
namespace stdalgos {
namespace TransformEScan {

namespace KE = flare::experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(0.05, 1.2) { m_gen.seed(1034343); }

  double operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(1, 3) { m_gen.seed(1034343); }

  int operator()() { return m_dist(m_gen); }
};

template <class TensorType>
void fill_zero(TensorType tensor) {
  flare::parallel_for(tensor.extent(0), FillZeroFunctor<TensorType>(tensor));
}

template <class TensorType>
void fill_tensor(TensorType dest_tensor, const std::string& name) {
  using value_type = typename TensorType::value_type;
  using exe_space  = typename TensorType::execution_space;

  const std::size_t ext = dest_tensor.extent(0);
  using aux_tensor_t      = flare::Tensor<value_type*, exe_space>;
  aux_tensor_t aux_tensor("aux_tensor", ext);
  auto v_h = create_mirror_tensor(flare::HostSpace(), aux_tensor);

  UnifDist<value_type> randObj;

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element") {
    assert(v_h.extent(0) == 1);
    v_h(0) = static_cast<value_type>(1);
  }

  else if (name == "two-elements-a") {
    assert(v_h.extent(0) == 2);
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-b") {
    assert(v_h.extent(0) == 2);
    v_h(0) = static_cast<value_type>(2);
    v_h(1) = static_cast<value_type>(-1);
  }

  else if (name == "small-a") {
    assert(v_h.extent(0) == 9);
    v_h(0) = static_cast<value_type>(3);
    v_h(1) = static_cast<value_type>(1);
    v_h(2) = static_cast<value_type>(4);
    v_h(3) = static_cast<value_type>(1);
    v_h(4) = static_cast<value_type>(5);
    v_h(5) = static_cast<value_type>(9);
    v_h(6) = static_cast<value_type>(2);
    v_h(7) = static_cast<value_type>(6);
    v_h(8) = static_cast<value_type>(2);
  }

  else if (name == "small-b") {
    assert(v_h.extent(0) >= 6);
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

// I had to write my own because std::transform_exclusive_scan is ONLY found
// with std=c++17
template <class it1, class it2, class ValType, class BopType, class UopType>
void my_host_transform_exclusive_scan(it1 first, it1 last, it2 dest,
                                      ValType init, BopType bop, UopType uop) {
  const auto num_elements = last - first;
  if (num_elements > 0) {
    while (first < last - 1) {
      *(dest++) = init;
      init      = bop(uop(*(first++)), init);
    }
    *dest = init;
  }
}

template <class TensorType1, class TensorType2, class ValueType, class BinaryOp,
          class UnaryOp>
void verify_data(TensorType1 data_tensor,  // contains data
                 TensorType2 test_tensor,  // the tensor to test
                 ValueType init_value, BinaryOp bop, UnaryOp uop) {
  //! always careful because tensors might not be deep copyable

  auto data_tensor_dc = create_deep_copyable_compatible_clone(data_tensor);
  auto data_tensor_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), data_tensor_dc);

  using gold_tensor_value_type = typename TensorType2::value_type;
  flare::Tensor<gold_tensor_value_type*, flare::HostSpace> gold_h(
      "goldh", data_tensor.extent(0));
  my_host_transform_exclusive_scan(KE::cbegin(data_tensor_h),
                                   KE::cend(data_tensor_h), KE::begin(gold_h),
                                   init_value, bop, uop);

  auto test_tensor_dc = create_deep_copyable_compatible_clone(test_tensor);
  auto test_tensor_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), test_tensor_dc);
  if (test_tensor_h.extent(0) > 0) {
    for (std::size_t i = 0; i < test_tensor_h.extent(0); ++i) {
      // std::cout << i << " " << std::setprecision(15) << data_tensor_h(i) << " "
      //           << gold_h(i) << " " << test_tensor_h(i) << " "
      //           << std::abs(gold_h(i) - test_tensor_h(i)) << std::endl;

      if (std::is_same<gold_tensor_value_type, int>::value) {
        REQUIRE_EQ(gold_h(i), test_tensor_h(i));
      } else {
        const auto error = std::abs(gold_h(i) - test_tensor_h(i));
        if (error > 1e-10) {
          std::cout << i << " " << std::setprecision(15) << data_tensor_h(i)
                    << " " << gold_h(i) << " " << test_tensor_h(i) << " "
                    << std::abs(gold_h(i) - test_tensor_h(i)) << std::endl;
        }
        REQUIRE_LT(error, 1e-10);
      }
    }
    // std::cout << " last el: " << test_tensor_h(test_tensor_h.extent(0)-1) <<
    // std::endl;
  }
}

template <class ValueType>
struct TimesTwoUnaryFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType& a) const { return (a * ValueType(2)); }
};

template <class ValueType>
struct SumBinaryFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a + b);
  }
};

std::string value_type_to_string(int) { return "int"; }

std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType, class InfoType, class BinaryOp,
          class UnaryOp>
void run_single_scenario(const InfoType& scenario_info, ValueType init_value,
                         BinaryOp bop, UnaryOp uop) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // std::cout << "transform_exclusive_scan custom op: " << name << ", "
  //           << tensor_tag_to_string(Tag{}) << ", "
  //           << value_type_to_string(ValueType()) << ", "
  //           << "init = " << init_value << std::endl;

  auto tensor_dest =
      create_tensor<ValueType>(Tag{}, tensor_ext, "transform_exclusive_scan");
  auto tensor_from =
      create_tensor<ValueType>(Tag{}, tensor_ext, "transform_exclusive_scan");
  fill_tensor(tensor_from, name);

  {
    fill_zero(tensor_dest);
    auto r = KE::transform_exclusive_scan(
        exespace(), KE::cbegin(tensor_from), KE::cend(tensor_from),
        KE::begin(tensor_dest), init_value, bop, uop);
    REQUIRE_EQ(r, KE::end(tensor_dest));
    verify_data(tensor_from, tensor_dest, init_value, bop, uop);
  }

  {
    fill_zero(tensor_dest);
    auto r = KE::transform_exclusive_scan(
        "label", exespace(), KE::cbegin(tensor_from), KE::cend(tensor_from),
        KE::begin(tensor_dest), init_value, bop, uop);
    REQUIRE_EQ(r, KE::end(tensor_dest));
    verify_data(tensor_from, tensor_dest, init_value, bop, uop);
  }

  {
    fill_zero(tensor_dest);
    auto r = KE::transform_exclusive_scan(exespace(), tensor_from, tensor_dest,
                                          init_value, bop, uop);
    REQUIRE_EQ(r, KE::end(tensor_dest));
    verify_data(tensor_from, tensor_dest, init_value, bop, uop);
  }

  {
    fill_zero(tensor_dest);
    auto r = KE::transform_exclusive_scan("label", exespace(), tensor_from,
                                          tensor_dest, init_value, bop, uop);
    REQUIRE_EQ(r, KE::end(tensor_dest));
    verify_data(tensor_from, tensor_dest, init_value, bop, uop);
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element", 1}, {"two-elements-a", 2},
      {"two-elements-b", 2}, {"small-a", 9},     {"small-b", 13},
      {"medium", 1103},      {"large", 10513}};

  for (const auto& it : scenarios) {
    using uop_t = TimesTwoUnaryFunctor<ValueType>;
    using bop_t = SumBinaryFunctor<ValueType>;
    run_single_scenario<Tag, ValueType>(it, ValueType{0}, bop_t(), uop_t());
    run_single_scenario<Tag, ValueType>(it, ValueType{1}, bop_t(), uop_t());
    run_single_scenario<Tag, ValueType>(it, ValueType{-2}, bop_t(), uop_t());
    run_single_scenario<Tag, ValueType>(it, ValueType{3}, bop_t(), uop_t());
  }
}

TEST_CASE("std_algorithms_numeric_ops_test, transform_exclusive_scan") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

template <class ValueType>
struct MultiplyFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a * b);
  }
};

TEST_CASE("std_algorithms_numeric_ops_test, transform_exclusive_scan_functor") {
  int dummy       = 0;
  using tensor_type = flare::Tensor<int*, exespace>;
  tensor_type dummy_tensor("dummy_tensor", 0);
  using unary_op_type =
      flare::experimental::detail::StdNumericScanIdentityReferenceUnaryFunctor<
          int>;
  using functor_type =
      flare::experimental::detail::TransformExclusiveScanFunctor<
          exespace, int, int, tensor_type, tensor_type, MultiplyFunctor<int>,
          unary_op_type>;
  functor_type functor(dummy, dummy_tensor, dummy_tensor, {}, {});
  using value_type = functor_type::value_type;

  value_type value1;
  functor.init(value1);
  REQUIRE_EQ(value1.val, 0);
  REQUIRE_EQ(value1.is_initial, true);

  value_type value2;
  value2.val        = 1;
  value2.is_initial = false;
  functor.join(value1, value2);
  REQUIRE_EQ(value1.val, 1);
  REQUIRE_EQ(value1.is_initial, false);

  functor.init(value1);
  functor.join(value2, value1);
  REQUIRE_EQ(value2.val, 1);
  REQUIRE_EQ(value2.is_initial, false);

  functor.init(value2);
  functor.join(value2, value1);
  REQUIRE_EQ(value2.val, 0);
  REQUIRE_EQ(value2.is_initial, true);

  value1.val        = 3;
  value1.is_initial = false;
  value2.val        = 2;
  value2.is_initial = false;
  functor.join(value2, value1);
  REQUIRE_EQ(value2.val, 6);
  REQUIRE_EQ(value2.is_initial, false);
}

}  // namespace TransformEScan
}  // namespace stdalgos
}  // namespace Test
