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

namespace KE = flare::experimental;

namespace Test {
namespace stdalgos {

template <class ValueType>
struct TimesTwoUnaryTransformFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType& a) const { return (a * 2.); }
};

template <class ValueType>
struct MultiplyAndHalveBinaryTransformFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a * b) * 0.5;
  }
};

template <class ValueType>
struct SumJoinFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return a + b;
  }
};

struct std_algorithms_numerics_test {
  flare::LayoutStride layout{20, 2};
    std_algorithms_numerics_test() = default;
    ~std_algorithms_numerics_test() = default;
  // value_type
  using static_tensor_t  = flare::Tensor<value_type[20]>;
  using dyn_tensor_t     = flare::Tensor<value_type*>;
  using strided_tensor_t = flare::Tensor<value_type*, flare::LayoutStride>;

  static_tensor_t m_static_tensor{"std-algo-test-1D-contiguous-tensor-static"};
  dyn_tensor_t m_dynamic_tensor{"std-algo-test-1D-contiguous-tensor-dyn", 20};
  strided_tensor_t m_strided_tensor{"std-algo-test-1D-strided-tensor", layout};

  // custom scalar (cs)
  using static_tensor_cs_t = flare::Tensor<CustomValueType[20]>;
  using dyn_tensor_cs_t    = flare::Tensor<CustomValueType*>;
  using strided_tensor_cs_t =
      flare::Tensor<CustomValueType*, flare::LayoutStride>;

  static_tensor_cs_t m_static_tensor_cs{
      "std-algo-test-1D-contiguous-tensor-static-custom-scalar"};
  dyn_tensor_cs_t m_dynamic_tensor_cs{
      "std-algo-test-1D-contiguous-tensor-dyn-custom_scalar", 20};
  strided_tensor_cs_t m_strided_tensor_cs{
      "std-algo-test-1D-strided-tensor-custom-scalar", layout};

  template <class TensorFromType, class TensorToType>
  void copyPoDTensorToCustom(TensorFromType v_from, TensorToType v_to) {
    for (std::size_t i = 0; i < v_from.extent(0); ++i) {
      v_to(i)() = v_from(i);
    }
  }

  void fillFixtureTensors() {
    static_tensor_t tmpTensor("tmpTensor");
    static_tensor_cs_t tmpTensorCs("tmpTensorCs");
    auto tmp_tensor_h = flare::create_mirror_tensor(flare::HostSpace(), tmpTensor);
    auto tmp_tensor_cs_h =
        flare::create_mirror_tensor(flare::HostSpace(), tmpTensorCs);
    tmp_tensor_h(0)  = 0.;
    tmp_tensor_h(1)  = 0.;
    tmp_tensor_h(2)  = 0.;
    tmp_tensor_h(3)  = 2.;
    tmp_tensor_h(4)  = 2.;
    tmp_tensor_h(5)  = 1.;
    tmp_tensor_h(6)  = 1.;
    tmp_tensor_h(7)  = 1.;
    tmp_tensor_h(8)  = 1.;
    tmp_tensor_h(9)  = 0.;
    tmp_tensor_h(10) = -2.;
    tmp_tensor_h(11) = -2.;
    tmp_tensor_h(12) = 0.;
    tmp_tensor_h(13) = 2.;
    tmp_tensor_h(14) = 2.;
    tmp_tensor_h(15) = 1.;
    tmp_tensor_h(16) = 1.;
    tmp_tensor_h(17) = 1.;
    tmp_tensor_h(18) = 1.;
    tmp_tensor_h(19) = 0.;

    copyPoDTensorToCustom(tmp_tensor_h, tmp_tensor_cs_h);

    flare::deep_copy(tmpTensor, tmp_tensor_h);
    flare::deep_copy(tmpTensorCs, tmp_tensor_cs_h);

    CopyFunctor<static_tensor_t, static_tensor_t> F1(tmpTensor, m_static_tensor);
    flare::parallel_for("_std_algo_copy1", 20, F1);

    CopyFunctor<static_tensor_t, dyn_tensor_t> F2(tmpTensor, m_dynamic_tensor);
    flare::parallel_for("_std_algo_copy2", 20, F2);

    CopyFunctor<static_tensor_t, strided_tensor_t> F3(tmpTensor, m_strided_tensor);
    flare::parallel_for("_std_algo_copy3", 20, F3);

    CopyFunctor<static_tensor_cs_t, static_tensor_cs_t> F4(tmpTensorCs,
                                                       m_static_tensor_cs);
    flare::parallel_for("_std_algo_copy4", 20, F4);

    CopyFunctor<static_tensor_cs_t, dyn_tensor_cs_t> F5(tmpTensorCs,
                                                    m_dynamic_tensor_cs);
    flare::parallel_for("_std_algo_copy5", 20, F5);

    CopyFunctor<static_tensor_cs_t, strided_tensor_cs_t> F6(tmpTensorCs,
                                                        m_strided_tensor_cs);
    flare::parallel_for("_std_algo_copy6", 20, F6);
  }
};


// -------------------------------------------------------------------
// test default case of transform_reduce
//
// test for both POD types and custom scalar types
// -------------------------------------------------------------------
template <class ExecutionSpace, class TensorType1, class TensorType2,
          class ValueType>
void run_and_check_transform_reduce_default(TensorType1 first_tensor,
                                            TensorType2 second_tensor,
                                            ValueType init_value,
                                            ValueType result_value) {
  // trivial cases
  const auto r1 = KE::transform_reduce(ExecutionSpace(), KE::cbegin(first_tensor),
                                       KE::cbegin(first_tensor),
                                       KE::cbegin(second_tensor), init_value);

  const auto r2 = KE::transform_reduce(
      "MYLABEL", ExecutionSpace(), KE::cbegin(first_tensor),
      KE::cbegin(first_tensor), KE::cbegin(second_tensor), init_value);
  REQUIRE_EQ(r1, init_value);
  REQUIRE_EQ(r2, init_value);

  // non-trivial cases
  const auto r3 = KE::transform_reduce(ExecutionSpace(), KE::cbegin(first_tensor),
                                       KE::cend(first_tensor),
                                       KE::cbegin(second_tensor), init_value);

  const auto r4 = KE::transform_reduce(
      "MYLABEL", ExecutionSpace(), KE::cbegin(first_tensor), KE::cend(first_tensor),
      KE::cbegin(second_tensor), init_value);

  const auto r5 = KE::transform_reduce(ExecutionSpace(), first_tensor,
                                       second_tensor, init_value);
  const auto r6 = KE::transform_reduce("MYLABEL", ExecutionSpace(), first_tensor,
                                       second_tensor, init_value);

  REQUIRE_EQ(r3, result_value);
  REQUIRE_EQ(r4, result_value);
  REQUIRE_EQ(r5, result_value);
  REQUIRE_EQ(r6, result_value);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "transform_reduce_default_functors_using_pod_value_type") {
  fillFixtureTensors();
  const value_type init0 = 0.;
  const value_type init5 = 5.;
  const value_type gold0 = 32.;
  const value_type gold5 = 37.;

  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor, m_dynamic_tensor, init0, gold0);
  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor, m_dynamic_tensor, init5, gold5);

  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor, m_strided_tensor, init0, gold0);
  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor, m_strided_tensor, init5, gold5);

  run_and_check_transform_reduce_default<exespace>(
      m_dynamic_tensor, m_strided_tensor, init0, gold0);
  run_and_check_transform_reduce_default<exespace>(
      m_dynamic_tensor, m_strided_tensor, init5, gold5);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "transform_reduce_default_functors_using_custom_value_type") {
  fillFixtureTensors();
  const CustomValueType init0{0.};
  const CustomValueType init5{5.};
  const CustomValueType gold0{32.};
  const CustomValueType gold5{37.};

  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor_cs, m_dynamic_tensor_cs, init0, gold0);
  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor_cs, m_dynamic_tensor_cs, init5, gold5);

  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor_cs, m_strided_tensor_cs, init0, gold0);
  run_and_check_transform_reduce_default<exespace>(
      m_static_tensor_cs, m_strided_tensor_cs, init5, gold5);

  run_and_check_transform_reduce_default<exespace>(
      m_dynamic_tensor_cs, m_strided_tensor_cs, init0, gold0);
  run_and_check_transform_reduce_default<exespace>(
      m_dynamic_tensor_cs, m_strided_tensor_cs, init5, gold5);
}

// -------------------------------------------------------------------
// transform_reduce for custom joiner and custom transform op
// test for both POD types and custom scalar types
//
// test overload1 accepting two intervals
//
// Note that in the std, the reducer is called BinaryReductionOp
// but in the flare naming convention, it corresponds to a "joiner"
// that knows how to join two values.
// the "joiner" is assumed to be commutative:
//
// https://en.cppreference.com/w/cpp/algorithm/transform_reduce
//
// -------------------------------------------------------------------

template <class ExecutionSpace, class TensorType1, class TensorType2,
          class ValueType, class... Args>
void run_and_check_transform_reduce_overloadA(TensorType1 first_tensor,
                                              TensorType2 second_tensor,
                                              ValueType init_value,
                                              ValueType result_value,
                                              Args&&... args) {
  // trivial cases
  const auto r1 = KE::transform_reduce(
      ExecutionSpace(), KE::cbegin(first_tensor), KE::cbegin(first_tensor),
      KE::cbegin(second_tensor), init_value, std::forward<Args>(args)...);

  const auto r2 =
      KE::transform_reduce("MYLABEL", ExecutionSpace(), KE::cbegin(first_tensor),
                           KE::cbegin(first_tensor), KE::cbegin(second_tensor),
                           init_value, std::forward<Args>(args)...);

  REQUIRE_EQ(r1, init_value);
  REQUIRE_EQ(r2, init_value);

  // non trivial cases
  const auto r3 = KE::transform_reduce(
      ExecutionSpace(), KE::cbegin(first_tensor), KE::cend(first_tensor),
      KE::cbegin(second_tensor), init_value, std::forward<Args>(args)...);

  const auto r4 = KE::transform_reduce(
      "MYLABEL", ExecutionSpace(), KE::cbegin(first_tensor), KE::cend(first_tensor),
      KE::cbegin(second_tensor), init_value, std::forward<Args>(args)...);

  const auto r5 =
      KE::transform_reduce(ExecutionSpace(), first_tensor, second_tensor,
                           init_value, std::forward<Args>(args)...);
  const auto r6 =
      KE::transform_reduce("MYLABEL", ExecutionSpace(), first_tensor, second_tensor,
                           init_value, std::forward<Args>(args)...);

  REQUIRE_EQ(r3, result_value);
  REQUIRE_EQ(r4, result_value);
  REQUIRE_EQ(r5, result_value);
  REQUIRE_EQ(r6, result_value);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "transform_reduce_custom_functors_overloadA_using_pod_value_type") {
  using joiner_type = SumJoinFunctor<value_type>;
  using transf_type = MultiplyAndHalveBinaryTransformFunctor<value_type>;

  const value_type init0 = 0.;
  const value_type init5 = 5.;
  const value_type gold0 = 16.;
  const value_type gold5 = 21.;

  fillFixtureTensors();
  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor, m_dynamic_tensor, init0, gold0, joiner_type(),
      transf_type());
  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor, m_dynamic_tensor, init5, gold5, joiner_type(),
      transf_type());

  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor, m_strided_tensor, init0, gold0, joiner_type(),
      transf_type());
  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor, m_strided_tensor, init5, gold5, joiner_type(),
      transf_type());
  run_and_check_transform_reduce_overloadA<exespace>(
      m_dynamic_tensor, m_strided_tensor, init0, gold0, joiner_type(),
      transf_type());
  run_and_check_transform_reduce_overloadA<exespace>(
      m_dynamic_tensor, m_strided_tensor, init5, gold5, joiner_type(),
      transf_type());
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "transform_reduce_custom_functors_overloadA_using_custom_value_type") {
  using joiner_type = SumJoinFunctor<CustomValueType>;
  using transf_type = MultiplyAndHalveBinaryTransformFunctor<CustomValueType>;

  const CustomValueType init0{0.};
  const CustomValueType init5{5.};
  const CustomValueType gold0{16.};
  const CustomValueType gold5{21.};

  fillFixtureTensors();
  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor_cs, m_dynamic_tensor_cs, init0, gold0, joiner_type(),
      transf_type());
  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor_cs, m_dynamic_tensor_cs, init5, gold5, joiner_type(),
      transf_type());

  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor_cs, m_strided_tensor_cs, init0, gold0, joiner_type(),
      transf_type());
  run_and_check_transform_reduce_overloadA<exespace>(
      m_static_tensor_cs, m_strided_tensor_cs, init5, gold5, joiner_type(),
      transf_type());

  run_and_check_transform_reduce_overloadA<exespace>(
      m_dynamic_tensor_cs, m_strided_tensor_cs, init0, gold0, joiner_type(),
      transf_type());
  run_and_check_transform_reduce_overloadA<exespace>(
      m_dynamic_tensor_cs, m_strided_tensor_cs, init5, gold5, joiner_type(),
      transf_type());
}

// -------------------------------------------------------------------
// transform_reduce for custom joiner and custom transform op
// test for both POD types and custom scalar types
//
// test overload1 accepting single interval/tensor
//
// Note that in the std, the reducer is called BinaryReductionOp
// but in the flare naming convention, it corresponds to a "joiner"
// that knows how to join two values.
// the "joiner" is assumed to be commutative:
//
// https://en.cppreference.com/w/cpp/algorithm/transform_reduce
//
// -------------------------------------------------------------------

template <class ExecutionSpace, class TensorType, class ValueType, class... Args>
void run_and_check_transform_reduce_overloadB(TensorType tensor,
                                              ValueType init_value,
                                              ValueType result_value,
                                              Args&&... args) {
  // trivial
  const auto r1 =
      KE::transform_reduce(ExecutionSpace(), KE::cbegin(tensor), KE::cbegin(tensor),
                           init_value, std::forward<Args>(args)...);

  const auto r2 = KE::transform_reduce("MYLABEL", ExecutionSpace(),
                                       KE::cbegin(tensor), KE::cbegin(tensor),
                                       init_value, std::forward<Args>(args)...);

  REQUIRE_EQ(r1, init_value);
  REQUIRE_EQ(r2, init_value);

  // non trivial
  const auto r3 =
      KE::transform_reduce(ExecutionSpace(), KE::cbegin(tensor), KE::cend(tensor),
                           init_value, std::forward<Args>(args)...);

  const auto r4 = KE::transform_reduce("MYLABEL", ExecutionSpace(),
                                       KE::cbegin(tensor), KE::cend(tensor),
                                       init_value, std::forward<Args>(args)...);
  const auto r5 = KE::transform_reduce(ExecutionSpace(), tensor, init_value,
                                       std::forward<Args>(args)...);

  const auto r6 = KE::transform_reduce("MYLABEL", ExecutionSpace(), tensor,
                                       init_value, std::forward<Args>(args)...);

  REQUIRE_EQ(r3, result_value);
  REQUIRE_EQ(r4, result_value);
  REQUIRE_EQ(r5, result_value);
  REQUIRE_EQ(r6, result_value);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "transform_reduce_custom_functors_overloadB_using_pod_value_type") {
  using joiner_type = SumJoinFunctor<value_type>;
  using transf_type = TimesTwoUnaryTransformFunctor<value_type>;

  const value_type init0 = 0.;
  const value_type init5 = 5.;
  const value_type gold0 = 24.;
  const value_type gold5 = 29.;

  fillFixtureTensors();
  run_and_check_transform_reduce_overloadB<exespace>(
      m_static_tensor, init0, gold0, joiner_type(), transf_type());
  run_and_check_transform_reduce_overloadB<exespace>(
      m_dynamic_tensor, init5, gold5, joiner_type(), transf_type());
  run_and_check_transform_reduce_overloadB<exespace>(
      m_strided_tensor, init0, gold0, joiner_type(), transf_type());
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "transform_reduce_custom_functors_overloadB_using_custom_value_type") {
  using joiner_type = SumJoinFunctor<CustomValueType>;
  using transf_type = TimesTwoUnaryTransformFunctor<CustomValueType>;

  const CustomValueType init0{0.};
  const CustomValueType init5{5.};
  const CustomValueType gold0{24.};
  const CustomValueType gold5{29.};

  fillFixtureTensors();
  run_and_check_transform_reduce_overloadB<exespace>(
      m_static_tensor_cs, init0, gold0, joiner_type(), transf_type());
  run_and_check_transform_reduce_overloadB<exespace>(
      m_dynamic_tensor_cs, init5, gold5, joiner_type(), transf_type());
  run_and_check_transform_reduce_overloadB<exespace>(
      m_strided_tensor_cs, init0, gold0, joiner_type(), transf_type());
}

// -------------------------------------------------------------------
// test reduce overload1
//
// test for both POD types and custom scalar types
// -------------------------------------------------------------------
template <class ExecutionSpace, class TensorType, class ValueType>
void run_and_check_reduce_overloadA(TensorType tensor, ValueType non_trivial_result,
                                    ValueType trivial_result) {
  // trivial cases
  const auto r1 =
      KE::reduce(ExecutionSpace(), KE::cbegin(tensor), KE::cbegin(tensor));
  const auto r2 = KE::reduce("MYLABEL", ExecutionSpace(), KE::cbegin(tensor),
                             KE::cbegin(tensor));
  REQUIRE_EQ(r1, trivial_result);
  REQUIRE_EQ(r2, trivial_result);

  // non trivial cases
  const auto r3 =
      KE::reduce(ExecutionSpace(), KE::cbegin(tensor), KE::cend(tensor));
  const auto r4 =
      KE::reduce("MYLABEL", ExecutionSpace(), KE::cbegin(tensor), KE::cend(tensor));
  const auto r5 = KE::reduce(ExecutionSpace(), tensor);
  const auto r6 = KE::reduce("MYLABEL", ExecutionSpace(), tensor);

  REQUIRE_EQ(r3, non_trivial_result);
  REQUIRE_EQ(r4, non_trivial_result);
  REQUIRE_EQ(r5, non_trivial_result);
  REQUIRE_EQ(r6, non_trivial_result);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "reduce_default_functors_overloadA_using_pod_value_type") {
  fillFixtureTensors();
  const value_type trivial_gold     = 0.;
  const value_type non_trivial_gold = 12.;
  run_and_check_reduce_overloadA<exespace>(m_static_tensor, non_trivial_gold,
                                           trivial_gold);
  run_and_check_reduce_overloadA<exespace>(m_dynamic_tensor, non_trivial_gold,
                                           trivial_gold);
  run_and_check_reduce_overloadA<exespace>(m_strided_tensor, non_trivial_gold,
                                           trivial_gold);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test,"reduce_default_functors_overloadA_using_custom_value_type") {
  fillFixtureTensors();
  const CustomValueType trivial_gold{0.};
  const CustomValueType non_trivial_gold{12.};
  run_and_check_reduce_overloadA<exespace>(m_static_tensor_cs, non_trivial_gold,
                                           trivial_gold);
  run_and_check_reduce_overloadA<exespace>(m_dynamic_tensor_cs, non_trivial_gold,
                                           trivial_gold);
  run_and_check_reduce_overloadA<exespace>(m_strided_tensor_cs, non_trivial_gold,
                                           trivial_gold);
}

// -------------------------------------------------------------------
// test reduce overload2 with init value
//
// test for both POD types and custom scalar types
// -------------------------------------------------------------------
template <class ExecutionSpace, class TensorType, class ValueType>
void run_and_check_reduce_overloadB(TensorType tensor, ValueType result_value,
                                    ValueType init_value) {
  // trivial cases
  const auto r1 = KE::reduce(ExecutionSpace(), KE::cbegin(tensor),
                             KE::cbegin(tensor), init_value);
  const auto r2 = KE::reduce("MYLABEL", ExecutionSpace(), KE::cbegin(tensor),
                             KE::cbegin(tensor), init_value);
  REQUIRE_EQ(r1, init_value);
  REQUIRE_EQ(r2, init_value);

  // non trivial cases
  const auto r3 = KE::reduce(ExecutionSpace(), KE::cbegin(tensor), KE::cend(tensor),
                             init_value);
  const auto r4 = KE::reduce("MYLABEL", ExecutionSpace(), KE::cbegin(tensor),
                             KE::cend(tensor), init_value);
  const auto r5 = KE::reduce(ExecutionSpace(), tensor, init_value);
  const auto r6 = KE::reduce("MYLABEL", ExecutionSpace(), tensor, init_value);

  REQUIRE_EQ(r3, result_value);
  REQUIRE_EQ(r4, result_value);
  REQUIRE_EQ(r5, result_value);
  REQUIRE_EQ(r6, result_value);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "reduce_default_functors_overloadB_using_pod_value_type") {
  fillFixtureTensors();
  const value_type init = 5.;
  const value_type gold = 17.;
  run_and_check_reduce_overloadB<exespace>(m_static_tensor, gold, init);
  run_and_check_reduce_overloadB<exespace>(m_dynamic_tensor, gold, init);
  run_and_check_reduce_overloadB<exespace>(m_strided_tensor, gold, init);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "reduce_default_functors_overloadB_using_custom_value_type") {
  fillFixtureTensors();
  const CustomValueType init{5.};
  const CustomValueType gold{17.};
  run_and_check_reduce_overloadB<exespace>(m_static_tensor_cs, gold, init);
  run_and_check_reduce_overloadB<exespace>(m_dynamic_tensor_cs, gold, init);
  run_and_check_reduce_overloadB<exespace>(m_strided_tensor_cs, gold, init);
}

// -------------------------------------------------------------------
// test reduce overload3 with init value
//
// test for both POD types and custom scalar types
// -------------------------------------------------------------------
template <class ExecutionSpace, class TensorType, class ValueType, class BinaryOp>
void run_and_check_reduce_overloadC(TensorType tensor, ValueType result_value,
                                    ValueType init_value, BinaryOp joiner) {
  // trivial cases
  const auto r1 = KE::reduce(ExecutionSpace(), KE::cbegin(tensor),
                             KE::cbegin(tensor), init_value, joiner);
  const auto r2 = KE::reduce("MYLABEL", ExecutionSpace(), KE::cbegin(tensor),
                             KE::cbegin(tensor), init_value, joiner);
  REQUIRE_EQ(r1, init_value);
  REQUIRE_EQ(r2, init_value);

  // non trivial cases
  const auto r3 = KE::reduce(ExecutionSpace(), KE::cbegin(tensor), KE::cend(tensor),
                             init_value, joiner);
  const auto r4 = KE::reduce("MYLABEL", ExecutionSpace(), KE::cbegin(tensor),
                             KE::cend(tensor), init_value, joiner);
  const auto r5 = KE::reduce(ExecutionSpace(), tensor, init_value, joiner);
  const auto r6 =
      KE::reduce("MYLABEL", ExecutionSpace(), tensor, init_value, joiner);

  REQUIRE_EQ(r3, result_value);
  REQUIRE_EQ(r4, result_value);
  REQUIRE_EQ(r5, result_value);
  REQUIRE_EQ(r6, result_value);
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test,"reduce_custom_functors_using_pod_value_type") {
  using joiner_type = SumJoinFunctor<value_type>;

  fillFixtureTensors();
  const value_type init = 5.;
  const value_type gold = 17.;
  run_and_check_reduce_overloadC<exespace>(m_static_tensor, gold, init,
                                           joiner_type());
  run_and_check_reduce_overloadC<exespace>(m_dynamic_tensor, gold, init,
                                           joiner_type());
  run_and_check_reduce_overloadC<exespace>(m_strided_tensor, gold, init,
                                           joiner_type());
}

TEST_CASE_FIXTURE(std_algorithms_numerics_test, "reduce_custom_functors_using_custom_value_type") {
  using joiner_type = SumJoinFunctor<CustomValueType>;

  fillFixtureTensors();
  const CustomValueType init{5.};
  const CustomValueType gold{17.};
  run_and_check_reduce_overloadC<exespace>(m_static_tensor_cs, gold, init,
                                           joiner_type());
  run_and_check_reduce_overloadC<exespace>(m_dynamic_tensor_cs, gold, init,
                                           joiner_type());
  run_and_check_reduce_overloadC<exespace>(m_strided_tensor_cs, gold, init,
                                           joiner_type());
}


}  // namespace stdalgos
}  // namespace Test
