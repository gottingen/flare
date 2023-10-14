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

struct std_algorithms_mod_seq_ops_test : std_algorithms_test {
 public:
  std_algorithms_mod_seq_ops_test() {
    flare::parallel_for(m_static_tensor.extent(0),
                         AssignIndexFunctor<static_tensor_t>(m_static_tensor));
  }
    ~std_algorithms_mod_seq_ops_test() = default;
};

//----------------------------------------------------------------------------

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy") {
  auto result = KE::copy(exespace(), KE::begin(m_static_tensor),
                         KE::end(m_static_tensor), KE::begin(m_strided_tensor));
  REQUIRE_EQ(KE::end(m_strided_tensor), result);
  compare_tensors(m_static_tensor, m_strided_tensor);

  auto result2 = KE::copy(exespace(), KE::begin(m_strided_tensor),
                          KE::end(m_strided_tensor), KE::begin(m_dynamic_tensor));
  REQUIRE_EQ(KE::end(m_dynamic_tensor), result2);
  compare_tensors(m_dynamic_tensor, m_strided_tensor);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy_tensor") {
  REQUIRE_EQ(KE::end(m_dynamic_tensor),
            KE::copy(exespace(), m_static_tensor, m_dynamic_tensor));
  compare_tensors(m_static_tensor, m_dynamic_tensor);

  REQUIRE_EQ(KE::end(m_strided_tensor),
            KE::copy(exespace(), m_dynamic_tensor, m_strided_tensor));
  compare_tensors(m_dynamic_tensor, m_strided_tensor);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy_n") {
  constexpr std::size_t n = 5;
  tensor_host_space_t expected("copy_n_expected");
  expected(0) = 0;
  expected(1) = 1;
  expected(2) = 2;
  expected(3) = 3;
  expected(4) = 4;
  expected(5) = 0;
  expected(6) = 0;
  expected(7) = 0;
  expected(8) = 0;
  expected(9) = 0;

  // pass iterators
  auto first = KE::begin(m_static_tensor);
  auto dest  = KE::begin(m_dynamic_tensor);
  REQUIRE_EQ(dest + n, KE::copy_n(exespace(), first, n, dest));
  compare_tensors(expected, m_dynamic_tensor);

  // pass tensors
  REQUIRE_EQ(KE::begin(m_strided_tensor) + n,
            KE::copy_n(exespace(), m_static_tensor, n, m_strided_tensor));
  compare_tensors(expected, m_strided_tensor);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy_backward") {
  auto first = KE::begin(m_static_tensor);
  auto last  = KE::end(m_static_tensor);
  auto dest  = KE::end(m_dynamic_tensor);

  // pass iterators
  REQUIRE_EQ(KE::begin(m_dynamic_tensor),
            KE::copy_backward(exespace(), first, last, dest));
  compare_tensors(m_static_tensor, m_dynamic_tensor);

  // pass tensors
  REQUIRE_EQ(KE::begin(m_strided_tensor),
            KE::copy_backward(exespace(), m_static_tensor, m_strided_tensor));
  compare_tensors(m_static_tensor, m_strided_tensor);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "reverse_copy") {
  tensor_host_space_t expected("reverse_copy_expected");
  expected(0) = 9;
  expected(1) = 8;
  expected(2) = 7;
  expected(3) = 6;
  expected(4) = 5;
  expected(5) = 4;
  expected(6) = 3;
  expected(7) = 2;
  expected(8) = 1;
  expected(9) = 0;

  auto first = KE::begin(m_static_tensor);
  auto last  = KE::end(m_static_tensor);
  auto dest  = KE::begin(m_dynamic_tensor);

  REQUIRE_EQ(KE::end(m_dynamic_tensor),
            KE::reverse_copy(exespace(), first, last, dest));
  compare_tensors(expected, m_dynamic_tensor);

  REQUIRE_EQ(KE::end(m_strided_tensor),
            KE::reverse_copy(exespace(), m_static_tensor, m_strided_tensor));
  compare_tensors(expected, m_strided_tensor);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "fill") {
  constexpr auto fill_value = 1.0;
  tensor_host_space_t expected("fill_n_expected");
  expected(0) = 0;
  expected(1) = 0;
  expected(2) = 0;
  expected(3) = 0;
  expected(4) = 0;
  expected(5) = 0;
  expected(6) = 0;
  expected(7) = 0;
  expected(8) = fill_value;
  expected(9) = fill_value;

  // pass iterators
  KE::fill(exespace(), KE::begin(m_dynamic_tensor) + 8, KE::end(m_dynamic_tensor),
           fill_value);
  compare_tensors(expected, m_dynamic_tensor);

  // pass tensor
  KE::fill(exespace(), m_strided_tensor, fill_value);
  verify_values(fill_value, m_strided_tensor);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "fill_n") {
  constexpr auto fill_n_value     = 100.0;
  constexpr auto fill_n_new_value = 200.0;

  // fill all elements
  // pass iterator
  REQUIRE_EQ(KE::end(m_static_tensor),
            KE::fill_n(exespace(), KE::begin(m_static_tensor),
                       m_static_tensor.extent(0), fill_n_value));
  verify_values(fill_n_value, m_static_tensor);

  // pass tensor
  REQUIRE_EQ(KE::end(m_strided_tensor),
            KE::fill_n(exespace(), m_strided_tensor, m_strided_tensor.extent(0),
                       fill_n_value));
  verify_values(fill_n_value, m_strided_tensor);

  // fill zero elements
  // pass tensor
  REQUIRE_EQ(KE::begin(m_dynamic_tensor),
            KE::fill_n(exespace(), m_dynamic_tensor, 0, fill_n_new_value));

  // fill single element
  // pass iterator
  REQUIRE_EQ(
      KE::begin(m_static_tensor) + 1,
      KE::fill_n(exespace(), KE::begin(m_static_tensor), 1, fill_n_new_value));

  tensor_host_space_t expected("fill_n_expected");
  expected(0) = fill_n_new_value;
  expected(1) = fill_n_value;
  expected(2) = fill_n_value;
  expected(3) = fill_n_value;
  expected(4) = fill_n_value;
  expected(5) = fill_n_value;
  expected(6) = fill_n_value;
  expected(7) = fill_n_value;
  expected(8) = fill_n_value;
  expected(9) = fill_n_value;
  compare_tensors(expected, m_static_tensor);
}

struct TransformFunctor {
  FLARE_INLINE_FUNCTION
  value_type operator()(const value_type& val) const {
    (void)val;
    return static_cast<value_type>(-1);
  }
};

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "transform_from_fixture_unary_op") {
  tensor_host_space_t gold_source("transform_expected");
  gold_source(0) = 0;
  gold_source(1) = 1;
  gold_source(2) = 2;
  gold_source(3) = 3;
  gold_source(4) = 4;
  gold_source(5) = 5;
  gold_source(6) = 6;
  gold_source(7) = 7;
  gold_source(8) = 8;
  gold_source(9) = 9;

  // transform static tensor, store results in dynamic tensor
  auto r1 = KE::transform(exespace(), KE::begin(m_static_tensor),
                          KE::end(m_static_tensor), KE::begin(m_dynamic_tensor),
                          TransformFunctor());
  REQUIRE_EQ(r1, KE::end(m_dynamic_tensor));
  compare_tensors(gold_source, m_static_tensor);
  verify_values(-1., m_dynamic_tensor);

  // transform dynamic tensor, store results in strided tensor
  auto r2 = KE::transform(exespace(), m_dynamic_tensor, m_strided_tensor,
                          TransformFunctor());
  REQUIRE_EQ(r2, KE::end(m_strided_tensor));
  verify_values(-1., m_dynamic_tensor);
  verify_values(-1., m_strided_tensor);

  // transform strided tensor, store results in static tensor
  auto r3 = KE::transform(exespace(), m_strided_tensor, m_static_tensor,
                          TransformFunctor());
  REQUIRE_EQ(r3, KE::end(m_static_tensor));
  verify_values(-1., m_static_tensor);
  verify_values(-1., m_strided_tensor);
}

struct TransformBinaryFunctor {
  FLARE_INLINE_FUNCTION
  value_type operator()(const value_type& val1, const value_type& val2) const {
    return val1 + val2;
  }
};

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "transform_from_fixture_binary_op") {
  tensor_host_space_t expected("transform_expected");
  expected(0) = 0;
  expected(1) = 1;
  expected(2) = 2;
  expected(3) = 3;
  expected(4) = 4;
  expected(5) = 5;
  expected(6) = 6;
  expected(7) = 7;
  expected(8) = 8;
  expected(9) = 9;

  auto r1 = KE::transform(exespace(), KE::begin(m_static_tensor),
                          KE::end(m_static_tensor), KE::begin(m_dynamic_tensor),
                          KE::begin(m_strided_tensor), TransformBinaryFunctor());
  REQUIRE_EQ(r1, KE::end(m_strided_tensor));
  compare_tensors(expected, m_strided_tensor);

  expected(0) = 0;
  expected(1) = 2;
  expected(2) = 4;
  expected(3) = 6;
  expected(4) = 8;
  expected(5) = 10;
  expected(6) = 12;
  expected(7) = 14;
  expected(8) = 16;
  expected(9) = 18;
  auto r2 = KE::transform("label", exespace(), m_static_tensor, m_strided_tensor,
                          m_dynamic_tensor, TransformBinaryFunctor());
  REQUIRE_EQ(r2, KE::end(m_dynamic_tensor));
  compare_tensors(expected, m_dynamic_tensor);
}

constexpr value_type generated_value = 2.0;

struct GenerateFunctor {
  FLARE_INLINE_FUNCTION
  value_type operator()() const { return generated_value; }
};

// cuda illegal instruction error appears for this one:
// constexpr int generate_f() { return generated_value; }

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "generate") {
  // tensor + functor
  KE::generate(exespace(), m_static_tensor, GenerateFunctor());
  verify_values(generated_value, m_static_tensor);

  // iterators + functor
  KE::generate(exespace(), KE::begin(m_strided_tensor), KE::end(m_strided_tensor),
               GenerateFunctor());
  verify_values(generated_value, m_strided_tensor);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "generate_n") {
  // iterator + functor
  REQUIRE_EQ(KE::end(m_static_tensor),
            KE::generate_n(exespace(), KE::begin(m_static_tensor),
                           m_static_tensor.extent(0), GenerateFunctor()));
  verify_values(generated_value, m_static_tensor);

  // tensor + functor
  REQUIRE_EQ(KE::end(m_dynamic_tensor),
            KE::generate_n(exespace(), m_dynamic_tensor, m_dynamic_tensor.extent(0),
                           GenerateFunctor()));
  verify_values(generated_value, m_dynamic_tensor);

  // tensor + functor, negative n
  REQUIRE_EQ(KE::begin(m_strided_tensor),
            KE::generate_n(exespace(), m_strided_tensor, -1, GenerateFunctor()));
}

// -----------------
// test swap_ranges
// -----------------
template <class TensorType>
struct StdModOpsSwapRangesFillFunctorA {
  TensorType m_tensor;
  StdModOpsSwapRangesFillFunctorA(TensorType tensor) : m_tensor(tensor) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { m_tensor(i) = i; }
};

template <class TensorType>
struct StdModOpsSwapRangesFillFunctorB {
  TensorType m_tensor;
  StdModOpsSwapRangesFillFunctorB(TensorType tensor) : m_tensor(tensor) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { m_tensor(i) = 100 - i; }
};

template <class TensorType>
void test_swap_ranges(TensorType tensor) {
  const auto ext = tensor.extent(0);

  auto FA = StdModOpsSwapRangesFillFunctorA<TensorType>(tensor);
  flare::parallel_for(ext, std::move(FA));

  using static_tensor_type = std_algorithms_test::static_tensor_t;
  static_tensor_type TensorB("TensorB");
  auto FB = StdModOpsSwapRangesFillFunctorB<static_tensor_type>(TensorB);
  flare::parallel_for(ext, std::move(FB));

  /* call swap_ranges */
  auto first1 = KE::begin(tensor) + 2;
  auto last1  = first1 + 4;
  auto first2 = KE::begin(TensorB) + 1;
  auto r      = KE::swap_ranges(exespace(), first1, last1, first2);
  REQUIRE_EQ(r, first2 + 4);

  static_tensor_type checkTensorA("tmp");
  using cp_func_a_t = CopyFunctor<TensorType, static_tensor_type>;
  parallel_for(ext, cp_func_a_t(tensor, checkTensorA));
  auto cvA_h =
      flare::create_mirror_tensor_and_copy(flare::HostSpace(), checkTensorA);
  REQUIRE_EQ(cvA_h(0), 0);
  REQUIRE_EQ(cvA_h(1), 1);
  REQUIRE_EQ(cvA_h(2), 99);
  REQUIRE_EQ(cvA_h(3), 98);
  REQUIRE_EQ(cvA_h(4), 97);
  REQUIRE_EQ(cvA_h(5), 96);
  REQUIRE_EQ(cvA_h(6), 6);
  REQUIRE_EQ(cvA_h(7), 7);
  REQUIRE_EQ(cvA_h(8), 8);
  REQUIRE_EQ(cvA_h(9), 9);

  /* check TensorB */
  static_tensor_type checkTensorB("tmpB");
  using cp_func_b_t = CopyFunctor<static_tensor_type, static_tensor_type>;
  flare::parallel_for(ext, cp_func_b_t(TensorB, checkTensorB));
  auto cvB_h =
      flare::create_mirror_tensor_and_copy(flare::HostSpace(), checkTensorB);
  REQUIRE_EQ(cvB_h(0), 100);
  REQUIRE_EQ(cvB_h(1), 2);
  REQUIRE_EQ(cvB_h(2), 3);
  REQUIRE_EQ(cvB_h(3), 4);
  REQUIRE_EQ(cvB_h(4), 5);
  REQUIRE_EQ(cvB_h(5), 95);
  REQUIRE_EQ(cvB_h(6), 94);
  REQUIRE_EQ(cvB_h(7), 93);
  REQUIRE_EQ(cvB_h(8), 92);
  REQUIRE_EQ(cvB_h(9), 91);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "swap_ranges") {
  test_swap_ranges(m_static_tensor);
  test_swap_ranges(m_dynamic_tensor);
  test_swap_ranges(m_strided_tensor);
}

}  // namespace stdalgos
}  // namespace Test
