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
    flare::parallel_for(m_static_view.extent(0),
                         AssignIndexFunctor<static_view_t>(m_static_view));
  }
    ~std_algorithms_mod_seq_ops_test() = default;
};

//----------------------------------------------------------------------------

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy") {
  auto result = KE::copy(exespace(), KE::begin(m_static_view),
                         KE::end(m_static_view), KE::begin(m_strided_view));
  REQUIRE_EQ(KE::end(m_strided_view), result);
  compare_views(m_static_view, m_strided_view);

  auto result2 = KE::copy(exespace(), KE::begin(m_strided_view),
                          KE::end(m_strided_view), KE::begin(m_dynamic_view));
  REQUIRE_EQ(KE::end(m_dynamic_view), result2);
  compare_views(m_dynamic_view, m_strided_view);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy_view") {
  REQUIRE_EQ(KE::end(m_dynamic_view),
            KE::copy(exespace(), m_static_view, m_dynamic_view));
  compare_views(m_static_view, m_dynamic_view);

  REQUIRE_EQ(KE::end(m_strided_view),
            KE::copy(exespace(), m_dynamic_view, m_strided_view));
  compare_views(m_dynamic_view, m_strided_view);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy_n") {
  constexpr std::size_t n = 5;
  view_host_space_t expected("copy_n_expected");
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
  auto first = KE::begin(m_static_view);
  auto dest  = KE::begin(m_dynamic_view);
  REQUIRE_EQ(dest + n, KE::copy_n(exespace(), first, n, dest));
  compare_views(expected, m_dynamic_view);

  // pass views
  REQUIRE_EQ(KE::begin(m_strided_view) + n,
            KE::copy_n(exespace(), m_static_view, n, m_strided_view));
  compare_views(expected, m_strided_view);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "copy_backward") {
  auto first = KE::begin(m_static_view);
  auto last  = KE::end(m_static_view);
  auto dest  = KE::end(m_dynamic_view);

  // pass iterators
  REQUIRE_EQ(KE::begin(m_dynamic_view),
            KE::copy_backward(exespace(), first, last, dest));
  compare_views(m_static_view, m_dynamic_view);

  // pass views
  REQUIRE_EQ(KE::begin(m_strided_view),
            KE::copy_backward(exespace(), m_static_view, m_strided_view));
  compare_views(m_static_view, m_strided_view);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "reverse_copy") {
  view_host_space_t expected("reverse_copy_expected");
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

  auto first = KE::begin(m_static_view);
  auto last  = KE::end(m_static_view);
  auto dest  = KE::begin(m_dynamic_view);

  REQUIRE_EQ(KE::end(m_dynamic_view),
            KE::reverse_copy(exespace(), first, last, dest));
  compare_views(expected, m_dynamic_view);

  REQUIRE_EQ(KE::end(m_strided_view),
            KE::reverse_copy(exespace(), m_static_view, m_strided_view));
  compare_views(expected, m_strided_view);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "fill") {
  constexpr auto fill_value = 1.0;
  view_host_space_t expected("fill_n_expected");
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
  KE::fill(exespace(), KE::begin(m_dynamic_view) + 8, KE::end(m_dynamic_view),
           fill_value);
  compare_views(expected, m_dynamic_view);

  // pass view
  KE::fill(exespace(), m_strided_view, fill_value);
  verify_values(fill_value, m_strided_view);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "fill_n") {
  constexpr auto fill_n_value     = 100.0;
  constexpr auto fill_n_new_value = 200.0;

  // fill all elements
  // pass iterator
  REQUIRE_EQ(KE::end(m_static_view),
            KE::fill_n(exespace(), KE::begin(m_static_view),
                       m_static_view.extent(0), fill_n_value));
  verify_values(fill_n_value, m_static_view);

  // pass view
  REQUIRE_EQ(KE::end(m_strided_view),
            KE::fill_n(exespace(), m_strided_view, m_strided_view.extent(0),
                       fill_n_value));
  verify_values(fill_n_value, m_strided_view);

  // fill zero elements
  // pass view
  REQUIRE_EQ(KE::begin(m_dynamic_view),
            KE::fill_n(exespace(), m_dynamic_view, 0, fill_n_new_value));

  // fill single element
  // pass iterator
  REQUIRE_EQ(
      KE::begin(m_static_view) + 1,
      KE::fill_n(exespace(), KE::begin(m_static_view), 1, fill_n_new_value));

  view_host_space_t expected("fill_n_expected");
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
  compare_views(expected, m_static_view);
}

struct TransformFunctor {
  FLARE_INLINE_FUNCTION
  value_type operator()(const value_type& val) const {
    (void)val;
    return static_cast<value_type>(-1);
  }
};

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "transform_from_fixture_unary_op") {
  view_host_space_t gold_source("transform_expected");
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

  // transform static view, store results in dynamic view
  auto r1 = KE::transform(exespace(), KE::begin(m_static_view),
                          KE::end(m_static_view), KE::begin(m_dynamic_view),
                          TransformFunctor());
  REQUIRE_EQ(r1, KE::end(m_dynamic_view));
  compare_views(gold_source, m_static_view);
  verify_values(-1., m_dynamic_view);

  // transform dynamic view, store results in strided view
  auto r2 = KE::transform(exespace(), m_dynamic_view, m_strided_view,
                          TransformFunctor());
  REQUIRE_EQ(r2, KE::end(m_strided_view));
  verify_values(-1., m_dynamic_view);
  verify_values(-1., m_strided_view);

  // transform strided view, store results in static view
  auto r3 = KE::transform(exespace(), m_strided_view, m_static_view,
                          TransformFunctor());
  REQUIRE_EQ(r3, KE::end(m_static_view));
  verify_values(-1., m_static_view);
  verify_values(-1., m_strided_view);
}

struct TransformBinaryFunctor {
  FLARE_INLINE_FUNCTION
  value_type operator()(const value_type& val1, const value_type& val2) const {
    return val1 + val2;
  }
};

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "transform_from_fixture_binary_op") {
  view_host_space_t expected("transform_expected");
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

  auto r1 = KE::transform(exespace(), KE::begin(m_static_view),
                          KE::end(m_static_view), KE::begin(m_dynamic_view),
                          KE::begin(m_strided_view), TransformBinaryFunctor());
  REQUIRE_EQ(r1, KE::end(m_strided_view));
  compare_views(expected, m_strided_view);

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
  auto r2 = KE::transform("label", exespace(), m_static_view, m_strided_view,
                          m_dynamic_view, TransformBinaryFunctor());
  REQUIRE_EQ(r2, KE::end(m_dynamic_view));
  compare_views(expected, m_dynamic_view);
}

constexpr value_type generated_value = 2.0;

struct GenerateFunctor {
  FLARE_INLINE_FUNCTION
  value_type operator()() const { return generated_value; }
};

// cuda illegal instruction error appears for this one:
// constexpr int generate_f() { return generated_value; }

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "generate") {
  // view + functor
  KE::generate(exespace(), m_static_view, GenerateFunctor());
  verify_values(generated_value, m_static_view);

  // iterators + functor
  KE::generate(exespace(), KE::begin(m_strided_view), KE::end(m_strided_view),
               GenerateFunctor());
  verify_values(generated_value, m_strided_view);
}

TEST_CASE_FIXTURE(std_algorithms_mod_seq_ops_test, "generate_n") {
  // iterator + functor
  REQUIRE_EQ(KE::end(m_static_view),
            KE::generate_n(exespace(), KE::begin(m_static_view),
                           m_static_view.extent(0), GenerateFunctor()));
  verify_values(generated_value, m_static_view);

  // view + functor
  REQUIRE_EQ(KE::end(m_dynamic_view),
            KE::generate_n(exespace(), m_dynamic_view, m_dynamic_view.extent(0),
                           GenerateFunctor()));
  verify_values(generated_value, m_dynamic_view);

  // view + functor, negative n
  REQUIRE_EQ(KE::begin(m_strided_view),
            KE::generate_n(exespace(), m_strided_view, -1, GenerateFunctor()));
}

// -----------------
// test swap_ranges
// -----------------
template <class ViewType>
struct StdModOpsSwapRangesFillFunctorA {
  ViewType m_view;
  StdModOpsSwapRangesFillFunctorA(ViewType view) : m_view(view) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { m_view(i) = i; }
};

template <class ViewType>
struct StdModOpsSwapRangesFillFunctorB {
  ViewType m_view;
  StdModOpsSwapRangesFillFunctorB(ViewType view) : m_view(view) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { m_view(i) = 100 - i; }
};

template <class ViewType>
void test_swap_ranges(ViewType view) {
  const auto ext = view.extent(0);

  /* fill view_a */
  auto FA = StdModOpsSwapRangesFillFunctorA<ViewType>(view);
  flare::parallel_for(ext, std::move(FA));

  /* fill view_b */
  using static_view_type = std_algorithms_test::static_view_t;
  static_view_type viewB("viewB");
  auto FB = StdModOpsSwapRangesFillFunctorB<static_view_type>(viewB);
  flare::parallel_for(ext, std::move(FB));

  /* call swap_ranges */
  auto first1 = KE::begin(view) + 2;
  auto last1  = first1 + 4;
  auto first2 = KE::begin(viewB) + 1;
  auto r      = KE::swap_ranges(exespace(), first1, last1, first2);
  REQUIRE_EQ(r, first2 + 4);

  /* check VIEW_A */
  static_view_type checkViewA("tmp");
  using cp_func_a_t = CopyFunctor<ViewType, static_view_type>;
  parallel_for(ext, cp_func_a_t(view, checkViewA));
  auto cvA_h =
      flare::create_mirror_view_and_copy(flare::HostSpace(), checkViewA);
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

  /* check viewB */
  static_view_type checkViewB("tmpB");
  using cp_func_b_t = CopyFunctor<static_view_type, static_view_type>;
  flare::parallel_for(ext, cp_func_b_t(viewB, checkViewB));
  auto cvB_h =
      flare::create_mirror_view_and_copy(flare::HostSpace(), checkViewB);
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
  test_swap_ranges(m_static_view);
  test_swap_ranges(m_dynamic_view);
  test_swap_ranges(m_strided_view);
}

}  // namespace stdalgos
}  // namespace Test
