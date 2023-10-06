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

struct random_access_iterator_test : std_algorithms_test {
 public:
    random_access_iterator_test() {
        flare::parallel_for(m_static_view.extent(0),
                            AssignIndexFunctor<static_view_t>(m_static_view));

        flare::parallel_for(m_static_view.extent(0),
                            AssignIndexFunctor<dyn_view_t>(m_dynamic_view));

        flare::parallel_for(m_static_view.extent(0),
                            AssignIndexFunctor<strided_view_t>(m_strided_view));
    }
    ~random_access_iterator_test() = default;
};

TEST_CASE_FIXTURE(random_access_iterator_test, "constructor") {
  // just tests that constructor works
  auto it1 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view);
  auto it2 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view);
  auto it3 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view);
  auto it4 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it5 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 3);
  auto it6 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 3);
  REQUIRE(true);
}

template <class IteratorType, class ValueType>
void test_random_access_it_verify(IteratorType it, ValueType gold_value) {
  using view_t = flare::View<typename IteratorType::value_type>;
  view_t checkView("checkView");
  CopyFromIteratorFunctor<IteratorType, view_t> cf(it, checkView);
  flare::parallel_for("_std_algo_copy", 1, cf);
  auto v_h =
      flare::create_mirror_view_and_copy(flare::HostSpace(), checkView);
  REQUIRE_EQ(v_h(), gold_value);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "dereference") {
  auto it1 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view);
  auto it2 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view);
  auto it3 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view);
  test_random_access_it_verify(it1, (value_type)0);
  test_random_access_it_verify(it2, (value_type)0);
  test_random_access_it_verify(it3, (value_type)0);

  auto it4 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it5 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 4);
  auto it6 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 5);
  test_random_access_it_verify(it4, (value_type)3);
  test_random_access_it_verify(it5, (value_type)4);
  test_random_access_it_verify(it6, (value_type)5);
}

template <class ItTypeFrom, class ViewTypeTo>
struct CopyFromIteratorUsingSubscriptFunctor {
  ItTypeFrom m_itFrom;
  ViewTypeTo m_viewTo;

  CopyFromIteratorUsingSubscriptFunctor(const ItTypeFrom itFromIn,
                                        const ViewTypeTo viewToIn)
      : m_itFrom(itFromIn), m_viewTo(viewToIn) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { m_viewTo(i) = m_itFrom[i]; }
};

template <class IteratorType>
void test_random_access_it_subscript_op_verify(IteratorType it) {
  using value_t = typename IteratorType::value_type;
  using view_t  = flare::View<value_t*>;
  view_t checkView("checkView", 3);
  CopyFromIteratorUsingSubscriptFunctor<IteratorType, view_t> cf(it, checkView);
  flare::parallel_for("_std_algo_copy", 3, cf);

  auto v_h =
      flare::create_mirror_view_and_copy(flare::HostSpace(), checkView);
  REQUIRE_EQ(v_h(0), (value_t)0);
  REQUIRE_EQ(v_h(1), (value_t)1);
  REQUIRE_EQ(v_h(2), (value_t)2);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "subscript_operator") {
  auto it1 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view);
  auto it2 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view);
  auto it3 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view);
  test_random_access_it_subscript_op_verify(it1);
  test_random_access_it_subscript_op_verify(it2);
  test_random_access_it_subscript_op_verify(it3);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "operatorsSet1") {
  auto it1 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it2 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 3);
  auto it3 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 3);
  ++it1;
  ++it2;
  ++it3;
  test_random_access_it_verify(it1, (value_type)4);
  test_random_access_it_verify(it2, (value_type)4);
  test_random_access_it_verify(it3, (value_type)4);

  --it1;
  --it2;
  --it3;
  test_random_access_it_verify(it1, (value_type)3);
  test_random_access_it_verify(it2, (value_type)3);
  test_random_access_it_verify(it3, (value_type)3);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "operatorsSet2") {
  auto it1  = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it2  = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 3);
  auto it3  = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 3);
  auto it11 = it1 + 3;
  auto it21 = it2 + 3;
  auto it31 = it3 + 3;
  test_random_access_it_verify(it11, (value_type)6);
  test_random_access_it_verify(it21, (value_type)6);
  test_random_access_it_verify(it31, (value_type)6);

  auto it12 = it11 - 4;
  auto it22 = it21 - 4;
  auto it32 = it31 - 4;
  test_random_access_it_verify(it12, (value_type)2);
  test_random_access_it_verify(it22, (value_type)2);
  test_random_access_it_verify(it32, (value_type)2);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "operatorsSet3") {
  auto it1 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it2 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 3);
  auto it3 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 3);
  it1 += 3;
  it2 += 3;
  it3 += 3;
  test_random_access_it_verify(it1, (value_type)6);
  test_random_access_it_verify(it2, (value_type)6);
  test_random_access_it_verify(it3, (value_type)6);

  it1 -= 4;
  it2 -= 4;
  it3 -= 4;
  test_random_access_it_verify(it1, (value_type)2);
  test_random_access_it_verify(it2, (value_type)2);
  test_random_access_it_verify(it3, (value_type)2);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "operatorsSet4") {
  auto it1 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it2 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 3);
  auto it3 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 3);

  auto it4 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 4);
  auto it5 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 4);
  auto it6 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 4);
  REQUIRE_NE(it1, it4);
  REQUIRE_NE(it2, it5);
  REQUIRE_NE(it3, it6);
  REQUIRE_LT(it1, it4);
  REQUIRE_LT(it2, it5);
  REQUIRE_LT(it3, it6);
  REQUIRE_LE(it1, it4);
  REQUIRE_LE(it2, it5);
  REQUIRE_LE(it3, it6);

  auto it7 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it8 = KE::detail::RandomAccessIterator<dyn_view_t>(m_dynamic_view, 3);
  auto it9 = KE::detail::RandomAccessIterator<strided_view_t>(m_strided_view, 3);
  REQUIRE_EQ(it1, it7);
  REQUIRE_EQ(it2, it8);
  REQUIRE_EQ(it3, it9);
  REQUIRE_GE(it1, it7);
  REQUIRE_GE(it2, it8);
  REQUIRE_GE(it3, it9);
  REQUIRE_GT(it4, it7);
  REQUIRE_GT(it5, it8);
  REQUIRE_GT(it6, it9);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "assignment_operator") {
  auto it1 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 3);
  auto it2 = KE::detail::RandomAccessIterator<static_view_t>(m_static_view, 5);
  REQUIRE_NE(it1, it2);

  it2 = it1;
  REQUIRE_EQ(it1, it2);
}

TEST_CASE_FIXTURE(random_access_iterator_test, "distance") {
  auto first = KE::begin(m_dynamic_view);
  auto last  = KE::end(m_dynamic_view);

  REQUIRE_EQ(0, KE::distance(first, first));
  REQUIRE_EQ(1, KE::distance(first, first + 1));
  REQUIRE_EQ(m_dynamic_view.extent(0), size_t(KE::distance(first, last)));
}

TEST_CASE_FIXTURE(random_access_iterator_test, "traits_helpers") {
  using T1_t = KE::detail::RandomAccessIterator<static_view_t>;
  using T2_t = KE::detail::RandomAccessIterator<dyn_view_t>;
  using T3_t = KE::detail::RandomAccessIterator<strided_view_t>;

  namespace KE = flare::experimental;
  static_assert(KE::detail::are_iterators_v<T1_t, T2_t, T3_t>);
  static_assert(KE::detail::are_random_access_iterators_v<T1_t, T2_t, T3_t>);
  static_assert(!KE::detail::are_iterators_v<int, T2_t, T3_t>);
}

}  // namespace stdalgos
}  // namespace Test
