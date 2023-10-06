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

namespace Test {
namespace stdalgos {
namespace TeamShiftLeft {

namespace KE = flare::experimental;

template <class ViewType, class DistancesViewType>
struct TestFunctorA {
  ViewType m_view;
  DistancesViewType m_distancesView;
  std::size_t m_shift;
  int m_apiPick;

  TestFunctorA(const ViewType view, const DistancesViewType distancesView,
               std::size_t shift, int apiPick)
      : m_view(view),
        m_distancesView(distancesView),
        m_shift(shift),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = flare::subview(m_view, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto it = KE::shift_left(member, KE::begin(myRowView), KE::end(myRowView),
                               m_shift);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::shift_left(member, myRowView, m_shift);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    }
  }
};

// shift_left is only supported starting from C++20,
// so put here a working version of the std algo copied from
// https://github.com/llvm/llvm-project/blob/main/libcxx/include/__algorithm/shift_left.h
template <class ForwardIterator>
ForwardIterator my_std_shift_left(
    ForwardIterator first, ForwardIterator last,
    typename std::iterator_traits<ForwardIterator>::difference_type n) {
  if (n == 0) {
    return last;
  }

  ForwardIterator m = first;
  for (; n > 0; --n) {
    if (m == last) {
      return first;
    }
    ++m;
  }
  return std::move(m, last, first);
}

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, std::size_t shift,
            int apiId) {
  /* description:
     randomly fill a rank-2 view and do a team-level KE::shift_left
     using shift as the shift count.
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [dataView, cloneOfDataViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{11, 523}, "dataView");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, distancesView, shift, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run std algo and check
  // -----------------------------------------------
  // here I can use cloneOfDataViewBeforeOp_h to run std algo on
  // since that contains a valid copy of the data
  auto distancesView_h = create_host_space_copy(distancesView);
  for (std::size_t i = 0; i < cloneOfDataViewBeforeOp_h.extent(0); ++i) {
    auto myRow = flare::subview(cloneOfDataViewBeforeOp_h, i, flare::ALL());
    auto it    = my_std_shift_left(KE::begin(myRow), KE::end(myRow), shift);
    const std::size_t stdDistance = KE::distance(KE::begin(myRow), it);
    REQUIRE_EQ(stdDistance, distancesView_h(i));
  }

  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  expect_equal_host_views(cloneOfDataViewBeforeOp_h, dataViewAfterOp_h);
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  // prepare a map where, for a given set of num cols
  // we provide a list of shifts to use for testing
  // key = num of columns
  // value = list of shifts
  // Note that the cornerCase number is here since the shiftLeft algo
  // should work even when the shift given is way larger than the range.
  constexpr std::size_t cornerCase                        = 110111;
  const std::map<int, std::vector<std::size_t>> scenarios = {
      {0, {0, cornerCase}},
      {2, {0, 1, 2, cornerCase}},
      {6, {0, 1, 2, 5, cornerCase}},
      {13, {0, 1, 2, 8, 11, cornerCase}},
      {56, {0, 1, 2, 8, 11, 33, 56, cornerCase}},
      {123, {0, 1, 11, 33, 56, 89, 112, cornerCase}},
      {3145, {0, 1, 11, 33, 56, 89, 112, 5677, cornerCase}}};

  for (int numTeams : teamSizesToTest) {
    for (const auto& scenario : scenarios) {
      const std::size_t numCols = scenario.first;
      for (int copyCount : scenario.second) {
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(numTeams, numCols, copyCount, apiId);
        }
      }
    }
  }
}

TEST_CASE("std_algorithms_shift_left_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamShiftLeft
}  // namespace stdalgos
}  // namespace Test
