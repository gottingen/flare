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
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace TeamMinMaxElement {

namespace KE = flare::experimental;

template <class ViewType, class DistancesViewType>
struct TestFunctorA {
  ViewType m_view;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const ViewType view, const DistancesViewType distancesView,
               int apiPick)
      : m_view(view), m_distancesView(distancesView), m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = flare::subview(m_view, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto itPair = KE::minmax_element(member, KE::cbegin(myRowView),
                                       KE::cend(myRowView));

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex, 0) =
            KE::distance(KE::cbegin(myRowView), itPair.first);
        m_distancesView(myRowIndex, 1) =
            KE::distance(KE::cbegin(myRowView), itPair.second);
      });
    }

    else if (m_apiPick == 1) {
      auto itPair = KE::minmax_element(member, myRowView);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex, 0) =
            KE::distance(KE::begin(myRowView), itPair.first);
        m_distancesView(myRowIndex, 1) =
            KE::distance(KE::begin(myRowView), itPair.second);
      });
    }
    else if (m_apiPick == 2) {
      using value_type = typename ViewType::value_type;
      auto itPair =
          KE::minmax_element(member, KE::cbegin(myRowView), KE::cend(myRowView),
                             CustomLessThanComparator<value_type>{});
      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex, 0) =
            KE::distance(KE::cbegin(myRowView), itPair.first);
        m_distancesView(myRowIndex, 1) =
            KE::distance(KE::cbegin(myRowView), itPair.second);
      });
    }

    else if (m_apiPick == 3) {
      using value_type = typename ViewType::value_type;
      auto itPair      = KE::minmax_element(member, myRowView,
                                       CustomLessThanComparator<value_type>{});
      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex, 0) =
            KE::distance(KE::begin(myRowView), itPair.first);
        m_distancesView(myRowIndex, 1) =
            KE::distance(KE::begin(myRowView), itPair.second);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     team-level KE::minmax_element on a rank-2 view where
     data is filled randomly and we use one team per row.
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  auto [dataView, cloneOfDataViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{0, 1153}, "dataView");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // each team stores the distance of the returned value from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  flare::View<std::size_t**> distancesView("distancesView", numTeams, 2);

  // use CTAD for functor
  TestFunctorA fnc(dataView, distancesView, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run std algo and check
  // -----------------------------------------------
  // here I can use cloneOfDataViewBeforeOp_h to run std algo on
  // since that contains a valid copy of the data
  auto distancesView_h   = create_host_space_copy(distancesView);
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  for (std::size_t i = 0; i < cloneOfDataViewBeforeOp_h.extent(0); ++i) {
    auto myRow = flare::subview(cloneOfDataViewBeforeOp_h, i, flare::ALL());

    std::size_t stdDistance[2];
    if (apiId <= 1) {
      auto itPair    = std::minmax_element(KE::cbegin(myRow), KE::cend(myRow));
      stdDistance[0] = KE::distance(KE::cbegin(myRow), itPair.first);
      stdDistance[1] = KE::distance(KE::cbegin(myRow), itPair.second);
    } else {
      auto itPair    = std::minmax_element(KE::cbegin(myRow), KE::cend(myRow),
                                        CustomLessThanComparator<value_type>{});
      stdDistance[0] = KE::distance(KE::cbegin(myRow), itPair.first);
      stdDistance[1] = KE::distance(KE::cbegin(myRow), itPair.second);
    }

    REQUIRE_EQ(stdDistance[0], distancesView_h(i, 0));
    REQUIRE_EQ(stdDistance[1], distancesView_h(i, 1));
  }

  // dataView should remain unchanged
  expect_equal_host_views(cloneOfDataViewBeforeOp_h, dataViewAfterOp_h);
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 5113}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_minmax_element_team_test, test") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedTwoRowsTag, double>();
  run_all_scenarios<StridedThreeRowsTag, int>();
}

}  // namespace TeamMinMaxElement
}  // namespace stdalgos
}  // namespace Test
