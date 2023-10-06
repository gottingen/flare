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
namespace TeamReplaceCopyIf {

namespace KE = flare::experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  FLARE_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  FLARE_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class SourceViewType, class DestViewType, class DistancesViewType,
          class ValueType>
struct TestFunctorA {
  SourceViewType m_sourceView;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  ValueType m_threshold;
  ValueType m_newValue;
  int m_apiPick;

  TestFunctorA(const SourceViewType sourceView, const DestViewType destView,
               const DistancesViewType distancesView, ValueType threshold,
               ValueType newVal, int apiPick)
      : m_sourceView(sourceView),
        m_destView(destView),
        m_distancesView(distancesView),
        m_threshold(threshold),
        m_newValue(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom =
        flare::subview(m_sourceView, myRowIndex, flare::ALL());
    auto myRowViewDest = flare::subview(m_destView, myRowIndex, flare::ALL());

    GreaterThanValueFunctor predicate(m_threshold);

    if (m_apiPick == 0) {
      auto it = KE::replace_copy_if(
          member, KE::begin(myRowViewFrom), KE::end(myRowViewFrom),
          KE::begin(myRowViewDest), predicate, m_newValue);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::replace_copy_if(member, myRowViewFrom, myRowViewDest,
                                    predicate, m_newValue);
      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level replace_copy_if where the values strictly
     greater than a threshold are replaced/copied into a new view
     with a new value, while those <= threshold are copied but unchanged
   */

  const auto threshold = static_cast<ValueType>(151);
  const auto newVal    = static_cast<ValueType>(1);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [sourceView, cloneOfSourceViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{5, 523}, "sourceView");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // create the destination view
  flare::View<ValueType**> destView("destView", numTeams, numCols);

  // replace_copy_if returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceView, destView, distancesView, threshold, newVal,
                   apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);
  flare::View<ValueType**, flare::HostSpace> stdDestView("stdDestView",
                                                           numTeams, numCols);
  GreaterThanValueFunctor predicate(threshold);
  for (std::size_t i = 0; i < sourceView.extent(0); ++i) {
    auto rowFrom =
        flare::subview(cloneOfSourceViewBeforeOp_h, i, flare::ALL());
    auto rowDest = flare::subview(stdDestView, i, flare::ALL());
    auto it      = std::replace_copy_if(KE::cbegin(rowFrom), KE::cend(rowFrom),
                                   KE::begin(rowDest), predicate, newVal);
    const std::size_t stdDistance = KE::distance(KE::begin(rowDest), it);
    REQUIRE_EQ(stdDistance, distancesView_h(i));
  }

  auto dataViewAfterOp_h = create_host_space_copy(destView);
  expect_equal_host_views(stdDestView, dataViewAfterOp_h);
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_replace_copy_if_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplaceCopyIf
}  // namespace stdalgos
}  // namespace Test
