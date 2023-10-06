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
namespace TeamMovebackward {

namespace KE = flare::experimental;

template <class SourceViewType, class DestViewType, class DistancesViewType>
struct TestFunctorA {
  SourceViewType m_sourceView;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const SourceViewType sourceView, const DestViewType destView,
               const DistancesViewType distancesView, int apiPick)
      : m_sourceView(sourceView),
        m_destView(destView),
        m_distancesView(distancesView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom =
        flare::subview(m_sourceView, myRowIndex, flare::ALL());
    auto myRowViewDest = flare::subview(m_destView, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto it =
          KE::move_backward(member, KE::cbegin(myRowViewFrom),
                            KE::cend(myRowViewFrom), KE::end(myRowViewDest));

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::move_backward(member, myRowViewFrom, myRowViewDest);
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
     randomly fill a source view and do team KE::move_backward
     into a destination view. The operation is done via a
     team parfor with one row per team.
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [sourceView, cloneOfSourceViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{11, 523}, "sourceView");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // create the destination view: for a meaningful test, the destination
  // view must have more columns that than the source view so that we
  // can check that the elements are copied into the right place
  constexpr std::size_t extra = 10;
  flare::View<ValueType**> destView("destView", numTeams, numCols + extra);
  // make host copy of destView that should remain unchanged
  auto destViewBeforeOp_h = create_host_space_copy(destView);

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceView, destView, distancesView, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  // KE::move_backwar has been applied to sourceView, so we should
  // NOT use sourceView henceforth, becuase all its elements
  // have been moved from.

  auto distancesView_h   = create_host_space_copy(distancesView);
  auto destViewAfterOp_h = create_host_space_copy(destView);
  for (std::size_t i = 0; i < destViewAfterOp_h.extent(0); ++i) {
    // first extra num of columns should be unchanged
    for (std::size_t j = 0; j < extra; ++j) {
      REQUIRE(destViewAfterOp_h(i, j) == destViewBeforeOp_h(i, j));
    }

    // after extra # of column (inclusive) should match the source view
    for (std::size_t j = extra; j < destViewBeforeOp_h.extent(1); ++j) {
      REQUIRE(cloneOfSourceViewBeforeOp_h(i, j - extra) ==
                  destViewAfterOp_h(i, j));
    }

    // each team should have returned an interator whose distance
    // from the beginning of the row should satisfy this
    REQUIRE(distancesView_h(i) == extra);
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 11113}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_move_backward_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamMovebackward
}  // namespace stdalgos
}  // namespace Test
