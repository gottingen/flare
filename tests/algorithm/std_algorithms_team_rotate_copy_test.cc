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
namespace TeamRotateCopy {

namespace KE = flare::experimental;

template <class SourceViewType, class DestViewType, class DistancesViewType>
struct TestFunctorA {
  SourceViewType m_sourceView;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  std::size_t m_pivotShift;
  int m_apiPick;

  TestFunctorA(const SourceViewType sourceView, const DestViewType destView,
               const DistancesViewType distancesView, std::size_t pivotShift,
               int apiPick)
      : m_sourceView(sourceView),
        m_destView(destView),
        m_distancesView(distancesView),
        m_pivotShift(pivotShift),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom =
        flare::subview(m_sourceView, myRowIndex, flare::ALL());
    auto myRowViewDest = flare::subview(m_destView, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto pivot = KE::cbegin(myRowViewFrom) + m_pivotShift;
      auto it =
          KE::rotate_copy(member, KE::cbegin(myRowViewFrom), pivot,
                          KE::cend(myRowViewFrom), KE::begin(myRowViewDest));

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it =
          KE::rotate_copy(member, myRowViewFrom, m_pivotShift, myRowViewDest);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, std::size_t pivotShift,
            int apiId) {
  /* description:
     randomly fill a rank-2 view and for a given pivot,
     do a team-level KE::rotateCopy
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

  // create the destination view
  flare::View<ValueType**> destView("destView", numTeams, numCols);

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceView, destView, distancesView, pivotShift, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run std algo and check
  // -----------------------------------------------
  flare::View<ValueType**, flare::HostSpace> stdDestView("stdDestView",
                                                           numTeams, numCols);
  auto distancesView_h = create_host_space_copy(distancesView);
  for (std::size_t i = 0; i < cloneOfSourceViewBeforeOp_h.extent(0); ++i) {
    auto myRowFrom =
        flare::subview(cloneOfSourceViewBeforeOp_h, i, flare::ALL());
    auto myRowDest = flare::subview(stdDestView, i, flare::ALL());

    auto pivot = KE::cbegin(myRowFrom) + pivotShift;
    auto it    = std::rotate_copy(KE::cbegin(myRowFrom), pivot,
                               KE::cend(myRowFrom), KE::begin(myRowDest));
    const std::size_t stdDistance = KE::distance(KE::begin(myRowDest), it);
    REQUIRE_EQ(stdDistance, distancesView_h(i));
  }

  auto destViewAfterOp_h = create_host_space_copy(destView);
  expect_equal_host_views(stdDestView, destViewAfterOp_h);
}

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist(int b, std::size_t seedIn) : m_dist(0, b) { m_gen.seed(seedIn); }
  int operator()() { return m_dist(m_gen); }
};

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1153}) {
      // given numTeams, numCols, randomly pick a few pivots to test
      constexpr int numPivotsToTest = 5;
      UnifDist<int> pivotsProducer(numCols, 3123377);
      for (int k = 0; k < numPivotsToTest; ++k) {
        const auto pivotIndex = pivotsProducer();
        // test all apis
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(numTeams, numCols, pivotIndex, apiId);
        }
      }
    }
  }
}

TEST_CASE("std_algorithms_rotate_copy_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, int>();
}

}  // namespace TeamRotateCopy
}  // namespace stdalgos
}  // namespace Test
