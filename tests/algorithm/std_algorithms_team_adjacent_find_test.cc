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
namespace TeamAdjacentFind {

namespace KE = flare::experimental;

template <class ValueType>
struct IsEqualFunctor {
  FLARE_INLINE_FUNCTION constexpr bool operator()(const ValueType& lhs,
                                                   const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DataViewType, class DistancesViewType, class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  DistancesViewType m_distancesView;
  int m_apiPick;
  BinaryPredType m_binaryPred;

  TestFunctorA(const DataViewType dataView,
               const DistancesViewType distancesView, int apiPick,
               BinaryPredType binaryPred)
      : m_dataView(dataView),
        m_distancesView(distancesView),
        m_apiPick(apiPick),
        m_binaryPred(binaryPred) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom = flare::subview(m_dataView, myRowIndex, flare::ALL());

    switch (m_apiPick) {
      case 0: {
        const auto it = KE::adjacent_find(member, KE::cbegin(myRowViewFrom),
                                          KE::cend(myRowViewFrom));
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });
        break;
      }

      case 1: {
        const auto it = KE::adjacent_find(member, myRowViewFrom);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });
        break;
      }

      case 2: {
        const auto it =
            KE::adjacent_find(member, KE::cbegin(myRowViewFrom),
                              KE::cend(myRowViewFrom), m_binaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });
        break;
      }

      case 3: {
        const auto it = KE::adjacent_find(member, myRowViewFrom, m_binaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool ensureAdjacentFindCanFind, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level adjacent_find
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr ValueType lowerBound = 5;
  constexpr ValueType upperBound = 523;
  const auto bounds              = make_bounds(lowerBound, upperBound);

  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataView");

  // If ensureAdjacentFindCanFind == true ensure there are two consecutive equal
  // elements in each row

  // dataView might not deep copyable (e.g. strided layout) so to prepare it
  // correctly, we make a new view that is for sure deep copyable, modify it on
  // the host, deep copy to device and then launch a kernel to copy to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(flare::HostSpace(), dataView_dc);

  if (ensureAdjacentFindCanFind && numCols > 1) {
    for (std::size_t i = 0; i < numTeams; ++i) {
      const auto j = numCols / 2;

      dataView_dc_h(i, j - 1) = dataView_dc_h(i, j);
    }
  }

  // copy to dataView_dc and then to dataView
  flare::deep_copy(dataView_dc, dataView_dc_h);

  CopyFunctorRank2 cpFun(dataView_dc, dataView);
  flare::parallel_for("copy", dataView.extent(0) * dataView.extent(1), cpFun);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // adjacent_find returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the beginning
  // of the interval that team operates on and then we check that these
  // distances match the std result
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  IsEqualFunctor<ValueType> binaryPred;
  TestFunctorA fnc(dataView, distancesView, apiId, binaryPred);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom            = flare::subview(dataView_dc_h, i, flare::ALL());
    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);
    const std::size_t beginEndDist = KE::distance(rowFromBegin, rowFromEnd);

    switch (apiId) {
      case 0:
      case 1: {
        const auto it = std::adjacent_find(rowFromBegin, rowFromEnd);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);
        REQUIRE_EQ(stdDistance, distancesView_h(i));

        if (numCols == 1) {
          REQUIRE_EQ(distancesView_h(i), beginEndDist);
        } else if (ensureAdjacentFindCanFind) {
          REQUIRE_NE(distancesView_h(i), beginEndDist);
        }

        break;
      }

      case 2:
      case 3: {
        const auto it =
            std::adjacent_find(rowFromBegin, rowFromEnd, binaryPred);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        REQUIRE_EQ(stdDistance, distancesView_h(i));

        if (numCols == 1) {
          REQUIRE_EQ(distancesView_h(i), beginEndDist);
        } else if (ensureAdjacentFindCanFind) {
          REQUIRE_NE(distancesView_h(i), beginEndDist);
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool ensureAdjacentFindCanFind) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(ensureAdjacentFindCanFind, numTeams,
                                     numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_adjacent_find_team_test,two_consecutive_equal_elements_exist") {
  constexpr bool ensureAdjacentFindCanFind = true;

  run_all_scenarios<DynamicTag, double>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedTwoRowsTag, int>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(ensureAdjacentFindCanFind);
}

TEST_CASE("std_algorithms_adjacent_find_team_test, two_consecutive_equal_elements_might_exist") {
  constexpr bool ensureAdjacentFindCanFind = false;

  run_all_scenarios<DynamicTag, double>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedTwoRowsTag, int>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(ensureAdjacentFindCanFind);
}

}  // namespace TeamAdjacentFind
}  // namespace stdalgos
}  // namespace Test
