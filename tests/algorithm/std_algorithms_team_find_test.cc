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
namespace TeamFind {

namespace KE = flare::experimental;

template <class DataViewType, class SearchedValuesViewType,
          class DistancesViewType>
struct TestFunctorA {
  DataViewType m_dataView;
  SearchedValuesViewType m_searchedValuesView;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const DataViewType dataView,
               const SearchedValuesViewType searchedValuesView,
               const DistancesViewType distancesView, int apiPick)
      : m_dataView(dataView),
        m_searchedValuesView(searchedValuesView),
        m_distancesView(distancesView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = flare::subview(m_dataView, myRowIndex, flare::ALL());
    const auto searchedValue = m_searchedValuesView(myRowIndex);

    switch (m_apiPick) {
      case 0: {
        auto it = KE::find(member, KE::cbegin(myRowViewFrom),
                           KE::cend(myRowViewFrom), searchedValue);

        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });

        break;
      }

      case 1: {
        auto it = KE::find(member, myRowViewFrom, searchedValue);

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
void test_A(const bool searchedValuesExist, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level find
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

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // find returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  // If searchedValuesExist == true we want to ensure that each value we're
  // looking for exists in dataView. To do that, for each numTeams, a random j
  // index from a range [0, numCols) is used to obtain a value from dataView.
  //
  // If searchedValuesExist == false we want to ensure the opposite, so every
  // value is less than a lower bound of dataView.
  flare::View<ValueType*> searchedValuesView("searchValuesView", numTeams);
  auto searchedValuesView_h =
      create_mirror_view(flare::HostSpace(), searchedValuesView);

  using rand_pool =
      flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);

  if (searchedValuesExist) {
    flare::View<std::size_t*, flare::DefaultHostExecutionSpace> randomIndices(
        "randomIndices", numTeams);

    flare::fill_random(randomIndices, pool, 0, numCols);

    for (std::size_t i = 0; i < numTeams; ++i) {
      const std::size_t j     = randomIndices(i);
      searchedValuesView_h(i) = dataViewBeforeOp_h(i, j);
    }
  } else {
    flare::fill_random(searchedValuesView_h, pool, 0, lowerBound);
  }

  flare::deep_copy(searchedValuesView, searchedValuesView_h);

  // use CTAD for functor
  TestFunctorA fnc(dataView, searchedValuesView, distancesView, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = flare::subview(dataViewBeforeOp_h, i, flare::ALL());
    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);

    auto it = std::find(rowFromBegin, rowFromEnd, searchedValuesView_h(i));

    const std::size_t stdDistance      = KE::distance(rowFromBegin, it);
    const std::size_t beginEndDistance = KE::distance(rowFromBegin, rowFromEnd);

    if (searchedValuesExist) {
      REQUIRE_LT(stdDistance, beginEndDistance);
    } else {
      REQUIRE_EQ(stdDistance, beginEndDistance);
    }

    REQUIRE_EQ(stdDistance, distancesView_h(i));
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool searchedValuesExist) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(searchedValuesExist, numTeams, numCols,
                                     apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_find_team_test, searched_values_exist") {
  constexpr bool searchedValuesExist = true;

  run_all_scenarios<DynamicTag, double>(searchedValuesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValuesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValuesExist);
}

TEST_CASE("std_algorithms_find_team_test, searched_values_do_not_exist") {
  constexpr bool searchedValuesExist = false;

  run_all_scenarios<DynamicTag, double>(searchedValuesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValuesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValuesExist);
}

}  // namespace TeamFind
}  // namespace stdalgos
}  // namespace Test
