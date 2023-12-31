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
namespace TeamCount {

namespace KE = flare::experimental;

template <class ViewType, class ValuesViewType, class CountsViewType>
struct TestFunctorA {
  ViewType m_view;
  ValuesViewType m_valuesView;
  CountsViewType m_countsView;
  int m_apiPick;

  TestFunctorA(const ViewType view, const ValuesViewType valuesView,
               const CountsViewType countsView, int apiPick)
      : m_view(view),
        m_valuesView(valuesView),
        m_countsView(countsView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto rowIndex = member.league_rank();
    const auto value    = m_valuesView(rowIndex);
    auto rowView        = flare::subview(m_view, rowIndex, flare::ALL());

    switch (m_apiPick) {
      case 0: {
        auto result =
            KE::count(member, KE::cbegin(rowView), KE::cend(rowView), value);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_countsView(rowIndex) = result; });

        break;
      }

      case 1: {
        auto result = KE::count(member, rowView, value);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_countsView(rowIndex) = result; });

        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool searched_value_exist, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level count
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.

  // Boundaries choosen so that every drawn number is at least once in the given
  // row
  const ValueType lowerBound = numCols / 4;
  const ValueType upperBound = 1 + numCols * 3 / 4;
  const auto bounds          = make_bounds(lowerBound, upperBound);

  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataView");

  // If searched_value_exist == true, we want to ensure that count result is >
  // 0, so we randomly pick a value to look for from a given row.
  //
  // If searched_value_exist == false, we want to ensure that count returns 0,
  // so we pick a value that's outside of view boundaries.
  flare::View<ValueType*> valuesView("valuesView", numTeams);
  auto valuesView_h = create_mirror_view(flare::HostSpace(), valuesView);

  using rand_pool =
      flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);

  if (searched_value_exist) {
    flare::View<std::size_t*, flare::DefaultHostExecutionSpace> randomIndices(
        "randomIndices", numTeams);
    flare::fill_random(randomIndices, pool, 0, numCols);

    for (std::size_t i = 0; i < numTeams; ++i) {
      const std::size_t j = randomIndices(i);
      valuesView_h(i)     = dataViewBeforeOp_h(i, j);
    }
  } else {
    flare::fill_random(valuesView_h, pool, upperBound, upperBound * 2);
  }

  flare::deep_copy(valuesView, valuesView_h);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // to verify that things work, each team stores the result of its count
  // call, and then we check that these match what we expect
  flare::View<std::size_t*> countsView("countsView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, valuesView, countsView, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto countsView_h = create_host_space_copy(countsView);
  if (searched_value_exist) {
    for (std::size_t i = 0; i < dataView.extent(0); ++i) {
      auto rowFrom = flare::subview(dataViewBeforeOp_h, i, flare::ALL());
      const auto rowFromBegin = KE::cbegin(rowFrom);
      const auto rowFromEnd   = KE::cend(rowFrom);
      const auto val          = valuesView_h(i);

      const std::size_t result = std::count(rowFromBegin, rowFromEnd, val);
      REQUIRE_EQ(result, countsView_h(i));
    }
  } else {
    for (std::size_t i = 0; i < countsView.extent(0); ++i) {
      constexpr std::size_t zero = 0;
      REQUIRE_EQ(countsView_h(i), zero);
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool searchedValueExist) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(searchedValueExist, numTeams, numCols,
                                     apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_count_team_test, count_returns_nonzero") {
  constexpr bool searchedValueExist = true;
  run_all_scenarios<DynamicTag, double>(searchedValueExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValueExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValueExist);
}

TEST_CASE("std_algorithms_count_team_test, count_returns_zero") {
  constexpr bool searchedValueExist = false;
  run_all_scenarios<DynamicTag, double>(searchedValueExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValueExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValueExist);
}

}  // namespace TeamCount
}  // namespace stdalgos
}  // namespace Test
