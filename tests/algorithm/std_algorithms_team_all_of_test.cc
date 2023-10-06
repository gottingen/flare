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
namespace TeamAllOf {

namespace KE = flare::experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  FLARE_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  FLARE_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class DataViewType, class AlllOfResultsViewType, class UnaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  AlllOfResultsViewType m_allOfResultsView;
  int m_apiPick;
  UnaryPredType m_unaryPred;

  TestFunctorA(const DataViewType dataView,
               const AlllOfResultsViewType allOfResultsView, int apiPick,
               UnaryPredType unaryPred)
      : m_dataView(dataView),
        m_allOfResultsView(allOfResultsView),
        m_apiPick(apiPick),
        m_unaryPred(unaryPred) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom = flare::subview(m_dataView, myRowIndex, flare::ALL());

    switch (m_apiPick) {
      case 0: {
        const bool result = KE::all_of(member, KE::cbegin(myRowViewFrom),
                                       KE::cend(myRowViewFrom), m_unaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_allOfResultsView(myRowIndex) = result;
        });
        break;
      }

      case 1: {
        const bool result = KE::all_of(member, myRowViewFrom, m_unaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_allOfResultsView(myRowIndex) = result;
        });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level all_of
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

  // to verify that things work, each team stores the result of its all_of call,
  // and then we check that these match what we expect
  flare::View<bool*> allOfResultsView("allOfResultsView", numTeams);

  GreaterThanValueFunctor unaryPred{lowerBound - 1};

  // use CTAD for functor
  TestFunctorA fnc(dataView, allOfResultsView, apiId, unaryPred);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto allOfResultsView_h = create_host_space_copy(allOfResultsView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = flare::subview(dataViewBeforeOp_h, i, flare::ALL());
    const bool result =
        std::all_of(KE::cbegin(rowFrom), KE::cend(rowFrom), unaryPred);
    REQUIRE_EQ(result, allOfResultsView_h(i));
  }
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

TEST_CASE("std_algorithms_all_of_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamAllOf
}  // namespace stdalgos
}  // namespace Test
