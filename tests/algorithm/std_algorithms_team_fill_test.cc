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
namespace TeamFill {

namespace KE = flare::experimental;

template <class ViewType>
struct TestFunctorA {
  ViewType m_view;
  int m_apiPick;

  TestFunctorA(const ViewType view, int apiPick)
      : m_view(view), m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto leagueRank = member.league_rank();
    const auto myRowIndex = leagueRank;
    auto myRowView        = flare::subview(m_view, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      KE::fill(member, KE::begin(myRowView), KE::end(myRowView), leagueRank);
    } else if (m_apiPick == 1) {
      KE::fill(member, myRowView, leagueRank);
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     create a rank-2 view, randomly fill with non trivial numbers
     and do a team-level KE::fill where each team fills
     with its league_rank value the row it is responsible for
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [dataView, _] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols,
      flare::pair<ValueType, ValueType>{11, 523}, "dataView");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // use CTAD for functor
  TestFunctorA fnc(dataView, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  // each row should be filled with the row index
  // since the league_rank of a team here coincides with row index
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  for (std::size_t i = 0; i < dataViewAfterOp_h.extent(0); ++i) {
    for (std::size_t j = 0; j < dataViewAfterOp_h.extent(1); ++j) {
      REQUIRE(dataViewAfterOp_h(i, j) == static_cast<ValueType>(i));
    }
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

TEST_CASE("std_algorithms_fill_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamFill
}  // namespace stdalgos
}  // namespace Test
