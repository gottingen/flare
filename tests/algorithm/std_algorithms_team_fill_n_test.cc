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
namespace TeamFill_n {

namespace KE = flare::experimental;

template <class TensorType, class DistancesTensorType>
struct TestFunctorA {
  TensorType m_tensor;
  DistancesTensorType m_distancesTensor;
  std::size_t m_fillCount;
  int m_apiPick;

  TestFunctorA(const TensorType tensor, const DistancesTensorType distancesTensor,
               std::size_t fillCount, int apiPick)
      : m_tensor(tensor),
        m_distancesTensor(distancesTensor),
        m_fillCount(fillCount),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto leagueRank = member.league_rank();
    const auto myRowIndex = leagueRank;
    auto myRowTensor        = flare::subtensor(m_tensor, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto it =
          KE::fill_n(member, KE::begin(myRowTensor), m_fillCount, leagueRank);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) = KE::distance(KE::begin(myRowTensor), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::fill_n(member, myRowTensor, m_fillCount, leagueRank);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) = KE::distance(KE::begin(myRowTensor), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, std::size_t fillCount,
            int apiId) {
  /* description:
     create a rank-2 tensor, run a team parfor with one row per team,
     such that n elements of each row are filled up with the league_rank value
     of the team in charge of it, while the other elements in the row
     are left unchanged
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a tensor in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [dataTensor, cloneOfDataTensorBeforeOp_h] =
      create_random_tensor_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{5, 523}, "dataTensor");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // fill_n returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expected value
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataTensor, distancesTensor, fillCount, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto dataTensorAfterOp_h = create_host_space_copy(dataTensor);
  auto distancesTensor_h   = create_host_space_copy(distancesTensor);
  for (std::size_t i = 0; i < dataTensor.extent(0); ++i) {
    // check that values match what we expect
    for (std::size_t j = 0; j < fillCount; ++j) {
      REQUIRE_EQ(dataTensorAfterOp_h(i, j), ValueType(i));
    }
    // all other elements should be unchanged from before op
    for (std::size_t j = fillCount; j < numCols; ++j) {
      REQUIRE_EQ(dataTensorAfterOp_h(i, j), cloneOfDataTensorBeforeOp_h(i, j));
    }

    // check that returned iterators are correct
    if (fillCount > 0) {
      REQUIRE_EQ(distancesTensor_h(i), std::size_t(fillCount));
    } else {
      REQUIRE_EQ(distancesTensor_h(i), std::size_t(0));
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  // prepare a map where, for a given set of num cols
  // we provide a list of counts of elements to fill
  // key = num of columns,
  // value = list of num of elemenents to fill
  const std::map<int, std::vector<int>> scenarios = {
      {0, {0}},
      {2, {0, 1, 2}},
      {6, {0, 1, 2, 5}},
      {13, {0, 1, 2, 8, 11}},
      {56, {0, 1, 2, 8, 11, 33, 56}},
      {123, {0, 1, 11, 33, 56, 89, 112}}};

  for (int numTeams : teamSizesToTest) {
    for (const auto& scenario : scenarios) {
      const std::size_t numCols = scenario.first;
      for (int numFills : scenario.second) {
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(numTeams, numCols, numFills, apiId);
        }
      }
    }
  }
}

TEST_CASE("std_algorithms_fill_n_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamFill_n
}  // namespace stdalgos
}  // namespace Test
