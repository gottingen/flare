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
namespace TeamReverse {

namespace KE = flare::experimental;

template <class TensorType>
struct TestFunctorA {
  TensorType m_tensor;
  int m_api_pick;

  TestFunctorA(const TensorType tensor, int apiPick)
      : m_tensor(tensor), m_api_pick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensor        = flare::subtensor(m_tensor, myRowIndex, flare::ALL());

    if (m_api_pick == 0) {
      KE::reverse(member, KE::begin(myRowTensor), KE::end(myRowTensor));
    } else if (m_api_pick == 1) {
      KE::reverse(member, myRowTensor);
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     create a rank-2 tensor, randomly fill, and do a team-level
     KE::reverse for each row where each team is
     responsible for a single row
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
          flare::pair<ValueType, ValueType>{11, 523}, "dataTensor");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // use CTAD for functor
  TestFunctorA fnc(dataTensor, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto dataTensorAfterOp_h = create_host_space_copy(dataTensor);
  for (std::size_t i = 0; i < dataTensorAfterOp_h.extent(0); ++i) {
    for (std::size_t j = 0; j < dataTensorAfterOp_h.extent(1); ++j) {
      REQUIRE(dataTensorAfterOp_h(i, j) ==
                  cloneOfDataTensorBeforeOp_h(i, numCols - j - 1));
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

TEST_CASE("std_algorithms_reverse_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReverse
}  // namespace stdalgos
}  // namespace Test
