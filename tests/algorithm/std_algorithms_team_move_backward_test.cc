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

template <class SourceTensorType, class DestTensorType, class DistancesTensorType>
struct TestFunctorA {
  SourceTensorType m_sourceTensor;
  DestTensorType m_destTensor;
  DistancesTensorType m_distancesTensor;
  int m_apiPick;

  TestFunctorA(const SourceTensorType sourceTensor, const DestTensorType destTensor,
               const DistancesTensorType distancesTensor, int apiPick)
      : m_sourceTensor(sourceTensor),
        m_destTensor(destTensor),
        m_distancesTensor(distancesTensor),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensorFrom =
        flare::subtensor(m_sourceTensor, myRowIndex, flare::ALL());
    auto myRowTensorDest = flare::subtensor(m_destTensor, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto it =
          KE::move_backward(member, KE::cbegin(myRowTensorFrom),
                            KE::cend(myRowTensorFrom), KE::end(myRowTensorDest));

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) =
            KE::distance(KE::begin(myRowTensorDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::move_backward(member, myRowTensorFrom, myRowTensorDest);
      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) =
            KE::distance(KE::begin(myRowTensorDest), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     randomly fill a source tensor and do team KE::move_backward
     into a destination tensor. The operation is done via a
     team parfor with one row per team.
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a tensor in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [sourceTensor, cloneOfSourceTensorBeforeOp_h] =
      create_random_tensor_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{11, 523}, "sourceTensor");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // create the destination tensor: for a meaningful test, the destination
  // tensor must have more columns that than the source tensor so that we
  // can check that the elements are copied into the right place
  constexpr std::size_t extra = 10;
  flare::Tensor<ValueType**> destTensor("destTensor", numTeams, numCols + extra);
  // make host copy of destTensor that should remain unchanged
  auto destTensorBeforeOp_h = create_host_space_copy(destTensor);

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceTensor, destTensor, distancesTensor, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  // KE::move_backwar has been applied to sourceTensor, so we should
  // NOT use sourceTensor henceforth, becuase all its elements
  // have been moved from.

  auto distancesTensor_h   = create_host_space_copy(distancesTensor);
  auto destTensorAfterOp_h = create_host_space_copy(destTensor);
  for (std::size_t i = 0; i < destTensorAfterOp_h.extent(0); ++i) {
    // first extra num of columns should be unchanged
    for (std::size_t j = 0; j < extra; ++j) {
      REQUIRE(destTensorAfterOp_h(i, j) == destTensorBeforeOp_h(i, j));
    }

    // after extra # of column (inclusive) should match the source tensor
    for (std::size_t j = extra; j < destTensorBeforeOp_h.extent(1); ++j) {
      REQUIRE(cloneOfSourceTensorBeforeOp_h(i, j - extra) ==
                  destTensorAfterOp_h(i, j));
    }

    // each team should have returned an interator whose distance
    // from the beginning of the row should satisfy this
    REQUIRE(distancesTensor_h(i) == extra);
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
