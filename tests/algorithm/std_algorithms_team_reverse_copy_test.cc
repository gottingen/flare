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
namespace TeamReverseCopy {

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
          KE::reverse_copy(member, KE::begin(myRowTensorFrom),
                           KE::end(myRowTensorFrom), KE::begin(myRowTensorDest));

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) =
            KE::distance(KE::begin(myRowTensorDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::reverse_copy(member, myRowTensorFrom, myRowTensorDest);
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
     randomly fill a source tensor and reverse_copy into a destination tensor.
     The operation is done via a team parfor with one row per team.
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
  // create the destination tensor
  flare::Tensor<ValueType**> destTensor("destTensor", numTeams, numCols);

  // to verify that things are correct each team stores the distance
  // of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceTensor, destTensor, distancesTensor, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto distancesTensor_h   = create_host_space_copy(distancesTensor);
  auto destTensorAfterOp_h = create_host_space_copy(destTensor);
  for (std::size_t i = 0; i < destTensorAfterOp_h.extent(0); ++i) {
    for (std::size_t j = 0; j < destTensorAfterOp_h.extent(1); ++j) {
      REQUIRE(destTensorAfterOp_h(i, j) ==
                  cloneOfSourceTensorBeforeOp_h(i, numCols - j - 1));
    }
    // each team should return an iterator past the last column
    REQUIRE(distancesTensor_h(i) == numCols);
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8113}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_reverse_copy_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReverseCopy
}  // namespace stdalgos
}  // namespace Test
