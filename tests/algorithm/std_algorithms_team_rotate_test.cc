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
namespace TeamRotate {

namespace KE = flare::experimental;

template <class TensorType, class DistancesTensorType>
struct TestFunctorA {
  TensorType m_tensor;
  DistancesTensorType m_distancesTensor;
  std::size_t m_pivotShift;
  int m_apiPick;

  TestFunctorA(const TensorType tensor, const DistancesTensorType distancesTensor,
               std::size_t pivotShift, int apiPick)
      : m_tensor(tensor),
        m_distancesTensor(distancesTensor),
        m_pivotShift(pivotShift),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensor        = flare::subtensor(m_tensor, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto pivot = KE::begin(myRowTensor) + m_pivotShift;
      auto it =
          KE::rotate(member, KE::begin(myRowTensor), pivot, KE::end(myRowTensor));

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) = KE::distance(KE::begin(myRowTensor), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::rotate(member, myRowTensor, m_pivotShift);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) = KE::distance(KE::begin(myRowTensor), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, std::size_t pivotShift,
            int apiId) {
  /* description:
     randomly fill a rank-2 tensor and for a given pivot,
     do a team-level KE::rotate
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

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataTensor, distancesTensor, pivotShift, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run std algo and check
  // -----------------------------------------------
  // here I can use cloneOfDataTensorBeforeOp_h to run std algo on
  // since that contains a valid copy of the data
  auto distancesTensor_h = create_host_space_copy(distancesTensor);
  for (std::size_t i = 0; i < cloneOfDataTensorBeforeOp_h.extent(0); ++i) {
    auto myRow = flare::subtensor(cloneOfDataTensorBeforeOp_h, i, flare::ALL());
    auto pivot = KE::begin(myRow) + pivotShift;

    auto it = std::rotate(KE::begin(myRow), pivot, KE::end(myRow));
    const std::size_t stdDistance = KE::distance(KE::begin(myRow), it);
    REQUIRE_EQ(stdDistance, distancesTensor_h(i));
  }

  auto dataTensorAfterOp_h = create_host_space_copy(dataTensor);
  expect_equal_host_tensors(cloneOfDataTensorBeforeOp_h, dataTensorAfterOp_h);
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

TEST_CASE("std_algorithms_rotate_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, int>();
}

}  // namespace TeamRotate
}  // namespace stdalgos
}  // namespace Test
