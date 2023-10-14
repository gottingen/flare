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
namespace TeamReplaceCopyIf {

namespace KE = flare::experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  FLARE_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  FLARE_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class SourceTensorType, class DestTensorType, class DistancesTensorType,
          class ValueType>
struct TestFunctorA {
  SourceTensorType m_sourceTensor;
  DestTensorType m_destTensor;
  DistancesTensorType m_distancesTensor;
  ValueType m_threshold;
  ValueType m_newValue;
  int m_apiPick;

  TestFunctorA(const SourceTensorType sourceTensor, const DestTensorType destTensor,
               const DistancesTensorType distancesTensor, ValueType threshold,
               ValueType newVal, int apiPick)
      : m_sourceTensor(sourceTensor),
        m_destTensor(destTensor),
        m_distancesTensor(distancesTensor),
        m_threshold(threshold),
        m_newValue(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensorFrom =
        flare::subtensor(m_sourceTensor, myRowIndex, flare::ALL());
    auto myRowTensorDest = flare::subtensor(m_destTensor, myRowIndex, flare::ALL());

    GreaterThanValueFunctor predicate(m_threshold);

    if (m_apiPick == 0) {
      auto it = KE::replace_copy_if(
          member, KE::begin(myRowTensorFrom), KE::end(myRowTensorFrom),
          KE::begin(myRowTensorDest), predicate, m_newValue);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) =
            KE::distance(KE::begin(myRowTensorDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::replace_copy_if(member, myRowTensorFrom, myRowTensorDest,
                                    predicate, m_newValue);
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
     use a rank-2 tensor randomly filled with values,
     and run a team-level replace_copy_if where the values strictly
     greater than a threshold are replaced/copied into a new tensor
     with a new value, while those <= threshold are copied but unchanged
   */

  const auto threshold = static_cast<ValueType>(151);
  const auto newVal    = static_cast<ValueType>(1);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a tensor in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [sourceTensor, cloneOfSourceTensorBeforeOp_h] =
      create_random_tensor_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{5, 523}, "sourceTensor");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // create the destination tensor
  flare::Tensor<ValueType**> destTensor("destTensor", numTeams, numCols);

  // replace_copy_if returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceTensor, destTensor, distancesTensor, threshold, newVal,
                   apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesTensor_h = create_host_space_copy(distancesTensor);
  flare::Tensor<ValueType**, flare::HostSpace> stdDestTensor("stdDestTensor",
                                                           numTeams, numCols);
  GreaterThanValueFunctor predicate(threshold);
  for (std::size_t i = 0; i < sourceTensor.extent(0); ++i) {
    auto rowFrom =
        flare::subtensor(cloneOfSourceTensorBeforeOp_h, i, flare::ALL());
    auto rowDest = flare::subtensor(stdDestTensor, i, flare::ALL());
    auto it      = std::replace_copy_if(KE::cbegin(rowFrom), KE::cend(rowFrom),
                                   KE::begin(rowDest), predicate, newVal);
    const std::size_t stdDistance = KE::distance(KE::begin(rowDest), it);
    REQUIRE_EQ(stdDistance, distancesTensor_h(i));
  }

  auto dataTensorAfterOp_h = create_host_space_copy(destTensor);
  expect_equal_host_tensors(stdDestTensor, dataTensorAfterOp_h);
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

TEST_CASE("std_algorithms_replace_copy_if_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplaceCopyIf
}  // namespace stdalgos
}  // namespace Test
