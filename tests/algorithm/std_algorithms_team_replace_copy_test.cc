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
namespace TeamReplaceCopy {

namespace KE = flare::experimental;

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

template <class SourceTensorType, class DestTensorType, class DistancesTensorType,
          class ValueType>
struct TestFunctorA {
  SourceTensorType m_sourceTensor;
  DestTensorType m_destTensor;
  DistancesTensorType m_distancesTensor;
  ValueType m_targetValue;
  ValueType m_newValue;
  int m_apiPick;

  TestFunctorA(const SourceTensorType sourceTensor, const DestTensorType destTensor,
               const DistancesTensorType distancesTensor, ValueType targetVal,
               ValueType newVal, int apiPick)
      : m_sourceTensor(sourceTensor),
        m_destTensor(destTensor),
        m_distancesTensor(distancesTensor),
        m_targetValue(targetVal),
        m_newValue(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensorFrom =
        flare::subtensor(m_sourceTensor, myRowIndex, flare::ALL());
    auto myRowTensorDest = flare::subtensor(m_destTensor, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      auto it = KE::replace_copy(
          member, KE::begin(myRowTensorFrom), KE::end(myRowTensorFrom),
          KE::begin(myRowTensorDest), m_targetValue, m_newValue);

      flare::single(flare::PerTeam(member), [=, *this]() {
        m_distancesTensor(myRowIndex) =
            KE::distance(KE::begin(myRowTensorDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::replace_copy(member, myRowTensorFrom, myRowTensorDest,
                                 m_targetValue, m_newValue);
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
     use a "source" and "destination" rank-2 tensors such that in the source,
     for each row, a random subset of elements is filled with a target value
     that we want to replace_copy with a new value into the destination tensor.
     The operation is done via a team parfor with one row per team.
   */

  const auto targetVal = static_cast<ValueType>(531);
  const auto newVal    = static_cast<ValueType>(123);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // construct in memory space associated with default exespace
  auto sourceTensor =
      create_tensor<ValueType>(LayoutTag{}, numTeams, numCols, "sourceTensor");

  // sourceTensor might not deep copyable (e.g. strided layout) so to fill it
  // we make a new tensor that is for sure deep copyable, modify it on the host
  // deep copy to device and then launch copy kernel to sourceTensor
  auto sourceTensor_dc =
      create_deep_copyable_compatible_tensor_with_same_extent(sourceTensor);
  auto sourceTensor_dc_h = create_mirror_tensor(flare::HostSpace(), sourceTensor_dc);

  // for each row, randomly select columns, fill with targetVal
  const std::size_t maxColInd = numCols > 0 ? numCols - 1 : 0;
  UnifDist<int> colCountProducer(maxColInd, 3123377);
  UnifDist<int> colIndicesProducer(maxColInd, 455225);
  for (std::size_t i = 0; i < sourceTensor_dc_h.extent(0); ++i) {
    const std::size_t currCount = colCountProducer();
    for (std::size_t j = 0; j < currCount; ++j) {
      const auto colInd          = colIndicesProducer();
      sourceTensor_dc_h(i, colInd) = targetVal;
    }
  }

  // copy to sourceTensor_dc and then to sourceTensor
  flare::deep_copy(sourceTensor_dc, sourceTensor_dc_h);
  // use CTAD
  CopyFunctorRank2 F1(sourceTensor_dc, sourceTensor);
  flare::parallel_for("copy", sourceTensor.extent(0) * sourceTensor.extent(1), F1);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // create the destination tensor where we to store the replace_copy
  flare::Tensor<ValueType**> destTensor("destTensor", numTeams, numCols);

  // replace_copy returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceTensor, destTensor, distancesTensor, targetVal, newVal,
                   apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesTensor_h = create_host_space_copy(distancesTensor);
  flare::Tensor<ValueType**, flare::HostSpace> stdDestTensor("stdDestTensor",
                                                           numTeams, numCols);
  for (std::size_t i = 0; i < sourceTensor_dc_h.extent(0); ++i) {
    auto rowFrom = flare::subtensor(sourceTensor_dc_h, i, flare::ALL());
    auto rowDest = flare::subtensor(stdDestTensor, i, flare::ALL());
    auto it      = std::replace_copy(KE::cbegin(rowFrom), KE::cend(rowFrom),
                                KE::begin(rowDest), targetVal, newVal);
    const std::size_t stdDistance = KE::distance(KE::begin(rowDest), it);
    REQUIRE_EQ(stdDistance, distancesTensor_h(i));
  }

  auto dataTensorAfterOp_h = create_host_space_copy(destTensor);
  expect_equal_host_tensors(stdDestTensor, dataTensorAfterOp_h);
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

TEST_CASE("std_algorithms_replace_copy_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplaceCopy
}  // namespace stdalgos
}  // namespace Test
