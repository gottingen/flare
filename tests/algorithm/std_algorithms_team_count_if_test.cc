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
namespace TeamCountIf {

namespace KE = flare::experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  FLARE_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  FLARE_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class TensorType, class CountsTensorType, class ValueType>
struct TestFunctorA {
  TensorType m_tensor;
  CountsTensorType m_countsTensor;
  ValueType m_threshold;
  int m_apiPick;

  TestFunctorA(const TensorType tensor, const CountsTensorType countsTensor,
               ValueType threshold, int apiPick)
      : m_tensor(tensor),
        m_countsTensor(countsTensor),
        m_threshold(threshold),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensor        = flare::subtensor(m_tensor, myRowIndex, flare::ALL());

    GreaterThanValueFunctor predicate(m_threshold);
    if (m_apiPick == 0) {
      auto myCount = KE::count_if(member, KE::begin(myRowTensor),
                                  KE::end(myRowTensor), predicate);

      flare::single(flare::PerTeam(member),
                     [=, *this]() { m_countsTensor(myRowIndex) = myCount; });
    } else if (m_apiPick == 1) {
      auto myCount = KE::count_if(member, myRowTensor, predicate);
      flare::single(flare::PerTeam(member),
                     [=, *this]() { m_countsTensor(myRowIndex) = myCount; });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 tensor randomly filled with values,
     and run a team-level count_if where only the values
     strictly greater than a threshold are counted
   */

  const auto threshold = static_cast<ValueType>(151);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a tensor in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr ValueType lowerBound = 5;
  constexpr ValueType upperBound = 523;
  const auto bounds              = make_bounds(lowerBound, upperBound);

  auto [dataTensor, cloneOfDataTensorBeforeOp_h] =
      create_random_tensor_and_host_clone(LayoutTag{}, numTeams, numCols, bounds,
                                        "dataTensor");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // to verify that things work, each team stores the result
  // of its count_if call, and then we check
  // that these match what we expect
  flare::Tensor<std::size_t*> countsTensor("countsTensor", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataTensor, countsTensor, threshold, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto countsTensor_h = create_host_space_copy(countsTensor);
  for (std::size_t i = 0; i < cloneOfDataTensorBeforeOp_h.extent(0); ++i) {
    std::size_t goldCountForRow = 0;
    for (std::size_t j = 0; j < cloneOfDataTensorBeforeOp_h.extent(1); ++j) {
      if (cloneOfDataTensorBeforeOp_h(i, j) > threshold) {
        goldCountForRow++;
      }
    }
    REQUIRE_EQ(goldCountForRow, countsTensor_h(i));
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

TEST_CASE("std_algorithms_count_if_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamCountIf
}  // namespace stdalgos
}  // namespace Test
