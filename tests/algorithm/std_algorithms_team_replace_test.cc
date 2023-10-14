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
namespace TeamReplace {

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

template <class TensorType, class ValueType>
struct TestFunctorA {
  TensorType m_tensor;
  ValueType m_targetValue;
  ValueType m_newValue;
  int m_apiPick;

  TestFunctorA(const TensorType tensor, ValueType oldVal, ValueType newVal,
               int apiPick)
      : m_tensor(tensor),
        m_targetValue(oldVal),
        m_newValue(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensor        = flare::subtensor(m_tensor, myRowIndex, flare::ALL());

    if (m_apiPick == 0) {
      KE::replace(member, KE::begin(myRowTensor), KE::end(myRowTensor),
                  m_targetValue, m_newValue);
    } else if (m_apiPick == 1) {
      KE::replace(member, myRowTensor, m_targetValue, m_newValue);
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     Randomly fill a tensor with some elements equal to a target value that we
     want to replace with a new value. Do the operation via a team parfor with
     one row per team.
   */

  const auto targetVal = static_cast<ValueType>(531);
  const auto newVal    = static_cast<ValueType>(123);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // Create a tensor in the memory space associated with default exespace with as
  // many rows as the number of teams and fill it with random values from an
  // arbitrary range. Pick range so that some of the values are equal to target.
  auto [dataTensor, dataTensorBeforeOp_h] = create_random_tensor_and_host_clone(
      LayoutTag{}, numTeams, numCols,
      flare::pair<ValueType, ValueType>{targetVal - 1, targetVal + 1},
      "dataTensor");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // use CTAD for functor
  TestFunctorA fnc(dataTensor, targetVal, newVal, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // conditions for test passing:
  // - the target elements are replaced with the new value
  // - all other elements are unchanged
  // -----------------------------------------------
  auto dataTensorAfterOp_h = create_host_space_copy(dataTensor);
  for (std::size_t i = 0; i < dataTensorAfterOp_h.extent(0); ++i) {
    for (std::size_t j = 0; j < dataTensorAfterOp_h.extent(1); ++j) {
      const auto correctVal = (dataTensorBeforeOp_h(i, j) == targetVal)
                                  ? newVal
                                  : dataTensorBeforeOp_h(i, j);
      REQUIRE_EQ(dataTensorAfterOp_h(i, j), correctVal);
    }
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

TEST_CASE("std_algorithms_replace_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplace
}  // namespace stdalgos
}  // namespace Test
