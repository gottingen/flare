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
namespace TeamForEach {

namespace KE = flare::experimental;

template <class ValueType>
struct PrefixIncrementFunctor {
  FLARE_INLINE_FUNCTION
  void operator()(ValueType& val) const { ++val; }
};

template <class DatATensorType, class UnaryPredType>
struct TestFunctorA {
  DatATensorType m_dataTensor;
  int m_apiPick;
  UnaryPredType m_unaryPred;

  TestFunctorA(const DatATensorType dataTensor, int apiPick,
               UnaryPredType unaryPred)
      : m_dataTensor(dataTensor), m_apiPick(apiPick), m_unaryPred(unaryPred) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensorFrom = flare::subtensor(m_dataTensor, myRowIndex, flare::ALL());

    switch (m_apiPick) {
      case 0: {
        KE::for_each(member, KE::begin(myRowTensorFrom), KE::end(myRowTensorFrom),
                     m_unaryPred);
        break;
      }

      case 1: {
        KE::for_each(member, myRowTensorFrom, m_unaryPred);
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 tensor randomly filled with values,
     and run a team-level for_each
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a tensor in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr ValueType lowerBound = 5;
  constexpr ValueType upperBound = 523;
  const auto bounds              = make_bounds(lowerBound, upperBound);

  auto [dataTensor, _] = create_random_tensor_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataTensor");

  // for_each modifies dataTensor, so make a separated host copy of if
  auto dataTensorBeforeOp_h = create_host_space_copy(dataTensor);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  PrefixIncrementFunctor<ValueType> unaryPred;

  // use CTAD for functor
  TestFunctorA fnc(dataTensor, apiId, unaryPred);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto dataTensorAfterOp_h = create_host_space_copy(dataTensor);
  for (std::size_t i = 0; i < dataTensorAfterOp_h.extent(0); ++i) {
    for (std::size_t j = 0; j < dataTensorAfterOp_h.extent(1); ++j) {
      REQUIRE_EQ(dataTensorBeforeOp_h(i, j) + 1, dataTensorAfterOp_h(i, j));
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

TEST_CASE("std_algorithms_for_each_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamForEach
}  // namespace stdalgos
}  // namespace Test
