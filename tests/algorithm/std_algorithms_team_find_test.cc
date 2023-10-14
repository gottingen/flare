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
namespace TeamFind {

namespace KE = flare::experimental;

template <class DatATensorType, class SearchedValuesTensorType,
          class DistancesTensorType>
struct TestFunctorA {
  DatATensorType m_dataTensor;
  SearchedValuesTensorType m_searchedValuesTensor;
  DistancesTensorType m_distancesTensor;
  int m_apiPick;

  TestFunctorA(const DatATensorType dataTensor,
               const SearchedValuesTensorType searchedValuesTensor,
               const DistancesTensorType distancesTensor, int apiPick)
      : m_dataTensor(dataTensor),
        m_searchedValuesTensor(searchedValuesTensor),
        m_distancesTensor(distancesTensor),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensorFrom = flare::subtensor(m_dataTensor, myRowIndex, flare::ALL());
    const auto searchedValue = m_searchedValuesTensor(myRowIndex);

    switch (m_apiPick) {
      case 0: {
        auto it = KE::find(member, KE::cbegin(myRowTensorFrom),
                           KE::cend(myRowTensorFrom), searchedValue);

        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesTensor(myRowIndex) =
              KE::distance(KE::cbegin(myRowTensorFrom), it);
        });

        break;
      }

      case 1: {
        auto it = KE::find(member, myRowTensorFrom, searchedValue);

        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesTensor(myRowIndex) =
              KE::distance(KE::begin(myRowTensorFrom), it);
        });

        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool searchedValuesExist, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 tensor randomly filled with values,
     and run a team-level find
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

  auto [dataTensor, dataTensorBeforeOp_h] = create_random_tensor_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataTensor");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // find returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  // If searchedValuesExist == true we want to ensure that each value we're
  // looking for exists in dataTensor. To do that, for each numTeams, a random j
  // index from a range [0, numCols) is used to obtain a value from dataTensor.
  //
  // If searchedValuesExist == false we want to ensure the opposite, so every
  // value is less than a lower bound of dataTensor.
  flare::Tensor<ValueType*> searchedValuesTensor("searchValuesTensor", numTeams);
  auto searchedValuesTensor_h =
      create_mirror_tensor(flare::HostSpace(), searchedValuesTensor);

  using rand_pool =
      flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);

  if (searchedValuesExist) {
    flare::Tensor<std::size_t*, flare::DefaultHostExecutionSpace> randomIndices(
        "randomIndices", numTeams);

    flare::fill_random(randomIndices, pool, 0, numCols);

    for (std::size_t i = 0; i < numTeams; ++i) {
      const std::size_t j     = randomIndices(i);
      searchedValuesTensor_h(i) = dataTensorBeforeOp_h(i, j);
    }
  } else {
    flare::fill_random(searchedValuesTensor_h, pool, 0, lowerBound);
  }

  flare::deep_copy(searchedValuesTensor, searchedValuesTensor_h);

  // use CTAD for functor
  TestFunctorA fnc(dataTensor, searchedValuesTensor, distancesTensor, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesTensor_h = create_host_space_copy(distancesTensor);

  for (std::size_t i = 0; i < dataTensor.extent(0); ++i) {
    auto rowFrom = flare::subtensor(dataTensorBeforeOp_h, i, flare::ALL());
    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);

    auto it = std::find(rowFromBegin, rowFromEnd, searchedValuesTensor_h(i));

    const std::size_t stdDistance      = KE::distance(rowFromBegin, it);
    const std::size_t beginEndDistance = KE::distance(rowFromBegin, rowFromEnd);

    if (searchedValuesExist) {
      REQUIRE_LT(stdDistance, beginEndDistance);
    } else {
      REQUIRE_EQ(stdDistance, beginEndDistance);
    }

    REQUIRE_EQ(stdDistance, distancesTensor_h(i));
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool searchedValuesExist) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(searchedValuesExist, numTeams, numCols,
                                     apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_find_team_test, searched_values_exist") {
  constexpr bool searchedValuesExist = true;

  run_all_scenarios<DynamicTag, double>(searchedValuesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValuesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValuesExist);
}

TEST_CASE("std_algorithms_find_team_test, searched_values_do_not_exist") {
  constexpr bool searchedValuesExist = false;

  run_all_scenarios<DynamicTag, double>(searchedValuesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValuesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValuesExist);
}

}  // namespace TeamFind
}  // namespace stdalgos
}  // namespace Test
