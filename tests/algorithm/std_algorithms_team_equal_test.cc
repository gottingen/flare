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
namespace TeamEqual {

namespace KE = flare::experimental;

template <class ValueType>
struct EqualFunctor {
  FLARE_INLINE_FUNCTION bool operator()(const ValueType& lhs,
                                         const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DatATensorType, class CompTensorType, class ResultsTensorType,
          class BinaryPredType>
struct TestFunctorA {
  DatATensorType m_dataTensor;
  CompTensorType m_compTensor;
  ResultsTensorType m_resultsTensor;
  int m_apiPick;
  BinaryPredType m_binaryPred;

  TestFunctorA(const DatATensorType dataTensor, const CompTensorType compTensor,
               const ResultsTensorType resultsTensor, int apiPick,
               BinaryPredType binaryPred)
      : m_dataTensor(dataTensor),
        m_compTensor(compTensor),
        m_resultsTensor(resultsTensor),
        m_apiPick(apiPick),
        m_binaryPred(binaryPred) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto rowIndex = member.league_rank();

    auto rowData         = flare::subtensor(m_dataTensor, rowIndex, flare::ALL());
    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    auto rowComp         = flare::subtensor(m_compTensor, rowIndex, flare::ALL());
    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    switch (m_apiPick) {
      case 0: {
        const bool result = KE::equal(member, dataBegin, dataEnd, compBegin);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsTensor(rowIndex) = result; });
        break;
      }

      case 1: {
        const bool result =
            KE::equal(member, dataBegin, dataEnd, compBegin, m_binaryPred);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsTensor(rowIndex) = result; });
        break;
      }

      case 2: {
        const bool result = KE::equal(member, rowData, rowComp);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsTensor(rowIndex) = result; });
        break;
      }

      case 3: {
        const bool result = KE::equal(member, rowData, rowComp, m_binaryPred);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsTensor(rowIndex) = result; });

        break;
      }

      case 4: {
        const bool result =
            KE::equal(member, dataBegin, dataEnd, compBegin, compEnd);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsTensor(rowIndex) = result; });
        break;
      }

      case 5: {
        const bool result = KE::equal(member, dataBegin, dataEnd, compBegin,
                                      compEnd, m_binaryPred);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsTensor(rowIndex) = result; });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool tensorsAreEqual, std::size_t numTeams, std::size_t numCols,
            int apiId) {
  /* description:
     use a rank-2 tensor randomly filled with values,
     and run a team-level equal
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

  // create a tensor to compare it with dataTensor. If tensorsAreEqual == true,
  // compTensor is a copy of dataTensor. If tensorsAreEqual == false, compTensor is
  // randomly filled
  auto compTensor   = create_deep_copyable_compatible_clone(dataTensor);
  auto compTensor_h = create_mirror_tensor(flare::HostSpace(), compTensor);
  if (tensorsAreEqual) {
    flare::deep_copy(compTensor_h, dataTensorBeforeOp_h);
  } else {
    using rand_pool =
        flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    flare::fill_random(compTensor_h, pool, lowerBound, upperBound);
  }

  flare::deep_copy(compTensor, compTensor_h);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // create the tensor to store results of equal()
  flare::Tensor<bool*> resultsTensor("resultsTensor", numTeams);

  EqualFunctor<ValueType> binaryPred{};

  // use CTAD for functor
  TestFunctorA fnc(dataTensor, compTensor, resultsTensor, apiId, binaryPred);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto resultsTensor_h = create_host_space_copy(resultsTensor);

  for (std::size_t i = 0; i < dataTensor.extent(0); ++i) {
    auto rowData = flare::subtensor(dataTensorBeforeOp_h, i, flare::ALL());
    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    auto rowComp         = flare::subtensor(compTensor_h, i, flare::ALL());
    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    switch (apiId) {
      case 0:
      case 2: {
        const bool result = std::equal(dataBegin, dataEnd, compBegin);

        if (tensorsAreEqual) {
          REQUIRE(resultsTensor_h(i));
        } else {
          REQUIRE_EQ(result, resultsTensor_h(i));
        }

        break;
      }

      case 1:
      case 3: {
        const bool result =
            std::equal(dataBegin, dataEnd, compBegin, binaryPred);

        if (tensorsAreEqual) {
          REQUIRE(resultsTensor_h(i));
        } else {
          REQUIRE_EQ(result, resultsTensor_h(i));
        }

        break;
      }

      case 4: {
        const bool result = std::equal(dataBegin, dataEnd, compBegin, compEnd);

        if (tensorsAreEqual) {
          REQUIRE(resultsTensor_h(i));
        } else {
          REQUIRE_EQ(result, resultsTensor_h(i));
        }

        break;
      }

      case 5: {
        const bool result =
            std::equal(dataBegin, dataEnd, compBegin, compEnd, binaryPred);

        if (tensorsAreEqual) {
          REQUIRE(resultsTensor_h(i));
        } else {
          REQUIRE_EQ(result, resultsTensor_h(i));
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool tensorsAreEqual) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3, 4, 5}) {
        test_A<LayoutTag, ValueType>(tensorsAreEqual, numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_equal_team_test, tensors_are_equal") {
  constexpr bool tensorsAreEqual = true;
  run_all_scenarios<DynamicTag, double>(tensorsAreEqual);
  run_all_scenarios<StridedTwoRowsTag, int>(tensorsAreEqual);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(tensorsAreEqual);
}

TEST_CASE("std_algorithms_equal_team_test, tensors_are_not_equal") {
  constexpr bool tensorsAreEqual = false;
  run_all_scenarios<DynamicTag, double>(tensorsAreEqual);
  run_all_scenarios<StridedTwoRowsTag, int>(tensorsAreEqual);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(tensorsAreEqual);
}

}  // namespace TeamEqual
}  // namespace stdalgos
}  // namespace Test
