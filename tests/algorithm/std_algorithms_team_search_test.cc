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
namespace TeamSearch {

namespace KE = flare::experimental;

template <class ValueType>
struct EqualFunctor {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DatATensorType, class SearchedSequencesTensorType,
          class DistancesTensorType, class BinaryPredType>
struct TestFunctorA {
  DatATensorType m_dataTensor;
  SearchedSequencesTensorType m_searchedSequencesTensor;
  DistancesTensorType m_distancesTensor;
  BinaryPredType m_binaryPred;
  int m_apiPick;

  TestFunctorA(const DatATensorType dataTensor,
               const SearchedSequencesTensorType searchedSequencesTensor,
               const DistancesTensorType distancesTensor, BinaryPredType binaryPred,
               int apiPick)
      : m_dataTensor(dataTensor),
        m_searchedSequencesTensor(searchedSequencesTensor),
        m_distancesTensor(distancesTensor),
        m_binaryPred(binaryPred),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowTensorFrom = flare::subtensor(m_dataTensor, myRowIndex, flare::ALL());
    auto myRowSearchedSeqTensor =
        flare::subtensor(m_searchedSequencesTensor, myRowIndex, flare::ALL());

    switch (m_apiPick) {
      case 0: {
        const auto it = KE::search(
            member, KE::cbegin(myRowTensorFrom), KE::cend(myRowTensorFrom),
            KE::cbegin(myRowSearchedSeqTensor), KE::cend(myRowSearchedSeqTensor));
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesTensor(myRowIndex) =
              KE::distance(KE::cbegin(myRowTensorFrom), it);
        });

        break;
      }

      case 1: {
        const auto it = KE::search(member, myRowTensorFrom, myRowSearchedSeqTensor);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesTensor(myRowIndex) =
              KE::distance(KE::begin(myRowTensorFrom), it);
        });

        break;
      }

      case 2: {
        const auto it = KE::search(
            member, KE::cbegin(myRowTensorFrom), KE::cend(myRowTensorFrom),
            KE::cbegin(myRowSearchedSeqTensor), KE::cend(myRowSearchedSeqTensor),
            m_binaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesTensor(myRowIndex) =
              KE::distance(KE::cbegin(myRowTensorFrom), it);
        });

        break;
      }

      case 3: {
        const auto it = KE::search(member, myRowTensorFrom, myRowSearchedSeqTensor,
                                   m_binaryPred);
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
void test_A(const bool sequencesExist, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 tensor randomly filled with values,
     and run a team-level search
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

  // create a tensor that stores a sequence to found in dataTensor. If
  // sequencesExist == true it is filled base on dataTensor content, to allow
  // search to actually find anything. If sequencesExist == false it is filled
  // with random values greater than upperBound
  const std::size_t halfCols = (numCols > 1) ? ((numCols + 1) / 2) : (1);
  const std::size_t seqSize  = (numCols > 1) ? (std::log2(numCols)) : (1);

  flare::Tensor<ValueType**> searchedSequencesTensor("searchedSequencesTensor",
                                                  numTeams, seqSize);
  auto searchedSequencesTensor_h = create_host_space_copy(searchedSequencesTensor);

  if (sequencesExist) {
    const std::size_t dataBegin = halfCols - seqSize;
    for (std::size_t i = 0; i < searchedSequencesTensor_h.extent(0); ++i) {
      for (std::size_t js = 0, jd = dataBegin; js < seqSize; ++js, ++jd) {
        searchedSequencesTensor_h(i, js) = dataTensorBeforeOp_h(i, jd);
      }
    }
  } else {
    using rand_pool =
        flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    flare::fill_random(searchedSequencesTensor_h, pool, upperBound,
                        upperBound * 2);
  }

  flare::deep_copy(searchedSequencesTensor, searchedSequencesTensor_h);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // search returns an iterator so to verify that it is correct each team stores
  // the distance of the returned iterator from the beginning of the interval
  // that team operates on and then we check that these distances match the std
  // result
  flare::Tensor<std::size_t*> distancesTensor("distancesTensor", numTeams);

  EqualFunctor<ValueType> binaryPred;

  // use CTAD for functor
  TestFunctorA fnc(dataTensor, searchedSequencesTensor, distancesTensor, binaryPred,
                   apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesTensor_h = create_host_space_copy(distancesTensor);

  for (std::size_t i = 0; i < dataTensor.extent(0); ++i) {
    auto rowFrom = flare::subtensor(dataTensorBeforeOp_h, i, flare::ALL());
    auto rowSearchedSeq =
        flare::subtensor(searchedSequencesTensor_h, i, flare::ALL());

    const auto rowFromBegin     = KE::cbegin(rowFrom);
    const auto rowFromEnd       = KE::cend(rowFrom);
    const auto rowSearchedBegin = KE::cbegin(rowSearchedSeq);
    const auto rowSearchedEnd   = KE::cend(rowSearchedSeq);

    const std::size_t beginEndDistance = KE::distance(rowFromBegin, rowFromEnd);

    switch (apiId) {
      case 0:
      case 1: {
        const auto it = std::search(rowFromBegin, rowFromEnd, rowSearchedBegin,
                                    rowSearchedEnd);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        if (sequencesExist) {
          REQUIRE_LT(distancesTensor_h(i), beginEndDistance);
        } else {
          REQUIRE_EQ(distancesTensor_h(i), beginEndDistance);
        }

        REQUIRE_EQ(stdDistance, distancesTensor_h(i));

        break;
      }

      case 2:
      case 3: {
        const auto it = std::search(rowFromBegin, rowFromEnd, rowSearchedBegin,
                                    rowSearchedEnd, binaryPred);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        if (sequencesExist) {
          REQUIRE_LT(distancesTensor_h(i), beginEndDistance);
        } else {
          REQUIRE_EQ(distancesTensor_h(i), beginEndDistance);
        }

        REQUIRE_EQ(stdDistance, distancesTensor_h(i));

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool sequencesExist) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(sequencesExist, numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_search_team_test, sequences_exist") {
  constexpr bool sequencesExist = true;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

TEST_CASE("std_algorithms_search_team_test, sequences_do_not_exist") {
  constexpr bool sequencesExist = false;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

}  // namespace TeamSearch
}  // namespace stdalgos
}  // namespace Test
