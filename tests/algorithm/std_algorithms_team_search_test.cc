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

template <class DataViewType, class SearchedSequencesViewType,
          class DistancesViewType, class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  SearchedSequencesViewType m_searchedSequencesView;
  DistancesViewType m_distancesView;
  BinaryPredType m_binaryPred;
  int m_apiPick;

  TestFunctorA(const DataViewType dataView,
               const SearchedSequencesViewType searchedSequencesView,
               const DistancesViewType distancesView, BinaryPredType binaryPred,
               int apiPick)
      : m_dataView(dataView),
        m_searchedSequencesView(searchedSequencesView),
        m_distancesView(distancesView),
        m_binaryPred(binaryPred),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = flare::subview(m_dataView, myRowIndex, flare::ALL());
    auto myRowSearchedSeqView =
        flare::subview(m_searchedSequencesView, myRowIndex, flare::ALL());

    switch (m_apiPick) {
      case 0: {
        const auto it = KE::search(
            member, KE::cbegin(myRowViewFrom), KE::cend(myRowViewFrom),
            KE::cbegin(myRowSearchedSeqView), KE::cend(myRowSearchedSeqView));
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });

        break;
      }

      case 1: {
        const auto it = KE::search(member, myRowViewFrom, myRowSearchedSeqView);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });

        break;
      }

      case 2: {
        const auto it = KE::search(
            member, KE::cbegin(myRowViewFrom), KE::cend(myRowViewFrom),
            KE::cbegin(myRowSearchedSeqView), KE::cend(myRowSearchedSeqView),
            m_binaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });

        break;
      }

      case 3: {
        const auto it = KE::search(member, myRowViewFrom, myRowSearchedSeqView,
                                   m_binaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
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
     use a rank-2 view randomly filled with values,
     and run a team-level search
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr ValueType lowerBound = 5;
  constexpr ValueType upperBound = 523;
  const auto bounds              = make_bounds(lowerBound, upperBound);

  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataView");

  // create a view that stores a sequence to found in dataView. If
  // sequencesExist == true it is filled base on dataView content, to allow
  // search to actually find anything. If sequencesExist == false it is filled
  // with random values greater than upperBound
  const std::size_t halfCols = (numCols > 1) ? ((numCols + 1) / 2) : (1);
  const std::size_t seqSize  = (numCols > 1) ? (std::log2(numCols)) : (1);

  flare::View<ValueType**> searchedSequencesView("searchedSequencesView",
                                                  numTeams, seqSize);
  auto searchedSequencesView_h = create_host_space_copy(searchedSequencesView);

  if (sequencesExist) {
    const std::size_t dataBegin = halfCols - seqSize;
    for (std::size_t i = 0; i < searchedSequencesView_h.extent(0); ++i) {
      for (std::size_t js = 0, jd = dataBegin; js < seqSize; ++js, ++jd) {
        searchedSequencesView_h(i, js) = dataViewBeforeOp_h(i, jd);
      }
    }
  } else {
    using rand_pool =
        flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    flare::fill_random(searchedSequencesView_h, pool, upperBound,
                        upperBound * 2);
  }

  flare::deep_copy(searchedSequencesView, searchedSequencesView_h);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // search returns an iterator so to verify that it is correct each team stores
  // the distance of the returned iterator from the beginning of the interval
  // that team operates on and then we check that these distances match the std
  // result
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  EqualFunctor<ValueType> binaryPred;

  // use CTAD for functor
  TestFunctorA fnc(dataView, searchedSequencesView, distancesView, binaryPred,
                   apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = flare::subview(dataViewBeforeOp_h, i, flare::ALL());
    auto rowSearchedSeq =
        flare::subview(searchedSequencesView_h, i, flare::ALL());

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
          REQUIRE_LT(distancesView_h(i), beginEndDistance);
        } else {
          REQUIRE_EQ(distancesView_h(i), beginEndDistance);
        }

        REQUIRE_EQ(stdDistance, distancesView_h(i));

        break;
      }

      case 2:
      case 3: {
        const auto it = std::search(rowFromBegin, rowFromEnd, rowSearchedBegin,
                                    rowSearchedEnd, binaryPred);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        if (sequencesExist) {
          REQUIRE_LT(distancesView_h(i), beginEndDistance);
        } else {
          REQUIRE_EQ(distancesView_h(i), beginEndDistance);
        }

        REQUIRE_EQ(stdDistance, distancesView_h(i));

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
