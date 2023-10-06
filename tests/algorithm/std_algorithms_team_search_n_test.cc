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
namespace TeamSearchN {

namespace KE = flare::experimental;

template <class ValueType>
struct EqualFunctor {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DataViewType, class SearchedValuesViewType,
          class DistancesViewType, class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  std::size_t m_seqSize;
  SearchedValuesViewType m_searchedValuesView;
  DistancesViewType m_distancesView;
  BinaryPredType m_binaryPred;
  int m_apiPick;

  TestFunctorA(const DataViewType dataView, std::size_t seqSize,
               const SearchedValuesViewType searchedValuesView,
               const DistancesViewType distancesView, BinaryPredType binaryPred,
               int apiPick)
      : m_dataView(dataView),
        m_seqSize(seqSize),
        m_searchedValuesView(searchedValuesView),
        m_distancesView(distancesView),
        m_binaryPred(binaryPred),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = flare::subview(m_dataView, myRowIndex, flare::ALL());
    auto rowFromBegin  = KE::begin(myRowViewFrom);
    auto rowFromEnd    = KE::end(myRowViewFrom);
    const auto searchedValue = m_searchedValuesView(myRowIndex);

    switch (m_apiPick) {
      case 0: {
        const auto it = KE::search_n(member, rowFromBegin, rowFromEnd,
                                     m_seqSize, searchedValue);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
        });

        break;
      }

      case 1: {
        const auto it =
            KE::search_n(member, myRowViewFrom, m_seqSize, searchedValue);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
        });

        break;
      }

      case 2: {
        const auto it = KE::search_n(member, rowFromBegin, rowFromEnd,
                                     m_seqSize, searchedValue, m_binaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
        });

        break;
      }

      case 3: {
        const auto it = KE::search_n(member, myRowViewFrom, m_seqSize,
                                     searchedValue, m_binaryPred);
        flare::single(flare::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
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
     and run a team-level search_n
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

  // If sequencesExist == true we need to inject some sequence of count test
  // value into dataView. If sequencesExist == false we set searchedVal to a
  // value that is not present in dataView

  const std::size_t halfCols = (numCols > 1) ? ((numCols + 1) / 2) : (1);
  const std::size_t seqSize  = (numCols > 1) ? (std::log2(numCols)) : (1);

  flare::View<ValueType*> searchedValuesView("searchedValuesView", numTeams);
  auto searchedValuesView_h = create_host_space_copy(searchedValuesView);

  // dataView might not deep copyable (e.g. strided layout) so to prepare it
  // correclty, we make a new view that is for sure deep copyable, modify it
  // on the host, deep copy to device and then launch a kernel to copy to
  // dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(flare::HostSpace(), dataView_dc);

  if (sequencesExist) {
    const std::size_t dataBegin = halfCols - seqSize;
    for (std::size_t i = 0; i < searchedValuesView.extent(0); ++i) {
      const ValueType searchedVal = dataView_dc_h(i, dataBegin);
      searchedValuesView_h(i)     = searchedVal;

      for (std::size_t j = dataBegin + 1; j < seqSize; ++j) {
        dataView_dc_h(i, j) = searchedVal;
      }
    }

    // copy to dataView_dc and then to dataView
    flare::deep_copy(dataView_dc, dataView_dc_h);

    CopyFunctorRank2 cpFun(dataView_dc, dataView);
    flare::parallel_for("copy", dataView.extent(0) * dataView.extent(1),
                         cpFun);
  } else {
    using rand_pool =
        flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    flare::fill_random(searchedValuesView_h, pool, upperBound, upperBound * 2);
  }

  flare::deep_copy(searchedValuesView, searchedValuesView_h);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // search_n returns an iterator so to verify that it is correct each team
  // stores the distance of the returned iterator from the beginning of the
  // interval that team operates on and then we check that these distances match
  // the std result
  flare::View<std::size_t*> distancesView("distancesView", numTeams);

  EqualFunctor<ValueType> binaryPred;

  // use CTAD for functor
  TestFunctorA fnc(dataView, seqSize, searchedValuesView, distancesView,
                   binaryPred, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = flare::subview(dataView_dc_h, i, flare::ALL());

    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);

    const ValueType searchedVal = searchedValuesView_h(i);

    const std::size_t beginEndDist = KE::distance(rowFromBegin, rowFromEnd);

    switch (apiId) {
      case 0:
      case 1: {
        const auto it =
            std::search_n(rowFromBegin, rowFromEnd, seqSize, searchedVal);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        if (sequencesExist) {
          REQUIRE_LT(distancesView_h(i), beginEndDist);
        } else {
          REQUIRE_EQ(distancesView_h(i), beginEndDist);
        }

        REQUIRE_EQ(stdDistance, distancesView_h(i));

        break;
      }

      case 2:
      case 3: {
        const auto it = std::search_n(rowFromBegin, rowFromEnd, seqSize,
                                      searchedVal, binaryPred);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        if (sequencesExist) {
          REQUIRE_LT(distancesView_h(i), beginEndDist);
        } else {
          REQUIRE_EQ(distancesView_h(i), beginEndDist);
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
    for (const auto& numCols : {2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(sequencesExist, numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_search_n_team_test, sequences_of_equal_elements_exist") {
  constexpr bool sequencesExist = true;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

TEST_CASE("std_algorithms_search_n_team_test, sequences_of_equal_elements_probably_does_not_exist") {
  constexpr bool sequencesExist = false;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

}  // namespace TeamSearchN
}  // namespace stdalgos
}  // namespace Test
