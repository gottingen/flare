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

template <class DataViewType, class CompViewType, class ResultsViewType,
          class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  CompViewType m_compView;
  ResultsViewType m_resultsView;
  int m_apiPick;
  BinaryPredType m_binaryPred;

  TestFunctorA(const DataViewType dataView, const CompViewType compView,
               const ResultsViewType resultsView, int apiPick,
               BinaryPredType binaryPred)
      : m_dataView(dataView),
        m_compView(compView),
        m_resultsView(resultsView),
        m_apiPick(apiPick),
        m_binaryPred(binaryPred) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto rowIndex = member.league_rank();

    auto rowData         = flare::subview(m_dataView, rowIndex, flare::ALL());
    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    auto rowComp         = flare::subview(m_compView, rowIndex, flare::ALL());
    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    switch (m_apiPick) {
      case 0: {
        const bool result = KE::equal(member, dataBegin, dataEnd, compBegin);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 1: {
        const bool result =
            KE::equal(member, dataBegin, dataEnd, compBegin, m_binaryPred);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 2: {
        const bool result = KE::equal(member, rowData, rowComp);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 3: {
        const bool result = KE::equal(member, rowData, rowComp, m_binaryPred);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });

        break;
      }

      case 4: {
        const bool result =
            KE::equal(member, dataBegin, dataEnd, compBegin, compEnd);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 5: {
        const bool result = KE::equal(member, dataBegin, dataEnd, compBegin,
                                      compEnd, m_binaryPred);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool viewsAreEqual, std::size_t numTeams, std::size_t numCols,
            int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level equal
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

  // create a view to compare it with dataView. If viewsAreEqual == true,
  // compView is a copy of dataView. If viewsAreEqual == false, compView is
  // randomly filled
  auto compView   = create_deep_copyable_compatible_clone(dataView);
  auto compView_h = create_mirror_view(flare::HostSpace(), compView);
  if (viewsAreEqual) {
    flare::deep_copy(compView_h, dataViewBeforeOp_h);
  } else {
    using rand_pool =
        flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    flare::fill_random(compView_h, pool, lowerBound, upperBound);
  }

  flare::deep_copy(compView, compView_h);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // create the view to store results of equal()
  flare::View<bool*> resultsView("resultsView", numTeams);

  EqualFunctor<ValueType> binaryPred{};

  // use CTAD for functor
  TestFunctorA fnc(dataView, compView, resultsView, apiId, binaryPred);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto resultsView_h = create_host_space_copy(resultsView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowData = flare::subview(dataViewBeforeOp_h, i, flare::ALL());
    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    auto rowComp         = flare::subview(compView_h, i, flare::ALL());
    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    switch (apiId) {
      case 0:
      case 2: {
        const bool result = std::equal(dataBegin, dataEnd, compBegin);

        if (viewsAreEqual) {
          REQUIRE(resultsView_h(i));
        } else {
          REQUIRE_EQ(result, resultsView_h(i));
        }

        break;
      }

      case 1:
      case 3: {
        const bool result =
            std::equal(dataBegin, dataEnd, compBegin, binaryPred);

        if (viewsAreEqual) {
          REQUIRE(resultsView_h(i));
        } else {
          REQUIRE_EQ(result, resultsView_h(i));
        }

        break;
      }

      case 4: {
        const bool result = std::equal(dataBegin, dataEnd, compBegin, compEnd);

        if (viewsAreEqual) {
          REQUIRE(resultsView_h(i));
        } else {
          REQUIRE_EQ(result, resultsView_h(i));
        }

        break;
      }

      case 5: {
        const bool result =
            std::equal(dataBegin, dataEnd, compBegin, compEnd, binaryPred);

        if (viewsAreEqual) {
          REQUIRE(resultsView_h(i));
        } else {
          REQUIRE_EQ(result, resultsView_h(i));
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool viewsAreEqual) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3, 4, 5}) {
        test_A<LayoutTag, ValueType>(viewsAreEqual, numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_equal_team_test, views_are_equal") {
  constexpr bool viewsAreEqual = true;
  run_all_scenarios<DynamicTag, double>(viewsAreEqual);
  run_all_scenarios<StridedTwoRowsTag, int>(viewsAreEqual);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(viewsAreEqual);
}

TEST_CASE("std_algorithms_equal_team_test, views_are_not_equal") {
  constexpr bool viewsAreEqual = false;
  run_all_scenarios<DynamicTag, double>(viewsAreEqual);
  run_all_scenarios<StridedTwoRowsTag, int>(viewsAreEqual);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(viewsAreEqual);
}

}  // namespace TeamEqual
}  // namespace stdalgos
}  // namespace Test
