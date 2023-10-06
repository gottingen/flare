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
namespace TeamLexicographicalCompare {

namespace KE = flare::experimental;

enum class TestCaseType { ViewsAreEqual, FirstIsLess, FirstIsGreater };

template <class ValueType>
struct LessFunctor {
  FLARE_INLINE_FUNCTION bool operator()(const ValueType& lhs,
                                         const ValueType& rhs) const {
    return lhs < rhs;
  }
};

template <class DataViewType, class CompViewType, class ResultsViewType,
          class BinaryCompType>
struct TestFunctorA {
  DataViewType m_dataView;
  CompViewType m_compView;
  ResultsViewType m_resultsView;
  int m_apiPick;
  BinaryCompType m_binaryComp;

  TestFunctorA(const DataViewType dataView, const CompViewType compView,
               const ResultsViewType resultsView, int apiPick,
               BinaryCompType binaryComp)
      : m_dataView(dataView),
        m_compView(compView),
        m_resultsView(resultsView),
        m_apiPick(apiPick),
        m_binaryComp(binaryComp) {}

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
        const bool result = KE::lexicographical_compare(
            member, dataBegin, dataEnd, compBegin, compEnd);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 1: {
        const bool result =
            KE::lexicographical_compare(member, rowData, rowComp);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 2: {
        const bool result = KE::lexicographical_compare(
            member, dataBegin, dataEnd, compBegin, compEnd, m_binaryComp);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 3: {
        const bool result =
            KE::lexicographical_compare(member, rowData, rowComp, m_binaryComp);
        flare::single(flare::PerTeam(member),
                       [=, *this]() { m_resultsView(rowIndex) = result; });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const TestCaseType testCase, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level lexicographical_compare
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

  // create a view to compare it with dataView. If testCase == ViewsAreEqual,
  // compView is a copy of dataView. If testCase == FirstIsLess, we want the
  // dataView to be lexicographically less (and compView - greater). If testCase
  // == FirstIsGreater, we want the dataView to be lexicographically greater
  // (and compView - less).
  auto compEqualView   = create_deep_copyable_compatible_clone(dataView);
  auto compEqualView_h = create_mirror_view(flare::HostSpace(), compEqualView);
  flare::deep_copy(compEqualView_h, dataViewBeforeOp_h);
  const auto middle = numCols / 2;
  switch (testCase) {
    case TestCaseType::ViewsAreEqual: {
      // Do nothing - deep_copy was already done
      break;
    }

    case TestCaseType::FirstIsLess: {
      for (std::size_t i = 0; i < compEqualView_h.extent(0); ++i) {
        compEqualView_h(i, middle) += 1;
      }

      break;
    }

    case TestCaseType::FirstIsGreater: {
      for (std::size_t i = 0; i < compEqualView_h.extent(0); ++i) {
        compEqualView_h(i, middle) -= 1;
      }

      break;
    }
  }

  flare::deep_copy(compEqualView, compEqualView_h);

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

  // create the view to store results of equal()
  flare::View<bool*> resultsView("resultsView", numTeams);

  LessFunctor<ValueType> binaryComp{};

  // use CTAD for functor
  TestFunctorA fnc(dataView, compEqualView, resultsView, apiId, binaryComp);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto resultsView_h = create_host_space_copy(resultsView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowData = flare::subview(dataViewBeforeOp_h, i, flare::ALL());
    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    auto rowComp         = flare::subview(compEqualView_h, i, flare::ALL());
    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    switch (apiId) {
      case 0:
      case 1: {
        const bool result = std::lexicographical_compare(dataBegin, dataEnd,
                                                         compBegin, compEnd);

        switch (testCase) {
          case TestCaseType::ViewsAreEqual:
          case TestCaseType::FirstIsGreater: {
            REQUIRE_FALSE(resultsView_h(i));
            REQUIRE_EQ(result, resultsView_h(i));
            break;
          }

          case TestCaseType::FirstIsLess: {
            REQUIRE(resultsView_h(i));
            REQUIRE_EQ(result, resultsView_h(i));
            break;
          }
        }

        break;
      }

      case 2:
      case 3: {
        const bool result = std::lexicographical_compare(
            dataBegin, dataEnd, compBegin, compEnd, binaryComp);

        switch (testCase) {
          case TestCaseType::ViewsAreEqual:
          case TestCaseType::FirstIsGreater: {
            REQUIRE_FALSE(resultsView_h(i));
            REQUIRE_EQ(result, resultsView_h(i));
            break;
          }

          case TestCaseType::FirstIsLess: {
            REQUIRE(resultsView_h(i));
            REQUIRE_EQ(result, resultsView_h(i));
            break;
          }
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const TestCaseType testCase) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(testCase, numTeams, numCols, apiId);
      }
    }
  }
}

TEST_CASE("std_algorithms_lexicographical_compare_team_test, views_are_equal") {
  constexpr TestCaseType testCaseType = TestCaseType::ViewsAreEqual;
  run_all_scenarios<DynamicTag, double>(testCaseType);
  run_all_scenarios<StridedTwoRowsTag, int>(testCaseType);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(testCaseType);
}

TEST_CASE("std_algorithms_lexicographical_compare_team_test, first_view_is_less") {
  constexpr TestCaseType testCaseType = TestCaseType::FirstIsLess;
  run_all_scenarios<DynamicTag, double>(testCaseType);
  run_all_scenarios<StridedTwoRowsTag, int>(testCaseType);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(testCaseType);
}

TEST_CASE("std_algorithms_lexicographical_compare_team_test, first_view_is_greater") {
  constexpr TestCaseType testCaseType = TestCaseType::FirstIsGreater;
  run_all_scenarios<DynamicTag, double>(testCaseType);
  run_all_scenarios<StridedTwoRowsTag, int>(testCaseType);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(testCaseType);
}

}  // namespace TeamLexicographicalCompare
}  // namespace stdalgos
}  // namespace Test
