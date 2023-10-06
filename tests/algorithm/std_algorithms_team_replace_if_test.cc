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
namespace TeamReplaceIf {

namespace KE = flare::experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  FLARE_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  FLARE_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class ViewType, class ValueType>
struct TestFunctorA {
  ViewType m_view;
  ValueType m_threshold;
  ValueType m_newVal;
  int m_apiPick;

  TestFunctorA(const ViewType view, ValueType threshold, ValueType newVal,
               int apiPick)
      : m_view(view),
        m_threshold(threshold),
        m_newVal(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  FLARE_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = flare::subview(m_view, myRowIndex, flare::ALL());

    GreaterThanValueFunctor predicate(m_threshold);
    if (m_apiPick == 0) {
      KE::replace_if(member, KE::begin(myRowView), KE::end(myRowView),
                     predicate, m_newVal);
    } else if (m_apiPick == 1) {
      KE::replace_if(member, myRowView, predicate, m_newVal);
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level replace_if where the values strictly greater
     than a threshold are replaced with a new value.
   */
  const auto threshold = static_cast<ValueType>(151);
  const auto newVal    = static_cast<ValueType>(1);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [dataView, cloneOfDataViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          flare::pair<ValueType, ValueType>{5, 523}, "dataView");

  // -----------------------------------------------
  // launch flare kernel
  // -----------------------------------------------
  using space_t = flare::DefaultExecutionSpace;
  flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());
  // use CTAD for functor
  TestFunctorA fnc(dataView, threshold, newVal, apiId);
  flare::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel
  // -----------------------------------------------
  flare::View<ValueType**, flare::HostSpace> stdDataView("stdDataView",
                                                           numTeams, numCols);
  // ensure that we use the same data to run the std algo on
  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    for (std::size_t j = 0; j < dataView.extent(1); ++j) {
      stdDataView(i, j) = cloneOfDataViewBeforeOp_h(i, j);
    }
  }
  GreaterThanValueFunctor predicate(threshold);
  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto thisRow = flare::subview(stdDataView, i, flare::ALL());
    std::replace_if(KE::begin(thisRow), KE::end(thisRow), predicate, newVal);
  }

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  expect_equal_host_views(stdDataView, dataViewAfterOp_h);
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

TEST_CASE("std_algorithms_replace_if_team_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplaceIf
}  // namespace stdalgos
}  // namespace Test
