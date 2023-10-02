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

#ifndef FLARE_ALGORITHM_ADJACENT_FIND_IMPL_H_
#define FLARE_ALGORITHM_ADJACENT_FIND_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class ReducerType, class PredicateType>
struct StdAdjacentFindFunctor {
  using index_type     = typename IteratorType::difference_type;
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    const auto& my_value   = m_first[i];
    const auto& next_value = m_first[i + 1];
    const bool are_equal   = m_p(my_value, next_value);

    // FIXME_NVHPC using a ternary operator causes problems
    red_value_type value = {::flare::reduction_identity<index_type>::min()};
    if (are_equal) {
      value.min_loc_true = i;
    }

    m_reducer.join(red_value, value);
  }

  FLARE_FUNCTION
  StdAdjacentFindFunctor(IteratorType first, ReducerType reducer,
                         PredicateType p)
      : m_first(std::move(first)),
        m_reducer(std::move(reducer)),
        m_p(std::move(p)) {}
};

//
// exespace impl
//
template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType adjacent_find_exespace_impl(const std::string& label,
                                         const ExecutionSpace& ex,
                                         IteratorType first, IteratorType last,
                                         PredicateType pred) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  const auto num_elements = flare::experimental::distance(first, last);

  if (num_elements <= 1) {
    return last;
  }

  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;

  reduction_value_type red_result;
  reducer_type reducer(red_result);

  // note that we use below num_elements-1 because
  // each index i in the reduction checks i and (i+1).
  ::flare::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements - 1),
      // use CTAD
      StdAdjacentFindFunctor(first, reducer, pred), reducer);

  // fence not needed because reducing into scalar
  if (red_result.min_loc_true ==
      ::flare::reduction_identity<index_type>::min()) {
    return last;
  } else {
    return first + red_result.min_loc_true;
  }
}

template <class ExecutionSpace, class IteratorType>
IteratorType adjacent_find_exespace_impl(const std::string& label,
                                         const ExecutionSpace& ex,
                                         IteratorType first,
                                         IteratorType last) {
  using value_type     = typename IteratorType::value_type;
  using default_pred_t = StdAlgoEqualBinaryPredicate<value_type>;
  return adjacent_find_exespace_impl(label, ex, first, last, default_pred_t());
}

//
// team impl
//
template <class TeamHandleType, class IteratorType, class PredicateType>
FLARE_FUNCTION IteratorType
adjacent_find_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                        IteratorType last, PredicateType pred) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  const auto num_elements = flare::experimental::distance(first, last);

  if (num_elements <= 1) {
    return last;
  }

  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;

  reduction_value_type red_result;
  reducer_type reducer(red_result);

  // note that we use below num_elements-1 because
  // each index i in the reduction checks i and (i+1).
  ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements - 1),
                            // use CTAD
                            StdAdjacentFindFunctor(first, reducer, pred),
                            reducer);

  teamHandle.team_barrier();

  if (red_result.min_loc_true ==
      ::flare::reduction_identity<index_type>::min()) {
    return last;
  } else {
    return first + red_result.min_loc_true;
  }
}

template <class TeamHandleType, class IteratorType>
FLARE_FUNCTION IteratorType adjacent_find_team_impl(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last) {
  using value_type     = typename IteratorType::value_type;
  using default_pred_t = StdAlgoEqualBinaryPredicate<value_type>;
  return adjacent_find_team_impl(teamHandle, first, last, default_pred_t());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ADJACENT_FIND_IMPL_H_
