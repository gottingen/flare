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

#ifndef FLARE_ALGORITHM_FIND_IF_AND_FIND_IF_NOT_IMPL_H_
#define FLARE_ALGORITHM_FIND_IF_AND_FIND_IF_NOT_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <bool is_find_if, class IndexType, class IteratorType,
          class ReducerType, class PredicateType>
struct StdFindIfOrNotFunctor {
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  FLARE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    const auto& my_value = m_first[i];

    // if doing find_if, look for when predicate is true
    // if doing find_if_not, look for when predicate is false
    const bool found_condition = is_find_if ? m_p(my_value) : !m_p(my_value);

    // FIXME_NVHPC using a ternary operator causes problems
    red_value_type rv = {::flare::reduction_identity<IndexType>::min()};
    if (found_condition) {
      rv.min_loc_true = i;
    }

    m_reducer.join(red_value, rv);
  }

  FLARE_FUNCTION
  StdFindIfOrNotFunctor(IteratorType first, ReducerType reducer,
                        PredicateType p)
      : m_first(std::move(first)),
        m_reducer(std::move(reducer)),
        m_p(std::move(p)) {}
};

//
// exespace impl
//
template <bool is_find_if, class ExecutionSpace, class IteratorType,
          class PredicateType>
IteratorType find_if_or_not_exespace_impl(const std::string& label,
                                          const ExecutionSpace& ex,
                                          IteratorType first, IteratorType last,
                                          PredicateType pred) {
  // checks
  detail::static_assert_random_access_and_accessible(
      ex, first);  // only need one It per type
  detail::expect_valid_range(first, last);

  if (first == last) {
    return last;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t = StdFindIfOrNotFunctor<is_find_if, index_type, IteratorType,
                                       reducer_type, PredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer, pred), reducer);

  // fence not needed because reducing into scalar

  // decide and return
  if (red_result.min_loc_true ==
      ::flare::reduction_identity<index_type>::min()) {
    // here, it means a valid loc has not been found,
    return last;
  } else {
    // a location has been found
    return first + red_result.min_loc_true;
  }
}

template <class ExecutionSpace, class InputIterator, class T>
InputIterator find_exespace_impl(const std::string& label, ExecutionSpace ex,
                                 InputIterator first, InputIterator last,
                                 const T& value) {
  return find_if_or_not_exespace_impl<true>(
      label, ex, first, last,
      ::flare::experimental::detail::StdAlgoEqualsValUnaryPredicate<T>(value));
}

//
// team impl
//
template <bool is_find_if, class TeamHandleType, class IteratorType,
          class PredicateType>
FLARE_FUNCTION IteratorType
find_if_or_not_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                         IteratorType last, PredicateType pred) {
  // checks
  detail::static_assert_random_access_and_accessible(
      teamHandle, first);  // only need one It per type
  detail::expect_valid_range(first, last);

  if (first == last) {
    return last;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t = StdFindIfOrNotFunctor<is_find_if, index_type, IteratorType,
                                       reducer_type, PredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements),
                            func_t(first, reducer, pred), reducer);

  teamHandle.team_barrier();

  // decide and return
  if (red_result.min_loc_true ==
      ::flare::reduction_identity<index_type>::min()) {
    // here, it means a valid loc has not been found,
    return last;
  } else {
    // a location has been found
    return first + red_result.min_loc_true;
  }
}

template <class TeamHandleType, class InputIterator, class T>
FLARE_FUNCTION InputIterator find_team_impl(const TeamHandleType& teamHandle,
                                             InputIterator first,
                                             InputIterator last,
                                             const T& value) {
  return find_if_or_not_team_impl<true>(
      teamHandle, first, last,
      ::flare::experimental::detail::StdAlgoEqualsValUnaryPredicate<T>(value));
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FIND_IF_AND_FIND_IF_NOT_IMPL_H_
