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

#ifndef FLARE_ALGORITHM_MIN_MAX_MINMAX_ELEMENT_IMPL_H_
#define FLARE_ALGORITHM_MIN_MAX_MINMAX_ELEMENT_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class ReducerType>
struct StdMinOrMaxElemFunctor {
  using index_type     = typename IteratorType::difference_type;
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ReducerType m_reducer;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    m_reducer.join(red_value, red_value_type{m_first[i], i});
  }

  FLARE_FUNCTION
  StdMinOrMaxElemFunctor(IteratorType first, ReducerType reducer)
      : m_first(std::move(first)), m_reducer(std::move(reducer)) {}
};

template <class IteratorType, class ReducerType>
struct StdMinMaxElemFunctor {
  using index_type     = typename IteratorType::difference_type;
  using red_value_type = typename ReducerType::value_type;
  IteratorType m_first;
  ReducerType m_reducer;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    const auto& my_value = m_first[i];
    m_reducer.join(red_value, red_value_type{my_value, my_value, i, i});
  }

  FLARE_FUNCTION
  StdMinMaxElemFunctor(IteratorType first, ReducerType reducer)
      : m_first(std::move(first)), m_reducer(std::move(reducer)) {}
};

//
// exespace impl
//
template <template <class... Args> class ReducerType, class ExecutionSpace,
          class IteratorType, class... Args>
IteratorType min_or_max_element_exespace_impl(const std::string& label,
                                              const ExecutionSpace& ex,
                                              IteratorType first,
                                              IteratorType last,
                                              Args&&... args) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    return last;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using value_type           = typename IteratorType::value_type;
  using reducer_type         = ReducerType<value_type, index_type, Args...>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t = StdMinOrMaxElemFunctor<IteratorType, reducer_type>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result, std::forward<Args>(args)...);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer), reducer);

  // fence not needed because reducing into scalar

  // return
  return first + red_result.loc;
}

template <template <class... Args> class ReducerType, class ExecutionSpace,
          class IteratorType, class... Args>
::flare::pair<IteratorType, IteratorType> minmax_element_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, Args&&... args) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    return {first, first};
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using value_type           = typename IteratorType::value_type;
  using reducer_type         = ReducerType<value_type, index_type, Args...>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t               = StdMinMaxElemFunctor<IteratorType, reducer_type>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result, std::forward<Args>(args)...);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer), reducer);

  // fence not needed because reducing into scalar

  // return
  return {first + red_result.min_loc, first + red_result.max_loc};
}

//
// team level impl
//
template <template <class... Args> class ReducerType, class TeamHandleType,
          class IteratorType, class... Args>
FLARE_FUNCTION IteratorType min_or_max_element_team_impl(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last,
    Args&&... args) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    return last;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using value_type           = typename IteratorType::value_type;
  using reducer_type         = ReducerType<value_type, index_type, Args...>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t = StdMinOrMaxElemFunctor<IteratorType, reducer_type>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result, std::forward<Args>(args)...);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements),
                            func_t(first, reducer), reducer);
  teamHandle.team_barrier();
  // maybe the barrier is not needed since reducing into scalar?

  // return
  return first + red_result.loc;
}

template <template <class... Args> class ReducerType, class TeamHandleType,
          class IteratorType, class... Args>
FLARE_FUNCTION ::flare::pair<IteratorType, IteratorType>
minmax_element_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                         IteratorType last, Args&&... args) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    return {first, first};
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using value_type           = typename IteratorType::value_type;
  using reducer_type         = ReducerType<value_type, index_type, Args...>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t               = StdMinMaxElemFunctor<IteratorType, reducer_type>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result, std::forward<Args>(args)...);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements),
                            func_t(first, reducer), reducer);
  teamHandle.team_barrier();
  // maybe the barrier is not needed since reducing into scalar?

  // return
  return {first + red_result.min_loc, first + red_result.max_loc};
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_MIN_MAX_MINMAX_ELEMENT_IMPL_H_
