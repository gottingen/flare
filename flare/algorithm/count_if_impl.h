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

#ifndef FLARE_ALGORITHM_COUNT_IF_IMPL_H_
#define FLARE_ALGORITHM_COUNT_IF_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class Predicate>
struct StdCountIfFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  Predicate m_predicate;

  FLARE_FUNCTION
  void operator()(index_type i, index_type& lsum) const {
    if (m_predicate(m_first[i])) {
      lsum++;
    }
  }

  FLARE_FUNCTION
  StdCountIfFunctor(IteratorType _first, Predicate _predicate)
      : m_first(std::move(_first)), m_predicate(std::move(_predicate)) {}
};

template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, Predicate predicate) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  typename IteratorType::difference_type count = 0;
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            // use CTAD
                            StdCountIfFunctor(first, predicate), count);
  ex.fence("flare::count_if: fence after operation");

  return count;
}

template <class ExecutionSpace, class IteratorType, class T>
auto count_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last,
                         const T& value) {
  return count_if_exespace_impl(
      label, ex, first, last,
      ::flare::experimental::detail::StdAlgoEqualsValUnaryPredicate<T>(value));
}

//
// team-level impl
//
template <class TeamHandleType, class IteratorType, class Predicate>
FLARE_FUNCTION typename IteratorType::difference_type count_if_team_impl(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last,
    Predicate predicate) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  typename IteratorType::difference_type count = 0;
  ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements),
                            // use CTAD
                            StdCountIfFunctor(first, predicate), count);
  teamHandle.team_barrier();

  return count;
}

template <class TeamHandleType, class IteratorType, class T>
FLARE_FUNCTION auto count_team_impl(const TeamHandleType& teamHandle,
                                     IteratorType first, IteratorType last,
                                     const T& value) {
  return count_if_team_impl(
      teamHandle, first, last,
      ::flare::experimental::detail::StdAlgoEqualsValUnaryPredicate<T>(value));
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_COUNT_IF_IMPL_H_
