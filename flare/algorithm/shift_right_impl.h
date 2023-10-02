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

#ifndef FLARE_ALGORITHM_SHIFT_RIGHT_IMPL_H_
#define FLARE_ALGORITHM_SHIFT_RIGHT_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/move.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class ExecutionSpace, class IteratorType>
IteratorType shift_right_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, typename IteratorType::difference_type n) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);
  FLARE_EXPECTS(n >= 0);

  // handle trivial cases
  if (n == 0) {
    return first;
  }

  if (n >= flare::experimental::distance(first, last)) {
    return last;
  }

  /*
    Suppose that n = 3, and [first,last) spans:

    | 0  | 1  |  2 | 1  | 2  | 1  | 2  | 2  | 10 | -3 | 1  | -6 | *
      ^                         				  ^
    first							 last

    shift_right modifies the range such that we have this data:
    |  x | x  | x  | 0  | 1  |  2 | 1  | 2  | 1  | 2  | 2  | 10 | *
                     ^
             return it points here

    and returns an iterator pointing to the new beginning.
    Note: elements marked x are in undefined state because have been moved.

    We implement this in two steps:
    step 1:
      we create a temporary view with extent = distance(first, last-n)
      and *move* assign the elements from [first, last-n) to tmp view, such that
      tmp view becomes:

      | 0  | 1  |  2 | 1  | 2  | 1  | 2  | 2  | 10 |

    step 2:
      move elements of tmp view back to range starting at first+n.
   */

  const auto num_elements_to_move =
      ::flare::experimental::distance(first, last - n);

  // create tmp view
  using value_type    = typename IteratorType::value_type;
  using tmp_view_type = flare::View<value_type*, ExecutionSpace>;
  tmp_view_type tmp_view("shift_right_impl", num_elements_to_move);
  using tmp_readwrite_iterator_type = decltype(begin(tmp_view));

  using index_type = typename IteratorType::difference_type;

  // step 1
  using step1_func_type =
      StdMoveFunctor<index_type, IteratorType, tmp_readwrite_iterator_type>;
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_to_move),
      step1_func_type(first, begin(tmp_view)));

  // step 2
  using step2_func_type =
      StdMoveFunctor<index_type, tmp_readwrite_iterator_type, IteratorType>;
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, tmp_view.extent(0)),
                         step2_func_type(begin(tmp_view), first + n));

  ex.fence("flare::shift_right: fence after operation");

  return first + n;
}

template <class Iterator>
struct StdShiftRightTeamSingleFunctor {
  Iterator m_first;
  Iterator m_last;
  std::size_t m_shift;

  FLARE_FUNCTION
  void operator()() const {
    // the impl function calling this functor guarantees that
    // - m_shift is non-negative
    // - m_first, m_last identify a valid range with m_last > m_first
    // - m_shift is less than m_last - m_first
    // so I can safely use std::size_t here
  }

  FLARE_FUNCTION
  StdShiftRightTeamSingleFunctor(Iterator _first, Iterator _last, std::size_t n)
      : m_first(std::move(_first)), m_last(std::move(_last)), m_shift(n) {}
};

template <class TeamHandleType, class IteratorType>
FLARE_FUNCTION IteratorType shift_right_team_impl(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last,
    typename IteratorType::difference_type n) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);
  FLARE_EXPECTS(n >= 0);

  // handle trivial cases
  if (n == 0) {
    return first;
  }

  if (n >= flare::experimental::distance(first, last)) {
    return last;
  }

  // we cannot use here a new allocation like we do for the
  // execution space impl because for this team impl we are
  // within a parallel region, so for now we solve serially

  const std::size_t numElementsToMove =
      ::flare::experimental::distance(first, last - n);
  flare::single(flare::PerTeam(teamHandle), [=]() {
    for (std::size_t i = 0; i < numElementsToMove; ++i) {
      last[-i - 1] = std::move(last[-n - i - 1]);
    }
  });
  teamHandle.team_barrier();

  return first + n;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_SHIFT_RIGHT_IMPL_H_
