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

#ifndef FLARE_ALGORITHM_SHIFT_LEFT_IMPL_H_
#define FLARE_ALGORITHM_SHIFT_LEFT_IMPL_H_

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
IteratorType shift_left_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, typename IteratorType::difference_type n) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);
  FLARE_EXPECTS(n >= 0);

  // handle trivial cases
  if (n == 0) {
    return last;
  }

  if (n >= flare::experimental::distance(first, last)) {
    return first;
  }

  /*
    Suppose that n = 5, and our [first,last) spans:

    | 0  | 1  |  2 | 1  | 2  | 1  | 2  | 2  | 10 | -3 | 1  | -6 | *
      ^                         				  ^
    first							 last

    shift_left modifies the range such that we have this data:
    | 1  | 2  | 2  | 10  | -3 | 1  | -6 | x | x  | x  | x  |  x | *
                                          ^
                                   return it pointing here


    and returns an iterator pointing to one past the new end.
    Note: elements marked x are in undefined state because have been moved.

    We implement this in two steps:
    step 1:
      we create a temporary view with extent = distance(first+n, last)
      and *move* assign the elements from [first+n, last) to tmp view, such that
      tmp view becomes:

      | 1  | 2  | 2  | 10  | -3 | 1  | -6 |

    step 2:
      move elements of tmp view back to range starting at first.
   */

  const auto num_elements_to_move =
      ::flare::experimental::distance(first + n, last);

  // create tmp view
  using value_type    = typename IteratorType::value_type;
  using tmp_view_type = flare::View<value_type*, ExecutionSpace>;
  tmp_view_type tmp_view("shift_left_impl", num_elements_to_move);
  using tmp_readwrite_iterator_type = decltype(begin(tmp_view));

  using index_type = typename IteratorType::difference_type;

  // step 1
  using step1_func_type =
      StdMoveFunctor<index_type, IteratorType, tmp_readwrite_iterator_type>;
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_to_move),
      step1_func_type(first + n, begin(tmp_view)));

  // step 2
  using step2_func_type =
      StdMoveFunctor<index_type, tmp_readwrite_iterator_type, IteratorType>;
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, tmp_view.extent(0)),
                         step2_func_type(begin(tmp_view), first));

  ex.fence("flare::shift_left: fence after operation");

  return last - n;
}

template <class TeamHandleType, class IteratorType>
FLARE_FUNCTION IteratorType shift_left_team_impl(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last,
    typename IteratorType::difference_type n) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);
  FLARE_EXPECTS(n >= 0);

  // handle trivial cases
  if (n == 0) {
    return last;
  }

  if (n >= flare::experimental::distance(first, last)) {
    return first;
  }

  // we cannot use here a new allocation like we do for the
  // execution space impl because for this team impl we are
  // within a parallel region, so for now we solve serially

  const std::size_t numElementsToMove =
      ::flare::experimental::distance(first + n, last);
  flare::single(flare::PerTeam(teamHandle), [=]() {
    for (std::size_t i = 0; i < numElementsToMove; ++i) {
      first[i] = std::move(first[i + n]);
    }
  });
  teamHandle.team_barrier();

  return last - n;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_SHIFT_LEFT_IMPL_H_
