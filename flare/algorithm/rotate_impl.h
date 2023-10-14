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

#ifndef FLARE_ALGORITHM_ROTATE_IMPL_H_
#define FLARE_ALGORITHM_ROTATE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/reverse_impl.h>
#include <flare/algorithm/move.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class ExecutionSpace, class IteratorType>
IteratorType rotate_with_pivot_in_left_half(const std::string& label,
                                            const ExecutionSpace& ex,
                                            IteratorType first,
                                            IteratorType n_first,
                                            IteratorType last) {
  /*
    This impl is specific for when the n_first iterator points to
    an element that is before or equal to the middle of the range.

    If we have:

    | 0 | 1 | 2 | 1 | 4 | 5 | 2 | 2 | 10 | -3 | 1 | -6 | -5 | 8 | 9 | 11 | *
      ^           ^              mid					   ^
    first       n_first							  last

    In step 1, we create a temporary tensor with extent = distance(n_first, last)
    and *move* the elements from [n_first, last) to tmp tensor, such that
    tmp tensor becomes:

    | 1 | 4 | 5 | 2 | 2 | 10 | -3 | 1 | -6 | -5 | 8 | 9 | 11 |

    In step 2, we move the elements in [first, n_first)
    to the new position where they are supposed to end up.

    In step 3, we move the elements from the tmp tensor to
    the range starting at first.
   */

  namespace KE                     = ::flare::experimental;
  const auto num_elements_on_left  = KE::distance(first, n_first);
  const auto num_elements_on_right = KE::distance(n_first, last);

  // create helper tmp tensor
  using value_type    = typename IteratorType::value_type;
  using tmp_tensor_type = flare::Tensor<value_type*, ExecutionSpace>;
  tmp_tensor_type tmp_tensor("rotate_impl_for_pivot_in_left_half_impl",
                         num_elements_on_right);
  using tmp_readwrite_iterator_type = decltype(begin(tmp_tensor));

  // index_type is the same and needed in all steps
  using index_type = typename IteratorType::difference_type;

  // stage 1
  using step1_func_type =
      StdMoveFunctor<index_type, IteratorType, tmp_readwrite_iterator_type>;
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_on_right),
      step1_func_type(n_first, begin(tmp_tensor)));

  // stage 2
  using step2_func_type =
      StdMoveFunctor<index_type, IteratorType, IteratorType>;
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_on_left),
      step2_func_type(first, first + num_elements_on_right));

  // step 3
  using step3_func_type =
      StdMoveFunctor<index_type, tmp_readwrite_iterator_type, IteratorType>;
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, tmp_tensor.extent(0)),
                         step3_func_type(begin(tmp_tensor), first));

  ex.fence("flare::rotate: fence after operation");
  return first + (last - n_first);
}

template <class ExecutionSpace, class IteratorType>
IteratorType rotate_with_pivot_in_right_half(const std::string& label,
                                             const ExecutionSpace& ex,
                                             IteratorType first,
                                             IteratorType n_first,
                                             IteratorType last) {
  /*
    This impl is specific for when the n_first iterator points to
    an element that is after the middle of the range.

    If we have:

    | 0 | 1 | 2 | 1 | 4 | 5 | 2 | 2 | 10 | -3 | 1 | -6 | -5 | 8 | 9 | 11 | *
      ^                          mid            ^                          ^
    first                                    n_first			  last

    In step 1, we create a temporary tensor with extent = distance(first, n_first)
    and *move* the elements from [first, n_first) to tmp tensor,
    such that tmp tensor becomes:

    | 0 | 1 | 2 | 1 | 4 | 5 | 2 | 2 | 10 | -3 | 1 |

    In step 2, we move the elements in [n_first, last)
    to the beginning where they are supposed to end up.

    In step 3, we move the elements from the tmp tensor to
    the range starting at first.
   */

  namespace KE                     = ::flare::experimental;
  const auto num_elements_on_left  = KE::distance(first, n_first);
  const auto num_elements_on_right = KE::distance(n_first, last);

  // create helper tmp tensor
  using value_type    = typename IteratorType::value_type;
  using tmp_tensor_type = flare::Tensor<value_type*, ExecutionSpace>;
  tmp_tensor_type tmp_tensor("rotate_impl_for_pivot_in_left_half_impl",
                         num_elements_on_left);
  using tmp_readwrite_iterator_type = decltype(begin(tmp_tensor));

  // index_type is the same and needed in all steps
  using index_type = typename IteratorType::difference_type;

  // stage 1
  using step1_func_type =
      StdMoveFunctor<index_type, IteratorType, tmp_readwrite_iterator_type>;
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_on_left),
      step1_func_type(first, begin(tmp_tensor)));

  // stage 2
  using step2_func_type =
      StdMoveFunctor<index_type, IteratorType, IteratorType>;
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_on_right),
      step2_func_type(n_first, first));

  // step 3:
  using step3_func_type =
      StdMoveFunctor<index_type, tmp_readwrite_iterator_type, IteratorType>;
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, tmp_tensor.extent(0)),
      step3_func_type(begin(tmp_tensor), first + num_elements_on_right));

  ex.fence("flare::rotate: fence after operation");
  return first + (last - n_first);
}

template <class ExecutionSpace, class IteratorType>
IteratorType rotate_exespace_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType n_first, IteratorType last) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(first, n_first);
  detail::expect_valid_range(n_first, last);

  // might be worth checking if for exespace we should do
  // something similar to what we do for team since it avoids a new allocation
  namespace KE                     = ::flare::experimental;
  const auto num_elements          = KE::distance(first, last);
  const auto n_distance_from_first = KE::distance(first, n_first);
  if (n_distance_from_first <= num_elements / 2) {
    return rotate_with_pivot_in_left_half(label, ex, first, n_first, last);
  } else {
    return rotate_with_pivot_in_right_half(label, ex, first, n_first, last);
  }
}

template <class TeamHandleType, class IteratorType>
FLARE_FUNCTION IteratorType rotate_team_impl(const TeamHandleType& teamHandle,
                                              IteratorType first,
                                              IteratorType n_first,
                                              IteratorType last) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(first, n_first);
  detail::expect_valid_range(n_first, last);

  namespace KE = ::flare::experimental;

  auto result = first + (last - n_first);
  // first reverse the whole range
  KE::detail::reverse_team_impl(teamHandle, first, last);
  // re-reverse each piece
  KE::detail::reverse_team_impl(teamHandle, first, result);
  KE::detail::reverse_team_impl(teamHandle, result, last);

  // no need for barrier here since reverse already calls it

  return result;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ROTATE_IMPL_H_
