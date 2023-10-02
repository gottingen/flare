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

#ifndef FLARE_ALGORITHM_MOVE_BACKWARD_IMPL_H_
#define FLARE_ALGORITHM_MOVE_BACKWARD_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType1, class IteratorType2>
struct StdMoveBackwardFunctor {
  using index_type = typename IteratorType1::difference_type;
  static_assert(std::is_signed<index_type>::value,
                "flare: StdMoveBackwardFunctor requires signed index type");

  IteratorType1 m_last;
  IteratorType2 m_dest_last;

  FLARE_FUNCTION
  void operator()(index_type i) const {
    m_dest_last[-i - 1] = std::move(m_last[-i - 1]);
  }

  FLARE_FUNCTION
  StdMoveBackwardFunctor(IteratorType1 _last, IteratorType2 _dest_last)
      : m_last(std::move(_last)), m_dest_last(std::move(_dest_last)) {}
};

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType2 move_backward_exespace_impl(const std::string& label,
                                          const ExecutionSpace& ex,
                                          IteratorType1 first,
                                          IteratorType1 last,
                                          IteratorType2 d_last) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first, d_last);
  detail::static_assert_iterators_have_matching_difference_type(first, d_last);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         StdMoveBackwardFunctor(last, d_last));
  ex.fence("flare::move_backward: fence after operation");

  // return
  return d_last - num_elements;
}

template <class TeamHandleType, class IteratorType1, class IteratorType2>
FLARE_FUNCTION IteratorType2
move_backward_team_impl(const TeamHandleType& teamHandle, IteratorType1 first,
                        IteratorType1 last, IteratorType2 d_last) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first, d_last);
  detail::static_assert_iterators_have_matching_difference_type(first, d_last);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(TeamThreadRange(teamHandle, 0, num_elements),
                         StdMoveBackwardFunctor(last, d_last));
  teamHandle.team_barrier();

  // return
  return d_last - num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_MOVE_BACKWARD_IMPL_H_
