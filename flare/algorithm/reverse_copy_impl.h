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

#ifndef FLARE_ALGORITHM_REVERSE_COPY_IMPL_H_
#define FLARE_ALGORITHM_REVERSE_COPY_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class InputIterator, class OutputIterator>
struct StdReverseCopyFunctor {
  using index_type = typename InputIterator::difference_type;
  static_assert(std::is_signed<index_type>::value,
                "flare: StdReverseCopyFunctor requires signed index type");

  InputIterator m_last;
  OutputIterator m_dest_first;

  FLARE_FUNCTION
  void operator()(index_type i) const { m_dest_first[i] = m_last[-1 - i]; }

  FLARE_FUNCTION
  StdReverseCopyFunctor(InputIterator _last, OutputIterator _dest_first)
      : m_last(std::move(_last)), m_dest_first(std::move(_dest_first)) {}
};

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator reverse_copy_exespace_impl(const std::string& label,
                                          const ExecutionSpace& ex,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator d_first) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first, d_first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         StdReverseCopyFunctor(last, d_first));
  ex.fence("flare::reverse_copy: fence after operation");

  // return
  return d_first + num_elements;
}

template <class TeamHandleType, class InputIterator, class OutputIterator>
FLARE_FUNCTION OutputIterator
reverse_copy_team_impl(const TeamHandleType& teamHandle, InputIterator first,
                       InputIterator last, OutputIterator d_first) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first, d_first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(TeamThreadRange(teamHandle, 0, num_elements),
                         StdReverseCopyFunctor(last, d_first));
  teamHandle.team_barrier();

  // return
  return d_first + num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REVERSE_COPY_IMPL_H_
