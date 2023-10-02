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

#ifndef FLARE_ALGORITHM_REVERSE_IMPL_H_
#define FLARE_ALGORITHM_REVERSE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <flare/algorithm/swap.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class InputIterator>
struct StdReverseFunctor {
  using index_type = typename InputIterator::difference_type;
  static_assert(std::is_signed<index_type>::value,
                "flare: StdReverseFunctor requires signed index type");

  InputIterator m_first;
  InputIterator m_last;

  FLARE_FUNCTION
  void operator()(index_type i) const {
    ::flare::experimental::swap(m_first[i], m_last[-i - 1]);
  }

  FLARE_FUNCTION
  StdReverseFunctor(InputIterator first, InputIterator last)
      : m_first(std::move(first)), m_last(std::move(last)) {}
};

template <class ExecutionSpace, class InputIterator>
void reverse_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                           InputIterator first, InputIterator last) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  // run
  if (last >= first + 2) {
    // only need half
    const auto num_elements = flare::experimental::distance(first, last) / 2;
    ::flare::parallel_for(label,
                           RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                           StdReverseFunctor(first, last));
    ex.fence("flare::reverse: fence after operation");
  }
}

template <class TeamHandleType, class InputIterator>
FLARE_FUNCTION void reverse_team_impl(const TeamHandleType& teamHandle,
                                       InputIterator first,
                                       InputIterator last) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  // run
  if (last >= first + 2) {
    // only need half
    const auto num_elements = flare::experimental::distance(first, last) / 2;
    ::flare::parallel_for(TeamThreadRange(teamHandle, 0, num_elements),
                           StdReverseFunctor(first, last));
    teamHandle.team_barrier();
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REVERSE_IMPL_H_
