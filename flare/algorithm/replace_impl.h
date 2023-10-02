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

#ifndef FLARE_ALGORITHM_REPLACE_IMPL_H_
#define FLARE_ALGORITHM_REPLACE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class InputIterator, class ValueType>
struct StdReplaceFunctor {
  using index_type = typename InputIterator::difference_type;
  InputIterator m_first;
  ValueType m_old_value;
  ValueType m_new_value;

  FLARE_FUNCTION
  void operator()(index_type i) const {
    if (m_first[i] == m_old_value) {
      m_first[i] = m_new_value;
    }
  }

  FLARE_FUNCTION
  StdReplaceFunctor(InputIterator first, ValueType old_value,
                    ValueType new_value)
      : m_first(std::move(first)),
        m_old_value(std::move(old_value)),
        m_new_value(std::move(new_value)) {}
};

template <class ExecutionSpace, class IteratorType, class ValueType>
void replace_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                           IteratorType first, IteratorType last,
                           const ValueType& old_value,
                           const ValueType& new_value) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         StdReplaceFunctor(first, old_value, new_value));
  ex.fence("flare::replace: fence after operation");
}

template <class TeamHandleType, class IteratorType, class ValueType>
FLARE_FUNCTION void replace_team_impl(const TeamHandleType& teamHandle,
                                       IteratorType first, IteratorType last,
                                       const ValueType& old_value,
                                       const ValueType& new_value) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(TeamThreadRange(teamHandle, 0, num_elements),
                         StdReplaceFunctor(first, old_value, new_value));
  teamHandle.team_barrier();
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REPLACE_IMPL_H_
