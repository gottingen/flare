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

#ifndef FLARE_ALGORITHM_FILL_AND_FILL_N_IMPL_H_
#define FLARE_ALGORITHM_FILL_AND_FILL_N_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class InputIterator, class T>
struct StdFillFunctor {
  using index_type = typename InputIterator::difference_type;
  InputIterator m_first;
  T m_value;

  FLARE_FUNCTION
  void operator()(index_type i) const { m_first[i] = m_value; }

  FLARE_FUNCTION
  StdFillFunctor(InputIterator _first, T _value)
      : m_first(std::move(_first)), m_value(std::move(_value)) {}
};

//
// exespace impl
//
template <class ExecutionSpace, class IteratorType, class T>
void fill_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                        IteratorType first, IteratorType last, const T& value) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         StdFillFunctor(first, value));
  ex.fence("flare::fill: fence after operation");
}

template <class ExecutionSpace, class IteratorType, class SizeType, class T>
IteratorType fill_n_exespace_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  SizeType n, const T& value) {
  auto last = first + n;
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  if (n <= 0) {
    return first;
  }

  fill_exespace_impl(label, ex, first, last, value);
  return last;
}

//
// team-level impl
//
template <class TeamHandleType, class IteratorType, class T>
FLARE_FUNCTION void fill_team_impl(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType last,
                                    const T& value) {
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(TeamThreadRange(teamHandle, 0, num_elements),
                         StdFillFunctor(first, value));

  teamHandle.team_barrier();
}

template <class TeamHandleType, class IteratorType, class SizeType, class T>
FLARE_FUNCTION IteratorType fill_n_team_impl(const TeamHandleType& teamHandle,
                                              IteratorType first, SizeType n,
                                              const T& value) {
  auto last = first + n;
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);

  if (n <= 0) {
    return first;
  }

  fill_team_impl(teamHandle, first, last, value);
  return last;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FILL_AND_FILL_N_IMPL_H_
