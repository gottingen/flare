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

#ifndef FLARE_ALGORITHM_REPLACE_COPY_IMPL_H_
#define FLARE_ALGORITHM_REPLACE_COPY_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class InputIterator, class OutputIterator, class ValueType>
struct StdReplaceCopyFunctor {
  using index_type = typename InputIterator::difference_type;

  InputIterator m_first_from;
  OutputIterator m_first_dest;
  ValueType m_old_value;
  ValueType m_new_value;

  FLARE_FUNCTION
  void operator()(index_type i) const {
    const auto& myvalue_from = m_first_from[i];

    if (myvalue_from == m_old_value) {
      m_first_dest[i] = m_new_value;
    } else {
      m_first_dest[i] = myvalue_from;
    }
  }

  FLARE_FUNCTION
  StdReplaceCopyFunctor(InputIterator first_from, OutputIterator first_dest,
                        ValueType old_value, ValueType new_value)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_old_value(std::move(old_value)),
        m_new_value(std::move(new_value)) {}
};

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType>
OutputIteratorType replace_copy_exespace_impl(const std::string& label,
                                              const ExecutionSpace& ex,
                                              InputIteratorType first_from,
                                              InputIteratorType last_from,
                                              OutputIteratorType first_dest,
                                              const ValueType& old_value,
                                              const ValueType& new_value) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  detail::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  detail::expect_valid_range(first_from, last_from);

  // run
  const auto num_elements =
      flare::experimental::distance(first_from, last_from);
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      StdReplaceCopyFunctor(first_from, first_dest, old_value, new_value));
  ex.fence("flare::replace_copy: fence after operation");

  // return
  return first_dest + num_elements;
}

template <class TeamHandleType, class InputIteratorType,
          class OutputIteratorType, class ValueType>
FLARE_FUNCTION OutputIteratorType replace_copy_team_impl(
    const TeamHandleType& teamHandle, InputIteratorType first_from,
    InputIteratorType last_from, OutputIteratorType first_dest,
    const ValueType& old_value, const ValueType& new_value) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first_from,
                                                   first_dest);
  detail::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  detail::expect_valid_range(first_from, last_from);

  // run
  const auto num_elements =
      flare::experimental::distance(first_from, last_from);
  ::flare::parallel_for(
      TeamThreadRange(teamHandle, 0, num_elements),
      StdReplaceCopyFunctor(first_from, first_dest, old_value, new_value));
  teamHandle.team_barrier();

  // return
  return first_dest + num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REPLACE_COPY_IMPL_H_
