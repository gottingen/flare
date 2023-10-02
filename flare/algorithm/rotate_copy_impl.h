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

#ifndef FLARE_ALGORITHM_ROTATE_COPY_IMPL_H_
#define FLARE_ALGORITHM_ROTATE_COPY_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class InputIterator, class OutputIterator>
struct StdRotateCopyFunctor {
  using index_type = typename InputIterator::difference_type;

  InputIterator m_first;
  InputIterator m_last;
  InputIterator m_first_n;
  OutputIterator m_dest_first;

  FLARE_FUNCTION
  void operator()(index_type i) const {
    const index_type shift = m_last - m_first_n;

    if (i < shift) {
      m_dest_first[i] = m_first_n[i];
    } else {
      m_dest_first[i] = m_first[i - shift];
    }
  }

  FLARE_FUNCTION
  StdRotateCopyFunctor(InputIterator first, InputIterator last,
                       InputIterator first_n, OutputIterator dest_first)
      : m_first(std::move(first)),
        m_last(std::move(last)),
        m_first_n(std::move(first_n)),
        m_dest_first(std::move(dest_first)) {}
};

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator rotate_copy_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, InputIterator first,
    InputIterator n_first, InputIterator last, OutputIterator d_first) {
  /*
    algorithm is implemented as follows:

    first 	   n_first		last
    |		      |                  |
    o  o  o  o  o  o  o  o  o  o  o  o

    dest+0 -> first_n
    dest+1 -> first_n+1
    dest+2 -> first_n+2
    dest+3 -> first
    dest+4 -> first+1
    dest+5 -> first+2
    dest+6 -> first+3
    dest+7 -> first+4
    dest+8 -> first+5
    ...
    let shift = last - first_n;

    then we have:
    if (i < shift){
      *(dest_first + i) = *(first_n + i);
    }
    else{
      *(dest_first + i) = *(from + i - shift);
    }
  */

  // checks
  detail::static_assert_random_access_and_accessible(ex, first, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first, d_first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(first, n_first);
  detail::expect_valid_range(n_first, last);

  if (first == last) {
    return d_first;
  }

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         StdRotateCopyFunctor(first, last, n_first, d_first));

  ex.fence("flare::rotate_copy: fence after operation");

  // return
  return d_first + num_elements;
}

template <class TeamHandleType, class InputIterator, class OutputIterator>
FLARE_FUNCTION OutputIterator rotate_copy_team_impl(
    const TeamHandleType& teamHandle, InputIterator first,
    InputIterator n_first, InputIterator last, OutputIterator d_first) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first, d_first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(first, n_first);
  detail::expect_valid_range(n_first, last);

  if (first == last) {
    return d_first;
  }

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(TeamThreadRange(teamHandle, 0, num_elements),
                         StdRotateCopyFunctor(first, last, n_first, d_first));

  teamHandle.team_barrier();

  // return
  return d_first + num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ROTATE_COPY_IMPL_H_
