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

#ifndef FLARE_ALGORITHM_SWAP_RANGES_IMPL_H_
#define FLARE_ALGORITHM_SWAP_RANGES_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <flare/algorithm/swap.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class IteratorType1, class IteratorType2>
struct StdSwapRangesFunctor {
  IteratorType1 m_first1;
  IteratorType2 m_first2;

  FLARE_FUNCTION
  void operator()(IndexType i) const {
    ::flare::experimental::swap(m_first1[i], m_first2[i]);
  }

  FLARE_FUNCTION
  StdSwapRangesFunctor(IteratorType1 _first1, IteratorType2 _first2)
      : m_first1(std::move(_first1)), m_first2(std::move(_first2)) {}
};

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType2 swap_ranges_impl(const std::string& label,
                               const ExecutionSpace& ex, IteratorType1 first1,
                               IteratorType1 last1, IteratorType2 first2) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first1, first2);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2);
  detail::expect_valid_range(first1, last1);

  // aliases
  using index_type = typename IteratorType1::difference_type;
  using func_t = StdSwapRangesFunctor<index_type, IteratorType1, IteratorType2>;

  // run
  const auto num_elements_to_swap =
      flare::experimental::distance(first1, last1);
  ::flare::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_to_swap),
      func_t(first1, first2));
  ex.fence("flare::swap_ranges: fence after operation");

  // return
  return first2 + num_elements_to_swap;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_SWAP_RANGES_IMPL_H_
