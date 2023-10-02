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

#ifndef FLARE_ALGORITHM_IS_SORTED_UNTIL_IMPL_H_
#define FLARE_ALGORITHM_IS_SORTED_UNTIL_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <flare/algorithm/find.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class ComparatorType, class ReducerType>
struct StdIsSortedUntilFunctor {
  using index_type = typename IteratorType::difference_type;
  using value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ComparatorType m_comparator;
  ReducerType m_reducer;

  FLARE_FUNCTION
  void operator()(const index_type i, value_type& reduction_result) const {
    const auto& val_i   = m_first[i];
    const auto& val_ip1 = m_first[i + 1];
    if (m_comparator(val_ip1, val_i)) {
      m_reducer.join(reduction_result, i);
    }
  }

  FLARE_FUNCTION
  StdIsSortedUntilFunctor(IteratorType first, ComparatorType comparator,
                          ReducerType reducer)
      : m_first(std::move(first)),
        m_comparator(std::move(comparator)),
        m_reducer(std::move(reducer)) {}
};

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last, ComparatorType comp) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  const auto num_elements = flare::experimental::distance(first, last);

  // trivial case
  if (num_elements <= 1) {
    return last;
  }

  /*
    Do a par_reduce computing the *min* index that breaks the sorting.
    If such an index is found, then the range is sorted until that element.
    If no such index is found, then the range is sorted until the end.
  */
  using index_type = typename IteratorType::difference_type;
  index_type reduction_result;
  ::flare::Min<index_type> reducer(reduction_result);
  ::flare::parallel_reduce(
      label,
      // use num_elements-1 because each index handles i and i+1
      RangePolicy<ExecutionSpace>(ex, 0, num_elements - 1),
      // use CTAD
      StdIsSortedUntilFunctor(first, comp, reducer), reducer);

  /* If the reduction result is equal to the initial value,
     it means the range is sorted until the end */
  index_type reduction_result_init;
  reducer.init(reduction_result_init);
  if (reduction_result == reduction_result_init) {
    return last;
  } else {
    /* If such an index is found, then the range is sorted until there and
       we need to return an iterator past the element found so do +1 */
    return first + (reduction_result + 1);
  }
}

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last) {
  using value_type = typename IteratorType::value_type;
  using pred_t     = detail::StdAlgoLessThanBinaryPredicate<value_type>;
  return is_sorted_until_impl(label, ex, first, last, pred_t());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_IS_SORTED_UNTIL_IMPL_H_
