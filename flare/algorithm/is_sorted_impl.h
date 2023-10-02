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

#ifndef FLARE_ALGORITHM_IS_SORTED_IMPL_H_
#define FLARE_ALGORITHM_IS_SORTED_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class ComparatorType>
struct StdIsSortedFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  ComparatorType m_comparator;

  FLARE_FUNCTION
  void operator()(const index_type i, std::size_t& update) const {
    const auto& val_i   = m_first[i];
    const auto& val_ip1 = m_first[i + 1];

    if (m_comparator(val_ip1, val_i)) {
      ++update;
    }
  }

  FLARE_FUNCTION
  StdIsSortedFunctor(IteratorType _first1, ComparatorType comparator)
      : m_first(std::move(_first1)), m_comparator(std::move(comparator)) {}
};

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted_impl(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last,
                    ComparatorType comp) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  const auto num_elements = flare::experimental::distance(first, last);
  if (num_elements <= 1) {
    return true;
  }

  // use num_elements-1 because each index handles i and i+1
  const auto num_elements_minus_one = num_elements - 1;
  using functor_type = StdIsSortedFunctor<IteratorType, ComparatorType>;

  // result is incremented by one if sorting breaks at index i
  std::size_t result = 0;
  ::flare::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_minus_one),
      functor_type(first, std::move(comp)), result);

  return result == 0;
}

template <class ExecutionSpace, class IteratorType>
bool is_sorted_impl(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last) {
  using value_type = typename IteratorType::value_type;
  using pred_t     = detail::StdAlgoLessThanBinaryPredicate<value_type>;
  return is_sorted_impl(label, ex, first, last, pred_t());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_IS_SORTED_IMPL_H_
