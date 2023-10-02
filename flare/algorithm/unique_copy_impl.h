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

#ifndef FLARE_ALGORITHM_UNIQUE_COPY_IMPL_H_
#define FLARE_ALGORITHM_UNIQUE_COPY_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/copy_n_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class InputIt, class OutputIt,
          class BinaryPredicateType>
struct StdUniqueCopyFunctor {
  InputIt m_first_from;
  InputIt m_last_from;
  OutputIt m_first_dest;
  BinaryPredicateType m_pred;

  FLARE_FUNCTION
  StdUniqueCopyFunctor(InputIt first_from, InputIt last_from,
                       OutputIt first_dest, BinaryPredicateType pred)
      : m_first_from(std::move(first_from)),
        m_last_from(std::move(last_from)),
        m_first_dest(std::move(first_dest)),
        m_pred(std::move(pred)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, IndexType& update,
                  const bool final_pass) const {
    const auto& val_i   = m_first_from[i];
    const auto& val_ip1 = m_first_from[i + 1];

    if (final_pass) {
      if (!m_pred(val_i, val_ip1)) {
        m_first_dest[update] = val_i;
      }
    }

    if (!m_pred(val_i, val_ip1)) {
      update += 1;
    }
  }
};

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class PredicateType>
OutputIterator unique_copy_impl(const std::string& label,
                                const ExecutionSpace& ex, InputIterator first,
                                InputIterator last, OutputIterator d_first,
                                PredicateType pred) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first, d_first);
  detail::expect_valid_range(first, last);

  // branch for trivial vs non trivial case
  const auto num_elements = flare::experimental::distance(first, last);
  if (num_elements == 0) {
    return d_first;
  } else if (num_elements == 1) {
    return detail::copy_impl("flare::copy_from_unique_copy", ex, first, last,
                           d_first);
  } else {
    // aliases
    using index_type = typename InputIterator::difference_type;
    using func_type  = StdUniqueCopyFunctor<index_type, InputIterator,
                                           OutputIterator, PredicateType>;

    // note here that we run scan for num_elements - 1
    // because of the way we implement this, the last element is always needed.
    // We avoid performing checks inside functor that we are within limits
    // and run a "safe" scan and then copy the last element.
    const auto scan_size = num_elements - 1;
    index_type count     = 0;
    ::flare::parallel_scan(label,
                            RangePolicy<ExecutionSpace>(ex, 0, scan_size),
                            func_type(first, last, d_first, pred), count);

    return detail::copy_impl("flare::copy_from_unique_copy", ex,
                           first + scan_size, last, d_first + count);
  }
}

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator unique_copy_impl(const std::string& label,
                                const ExecutionSpace& ex, InputIterator first,
                                InputIterator last, OutputIterator d_first) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first, d_first);
  detail::expect_valid_range(first, last);

  // aliases
  using value_type1 = typename InputIterator::value_type;
  using value_type2 = typename OutputIterator::value_type;

  // default binary predicate uses ==
  using binary_pred_t = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;

  // run
  return unique_copy_impl(label, ex, first, last, d_first, binary_pred_t());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_UNIQUE_COPY_IMPL_H_
