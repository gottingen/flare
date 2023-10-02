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

#ifndef FLARE_ALGORITHM_COPY_IF_IMPL_H_
#define FLARE_ALGORITHM_COPY_IF_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class FirstFrom, class FirstDest, class PredType>
struct StdCopyIfFunctor {
  FirstFrom m_first_from;
  FirstDest m_first_dest;
  PredType m_pred;

  FLARE_FUNCTION
  StdCopyIfFunctor(FirstFrom first_from, FirstDest first_dest, PredType pred)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_pred(std::move(pred)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, IndexType& update,
                  const bool final_pass) const {
    const auto& myval = m_first_from[i];
    if (final_pass) {
      if (m_pred(myval)) {
        m_first_dest[update] = myval;
      }
    }

    if (m_pred(myval)) {
      update += 1;
    }
  }
};

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class PredicateType>
OutputIterator copy_if_impl(const std::string& label, const ExecutionSpace& ex,
                            InputIterator first, InputIterator last,
                            OutputIterator d_first, PredicateType pred) {
  /*
    To explain the impl, suppose that our data is:

    | 1 | 1 | 2 | 2 | 3 | -2 | 4 | 4 | 4 | 5 | 7 | -10 |

    and we want to copy only the even entries,
    We can use an exclusive scan where the "update"
    is incremented only for the elements that satisfy the predicate.
    This way, the update allows us to track where in the destination
    we need to copy the elements:

    In this case, counting only the even entries, the exlusive scan
    during the final pass would yield:

    | 0 | 0 | 0 | 1 | 2 | 2 | 3 | 4 | 5 | 6 | 6 | 6 |
              *   *       *   *   *   *           *

    which provides the indexing in the destination where
    each starred (*) element needs to be copied to since
    the starred elements are those that satisfy the predicate.
   */

  // checks
  detail::static_assert_random_access_and_accessible(ex, first, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first, d_first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    return d_first;
  } else {
    // aliases
    using index_type = typename InputIterator::difference_type;
    using func_type  = StdCopyIfFunctor<index_type, InputIterator,
                                       OutputIterator, PredicateType>;

    // run
    const auto num_elements = flare::experimental::distance(first, last);
    index_type count        = 0;
    ::flare::parallel_scan(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_type(first, d_first, pred), count);

    // fence not needed because of the scan accumulating into count
    return d_first + count;
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_COPY_IF_IMPL_H_
