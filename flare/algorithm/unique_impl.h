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

#ifndef FLARE_ALGORITHM_UNIQUE_IMPL_H_
#define FLARE_ALGORITHM_UNIQUE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/move.h>
#include <flare/algorithm/distance.h>
#include <flare/algorithm/adjacent_find.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class InputIt, class OutputIt,
          class BinaryPredicateType>
struct StdUniqueFunctor {
  InputIt m_first_from;
  InputIt m_last_from;
  OutputIt m_first_dest;
  BinaryPredicateType m_pred;

  FLARE_FUNCTION
  StdUniqueFunctor(InputIt first_from, InputIt last_from, OutputIt first_dest,
                   BinaryPredicateType pred)
      : m_first_from(std::move(first_from)),
        m_last_from(std::move(last_from)),
        m_first_dest(std::move(first_dest)),
        m_pred(std::move(pred)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, IndexType& update,
                  const bool final_pass) const {
    auto& val_i         = m_first_from[i];
    const auto& val_ip1 = m_first_from[i + 1];

    if (final_pass) {
      if (!m_pred(val_i, val_ip1)) {
        m_first_dest[update] = std::move(val_i);
      }
    }

    if (!m_pred(val_i, val_ip1)) {
      update += 1;
    }
  }
};

template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType unique_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last,
                         PredicateType pred) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  const auto num_elements = flare::experimental::distance(first, last);
  if (num_elements == 0) {
    return first;
  } else if (num_elements == 1) {
    return last;
  } else {
    // ----------
    // step 1:
    // find first location of adjacent equal elements
    // ----------
    auto it_found =
        ::flare::experimental::adjacent_find(ex, first, last, pred);

    // if none, all elements are unique, so nothing to do
    if (it_found == last) {
      return last;
    } else {
      // if here, we found some equal adjacent elements,
      // so count all preceeding unique elements
      const auto num_unique_found_in_step_one = it_found - first;

      // ----------
      // step 2:
      // ----------
      // since we found some unique elements, we don't need to explore
      // the full range [first, last), but only need to focus on the
      // remaining range [it_found, last)
      const auto num_elements_to_explore = last - it_found;

      // create a tmp view to use to *move* all unique elements
      // using the same algorithm used for unique_copy but we now move things
      using value_type    = typename IteratorType::value_type;
      using tmp_view_type = flare::View<value_type*, ExecutionSpace>;
      tmp_view_type tmp_view("std_unique_tmp_view", num_elements_to_explore);

      // scan extent is: num_elements_to_explore - 1
      // for same reason as the one explained in unique_copy
      const auto scan_size = num_elements_to_explore - 1;
      auto tmp_first       = ::flare::experimental::begin(tmp_view);
      using output_it      = decltype(tmp_first);

      using index_type = typename IteratorType::difference_type;
      using func_type =
          StdUniqueFunctor<index_type, IteratorType, output_it, PredicateType>;
      index_type count = 0;
      ::flare::parallel_scan(
          label, RangePolicy<ExecutionSpace>(ex, 0, scan_size),
          func_type(it_found, last, tmp_first, pred), count);

      // move last element too, for the same reason as the unique_copy
      auto unused_r = detail::move_exespace_impl("flare::move_from_unique", ex,
                                               it_found + scan_size, last,
                                               tmp_first + count);
      (void)unused_r;  // r1 not used

      // ----------
      // step 3
      // ----------
      // move back from tmp to original range,
      // ensuring we start overwriting after the original unique found
      using tmp_readwrite_iterator_type = decltype(begin(tmp_view));
      using step3_func_t =
          StdMoveFunctor<index_type, tmp_readwrite_iterator_type, IteratorType>;

      ::flare::parallel_for(
          "unique_step3_parfor",
          RangePolicy<ExecutionSpace>(ex, 0, tmp_view.extent(0)),
          step3_func_t(begin(tmp_view),
                       (first + num_unique_found_in_step_one)));

      ex.fence("flare::unique: fence after operation");

      // return iterator to one passed the last written
      // (the +1 is needed to account for the last element, see above)
      return (first + num_unique_found_in_step_one + count + 1);
    }
  }
}

template <class ExecutionSpace, class IteratorType>
IteratorType unique_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last) {
  using value_type    = typename IteratorType::value_type;
  using binary_pred_t = StdAlgoEqualBinaryPredicate<value_type>;
  return unique_impl(label, ex, first, last, binary_pred_t());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_UNIQUE_IMPL_H_
