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

#ifndef FLARE_ALGORITHM_IS_PARTITIONED_IMPL_H_
#define FLARE_ALGORITHM_IS_PARTITIONED_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class ReducerType, class PredicateType>
struct StdIsPartitionedFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename IteratorType::difference_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& redValue) const {
    const auto predicate_value = m_p(m_first[i]);
    constexpr index_type m_red_id_min =
        ::flare::reduction_identity<index_type>::min();
    constexpr index_type m_red_id_max =
        ::flare::reduction_identity<index_type>::max();

    // FIXME_NVHPC using a ternary operator causes problems
    red_value_type rv = {m_red_id_max, i};
    if (predicate_value) {
      rv = {i, m_red_id_min};
    }

    m_reducer.join(redValue, rv);
  }

  FLARE_FUNCTION
  StdIsPartitionedFunctor(IteratorType first, ReducerType reducer,
                          PredicateType p)
      : m_first(std::move(first)),
        m_reducer(std::move(reducer)),
        m_p(std::move(p)) {}
};

template <class ExecutionSpace, class IteratorType, class PredicateType>
bool is_partitioned_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last,
                         PredicateType pred) {
  // true if all elements in the range [first, last) that satisfy
  // the predicate "pred" appear before all elements that don't.
  // Also returns true if [first, last) is empty.
  // also true if all elements satisfy the predicate.

  // we implement it by finding:
  // - the max location where predicate is true  (max_loc_true)
  // - the min location where predicate is false (min_loc_false)
  // so the range is partitioned if max_loc_true < (min_loc_false)

  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  // trivial case
  if (first == last) {
    return true;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = StdIsPartitioned<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t =
      StdIsPartitionedFunctor<IteratorType, reducer_type, PredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer, pred), reducer);

  // fence not needed because reducing into scalar

  // decide and return
  constexpr index_type red_id_min =
      ::flare::reduction_identity<index_type>::min();
  constexpr index_type red_id_max =
      ::flare::reduction_identity<index_type>::max();

  if (red_result.max_loc_true != red_id_max &&
      red_result.min_loc_false != red_id_min) {
    // this occurs when the reduction yields nontrivial values
    return red_result.max_loc_true < red_result.min_loc_false;
  } else if (red_result.max_loc_true == red_id_max &&
             red_result.min_loc_false == 0) {
    // this occurs when all values do NOT satisfy
    // the predicate, and this corner case should also be true
    return true;
  } else if (first + red_result.max_loc_true == --last) {
    // this occurs when all values DO satisfy the predicate,
    // this corner case should also be true
    return true;
  } else {
    return false;
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_IS_PARTITIONED_IMPL_H_
