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

#ifndef FLARE_ALGORITHM_PARTITION_POINT_IMPL_H_
#define FLARE_ALGORITHM_PARTITION_POINT_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class ReducerType, class PredicateType>
struct StdPartitionPointFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename IteratorType::difference_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& redValue) const {
    const auto predicate_value = m_p(m_first[i]);

    // FIXME_NVHPC using a ternary operator causes problems
    red_value_type rv = {i};
    if (predicate_value) {
      rv = {::flare::reduction_identity<index_type>::min()};
    }

    m_reducer.join(redValue, rv);
  }

  FLARE_FUNCTION
  StdPartitionPointFunctor(IteratorType first, ReducerType reducer,
                           PredicateType p)
      : m_first(std::move(first)),
        m_reducer(std::move(reducer)),
        m_p(std::move(p)) {}
};

template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType partition_point_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last, PredicateType pred) {
  // locates the end of the first partition, that is, the first
  // element that does not satisfy p or last if all elements satisfy p.
  // Implementation below finds the first location where p is false.

  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    return first;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = StdPartitionPoint<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t =
      StdPartitionPointFunctor<IteratorType, reducer_type, PredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer, pred), reducer);

  // fence not needed because reducing into scalar

  // decide and return
  if (red_result.min_loc_false ==
      ::flare::reduction_identity<index_type>::min()) {
    // if all elements are true, return last
    return last;
  } else {
    return first + red_result.min_loc_false;
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_PARTITION_POINT_IMPL_H_
