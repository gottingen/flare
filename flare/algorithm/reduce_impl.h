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

#ifndef FLARE_ALGORITHM_REDUCE_IMPL_H_
#define FLARE_ALGORITHM_REDUCE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/reducer_with_arbitrary_joiner_no_neutral_element_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class ValueType>
struct StdReduceDefaultFunctor {
  using index_type = typename IteratorType::difference_type;

  const IteratorType m_first;

  FLARE_FUNCTION
  void operator()(const index_type i, ValueType& update) const {
    update += m_first[i];
  }
};

template <class ValueType>
struct StdReduceDefaultJoinFunctor {
  FLARE_FUNCTION
  constexpr ValueType operator()(const ValueType& a, const ValueType& b) const {
    return a + b;
  }
};

template <class IteratorType, class ReducerType>
struct StdReduceFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename IteratorType::difference_type;

  const IteratorType m_first;
  const ReducerType m_reducer;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    auto tmp_wrapped_value = red_value_type{m_first[i], false};

    if (red_value.is_initial) {
      red_value = tmp_wrapped_value;
    } else {
      m_reducer.join(red_value, tmp_wrapped_value);
    }
  }

  FLARE_FUNCTION
  StdReduceFunctor(IteratorType first, ReducerType reducer)
      : m_first(std::move(first)), m_reducer(std::move(reducer)) {}
};

//------------------------------
// reduce_custom_functors_impl
//------------------------------
template <class ExecutionSpace, class IteratorType, class ValueType,
          class JoinerType>
ValueType reduce_custom_functors_impl(const std::string& label,
                                      const ExecutionSpace& ex,
                                      IteratorType first, IteratorType last,
                                      ValueType init_reduction_value,
                                      JoinerType joiner) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    // init is returned, unmodified
    return init_reduction_value;
  }

  // aliases
  using reducer_type =
      ReducerWithArbitraryJoinerNoNeutralElement<ValueType, JoinerType>;
  using functor_type         = StdReduceFunctor<IteratorType, reducer_type>;
  using reduction_value_type = typename reducer_type::value_type;

  // run
  reduction_value_type result;
  reducer_type reducer(result, joiner);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            functor_type(first, reducer), reducer);

  // fence not needed since reducing into scalar
  return joiner(result.val, init_reduction_value);
}

template <typename ValueType>
using has_reduction_identity_sum_t =
    decltype(flare::reduction_identity<ValueType>::sum());

template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce_default_functors_impl(const std::string& label,
                                       const ExecutionSpace& ex,
                                       IteratorType first, IteratorType last,
                                       ValueType init_reduction_value) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  using value_type = flare::detail::remove_cvref_t<ValueType>;

  if (::flare::is_detected<has_reduction_identity_sum_t, value_type>::value) {
    if (first == last) {
      // init is returned, unmodified
      return init_reduction_value;
    }

    using functor_type =
        detail::StdReduceDefaultFunctor<IteratorType, value_type>;

    // run
    value_type tmp;
    const auto num_elements = flare::experimental::distance(first, last);
    ::flare::parallel_reduce(label,
                              RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                              functor_type{first}, tmp);
    // fence not needed since reducing into scalar
    tmp += init_reduction_value;
    return tmp;
  } else {
    using joiner_type = detail::StdReduceDefaultJoinFunctor<value_type>;
    return reduce_custom_functors_impl(
        label, ex, first, last, std::move(init_reduction_value), joiner_type());
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REDUCE_IMPL_H_
