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

#ifndef FLARE_ALGORITHM_TRANSFORM_REDUCE_IMPL_H_
#define FLARE_ALGORITHM_TRANSFORM_REDUCE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class ValueType>
struct StdTranformReduceDefaultBinaryTransformFunctor {
  FLARE_FUNCTION
  constexpr ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a * b);
  }
};

template <class ValueType>
struct StdTranformReduceDefaultJoinFunctor {
  FLARE_FUNCTION
  constexpr ValueType operator()(const ValueType& a, const ValueType& b) const {
    return a + b;
  }
};

template <class IteratorType, class ReducerType, class TransformType>
struct StdTransformReduceSingleIntervalFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename IteratorType::difference_type;

  const IteratorType m_first;
  const ReducerType m_reducer;
  const TransformType m_transform;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    auto tmp_wrapped_value = red_value_type{m_transform(m_first[i]), false};
    if (red_value.is_initial) {
      red_value = tmp_wrapped_value;
    } else {
      m_reducer.join(red_value, tmp_wrapped_value);
    }
  }

  FLARE_FUNCTION
  StdTransformReduceSingleIntervalFunctor(IteratorType first,
                                          ReducerType reducer,
                                          TransformType transform)
      : m_first(std::move(first)),
        m_reducer(std::move(reducer)),
        m_transform(std::move(transform)) {}
};

template <class IndexType, class IteratorType1, class IteratorType2,
          class ReducerType, class TransformType>
struct StdTransformReduceTwoIntervalsFunctor {
  using red_value_type = typename ReducerType::value_type;

  const IteratorType1 m_first1;
  const IteratorType2 m_first2;
  const ReducerType m_reducer;
  const TransformType m_transform;

  FLARE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    auto tmp_wrapped_value =
        red_value_type{m_transform(m_first1[i], m_first2[i]), false};

    if (red_value.is_initial) {
      red_value = tmp_wrapped_value;
    } else {
      m_reducer.join(red_value, tmp_wrapped_value);
    }
  }

  FLARE_FUNCTION
  StdTransformReduceTwoIntervalsFunctor(IteratorType1 first1,
                                        IteratorType2 first2,
                                        ReducerType reducer,
                                        TransformType transform)
      : m_first1(std::move(first1)),
        m_first2(std::move(first2)),
        m_reducer(std::move(reducer)),
        m_transform(std::move(transform)) {}
};

//------------------------------
//
// impl functions
//
//------------------------------

template <class ExecutionSpace, class IteratorType, class ValueType,
          class JoinerType, class UnaryTransformerType>
ValueType transform_reduce_custom_functors_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, ValueType init_reduction_value, JoinerType joiner,
    UnaryTransformerType transformer) {
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
  using functor_type =
      StdTransformReduceSingleIntervalFunctor<IteratorType, reducer_type,
                                              UnaryTransformerType>;
  using reduction_value_type = typename reducer_type::value_type;

  // run
  reduction_value_type result;
  reducer_type reducer(result, joiner);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            functor_type(first, reducer, transformer), reducer);

  // fence not needed since reducing into scalar

  // as per standard, transform is not applied to the init value
  // https://en.cppreference.com/w/cpp/algorithm/transform_reduce
  return joiner(result.val, init_reduction_value);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType, class JoinerType, class BinaryTransformerType>
ValueType transform_reduce_custom_functors_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, ValueType init_reduction_value,
    JoinerType joiner, BinaryTransformerType transformer) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first1, first2);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2);
  detail::expect_valid_range(first1, last1);

  if (first1 == last1) {
    // init is returned, unmodified
    return init_reduction_value;
  }

  // aliases
  using index_type = typename IteratorType1::difference_type;
  using reducer_type =
      ReducerWithArbitraryJoinerNoNeutralElement<ValueType, JoinerType>;
  using functor_type =
      StdTransformReduceTwoIntervalsFunctor<index_type, IteratorType1,
                                            IteratorType2, reducer_type,
                                            BinaryTransformerType>;
  using reduction_value_type = typename reducer_type::value_type;

  // run
  reduction_value_type result;
  reducer_type reducer(result, joiner);

  const auto num_elements = flare::experimental::distance(first1, last1);
  ::flare::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      functor_type(first1, first2, reducer, transformer), reducer);

  // fence not needed since reducing into scalar
  return joiner(result.val, init_reduction_value);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType>
ValueType transform_reduce_default_functors_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, ValueType init_reduction_value) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first1, first2);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2);
  detail::expect_valid_range(first1, last1);

  // aliases
  using transformer_type =
      detail::StdTranformReduceDefaultBinaryTransformFunctor<ValueType>;
  using joiner_type = detail::StdTranformReduceDefaultJoinFunctor<ValueType>;

  return transform_reduce_custom_functors_impl(
      label, ex, first1, last1, first2, std::move(init_reduction_value),
      joiner_type(), transformer_type());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_TRANSFORM_REDUCE_IMPL_H_
