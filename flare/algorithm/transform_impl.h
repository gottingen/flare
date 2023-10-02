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

#ifndef FLARE_ALGORITHM_TRANSFORM_IMPL_H_
#define FLARE_ALGORITHM_TRANSFORM_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class InputIterator, class OutputIterator,
          class UnaryFunctorType>
struct StdTransformFunctor {
  InputIterator m_first;
  OutputIterator m_d_first;
  UnaryFunctorType m_unary_op;

  FLARE_FUNCTION
  void operator()(IndexType i) const { m_d_first[i] = m_unary_op(m_first[i]); }

  FLARE_FUNCTION
  StdTransformFunctor(InputIterator _first, OutputIterator _m_d_first,
                      UnaryFunctorType _functor)
      : m_first(std::move(_first)),
        m_d_first(std::move(_m_d_first)),
        m_unary_op(std::move(_functor)) {}
};

template <class IndexType, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryFunctorType>
struct StdTransformBinaryFunctor {
  InputIterator1 m_first1;
  InputIterator2 m_first2;
  OutputIterator m_d_first;
  BinaryFunctorType m_binary_op;

  FLARE_FUNCTION
  void operator()(IndexType i) const {
    m_d_first[i] = m_binary_op(m_first1[i], m_first2[i]);
  }

  FLARE_FUNCTION
  StdTransformBinaryFunctor(InputIterator1 _first1, InputIterator2 _first2,
                            OutputIterator _m_d_first,
                            BinaryFunctorType _functor)
      : m_first1(std::move(_first1)),
        m_first2(std::move(_first2)),
        m_d_first(std::move(_m_d_first)),
        m_binary_op(std::move(_functor)) {}
};

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class UnaryOperation>
OutputIterator transform_impl(const std::string& label,
                              const ExecutionSpace& ex, InputIterator first1,
                              InputIterator last1, OutputIterator d_first,
                              UnaryOperation unary_op) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first1, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first1, d_first);
  detail::expect_valid_range(first1, last1);

  // aliases
  using index_type = typename InputIterator::difference_type;
  using func_t = StdTransformFunctor<index_type, InputIterator, OutputIterator,
                                     UnaryOperation>;

  // run
  const auto num_elements = flare::experimental::distance(first1, last1);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         func_t(first1, d_first, unary_op));
  ex.fence("flare::transform: fence after operation");

  // return
  return d_first + num_elements;
}

template <class ExecutionSpace, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryOperation>
OutputIterator transform_impl(const std::string& label,
                              const ExecutionSpace& ex, InputIterator1 first1,
                              InputIterator1 last1, InputIterator2 first2,
                              OutputIterator d_first,
                              BinaryOperation binary_op) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first1, first2, d_first);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2,
                                                              d_first);
  detail::expect_valid_range(first1, last1);

  // aliases
  using index_type = typename InputIterator1::difference_type;
  using func_t =
      StdTransformBinaryFunctor<index_type, InputIterator1, InputIterator2,
                                OutputIterator, BinaryOperation>;

  // run
  const auto num_elements = flare::experimental::distance(first1, last1);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         func_t(first1, first2, d_first, binary_op));
  ex.fence("flare::transform: fence after operation");
  return d_first + num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_TRANSFORM_IMPL_H_
