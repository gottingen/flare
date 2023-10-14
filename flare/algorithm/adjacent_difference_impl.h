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

#ifndef FLARE_ALGORITHM_ADJACENT_DIFFERENCE_IMPL_H_
#define FLARE_ALGORITHM_ADJACENT_DIFFERENCE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class ValueType1, class ValueType2, class RetType = ValueType2>
struct StdAdjacentDifferenceDefaultBinaryOpFunctor {
  FLARE_FUNCTION
  constexpr RetType operator()(const ValueType1& a, const ValueType2& b) const {
    return a - b;
  }
};

template <class InputIteratorType, class OutputIteratorType,
          class BinaryOperator>
struct StdAdjacentDiffFunctor {
  using index_type = typename InputIteratorType::difference_type;

  const InputIteratorType m_first_from;
  const OutputIteratorType m_first_dest;
  BinaryOperator m_op;

  FLARE_FUNCTION
  void operator()(const index_type i) const {
    const auto& my_value = m_first_from[i];
    if (i == 0) {
      m_first_dest[i] = my_value;
    } else {
      const auto& left_value = m_first_from[i - 1];
      m_first_dest[i]        = m_op(my_value, left_value);
    }
  }

  FLARE_FUNCTION
  StdAdjacentDiffFunctor(InputIteratorType first_from,
                         OutputIteratorType first_dest, BinaryOperator op)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_op(std::move(op)) {}
};

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOp>
OutputIteratorType adjacent_difference_impl(const std::string& label,
                                            const ExecutionSpace& ex,
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            BinaryOp bin_op) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  detail::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  detail::expect_valid_range(first_from, last_from);

  if (first_from == last_from) {
    return first_dest;
  }

  // aliases
  using value_type    = typename OutputIteratorType::value_type;
  using aux_tensor_type = ::flare::Tensor<value_type*, ExecutionSpace>;
  using functor_t =
      StdAdjacentDiffFunctor<InputIteratorType, OutputIteratorType, BinaryOp>;

  // run
  const auto num_elements =
      flare::experimental::distance(first_from, last_from);
  aux_tensor_type aux_tensor("aux_tensor", num_elements);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         functor_t(first_from, first_dest, bin_op));
  ex.fence("flare::adjacent_difference: fence after operation");

  // return
  return first_dest + num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ADJACENT_DIFFERENCE_IMPL_H_
