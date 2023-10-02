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

#ifndef FLARE_ALGORITHM_TRANSFORM_INCLUSIVE_SCAN_IMPL_H_
#define FLARE_ALGORITHM_TRANSFORM_INCLUSIVE_SCAN_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/value_wrapper_for_no_neutral_element_impl.h>
#include <flare/algorithm/identity_reference_unary_functor_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class ExeSpace, class IndexType, class ValueType, class FirstFrom,
          class FirstDest, class BinaryOpType, class UnaryOpType>
struct TransformInclusiveScanNoInitValueFunctor {
  using execution_space = ExeSpace;
  using value_type      = ValueWrapperForNoNeutralElement<ValueType>;

  FirstFrom m_first_from;
  FirstDest m_first_dest;
  BinaryOpType m_binary_op;
  UnaryOpType m_unary_op;

  FLARE_FUNCTION
  TransformInclusiveScanNoInitValueFunctor(FirstFrom first_from,
                                           FirstDest first_dest,
                                           BinaryOpType bop, UnaryOpType uop)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_binary_op(std::move(bop)),
        m_unary_op(std::move(uop)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, value_type& update,
                  const bool final_pass) const {
    const auto tmp = value_type{m_unary_op(m_first_from[i]), false};
    this->join(update, tmp);
    if (final_pass) {
      m_first_dest[i] = update.val;
    }
  }

  FLARE_FUNCTION
  void init(value_type& update) const {
    update.val        = {};
    update.is_initial = true;
  }

  FLARE_FUNCTION
  void join(value_type& update, const value_type& input) const {
    if (input.is_initial) return;

    if (update.is_initial) {
      update.val = input.val;
    } else {
      update.val = m_binary_op(update.val, input.val);
    }
    update.is_initial = false;
  }
};

template <class ExeSpace, class IndexType, class ValueType, class FirstFrom,
          class FirstDest, class BinaryOpType, class UnaryOpType>
struct TransformInclusiveScanWithInitValueFunctor {
  using execution_space = ExeSpace;
  using value_type      = ValueWrapperForNoNeutralElement<ValueType>;

  FirstFrom m_first_from;
  FirstDest m_first_dest;
  BinaryOpType m_binary_op;
  UnaryOpType m_unary_op;
  ValueType m_init;

  FLARE_FUNCTION
  TransformInclusiveScanWithInitValueFunctor(FirstFrom first_from,
                                             FirstDest first_dest,
                                             BinaryOpType bop, UnaryOpType uop,
                                             ValueType init)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_binary_op(std::move(bop)),
        m_unary_op(std::move(uop)),
        m_init(std::move(init)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, value_type& update,
                  const bool final_pass) const {
    const auto tmp = value_type{m_unary_op(m_first_from[i]), false};
    this->join(update, tmp);

    if (final_pass) {
      m_first_dest[i] = m_binary_op(update.val, m_init);
    }
  }

  FLARE_FUNCTION
  void init(value_type& update) const {
    update.val        = {};
    update.is_initial = true;
  }

  FLARE_FUNCTION
  void join(value_type& update, const value_type& input) const {
    if (input.is_initial) return;

    if (update.is_initial) {
      update.val = input.val;
    } else {
      update.val = m_binary_op(update.val, input.val);
    }
    update.is_initial = false;
  }
};

// -------------------------------------------------------------
// transform_inclusive_scan_impl without init_value
// -------------------------------------------------------------
template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOpType, class UnaryOpType>
OutputIteratorType transform_inclusive_scan_impl(const std::string& label,
                                                 const ExecutionSpace& ex,
                                                 InputIteratorType first_from,
                                                 InputIteratorType last_from,
                                                 OutputIteratorType first_dest,
                                                 BinaryOpType binary_op,
                                                 UnaryOpType unary_op) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  detail::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  detail::expect_valid_range(first_from, last_from);

  // aliases
  using index_type = typename InputIteratorType::difference_type;
  using value_type =
      std::remove_const_t<typename InputIteratorType::value_type>;
  using func_type = TransformInclusiveScanNoInitValueFunctor<
      ExecutionSpace, index_type, value_type, InputIteratorType,
      OutputIteratorType, BinaryOpType, UnaryOpType>;

  // run
  const auto num_elements =
      flare::experimental::distance(first_from, last_from);
  ::flare::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(first_from, first_dest, binary_op, unary_op));
  ex.fence("flare::transform_inclusive_scan: fence after operation");

  // return
  return first_dest + num_elements;
}

// -------------------------------------------------------------
// transform_inclusive_scan_impl with init_value
// -------------------------------------------------------------
template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOpType, class UnaryOpType,
          class ValueType>
OutputIteratorType transform_inclusive_scan_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIteratorType first_from, InputIteratorType last_from,
    OutputIteratorType first_dest, BinaryOpType binary_op, UnaryOpType unary_op,
    ValueType init_value) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  detail::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  detail::expect_valid_range(first_from, last_from);

  // aliases
  using index_type = typename InputIteratorType::difference_type;
  using func_type  = TransformInclusiveScanWithInitValueFunctor<
      ExecutionSpace, index_type, ValueType, InputIteratorType,
      OutputIteratorType, BinaryOpType, UnaryOpType>;

  // run
  const auto num_elements =
      flare::experimental::distance(first_from, last_from);
  ::flare::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(first_from, first_dest, binary_op, unary_op, init_value));
  ex.fence("flare::transform_inclusive_scan: fence after operation");

  // return
  return first_dest + num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_TRANSFORM_INCLUSIVE_SCAN_IMPL_H_
