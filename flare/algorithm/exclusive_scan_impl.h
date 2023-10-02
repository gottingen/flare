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

#ifndef FLARE_ALGORITHM_EXCLUSIVE_SCAN_IMPL_H_
#define FLARE_ALGORITHM_EXCLUSIVE_SCAN_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/value_wrapper_for_no_neutral_element_impl.h>
#include <flare/algorithm/identity_reference_unary_functor_impl.h>
#include <flare/algorithm/transform_exclusive_scan.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class ExeSpace, class IndexType, class ValueType, class FirstFrom,
          class FirstDest>
struct ExclusiveScanDefaultFunctorForKnownNeutralElement {
  using execution_space = ExeSpace;

  ValueType m_init_value;
  FirstFrom m_first_from;
  FirstDest m_first_dest;

  FLARE_FUNCTION
  ExclusiveScanDefaultFunctorForKnownNeutralElement(ValueType init,
                                                    FirstFrom first_from,
                                                    FirstDest first_dest)
      : m_init_value(std::move(init)),
        m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, ValueType& update,
                  const bool final_pass) const {
    if (final_pass) m_first_dest[i] = update + m_init_value;
    update += m_first_from[i];
  }
};

template <class ExeSpace, class IndexType, class ValueType, class FirstFrom,
          class FirstDest>
struct ExclusiveScanDefaultFunctor {
  using execution_space = ExeSpace;
  using value_type =
      ::flare::experimental::detail::ValueWrapperForNoNeutralElement<ValueType>;

  ValueType m_init_value;
  FirstFrom m_first_from;
  FirstDest m_first_dest;

  FLARE_FUNCTION
  ExclusiveScanDefaultFunctor(ValueType init, FirstFrom first_from,
                              FirstDest first_dest)
      : m_init_value(std::move(init)),
        m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, value_type& update,
                  const bool final_pass) const {
    if (final_pass) {
      if (i == 0) {
        m_first_dest[i] = m_init_value;
      } else {
        m_first_dest[i] = update.val + m_init_value;
      }
    }

    const auto tmp = value_type{m_first_from[i], false};
    this->join(update, tmp);
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
      update.val        = input.val;
      update.is_initial = false;
    } else {
      update.val = update.val + input.val;
    }
  }
};

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType>
OutputIteratorType exclusive_scan_custom_op_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIteratorType first_from, InputIteratorType last_from,
    OutputIteratorType first_dest, ValueType init_value, BinaryOpType bop) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  detail::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  detail::expect_valid_range(first_from, last_from);

  // aliases
  using index_type    = typename InputIteratorType::difference_type;
  using unary_op_type = StdNumericScanIdentityReferenceUnaryFunctor<ValueType>;
  using func_type =
      TransformExclusiveScanFunctor<ExecutionSpace, index_type, ValueType,
                                    InputIteratorType, OutputIteratorType,
                                    BinaryOpType, unary_op_type>;

  // run
  const auto num_elements =
      flare::experimental::distance(first_from, last_from);
  ::flare::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(init_value, first_from, first_dest, bop, unary_op_type()));
  ex.fence("flare::exclusive_scan_custom_op: fence after operation");

  // return
  return first_dest + num_elements;
}

template <typename ValueType>
using ex_scan_has_reduction_identity_sum_t =
    decltype(flare::reduction_identity<ValueType>::sum());

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType>
OutputIteratorType exclusive_scan_default_op_impl(const std::string& label,
                                                  const ExecutionSpace& ex,
                                                  InputIteratorType first_from,
                                                  InputIteratorType last_from,
                                                  OutputIteratorType first_dest,
                                                  ValueType init_value) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  detail::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  detail::expect_valid_range(first_from, last_from);

  // does it make sense to do this static_assert too?
  // using input_iterator_value_type = typename InputIteratorType::value_type;
  // static_assert
  //   (std::is_convertible<std::remove_cv_t<input_iterator_value_type>,
  //   ValueType>::value,
  //    "exclusive_scan: InputIteratorType::value_type not convertible to
  //    ValueType");

  // we are unnecessarily duplicating code, but this is on purpose
  // so that we can use the default_op for OpenMPTarget.
  // Originally, I had this implemented as:
  // '''
  // using bop_type   = StdExclusiveScanDefaultJoinFunctor<ValueType>;
  // call exclusive_scan_custom_op_impl(..., bop_type());
  // '''
  // which avoids duplicating the functors, but for OpenMPTarget
  // I cannot use a custom binary op.
  // This is the same problem that occurs for reductions.

  // aliases
  using index_type = typename InputIteratorType::difference_type;
  using func_type  = std::conditional_t<
      ::flare::is_detected<ex_scan_has_reduction_identity_sum_t,
                            ValueType>::value,
      ExclusiveScanDefaultFunctorForKnownNeutralElement<
          ExecutionSpace, index_type, ValueType, InputIteratorType,
          OutputIteratorType>,
      ExclusiveScanDefaultFunctor<ExecutionSpace, index_type, ValueType,
                                  InputIteratorType, OutputIteratorType>>;

  // run
  const auto num_elements =
      flare::experimental::distance(first_from, last_from);
  ::flare::parallel_scan(label,
                          RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                          func_type(init_value, first_from, first_dest));

  ex.fence("flare::exclusive_scan_default_op: fence after operation");

  return first_dest + num_elements;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_EXCLUSIVE_SCAN_IMPL_H_
