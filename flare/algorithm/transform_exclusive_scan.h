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

#ifndef FLARE_ALGORITHM_TRANSFORM_EXCLUSIVE_SCAN_H_
#define FLARE_ALGORITHM_TRANSFORM_EXCLUSIVE_SCAN_H_

#include <flare/algorithm/transform_exclusive_scan_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType,
          class UnaryOpType>
std::enable_if_t<::flare::experimental::detail::are_iterators<
                     InputIteratorType, OutputIteratorType>::value,
                 OutputIteratorType>
transform_exclusive_scan(const ExecutionSpace& ex, InputIteratorType first,
                         InputIteratorType last, OutputIteratorType first_dest,
                         ValueType init_value, BinaryOpType binary_op,
                         UnaryOpType unary_op) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  return detail::transform_exclusive_scan_impl(
      "flare::transform_exclusive_scan_custom_functors_iterator_api", ex,
      first, last, first_dest, init_value, binary_op, unary_op);
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType,
          class UnaryOpType>
std::enable_if_t<::flare::experimental::detail::are_iterators<
                     InputIteratorType, OutputIteratorType>::value,
                 OutputIteratorType>
transform_exclusive_scan(const std::string& label, const ExecutionSpace& ex,
                         InputIteratorType first, InputIteratorType last,
                         OutputIteratorType first_dest, ValueType init_value,
                         BinaryOpType binary_op, UnaryOpType unary_op) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  return detail::transform_exclusive_scan_impl(label, ex, first, last, first_dest,
                                             init_value, binary_op, unary_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(
    const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& view_from,
    const ::flare::View<DataType2, Properties2...>& view_dest,
    ValueType init_value, BinaryOpType binary_op, UnaryOpType unary_op) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  namespace KE = ::flare::experimental;
  return detail::transform_exclusive_scan_impl(
      "flare::transform_exclusive_scan_custom_functors_view_api", ex,
      KE::cbegin(view_from), KE::cend(view_from), KE::begin(view_dest),
      init_value, binary_op, unary_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& view_from,
    const ::flare::View<DataType2, Properties2...>& view_dest,
    ValueType init_value, BinaryOpType binary_op, UnaryOpType unary_op) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  namespace KE = ::flare::experimental;
  return detail::transform_exclusive_scan_impl(
      label, ex, KE::cbegin(view_from), KE::cend(view_from),
      KE::begin(view_dest), init_value, binary_op, unary_op);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_TRANSFORM_EXCLUSIVE_SCAN_H_
