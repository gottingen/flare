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

#ifndef FLARE_ALGORITHM_REDUCE_H_
#define FLARE_ALGORITHM_REDUCE_H_

#include <flare/algorithm/reduce_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set 1
//
template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const ExecutionSpace& ex,
                                         IteratorType first,
                                         IteratorType last) {
  return detail::reduce_default_functors_impl(
      "flare::reduce_default_functors_iterator_api", ex, first, last,
      typename IteratorType::value_type());
}

template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const std::string& label,
                                         const ExecutionSpace& ex,
                                         IteratorType first,
                                         IteratorType last) {
  return detail::reduce_default_functors_impl(
      label, ex, first, last, typename IteratorType::value_type());
}

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const ExecutionSpace& ex,
            const ::flare::Tensor<DataType, Properties...>& tensor) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  using tensor_type  = ::flare::Tensor<DataType, Properties...>;
  using value_type = typename tensor_type::value_type;

  return detail::reduce_default_functors_impl(
      "flare::reduce_default_functors_tensor_api", ex, KE::cbegin(tensor),
      KE::cend(tensor), value_type());
}

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const std::string& label, const ExecutionSpace& ex,
            const ::flare::Tensor<DataType, Properties...>& tensor) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  using tensor_type  = ::flare::Tensor<DataType, Properties...>;
  using value_type = typename tensor_type::value_type;

  return detail::reduce_default_functors_impl(label, ex, KE::cbegin(tensor),
                                            KE::cend(tensor), value_type());
}

//
// overload set2:
//
template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last, ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::reduce_default_functors_impl(
      "flare::reduce_default_functors_iterator_api", ex, first, last,
      init_reduction_value);
}

template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::reduce_default_functors_impl(label, ex, first, last,
                                            init_reduction_value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
ValueType reduce(const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType, Properties...>& tensor,
                 ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  return detail::reduce_default_functors_impl(
      "flare::reduce_default_functors_tensor_api", ex, KE::cbegin(tensor),
      KE::cend(tensor), init_reduction_value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType, Properties...>& tensor,
                 ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  return detail::reduce_default_functors_impl(
      label, ex, KE::cbegin(tensor), KE::cend(tensor), init_reduction_value);
}

//
// overload set 3
//
template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryOp>
ValueType reduce(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last, ValueType init_reduction_value,
                 BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::reduce_custom_functors_impl(
      "flare::reduce_default_functors_iterator_api", ex, first, last,
      init_reduction_value, joiner);
}

template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::reduce_custom_functors_impl(label, ex, first, last,
                                           init_reduction_value, joiner);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryOp>
ValueType reduce(const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType, Properties...>& tensor,
                 ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  return detail::reduce_custom_functors_impl(
      "flare::reduce_custom_functors_tensor_api", ex, KE::cbegin(tensor),
      KE::cend(tensor), init_reduction_value, joiner);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType, Properties...>& tensor,
                 ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  return detail::reduce_custom_functors_impl(label, ex, KE::cbegin(tensor),
                                           KE::cend(tensor), init_reduction_value,
                                           joiner);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REDUCE_H_
