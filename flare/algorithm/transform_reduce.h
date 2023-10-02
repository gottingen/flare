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

#ifndef FLARE_ALGORITHM_TRANSFORM_REDUCE_H_
#define FLARE_ALGORITHM_TRANSFORM_REDUCE_H_

#include <flare/algorithm/transform_reduce_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

// ----------------------------
// overload set1:
// no custom functors passed, so equivalent to
// transform_reduce(first1, last1, first2, init, plus<>(), multiplies<>());
// ----------------------------
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType>
ValueType transform_reduce(const ExecutionSpace& ex, IteratorType1 first1,
                           IteratorType1 last1, IteratorType2 first2,
                           ValueType init_reduction_value) {
  return detail::transform_reduce_default_functors_impl(
      "flare::transform_reduce_default_functors_iterator_api", ex, first1,
      last1, first2, std::move(init_reduction_value));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType>
ValueType transform_reduce(const std::string& label, const ExecutionSpace& ex,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value) {
  return detail::transform_reduce_default_functors_impl(
      label, ex, first1, last1, first2, std::move(init_reduction_value));
}

// overload1 accepting views
template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType>
ValueType transform_reduce(
    const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& first_view,
    const ::flare::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(first_view);
  detail::static_assert_is_admissible_to_flare_std_algorithms(second_view);

  return detail::transform_reduce_default_functors_impl(
      "flare::transform_reduce_default_functors_iterator_api", ex,
      KE::cbegin(first_view), KE::cend(first_view), KE::cbegin(second_view),
      std::move(init_reduction_value));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType>
ValueType transform_reduce(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& first_view,
    const ::flare::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(first_view);
  detail::static_assert_is_admissible_to_flare_std_algorithms(second_view);

  return detail::transform_reduce_default_functors_impl(
      label, ex, KE::cbegin(first_view), KE::cend(first_view),
      KE::cbegin(second_view), std::move(init_reduction_value));
}

//
// overload set2:
// accepts a custom transform and joiner functor
//

// Note the std refers to the arg BinaryReductionOp
// but in the flare naming convention, it corresponds
// to a "joiner" that knows how to join two values
// NOTE: "joiner/transformer" need to be commutative.

// https://en.cppreference.com/w/cpp/algorithm/transform_reduce

// api accepting iterators
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType, class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const ExecutionSpace& ex, IteratorType1 first1,
                           IteratorType1 last1, IteratorType2 first2,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::transform_reduce_custom_functors_impl(
      "flare::transform_reduce_custom_functors_iterator_api", ex, first1,
      last1, first2, std::move(init_reduction_value), std::move(joiner),
      std::move(transformer));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType, class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const std::string& label, const ExecutionSpace& ex,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2, ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::transform_reduce_custom_functors_impl(
      label, ex, first1, last1, first2, std::move(init_reduction_value),
      std::move(joiner), std::move(transformer));
}

// accepting views
template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(
    const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& first_view,
    const ::flare::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value, BinaryJoinerType joiner,
    BinaryTransform transformer) {
  namespace KE = ::flare::experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  detail::static_assert_is_admissible_to_flare_std_algorithms(first_view);
  detail::static_assert_is_admissible_to_flare_std_algorithms(second_view);

  return detail::transform_reduce_custom_functors_impl(
      "flare::transform_reduce_custom_functors_view_api", ex,
      KE::cbegin(first_view), KE::cend(first_view), KE::cbegin(second_view),
      std::move(init_reduction_value), std::move(joiner),
      std::move(transformer));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& first_view,
    const ::flare::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value, BinaryJoinerType joiner,
    BinaryTransform transformer) {
  namespace KE = ::flare::experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  detail::static_assert_is_admissible_to_flare_std_algorithms(first_view);
  detail::static_assert_is_admissible_to_flare_std_algorithms(second_view);

  return detail::transform_reduce_custom_functors_impl(
      label, ex, KE::cbegin(first_view), KE::cend(first_view),
      KE::cbegin(second_view), std::move(init_reduction_value),
      std::move(joiner), std::move(transformer));
}

//
// overload set3:
//
// accepting iterators
template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryJoinerType, class UnaryTransform>
// need this to avoid ambiguous call
std::enable_if_t<
    ::flare::experimental::detail::are_iterators<IteratorType>::value, ValueType>
transform_reduce(const ExecutionSpace& ex, IteratorType first1,
                 IteratorType last1, ValueType init_reduction_value,
                 BinaryJoinerType joiner, UnaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::transform_reduce_custom_functors_impl(
      "flare::transform_reduce_custom_functors_iterator_api", ex, first1,
      last1, std::move(init_reduction_value), std::move(joiner),
      std::move(transformer));
}

template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryJoinerType, class UnaryTransform>
// need this to avoid ambiguous call
std::enable_if_t<
    ::flare::experimental::detail::are_iterators<IteratorType>::value, ValueType>
transform_reduce(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first1, IteratorType last1,
                 ValueType init_reduction_value, BinaryJoinerType joiner,
                 UnaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return detail::transform_reduce_custom_functors_impl(
      label, ex, first1, last1, std::move(init_reduction_value),
      std::move(joiner), std::move(transformer));
}

// accepting views
template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const ExecutionSpace& ex,
                           const ::flare::View<DataType, Properties...>& view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform transformer) {
  namespace KE = ::flare::experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  return detail::transform_reduce_custom_functors_impl(
      "flare::transform_reduce_custom_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), std::move(init_reduction_value), std::move(joiner),
      std::move(transformer));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const std::string& label, const ExecutionSpace& ex,
                           const ::flare::View<DataType, Properties...>& view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform transformer) {
  namespace KE = ::flare::experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  return detail::transform_reduce_custom_functors_impl(
      label, ex, KE::cbegin(view), KE::cend(view),
      std::move(init_reduction_value), std::move(joiner),
      std::move(transformer));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_TRANSFORM_REDUCE_H_
