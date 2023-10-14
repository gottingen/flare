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

#ifndef FLARE_ALGORITHM_TRANSFORM_H_
#define FLARE_ALGORITHM_TRANSFORM_H_

#include <flare/algorithm/transform_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class UnaryOperation>
std::enable_if_t< ::flare::experimental::detail::are_iterators<
                      InputIterator, OutputIterator>::value,
                  OutputIterator>
transform(const ExecutionSpace& ex, InputIterator first1, InputIterator last1,
          OutputIterator d_first, UnaryOperation unary_op) {
  return detail::transform_impl("flare::transform_iterator_api_default", ex,
                              first1, last1, d_first, std::move(unary_op));
}

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class UnaryOperation>
std::enable_if_t< ::flare::experimental::detail::are_iterators<
                      InputIterator, OutputIterator>::value,
                  OutputIterator>
transform(const std::string& label, const ExecutionSpace& ex,
          InputIterator first1, InputIterator last1, OutputIterator d_first,
          UnaryOperation unary_op) {
  return detail::transform_impl(label, ex, first1, last1, d_first,
                              std::move(unary_op));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class UnaryOperation>
auto transform(const ExecutionSpace& ex,
               const ::flare::Tensor<DataType1, Properties1...>& source,
               ::flare::Tensor<DataType2, Properties2...>& dest,
               UnaryOperation unary_op) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::transform_impl("flare::transform_tensor_api_default", ex,
                              begin(source), end(source), begin(dest),
                              std::move(unary_op));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class UnaryOperation>
auto transform(const std::string& label, const ExecutionSpace& ex,
               const ::flare::Tensor<DataType1, Properties1...>& source,
               ::flare::Tensor<DataType2, Properties2...>& dest,
               UnaryOperation unary_op) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::transform_impl(label, ex, begin(source), end(source),
                              begin(dest), std::move(unary_op));
}

template <class ExecutionSpace, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryOperation>
std::enable_if_t< ::flare::experimental::detail::are_iterators<
                      InputIterator1, InputIterator2, OutputIterator>::value,
                  OutputIterator>
transform(const ExecutionSpace& ex, InputIterator1 first1, InputIterator1 last1,
          InputIterator2 first2, OutputIterator d_first,
          BinaryOperation binary_op) {
  return detail::transform_impl("flare::transform_iterator_api_default", ex,
                              first1, last1, first2, d_first,
                              std::move(binary_op));
}

template <class ExecutionSpace, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryOperation>
std::enable_if_t< ::flare::experimental::detail::are_iterators<
                      InputIterator1, InputIterator2, OutputIterator>::value,
                  OutputIterator>
transform(const std::string& label, const ExecutionSpace& ex,
          InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
          OutputIterator d_first, BinaryOperation binary_op) {
  return detail::transform_impl(label, ex, first1, last1, first2, d_first,
                              std::move(binary_op));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class DataType3,
          class... Properties3, class BinaryOperation>
auto transform(const ExecutionSpace& ex,
               const ::flare::Tensor<DataType1, Properties1...>& source1,
               const ::flare::Tensor<DataType2, Properties2...>& source2,
               ::flare::Tensor<DataType3, Properties3...>& dest,
               BinaryOperation binary_op) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(source2);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::transform_impl("flare::transform_tensor_api_default", ex,
                              begin(source1), end(source1), begin(source2),
                              begin(dest), std::move(binary_op));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class DataType3,
          class... Properties3, class BinaryOperation>
auto transform(const std::string& label, const ExecutionSpace& ex,
               const ::flare::Tensor<DataType1, Properties1...>& source1,
               const ::flare::Tensor<DataType2, Properties2...>& source2,
               ::flare::Tensor<DataType3, Properties3...>& dest,
               BinaryOperation binary_op) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(source2);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::transform_impl(label, ex, begin(source1), end(source1),
                              begin(source2), begin(dest),
                              std::move(binary_op));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_TRANSFORM_H_
