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

#ifndef FLARE_ALGORITHM_ADJACENT_DIFFERENCE_H_
#define FLARE_ALGORITHM_ADJACENT_DIFFERENCE_H_

#include <flare/algorithm/adjacent_difference_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType>
std::enable_if_t<!::flare::is_tensor<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const ExecutionSpace& ex, InputIteratorType first_from,
                    InputIteratorType last_from,
                    OutputIteratorType first_dest) {
  using value_type1 = typename InputIteratorType::value_type;
  using value_type2 = typename OutputIteratorType::value_type;
  using binary_op =
      detail::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;
  return detail::adjacent_difference_impl(
      "flare::adjacent_difference_iterator_api", ex, first_from, last_from,
      first_dest, binary_op());
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOp>
std::enable_if_t<!::flare::is_tensor<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const ExecutionSpace& ex, InputIteratorType first_from,
                    InputIteratorType last_from, OutputIteratorType first_dest,
                    BinaryOp bin_op) {
  return detail::adjacent_difference_impl(
      "flare::adjacent_difference_iterator_api", ex, first_from, last_from,
      first_dest, bin_op);
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType>
std::enable_if_t<!::flare::is_tensor<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const std::string& label, const ExecutionSpace& ex,
                    InputIteratorType first_from, InputIteratorType last_from,
                    OutputIteratorType first_dest) {
  using value_type1 = typename InputIteratorType::value_type;
  using value_type2 = typename OutputIteratorType::value_type;
  using binary_op =
      detail::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;
  return detail::adjacent_difference_impl(label, ex, first_from, last_from,
                                        first_dest, binary_op());
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOp>
std::enable_if_t<!::flare::is_tensor<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const std::string& label, const ExecutionSpace& ex,
                    InputIteratorType first_from, InputIteratorType last_from,
                    OutputIteratorType first_dest, BinaryOp bin_op) {
  return detail::adjacent_difference_impl(label, ex, first_from, last_from,
                                        first_dest, bin_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto adjacent_difference(
    const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor_from,
    const ::flare::Tensor<DataType2, Properties2...>& tensor_dest) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_dest);

  using tensor_type1  = ::flare::Tensor<DataType1, Properties1...>;
  using tensor_type2  = ::flare::Tensor<DataType2, Properties2...>;
  using value_type1 = typename tensor_type1::value_type;
  using value_type2 = typename tensor_type2::value_type;
  using binary_op =
      detail::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;
  return detail::adjacent_difference_impl(
      "flare::adjacent_difference_tensor_api", ex, KE::cbegin(tensor_from),
      KE::cend(tensor_from), KE::begin(tensor_dest), binary_op());
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryOp>
auto adjacent_difference(
    const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor_from,
    const ::flare::Tensor<DataType2, Properties2...>& tensor_dest,
    BinaryOp bin_op) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_dest);
  return detail::adjacent_difference_impl(
      "flare::adjacent_difference_tensor_api", ex, KE::cbegin(tensor_from),
      KE::cend(tensor_from), KE::begin(tensor_dest), bin_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto adjacent_difference(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor_from,
    const ::flare::Tensor<DataType2, Properties2...>& tensor_dest) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_dest);

  using tensor_type1  = ::flare::Tensor<DataType1, Properties1...>;
  using tensor_type2  = ::flare::Tensor<DataType2, Properties2...>;
  using value_type1 = typename tensor_type1::value_type;
  using value_type2 = typename tensor_type2::value_type;
  using binary_op =
      detail::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;

  return detail::adjacent_difference_impl(label, ex, KE::cbegin(tensor_from),
                                        KE::cend(tensor_from),
                                        KE::begin(tensor_dest), binary_op());
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryOp>
auto adjacent_difference(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor_from,
    const ::flare::Tensor<DataType2, Properties2...>& tensor_dest,
    BinaryOp bin_op) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor_dest);
  return detail::adjacent_difference_impl(label, ex, KE::cbegin(tensor_from),
                                        KE::cend(tensor_from),
                                        KE::begin(tensor_dest), bin_op);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ADJACENT_DIFFERENCE_H_
