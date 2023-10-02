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
std::enable_if_t<!::flare::is_view<InputIteratorType>::value,
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
std::enable_if_t<!::flare::is_view<InputIteratorType>::value,
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
std::enable_if_t<!::flare::is_view<InputIteratorType>::value,
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
std::enable_if_t<!::flare::is_view<InputIteratorType>::value,
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
    const ::flare::View<DataType1, Properties1...>& view_from,
    const ::flare::View<DataType2, Properties2...>& view_dest) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);

  using view_type1  = ::flare::View<DataType1, Properties1...>;
  using view_type2  = ::flare::View<DataType2, Properties2...>;
  using value_type1 = typename view_type1::value_type;
  using value_type2 = typename view_type2::value_type;
  using binary_op =
      detail::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;
  return detail::adjacent_difference_impl(
      "flare::adjacent_difference_view_api", ex, KE::cbegin(view_from),
      KE::cend(view_from), KE::begin(view_dest), binary_op());
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryOp>
auto adjacent_difference(
    const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& view_from,
    const ::flare::View<DataType2, Properties2...>& view_dest,
    BinaryOp bin_op) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);
  return detail::adjacent_difference_impl(
      "flare::adjacent_difference_view_api", ex, KE::cbegin(view_from),
      KE::cend(view_from), KE::begin(view_dest), bin_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto adjacent_difference(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& view_from,
    const ::flare::View<DataType2, Properties2...>& view_dest) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);

  using view_type1  = ::flare::View<DataType1, Properties1...>;
  using view_type2  = ::flare::View<DataType2, Properties2...>;
  using value_type1 = typename view_type1::value_type;
  using value_type2 = typename view_type2::value_type;
  using binary_op =
      detail::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;

  return detail::adjacent_difference_impl(label, ex, KE::cbegin(view_from),
                                        KE::cend(view_from),
                                        KE::begin(view_dest), binary_op());
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryOp>
auto adjacent_difference(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::View<DataType1, Properties1...>& view_from,
    const ::flare::View<DataType2, Properties2...>& view_dest,
    BinaryOp bin_op) {
  namespace KE = ::flare::experimental;
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);
  return detail::adjacent_difference_impl(label, ex, KE::cbegin(view_from),
                                        KE::cend(view_from),
                                        KE::begin(view_dest), bin_op);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ADJACENT_DIFFERENCE_H_
