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

#ifndef FLARE_ALGORITHM_FIND_FIRST_OF_H_
#define FLARE_ALGORITHM_FIND_FIRST_OF_H_

#include <flare/algorithm/find_first_of_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//

// overload set 1: no binary predicate passed
template <
    typename ExecutionSpace, typename IteratorType1, typename IteratorType2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType1 find_first_of(const ExecutionSpace& ex, IteratorType1 first,
                            IteratorType1 last, IteratorType2 s_first,
                            IteratorType2 s_last) {
  return detail::find_first_of_exespace_impl(
      "flare::find_first_of_iterator_api_default", ex, first, last, s_first,
      s_last);
}

template <
    typename ExecutionSpace, typename IteratorType1, typename IteratorType2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType1 find_first_of(const std::string& label, const ExecutionSpace& ex,
                            IteratorType1 first, IteratorType1 last,
                            IteratorType2 s_first, IteratorType2 s_last) {
  return detail::find_first_of_exespace_impl(label, ex, first, last, s_first,
                                           s_last);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto find_first_of(const ExecutionSpace& ex,
                   const ::flare::Tensor<DataType1, Properties1...>& tensor,
                   const ::flare::Tensor<DataType2, Properties2...>& s_tensor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::static_assert_is_admissible_to_flare_std_algorithms(s_tensor);

  namespace KE = ::flare::experimental;
  return detail::find_first_of_exespace_impl(
      "flare::find_first_of_tensor_api_default", ex, KE::begin(tensor),
      KE::end(tensor), KE::begin(s_tensor), KE::end(s_tensor));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto find_first_of(const std::string& label, const ExecutionSpace& ex,
                   const ::flare::Tensor<DataType1, Properties1...>& tensor,
                   const ::flare::Tensor<DataType2, Properties2...>& s_tensor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::static_assert_is_admissible_to_flare_std_algorithms(s_tensor);

  namespace KE = ::flare::experimental;
  return detail::find_first_of_exespace_impl(label, ex, KE::begin(tensor),
                                           KE::end(tensor), KE::begin(s_tensor),
                                           KE::end(s_tensor));
}

// overload set 2: binary predicate passed
template <
    typename ExecutionSpace, typename IteratorType1, typename IteratorType2,
    typename BinaryPredicateType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType1 find_first_of(const ExecutionSpace& ex, IteratorType1 first,
                            IteratorType1 last, IteratorType2 s_first,
                            IteratorType2 s_last,
                            const BinaryPredicateType& pred) {
  return detail::find_first_of_exespace_impl(
      "flare::find_first_of_iterator_api_default", ex, first, last, s_first,
      s_last, pred);
}

template <
    typename ExecutionSpace, typename IteratorType1, typename IteratorType2,
    typename BinaryPredicateType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType1 find_first_of(const std::string& label, const ExecutionSpace& ex,
                            IteratorType1 first, IteratorType1 last,
                            IteratorType2 s_first, IteratorType2 s_last,
                            const BinaryPredicateType& pred) {
  return detail::find_first_of_exespace_impl(label, ex, first, last, s_first,
                                           s_last, pred);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename BinaryPredicateType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto find_first_of(const ExecutionSpace& ex,
                   const ::flare::Tensor<DataType1, Properties1...>& tensor,
                   const ::flare::Tensor<DataType2, Properties2...>& s_tensor,
                   const BinaryPredicateType& pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::static_assert_is_admissible_to_flare_std_algorithms(s_tensor);

  namespace KE = ::flare::experimental;
  return detail::find_first_of_exespace_impl(
      "flare::find_first_of_tensor_api_default", ex, KE::begin(tensor),
      KE::end(tensor), KE::begin(s_tensor), KE::end(s_tensor), pred);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename BinaryPredicateType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto find_first_of(const std::string& label, const ExecutionSpace& ex,
                   const ::flare::Tensor<DataType1, Properties1...>& tensor,
                   const ::flare::Tensor<DataType2, Properties2...>& s_tensor,
                   const BinaryPredicateType& pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::static_assert_is_admissible_to_flare_std_algorithms(s_tensor);

  namespace KE = ::flare::experimental;
  return detail::find_first_of_exespace_impl(label, ex, KE::begin(tensor),
                                           KE::end(tensor), KE::begin(s_tensor),
                                           KE::end(s_tensor), pred);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

// overload set 1: no binary predicate passed
template <typename TeamHandleType, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION IteratorType1 find_first_of(const TeamHandleType& teamHandle,
                                            IteratorType1 first,
                                            IteratorType1 last,
                                            IteratorType2 s_first,
                                            IteratorType2 s_last) {
  return detail::find_first_of_team_impl(teamHandle, first, last, s_first,
                                       s_last);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto find_first_of(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType1, Properties1...>& tensor,
    const ::flare::Tensor<DataType2, Properties2...>& s_tensor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::static_assert_is_admissible_to_flare_std_algorithms(s_tensor);

  namespace KE = ::flare::experimental;
  return detail::find_first_of_team_impl(teamHandle, KE::begin(tensor),
                                       KE::end(tensor), KE::begin(s_tensor),
                                       KE::end(s_tensor));
}

// overload set 2: binary predicate passed
template <typename TeamHandleType, typename IteratorType1,
          typename IteratorType2, typename BinaryPredicateType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>

FLARE_FUNCTION IteratorType1 find_first_of(const TeamHandleType& teamHandle,
                                            IteratorType1 first,
                                            IteratorType1 last,
                                            IteratorType2 s_first,
                                            IteratorType2 s_last,
                                            const BinaryPredicateType& pred) {
  return detail::find_first_of_team_impl(teamHandle, first, last, s_first, s_last,
                                       pred);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2,
          typename BinaryPredicateType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto find_first_of(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType1, Properties1...>& tensor,
    const ::flare::Tensor<DataType2, Properties2...>& s_tensor,
    const BinaryPredicateType& pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::static_assert_is_admissible_to_flare_std_algorithms(s_tensor);

  namespace KE = ::flare::experimental;
  return detail::find_first_of_team_impl(teamHandle, KE::begin(tensor),
                                       KE::end(tensor), KE::begin(s_tensor),
                                       KE::end(s_tensor), pred);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FIND_FIRST_OF_H_
