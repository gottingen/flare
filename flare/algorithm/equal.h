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

#ifndef FLARE_ALGORITHM_EQUAL_H_
#define FLARE_ALGORITHM_EQUAL_H_

#include <flare/algorithm/equal_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2> &&
                               flare::is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2) {
  return detail::equal_exespace_impl("flare::equal_iterator_api_default", ex,
                                   first1, last1, first2);
}

template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2) {
  return detail::equal_exespace_impl(label, ex, first1, last1, first2);
}

template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2, typename BinaryPredicateType,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2, BinaryPredicateType predicate) {
  return detail::equal_exespace_impl("flare::equal_iterator_api_default", ex,
                                   first1, last1, first2, std::move(predicate));
}

template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2, typename BinaryPredicateType,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
           BinaryPredicateType predicate) {
  return detail::equal_exespace_impl(label, ex, first1, last1, first2,
                                   std::move(predicate));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool equal(const ExecutionSpace& ex,
           const ::flare::Tensor<DataType1, Properties1...>& tensor1,
           ::flare::Tensor<DataType2, Properties2...>& tensor2) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::equal_exespace_impl("flare::equal_tensor_api_default", ex,
                                   KE::cbegin(tensor1), KE::cend(tensor1),
                                   KE::cbegin(tensor2));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool equal(const std::string& label, const ExecutionSpace& ex,
           const ::flare::Tensor<DataType1, Properties1...>& tensor1,
           ::flare::Tensor<DataType2, Properties2...>& tensor2) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::equal_exespace_impl(label, ex, KE::cbegin(tensor1),
                                   KE::cend(tensor1), KE::cbegin(tensor2));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename BinaryPredicateType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool equal(const ExecutionSpace& ex,
           const ::flare::Tensor<DataType1, Properties1...>& tensor1,
           ::flare::Tensor<DataType2, Properties2...>& tensor2,
           BinaryPredicateType predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::equal_exespace_impl("flare::equal_tensor_api_default", ex,
                                   KE::cbegin(tensor1), KE::cend(tensor1),
                                   KE::cbegin(tensor2), std::move(predicate));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename BinaryPredicateType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool equal(const std::string& label, const ExecutionSpace& ex,
           const ::flare::Tensor<DataType1, Properties1...>& tensor1,
           ::flare::Tensor<DataType2, Properties2...>& tensor2,
           BinaryPredicateType predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::equal_exespace_impl(label, ex, KE::cbegin(tensor1),
                                   KE::cend(tensor1), KE::cbegin(tensor2),
                                   std::move(predicate));
}

template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2, IteratorType2 last2) {
  return detail::equal_exespace_impl("flare::equal_iterator_api_default", ex,
                                   first1, last1, first2, last2);
}

template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
           IteratorType2 last2) {
  return detail::equal_exespace_impl(label, ex, first1, last1, first2, last2);
}

template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2, typename BinaryPredicateType,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2, IteratorType2 last2,
           BinaryPredicateType predicate) {
  return detail::equal_exespace_impl("flare::equal_iterator_api_default", ex,
                                   first1, last1, first2, last2,
                                   std::move(predicate));
}

template <typename ExecutionSpace, typename IteratorType1,
          typename IteratorType2, typename BinaryPredicateType,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_execution_space_v<ExecutionSpace>,
                           int> = 0>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
           IteratorType2 last2, BinaryPredicateType predicate) {
  return detail::equal_exespace_impl(label, ex, first1, last1, first2, last2,
                                   std::move(predicate));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_team_handle_v<TeamHandleType>,
                           int> = 0>
FLARE_FUNCTION bool equal(const TeamHandleType& teamHandle,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2) {
  return detail::equal_team_impl(teamHandle, first1, last1, first2);
}

template <typename TeamHandleType, typename IteratorType1,
          typename IteratorType2, typename BinaryPredicateType,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_team_handle_v<TeamHandleType>,
                           int> = 0>
FLARE_FUNCTION bool equal(const TeamHandleType& teamHandle,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           BinaryPredicateType predicate) {
  return detail::equal_team_impl(teamHandle, first1, last1, first2,
                               std::move(predicate));
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION bool equal(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::equal_team_impl(teamHandle, KE::cbegin(tensor1), KE::cend(tensor1),
                               KE::cbegin(tensor2));
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2,
          typename BinaryPredicateType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION bool equal(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2,
    BinaryPredicateType predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::equal_team_impl(teamHandle, KE::cbegin(tensor1), KE::cend(tensor1),
                               KE::cbegin(tensor2), std::move(predicate));
}

template <typename TeamHandleType, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_team_handle_v<TeamHandleType>,
                           int> = 0>
FLARE_FUNCTION bool equal(const TeamHandleType& teamHandle,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2, IteratorType2 last2) {
  return detail::equal_team_impl(teamHandle, first1, last1, first2, last2);
}

template <typename TeamHandleType, typename IteratorType1,
          typename IteratorType2, typename BinaryPredicateType,
          std::enable_if_t<::flare::experimental::detail::are_iterators_v<
                               IteratorType1, IteratorType2>&& ::flare::
                               is_team_handle_v<TeamHandleType>,
                           int> = 0>
FLARE_FUNCTION bool equal(const TeamHandleType& teamHandle,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2, IteratorType2 last2,
                           BinaryPredicateType predicate) {
  return detail::equal_team_impl(teamHandle, first1, last1, first2, last2,
                               std::move(predicate));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_EQUAL_H_
