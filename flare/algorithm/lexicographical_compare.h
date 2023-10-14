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

#ifndef FLARE_ALGORITHM_LEXICOGRAPHICAL_COMPARE_H_
#define FLARE_ALGORITHM_LEXICOGRAPHICAL_COMPARE_H_

#include <flare/algorithm/lexicographical_compare_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    class ExecutionSpace, class IteratorType1, class IteratorType2,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(const ExecutionSpace& ex, IteratorType1 first1,
                             IteratorType1 last1, IteratorType2 first2,
                             IteratorType2 last2) {
  return detail::lexicographical_compare_exespace_impl(
      "flare::lexicographical_compare_iterator_api_default", ex, first1, last1,
      first2, last2);
}

template <
    class ExecutionSpace, class IteratorType1, class IteratorType2,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(const std::string& label, const ExecutionSpace& ex,
                             IteratorType1 first1, IteratorType1 last1,
                             IteratorType2 first2, IteratorType2 last2) {
  return detail::lexicographical_compare_exespace_impl(label, ex, first1, last1,
                                                     first2, last2);
}

template <
    class ExecutionSpace, class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(
    const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::lexicographical_compare_exespace_impl(
      "flare::lexicographical_compare_tensor_api_default", ex, KE::cbegin(tensor1),
      KE::cend(tensor1), KE::cbegin(tensor2), KE::cend(tensor2));
}

template <
    class ExecutionSpace, class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::lexicographical_compare_exespace_impl(
      label, ex, KE::cbegin(tensor1), KE::cend(tensor1), KE::cbegin(tensor2),
      KE::cend(tensor2));
}

template <
    class ExecutionSpace, class IteratorType1, class IteratorType2,
    class ComparatorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(const ExecutionSpace& ex, IteratorType1 first1,
                             IteratorType1 last1, IteratorType2 first2,
                             IteratorType2 last2, ComparatorType comp) {
  return detail::lexicographical_compare_exespace_impl(
      "flare::lexicographical_compare_iterator_api_default", ex, first1, last1,
      first2, last2, comp);
}

template <
    class ExecutionSpace, class IteratorType1, class IteratorType2,
    class ComparatorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(const std::string& label, const ExecutionSpace& ex,
                             IteratorType1 first1, IteratorType1 last1,
                             IteratorType2 first2, IteratorType2 last2,
                             ComparatorType comp) {
  return detail::lexicographical_compare_exespace_impl(label, ex, first1, last1,
                                                     first2, last2, comp);
}

template <
    class ExecutionSpace, class DataType1, class... Properties1,
    class DataType2, class... Properties2, class ComparatorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(
    const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2, ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::lexicographical_compare_exespace_impl(
      "flare::lexicographical_compare_tensor_api_default", ex, KE::cbegin(tensor1),
      KE::cend(tensor1), KE::cbegin(tensor2), KE::cend(tensor2), comp);
}

template <
    class ExecutionSpace, class DataType1, class... Properties1,
    class DataType2, class... Properties2, class ComparatorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool lexicographical_compare(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2, ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::lexicographical_compare_exespace_impl(
      label, ex, KE::cbegin(tensor1), KE::cend(tensor1), KE::cbegin(tensor2),
      KE::cend(tensor2), comp);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <class TeamHandleType, class IteratorType1, class IteratorType2,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION bool lexicographical_compare(const TeamHandleType& teamHandle,
                                             IteratorType1 first1,
                                             IteratorType1 last1,
                                             IteratorType2 first2,
                                             IteratorType2 last2) {
  return detail::lexicographical_compare_team_impl(teamHandle, first1, last1,
                                                 first2, last2);
}

template <class TeamHandleType, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION bool lexicographical_compare(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::lexicographical_compare_team_impl(
      teamHandle, KE::cbegin(tensor1), KE::cend(tensor1), KE::cbegin(tensor2),
      KE::cend(tensor2));
}

template <class TeamHandleType, class IteratorType1, class IteratorType2,
          class ComparatorType,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION bool lexicographical_compare(
    const TeamHandleType& teamHandle, IteratorType1 first1, IteratorType1 last1,
    IteratorType2 first2, IteratorType2 last2, ComparatorType comp) {
  return detail::lexicographical_compare_team_impl(teamHandle, first1, last1,
                                                 first2, last2, comp);
}

template <class TeamHandleType, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ComparatorType,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION bool lexicographical_compare(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType1, Properties1...>& tensor1,
    ::flare::Tensor<DataType2, Properties2...>& tensor2, ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor1);
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor2);

  namespace KE = ::flare::experimental;
  return detail::lexicographical_compare_team_impl(
      teamHandle, KE::cbegin(tensor1), KE::cend(tensor1), KE::cbegin(tensor2),
      KE::cend(tensor2), comp);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_LEXICOGRAPHICAL_COMPARE_H_
