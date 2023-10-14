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

#ifndef FLARE_ALGORITHM_MINMAX_ELEMENT_H_
#define FLARE_ALGORITHM_MINMAX_ELEMENT_H_

#include <flare/algorithm/min_max_minmax_element_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last) {
  return detail::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      "flare::minmax_element_iterator_api_default", ex, first, last);
}

template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last) {
  return detail::minmax_element_exespace_impl<MinMaxFirstLastLoc>(label, ex,
                                                                first, last);
}

template <
    typename ExecutionSpace, typename IteratorType, typename ComparatorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last, ComparatorType comp) {
  return detail::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      "flare::minmax_element_iterator_api_default", ex, first, last,
      std::move(comp));
}

template <
    typename ExecutionSpace, typename IteratorType, typename ComparatorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last,
                    ComparatorType comp) {

  return detail::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, first, last, std::move(comp));
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex,
                    const ::flare::Tensor<DataType, Properties...>& v) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  return detail::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      "flare::minmax_element_tensor_api_default", ex, begin(v), end(v));
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::flare::Tensor<DataType, Properties...>& v) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  return detail::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      label, ex, begin(v), end(v));
}

template <
    typename ExecutionSpace, typename DataType, typename ComparatorType,
    typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex,
                    const ::flare::Tensor<DataType, Properties...>& v,
                    ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  return detail::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      "flare::minmax_element_tensor_api_default", ex, begin(v), end(v),
      std::move(comp));
}

template <
    typename ExecutionSpace, typename DataType, typename ComparatorType,
    typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::flare::Tensor<DataType, Properties...>& v,
                    ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  return detail::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, begin(v), end(v), std::move(comp));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto minmax_element(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType last) {
  return detail::minmax_element_team_impl<MinMaxFirstLastLoc>(teamHandle, first,
                                                            last);
}

template <typename TeamHandleType, typename IteratorType,
          typename ComparatorType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto minmax_element(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType last,
                                    ComparatorType comp) {

  return detail::minmax_element_team_impl<MinMaxFirstLastLocCustomComparator>(
      teamHandle, first, last, std::move(comp));
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto minmax_element(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType, Properties...>& v) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  return detail::minmax_element_team_impl<MinMaxFirstLastLoc>(teamHandle,
                                                            begin(v), end(v));
}

template <typename TeamHandleType, typename DataType, typename ComparatorType,
          typename... Properties,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto minmax_element(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType, Properties...>& v, ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  return detail::minmax_element_team_impl<MinMaxFirstLastLocCustomComparator>(
      teamHandle, begin(v), end(v), std::move(comp));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_MINMAX_ELEMENT_H_
