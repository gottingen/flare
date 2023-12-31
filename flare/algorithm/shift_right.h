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

#ifndef FLARE_ALGORITHM_SHIFT_RIGHT_H_
#define FLARE_ALGORITHM_SHIFT_RIGHT_H_

#include <flare/algorithm/shift_right_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType shift_right(const ExecutionSpace& ex, IteratorType first,
                         IteratorType last,
                         typename IteratorType::difference_type n) {
  return detail::shift_right_exespace_impl(
      "flare::shift_right_iterator_api_default", ex, first, last, n);
}

template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType shift_right(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last,
                         typename IteratorType::difference_type n) {
  return detail::shift_right_exespace_impl(label, ex, first, last, n);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto shift_right(const ExecutionSpace& ex,
                 const ::flare::View<DataType, Properties...>& view,
                 typename decltype(begin(view))::difference_type n) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::shift_right_exespace_impl("flare::shift_right_view_api_default",
                                         ex, begin(view), end(view), n);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto shift_right(const std::string& label, const ExecutionSpace& ex,
                 const ::flare::View<DataType, Properties...>& view,
                 typename decltype(begin(view))::difference_type n) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::shift_right_exespace_impl(label, ex, begin(view), end(view), n);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION IteratorType
shift_right(const TeamHandleType& teamHandle, IteratorType first,
            IteratorType last, typename IteratorType::difference_type n) {
  return detail::shift_right_team_impl(teamHandle, first, last, n);
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto shift_right(
    const TeamHandleType& teamHandle,
    const ::flare::View<DataType, Properties...>& view,
    typename decltype(begin(view))::difference_type n) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::shift_right_team_impl(teamHandle, begin(view), end(view), n);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_SHIFT_RIGHT_H_
