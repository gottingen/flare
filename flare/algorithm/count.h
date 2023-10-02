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

#ifndef FLARE_ALGORITHM_COUNT_H_
#define FLARE_ALGORITHM_COUNT_H_

#include <flare/algorithm/count_if_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    class ExecutionSpace, class IteratorType, class T,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
typename IteratorType::difference_type count(const ExecutionSpace& ex,
                                             IteratorType first,
                                             IteratorType last,
                                             const T& value) {
  return detail::count_exespace_impl("flare::count_iterator_api_default", ex,
                                   first, last, value);
}

template <
    class ExecutionSpace, class IteratorType, class T,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
typename IteratorType::difference_type count(const std::string& label,
                                             const ExecutionSpace& ex,
                                             IteratorType first,
                                             IteratorType last,
                                             const T& value) {
  return detail::count_exespace_impl(label, ex, first, last, value);
}

template <
    class ExecutionSpace, class DataType, class... Properties, class T,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto count(const ExecutionSpace& ex,
           const ::flare::View<DataType, Properties...>& v, const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::count_exespace_impl("flare::count_view_api_default", ex,
                                   KE::cbegin(v), KE::cend(v), value);
}

template <
    class ExecutionSpace, class DataType, class... Properties, class T,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto count(const std::string& label, const ExecutionSpace& ex,
           const ::flare::View<DataType, Properties...>& v, const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::count_exespace_impl(label, ex, KE::cbegin(v), KE::cend(v),
                                   value);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

template <class TeamHandleType, class IteratorType, class T,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>

FLARE_FUNCTION typename IteratorType::difference_type count(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last,
    const T& value) {
  return detail::count_team_impl(teamHandle, first, last, value);
}

template <class TeamHandleType, class DataType, class... Properties, class T,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto count(const TeamHandleType& teamHandle,
                           const ::flare::View<DataType, Properties...>& v,
                           const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::count_team_impl(teamHandle, KE::cbegin(v), KE::cend(v), value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_COUNT_H_
