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

#ifndef FLARE_ALGORITHM_FIND_H_
#define FLARE_ALGORITHM_FIND_H_

#include <flare/algorithm/find_if_or_not_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename InputIterator, typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
InputIterator find(const ExecutionSpace& ex, InputIterator first,
                   InputIterator last, const T& value) {
  return detail::find_exespace_impl("flare::find_iterator_api_default", ex,
                                  first, last, value);
}

template <
    typename ExecutionSpace, typename InputIterator, typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
InputIterator find(const std::string& label, const ExecutionSpace& ex,
                   InputIterator first, InputIterator last, const T& value) {
  return detail::find_exespace_impl(label, ex, first, last, value);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto find(const ExecutionSpace& ex,
          const ::flare::View<DataType, Properties...>& view, const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  namespace KE = ::flare::experimental;
  return detail::find_exespace_impl("flare::find_view_api_default", ex,
                                  KE::begin(view), KE::end(view), value);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto find(const std::string& label, const ExecutionSpace& ex,
          const ::flare::View<DataType, Properties...>& view, const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  namespace KE = ::flare::experimental;
  return detail::find_exespace_impl(label, ex, KE::begin(view), KE::end(view),
                                  value);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename InputIterator, typename T,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION InputIterator find(const TeamHandleType& teamHandle,
                                   InputIterator first, InputIterator last,
                                   const T& value) {
  return detail::find_team_impl(teamHandle, first, last, value);
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          typename T,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto find(const TeamHandleType& teamHandle,
                          const ::flare::View<DataType, Properties...>& view,
                          const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  namespace KE = ::flare::experimental;
  return detail::find_team_impl(teamHandle, KE::begin(view), KE::end(view),
                              value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FIND_H_
