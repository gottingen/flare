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

#ifndef FLARE_ALGORITHM_REPLACE_H_
#define FLARE_ALGORITHM_REPLACE_H_

#include <flare/algorithm/replace_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename Iterator, typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace(const ExecutionSpace& ex, Iterator first, Iterator last,
             const ValueType& old_value, const ValueType& new_value) {
  detail::replace_exespace_impl("flare::replace_iterator_api", ex, first, last,
                              old_value, new_value);
}

template <
    typename ExecutionSpace, typename Iterator, typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace(const std::string& label, const ExecutionSpace& ex, Iterator first,
             Iterator last, const ValueType& old_value,
             const ValueType& new_value) {
  detail::replace_exespace_impl(label, ex, first, last, old_value, new_value);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace(const ExecutionSpace& ex,
             const ::flare::View<DataType1, Properties1...>& view,
             const ValueType& old_value, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  namespace KE = ::flare::experimental;
  detail::replace_exespace_impl("flare::replace_view_api", ex, KE::begin(view),
                              KE::end(view), old_value, new_value);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace(const std::string& label, const ExecutionSpace& ex,
             const ::flare::View<DataType1, Properties1...>& view,
             const ValueType& old_value, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  namespace KE = ::flare::experimental;
  detail::replace_exespace_impl(label, ex, KE::begin(view), KE::end(view),
                              old_value, new_value);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename Iterator, typename ValueType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION void replace(const TeamHandleType& teamHandle, Iterator first,
                             Iterator last, const ValueType& old_value,
                             const ValueType& new_value) {
  detail::replace_team_impl(teamHandle, first, last, old_value, new_value);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename ValueType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION void replace(
    const TeamHandleType& teamHandle,
    const ::flare::View<DataType1, Properties1...>& view,
    const ValueType& old_value, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  namespace KE = ::flare::experimental;
  detail::replace_team_impl(teamHandle, KE::begin(view), KE::end(view), old_value,
                          new_value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REPLACE_H_
