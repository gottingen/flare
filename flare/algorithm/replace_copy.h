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

#ifndef FLARE_ALGORITHM_REPLACE_COPY_H_
#define FLARE_ALGORITHM_REPLACE_COPY_H_

#include <flare/algorithm/replace_copy_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
OutputIterator replace_copy(const ExecutionSpace& ex, InputIterator first_from,
                            InputIterator last_from, OutputIterator first_dest,
                            const ValueType& old_value,
                            const ValueType& new_value) {
  return detail::replace_copy_exespace_impl("flare::replace_copy_iterator_api",
                                          ex, first_from, last_from, first_dest,
                                          old_value, new_value);
}

template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
OutputIterator replace_copy(const std::string& label, const ExecutionSpace& ex,
                            InputIterator first_from, InputIterator last_from,
                            OutputIterator first_dest,
                            const ValueType& old_value,
                            const ValueType& new_value) {
  return detail::replace_copy_exespace_impl(label, ex, first_from, last_from,
                                          first_dest, old_value, new_value);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto replace_copy(const ExecutionSpace& ex,
                  const ::flare::View<DataType1, Properties1...>& view_from,
                  const ::flare::View<DataType2, Properties2...>& view_dest,
                  const ValueType& old_value, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);
  namespace KE = ::flare::experimental;
  return detail::replace_copy_exespace_impl(
      "flare::replace_copy_view_api", ex, KE::cbegin(view_from),
      KE::cend(view_from), KE::begin(view_dest), old_value, new_value);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto replace_copy(const std::string& label, const ExecutionSpace& ex,
                  const ::flare::View<DataType1, Properties1...>& view_from,
                  const ::flare::View<DataType2, Properties2...>& view_dest,
                  const ValueType& old_value, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);
  namespace KE = ::flare::experimental;
  return detail::replace_copy_exespace_impl(
      label, ex, KE::cbegin(view_from), KE::cend(view_from),
      KE::begin(view_dest), old_value, new_value);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename InputIterator,
          typename OutputIterator, typename ValueType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION OutputIterator replace_copy(const TeamHandleType& teamHandle,
                                            InputIterator first_from,
                                            InputIterator last_from,
                                            OutputIterator first_dest,
                                            const ValueType& old_value,
                                            const ValueType& new_value) {
  return detail::replace_copy_team_impl(teamHandle, first_from, last_from,
                                      first_dest, old_value, new_value);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename ValueType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto replace_copy(
    const TeamHandleType& teamHandle,
    const ::flare::View<DataType1, Properties1...>& view_from,
    const ::flare::View<DataType2, Properties2...>& view_dest,
    const ValueType& old_value, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);
  namespace KE = ::flare::experimental;
  return detail::replace_copy_team_impl(teamHandle, KE::cbegin(view_from),
                                      KE::cend(view_from), KE::begin(view_dest),
                                      old_value, new_value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REPLACE_COPY_H_
