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

#ifndef FLARE_ALGORITHM_REVERSE_COPY_H_
#define FLARE_ALGORITHM_REVERSE_COPY_H_

#include <flare/algorithm/reverse_copy_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
OutputIterator reverse_copy(const ExecutionSpace& ex, InputIterator first,
                            InputIterator last, OutputIterator d_first) {
  return detail::reverse_copy_exespace_impl(
      "flare::reverse_copy_iterator_api_default", ex, first, last, d_first);
}

template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
OutputIterator reverse_copy(const std::string& label, const ExecutionSpace& ex,
                            InputIterator first, InputIterator last,
                            OutputIterator d_first) {
  return detail::reverse_copy_exespace_impl(label, ex, first, last, d_first);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto reverse_copy(const ExecutionSpace& ex,
                  const ::flare::View<DataType1, Properties1...>& source,
                  ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::reverse_copy_exespace_impl(
      "flare::reverse_copy_view_api_default", ex, cbegin(source), cend(source),
      begin(dest));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto reverse_copy(const std::string& label, const ExecutionSpace& ex,
                  const ::flare::View<DataType1, Properties1...>& source,
                  ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::reverse_copy_exespace_impl(label, ex, cbegin(source),
                                          cend(source), begin(dest));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION OutputIterator reverse_copy(const TeamHandleType& teamHandle,
                                            InputIterator first,
                                            InputIterator last,
                                            OutputIterator d_first) {
  return detail::reverse_copy_team_impl(teamHandle, first, last, d_first);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto reverse_copy(
    const TeamHandleType& teamHandle,
    const ::flare::View<DataType1, Properties1...>& source,
    ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::reverse_copy_team_impl(teamHandle, cbegin(source), cend(source),
                                      begin(dest));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REVERSE_COPY_H_
