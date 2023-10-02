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

#ifndef FLARE_ALGORITHM_MOVE_BACKWARD_H_
#define FLARE_ALGORITHM_MOVE_BACKWARD_H_

#include <flare/algorithm/move_backward_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType1, typename IteratorType2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType2 move_backward(const ExecutionSpace& ex, IteratorType1 first,
                            IteratorType1 last, IteratorType2 d_last) {
  return detail::move_backward_exespace_impl(
      "flare::move_backward_iterator_api_default", ex, first, last, d_last);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto move_backward(const ExecutionSpace& ex,
                   const ::flare::View<DataType1, Properties1...>& source,
                   ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::move_backward_exespace_impl(
      "flare::move_backward_view_api_default", ex, begin(source), end(source),
      end(dest));
}

template <
    typename ExecutionSpace, typename IteratorType1, typename IteratorType2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType2 move_backward(const std::string& label, const ExecutionSpace& ex,
                            IteratorType1 first, IteratorType1 last,
                            IteratorType2 d_last) {
  return detail::move_backward_exespace_impl(label, ex, first, last, d_last);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto move_backward(const std::string& label, const ExecutionSpace& ex,
                   const ::flare::View<DataType1, Properties1...>& source,
                   ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::move_backward_exespace_impl(label, ex, begin(source),
                                           end(source), end(dest));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType1,
          typename IteratorType2,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION IteratorType2 move_backward(const TeamHandleType& teamHandle,
                                            IteratorType1 first,
                                            IteratorType1 last,
                                            IteratorType2 d_last) {
  return detail::move_backward_team_impl(teamHandle, first, last, d_last);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto move_backward(
    const TeamHandleType& teamHandle,
    const ::flare::View<DataType1, Properties1...>& source,
    ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::move_backward_team_impl(teamHandle, begin(source), end(source),
                                       end(dest));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_MOVE_BACKWARD_H_
