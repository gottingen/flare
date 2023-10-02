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

#ifndef FLARE_ALGORITHM_ROTATE_H_
#define FLARE_ALGORITHM_ROTATE_H_

#include <flare/algorithm/rotate_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType rotate(const ExecutionSpace& ex, IteratorType first,
                    IteratorType n_first, IteratorType last) {
  return detail::rotate_exespace_impl("flare::rotate_iterator_api_default", ex,
                                    first, n_first, last);
}

template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType rotate(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType n_first,
                    IteratorType last) {
  return detail::rotate_exespace_impl(label, ex, first, n_first, last);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto rotate(const ExecutionSpace& ex,
            const ::flare::View<DataType, Properties...>& view,
            std::size_t n_location) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::rotate_exespace_impl("flare::rotate_view_api_default", ex,
                                    begin(view), begin(view) + n_location,
                                    end(view));
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto rotate(const std::string& label, const ExecutionSpace& ex,
            const ::flare::View<DataType, Properties...>& view,
            std::size_t n_location) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::rotate_exespace_impl(label, ex, begin(view),
                                    begin(view) + n_location, end(view));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION IteratorType rotate(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType n_first,
                                    IteratorType last) {
  return detail::rotate_team_impl(teamHandle, first, n_first, last);
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto rotate(const TeamHandleType& teamHandle,
                            const ::flare::View<DataType, Properties...>& view,
                            std::size_t n_location) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::rotate_team_impl(teamHandle, begin(view),
                                begin(view) + n_location, end(view));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ROTATE_H_
