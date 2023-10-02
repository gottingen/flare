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

#ifndef FLARE_ALGORITHM_FILL_N_H_
#define FLARE_ALGORITHM_FILL_N_H_

#include <flare/algorithm/fill_n_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <
    typename ExecutionSpace, typename IteratorType, typename SizeType,
    typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType fill_n(const ExecutionSpace& ex, IteratorType first, SizeType n,
                    const T& value) {
  return detail::fill_n_exespace_impl("flare::fill_n_iterator_api_default", ex,
                                    first, n, value);
}

template <
    typename ExecutionSpace, typename IteratorType, typename SizeType,
    typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType fill_n(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, SizeType n, const T& value) {
  return detail::fill_n_exespace_impl(label, ex, first, n, value);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename SizeType, typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto fill_n(const ExecutionSpace& ex,
            const ::flare::View<DataType, Properties...>& view, SizeType n,
            const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  return detail::fill_n_exespace_impl("flare::fill_n_view_api_default", ex,
                                    begin(view), n, value);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename SizeType, typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto fill_n(const std::string& label, const ExecutionSpace& ex,
            const ::flare::View<DataType, Properties...>& view, SizeType n,
            const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  return detail::fill_n_exespace_impl(label, ex, begin(view), n, value);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType, typename SizeType,
          typename T,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION IteratorType fill_n(const TeamHandleType& th,
                                    IteratorType first, SizeType n,
                                    const T& value) {
  return detail::fill_n_team_impl(th, first, n, value);
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          typename SizeType, typename T,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto fill_n(const TeamHandleType& th,
                            const ::flare::View<DataType, Properties...>& view,
                            SizeType n, const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::fill_n_team_impl(th, begin(view), n, value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FILL_N_H_
