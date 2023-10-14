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

#ifndef FLARE_ALGORITHM_FILL_H_
#define FLARE_ALGORITHM_FILL_H_

#include <flare/algorithm/fill_n_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType, typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void fill(const ExecutionSpace& ex, IteratorType first, IteratorType last,
          const T& value) {
  detail::fill_exespace_impl("flare::fill_iterator_api_default", ex, first, last,
                           value);
}

template <
    typename ExecutionSpace, typename IteratorType, typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void fill(const std::string& label, const ExecutionSpace& ex,
          IteratorType first, IteratorType last, const T& value) {
  detail::fill_exespace_impl(label, ex, first, last, value);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void fill(const ExecutionSpace& ex,
          const ::flare::Tensor<DataType, Properties...>& tensor, const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::fill_exespace_impl("flare::fill_tensor_api_default", ex, begin(tensor),
                           end(tensor), value);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename T,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void fill(const std::string& label, const ExecutionSpace& ex,
          const ::flare::Tensor<DataType, Properties...>& tensor, const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::fill_exespace_impl(label, ex, begin(tensor), end(tensor), value);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType, typename T,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION void fill(const TeamHandleType& th, IteratorType first,
                          IteratorType last, const T& value) {
  detail::fill_team_impl(th, first, last, value);
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          typename T,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION void fill(const TeamHandleType& th,
                          const ::flare::Tensor<DataType, Properties...>& tensor,
                          const T& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  detail::fill_team_impl(th, begin(tensor), end(tensor), value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FILL_H_
