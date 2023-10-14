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

#ifndef FLARE_ALGORITHM_NONE_OF_H_
#define FLARE_ALGORITHM_NONE_OF_H_

#include <flare/algorithm/all_of_any_of_none_of_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType, typename Predicate,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool none_of(const ExecutionSpace& ex, IteratorType first, IteratorType last,
             Predicate predicate) {
  return detail::none_of_exespace_impl("flare::none_of_iterator_api_default", ex,
                                     first, last, predicate);
}

template <
    typename ExecutionSpace, typename IteratorType, typename Predicate,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool none_of(const std::string& label, const ExecutionSpace& ex,
             IteratorType first, IteratorType last, Predicate predicate) {
  return detail::none_of_exespace_impl(label, ex, first, last, predicate);
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename Predicate,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool none_of(const ExecutionSpace& ex,
             const ::flare::Tensor<DataType, Properties...>& v,
             Predicate predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::none_of_exespace_impl("flare::none_of_tensor_api_default", ex,
                                     KE::cbegin(v), KE::cend(v),
                                     std::move(predicate));
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    typename Predicate,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
bool none_of(const std::string& label, const ExecutionSpace& ex,
             const ::flare::Tensor<DataType, Properties...>& v,
             Predicate predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::none_of_exespace_impl(label, ex, KE::cbegin(v), KE::cend(v),
                                     std::move(predicate));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType, typename Predicate>
FLARE_FUNCTION
    std::enable_if_t<::flare::is_team_handle<TeamHandleType>::value, bool>
    none_of(const TeamHandleType& teamHandle, IteratorType first,
            IteratorType last, Predicate predicate) {
  return detail::none_of_team_impl(teamHandle, first, last, predicate);
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          typename Predicate>
FLARE_FUNCTION
    std::enable_if_t<::flare::is_team_handle<TeamHandleType>::value, bool>
    none_of(const TeamHandleType& teamHandle,
            const ::flare::Tensor<DataType, Properties...>& v,
            Predicate predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::none_of_team_impl(teamHandle, KE::cbegin(v), KE::cend(v),
                                 std::move(predicate));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_NONE_OF_H_
