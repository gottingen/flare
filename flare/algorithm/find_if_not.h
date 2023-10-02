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

#ifndef FLARE_ALGORITHM_FIND_IF_NOT_H_
#define FLARE_ALGORITHM_FIND_IF_NOT_H_

#include <flare/algorithm/find_if_or_not_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType, typename Predicate,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType find_if_not(const ExecutionSpace& ex, IteratorType first,
                         IteratorType last, Predicate predicate) {
  return detail::find_if_or_not_exespace_impl<false>(
      "flare::find_if_not_iterator_api_default", ex, first, last,
      std::move(predicate));
}

template <
    typename ExecutionSpace, typename IteratorType, typename Predicate,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType find_if_not(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last,
                         Predicate predicate) {
  return detail::find_if_or_not_exespace_impl<false>(label, ex, first, last,
                                                   std::move(predicate));
}

template <typename ExecutionSpace, typename DataType, typename... Properties,
          typename Predicate,
          std::enable_if_t<::flare::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto find_if_not(const ExecutionSpace& ex,
                 const ::flare::View<DataType, Properties...>& v,
                 Predicate predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::find_if_or_not_exespace_impl<false>(
      "flare::find_if_not_view_api_default", ex, KE::begin(v), KE::end(v),
      std::move(predicate));
}

template <typename ExecutionSpace, typename DataType, typename... Properties,
          typename Predicate,
          std::enable_if_t<::flare::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto find_if_not(const std::string& label, const ExecutionSpace& ex,
                 const ::flare::View<DataType, Properties...>& v,
                 Predicate predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::find_if_or_not_exespace_impl<false>(
      label, ex, KE::begin(v), KE::end(v), std::move(predicate));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType, typename Predicate,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION IteratorType find_if_not(const TeamHandleType& teamHandle,
                                         IteratorType first, IteratorType last,
                                         Predicate predicate) {
  return detail::find_if_or_not_team_impl<false>(teamHandle, first, last,
                                               std::move(predicate));
}

template <
    typename TeamHandleType, typename DataType, typename... Properties,
    typename Predicate,
    std::enable_if_t<::flare::is_team_handle<TeamHandleType>::value, int> = 0>
FLARE_FUNCTION auto find_if_not(
    const TeamHandleType& teamHandle,
    const ::flare::View<DataType, Properties...>& v, Predicate predicate) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::find_if_or_not_team_impl<false>(
      teamHandle, KE::begin(v), KE::end(v), std::move(predicate));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FIND_IF_NOT_H_
