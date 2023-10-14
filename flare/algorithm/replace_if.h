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

#ifndef FLARE_ALGORITHM_REPLACE_IF_H_
#define FLARE_ALGORITHM_REPLACE_IF_H_

#include <flare/algorithm/replace_if_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename InputIterator, typename Predicate,
    typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace_if(const ExecutionSpace& ex, InputIterator first,
                InputIterator last, Predicate pred,
                const ValueType& new_value) {
  detail::replace_if_exespace_impl("flare::replace_if_iterator_api", ex, first,
                                 last, pred, new_value);
}

template <
    typename ExecutionSpace, typename InputIterator, typename Predicate,
    typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace_if(const std::string& label, const ExecutionSpace& ex,
                InputIterator first, InputIterator last, Predicate pred,
                const ValueType& new_value) {
  detail::replace_if_exespace_impl(label, ex, first, last, pred, new_value);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename Predicate, typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace_if(const ExecutionSpace& ex,
                const ::flare::Tensor<DataType1, Properties1...>& tensor,
                Predicate pred, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  namespace KE = ::flare::experimental;
  detail::replace_if_exespace_impl("flare::replace_if_tensor_api", ex,
                                 KE::begin(tensor), KE::end(tensor), pred,
                                 new_value);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename Predicate, typename ValueType,
    std::enable_if_t<::flare::is_execution_space_v<ExecutionSpace>, int> = 0>
void replace_if(const std::string& label, const ExecutionSpace& ex,
                const ::flare::Tensor<DataType1, Properties1...>& tensor,
                Predicate pred, const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  namespace KE = ::flare::experimental;
  detail::replace_if_exespace_impl(label, ex, KE::begin(tensor), KE::end(tensor),
                                 pred, new_value);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename InputIterator, typename Predicate,
          typename ValueType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION void replace_if(const TeamHandleType& teamHandle,
                                InputIterator first, InputIterator last,
                                Predicate pred, const ValueType& new_value) {
  detail::replace_if_team_impl(teamHandle, first, last, pred, new_value);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename Predicate, typename ValueType,
          std::enable_if_t<::flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION void replace_if(
    const TeamHandleType& teamHandle,
    const ::flare::Tensor<DataType1, Properties1...>& tensor, Predicate pred,
    const ValueType& new_value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  namespace KE = ::flare::experimental;
  detail::replace_if_team_impl(teamHandle, KE::begin(tensor), KE::end(tensor), pred,
                             new_value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REPLACE_IF_H_
