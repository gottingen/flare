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

#ifndef FLARE_ALGORITHM_FOR_EACH_H_
#define FLARE_ALGORITHM_FOR_EACH_H_

#include <flare/algorithm/for_each_for_each_n_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    class ExecutionSpace, class IteratorType, class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& ex,
                          IteratorType first, IteratorType last,
                          UnaryFunctorType functor) {
  return detail::for_each_exespace_impl(label, ex, first, last,
                                      std::move(functor));
}

template <
    class ExecutionSpace, class IteratorType, class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
UnaryFunctorType for_each(const ExecutionSpace& ex, IteratorType first,
                          IteratorType last, UnaryFunctorType functor) {
  return detail::for_each_exespace_impl("flare::for_each_iterator_api_default",
                                      ex, first, last, std::move(functor));
}

template <
    class ExecutionSpace, class DataType, class... Properties,
    class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& ex,
                          const ::flare::View<DataType, Properties...>& v,
                          UnaryFunctorType functor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::for_each_exespace_impl(label, ex, KE::begin(v), KE::end(v),
                                      std::move(functor));
}

template <
    class ExecutionSpace, class DataType, class... Properties,
    class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
UnaryFunctorType for_each(const ExecutionSpace& ex,
                          const ::flare::View<DataType, Properties...>& v,
                          UnaryFunctorType functor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::for_each_exespace_impl("flare::for_each_view_api_default", ex,
                                      KE::begin(v), KE::end(v),
                                      std::move(functor));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

template <class TeamHandleType, class IteratorType, class UnaryFunctorType,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION UnaryFunctorType for_each(const TeamHandleType& teamHandle,
                                          IteratorType first, IteratorType last,
                                          UnaryFunctorType functor) {
  return detail::for_each_team_impl(teamHandle, first, last, std::move(functor));
}

template <class TeamHandleType, class DataType, class... Properties,
          class UnaryFunctorType,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION UnaryFunctorType
for_each(const TeamHandleType& teamHandle,
         const ::flare::View<DataType, Properties...>& v,
         UnaryFunctorType functor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::for_each_team_impl(teamHandle, KE::begin(v), KE::end(v),
                                  std::move(functor));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FOR_EACH_H_
