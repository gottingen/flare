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

#ifndef FLARE_ALGORITHM_FOR_EACH_N_H_
#define FLARE_ALGORITHM_FOR_EACH_N_H_

#include <flare/algorithm/for_each_for_each_n_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

//
// overload set accepting execution space
//
template <
    class ExecutionSpace, class IteratorType, class SizeType,
    class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType for_each_n(const std::string& label, const ExecutionSpace& ex,
                        IteratorType first, SizeType n,
                        UnaryFunctorType functor) {
  return detail::for_each_n_exespace_impl(label, ex, first, n,
                                        std::move(functor));
}

template <
    class ExecutionSpace, class IteratorType, class SizeType,
    class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
IteratorType for_each_n(const ExecutionSpace& ex, IteratorType first,
                        SizeType n, UnaryFunctorType functor) {
  return detail::for_each_n_exespace_impl(
      "flare::for_each_n_iterator_api_default", ex, first, n,
      std::move(functor));
}

template <
    class ExecutionSpace, class DataType, class... Properties, class SizeType,
    class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto for_each_n(const std::string& label, const ExecutionSpace& ex,
                const ::flare::View<DataType, Properties...>& v, SizeType n,
                UnaryFunctorType functor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::for_each_n_exespace_impl(label, ex, KE::begin(v), n,
                                        std::move(functor));
}

template <
    class ExecutionSpace, class DataType, class... Properties, class SizeType,
    class UnaryFunctorType,
    std::enable_if_t<flare::is_execution_space_v<ExecutionSpace>, int> = 0>
auto for_each_n(const ExecutionSpace& ex,
                const ::flare::View<DataType, Properties...>& v, SizeType n,
                UnaryFunctorType functor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::for_each_n_exespace_impl("flare::for_each_n_view_api_default",
                                        ex, KE::begin(v), n,
                                        std::move(functor));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

template <class TeamHandleType, class IteratorType, class SizeType,
          class UnaryFunctorType,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION IteratorType for_each_n(const TeamHandleType& teamHandle,
                                        IteratorType first, SizeType n,
                                        UnaryFunctorType functor) {
  return detail::for_each_n_team_impl(teamHandle, first, n, std::move(functor));
}

template <class TeamHandleType, class DataType, class... Properties,
          class SizeType, class UnaryFunctorType,
          std::enable_if_t<flare::is_team_handle_v<TeamHandleType>, int> = 0>
FLARE_FUNCTION auto for_each_n(
    const TeamHandleType& teamHandle,
    const ::flare::View<DataType, Properties...>& v, SizeType n,
    UnaryFunctorType functor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  namespace KE = ::flare::experimental;
  return detail::for_each_n_team_impl(teamHandle, KE::begin(v), n,
                                    std::move(functor));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FOR_EACH_N_H_
