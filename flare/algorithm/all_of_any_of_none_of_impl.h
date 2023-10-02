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

#ifndef FLARE_ALGORITHM_ALL_OF_ANY_OF_NONE_OF_IMPL_H_
#define FLARE_ALGORITHM_ALL_OF_ANY_OF_NONE_OF_IMPL_H_

#include <flare/algorithm/find_if_or_not_impl.h>

namespace flare {
namespace experimental {
namespace detail {

//
// exespace impl
//
template <class ExecutionSpace, class InputIterator, class Predicate>
bool all_of_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                          InputIterator first, InputIterator last,
                          Predicate predicate) {
  return (find_if_or_not_exespace_impl<false>(label, ex, first, last,
                                              predicate) == last);
}

template <class ExecutionSpace, class InputIterator, class Predicate>
bool any_of_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                          InputIterator first, InputIterator last,
                          Predicate predicate) {
  return (find_if_or_not_exespace_impl<true>(label, ex, first, last,
                                             predicate) != last);
}

template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                           IteratorType first, IteratorType last,
                           Predicate predicate) {
  return (find_if_or_not_exespace_impl<true>(label, ex, first, last,
                                             predicate) == last);
}

//
// team impl
//
template <class TeamHandleType, class InputIterator, class Predicate>
FLARE_FUNCTION bool all_of_team_impl(const TeamHandleType& teamHandle,
                                      InputIterator first, InputIterator last,
                                      Predicate predicate) {
  return (find_if_or_not_team_impl<false>(teamHandle, first, last, predicate) ==
          last);
}

template <class TeamHandleType, class InputIterator, class Predicate>
FLARE_FUNCTION bool any_of_team_impl(const TeamHandleType& teamHandle,
                                      InputIterator first, InputIterator last,
                                      Predicate predicate) {
  return (find_if_or_not_team_impl<true>(teamHandle, first, last, predicate) !=
          last);
}

template <class TeamHandleType, class IteratorType, class Predicate>
FLARE_FUNCTION bool none_of_team_impl(const TeamHandleType& teamHandle,
                                       IteratorType first, IteratorType last,
                                       Predicate predicate) {
  return (find_if_or_not_team_impl<true>(teamHandle, first, last, predicate) ==
          last);
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ALL_OF_ANY_OF_NONE_OF_IMPL_H_
