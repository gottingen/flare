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

#ifndef FLARE_ALGORITHM_REMOVE_IF_H_
#define FLARE_ALGORITHM_REMOVE_IF_H_

#include <flare/algorithm/remove_all_variants_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class Iterator, class UnaryPredicate>
Iterator remove_if(const ExecutionSpace& ex, Iterator first, Iterator last,
                   UnaryPredicate pred) {
  return detail::remove_if_impl("flare::remove_if_iterator_api_default", ex,
                              first, last, pred);
}

template <class ExecutionSpace, class Iterator, class UnaryPredicate>
Iterator remove_if(const std::string& label, const ExecutionSpace& ex,
                   Iterator first, Iterator last, UnaryPredicate pred) {
  return detail::remove_if_impl(label, ex, first, last, pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class UnaryPredicate>
auto remove_if(const ExecutionSpace& ex,
               const ::flare::View<DataType, Properties...>& view,
               UnaryPredicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);

  return detail::remove_if_impl("flare::remove_if_iterator_api_default", ex,
                              ::flare::experimental::begin(view),
                              ::flare::experimental::end(view), pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class UnaryPredicate>
auto remove_if(const std::string& label, const ExecutionSpace& ex,
               const ::flare::View<DataType, Properties...>& view,
               UnaryPredicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::remove_if_impl(label, ex, ::flare::experimental::begin(view),
                              ::flare::experimental::end(view), pred);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REMOVE_IF_H_
