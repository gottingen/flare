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

#ifndef FLARE_ALGORITHM_REMOVE_COPY_IF_H_
#define FLARE_ALGORITHM_REMOVE_COPY_IF_H_

#include <flare/algorithm/remove_all_variants_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class UnaryPredicate>
OutputIterator remove_copy_if(const ExecutionSpace& ex,
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_dest,
                              const UnaryPredicate& pred) {
  return detail::remove_copy_if_impl(
      "flare::remove_copy_if_iterator_api_default", ex, first_from, last_from,
      first_dest, pred);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class UnaryPredicate>
OutputIterator remove_copy_if(const std::string& label,
                              const ExecutionSpace& ex,
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_dest,
                              const UnaryPredicate& pred) {
  return detail::remove_copy_if_impl(label, ex, first_from, last_from, first_dest,
                                   pred);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class UnaryPredicate>
auto remove_copy_if(const ExecutionSpace& ex,
                    const ::flare::View<DataType1, Properties1...>& view_from,
                    const ::flare::View<DataType2, Properties2...>& view_dest,
                    const UnaryPredicate& pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);

  return detail::remove_copy_if_impl(
      "flare::remove_copy_if_iterator_api_default", ex,
      ::flare::experimental::cbegin(view_from),
      ::flare::experimental::cend(view_from),
      ::flare::experimental::begin(view_dest), pred);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class UnaryPredicate>
auto remove_copy_if(const std::string& label, const ExecutionSpace& ex,
                    const ::flare::View<DataType1, Properties1...>& view_from,
                    const ::flare::View<DataType2, Properties2...>& view_dest,
                    const UnaryPredicate& pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_from);
  detail::static_assert_is_admissible_to_flare_std_algorithms(view_dest);

  return detail::remove_copy_if_impl(
      label, ex, ::flare::experimental::cbegin(view_from),
      ::flare::experimental::cend(view_from),
      ::flare::experimental::begin(view_dest), pred);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REMOVE_COPY_IF_H_
