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

#ifndef FLARE_ALGORITHM_REMOVE_H_
#define FLARE_ALGORITHM_REMOVE_H_

#include <flare/algorithm/remove_all_variants_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class Iterator, class ValueType>
Iterator remove(const ExecutionSpace& ex, Iterator first, Iterator last,
                const ValueType& value) {
  return detail::remove_impl("flare::remove_iterator_api_default", ex, first,
                           last, value);
}

template <class ExecutionSpace, class Iterator, class ValueType>
Iterator remove(const std::string& label, const ExecutionSpace& ex,
                Iterator first, Iterator last, const ValueType& value) {
  return detail::remove_impl(label, ex, first, last, value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
auto remove(const ExecutionSpace& ex,
            const ::flare::View<DataType, Properties...>& view,
            const ValueType& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::remove_impl("flare::remove_iterator_api_default", ex,
                           ::flare::experimental::begin(view),
                           ::flare::experimental::end(view), value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
auto remove(const std::string& label, const ExecutionSpace& ex,
            const ::flare::View<DataType, Properties...>& view,
            const ValueType& value) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(view);
  return detail::remove_impl(label, ex, ::flare::experimental::begin(view),
                           ::flare::experimental::end(view), value);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REMOVE_H_
