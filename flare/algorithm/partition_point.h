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

#ifndef FLARE_ALGORITHM_PARTITION_POINT_H_
#define FLARE_ALGORITHM_PARTITION_POINT_H_

#include <flare/algorithm/partition_point_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class IteratorType, class UnaryPredicate>
IteratorType partition_point(const ExecutionSpace& ex, IteratorType first,
                             IteratorType last, UnaryPredicate p) {
  return detail::partition_point_impl(
      "flare::partitioned_point_iterator_api_default", ex, first, last,
      std::move(p));
}

template <class ExecutionSpace, class IteratorType, class UnaryPredicate>
IteratorType partition_point(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, IteratorType last,
                             UnaryPredicate p) {
  return detail::partition_point_impl(label, ex, first, last, std::move(p));
}

template <class ExecutionSpace, class UnaryPredicate, class DataType,
          class... Properties>
auto partition_point(const std::string& label, const ExecutionSpace& ex,
                     const ::flare::View<DataType, Properties...>& v,
                     UnaryPredicate p) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);
  return detail::partition_point_impl(label, ex, begin(v), end(v), std::move(p));
}

template <class ExecutionSpace, class UnaryPredicate, class DataType,
          class... Properties>
auto partition_point(const ExecutionSpace& ex,
                     const ::flare::View<DataType, Properties...>& v,
                     UnaryPredicate p) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);
  return detail::partition_point_impl("flare::partition_point_view_api_default",
                                    ex, begin(v), end(v), std::move(p));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_PARTITION_POINT_H_
