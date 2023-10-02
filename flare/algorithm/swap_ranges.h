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

#ifndef FLARE_ALGORITHM_SWAP_RANGES_H_
#define FLARE_ALGORITHM_SWAP_RANGES_H_

#include <flare/algorithm/swap_ranges_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType2 swap_ranges(const ExecutionSpace& ex, IteratorType1 first1,
                          IteratorType1 last1, IteratorType2 first2) {
  return detail::swap_ranges_impl("flare::swap_ranges_iterator_api_default", ex,
                                first1, last1, first2);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto swap_ranges(const ExecutionSpace& ex,
                 const ::flare::View<DataType1, Properties1...>& source,
                 ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  assert(source.extent(0) == dest.extent(0));
  return detail::swap_ranges_impl("flare::swap_ranges_view_api_default", ex,
                                begin(source), end(source), begin(dest));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType2 swap_ranges(const std::string& label, const ExecutionSpace& ex,
                          IteratorType1 first1, IteratorType1 last1,
                          IteratorType2 first2) {
  return detail::swap_ranges_impl(label, ex, first1, last1, first2);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto swap_ranges(const std::string& label, const ExecutionSpace& ex,
                 const ::flare::View<DataType1, Properties1...>& source,
                 ::flare::View<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  assert(source.extent(0) == dest.extent(0));
  return detail::swap_ranges_impl(label, ex, begin(source), end(source),
                                begin(dest));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_SWAP_RANGES_H_
