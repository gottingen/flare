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

#ifndef FLARE_ALGORITHM_DISTANCE_H_
#define FLARE_ALGORITHM_DISTANCE_H_

#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/random_access_iterator_impl.h>

namespace flare {
namespace experimental {

template <class IteratorType>
FLARE_INLINE_FUNCTION constexpr typename IteratorType::difference_type
distance(IteratorType first, IteratorType last) {
  static_assert(
      ::flare::experimental::detail::are_random_access_iterators<
          IteratorType>::value,
      "flare::experimental::distance: only implemented for random access "
      "iterators.");

  return last - first;
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_DISTANCE_H_
