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

#ifndef FLARE_ALGORITHM_SWAP_H_
#define FLARE_ALGORITHM_SWAP_H_

#include <flare/core.h>

namespace flare {
namespace experimental {

// swap
template <class T>
FLARE_INLINE_FUNCTION void swap(T& a, T& b) noexcept {
  static_assert(
      std::is_move_assignable<T>::value && std::is_move_constructible<T>::value,
      "flare::experimental::swap arguments must be move assignable "
      "and move constructible");

  T tmp = std::move(a);
  a     = std::move(b);
  b     = std::move(tmp);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_SWAP_H_
