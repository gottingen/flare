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

#ifndef FLARE_ALGORITHM_ITER_SWAP_H_
#define FLARE_ALGORITHM_ITER_SWAP_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/swap.h>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType1, class IteratorType2>
struct StdIterSwapFunctor {
  IteratorType1 m_a;
  IteratorType2 m_b;

  FLARE_FUNCTION
  void operator()(int i) const {
    (void)i;
    ::flare::experimental::swap(*m_a, *m_b);
  }

  FLARE_FUNCTION
  StdIterSwapFunctor(IteratorType1 _a, IteratorType2 _b)
      : m_a(std::move(_a)), m_b(std::move(_b)) {}
};

template <class IteratorType1, class IteratorType2>
void iter_swap_impl(IteratorType1 a, IteratorType2 b) {
  // is there a better way to do this maybe?
  ::flare::parallel_for(
      1, StdIterSwapFunctor<IteratorType1, IteratorType2>(a, b));
  flare::DefaultExecutionSpace().fence(
      "flare::iter_swap: fence after operation");
}
}  // namespace detail
//----------------------------------------------------------------------------

// iter_swap
template <class IteratorType1, class IteratorType2>
void iter_swap(IteratorType1 a, IteratorType2 b) {
  detail::iter_swap_impl(a, b);
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_ITER_SWAP_H_
