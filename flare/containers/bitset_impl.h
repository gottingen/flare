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

#ifndef FLARE_CONTAINERS_BITSET_IMPL_H_
#define FLARE_CONTAINERS_BITSET_IMPL_H_

#include <flare/core/defines.h>
#include <flare/core/common/bit_ops.h>
#include <cstdint>

#include <cstdio>
#include <climits>
#include <iostream>
#include <iomanip>

namespace flare {
namespace detail {

FLARE_FORCEINLINE_FUNCTION
unsigned rotate_right(unsigned i, int r) {
  constexpr int size = static_cast<int>(sizeof(unsigned) * CHAR_BIT);
  return r ? ((i >> r) | (i << (size - r))) : i;
}

template <typename Bitset>
struct BitsetCount {
  using bitset_type = Bitset;
  using execution_space =
      typename bitset_type::execution_space::execution_space;
  using size_type  = typename bitset_type::size_type;
  using value_type = size_type;

  bitset_type m_bitset;

  BitsetCount(bitset_type const& bitset) : m_bitset(bitset) {}

  size_type apply() const {
    size_type count = 0u;
    parallel_reduce("flare::detail::BitsetCount::apply",
                    m_bitset.m_blocks.extent(0), *this, count);
    return count;
  }

  FLARE_INLINE_FUNCTION
  void init(value_type& count) const { count = 0u; }

  FLARE_INLINE_FUNCTION
  void join(value_type& count, const size_type& incr) const { count += incr; }

  FLARE_INLINE_FUNCTION
  void operator()(size_type i, value_type& count) const {
    count += bit_count(m_bitset.m_blocks[i]);
  }
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CONTAINERS_BITSET_IMPL_H_
