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


#include <flare/core/common/error.h>
#include <flare/core/memory/memory_pool.h>
#include <ostream>
#include <sstream>
#include <cstdint>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

/* Verify size constraints:
 *   min_block_alloc_size <= max_block_alloc_size
 *   max_block_alloc_size <= min_superblock_size
 *   min_superblock_size  <= max_superblock_size
 *   min_superblock_size  <= min_total_alloc_size
 *   min_superblock_size  <= min_block_alloc_size *
 *                           max_block_per_superblock
 */
void memory_pool_bounds_verification(size_t min_block_alloc_size,
                                     size_t max_block_alloc_size,
                                     size_t min_superblock_size,
                                     size_t max_superblock_size,
                                     size_t max_block_per_superblock,
                                     size_t min_total_alloc_size) {
  const size_t max_superblock = min_block_alloc_size * max_block_per_superblock;

  if ((size_t(max_superblock_size) < min_superblock_size) ||
      (min_total_alloc_size < min_superblock_size) ||
      (max_superblock < min_superblock_size) ||
      (min_superblock_size < max_block_alloc_size) ||
      (max_block_alloc_size < min_block_alloc_size)) {
    std::ostringstream msg;

    msg << "flare::MemoryPool size constraint violation";

    if (size_t(max_superblock_size) < min_superblock_size) {
      msg << " : max_superblock_size(" << max_superblock_size
          << ") < min_superblock_size(" << min_superblock_size << ")";
    }

    if (min_total_alloc_size < min_superblock_size) {
      msg << " : min_total_alloc_size(" << min_total_alloc_size
          << ") < min_superblock_size(" << min_superblock_size << ")";
    }

    if (max_superblock < min_superblock_size) {
      msg << " : max_superblock(" << max_superblock
          << ") < min_superblock_size(" << min_superblock_size << ")";
    }

    if (min_superblock_size < max_block_alloc_size) {
      msg << " : min_superblock_size(" << min_superblock_size
          << ") < max_block_alloc_size(" << max_block_alloc_size << ")";
    }

    if (max_block_alloc_size < min_block_alloc_size) {
      msg << " : max_block_alloc_size(" << max_block_alloc_size
          << ") < min_block_alloc_size(" << min_block_alloc_size << ")";
    }

    flare::detail::throw_runtime_exception(msg.str());
  }
}

// This has way too many parameters, but it is entirely for moving the iostream
// inclusion out of the header file with as few changes as possible
void _print_memory_pool_state(std::ostream& s, uint32_t const* sb_state_ptr,
                              int32_t sb_count, uint32_t sb_size_lg2,
                              uint32_t sb_state_size, uint32_t state_shift,
                              uint32_t state_used_mask) {
  s << "pool_size(" << (size_t(sb_count) << sb_size_lg2) << ")"
    << " superblock_size(" << (1LU << sb_size_lg2) << ")" << std::endl;

  for (int32_t i = 0; i < sb_count; ++i, sb_state_ptr += sb_state_size) {
    if (*sb_state_ptr) {
      const uint32_t block_count_lg2 = (*sb_state_ptr) >> state_shift;
      const uint32_t block_size_lg2  = sb_size_lg2 - block_count_lg2;
      const uint32_t block_count     = 1u << block_count_lg2;
      const uint32_t block_used      = (*sb_state_ptr) & state_used_mask;

      s << "Superblock[ " << i << " / " << sb_count << " ] {"
        << " block_size(" << (1 << block_size_lg2) << ")"
        << " block_count( " << block_used << " / " << block_count << " )"
        << std::endl;
    }
  }
}

}  // namespace detail
}  // namespace flare
