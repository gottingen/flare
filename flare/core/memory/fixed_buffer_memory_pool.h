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

#ifndef FLARE_CORE_MEMORY_FIXED_BUFFER_MEMORY_POOL_H_
#define FLARE_CORE_MEMORY_FIXED_BUFFER_MEMORY_POOL_H_

#include <flare/core_fwd.h>
#include <flare/core/atomic.h>

#include <flare/core/memory/pointer_ownership.h>
#include <flare/core/task/simple_task_scheduler.h>

namespace flare {
namespace detail {

template <class DeviceType, size_t Size, size_t Align = 1,
          class SizeType = typename DeviceType::execution_space::size_type>
class FixedBlockSizeMemoryPool
    : private MemorySpaceInstanceStorage<typename DeviceType::memory_space> {
 public:
  using memory_space = typename DeviceType::memory_space;
  using size_type    = SizeType;

 private:
  using memory_space_storage_base =
      MemorySpaceInstanceStorage<typename DeviceType::memory_space>;
  using tracker_type = flare::detail::SharedAllocationTracker;
  using record_type  = flare::detail::SharedAllocationRecord<memory_space>;

  struct alignas(Align) Block {
    union {
      char ignore;
      char data[Size];
    };
  };

  static constexpr auto actual_size = sizeof(Block);

  // TODO shared allocation tracker
  // TODO @optimization put the index values on different cache lines (CPU) or
  // pages (GPU)?

  tracker_type m_tracker                         = {};
  size_type m_num_blocks                         = 0;
  size_type m_first_free_idx                     = 0;
  size_type m_last_free_idx                      = 0;
  flare::OwningRawPtr<Block> m_first_block      = nullptr;
  flare::OwningRawPtr<size_type> m_free_indices = nullptr;

  enum : size_type { IndexInUse = ~size_type(0) };

 public:
  FixedBlockSizeMemoryPool(memory_space const& mem_space, size_type num_blocks)
      : memory_space_storage_base(mem_space),
        m_tracker(),
        m_num_blocks(num_blocks),
        m_first_free_idx(0),
        m_last_free_idx(num_blocks) {
    // TODO alignment?
    auto block_record = record_type::allocate(
        mem_space, "FixedBlockSizeMemPool_blocks", num_blocks * sizeof(Block));
    FLARE_ASSERT(intptr_t(block_record->data()) % Align == 0);
    m_tracker.assign_allocated_record_to_uninitialized(block_record);
    m_first_block = (Block*)block_record->data();

    auto idx_record =
        record_type::allocate(mem_space, "flare::FixedBlockSizeMemPool_blocks",
                              num_blocks * sizeof(size_type));
    FLARE_ASSERT(intptr_t(idx_record->data()) % alignof(size_type) == 0);
    m_tracker.assign_allocated_record_to_uninitialized(idx_record);
    m_free_indices = (size_type*)idx_record->data();

    for (size_type i = 0; i < num_blocks; ++i) {
      m_free_indices[i] = i;
    }

    flare::memory_fence();
  }

  // For compatibility with MemoryPool<>
  FixedBlockSizeMemoryPool(memory_space const& mem_space,
                           size_t mempool_capacity, unsigned, unsigned,
                           unsigned)
      : FixedBlockSizeMemoryPool(
            mem_space, mempool_capacity /
                           actual_size) { /* forwarding ctor, must be empty */
  }

  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool() = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool(
      FixedBlockSizeMemoryPool&&) = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool(
      FixedBlockSizeMemoryPool const&)                        = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool& operator=(
      FixedBlockSizeMemoryPool&&) = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool& operator=(
      FixedBlockSizeMemoryPool const&) = default;

  FLARE_INLINE_FUNCTION
  void* allocate(size_type alloc_size) const noexcept {
    (void)alloc_size;
    FLARE_EXPECTS(alloc_size <= Size);
    auto free_idx_counter = flare::atomic_fetch_add(
        (volatile size_type*)&m_first_free_idx, size_type(1));
    auto free_idx_idx = free_idx_counter % m_num_blocks;

    // We don't have exclusive access to m_free_indices[free_idx_idx] because
    // the allocate counter might have lapped us since we incremented it
    auto current_free_idx = m_free_indices[free_idx_idx];
    size_type free_idx    = IndexInUse;
    free_idx = flare::atomic_compare_exchange(&m_free_indices[free_idx_idx],
                                               current_free_idx, free_idx);
    flare::memory_fence();

    // TODO figure out how to decrement here?

    if (free_idx == IndexInUse) {
      return nullptr;
    } else {
      return (void*)&m_first_block[free_idx];
    }
  }

  FLARE_INLINE_FUNCTION
  void deallocate(void* ptr, size_type /*alloc_size*/) const noexcept {
    // figure out which block we are
    auto offset = intptr_t(ptr) - intptr_t(m_first_block);

    FLARE_EXPECTS(offset % actual_size == 0 &&
                   offset / actual_size < m_num_blocks);

    flare::memory_fence();
    auto last_idx_idx = flare::atomic_fetch_add(
        (volatile size_type*)&m_last_free_idx, size_type(1));
    last_idx_idx %= m_num_blocks;
    m_free_indices[last_idx_idx] = offset / actual_size;
  }
};

#if 0
template <
  class DeviceType,
  size_t Size,
  size_t Align=1,
  class SizeType = typename DeviceType::execution_space::size_type
>
class FixedBlockSizeChaseLevMemoryPool
  : private MemorySpaceInstanceStorage<typename DeviceType::memory_space>
{
public:

  using memory_space = typename DeviceType::memory_space;
  using size_type = SizeType;

private:

  using memory_space_storage_base = MemorySpaceInstanceStorage<typename DeviceType::memory_space>;
  using tracker_type = flare::detail::SharedAllocationTracker;
  using record_type = flare::detail::SharedAllocationRecord<memory_space>;

  struct alignas(Align) Block { union { char ignore; char data[Size]; }; };

  static constexpr auto actual_size = sizeof(Block);

  tracker_type m_tracker = { };
  size_type m_num_blocks = 0;
  size_type m_first_free_idx = 0;
  size_type m_last_free_idx = 0;


  enum : size_type { IndexInUse = ~size_type(0) };

public:

  FixedBlockSizeMemoryPool(
    memory_space const& mem_space,
    size_type num_blocks
  ) : memory_space_storage_base(mem_space),
    m_tracker(),
    m_num_blocks(num_blocks),
    m_first_free_idx(0),
    m_last_free_idx(num_blocks)
  {
    // TODO alignment?
    auto block_record = record_type::allocate(
      mem_space, "FixedBlockSizeMemPool_blocks", num_blocks * sizeof(Block)
    );
    FLARE_ASSERT(intptr_t(block_record->data()) % Align == 0);
    m_tracker.assign_allocated_record_to_uninitialized(block_record);
    m_first_block = (Block*)block_record->data();

    auto idx_record = record_type::allocate(
      mem_space, "FixedBlockSizeMemPool_blocks", num_blocks * sizeof(size_type)
    );
    FLARE_ASSERT(intptr_t(idx_record->data()) % alignof(size_type) == 0);
    m_tracker.assign_allocated_record_to_uninitialized(idx_record);
    m_free_indices = (size_type*)idx_record->data();

    for(size_type i = 0; i < num_blocks; ++i) {
      m_free_indices[i] = i;
    }

    flare::memory_fence();
  }

  // For compatibility with MemoryPool<>
  FixedBlockSizeMemoryPool(
    memory_space const& mem_space,
    size_t mempool_capacity,
    unsigned, unsigned, unsigned
  ) : FixedBlockSizeMemoryPool(mem_space, mempool_capacity / actual_size)
  { /* forwarding ctor, must be empty */ }

  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool() = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool(FixedBlockSizeMemoryPool&&) = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool(FixedBlockSizeMemoryPool const&) = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool& operator=(FixedBlockSizeMemoryPool&&) = default;
  FLARE_DEFAULTED_FUNCTION FixedBlockSizeMemoryPool& operator=(FixedBlockSizeMemoryPool const&) = default;


  FLARE_INLINE_FUNCTION
  void* allocate(size_type alloc_size) const noexcept
  {
    FLARE_EXPECTS(alloc_size <= Size);
    auto free_idx_counter = flare::atomic_fetch_add((volatile size_type*)&m_first_free_idx, size_type(1));
    auto free_idx_idx = free_idx_counter % m_num_blocks;

    // We don't have exclusive access to m_free_indices[free_idx_idx] because
    // the allocate counter might have lapped us since we incremented it
    auto current_free_idx = m_free_indices[free_idx_idx];
    size_type free_idx = IndexInUse;
    free_idx =
      flare::atomic_compare_exchange(&m_free_indices[free_idx_idx], current_free_idx, free_idx);
    flare::memory_fence();

    // TODO figure out how to decrement here?

    if(free_idx == IndexInUse) {
      return nullptr;
    }
    else {
      return (void*)&m_first_block[free_idx];
    }
  }

  FLARE_INLINE_FUNCTION
  void deallocate(void* ptr, size_type alloc_size) const noexcept
  {
    // figure out which block we are
    auto offset = intptr_t(ptr) - intptr_t(m_first_block);

    FLARE_EXPECTS(offset % actual_size == 0 && offset/actual_size < m_num_blocks);

    flare::memory_fence();
    auto last_idx_idx = flare::atomic_fetch_add((volatile size_type*)&m_last_free_idx, size_type(1));
    last_idx_idx %= m_num_blocks;
    m_free_indices[last_idx_idx] = offset / actual_size;
  }

};
#endif

}  // end namespace detail
}  // end namespace flare

#endif  // FLARE_CORE_MEMORY_FIXED_BUFFER_MEMORY_POOL_H_
