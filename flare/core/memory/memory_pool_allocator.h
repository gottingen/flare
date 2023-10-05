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

// Experimental unified task-data parallel manycore LDRD

#ifndef FLARE_CORE_MEMORY_MEMORY_POOL_ALLOCATOR_H_
#define FLARE_CORE_MEMORY_MEMORY_POOL_ALLOCATOR_H_

#include <flare/core/defines.h>

#include <flare/core_fwd.h>

namespace flare::detail {

    template<class MemoryPool, class T>
    class MemoryPoolAllocator {
    public:
        using memory_pool = MemoryPool;

    private:
        memory_pool m_pool;

    public:
        FLARE_DEFAULTED_FUNCTION
        MemoryPoolAllocator() = default;

        FLARE_DEFAULTED_FUNCTION
        MemoryPoolAllocator(MemoryPoolAllocator const &) = default;

        FLARE_DEFAULTED_FUNCTION
        MemoryPoolAllocator(MemoryPoolAllocator &&) = default;

        FLARE_DEFAULTED_FUNCTION
        MemoryPoolAllocator &operator=(MemoryPoolAllocator const &) = default;

        FLARE_DEFAULTED_FUNCTION
        MemoryPoolAllocator &operator=(MemoryPoolAllocator &&) = default;

        FLARE_DEFAULTED_FUNCTION
        ~MemoryPoolAllocator() = default;

        FLARE_INLINE_FUNCTION
        explicit MemoryPoolAllocator(memory_pool const &arg_pool)
                : m_pool(arg_pool) {}

        FLARE_INLINE_FUNCTION
        explicit MemoryPoolAllocator(memory_pool &&arg_pool)
                : m_pool(std::move(arg_pool)) {}

    public:
        using value_type = T;
        using pointer = T *;
        using size_type = typename MemoryPool::memory_space::size_type;
        using difference_type = std::make_signed_t<size_type>;

        template<class U>
        struct rebind {
            using other = MemoryPoolAllocator<MemoryPool, U>;
        };

        FLARE_INLINE_FUNCTION
        pointer allocate(size_t n) {
            void *rv = m_pool.allocate(n * sizeof(T));
            if (rv == nullptr) {
                flare::abort("flare MemoryPool allocator failed to allocate memory");
            }
            return reinterpret_cast<T *>(rv);
        }

        FLARE_INLINE_FUNCTION
        void deallocate(T *ptr, size_t n) { m_pool.deallocate(ptr, n * sizeof(T)); }

        FLARE_INLINE_FUNCTION
        size_type max_size() const { return m_pool.max_block_size(); }

        FLARE_INLINE_FUNCTION
        bool operator==(MemoryPoolAllocator const &other) const {
            return m_pool == other.m_pool;
        }

        FLARE_INLINE_FUNCTION
        bool operator!=(MemoryPoolAllocator const &other) const {
            return !(*this == other);
        }
    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_MEMORY_MEMORY_POOL_ALLOCATOR_H_
