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

#ifndef MEMPOOL_TEST_H_
#define MEMPOOL_TEST_H_

#include <flare/core.h>

namespace TestMemoryPool {

    template<typename MemSpace = flare::HostSpace>
    void test_host_memory_pool_defaults() {
        using Space = typename MemSpace::execution_space;
        using MemPool = typename flare::MemoryPool<Space>;

        {
            const size_t MemoryCapacity = 32000;
            const size_t MinBlockSize = 64;
            const size_t MaxBlockSize = 1024;
            const size_t SuperBlockSize = 4096;

            MemPool pool(MemSpace(), MemoryCapacity, MinBlockSize, MaxBlockSize,
                         SuperBlockSize);

            typename MemPool::usage_statistics stats;

            pool.get_usage_statistics(stats);

            REQUIRE_LE(MemoryCapacity, stats.capacity_bytes);
            REQUIRE_LE(MinBlockSize, stats.min_block_bytes);
            REQUIRE_LE(MaxBlockSize, stats.max_block_bytes);
            REQUIRE_LE(SuperBlockSize, stats.superblock_bytes);
        }

        {
            const size_t MemoryCapacity = 10000;

            MemPool pool(MemSpace(), MemoryCapacity);

            typename MemPool::usage_statistics stats;

            pool.get_usage_statistics(stats);

            REQUIRE_LE(MemoryCapacity, stats.capacity_bytes);
            REQUIRE_LE(64u /* default */, stats.min_block_bytes);
            REQUIRE_LE(stats.min_block_bytes, stats.max_block_bytes);
            REQUIRE_LE(stats.max_block_bytes, stats.superblock_bytes);
            REQUIRE_LE(stats.superblock_bytes, stats.capacity_bytes);
        }

        {
            const size_t MemoryCapacity = 10000;
            const size_t MinBlockSize = 32;  // power of two is exact

            MemPool pool(MemSpace(), MemoryCapacity, MinBlockSize);

            typename MemPool::usage_statistics stats;

            pool.get_usage_statistics(stats);

            REQUIRE_LE(MemoryCapacity, stats.capacity_bytes);
            REQUIRE_EQ(MinBlockSize, stats.min_block_bytes);
            REQUIRE_LE(stats.min_block_bytes, stats.max_block_bytes);
            REQUIRE_LE(stats.max_block_bytes, stats.superblock_bytes);
            REQUIRE_LE(stats.superblock_bytes, stats.capacity_bytes);
        }

        {
            const size_t MemoryCapacity = 32000;
            const size_t MinBlockSize = 32;    // power of two is exact
            const size_t MaxBlockSize = 1024;  // power of two is exact

            MemPool pool(MemSpace(), MemoryCapacity, MinBlockSize, MaxBlockSize);

            typename MemPool::usage_statistics stats;

            pool.get_usage_statistics(stats);

            REQUIRE_LE(MemoryCapacity, stats.capacity_bytes);
            REQUIRE_EQ(MinBlockSize, stats.min_block_bytes);
            REQUIRE_EQ(MaxBlockSize, stats.max_block_bytes);
            REQUIRE_LE(stats.max_block_bytes, stats.superblock_bytes);
            REQUIRE_LE(stats.superblock_bytes, stats.capacity_bytes);
        }
    }

    template<typename MemSpace = flare::HostSpace>
    void test_host_memory_pool_stats() {
        using Space = typename MemSpace::execution_space;
        using MemPool = typename flare::MemoryPool<Space>;

        const size_t MemoryCapacity = 32000;
        const size_t MinBlockSize = 64;
        const size_t MaxBlockSize = 1024;
        const size_t SuperBlockSize = 4096;

        MemPool pool(MemSpace(), MemoryCapacity, MinBlockSize, MaxBlockSize,
                     SuperBlockSize);

        {
            typename MemPool::usage_statistics stats;

            pool.get_usage_statistics(stats);

            REQUIRE_LE(MemoryCapacity, stats.capacity_bytes);
            REQUIRE_LE(MinBlockSize, stats.min_block_bytes);
            REQUIRE_LE(MaxBlockSize, stats.max_block_bytes);
            REQUIRE_LE(SuperBlockSize, stats.superblock_bytes);
        }

        void *p0064 = pool.allocate(64);
        void *p0128 = pool.allocate(128);
        void *p0256 = pool.allocate(256);
        void *p1024 = pool.allocate(1024);

        // Aborts because exceeds max block size:
        // void * p2048 = pool.allocate(2048);

        REQUIRE_NE(p0064, nullptr);
        REQUIRE_NE(p0128, nullptr);
        REQUIRE_NE(p0256, nullptr);
        REQUIRE_NE(p1024, nullptr);

        pool.deallocate(p0064, 64);
        pool.deallocate(p0128, 128);
        pool.deallocate(p0256, 256);
        pool.deallocate(p1024, 1024);
    }

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

    template<class DeviceType>
    struct TestMemoryPool_Functor {
        using ptrs_type = flare::View<uintptr_t *, DeviceType>;
        using pool_type = flare::MemoryPool<DeviceType>;

        pool_type pool;
        ptrs_type ptrs;

        TestMemoryPool_Functor(const pool_type &arg_pool, size_t n)
                : pool(arg_pool), ptrs("ptrs", n) {}

        // Specify reduction argument value_type to avoid
        // confusion with tag-dispatch.

        using value_type = long;

        struct TagAlloc {
        };

        FLARE_INLINE_FUNCTION
        void operator()(TagAlloc, int i, long &update) const noexcept {
            unsigned alloc_size = 32 * (1 + (i % 5));
            ptrs(i) = (uintptr_t) pool.allocate(alloc_size);
            if (ptrs(i)) {
                ++update;
            }
        }

        struct TagDealloc {
        };

        FLARE_INLINE_FUNCTION
        void operator()(TagDealloc, int i, long &update) const noexcept {
            if (ptrs(i) && (0 == i % 3)) {
                unsigned alloc_size = 32 * (1 + (i % 5));
                pool.deallocate((void *) ptrs(i), alloc_size);
                ptrs(i) = 0;
                ++update;
            }
        }

        struct TagRealloc {
        };

        FLARE_INLINE_FUNCTION
        void operator()(TagRealloc, int i, long &update) const noexcept {
            if (0 == ptrs(i)) {
                unsigned alloc_size = 32 * (1 + (i % 5));
                ptrs(i) = (uintptr_t) pool.allocate(alloc_size);
                if (ptrs(i)) {
                    ++update;
                }
            }
        }

        struct TagMixItUp {
        };

        FLARE_INLINE_FUNCTION
        void operator()(TagMixItUp, int i, long &update) const noexcept {
            if (ptrs(i) && (0 == i % 3)) {
                unsigned alloc_size = 32 * (1 + (i % 5));

                pool.deallocate((void *) ptrs(i), alloc_size);

                ptrs(i) = (uintptr_t) pool.allocate(alloc_size);

                if (ptrs(i)) {
                    ++update;
                }
            }
        }
    };

    template<class PoolType>
    void print_memory_pool_stats(typename PoolType::usage_statistics const &stats) {
        std::cout << "MemoryPool {" << std::endl
                  << "  bytes capacity = " << stats.capacity_bytes << std::endl
                  << "  bytes used     = " << stats.consumed_bytes << std::endl
                  << "  bytes reserved = " << stats.reserved_bytes << std::endl
                  << "  bytes free     = "
                  << (stats.capacity_bytes -
                      (stats.consumed_bytes + stats.reserved_bytes))
                  << std::endl
                  << "  block used     = " << stats.consumed_blocks << std::endl
                  << "  block reserved = " << stats.reserved_blocks << std::endl
                  << "  super used     = " << stats.consumed_superblocks << std::endl
                  << "  super reserved = "
                  << (stats.capacity_superblocks - stats.consumed_superblocks)
                  << std::endl
                  << "}" << std::endl;
    }

    template<class DeviceType>
    void test_memory_pool_v2(const bool print_statistics,
                             const bool print_superblocks) {
        using memory_space = typename DeviceType::memory_space;
        using execution_space = typename DeviceType::execution_space;
        using pool_type = flare::MemoryPool<DeviceType>;
        using functor_type = TestMemoryPool_Functor<DeviceType>;

        using TagAlloc = typename functor_type::TagAlloc;
        using TagDealloc = typename functor_type::TagDealloc;
        using TagRealloc = typename functor_type::TagRealloc;
        using TagMixItUp = typename functor_type::TagMixItUp;

        const size_t total_alloc_size = 10000000;
        const unsigned min_block_size = 64;
        const unsigned max_block_size = 256;
        const long nfill = 70000;

        for (uint32_t k = 0, min_superblock_size = 10000; k < 3;
             ++k, min_superblock_size *= 10) {
            typename pool_type::usage_statistics stats;

            pool_type pool(memory_space(), total_alloc_size, min_block_size,
                           max_block_size, min_superblock_size);

            functor_type functor(pool, nfill);

            long result = 0;
            long ndel = 0;

            flare::parallel_reduce(
                    flare::RangePolicy<execution_space, TagAlloc>(0, nfill), functor,
                    result);

            pool.get_usage_statistics(stats);

            const int fill_error =
                    (nfill != result) || (nfill != long(stats.consumed_blocks));

            if (fill_error || print_statistics)
                print_memory_pool_stats<pool_type>(stats);
            if (fill_error || print_superblocks) pool.print_state(std::cout);

            REQUIRE_EQ(nfill, result);
            REQUIRE_EQ(nfill, long(stats.consumed_blocks));

            flare::parallel_reduce(
                    flare::RangePolicy<execution_space, TagDealloc>(0, nfill), functor,
                    ndel);

            pool.get_usage_statistics(stats);

            const int del_error = (nfill - ndel) != long(stats.consumed_blocks);

            if (del_error || print_statistics)
                print_memory_pool_stats<pool_type>(stats);
            if (del_error || print_superblocks) pool.print_state(std::cout);

            REQUIRE_EQ((nfill - ndel), long(stats.consumed_blocks));

            flare::parallel_reduce(
                    flare::RangePolicy<execution_space, TagRealloc>(0, nfill), functor,
                    result);

            pool.get_usage_statistics(stats);

            const int refill_error =
                    (ndel != result) || (nfill != long(stats.consumed_blocks));

            if (refill_error || print_statistics)
                print_memory_pool_stats<pool_type>(stats);
            if (refill_error || print_superblocks) pool.print_state(std::cout);

            REQUIRE_EQ(ndel, result);
            REQUIRE_EQ(nfill, long(stats.consumed_blocks));

            flare::parallel_reduce(
                    flare::RangePolicy<execution_space, TagMixItUp>(0, nfill), functor,
                    result);

            pool.get_usage_statistics(stats);

            const int mix_error =
                    (ndel != result) || (nfill != long(stats.consumed_blocks));

            if (mix_error || print_statistics)
                print_memory_pool_stats<pool_type>(stats);
            if (mix_error || print_superblocks) pool.print_state(std::cout);

            REQUIRE_EQ(ndel, result);
            REQUIRE_EQ(nfill, long(stats.consumed_blocks));
        }
    }

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

    template<class DeviceType>
    struct TestMemoryPoolCorners {
        using ptrs_type = flare::View<uintptr_t *, DeviceType>;
        using pool_type = flare::MemoryPool<DeviceType>;

        pool_type pool;
        ptrs_type ptrs;
        uint32_t size;
        uint32_t stride;

        TestMemoryPoolCorners(const pool_type &arg_pool, const ptrs_type &arg_ptrs,
                              const uint32_t arg_base, const uint32_t arg_stride)
                : pool(arg_pool), ptrs(arg_ptrs), size(arg_base), stride(arg_stride) {}

        // Specify reduction argument value_type to
        // avoid confusion with tag-dispatch.

        using value_type = long;

        FLARE_INLINE_FUNCTION
        void operator()(int i, long &err) const noexcept {
            unsigned alloc_size = size << (i % stride);
            if (0 == ptrs(i)) {
                ptrs(i) = (uintptr_t) pool.allocate(alloc_size);
                if (ptrs(i) && !alloc_size) {
                    ++err;
                }
            }
        }

        struct TagDealloc {
        };

        FLARE_INLINE_FUNCTION
        void operator()(int i) const noexcept {
            unsigned alloc_size = size << (i % stride);
            if (ptrs(i)) {
                pool.deallocate((void *) ptrs(i), alloc_size);
            }
            ptrs(i) = 0;
        }
    };

    template<class DeviceType>
    void test_memory_pool_corners(const bool print_statistics,
                                  const bool print_superblocks) {
        using memory_space = typename DeviceType::memory_space;
        using execution_space = typename DeviceType::execution_space;
        using pool_type = flare::MemoryPool<DeviceType>;
        using functor_type = TestMemoryPoolCorners<DeviceType>;
        using ptrs_type = typename functor_type::ptrs_type;

        {
            // superblock size 1 << 14
            const size_t min_superblock_size = 1u << 14;

            // four superblocks
            const size_t total_alloc_size = min_superblock_size * 4;

            // block sizes  {  64 , 128 , 256 , 512 }
            // block counts { 256 , 128 ,  64 ,  32 }
            const unsigned min_block_size = 64;
            const unsigned max_block_size = 512;
            const unsigned num_blocks = 480;

            pool_type pool(memory_space(), total_alloc_size, min_block_size,
                           max_block_size, min_superblock_size);

            // Allocate one block from each superblock to lock that
            // superblock into the block size.

            ptrs_type ptrs("ptrs", num_blocks);

            long err = 0;

            flare::parallel_reduce(flare::RangePolicy<execution_space>(0, 4),
                                   functor_type(pool, ptrs, 64, 4), err);

            if (print_statistics || err) {
                typename pool_type::usage_statistics stats;

                pool.get_usage_statistics(stats);

                print_memory_pool_stats<pool_type>(stats);
            }

            if (print_superblocks || err) {
                pool.print_state(std::cout);
            }

            // Now fill remaining allocations with small size

            flare::parallel_reduce(flare::RangePolicy<execution_space>(0, num_blocks),
                                   functor_type(pool, ptrs, 64, 1), err);

            if (print_statistics || err) {
                typename pool_type::usage_statistics stats;

                pool.get_usage_statistics(stats);

                print_memory_pool_stats<pool_type>(stats);
            }

            if (print_superblocks || err) {
                pool.print_state(std::cout);
            }
        }
    }

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

    template<class DeviceType, class Enable = void>
    struct TestMemoryPoolHuge {
        enum : size_t {
            num_superblock = 0
        };

        using value_type = long;

        FLARE_INLINE_FUNCTION
        void operator()(int /*i*/, long & /*err*/) const noexcept {}

        FLARE_INLINE_FUNCTION
        void operator()(int /*i*/) const noexcept {}
    };

    template<class DeviceType>
    struct TestMemoryPoolHuge<
            DeviceType,
            std::enable_if_t<std::is_same<flare::HostSpace,
                    typename DeviceType::memory_space>::value>> {
        using ptrs_type = flare::View<uintptr_t *, DeviceType>;
        using pool_type = flare::MemoryPool<DeviceType>;
        using memory_space = typename DeviceType::memory_space;

        pool_type pool;
        ptrs_type ptrs;

        enum : size_t {
            min_block_size = 512,
            max_block_size = 1lu << 31,
            min_superblock_size = max_block_size,
            num_superblock = 4,
            total_alloc_size = num_superblock * max_block_size
        };

        TestMemoryPoolHuge()
                : pool(memory_space(), total_alloc_size, min_block_size, max_block_size,
                       min_superblock_size),
                  ptrs("ptrs", num_superblock) {}

        // Specify reduction argument value_type to
        // avoid confusion with tag-dispatch.

        using value_type = long;

        void operator()(int i, long &err) const noexcept {
            if (i < int(num_superblock)) {
                ptrs(i) = (uintptr_t) pool.allocate(max_block_size);
#if 0
                printf("TestMemoryPoolHuge size(0x%lx) ptr(0x%lx)\n"
                      , max_block_size
                      , ptrs(i) );
#endif
                if (!ptrs(i)) {
                    flare::abort("TestMemoryPoolHuge");
                    ++err;
                }
            }
        }

        void operator()(int i) const noexcept {
            if (i < int(num_superblock)) {
                pool.deallocate((void *) ptrs(i), max_block_size);
                ptrs(i) = 0;
            }
        }
    };

    template<class DeviceType>
    void test_memory_pool_huge() {
        using execution_space = typename DeviceType::execution_space;
        using functor_type = TestMemoryPoolHuge<DeviceType>;
        using policy_type = flare::RangePolicy<execution_space>;

        functor_type f;
        policy_type policy(0, functor_type::num_superblock);

        long err = 0;

        flare::parallel_reduce(policy, f, err);
        flare::parallel_for(policy, f);
    }

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

}  // namespace TestMemoryPool

namespace Test {

    TEST_CASE("TEST_CATEGORY, memory_pool") {
        TestMemoryPool::test_host_memory_pool_defaults<>();
        TestMemoryPool::test_host_memory_pool_stats<>();
        TestMemoryPool::test_memory_pool_v2<TEST_EXECSPACE>(false, false);
        TestMemoryPool::test_memory_pool_corners<TEST_EXECSPACE>(false, false);
#ifdef FLARE_ENABLE_LARGE_MEM_TESTS
        TestMemoryPool::test_memory_pool_huge<TEST_EXECSPACE>();
#endif
    }

}  // namespace Test

#endif  // MEMPOOL_TEST_H_
