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

#ifndef FLARE_CORE_MEMORY_SCRATCH_SPACE_H_
#define FLARE_CORE_MEMORY_SCRATCH_SPACE_H_

#include <cstdio>
#include <cstddef>
#include <flare/core_fwd.h>
#include <flare/core/common/concepts.h>

namespace flare {

    /** \brief  Scratch memory space associated with an execution space.
     *
     */
    template<class ExecSpace>
    class ScratchMemorySpace {
        static_assert(
                is_execution_space<ExecSpace>::value,
                "Instantiating ScratchMemorySpace on non-execution-space type.");

    public:
        // Minimal overalignment used by tensor scratch allocations
        constexpr static int ALIGN = 8;

    private:
        mutable char *m_iter_L0 = nullptr;
        mutable char *m_iter_L1 = nullptr;
        char *m_end_L0 = nullptr;
        char *m_end_L1 = nullptr;

        mutable int m_multiplier = 0;
        mutable int m_offset = 0;
        mutable int m_default_level = 0;

    public:
        //! Tag this class as a memory space
        using memory_space = ScratchMemorySpace<ExecSpace>;
        using execution_space = ExecSpace;
        //! This execution space preferred device_type
        using device_type = flare::Device<execution_space, memory_space>;

        using array_layout = typename ExecSpace::array_layout;
        using size_type = typename ExecSpace::size_type;

        static constexpr const char *name() { return "ScratchMemorySpace"; }


        template<typename IntType>
        FLARE_INLINE_FUNCTION void *get_shmem(const IntType &size,
                                              int level = -1) const {
            return get_shmem_common</*alignment_requested*/ false>(size, 1, level);
        }

        template<typename IntType>
        FLARE_INLINE_FUNCTION void *get_shmem_aligned(const IntType &size,
                                                      const ptrdiff_t alignment,
                                                      int level = -1) const {
            return get_shmem_common</*alignment_requested*/ true>(size, alignment,
                                                                  level);
        }

    private:
        template<bool alignment_requested, typename IntType>
        FLARE_INLINE_FUNCTION void *get_shmem_common(
                const IntType &size, [[maybe_unused]] const ptrdiff_t alignment,
                int level = -1) const {
            if (level == -1) level = m_default_level;
            auto &m_iter = (level == 0) ? m_iter_L0 : m_iter_L1;
            auto m_iter_old = m_iter;
            if constexpr (alignment_requested) {
                const ptrdiff_t missalign = size_t(m_iter) % alignment;
                if (missalign) m_iter += alignment - missalign;
            }

            // This is each thread's start pointer for its allocation
            // Note: for team scratch m_offset is 0, since every
            // thread will get back the same shared pointer
            void *tmp = m_iter + m_offset * size;
            uintptr_t increment = size * m_multiplier;

            const auto end_iter =
                    reinterpret_cast<uintptr_t>((level == 0) ? m_end_L0 : m_end_L1);
            auto current_iter = reinterpret_cast<uintptr_t>(m_iter);
            auto capacity = end_iter - current_iter;

            if (increment > capacity) {
                // Request did overflow: return nullptr and reset m_iter
                m_iter = m_iter_old;
                tmp = nullptr;
#ifdef FLARE_ENABLE_DEBUG
                // mfh 23 Jun 2015: printf call consumes 25 registers
                // in a CUDA build, so only print in debug mode.  The
                // function still returns nullptr if not enough memory.
                flare::printf(
                    "ScratchMemorySpace<...>::get_shmem: Failed to allocate "
                    "%ld byte(s); remaining capacity is %ld byte(s)\n",
                    long(size), long(capacity));
#endif  // FLARE_ENABLE_DEBUG
            } else {
                m_iter += increment;
            }
            return tmp;
        }

    public:
        FLARE_DEFAULTED_FUNCTION
        ScratchMemorySpace() = default;

        template<typename IntType>
        FLARE_INLINE_FUNCTION ScratchMemorySpace(void *ptr_L0,
                                                 const IntType &size_L0,
                                                 void *ptr_L1 = nullptr,
                                                 const IntType &size_L1 = 0)
                : m_iter_L0(static_cast<char *>(ptr_L0)),
                  m_iter_L1(static_cast<char *>(ptr_L1)),
                  m_end_L0(static_cast<char *>(ptr_L0) + size_L0),
                  m_end_L1(static_cast<char *>(ptr_L1) + size_L1),
                  m_multiplier(1),
                  m_offset(0),
                  m_default_level(0) {}

        FLARE_INLINE_FUNCTION
        const ScratchMemorySpace &set_team_thread_mode(const int &level,
                                                       const int &multiplier,
                                                       const int &offset) const {
            m_default_level = level;
            m_multiplier = multiplier;
            m_offset = offset;
            return *this;
        }
    };

}  // namespace flare

#endif  // FLARE_CORE_MEMORY_SCRATCH_SPACE_H_
