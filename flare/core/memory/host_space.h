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

#ifndef FLARE_CORE_MEMORY_HOST_SPACE_H_
#define FLARE_CORE_MEMORY_HOST_SPACE_H_

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <flare/core_fwd.h>
#include <flare/core/common/concepts.h>
#include <flare/core/memory/memory_traits.h>

#include <flare/core/common/traits.h>
#include <flare/core/common/error.h>
#include <flare/core/memory/shared_alloc.h>
#include <flare/core/profile/tools.h>

#include <flare/core/memory/host_space_deepcopy.h>
#include <flare/core/memory/memory_space.h>

/*--------------------------------------------------------------------------*/

namespace flare {
    /// \class HostSpace
    /// \brief Memory management for host memory.
    ///
    /// HostSpace is a memory space that governs host memory.  "Host"
    /// memory means the usual CPU-accessible memory.
    class HostSpace {
    public:
        //! Tag this class as a flare memory space
        using memory_space = HostSpace;
        using size_type = size_t;

        /// \typedef execution_space
        /// \brief Default execution space for this memory space.
        ///
        /// Every memory space has a default execution space.  This is
        /// useful for things like initializing a Tensor (which happens in
        /// parallel using the Tensor's default execution space).
        using execution_space = DefaultHostExecutionSpace;

        //! This memory space preferred device_type
        using device_type = flare::Device<execution_space, memory_space>;

        HostSpace() = default;

        HostSpace(HostSpace &&rhs) = default;

        HostSpace(const HostSpace &rhs) = default;

        HostSpace &operator=(HostSpace &&) = default;

        HostSpace &operator=(const HostSpace &) = default;

        ~HostSpace() = default;

        /**\brief  Allocate untracked memory in the space */
        void *allocate(const size_t arg_alloc_size) const;

        void *allocate(const char *arg_label, const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0) const;

        /**\brief  Deallocate untracked memory in the space */
        void deallocate(void *const arg_alloc_ptr, const size_t arg_alloc_size) const;

        void deallocate(const char *arg_label, void *const arg_alloc_ptr,
                        const size_t arg_alloc_size,
                        const size_t arg_logical_size = 0) const;

    private:
        template<class, class, class, class>
        friend
        class flare::experimental::LogicalMemorySpace;

        void *impl_allocate(const char *arg_label, const size_t arg_alloc_size,
                            const size_t arg_logical_size = 0,
                            const flare::Tools::SpaceHandle =
                            flare::Tools::make_space_handle(name())) const;

        void impl_deallocate(const char *arg_label, void *const arg_alloc_ptr,
                             const size_t arg_alloc_size,
                             const size_t arg_logical_size = 0,
                             const flare::Tools::SpaceHandle =
                             flare::Tools::make_space_handle(name())) const;

    public:
        /**\brief Return Name of the MemorySpace */
        static constexpr const char *name() { return m_name; }

    private:
        static constexpr const char *m_name = "Host";

        friend class flare::detail::SharedAllocationRecord<flare::HostSpace, void>;
    };

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare::detail {

    static_assert(flare::detail::MemorySpaceAccess<flare::HostSpace,
                          flare::HostSpace>::assignable,
                  "");

    template<typename S>
    struct HostMirror {
    private:
        // If input execution space can access HostSpace then keep it.
        // Example: flare::OpenMP can access, flare::Cuda cannot
        enum {
            keep_exe = flare::detail::MemorySpaceAccess<
                    typename S::execution_space::memory_space,
                    flare::HostSpace>::accessible
        };

        // If HostSpace can access memory space then keep it.
        // Example:  Cannot access flare::CudaSpace, can access flare::CudaUVMSpace
        enum {
            keep_mem =
            flare::detail::MemorySpaceAccess<flare::HostSpace,
                    typename S::memory_space>::accessible
        };

    public:
        using Space = std::conditional_t<
                keep_exe && keep_mem, S,
                std::conditional_t<keep_mem,
                        flare::Device<flare::HostSpace::execution_space,
                                typename S::memory_space>,
                        flare::HostSpace>>;
    };

    template<>
    class SharedAllocationRecord<flare::HostSpace, void>
            : public SharedAllocationRecordCommon<flare::HostSpace> {
    private:
        friend flare::HostSpace;

        friend class SharedAllocationRecordCommon<flare::HostSpace>;

        using base_t = SharedAllocationRecordCommon<flare::HostSpace>;
        using RecordBase = SharedAllocationRecord<void, void>;

        SharedAllocationRecord(const SharedAllocationRecord &) = delete;

        SharedAllocationRecord &operator=(const SharedAllocationRecord &) = delete;

#ifdef FLARE_ENABLE_DEBUG
        /**\brief  Root record for tracked allocations from this HostSpace instance */
        static RecordBase s_root_record;
#endif

        flare::HostSpace m_space;

    protected:
        ~SharedAllocationRecord();

        SharedAllocationRecord() = default;

        template<typename ExecutionSpace>
        SharedAllocationRecord(
                const ExecutionSpace & /* exec_space*/, const flare::HostSpace &arg_space,
                const std::string &arg_label, const size_t arg_alloc_size,
                const RecordBase::function_type arg_dealloc = &deallocate)
                : base_t(
#ifdef FLARE_ENABLE_DEBUG
                &SharedAllocationRecord<flare::HostSpace, void>::s_root_record,
#endif
                detail::checked_allocation_with_header(arg_space, arg_label,
                                                       arg_alloc_size),
                sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
                arg_label),
                  m_space(arg_space) {
            this->base_t::_fill_host_accessible_header_info(*RecordBase::m_alloc_ptr,
                                                            arg_label);
        }

        SharedAllocationRecord(
                const flare::HostSpace &arg_space, const std::string &arg_label,
                const size_t arg_alloc_size,
                const RecordBase::function_type arg_dealloc = &deallocate);

    public:
        FLARE_INLINE_FUNCTION static SharedAllocationRecord *allocate(
                const flare::HostSpace &arg_space, const std::string &arg_label,
                const size_t arg_alloc_size) {
            FLARE_IF_ON_HOST((return new SharedAllocationRecord(arg_space, arg_label,
                                     arg_alloc_size);))
            FLARE_IF_ON_DEVICE(((void) arg_space; (void) arg_label; (void) arg_alloc_size;
                                       return nullptr;))
        }
    };

    template<>
    struct DeepCopy<HostSpace, HostSpace, DefaultHostExecutionSpace> {
        DeepCopy(void *dst, const void *src, size_t n) {
            hostspace_parallel_deepcopy(dst, src, n);
        }

        DeepCopy(const DefaultHostExecutionSpace &exec, void *dst, const void *src,
                 size_t n) {
            hostspace_parallel_deepcopy_async(exec, dst, src, n);
        }
    };

    template<class ExecutionSpace>
    struct DeepCopy<HostSpace, HostSpace, ExecutionSpace> {
        DeepCopy(void *dst, const void *src, size_t n) {
            hostspace_parallel_deepcopy(dst, src, n);
        }

        DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
            exec.fence(
                    "flare::detail::DeepCopy<HostSpace, HostSpace, "
                    "ExecutionSpace>::DeepCopy: fence before copy");
            hostspace_parallel_deepcopy_async(dst, src, n);
        }
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_MEMORY_HOST_SPACE_H_
