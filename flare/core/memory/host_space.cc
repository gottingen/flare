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

#include <flare/core/defines.h>

#include <flare/core/common/error.h>
#include <flare/core/memory/memory_space.h>
#include <flare/core/profile/tools.h>

/*--------------------------------------------------------------------------*/

#if (defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)) && !defined(FLARE_ON_CUDA_DEVICE)

// Intel specialized allocator does not interoperate with CUDA memory allocation

#define FLARE_ENABLE_INTEL_MM_ALLOC

#endif

/*--------------------------------------------------------------------------*/

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <sstream>
#include <cstring>

#ifdef FLARE_COMPILER_INTEL
#include <aligned_new>
#endif

#include <flare/core/memory/host_space.h>
#include <flare/core/common/error.h>
#include <flare/core/atomic.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {


    void *HostSpace::allocate(const size_t arg_alloc_size) const {
        return allocate("[unlabeled]", arg_alloc_size);
    }

    void *HostSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                              const size_t

                              arg_logical_size) const {
        return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
    }

    void *HostSpace::impl_allocate(
            const char *arg_label, const size_t arg_alloc_size,
            const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        const size_t reported_size =
                (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
        static_assert(sizeof(void *) == sizeof(uintptr_t),
                      "Error sizeof(void*) != sizeof(uintptr_t)");

        static_assert(
                flare::detail::is_integral_power_of_two(flare::detail::MEMORY_ALIGNMENT),
                "Memory alignment must be power of two");

        constexpr uintptr_t alignment = flare::detail::MEMORY_ALIGNMENT;
        constexpr uintptr_t alignment_mask = alignment - 1;

        void *ptr = nullptr;

        if (arg_alloc_size)
            ptr = operator new(arg_alloc_size, std::align_val_t(alignment),
                               std::nothrow_t{});

        if ((ptr == nullptr) || (reinterpret_cast<uintptr_t>(ptr) == ~uintptr_t(0)) ||
            (reinterpret_cast<uintptr_t>(ptr) & alignment_mask)) {
            experimental::RawMemoryAllocationFailure::FailureMode failure_mode =
                    experimental::RawMemoryAllocationFailure::FailureMode::
                    AllocationNotAligned;
            if (ptr == nullptr) {
                failure_mode = experimental::RawMemoryAllocationFailure::FailureMode::
                OutOfMemoryError;
            }

            experimental::RawMemoryAllocationFailure::AllocationMechanism alloc_mec =
                    experimental::RawMemoryAllocationFailure::AllocationMechanism::
                    StdMalloc;

            throw flare::experimental::RawMemoryAllocationFailure(
                    arg_alloc_size, alignment, failure_mode, alloc_mec);
        }
        if (flare::Profiling::profileLibraryLoaded()) {
            flare::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
        }
        return ptr;
    }

    void HostSpace::deallocate(void *const arg_alloc_ptr,
                               const size_t arg_alloc_size) const {
        deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
    }

    void HostSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                               const size_t arg_alloc_size,
                               const size_t

                               arg_logical_size) const {
        impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
    }

    void HostSpace::impl_deallocate(
            const char *arg_label, void *const arg_alloc_ptr,
            const size_t arg_alloc_size, const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        if (arg_alloc_ptr) {
            flare::fence("HostSpace::impl_deallocate before free");
            size_t reported_size =
                    (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
            if (flare::Profiling::profileLibraryLoaded()) {
                flare::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                                 reported_size);
            }
            constexpr uintptr_t alignment = flare::detail::MEMORY_ALIGNMENT;
            operator delete(arg_alloc_ptr, std::align_val_t(alignment),
                            std::nothrow_t{});
        }
    }

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
    namespace detail {

#ifdef FLARE_ENABLE_DEBUG
        SharedAllocationRecord<void, void>
                SharedAllocationRecord<flare::HostSpace, void>::s_root_record;
#endif

        SharedAllocationRecord<flare::HostSpace, void>::~SharedAllocationRecord() {
            m_space.deallocate(m_label.c_str(),
                               SharedAllocationRecord<void, void>::m_alloc_ptr,
                               SharedAllocationRecord<void, void>::m_alloc_size,
                               (SharedAllocationRecord<void, void>::m_alloc_size -
                                sizeof(SharedAllocationHeader)));
        }

        SharedAllocationHeader *_do_allocation(flare::HostSpace const &space,
                                               std::string const &label,
                                               size_t alloc_size) {
            try {
                return reinterpret_cast<SharedAllocationHeader *>(
                        space.allocate(alloc_size));
            } catch (experimental::RawMemoryAllocationFailure const &failure) {
                if (failure.failure_mode() == experimental::RawMemoryAllocationFailure::
                FailureMode::AllocationNotAligned) {
                    // TODO: delete the misaligned memory
                }

                std::cerr << "flare failed to allocate memory for label \"" << label
                          << "\".  Allocation using MemorySpace named \"" << space.name()
                          << " failed with the following error:  ";
                failure.print_error_message(std::cerr);
                std::cerr.flush();
                flare::detail::throw_runtime_exception("Memory allocation failure");
            }
            return nullptr;  // unreachable
        }

        SharedAllocationRecord<flare::HostSpace, void>::SharedAllocationRecord(
                const flare::HostSpace &arg_space, const std::string &arg_label,
                const size_t arg_alloc_size,
                const SharedAllocationRecord<void, void>::function_type arg_dealloc)
        // Pass through allocated [ SharedAllocationHeader , user_memory ]
        // Pass through deallocation function
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

    }  // namespace detail
}  // namespace flare

#include <flare/core/memory/shared_alloc_impl.h>

namespace flare {
    namespace detail {

        // To avoid additional compilation cost for something that's (mostly?) not
        // performance sensitive, we explicity instantiate these CRTP base classes here,
        // where we have access to the associated *_timpl.hpp header files.
        template
        class SharedAllocationRecordCommon<flare::HostSpace>;

    }  // end namespace detail
}  // end namespace flare
