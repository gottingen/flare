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

#ifdef FLARE_ON_CUDA_DEVICE

#include <flare/core.h>
#include <flare/backend/cuda/cuda.h>
#include <flare/backend/cuda/cuda_space.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>

//#include <flare/backend/cuda/cuda_block_size_deduction.h>
#include <flare/core/common/error.h>
#include <flare/core/memory/memory_space.h>

#include <flare/core/profile/tools.h>

cudaStream_t flare::detail::cuda_get_deep_copy_stream() {
    static cudaStream_t s = nullptr;
    if (s == nullptr) {
        FLARE_IMPL_CUDA_SAFE_CALL(
                (CudaInternal::singleton().cuda_stream_create_wrapper(&s)));
    }
    return s;
}

const std::unique_ptr<flare::Cuda> &flare::detail::cuda_get_deep_copy_space(
        bool initialize) {
    static std::unique_ptr<Cuda> space = nullptr;
    if (!space && initialize)
        space = std::make_unique<Cuda>(flare::detail::cuda_get_deep_copy_stream());
    return space;
}

namespace flare::detail {

    namespace {

        static std::atomic<int> num_uvm_allocations(0);

    }  // namespace

    void DeepCopyCuda(void *dst, const void *src, size_t n) {
        FLARE_IMPL_CUDA_SAFE_CALL((CudaInternal::singleton().cuda_memcpy_wrapper(
                dst, src, n, cudaMemcpyDefault)));
    }

    void DeepCopyAsyncCuda(const Cuda &instance, void *dst, const void *src,
                           size_t n) {
        FLARE_IMPL_CUDA_SAFE_CALL(
                (instance.impl_internal_space_instance()->cuda_memcpy_async_wrapper(
                        dst, src, n, cudaMemcpyDefault)));
    }

    void DeepCopyAsyncCuda(void *dst, const void *src, size_t n) {
        cudaStream_t s = cuda_get_deep_copy_stream();
        FLARE_IMPL_CUDA_SAFE_CALL(
                (CudaInternal::singleton().cuda_memcpy_async_wrapper(
                        dst, src, n, cudaMemcpyDefault, s)));
        detail::cuda_stream_synchronize(
                s,
                flare::Tools::experimental::SpecialSynchronizationCases::
                DeepCopyResourceSynchronization,
                "flare::detail::DeepCopyAsyncCuda: Deep Copy Stream Sync");
    }

}  // namespace flare::detail

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace flare {

#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
    // The purpose of the following variable is to allow a state-based choice
    // for pinning UVM allocations to the CPU. For now this is considered
    // an experimental debugging capability - with the potential to work around
    // some CUDA issues.
    bool CudaUVMSpace::flare_impl_cuda_pin_uvm_to_host_v = false;

    bool CudaUVMSpace::cuda_pin_uvm_to_host() {
      return CudaUVMSpace::flare_impl_cuda_pin_uvm_to_host_v;
    }
    void CudaUVMSpace::cuda_set_pin_uvm_to_host(bool val) {
      CudaUVMSpace::flare_impl_cuda_pin_uvm_to_host_v = val;
    }
#endif
}  // namespace flare

#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
bool flare_impl_cuda_pin_uvm_to_host() {
  return flare::CudaUVMSpace::cuda_pin_uvm_to_host();
}

void flare_impl_cuda_set_pin_uvm_to_host(bool val) {
  flare::CudaUVMSpace::cuda_set_pin_uvm_to_host(val);
}
#endif

namespace flare {

    CudaSpace::CudaSpace() : m_device(flare::Cuda().cuda_device()) {}

    CudaUVMSpace::CudaUVMSpace() : m_device(flare::Cuda().cuda_device()) {}

    CudaHostPinnedSpace::CudaHostPinnedSpace() {}

    size_t memory_threshold_g = 40000;  // 40 kB

    void *CudaSpace::allocate(const size_t arg_alloc_size) const {
        return allocate("[unlabeled]", arg_alloc_size);
    }

    void *CudaSpace::allocate(const Cuda &exec_space, const char *arg_label,
                              const size_t arg_alloc_size,
                              const size_t arg_logical_size) const {
        return impl_allocate(exec_space, arg_label, arg_alloc_size, arg_logical_size);
    }

    void *CudaSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                              const size_t arg_logical_size) const {
        return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
    }

    namespace {
        void *impl_allocate_common(const Cuda &exec_space, const char *arg_label,
                                   const size_t arg_alloc_size,
                                   const size_t arg_logical_size,
                                   const flare::Tools::SpaceHandle arg_handle,
                                   bool exec_space_provided) {
            void *ptr = nullptr;

#ifndef CUDART_VERSION
#error CUDART_VERSION undefined!
#elif (defined(FLARE_ENABLE_IMPL_CUDA_MALLOC_ASYNC) && CUDART_VERSION >= 11020)
            cudaError_t error_code;
            if (arg_alloc_size >= memory_threshold_g) {
                if (exec_space_provided) {
                    error_code =
                            exec_space.impl_internal_space_instance()->cuda_malloc_async_wrapper(
                                    &ptr, arg_alloc_size);
                    exec_space.fence("flare::Cuda: backend fence after async malloc");
                } else {
                    error_code = detail::CudaInternal::singleton().cuda_malloc_async_wrapper(
                            &ptr, arg_alloc_size);
                    detail::cuda_device_synchronize(
                            "flare::Cuda: backend fence after async malloc");
                }
            } else {
                error_code =
                        (exec_space_provided
                         ? exec_space.impl_internal_space_instance()->cuda_malloc_wrapper(
                                        &ptr, arg_alloc_size)
                         : detail::CudaInternal::singleton().cuda_malloc_wrapper(
                                        &ptr, arg_alloc_size));
            }
#else
            cudaError_t error_code;
            if (exec_space_provided) {
              error_code = exec_space.impl_internal_space_instance()->cuda_malloc_wrapper(
                  &ptr, arg_alloc_size);
            } else {
              error_code = detail::CudaInternal::singleton().cuda_malloc_wrapper(
                  &ptr, arg_alloc_size);
            }
#endif
            if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
                // This is the only way to clear the last error, which
                // we should do here since we're turning it into an
                // exception here
                exec_space.impl_internal_space_instance()->cuda_get_last_error_wrapper();
                throw experimental::CudaRawMemoryAllocationFailure(
                        arg_alloc_size, error_code,
                        experimental::RawMemoryAllocationFailure::AllocationMechanism::
                        CudaMalloc);
            }

            if (flare::Profiling::profileLibraryLoaded()) {
                const size_t reported_size =
                        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
                flare::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
            }
            return ptr;
        }
    }  // namespace

    void *CudaSpace::impl_allocate(
            const char *arg_label, const size_t arg_alloc_size,
            const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        return impl_allocate_common(flare::Cuda{}, arg_label, arg_alloc_size,
                                    arg_logical_size, arg_handle, false);
    }

    void *CudaSpace::impl_allocate(
            const Cuda &exec_space, const char *arg_label, const size_t arg_alloc_size,
            const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        return impl_allocate_common(exec_space, arg_label, arg_alloc_size,
                                    arg_logical_size, arg_handle, true);
    }

    void *CudaUVMSpace::allocate(const size_t arg_alloc_size) const {
        return allocate("[unlabeled]", arg_alloc_size);
    }

    void *CudaUVMSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                                 const size_t arg_logical_size) const {
        return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
    }

    void *CudaUVMSpace::impl_allocate(
            const char *arg_label, const size_t arg_alloc_size,
            const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        void *ptr = nullptr;

        Cuda::impl_static_fence(
                "flare::CudaUVMSpace::impl_allocate: Pre UVM Allocation");
        if (arg_alloc_size > 0) {
            flare::detail::num_uvm_allocations++;

            auto error_code =
                    detail::CudaInternal::singleton().cuda_malloc_managed_wrapper(
                            &ptr, arg_alloc_size, cudaMemAttachGlobal);

#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
            if (flare::CudaUVMSpace::cuda_pin_uvm_to_host())
              FLARE_IMPL_CUDA_SAFE_CALL(
                  (detail::CudaInternal::singleton().cuda_mem_advise_wrapper(
                      ptr, arg_alloc_size, cudaMemAdviseSetPreferredLocation,
                      cudaCpuDeviceId)));
#endif

            if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
                // This is the only way to clear the last error, which
                // we should do here since we're turning it into an
                // exception here
                detail::CudaInternal::singleton().cuda_get_last_error_wrapper();
                throw experimental::CudaRawMemoryAllocationFailure(
                        arg_alloc_size, error_code,
                        experimental::RawMemoryAllocationFailure::AllocationMechanism::
                        CudaMallocManaged);
            }
        }
        Cuda::impl_static_fence(
                "flare::CudaUVMSpace::impl_allocate: Post UVM Allocation");
        if (flare::Profiling::profileLibraryLoaded()) {
            const size_t reported_size =
                    (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
            flare::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
        }
        return ptr;
    }

    void *CudaHostPinnedSpace::allocate(const size_t arg_alloc_size) const {
        return allocate("[unlabeled]", arg_alloc_size);
    }

    void *CudaHostPinnedSpace::allocate(const char *arg_label,
                                        const size_t arg_alloc_size,
                                        const size_t arg_logical_size) const {
        return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
    }

    void *CudaHostPinnedSpace::impl_allocate(
            const char *arg_label, const size_t arg_alloc_size,
            const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        void *ptr = nullptr;

        auto error_code = detail::CudaInternal::singleton().cuda_host_alloc_wrapper(
                &ptr, arg_alloc_size, cudaHostAllocDefault);
        if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
            // This is the only way to clear the last error, which
            // we should do here since we're turning it into an
            // exception here
            detail::CudaInternal::singleton().cuda_get_last_error_wrapper();
            throw experimental::CudaRawMemoryAllocationFailure(
                    arg_alloc_size, error_code,
                    experimental::RawMemoryAllocationFailure::AllocationMechanism::
                    CudaHostAlloc);
        }
        if (flare::Profiling::profileLibraryLoaded()) {
            const size_t reported_size =
                    (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
            flare::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
        }
        return ptr;
    }

    void CudaSpace::deallocate(void *const arg_alloc_ptr,
                               const size_t arg_alloc_size) const {
        deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
    }

    void CudaSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                               const size_t arg_alloc_size,
                               const size_t arg_logical_size) const {
        impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
    }

    void CudaSpace::impl_deallocate(const char *arg_label, void *const arg_alloc_ptr,
                                    const size_t arg_alloc_size, const size_t arg_logical_size,
                                    const flare::Tools::SpaceHandle arg_handle) const {
        if (flare::Profiling::profileLibraryLoaded()) {
            const size_t reported_size =
                    (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
            flare::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr, reported_size);
        }
        try {
#ifndef CUDART_VERSION
#error CUDART_VERSION undefined!
#elif (defined(FLARE_ENABLE_IMPL_CUDA_MALLOC_ASYNC) && CUDART_VERSION >= 11020)
            if (arg_alloc_size >= memory_threshold_g) {
                detail::cuda_device_synchronize(
                        "flare::Cuda: backend fence before async free");
                FLARE_IMPL_CUDA_SAFE_CALL(
                        (detail::CudaInternal::singleton().cuda_free_async_wrapper(arg_alloc_ptr)));
                detail::cuda_device_synchronize(
                        "flare::Cuda: backend fence after async free");
            } else {
                FLARE_IMPL_CUDA_SAFE_CALL(
                        (detail::CudaInternal::singleton().cuda_free_wrapper(arg_alloc_ptr)));
            }
#else
            FLARE_IMPL_CUDA_SAFE_CALL(
                (detail::CudaInternal::singleton().cuda_free_wrapper(arg_alloc_ptr)));
#endif
        } catch (...) {
        }
    }

    void CudaUVMSpace::deallocate(void *const arg_alloc_ptr,
                                  const size_t arg_alloc_size) const {
        deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
    }

    void CudaUVMSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                                  const size_t arg_alloc_size, const size_t arg_logical_size) const {
        impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
    }

    void CudaUVMSpace::impl_deallocate(
            const char *arg_label, void *const arg_alloc_ptr,
            const size_t arg_alloc_size, const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        Cuda::impl_static_fence(
                "flare::CudaUVMSpace::impl_deallocate: Pre UVM Deallocation");
        if (flare::Profiling::profileLibraryLoaded()) {
            const size_t reported_size =
                    (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
            flare::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                             reported_size);
        }
        try {
            if (arg_alloc_ptr != nullptr) {
                flare::detail::num_uvm_allocations--;
                FLARE_IMPL_CUDA_SAFE_CALL(
                        (detail::CudaInternal::singleton().cuda_free_wrapper(arg_alloc_ptr)));
            }
        } catch (...) {
        }
        Cuda::impl_static_fence(
                "flare::CudaUVMSpace::impl_deallocate: Post UVM Deallocation");
    }

    void CudaHostPinnedSpace::deallocate(void *const arg_alloc_ptr,
                                         const size_t arg_alloc_size) const {
        deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
    }

    void CudaHostPinnedSpace::deallocate(const char *arg_label,
                                         void *const arg_alloc_ptr,
                                         const size_t arg_alloc_size,
                                         const size_t arg_logical_size) const {
        impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
    }

    void CudaHostPinnedSpace::impl_deallocate(
            const char *arg_label, void *const arg_alloc_ptr,
            const size_t arg_alloc_size, const size_t arg_logical_size,
            const flare::Tools::SpaceHandle arg_handle) const {
        if (flare::Profiling::profileLibraryLoaded()) {
            const size_t reported_size =
                    (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
            flare::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                             reported_size);
        }
        try {
            FLARE_IMPL_CUDA_SAFE_CALL((
                                              detail::CudaInternal::singleton().cuda_free_host_wrapper(arg_alloc_ptr)));
        } catch (...) {
        }
    }

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

#ifdef FLARE_ENABLE_DEBUG
    SharedAllocationRecord<void, void>
        SharedAllocationRecord<flare::CudaSpace, void>::s_root_record;

    SharedAllocationRecord<void, void>
        SharedAllocationRecord<flare::CudaUVMSpace, void>::s_root_record;

    SharedAllocationRecord<void, void>
        SharedAllocationRecord<flare::CudaHostPinnedSpace, void>::s_root_record;
#endif

    SharedAllocationRecord<flare::CudaSpace, void>::~SharedAllocationRecord() {
        auto alloc_size = SharedAllocationRecord<void, void>::m_alloc_size;
        m_space.deallocate(m_label.c_str(),
                           SharedAllocationRecord<void, void>::m_alloc_ptr,
                           alloc_size, (alloc_size - sizeof(SharedAllocationHeader)));
    }

    void SharedAllocationRecord<flare::CudaSpace, void>::deep_copy_header_no_exec(
            void *ptr, const void *header) {
        flare::Cuda exec;
        flare::detail::DeepCopy<CudaSpace, HostSpace>(exec, ptr, header,
                                                      sizeof(SharedAllocationHeader));
        exec.fence(
                "SharedAllocationRecord<flare::CudaSpace, "
                "void>::SharedAllocationRecord(): fence after copying header from "
                "HostSpace");
    }

    SharedAllocationRecord<flare::CudaUVMSpace, void>::~SharedAllocationRecord() {
        m_space.deallocate(m_label.c_str(),
                           SharedAllocationRecord<void, void>::m_alloc_ptr,
                           SharedAllocationRecord<void, void>::m_alloc_size,
                           (SharedAllocationRecord<void, void>::m_alloc_size -
                            sizeof(SharedAllocationHeader)));
    }

    SharedAllocationRecord<flare::CudaHostPinnedSpace,
            void>::~SharedAllocationRecord() {
        m_space.deallocate(m_label.c_str(),
                           SharedAllocationRecord<void, void>::m_alloc_ptr,
                           SharedAllocationRecord<void, void>::m_alloc_size,
                           (SharedAllocationRecord<void, void>::m_alloc_size -
                            sizeof(SharedAllocationHeader)));
    }


    SharedAllocationRecord<flare::CudaSpace, void>::SharedAllocationRecord(
            const flare::CudaSpace &arg_space, const std::string &arg_label,
            const size_t arg_alloc_size,
            const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
            : base_t(
#ifdef FLARE_ENABLE_DEBUG
            &SharedAllocationRecord<flare::CudaSpace, void>::s_root_record,
#endif
            detail::checked_allocation_with_header(arg_space, arg_label,
                                                   arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
            arg_label),
              m_space(arg_space) {

        SharedAllocationHeader header;

        this->base_t::_fill_host_accessible_header_info(header, arg_label);

        // Copy to device memory
        flare::Cuda exec;
        flare::detail::DeepCopy<CudaSpace, HostSpace>(
                exec, RecordBase::m_alloc_ptr, &header, sizeof(SharedAllocationHeader));
        exec.fence(
                "SharedAllocationRecord<flare::CudaSpace, "
                "void>::SharedAllocationRecord(): fence after copying header from "
                "HostSpace");
    }

    SharedAllocationRecord<flare::CudaSpace, void>::SharedAllocationRecord(
            const flare::Cuda &arg_exec_space, const flare::CudaSpace &arg_space,
            const std::string &arg_label, const size_t arg_alloc_size,
            const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
            : base_t(
#ifdef FLARE_ENABLE_DEBUG
            &SharedAllocationRecord<flare::CudaSpace, void>::s_root_record,
#endif
            detail::checked_allocation_with_header(arg_exec_space, arg_space,
                                                   arg_label, arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
            arg_label),
              m_space(arg_space) {

        SharedAllocationHeader header;

        this->base_t::_fill_host_accessible_header_info(header, arg_label);

        // Copy to device memory
        flare::detail::DeepCopy<CudaSpace, HostSpace>(arg_exec_space,
                                                      RecordBase::m_alloc_ptr, &header,
                                                      sizeof(SharedAllocationHeader));
    }

    SharedAllocationRecord<flare::CudaUVMSpace, void>::SharedAllocationRecord(
            const flare::CudaUVMSpace &arg_space, const std::string &arg_label,
            const size_t arg_alloc_size,
            const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
            : base_t(
#ifdef FLARE_ENABLE_DEBUG
            &SharedAllocationRecord<flare::CudaUVMSpace, void>::s_root_record,
#endif
            detail::checked_allocation_with_header(arg_space, arg_label,
                                                   arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
            arg_label),
              m_space(arg_space) {
        this->base_t::_fill_host_accessible_header_info(*base_t::m_alloc_ptr,
                                                        arg_label);
    }

    SharedAllocationRecord<flare::CudaHostPinnedSpace, void>::
    SharedAllocationRecord(
            const flare::CudaHostPinnedSpace &arg_space,
            const std::string &arg_label, const size_t arg_alloc_size,
            const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
            : base_t(
#ifdef FLARE_ENABLE_DEBUG
            &SharedAllocationRecord<flare::CudaHostPinnedSpace,
                                    void>::s_root_record,
#endif
            detail::checked_allocation_with_header(arg_space, arg_label,
                                                   arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
            arg_label),
              m_space(arg_space) {
        this->base_t::_fill_host_accessible_header_info(*base_t::m_alloc_ptr,
                                                        arg_label);
    }

    void cuda_prefetch_pointer(const Cuda &space, const void *ptr, size_t bytes,
                               bool to_device) {
        if ((ptr == nullptr) || (bytes == 0)) return;
        cudaPointerAttributes attr;
        FLARE_IMPL_CUDA_SAFE_CALL((
                                          space.impl_internal_space_instance()->cuda_pointer_get_attributes_wrapper(
                                                  &attr, ptr)));
        // I measured this and it turns out prefetching towards the host slows
        // DualTensor syncs down. Probably because the latency is not too bad in the
        // first place for the pull down. If we want to change that provde
        // cudaCpuDeviceId as the device if to_device is false
        bool is_managed = attr.type == cudaMemoryTypeManaged;
        if (to_device && is_managed &&
            space.cuda_device_prop().concurrentManagedAccess) {
            FLARE_IMPL_CUDA_SAFE_CALL(
                    (space.impl_internal_space_instance()->cuda_mem_prefetch_async_wrapper(
                            ptr, bytes, space.cuda_device())));
        }
    }

}  // namespace flare::detail

#include <flare/core/memory/shared_alloc_impl.h>

namespace flare::detail {

    // To avoid additional compilation cost for something that's (mostly?) not
    // performance sensitive, we explicity instantiate these CRTP base classes here,
    // where we have access to the associated *_timpl.hpp header files.
    template
    class SharedAllocationRecordCommon<flare::CudaSpace>;

    template
    class HostInaccessibleSharedAllocationRecordCommon<flare::CudaSpace>;

    template
    class SharedAllocationRecordCommon<flare::CudaUVMSpace>;

    template
    class SharedAllocationRecordCommon<flare::CudaHostPinnedSpace>;

}  // end namespace flare::detail

#else
void FLARE_CORE_SRC_CUDA_CUDASPACE_PREVENT_LINK_ERROR() {}
#endif  // FLARE_ON_CUDA_DEVICE
