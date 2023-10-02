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


#ifndef FLARE_BACKEND_CUDA_CUDA_SPACE_H_
#define FLARE_BACKEND_CUDA_CUDA_SPACE_H_

#include <flare/core/defines.h>
#if defined(FLARE_ON_CUDA_DEVICE)

#include <flare/core_fwd.h>

#include <iosfwd>
#include <typeinfo>
#include <string>
#include <memory>

#include <flare/core/memory/host_space.h>
#include <flare/core/memory/shared_alloc.h>

#include <flare/core/profile/interface.h>

#include <flare/backend/cuda/cuda_abort.h>

#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
extern "C" bool flare_impl_cuda_pin_uvm_to_host();
extern "C" void flare_impl_cuda_set_pin_uvm_to_host(bool);
#endif

/*--------------------------------------------------------------------------*/

namespace flare {
namespace detail {

template <typename T>
struct is_cuda_type_space : public std::false_type {};

}  // namespace detail

/** \brief  Cuda on-device memory management */

class CudaSpace {
 public:
  //! Tag this class as a flare memory space
  using memory_space    = CudaSpace;
  using execution_space = flare::Cuda;
  using device_type     = flare::Device<execution_space, memory_space>;

  using size_type = unsigned int;

  /*--------------------------------*/

  CudaSpace();
  CudaSpace(CudaSpace&& rhs)      = default;
  CudaSpace(const CudaSpace& rhs) = default;
  CudaSpace& operator=(CudaSpace&& rhs) = default;
  CudaSpace& operator=(const CudaSpace& rhs) = default;
  ~CudaSpace()                               = default;

  /**\brief  Allocate untracked memory in the cuda space */
  void* allocate(const Cuda& exec_space, const size_t arg_alloc_size) const;
  void* allocate(const Cuda& exec_space, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class flare::experimental::LogicalMemorySpace;
  void* impl_allocate(const Cuda& exec_space, const char* arg_label,
                      const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const flare::Tools::SpaceHandle =
                          flare::Tools::make_space_handle(name())) const;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const flare::Tools::SpaceHandle =
                          flare::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const flare::Tools::SpaceHandle =
                           flare::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

 private:
  int m_device;  ///< Which Cuda device

  static constexpr const char* m_name = "Cuda";
  friend class flare::detail::SharedAllocationRecord<flare::CudaSpace, void>;
};

template <>
struct detail::is_cuda_type_space<CudaSpace> : public std::true_type {};

}  // namespace flare

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace flare {

/** \brief  Cuda memory that is accessible to Host execution space
 *          through Cuda's unified virtual memory (UVM) runtime.
 */
class CudaUVMSpace {
 public:
  //! Tag this class as a flare memory space
  using memory_space    = CudaUVMSpace;
  using execution_space = Cuda;
  using device_type     = flare::Device<execution_space, memory_space>;
  using size_type       = unsigned int;


  CudaUVMSpace();
  CudaUVMSpace(CudaUVMSpace&& rhs)      = default;
  CudaUVMSpace(const CudaUVMSpace& rhs) = default;
  CudaUVMSpace& operator=(CudaUVMSpace&& rhs) = default;
  CudaUVMSpace& operator=(const CudaUVMSpace& rhs) = default;
  ~CudaUVMSpace()                                  = default;

  /**\brief  Allocate untracked memory in the cuda space */
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class flare::experimental::LogicalMemorySpace;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const flare::Tools::SpaceHandle =
                          flare::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const flare::Tools::SpaceHandle =
                           flare::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
  static bool cuda_pin_uvm_to_host();
  static void cuda_set_pin_uvm_to_host(bool val);
#endif
  /*--------------------------------*/

 private:
  int m_device;  ///< Which Cuda device

#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
  static bool flare_impl_cuda_pin_uvm_to_host_v;
#endif
  static constexpr const char* m_name = "CudaUVM";
};

template <>
struct detail::is_cuda_type_space<CudaUVMSpace> : public std::true_type {};

}  // namespace flare

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace flare {

/** \brief  Host memory that is accessible to Cuda execution space
 *          through Cuda's host-pinned memory allocation.
 */
class CudaHostPinnedSpace {
 public:
  //! Tag this class as a flare memory space
  /** \brief  Memory is in HostSpace so use the HostSpace::execution_space */
  using execution_space = HostSpace::execution_space;
  using memory_space    = CudaHostPinnedSpace;
  using device_type     = flare::Device<execution_space, memory_space>;
  using size_type       = unsigned int;

  /*--------------------------------*/

  CudaHostPinnedSpace();
  CudaHostPinnedSpace(CudaHostPinnedSpace&& rhs)      = default;
  CudaHostPinnedSpace(const CudaHostPinnedSpace& rhs) = default;
  CudaHostPinnedSpace& operator=(CudaHostPinnedSpace&& rhs) = default;
  CudaHostPinnedSpace& operator=(const CudaHostPinnedSpace& rhs) = default;
  ~CudaHostPinnedSpace()                                         = default;

  /**\brief  Allocate untracked memory in the space */
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class flare::experimental::LogicalMemorySpace;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const flare::Tools::SpaceHandle =
                          flare::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const flare::Tools::SpaceHandle =
                           flare::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

 private:
  static constexpr const char* m_name = "CudaHostPinned";

  /*--------------------------------*/
};

template <>
struct detail::is_cuda_type_space<CudaHostPinnedSpace> : public std::true_type {};

}  // namespace flare

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace flare {
namespace detail {

cudaStream_t cuda_get_deep_copy_stream();

const std::unique_ptr<flare::Cuda>& cuda_get_deep_copy_space(
    bool initialize = true);

static_assert(flare::detail::MemorySpaceAccess<flare::CudaSpace,
                                              flare::CudaSpace>::assignable,
              "");
static_assert(flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                                              flare::CudaUVMSpace>::assignable,
              "");
static_assert(
    flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                                    flare::CudaHostPinnedSpace>::assignable,
    "");

//----------------------------------------

template <>
struct MemorySpaceAccess<flare::HostSpace, flare::CudaSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::HostSpace, flare::CudaUVMSpace> {
  // HostSpace::execution_space != CudaUVMSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::HostSpace, flare::CudaHostPinnedSpace> {
  // HostSpace::execution_space == CudaHostPinnedSpace::execution_space
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

//----------------------------------------

template <>
struct MemorySpaceAccess<flare::CudaSpace, flare::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::CudaSpace, flare::CudaUVMSpace> {
  // CudaSpace::execution_space == CudaUVMSpace::execution_space
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::CudaSpace, flare::CudaHostPinnedSpace> {
  // CudaSpace::execution_space != CudaHostPinnedSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };  // CudaSpace::execution_space
  enum : bool { deepcopy = true };
};

//----------------------------------------
// CudaUVMSpace::execution_space == Cuda
// CudaUVMSpace accessible to both Cuda and Host

template <>
struct MemorySpaceAccess<flare::CudaUVMSpace, flare::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };  // Cuda cannot access HostSpace
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::CudaUVMSpace, flare::CudaSpace> {
  // CudaUVMSpace::execution_space == CudaSpace::execution_space
  // Can access CudaUVMSpace from Host but cannot access CudaSpace from Host
  enum : bool { assignable = false };

  // CudaUVMSpace::execution_space can access CudaSpace
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::CudaUVMSpace, flare::CudaHostPinnedSpace> {
  // CudaUVMSpace::execution_space != CudaHostPinnedSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };  // CudaUVMSpace::execution_space
  enum : bool { deepcopy = true };
};

//----------------------------------------
// CudaHostPinnedSpace::execution_space == HostSpace::execution_space
// CudaHostPinnedSpace accessible to both Cuda and Host

template <>
struct MemorySpaceAccess<flare::CudaHostPinnedSpace, flare::HostSpace> {
  enum : bool { assignable = false };  // Cannot access from Cuda
  enum : bool { accessible = true };   // CudaHostPinnedSpace::execution_space
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::CudaHostPinnedSpace, flare::CudaSpace> {
  enum : bool { assignable = false };  // Cannot access from Host
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::CudaHostPinnedSpace, flare::CudaUVMSpace> {
  enum : bool { assignable = false };  // different execution_space
  enum : bool { accessible = true };   // same accessibility
  enum : bool { deepcopy = true };
};

//----------------------------------------

}  // namespace detail
}  // namespace flare

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace flare {
namespace detail {

void DeepCopyCuda(void* dst, const void* src, size_t n);
void DeepCopyAsyncCuda(const Cuda& instance, void* dst, const void* src,
                       size_t n);
void DeepCopyAsyncCuda(void* dst, const void* src, size_t n);

template <class MemSpace>
struct DeepCopy<MemSpace, HostSpace, Cuda,
                std::enable_if_t<is_cuda_type_space<MemSpace>::value>> {
  DeepCopy(void* dst, const void* src, size_t n) { DeepCopyCuda(dst, src, n); }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    DeepCopyAsyncCuda(instance, dst, src, n);
  }
};

template <class MemSpace>
struct DeepCopy<HostSpace, MemSpace, Cuda,
                std::enable_if_t<is_cuda_type_space<MemSpace>::value>> {
  DeepCopy(void* dst, const void* src, size_t n) { DeepCopyCuda(dst, src, n); }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    DeepCopyAsyncCuda(instance, dst, src, n);
  }
};

template <class MemSpace1, class MemSpace2>
struct DeepCopy<MemSpace1, MemSpace2, Cuda,
                std::enable_if_t<is_cuda_type_space<MemSpace1>::value &&
                                 is_cuda_type_space<MemSpace2>::value>> {
  DeepCopy(void* dst, const void* src, size_t n) { DeepCopyCuda(dst, src, n); }
  DeepCopy(const Cuda& instance, void* dst, const void* src, size_t n) {
    DeepCopyAsyncCuda(instance, dst, src, n);
  }
};

template <class MemSpace1, class MemSpace2, class ExecutionSpace>
struct DeepCopy<MemSpace1, MemSpace2, ExecutionSpace,
                std::enable_if_t<is_cuda_type_space<MemSpace1>::value &&
                                 is_cuda_type_space<MemSpace2>::value &&
                                 !std::is_same<ExecutionSpace, Cuda>::value>> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyCuda(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence(fence_string());
    DeepCopyAsyncCuda(dst, src, n);
  }

 private:
  static const std::string& fence_string() {
    static const std::string string =
        std::string("flare::detail::DeepCopy<") + MemSpace1::name() + "Space, " +
        MemSpace2::name() +
        "Space, ExecutionSpace>::DeepCopy: fence before copy";
    return string;
  }
};

template <class MemSpace, class ExecutionSpace>
struct DeepCopy<MemSpace, HostSpace, ExecutionSpace,
                std::enable_if_t<is_cuda_type_space<MemSpace>::value &&
                                 !std::is_same<ExecutionSpace, Cuda>::value>> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyCuda(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence(fence_string());
    DeepCopyAsyncCuda(dst, src, n);
  }

 private:
  static const std::string& fence_string() {
    static const std::string string =
        std::string("flare::detail::DeepCopy<") + MemSpace::name() +
        "Space, HostSpace, ExecutionSpace>::DeepCopy: fence before copy";
    return string;
  }
};

template <class MemSpace, class ExecutionSpace>
struct DeepCopy<HostSpace, MemSpace, ExecutionSpace,
                std::enable_if_t<is_cuda_type_space<MemSpace>::value &&
                                 !std::is_same<ExecutionSpace, Cuda>::value>> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyCuda(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence(fence_string());
    DeepCopyAsyncCuda(dst, src, n);
  }

 private:
  static const std::string& fence_string() {
    static const std::string string =
        std::string("flare::detail::DeepCopy<HostSpace, ") + MemSpace::name() +
        "Space, ExecutionSpace>::DeepCopy: fence before copy";
    return string;
  }
};

}  // namespace detail
}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

template <>
class SharedAllocationRecord<flare::CudaSpace, void>
    : public HostInaccessibleSharedAllocationRecordCommon<flare::CudaSpace> {
 private:
  friend class SharedAllocationRecord<flare::CudaUVMSpace, void>;
  friend class SharedAllocationRecordCommon<flare::CudaSpace>;
  friend class HostInaccessibleSharedAllocationRecordCommon<flare::CudaSpace>;

  using RecordBase = SharedAllocationRecord<void, void>;
  using base_t =
      HostInaccessibleSharedAllocationRecordCommon<flare::CudaSpace>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

#ifdef FLARE_ENABLE_DEBUG
  static RecordBase s_root_record;
#endif

  const flare::CudaSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/, const flare::CudaSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
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

    deep_copy_header_no_exec(RecordBase::m_alloc_ptr, &header);
  }

  SharedAllocationRecord(
      const flare::Cuda& exec_space, const flare::CudaSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);

  SharedAllocationRecord(
      const flare::CudaSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);

  static void deep_copy_header_no_exec(void*, const void*);
};

template <>
class SharedAllocationRecord<flare::CudaUVMSpace, void>
    : public SharedAllocationRecordCommon<flare::CudaUVMSpace> {
 private:
  friend class SharedAllocationRecordCommon<flare::CudaUVMSpace>;

  using base_t     = SharedAllocationRecordCommon<flare::CudaUVMSpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static RecordBase s_root_record;

  const flare::CudaUVMSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/,
      const flare::CudaUVMSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
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

  SharedAllocationRecord(
      const flare::CudaUVMSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);
};

template <>
class SharedAllocationRecord<flare::CudaHostPinnedSpace, void>
    : public SharedAllocationRecordCommon<flare::CudaHostPinnedSpace> {
 private:
  friend class SharedAllocationRecordCommon<flare::CudaHostPinnedSpace>;

  using RecordBase = SharedAllocationRecord<void, void>;
  using base_t     = SharedAllocationRecordCommon<flare::CudaHostPinnedSpace>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static RecordBase s_root_record;

  const flare::CudaHostPinnedSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/,
      const flare::CudaHostPinnedSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
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

  SharedAllocationRecord(
      const flare::CudaHostPinnedSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);
};

}  // namespace detail
}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif  // FLARE_ON_CUDA_DEVICE
#endif  // FLARE_BACKEND_CUDA_CUDA_SPACE_H_
