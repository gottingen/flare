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


#ifndef FLARE_CORE_MEMORY_HBW_SPACE_H_
#define FLARE_CORE_MEMORY_HBW_SPACE_H_

#include <flare/core/defines.h>
#ifdef FLARE_ENABLE_HBWSPACE

#include <flare/core/memory/host_space.h>

namespace flare {

namespace experimental {

/// \class HBWSpace
/// \brief Memory management for host memory.
///
/// HBWSpace is a memory space that governs host memory.  "Host"
/// memory means the usual CPU-accessible memory.
class HBWSpace {
 public:
  //! Tag this class as a flare memory space
  using memory_space = HBWSpace;
  using size_type    = size_t;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
  using execution_space = flare::DefaultHostExecutionSpace;

  //! This memory space preferred device_type
  using device_type = flare::Device<execution_space, memory_space>;

  /**\brief  Default memory space instance */
  HBWSpace();
  HBWSpace(const HBWSpace& rhs) = default;
  HBWSpace& operator=(const HBWSpace&) = default;
  ~HBWSpace()                          = default;

  /**\brief  Non-default memory space instance to choose allocation mechansim,
   * if available */

  enum AllocationMechanism {
    STD_MALLOC,
    POSIX_MEMALIGN,
    POSIX_MMAP,
    INTEL_MM_ALLOC
  };

  explicit HBWSpace(const AllocationMechanism&);

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
  friend class LogicalMemorySpace;

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
  static constexpr const char* name() { return "HBW"; }

 private:
  AllocationMechanism m_alloc_mech;
  friend class flare::detail::SharedAllocationRecord<
      flare::experimental::HBWSpace, void>;
};

}  // namespace experimental

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {

namespace detail {

template <>
class SharedAllocationRecord<flare::experimental::HBWSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  friend flare::experimental::HBWSpace;

  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

#ifdef FLARE_ENABLE_DEBUG
  /**\brief  Root record for tracked allocations from this HBWSpace instance */
  static RecordBase s_root_record;
#endif

  const flare::experimental::HBWSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  SharedAllocationRecord(
      const flare::experimental::HBWSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  inline std::string get_label() const {
    return std::string(RecordBase::head()->m_label);
  }

  FLARE_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const flare::experimental::HBWSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size) {
    FLARE_IF_ON_HOST((return new SharedAllocationRecord(arg_space, arg_label,
                                                         arg_alloc_size);))
    FLARE_IF_ON_DEVICE(((void)arg_space; (void)arg_label; (void)arg_alloc_size;
                         return nullptr;))
  }

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const flare::experimental::HBWSpace& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  static void print_records(std::ostream&,
                            const flare::experimental::HBWSpace&,
                            bool detail = false);
};

}  // namespace detail

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {

namespace detail {

static_assert(
    flare::detail::MemorySpaceAccess<flare::experimental::HBWSpace,
                                    flare::experimental::HBWSpace>::assignable,
    "");

template <>
struct MemorySpaceAccess<flare::HostSpace, flare::experimental::HBWSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<flare::experimental::HBWSpace, flare::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

}  // namespace detail

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {

namespace detail {

template <>
struct DeepCopy<flare::experimental::HBWSpace, flare::experimental::HBWSpace,
                DefaultHostExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const DefaultHostExecutionSpace& exec, void* dst, const void* src,
           size_t n) {
    hostspace_parallel_deepcopy(exec, dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<flare::experimental::HBWSpace, flare::experimental::HBWSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "flare::detail::DeepCopy<flare::experimental::HBWSpace, "
        "flare::experimental::HBWSpace,ExecutionSpace::DeepCopy: fence "
        "before copy");
    hostspace_parallel_deepcopy_async(dst, src, n);
  }
};

template <>
struct DeepCopy<HostSpace, flare::experimental::HBWSpace,
                DefaultHostExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const DefaultHostExecutionSpace& exec, void* dst, const void* src,
           size_t n) {
    hostspace_parallel_deepcopy(exec, dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<HostSpace, flare::experimental::HBWSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "flare::detail::DeepCopy<HostSpace, flare::experimental::HBWSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    hostspace_parallel_deepcopy_async(copy_space, dst, src, n);
  }
};

template <>
struct DeepCopy<flare::experimental::HBWSpace, HostSpace,
                DefaultHostExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const DefaultHostExecutionSpace& exec, void* dst, const void* src,
           size_t n) {
    hostspace_parallel_deepcopy(exec, dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<flare::experimental::HBWSpace, HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    hostspace_parallel_deepcopy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "flare::detail::DeepCopy<flare::experimental::HBWSpace, HostSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    hostspace_parallel_deepcopy_async(dst, src, n);
  }
};

}  // namespace detail

}  // namespace flare

#endif
#endif  // #define FLARE_CORE_MEMORY_HBW_SPACE_H_
