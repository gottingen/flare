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

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <flare/core/memory/hbw_space.h>
#include <flare/core/common/error.h>
#include <flare/core/memory/memory_space.h>
#include <flare/core/atomic.h>
#ifdef FLARE_ENABLE_HBWSPACE
#include <memkind.h>
#endif

#include <flare/core/profile/tools.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#ifdef FLARE_ENABLE_HBWSPACE
#define MEMKIND_TYPE MEMKIND_HBW  // hbw_get_kind(HBW_PAGESIZE_4KB)

/*--------------------------------------------------------------------------*/

namespace flare {
namespace experimental {

/* Default allocation mechanism */
HBWSpace::HBWSpace() : m_alloc_mech(HBWSpace::STD_MALLOC) {
  printf("Init\n");
  setenv("MEMKIND_HBW_NODES", "1", 0);
}

/* Default allocation mechanism */
HBWSpace::HBWSpace(const HBWSpace::AllocationMechanism &arg_alloc_mech)
    : m_alloc_mech(HBWSpace::STD_MALLOC) {
  printf("Init2\n");
  setenv("MEMKIND_HBW_NODES", "1", 0);
  if (arg_alloc_mech == STD_MALLOC) {
    m_alloc_mech = HBWSpace::STD_MALLOC;
  }
}

void *HBWSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void *HBWSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                         const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *HBWSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const flare::Tools::SpaceHandle arg_handle) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      flare::detail::power_of_two<flare::detail::MEMORY_ALIGNMENT>::value,
      "Memory alignment must be power of two");

  constexpr uintptr_t alignment      = flare::detail::MEMORY_ALIGNMENT;
  constexpr uintptr_t alignment_mask = alignment - 1;

  void *ptr = nullptr;

  if (arg_alloc_size) {
    if (m_alloc_mech == STD_MALLOC) {
      // Over-allocate to and round up to guarantee proper alignment.
      size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

      void *alloc_ptr = memkind_malloc(MEMKIND_TYPE, size_padded);

      if (alloc_ptr) {
        uintptr_t address = reinterpret_cast<uintptr_t>(alloc_ptr);

        // offset enough to record the alloc_ptr
        address += sizeof(void *);
        uintptr_t rem    = address % alignment;
        uintptr_t offset = rem ? (alignment - rem) : 0u;
        address += offset;
        ptr = reinterpret_cast<void *>(address);
        // record the alloc'd pointer
        address -= sizeof(void *);
        *reinterpret_cast<void **>(address) = alloc_ptr;
      }
    }
  }

  if ((ptr == nullptr) || (reinterpret_cast<uintptr_t>(ptr) == ~uintptr_t(0)) ||
      (reinterpret_cast<uintptr_t>(ptr) & alignment_mask)) {
    std::ostringstream msg;
    msg << "flare::experimental::HBWSpace::allocate[ ";
    switch (m_alloc_mech) {
      case STD_MALLOC: msg << "STD_MALLOC"; break;
      case POSIX_MEMALIGN: msg << "POSIX_MEMALIGN"; break;
      case POSIX_MMAP: msg << "POSIX_MMAP"; break;
      case INTEL_MM_ALLOC: msg << "INTEL_MM_ALLOC"; break;
    }
    msg << " ]( " << arg_alloc_size << " ) FAILED";
    if (ptr == nullptr) {
      msg << " nullptr";
    } else {
      msg << " NOT ALIGNED " << ptr;
    }

    std::cerr << msg.str() << std::endl;
    std::cerr.flush();

    flare::detail::throw_runtime_exception(msg.str());
  }
  if (flare::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    flare::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void HBWSpace::deallocate(void *const arg_alloc_ptr,
                          const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void HBWSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                          const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void HBWSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const flare::Tools::SpaceHandle arg_handle) const {
  if (arg_alloc_ptr) {
    if (flare::Profiling::profileLibraryLoaded()) {
      const size_t reported_size =
          (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
      flare::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                        reported_size);
    }

    if (m_alloc_mech == STD_MALLOC) {
      void *alloc_ptr = *(reinterpret_cast<void **>(arg_alloc_ptr) - 1);
      memkind_free(MEMKIND_TYPE, alloc_ptr);
    }
  }
}

}  // namespace experimental
}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

#ifdef FLARE_ENABLE_DEBUG
SharedAllocationRecord<void, void>
    SharedAllocationRecord<flare::experimental::HBWSpace, void>::s_root_record;
#endif

void SharedAllocationRecord<flare::experimental::HBWSpace, void>::deallocate(
    SharedAllocationRecord<void, void> *arg_rec) {
  delete static_cast<SharedAllocationRecord *>(arg_rec);
}

SharedAllocationRecord<flare::experimental::HBWSpace,
                       void>::~SharedAllocationRecord() {
  m_space.deallocate(m_label.c_str(),
                     SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size,
                     (SharedAllocationRecord<void, void>::m_alloc_size -
                      sizeof(SharedAllocationHeader)));
}

SharedAllocationRecord<flare::experimental::HBWSpace, void>::
    SharedAllocationRecord(
        const flare::experimental::HBWSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef FLARE_ENABLE_DEBUG
          &SharedAllocationRecord<flare::experimental::HBWSpace,
                                  void>::s_root_record,
#endif
          detail::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record =
      static_cast<SharedAllocationRecord<void, void> *>(this);

  strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length - 1);
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr
      ->m_label[SharedAllocationHeader::maximum_label_length - 1] = '\0';
}

//----------------------------------------------------------------------------

void *
SharedAllocationRecord<flare::experimental::HBWSpace, void>::allocate_tracked(
    const flare::experimental::HBWSpace &arg_space,
    const std::string &arg_alloc_label, const size_t arg_alloc_size) {
  if (!arg_alloc_size) return nullptr;

  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<flare::experimental::HBWSpace,
                            void>::deallocate_tracked(void *const
                                                          arg_alloc_ptr) {
  if (arg_alloc_ptr != nullptr) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void *SharedAllocationRecord<flare::experimental::HBWSpace, void>::
    reallocate_tracked(void *const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  flare::detail::DeepCopy<flare::experimental::HBWSpace,
                         flare::experimental::HBWSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));
  flare::fence(
      "SharedAllocationRecord<flare::experimental::HBWSpace, "
      "void>::reallocate_tracked(): fence after copying data");

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

SharedAllocationRecord<flare::experimental::HBWSpace, void>
    *SharedAllocationRecord<flare::experimental::HBWSpace, void>::get_record(
        void *alloc_ptr) {
  using Header = SharedAllocationHeader;
  using RecordHost =
      SharedAllocationRecord<flare::experimental::HBWSpace, void>;

  SharedAllocationHeader const *const head =
      alloc_ptr ? Header::get_header(alloc_ptr) : nullptr;
  RecordHost *const record =
      head ? static_cast<RecordHost *>(head->m_record) : nullptr;

  if (!alloc_ptr || record->m_alloc_ptr != head) {
    flare::detail::throw_runtime_exception(std::string(
        "flare::detail::SharedAllocationRecord< flare::experimental::HBWSpace "
        ", void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord<flare::experimental::HBWSpace, void>::
    print_records(std::ostream &s, const flare::experimental::HBWSpace &space,
                  bool detail) {
#ifdef FLARE_ENABLE_DEBUG
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, "HBWSpace", &s_root_record, detail);
#else
  throw_runtime_exception(
      "SharedAllocationRecord<HBWSpace>::print_records"
      " only works with FLARE_ENABLE_DEBUG enabled");
#endif
}

}  // namespace detail
}  // namespace flare

#endif
