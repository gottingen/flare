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

#ifndef FLARE_BACKEND_OPENMP_OPENMP_H_
#define FLARE_BACKEND_OPENMP_OPENMP_H_

#include <flare/core/defines.h>
#if defined(FLARE_ENABLE_OPENMP)

#include <flare/core_fwd.h>

#include <cstddef>
#include <iosfwd>
#include <flare/core/memory/host_space.h>

#ifdef FLARE_ENABLE_HBWSPACE
#include <flare/core/memory/hbw_space.h>
#endif

#include <flare/core/memory/scratch_space.h>
#include <flare/core/parallel/parallel.h>
#include <flare/core/memory/layout.h>
#include <flare/core/memory/host_shared_ptr.h>
#include <flare/core/profile/interface.h>
#include <flare/core/common/initialization_settings.h>

#include <omp.h>

#include <vector>

/*--------------------------------------------------------------------------*/

namespace flare {

namespace detail {
class OpenMPInternal;

}  // namespace detail

/// \class OpenMP
/// \brief flare device for multicore processors in the host memory space.
class OpenMP {
 public:
  //! Tag this class as a flare execution space
  using execution_space = OpenMP;

  using memory_space =
#ifdef FLARE_ENABLE_HBWSPACE
      experimental::HBWSpace;
#else
      HostSpace;
#endif

  //! This execution space preferred device_type
  using device_type          = flare::Device<execution_space, memory_space>;
  using array_layout         = LayoutRight;
  using size_type            = memory_space::size_type;
  using scratch_memory_space = ScratchMemorySpace<OpenMP>;

  OpenMP();

  OpenMP(int pool_size);

  /// \brief Print configuration information to the given output stream.
  void print_configuration(std::ostream& os, bool verbose = false) const;

  /// \brief is the instance running a parallel algorithm
  static bool in_parallel(OpenMP const& = OpenMP()) noexcept;

  /// \brief Wait until all dispatched functors complete on the given instance
  ///
  ///  This is a no-op on OpenMP
  static void impl_static_fence(std::string const& name);

  void fence(std::string const& name =
                 "flare::OpenMP::fence: Unnamed Instance Fence") const;

  /// \brief Does the given instance return immediately after launching
  /// a parallel algorithm
  ///
  /// This always returns false on OpenMP
  inline static bool is_asynchronous(OpenMP const& = OpenMP()) noexcept;


  int concurrency() const;

  static void impl_initialize(InitializationSettings const&);

  /// \brief is the default execution space initialized for current 'master'
  /// thread
  static bool impl_is_initialized() noexcept;

  /// \brief Free any resources being consumed by the default execution space
  static void impl_finalize();

  int impl_thread_pool_size() const noexcept;

  int impl_thread_pool_size(int depth) const;

  /** \brief  The rank of the executing thread in this thread pool */
  inline static int impl_thread_pool_rank() noexcept;

  // use UniqueToken
  static int impl_max_hardware_threads() noexcept;

  // use UniqueToken
  FLARE_INLINE_FUNCTION
  static int impl_hardware_thread_id() noexcept;

  static int impl_get_current_max_threads() noexcept;

  detail::OpenMPInternal* impl_internal_space_instance() const {
    return m_space_instance.get();
  }

  static constexpr const char* name() noexcept { return "OpenMP"; }
  uint32_t impl_instance_id() const noexcept { return 1; }

 private:
  friend bool operator==(OpenMP const& lhs, OpenMP const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(OpenMP const& lhs, OpenMP const& rhs) {
    return !(lhs == rhs);
  }
  flare::detail::HostSharedPtr<detail::OpenMPInternal> m_space_instance;
};

inline int OpenMP::impl_thread_pool_rank() noexcept {
  FLARE_IF_ON_HOST((return omp_get_thread_num();))
  FLARE_IF_ON_DEVICE((return -1;))
}

inline void OpenMP::impl_static_fence(std::string const& name) {
  flare::Tools::experimental::detail::profile_fence_event<flare::OpenMP>(
      name,
      flare::Tools::experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      []() {});
}

inline bool OpenMP::is_asynchronous(OpenMP const& /*instance*/) noexcept {
  return false;
}

inline int OpenMP::impl_thread_pool_size(int depth) const {
  return depth < 2 ? impl_thread_pool_size() : 1;
}

FLARE_INLINE_FUNCTION
int OpenMP::impl_hardware_thread_id() noexcept {
  FLARE_IF_ON_HOST((return omp_get_thread_num();))

  FLARE_IF_ON_DEVICE((return -1;))
}

namespace Tools {
namespace experimental {
template <>
struct DeviceTypeTraits<OpenMP> {
  static constexpr DeviceType id = DeviceType::OpenMP;
  static int device_id(const OpenMP&) { return 0; }
};
}  // namespace experimental
}  // namespace Tools
}  // namespace flare

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace flare {
namespace detail {

template <>
struct MemorySpaceAccess<flare::OpenMP::memory_space,
                         flare::OpenMP::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};

}  // namespace detail
}  // namespace flare

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#include <flare/backend/openmp/openmp_instance.h>
#include <flare/backend/openmp/openmp_team.h>
#include <flare/core/policy/exp_mdrange_policy.h>
/*--------------------------------------------------------------------------*/

#endif /* #if defined( FLARE_ENABLE_OPENMP ) && defined( _OPENMP ) */
#endif  // FLARE_BACKEND_OPENMP_OPENMP_H_
