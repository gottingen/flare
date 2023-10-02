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

/// \file serial.h
/// \brief Declaration and definition of flare::Serial device.

#ifndef FLARE_BACKEND_SERIAL_SERIAL_H_
#define FLARE_BACKEND_SERIAL_SERIAL_H_

#include <flare/core/defines.h>
#if defined(FLARE_ENABLE_SERIAL)

#include <cstddef>
#include <iosfwd>
#include <mutex>
#include <thread>
#include <flare/core_fwd.h>
#include <flare/core/memory/layout.h>
#include <flare/core/memory/host_space.h>
#include <flare/core/memory/scratch_space.h>
#include <flare/core/memory/memory_traits.h>
#include <flare/core/common/host_thread_team.h>
#include <flare/core/common/functor_analysis.h>
#include <flare/core/profile/tools.h>
#include <flare/core/memory/host_shared_ptr.h>
#include <flare/core/common/initialization_settings.h>

namespace flare {

namespace detail {
class SerialInternal {
 public:
  SerialInternal() = default;

  bool is_initialized();

  void initialize();

  void finalize();

  static SerialInternal& singleton();

  std::mutex m_thread_team_data_mutex;

  // Resize thread team data scratch memory
  void resize_thread_team_data(size_t pool_reduce_bytes,
                               size_t team_reduce_bytes,
                               size_t team_shared_bytes,
                               size_t thread_local_bytes);

  HostThreadTeamData m_thread_team_data;
  bool m_is_initialized = false;
};
}  // namespace detail

/// \class Serial
/// \brief flare device for non-parallel execution
///
/// A "device" represents a parallel execution model.  It tells flare
/// how to parallelize the execution of kernels in a parallel_for or
/// parallel_reduce.  For example, the Threads device uses
/// C++11 threads on a CPU, the OpenMP device uses the OpenMP language
/// extensions, and the Cuda device uses NVIDIA's CUDA programming
/// model.  The Serial device executes "parallel" kernels
/// sequentially.  This is useful if you really do not want to use
/// threads, or if you want to explore different combinations of MPI
/// and shared-memory parallel programming models.
class Serial {
 public:
  //! \name Type declarations that all flare devices must provide.
  //@{

  //! Tag this class as an execution space:
  using execution_space = Serial;
  //! This device's preferred memory space.
  using memory_space = flare::HostSpace;
  //! The size_type alias best suited for this device.
  using size_type = memory_space::size_type;
  //! This execution space preferred device_type
  using device_type = flare::Device<execution_space, memory_space>;

  //! This device's preferred array layout.
  using array_layout = LayoutRight;

  /// \brief  Scratch memory space
  using scratch_memory_space = ScratchMemorySpace<flare::Serial>;

  //@}

  Serial();

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  ///
  /// For the Serial device, this method <i>always</i> returns false,
  /// because parallel_for or parallel_reduce with the Serial device
  /// always execute sequentially.
  inline static int in_parallel() { return false; }

  /// \brief Wait until all dispatched functors complete.
  ///
  /// The parallel_for or parallel_reduce dispatch of a functor may
  /// return asynchronously, before the functor completes.  This
  /// method does not return until all dispatched functors on this
  /// device have completed.
  static void impl_static_fence(const std::string& name) {
    flare::Tools::experimental::detail::profile_fence_event<flare::Serial>(
        name,
        flare::Tools::experimental::SpecialSynchronizationCases::
            GlobalDeviceSynchronization,
        []() {});  // TODO: correct device ID
    flare::memory_fence();
  }

  void fence(const std::string& name =
                 "flare::Serial::fence: Unnamed Instance Fence") const {
    flare::Tools::experimental::detail::profile_fence_event<flare::Serial>(
        name, flare::Tools::experimental::detail::DirectFenceIDHandle{1},
        []() {});  // TODO: correct device ID
    flare::memory_fence();
  }

  /** \brief  Return the maximum amount of concurrency.  */
  int concurrency() const { return 1; }

  //! Print configuration information to the given output stream.
  void print_configuration(std::ostream& os, bool verbose = false) const;

  static void impl_initialize(InitializationSettings const&);

  static bool impl_is_initialized();

  //! Free any resources being consumed by the device.
  static void impl_finalize();

  //--------------------------------------------------------------------------

  inline static int impl_thread_pool_size(int = 0) { return 1; }
  FLARE_INLINE_FUNCTION static int impl_thread_pool_rank() { return 0; }

  //--------------------------------------------------------------------------

  FLARE_INLINE_FUNCTION static unsigned impl_hardware_thread_id() {
    return impl_thread_pool_rank();
  }
  inline static unsigned impl_max_hardware_threads() {
    return impl_thread_pool_size(0);
  }

  uint32_t impl_instance_id() const noexcept { return 1; }

  static const char* name();

  detail::SerialInternal* impl_internal_space_instance() const {
    return m_space_instance.get();
  }

 private:
  flare::detail::HostSharedPtr<detail::SerialInternal> m_space_instance;
  friend bool operator==(Serial const& lhs, Serial const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(Serial const& lhs, Serial const& rhs) {
    return !(lhs == rhs);
  }
  //--------------------------------------------------------------------------
};

namespace Tools {
namespace experimental {
template <>
struct DeviceTypeTraits<Serial> {
  static constexpr DeviceType id = DeviceType::Serial;
  static int device_id(const Serial&) { return 0; }
};
}  // namespace experimental
}  // namespace Tools
}  // namespace flare

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace flare {
namespace detail {

template <>
struct MemorySpaceAccess<flare::Serial::memory_space,
                         flare::Serial::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};

}  // namespace detail
}  // namespace flare

#include <flare/backend/serial/serial_parallel_range.h>
#include <flare/backend/serial/serial_parallel_mdrange.h>
#include <flare/backend/serial/serial_parallel_team.h>
#include <flare/backend/serial/serial_unique_token.h>

#endif  // defined( FLARE_ENABLE_SERIAL )
#endif  // FLARE_BACKEND_SERIAL_SERIAL_H_
