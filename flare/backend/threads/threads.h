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

#ifndef FLARE_BACKEND_THREADS_THREADS_H_
#define FLARE_BACKEND_THREADS_THREADS_H_

#include <flare/core/defines.h>
#if defined(FLARE_ENABLE_THREADS)

#include <flare/core_fwd.h>

#include <cstddef>
#include <iosfwd>
#include <flare/core/memory/host_space.h>
#include <flare/core/memory/scratch_space.h>
#include <flare/core/memory/layout.h>
#include <flare/core/memory/memory_traits.h>
#include <flare/core/profile/interface.h>
#include <flare/core/common/initialization_settings.h>

/*--------------------------------------------------------------------------*/

namespace flare {
namespace detail {
class ThreadsExec;
enum class fence_is_static { yes, no };
}  // namespace detail
}  // namespace flare

/*--------------------------------------------------------------------------*/

namespace flare {

/** \brief  Execution space for a pool of C++11 threads on a CPU. */
class Threads {
 public:
  //! \name Type declarations that all flare devices must provide.
  //@{
  //! Tag this class as a flare execution space
  using execution_space = Threads;
  using memory_space    = flare::HostSpace;

  //! This execution space preferred device_type
  using device_type = flare::Device<execution_space, memory_space>;

  using array_layout = flare::LayoutRight;
  using size_type    = memory_space::size_type;

  using scratch_memory_space = ScratchMemorySpace<Threads>;

  //@}
  /*------------------------------------------------------------------------*/
  //! \name Static functions that all flare devices must implement.
  //@{

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  static int in_parallel();

  /// \brief Print configuration information to the given output stream.
  void print_configuration(std::ostream& os, bool verbose = false) const;

  /// \brief Wait until all dispatched functors complete.
  ///
  /// The parallel_for or parallel_reduce dispatch of a functor may
  /// return asynchronously, before the functor completes.  This
  /// method does not return until all dispatched functors on this
  /// device have completed.
  static void impl_static_fence(const std::string& name);

  void fence(const std::string& name =
                 "flare::Threads::fence: Unnamed Instance Fence") const;

  /** \brief  Return the maximum amount of concurrency.  */
  int concurrency() const;

  /// \brief Free any resources being consumed by the device.
  ///
  /// For the Threads device, this terminates spawned worker threads.
  static void impl_finalize();

  //@}
  /*------------------------------------------------------------------------*/
  /*------------------------------------------------------------------------*/
  //! \name Space-specific functions
  //@{

  static void impl_initialize(InitializationSettings const&);

  static int impl_is_initialized();

  static Threads& impl_instance(int = 0);

  //----------------------------------------

  static int impl_thread_pool_size(int depth = 0);

  static int impl_thread_pool_rank_host();

  static FLARE_FUNCTION int impl_thread_pool_rank() {
    FLARE_IF_ON_HOST((return impl_thread_pool_rank_host();))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  inline static unsigned impl_max_hardware_threads() {
    return impl_thread_pool_size(0);
  }
  FLARE_INLINE_FUNCTION static unsigned impl_hardware_thread_id() {
    return impl_thread_pool_rank();
  }

  uint32_t impl_instance_id() const noexcept { return 1; }

  static const char* name();
  //@}
  //----------------------------------------
 private:
  friend bool operator==(Threads const&, Threads const&) { return true; }
  friend bool operator!=(Threads const&, Threads const&) { return false; }
};

namespace Tools {
namespace experimental {
template <>
struct DeviceTypeTraits<Threads> {
  static constexpr DeviceType id = DeviceType::Threads;
  static int device_id(const Threads&) { return 0; }
};
}  // namespace experimental
}  // namespace Tools
}  // namespace flare

/*--------------------------------------------------------------------------*/

namespace flare {
namespace detail {

template <>
struct MemorySpaceAccess<flare::Threads::memory_space,
                         flare::Threads::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};

}  // namespace detail
}  // namespace flare

/*--------------------------------------------------------------------------*/

#include <flare/core/policy/exec_policy.h>
#include <flare/core/parallel/parallel.h>
#include <flare/backend/threads/threads_exec.h>
#include <flare/backend/threads/threads_team.h>
#include <flare/backend/threads/threads_parallel_range.h>
#include <flare/backend/threads/threads_parallel_mdrange.h>
#include <flare/backend/threads/threads_parallel_team.h>
#include <flare/backend/threads/threads_unique_token.h>

#include <flare/core/policy/exp_mdrange_policy.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( FLARE_ENABLE_THREADS ) */
#endif  // FLARE_BACKEND_THREADS_THREADS_H_
