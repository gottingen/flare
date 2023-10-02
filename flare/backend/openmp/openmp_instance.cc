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


#include <flare/core.h>

#include <flare/core/common/error.h>
#include <flare/core/common/cpu_discovery.h>
#include <flare/core/profile/tools.h>
#include <flare/core/common/exec_space_manager.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace flare {
namespace detail {

void OpenMPInternal::acquire_lock() {
  while (1 == flare::atomic_compare_exchange(&m_pool_mutex, 0, 1,
                                             flare::MemoryOrderAcquire(),
                                             flare::MemoryScopeDevice())) {
    // do nothing
  }
}

void OpenMPInternal::release_lock() {
  flare::atomic_store(&m_pool_mutex, 0, flare::MemoryOrderRelease(),
                      flare::MemoryScopeDevice());
}

void OpenMPInternal::clear_thread_data() {
  const size_t member_bytes =
      sizeof(int64_t) *
      HostThreadTeamData::align_to_int64(sizeof(HostThreadTeamData));

  const int old_alloc_bytes =
      m_pool[0] ? (member_bytes + m_pool[0]->scratch_bytes()) : 0;

  OpenMP::memory_space space;

#pragma omp parallel num_threads(m_pool_size)
  {
    const int rank = omp_get_thread_num();

    if (nullptr != m_pool[rank]) {
      m_pool[rank]->disband_pool();

      space.deallocate(m_pool[rank], old_alloc_bytes);

      m_pool[rank] = nullptr;
    }
  }
  /* END #pragma omp parallel */
}

void OpenMPInternal::resize_thread_data(size_t pool_reduce_bytes,
                                        size_t team_reduce_bytes,
                                        size_t team_shared_bytes,
                                        size_t thread_local_bytes) {
  const size_t member_bytes =
      sizeof(int64_t) *
      HostThreadTeamData::align_to_int64(sizeof(HostThreadTeamData));

  HostThreadTeamData *root = m_pool[0];

  const size_t old_pool_reduce  = root ? root->pool_reduce_bytes() : 0;
  const size_t old_team_reduce  = root ? root->team_reduce_bytes() : 0;
  const size_t old_team_shared  = root ? root->team_shared_bytes() : 0;
  const size_t old_thread_local = root ? root->thread_local_bytes() : 0;
  const size_t old_alloc_bytes =
      root ? (member_bytes + root->scratch_bytes()) : 0;

  // Allocate if any of the old allocation is tool small:

  const bool allocate = (old_pool_reduce < pool_reduce_bytes) ||
                        (old_team_reduce < team_reduce_bytes) ||
                        (old_team_shared < team_shared_bytes) ||
                        (old_thread_local < thread_local_bytes);

  if (allocate) {
    if (pool_reduce_bytes < old_pool_reduce) {
      pool_reduce_bytes = old_pool_reduce;
    }
    if (team_reduce_bytes < old_team_reduce) {
      team_reduce_bytes = old_team_reduce;
    }
    if (team_shared_bytes < old_team_shared) {
      team_shared_bytes = old_team_shared;
    }
    if (thread_local_bytes < old_thread_local) {
      thread_local_bytes = old_thread_local;
    }

    const size_t alloc_bytes =
        member_bytes +
        HostThreadTeamData::scratch_size(pool_reduce_bytes, team_reduce_bytes,
                                         team_shared_bytes, thread_local_bytes);

    OpenMP::memory_space space;

    memory_fence();

    for (int rank = 0; rank < m_pool_size; ++rank) {
      if (nullptr != m_pool[rank]) {
        m_pool[rank]->disband_pool();

        space.deallocate(m_pool[rank], old_alloc_bytes);
      }

      void *ptr = nullptr;
      try {
        ptr = space.allocate(alloc_bytes);
      } catch (
          flare::experimental::RawMemoryAllocationFailure const &failure) {
        // For now, just rethrow the error message the existing way
        flare::detail::throw_runtime_exception(failure.get_error_message());
      }

      m_pool[rank] = new (ptr) HostThreadTeamData();

      m_pool[rank]->scratch_assign(((char *)ptr) + member_bytes, alloc_bytes,
                                   pool_reduce_bytes, team_reduce_bytes,
                                   team_shared_bytes, thread_local_bytes);
    }

    HostThreadTeamData::organize_pool(m_pool, m_pool_size);
  }
}

OpenMPInternal &OpenMPInternal::singleton() {
  static OpenMPInternal *self = nullptr;
  if (self == nullptr) {
    self = new OpenMPInternal(get_current_max_threads());
  }

  return *self;
}

int OpenMPInternal::get_current_max_threads() noexcept {
  // Using omp_get_max_threads(); is problematic in conjunction with
  // Hwloc on Intel (essentially an initial call to the OpenMP runtime
  // without a parallel region before will set a process mask for a single core
  // The runtime will than bind threads for a parallel region to other cores on
  // the entering the first parallel region and make the process mask the
  // aggregate of the thread masks. The intend seems to be to make serial code
  // run fast, if you compile with OpenMP enabled but don't actually use
  // parallel regions or so static int omp_max_threads = omp_get_max_threads();

  int count = 0;
#pragma omp parallel
  {
#pragma omp atomic
    ++count;
  }
  return count;
}

void OpenMPInternal::initialize(int thread_count) {
  if (m_initialized) {
    flare::abort(
        "Calling OpenMP::initialize after OpenMP::finalize is illegal\n");
  }

  if (omp_in_parallel()) {
    std::string msg("flare::OpenMP::initialize ERROR : in parallel");
    flare::detail::throw_runtime_exception(msg);
  }

  {
    if (flare::show_warnings() && !std::getenv("OMP_PROC_BIND")) {
      std::cerr
          << R"WARNING(flare::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
  In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
  For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
  For unit testing set OMP_PROC_BIND=false
)WARNING" << std::endl;

      if (mpi_detected()) {
        std::cerr
            << R"WARNING(MPI detected: For OpenMP binding to work as intended, MPI ranks must be bound to exclusive CPU sets.
)WARNING" << std::endl;
      }
    }

    // Before any other call to OMP query the maximum number of threads
    // and save the value for re-initialization unit testing.

    detail::g_openmp_hardware_max_threads = get_current_max_threads();

    int process_num_threads = detail::g_openmp_hardware_max_threads;

    if (flare::hwloc::available()) {
      process_num_threads = flare::hwloc::get_available_numa_count() *
                            flare::hwloc::get_available_cores_per_numa() *
                            flare::hwloc::get_available_threads_per_core();
    }

    // if thread_count  < 0, use g_openmp_hardware_max_threads;
    // if thread_count == 0, set g_openmp_hardware_max_threads to
    // process_num_threads if thread_count  > 0, set
    // g_openmp_hardware_max_threads to thread_count
    if (thread_count < 0) {
      thread_count = detail::g_openmp_hardware_max_threads;
    } else if (thread_count == 0) {
      if (detail::g_openmp_hardware_max_threads != process_num_threads) {
        detail::g_openmp_hardware_max_threads = process_num_threads;
        omp_set_num_threads(detail::g_openmp_hardware_max_threads);
      }
    } else {
      if (flare::show_warnings() && thread_count > process_num_threads) {
        std::cerr << "flare::OpenMP::initialize WARNING: You are likely "
                     "oversubscribing your CPU cores.\n"
                  << "  process threads available : " << std::setw(3)
                  << process_num_threads
                  << ",  requested thread : " << std::setw(3) << thread_count
                  << std::endl;
      }
      detail::g_openmp_hardware_max_threads = thread_count;
      omp_set_num_threads(detail::g_openmp_hardware_max_threads);
    }

// setup thread local
#pragma omp parallel num_threads(detail::g_openmp_hardware_max_threads)
    { detail::SharedAllocationRecord<void, void>::tracking_enable(); }

    auto &instance       = OpenMPInternal::singleton();
    instance.m_pool_size = detail::g_openmp_hardware_max_threads;

    // New, unified host thread team data:
    {
      size_t pool_reduce_bytes  = 32 * thread_count;
      size_t team_reduce_bytes  = 32 * thread_count;
      size_t team_shared_bytes  = 1024 * thread_count;
      size_t thread_local_bytes = 1024;

      instance.resize_thread_data(pool_reduce_bytes, team_reduce_bytes,
                                  team_shared_bytes, thread_local_bytes);
    }
  }

  // Check for over-subscription
  auto const reported_ranks = mpi_ranks_per_node();
  auto const mpi_local_size = reported_ranks < 0 ? 1 : reported_ranks;
  int const procs_per_node  = std::thread::hardware_concurrency();
  if (flare::show_warnings() &&
      (mpi_local_size * long(thread_count) > procs_per_node)) {
    std::cerr << "flare::OpenMP::initialize WARNING: You are likely "
                 "oversubscribing your CPU cores."
              << std::endl;
    std::cerr << "                                    Detected: "
              << procs_per_node << " cores per node." << std::endl;
    std::cerr << "                                    Detected: "
              << mpi_local_size << " MPI_ranks per node." << std::endl;
    std::cerr << "                                    Requested: "
              << thread_count << " threads per process." << std::endl;
  }

  m_initialized = true;
}

void OpenMPInternal::finalize() {
  if (omp_in_parallel()) {
    std::string msg("flare::OpenMP::finalize ERROR ");
    if (this != &singleton()) msg.append(": not initialized");
    if (omp_in_parallel()) msg.append(": in parallel");
    flare::detail::throw_runtime_exception(msg);
  }

  if (this == &singleton()) {
    auto const &instance = singleton();
    // Silence Cuda Warning
    const int nthreads =
        instance.m_pool_size <= detail::g_openmp_hardware_max_threads
            ? detail::g_openmp_hardware_max_threads
            : instance.m_pool_size;
    (void)nthreads;

#pragma omp parallel num_threads(nthreads)
    { detail::SharedAllocationRecord<void, void>::tracking_disable(); }

    // allow main thread to track
    detail::SharedAllocationRecord<void, void>::tracking_enable();

    detail::g_openmp_hardware_max_threads = 1;
  }

  m_initialized = false;

  flare::Profiling::finalize();
}

void OpenMPInternal::print_configuration(std::ostream &s) const {
  s << "flare::OpenMP";

  if (m_initialized) {
    const int numa_count      = 1;
    const int core_per_numa   = detail::g_openmp_hardware_max_threads;
    const int thread_per_core = 1;

    s << " thread_pool_topology[ " << numa_count << " x " << core_per_numa
      << " x " << thread_per_core << " ]" << std::endl;
  } else {
    s << " not initialized" << std::endl;
  }
}

bool OpenMPInternal::verify_is_initialized(const char *const label) const {
  if (!m_initialized) {
    std::cerr << "flare::OpenMP " << label
              << " : ERROR OpenMP is not initialized" << std::endl;
  }
  return m_initialized;
}
}  // namespace detail
}  // namespace flare
