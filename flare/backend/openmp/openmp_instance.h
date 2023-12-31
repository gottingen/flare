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

#ifndef FLARE_BACKEND_OPENMP_OPENMP_INSTANCE_H_
#define FLARE_BACKEND_OPENMP_OPENMP_INSTANCE_H_

#include <flare/core/defines.h>
#if !defined(_OPENMP) && !defined(__CUDA_ARCH__) && !defined(__CUDACC__)
#error \
    "You enabled flare OpenMP support without enabling OpenMP in the compiler!"
#endif

#include <flare/backend/openmp/openmp.h>

#include <flare/core/common/traits.h>
#include <flare/core/common/host_thread_team.h>

#include <flare/core/atomic.h>

#include <flare/core/common/concurrent_bitset.h>

#include <omp.h>

#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

/*--------------------------------------------------------------------------*/
namespace flare {
namespace detail {

inline bool execute_in_serial(OpenMP const& space = OpenMP()) {
  return (OpenMP::in_parallel(space) &&
          !(omp_get_nested() && (omp_get_level() == 1)));
}

}  // namespace detail
}  // namespace flare

namespace flare {
namespace detail {

class OpenMPInternal;

inline int g_openmp_hardware_max_threads = 1;

struct OpenMPTraits {
  static constexpr int MAX_THREAD_COUNT = 512;
};

class OpenMPInternal {
 private:
  OpenMPInternal(int arg_pool_size)
      : m_pool_size{arg_pool_size}, m_level{omp_get_level()}, m_pool() {}

  ~OpenMPInternal() { clear_thread_data(); }

  static int get_current_max_threads() noexcept;

  bool m_initialized = false;

  int m_pool_size;
  int m_level;
  int m_pool_mutex = 0;

  HostThreadTeamData* m_pool[OpenMPTraits::MAX_THREAD_COUNT];

 public:
  friend class flare::OpenMP;

  static OpenMPInternal& singleton();

  void initialize(int thread_cound);

  void finalize();

  void clear_thread_data();

  int thread_pool_size() const { return m_pool_size; }

  // Acquire lock used to protect access to m_pool
  void acquire_lock();

  // Release lock used to protect access to m_pool
  void release_lock();

  void resize_thread_data(size_t pool_reduce_bytes, size_t team_reduce_bytes,
                          size_t team_shared_bytes, size_t thread_local_bytes);

  HostThreadTeamData* get_thread_data() const noexcept {
    return m_pool[m_level == omp_get_level() ? 0 : omp_get_thread_num()];
  }

  HostThreadTeamData* get_thread_data(int i) const noexcept {
    return m_pool[i];
  }

  bool is_initialized() const { return m_initialized; }

  bool verify_is_initialized(const char* const label) const;

  void print_configuration(std::ostream& s) const;
};

}  // namespace detail


namespace experimental {
namespace detail {
// Partitioning an Execution Space: expects space and integer arguments for
// relative weight
template <typename T>
inline std::vector<OpenMP> create_OpenMP_instances(
    OpenMP const& main_instance, std::vector<T> const& weights) {
  static_assert(
      std::is_arithmetic<T>::value,
      "flare Error: partitioning arguments must be integers or floats");
  if (weights.size() == 0) {
    flare::abort("flare::abort: Partition weights vector is empty.");
  }
  std::vector<OpenMP> instances(weights.size());
  double total_weight = std::accumulate(weights.begin(), weights.end(), 0.);
  int const main_pool_size =
      main_instance.impl_internal_space_instance()->thread_pool_size();

  int resources_left = main_pool_size;
  for (unsigned int i = 0; i < weights.size() - 1; ++i) {
    int instance_pool_size = (weights[i] / total_weight) * main_pool_size;
    if (instance_pool_size == 0) {
      flare::abort("flare::abort: Instance has no resource allocated to it");
    }
    instances[i] = OpenMP(instance_pool_size);
    resources_left -= instance_pool_size;
  }
  // Last instance get all resources left
  if (resources_left <= 0) {
    flare::abort(
        "flare::abort: Partition not enough resources left to create the last "
        "instance.");
  }
  instances[weights.size() - 1] = resources_left;

  return instances;
}
}  // namespace detail

template <typename... Args>
std::vector<OpenMP> partition_space(OpenMP const& main_instance, Args... args) {
  // Unpack the arguments and create the weight vector. Note that if not all of
  // the types are the same, you will get a narrowing warning.
  std::vector<std::common_type_t<Args...>> const weights = {args...};
  return detail::create_OpenMP_instances(main_instance, weights);
}

template <typename T>
std::vector<OpenMP> partition_space(OpenMP const& main_instance,
                                    std::vector<T> const& weights) {
  return detail::create_OpenMP_instances(main_instance, weights);
}
}  // namespace experimental


}  // namespace flare

#endif  // FLARE_BACKEND_OPENMP_OPENMP_INSTANCE_H_
