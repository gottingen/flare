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

#ifndef FLARE_BACKEND_OPENMP_OPENMP_PARALLEL_FOR_H_
#define FLARE_BACKEND_OPENMP_OPENMP_PARALLEL_FOR_H_

#include <omp.h>
#include <flare/backend/openmp/openmp_instance.h>
#include <flare/core/policy/exp_mdrange_policy.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define FLARE_PRAGMA_IVDEP_IF_ENABLED
#if defined(FLARE_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(FLARE_ENABLE_PRAGMA_IVDEP)
#undef FLARE_PRAGMA_IVDEP_IF_ENABLED
#define FLARE_PRAGMA_IVDEP_IF_ENABLED _Pragma("ivdep")
#endif

#ifndef FLARE_COMPILER_NVHPC
#define FLARE_OPENMP_OPTIONAL_CHUNK_SIZE , m_policy.chunk_size()
#else
#define FLARE_OPENMP_OPTIONAL_CHUNK_SIZE
#endif

namespace flare {
namespace detail {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, flare::RangePolicy<Traits...>, flare::OpenMP> {
 private:
  using Policy  = flare::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  OpenMPInternal* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  inline static void exec_range(const FunctorType& functor, const Member ibeg,
                                const Member iend) {
    FLARE_PRAGMA_IVDEP_IF_ENABLED
    for (auto iwork = ibeg; iwork < iend; ++iwork) {
      exec_work(functor, iwork);
    }
  }

  template <class Enable = WorkTag>
  inline static std::enable_if_t<std::is_void<WorkTag>::value &&
                                 std::is_same<Enable, WorkTag>::value>
  exec_work(const FunctorType& functor, const Member iwork) {
    functor(iwork);
  }

  template <class Enable = WorkTag>
  inline static std::enable_if_t<!std::is_void<WorkTag>::value &&
                                 std::is_same<Enable, WorkTag>::value>
  exec_work(const FunctorType& functor, const Member iwork) {
    functor(WorkTag{}, iwork);
  }

  template <class Policy>
  std::enable_if_t<std::is_same<typename Policy::schedule_type::type,
                                flare::Dynamic>::value>
  execute_parallel() const {
    // prevent bug in NVHPC 21.9/CUDA 11.4 (entering zero iterations loop)
    if (m_policy.begin() >= m_policy.end()) return;
#pragma omp parallel for schedule(dynamic FLARE_OPENMP_OPTIONAL_CHUNK_SIZE) \
    num_threads(m_instance->thread_pool_size())
    FLARE_PRAGMA_IVDEP_IF_ENABLED
    for (auto iwork = m_policy.begin(); iwork < m_policy.end(); ++iwork) {
      exec_work(m_functor, iwork);
    }
  }

  template <class Policy>
  std::enable_if_t<!std::is_same<typename Policy::schedule_type::type,
                                 flare::Dynamic>::value>
  execute_parallel() const {
// Specifying an chunksize with GCC compiler leads to performance regression
// with static schedule.
#ifdef FLARE_COMPILER_GNU
#pragma omp parallel for schedule(static) \
    num_threads(m_instance->thread_pool_size())
#else
#pragma omp parallel for schedule(static FLARE_OPENMP_OPTIONAL_CHUNK_SIZE) \
    num_threads(m_instance->thread_pool_size())
#endif
    FLARE_PRAGMA_IVDEP_IF_ENABLED
    for (auto iwork = m_policy.begin(); iwork < m_policy.end(); ++iwork) {
      exec_work(m_functor, iwork);
    }
  }

 public:
  inline void execute() const {
    if (execute_in_serial(m_policy.space())) {
      exec_range(m_functor, m_policy.begin(), m_policy.end());
      return;
    }

#ifndef FLARE_INTERNAL_DISABLE_NATIVE_OPENMP
    execute_parallel<Policy>();
#else
    constexpr bool is_dynamic =
        std::is_same<typename Policy::schedule_type::type,
                     flare::Dynamic>::value;
#pragma omp parallel num_threads(m_instance->thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      data.set_work_partition(m_policy.end() - m_policy.begin(),
                              m_policy.chunk_size());

      if (is_dynamic) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      std::pair<int64_t, int64_t> range(0, 0);

      do {
        range = is_dynamic ? data.get_work_stealing_chunk()
                           : data.get_work_partition();

        exec_range(m_functor, range.first + m_policy.begin(),
                   range.second + m_policy.begin());

      } while (is_dynamic && 0 <= range.first);
    }
#endif
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(nullptr), m_functor(arg_functor), m_policy(arg_policy) {
    m_instance = arg_policy.space().impl_internal_space_instance();
  }
};

// MDRangePolicy impl
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, flare::MDRangePolicy<Traits...>,
                  flare::OpenMP> {
 private:
  using MDRangePolicy = flare::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using WorkTag       = typename MDRangePolicy::work_tag;

  using Member = typename Policy::member_type;

  using index_type   = typename Policy::index_type;
  using iterate_type = typename flare::detail::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  OpenMPInternal* m_instance;
  const iterate_type m_iter;

  inline void exec_range(const Member ibeg, const Member iend) const {
    FLARE_PRAGMA_IVDEP_IF_ENABLED
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      m_iter(iwork);
    }
  }

  template <class Policy>
  typename std::enable_if_t<std::is_same<typename Policy::schedule_type::type,
                                         flare::Dynamic>::value>
  execute_parallel() const {
#pragma omp parallel for schedule(dynamic, 1) \
    num_threads(m_instance->thread_pool_size())
    FLARE_PRAGMA_IVDEP_IF_ENABLED
    for (index_type iwork = 0; iwork < m_iter.m_rp.m_num_tiles; ++iwork) {
      m_iter(iwork);
    }
  }

  template <class Policy>
  typename std::enable_if<!std::is_same<typename Policy::schedule_type::type,
                                        flare::Dynamic>::value>::type
  execute_parallel() const {
#pragma omp parallel for schedule(static, 1) \
    num_threads(m_instance->thread_pool_size())
    FLARE_PRAGMA_IVDEP_IF_ENABLED
    for (index_type iwork = 0; iwork < m_iter.m_rp.m_num_tiles; ++iwork) {
      m_iter(iwork);
    }
  }

 public:
  inline void execute() const {
#ifndef FLARE_COMPILER_INTEL
    if (execute_in_serial(m_iter.m_rp.space())) {
      exec_range(0, m_iter.m_rp.m_num_tiles);
      return;
    }
#endif

#ifndef FLARE_INTERNAL_DISABLE_NATIVE_OPENMP
    execute_parallel<Policy>();
#else
    constexpr bool is_dynamic =
        std::is_same<typename Policy::schedule_type::type,
                     flare::Dynamic>::value;

#pragma omp parallel num_threads(m_instance->thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      data.set_work_partition(m_iter.m_rp.m_num_tiles, 1);

      if (is_dynamic) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      std::pair<int64_t, int64_t> range(0, 0);

      do {
        range = is_dynamic ? data.get_work_stealing_chunk()
                           : data.get_work_partition();

        exec_range(range.first, range.second);

      } while (is_dynamic && 0 <= range.first);
    }
    // END #pragma omp parallel
#endif
  }

  inline ParallelFor(const FunctorType& arg_functor, MDRangePolicy arg_policy)
      : m_instance(nullptr), m_iter(arg_policy, arg_functor) {
    m_instance = arg_policy.space().impl_internal_space_instance();
  }
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
};

}  // namespace detail
}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, flare::TeamPolicy<Properties...>,
                  flare::OpenMP> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy =
      flare::detail::TeamPolicyInternal<flare::OpenMP, Properties...>;
  using WorkTag  = typename Policy::work_tag;
  using SchedTag = typename Policy::schedule_type::type;
  using Member   = typename Policy::member_type;

  OpenMPInternal* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const size_t m_shmem_size;

  template <class TagType>
  inline static std::enable_if_t<(std::is_void<TagType>::value)> exec_team(
      const FunctorType& functor, HostThreadTeamData& data,
      const int league_rank_begin, const int league_rank_end,
      const int league_size) {
    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(Member(data, r, league_size));

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

  template <class TagType>
  inline static std::enable_if_t<(!std::is_void<TagType>::value)> exec_team(
      const FunctorType& functor, HostThreadTeamData& data,
      const int league_rank_begin, const int league_rank_end,
      const int league_size) {
    const TagType t{};

    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(t, Member(data, r, league_size));

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

 public:
  inline void execute() const {
    enum { is_dynamic = std::is_same<SchedTag, flare::Dynamic>::value };

    const size_t pool_reduce_size  = 0;  // Never shrinks
    const size_t team_reduce_size  = TEAM_REDUCE_SIZE * m_policy.team_size();
    const size_t team_shared_size  = m_shmem_size;
    const size_t thread_local_size = 0;  // Never shrinks

    m_instance->acquire_lock();

    m_instance->resize_thread_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    if (execute_in_serial(m_policy.space())) {
      ParallelFor::template exec_team<WorkTag>(
          m_functor, *(m_instance->get_thread_data()), 0,
          m_policy.league_size(), m_policy.league_size());

      m_instance->release_lock();

      return;
    }

#pragma omp parallel num_threads(m_instance->thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const int active = data.organize_team(m_policy.team_size());

      if (active) {
        data.set_work_partition(
            m_policy.league_size(),
            (0 < m_policy.chunk_size() ? m_policy.chunk_size()
                                       : m_policy.team_iter()));
      }

      if (is_dynamic) {
        // Must synchronize to make sure each team has set its
        // partition before beginning the work stealing loop.
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      if (active) {
        std::pair<int64_t, int64_t> range(0, 0);

        do {
          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelFor::template exec_team<WorkTag>(m_functor, data, range.first,
                                                   range.second,
                                                   m_policy.league_size());

        } while (is_dynamic && 0 <= range.first);
      }

      data.disband_team();
    }

    m_instance->release_lock();
  }

  inline ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_instance(nullptr),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_shmem_size(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, arg_policy.team_size())) {
    m_instance = arg_policy.space().impl_internal_space_instance();
  }
};

}  // namespace detail
}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#undef FLARE_PRAGMA_IVDEP_IF_ENABLED
#undef FLARE_OPENMP_OPTIONAL_CHUNK_SIZE

#endif  // FLARE_BACKEND_OPENMP_OPENMP_PARALLEL_FOR_H_
