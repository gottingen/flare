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

#ifndef FLARE_BACKEND_OPENMP_OPENMP_PARALLEL_SCAN_H_
#define FLARE_BACKEND_OPENMP_OPENMP_PARALLEL_SCAN_H_

#include <omp.h>
#include <flare/backend/openmp/openmp_instance.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, flare::RangePolicy<Traits...>,
                   flare::OpenMP> {
 private:
  using Policy = flare::RangePolicy<Traits...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType, void>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OpenMPInternal* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static std::enable_if_t<std::is_void<TagType>::value> exec_range(
      const FunctorType& functor, const Member ibeg, const Member iend,
      reference_type update, const bool final) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update, final);
    }
  }

  template <class TagType>
  inline static std::enable_if_t<!std::is_void<TagType>::value> exec_range(
      const FunctorType& functor, const Member ibeg, const Member iend,
      reference_type update, const bool final) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update, final);
    }
  }

 public:
  inline void execute() const {
    const int value_count          = Analysis::value_count(m_functor);
    const size_t pool_reduce_bytes = 2 * Analysis::value_size(m_functor);

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

    if (execute_in_serial(m_policy.space())) {
      typename Analysis::Reducer final_reducer(m_functor);

      reference_type update = final_reducer.init(
          pointer_type(m_instance->get_thread_data(0)->pool_reduce_local()));

      ParallelScan::template exec_range<WorkTag>(m_functor, m_policy.begin(),
                                                 m_policy.end(), update, true);

      return;
    }

#pragma omp parallel num_threads(m_instance->thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());
      typename Analysis::Reducer final_reducer(m_functor);

      const WorkRange range(m_policy, omp_get_thread_num(),
                            omp_get_num_threads());

      reference_type update_sum = final_reducer.init(
          reinterpret_cast<pointer_type>(data.pool_reduce_local()));

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_sum, false);

      if (data.pool_rendezvous()) {
        pointer_type ptr_prev = nullptr;

        const int n = omp_get_num_threads();

        for (int i = 0; i < n; ++i) {
          pointer_type ptr =
              (pointer_type)data.pool_member(i)->pool_reduce_local();

          if (i) {
            for (int j = 0; j < value_count; ++j) {
              ptr[j + value_count] = ptr_prev[j + value_count];
            }
            final_reducer.join(ptr + value_count, ptr_prev);
          } else {
            final_reducer.init(ptr + value_count);
          }

          ptr_prev = ptr;
        }

        data.pool_rendezvous_release();
      }

      reference_type update_base = final_reducer.reference(
          reinterpret_cast<pointer_type>(data.pool_reduce_local()) +
          value_count);

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_base, true);
    }
  }

  //----------------------------------------

  inline ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_instance(nullptr), m_functor(arg_functor), m_policy(arg_policy) {
    m_instance = arg_policy.space().impl_internal_space_instance();
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, flare::RangePolicy<Traits...>,
                            ReturnType, flare::OpenMP> {
 private:
  using Policy = flare::RangePolicy<Traits...>;

  using Analysis = FunctorAnalysis<FunctorPatternInterface::SCAN, Policy,
                                   FunctorType, ReturnType>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using value_type     = typename Analysis::value_type;
  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OpenMPInternal* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static std::enable_if_t<std::is_void<TagType>::value> exec_range(
      const FunctorType& functor, const Member ibeg, const Member iend,
      reference_type update, const bool final) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update, final);
    }
  }

  template <class TagType>
  inline static std::enable_if_t<!std::is_void<TagType>::value> exec_range(
      const FunctorType& functor, const Member ibeg, const Member iend,
      reference_type update, const bool final) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update, final);
    }
  }

 public:
  inline void execute() const {
    const int value_count          = Analysis::value_count(m_functor);
    const size_t pool_reduce_bytes = 2 * Analysis::value_size(m_functor);

    m_instance->acquire_lock();

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

    if (execute_in_serial(m_policy.space())) {
      typename Analysis::Reducer final_reducer(m_functor);

      reference_type update = final_reducer.init(
          pointer_type(m_instance->get_thread_data(0)->pool_reduce_local()));

      this->template exec_range<WorkTag>(m_functor, m_policy.begin(),
                                         m_policy.end(), update, true);

      *m_result_ptr = update;

      m_instance->release_lock();

      return;
    }

#pragma omp parallel num_threads(m_instance->thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());
      typename Analysis::Reducer final_reducer(m_functor);

      const WorkRange range(m_policy, omp_get_thread_num(),
                            omp_get_num_threads());
      reference_type update_sum = final_reducer.init(
          reinterpret_cast<pointer_type>(data.pool_reduce_local()));

      ParallelScanWithTotal::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_sum, false);

      if (data.pool_rendezvous()) {
        pointer_type ptr_prev = nullptr;

        const int n = omp_get_num_threads();

        for (int i = 0; i < n; ++i) {
          pointer_type ptr =
              (pointer_type)data.pool_member(i)->pool_reduce_local();

          if (i) {
            for (int j = 0; j < value_count; ++j) {
              ptr[j + value_count] = ptr_prev[j + value_count];
            }
            final_reducer.join(ptr + value_count, ptr_prev);
          } else {
            final_reducer.init(ptr + value_count);
          }

          ptr_prev = ptr;
        }

        data.pool_rendezvous_release();
      }

      reference_type update_base = final_reducer.reference(
          reinterpret_cast<pointer_type>(data.pool_reduce_local()) +
          value_count);

      ParallelScanWithTotal::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_base, true);

      if (omp_get_thread_num() == omp_get_num_threads() - 1) {
        *m_result_ptr = update_base;
      }
    }

    m_instance->release_lock();
  }

  //----------------------------------------

  template <class TensorType>
  ParallelScanWithTotal(const FunctorType& arg_functor,
                        const Policy& arg_policy,
                        const TensorType& arg_result_tensor)
      : m_instance(nullptr),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_result_ptr(arg_result_tensor.data()) {
    static_assert(
        flare::detail::MemorySpaceAccess<typename TensorType::memory_space,
                                        flare::HostSpace>::accessible,
        "flare::OpenMP parallel_scan result must be host-accessible!");
    m_instance = arg_policy.space().impl_internal_space_instance();
  }

  //----------------------------------------
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_BACKEND_OPENMP_OPENMP_PARALLEL_SCAN_H_
