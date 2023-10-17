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

#ifndef FLARE_BACKEND_THREADS_THREADS_PARALLEL_MDRANGE_H_
#define FLARE_BACKEND_THREADS_THREADS_PARALLEL_MDRANGE_H_

#include <flare/core/parallel/parallel.h>
#include <flare/backend/threads/threads_exec.h>
#include <flare/core/policy/exp_mdrange_policy.h>

namespace flare {
namespace detail {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, flare::MDRangePolicy<Traits...>,
                  flare::Threads> {
 private:
  using MDRangePolicy = flare::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using WorkTag = typename MDRangePolicy::work_tag;

  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using iterate_type = typename flare::detail::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  const iterate_type m_iter;

  inline void exec_range(const Member ibeg, const Member iend) const {
    for (Member i = ibeg; i < iend; ++i) {
      m_iter(i);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    exec_schedule<typename Policy::schedule_type::type>(exec, arg);
  }

  template <class Schedule>
  static std::enable_if_t<std::is_same<Schedule, flare::Static>::value>
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    auto const num_tiles = self.m_iter.m_rp.m_num_tiles;
    WorkRange range(Policy(0, num_tiles).set_chunk_size(1), exec.pool_rank(),
                    exec.pool_size());

    self.exec_range(range.begin(), range.end());

    exec.fan_in();
  }

  template <class Schedule>
  static std::enable_if_t<std::is_same<Schedule, flare::Dynamic>::value>
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    auto const num_tiles = self.m_iter.m_rp.m_num_tiles;
    WorkRange range(Policy(0, num_tiles).set_chunk_size(1), exec.pool_rank(),
                    exec.pool_size());

    exec.set_work_range(range.begin(), range.end(), 1);
    exec.reset_steal_target();
    exec.barrier();

    long work_index = exec.get_work_index();

    while (work_index != -1) {
      const Member begin = static_cast<Member>(work_index);
      const Member end   = begin + 1 < num_tiles ? begin + 1 : num_tiles;

      self.exec_range(begin, end);
      work_index = exec.get_work_index();
    }

    exec.fan_in();
  }

 public:
  inline void execute() const {
    ThreadsExec::start(&ParallelFor::exec, this);
    ThreadsExec::fence();
  }

  ParallelFor(const FunctorType &arg_functor, const MDRangePolicy &arg_policy)
      : m_iter(arg_policy, arg_functor) {}

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy &, const Functor &) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType,
                     flare::MDRangePolicy<Traits...>, flare::Threads> {
 private:
  using MDRangePolicy = flare::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using FunctorType   = typename CombinedFunctorReducerType::functor_type;
  using ReducerType   = typename CombinedFunctorReducerType::reducer_type;

  using WorkTag   = typename MDRangePolicy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using pointer_type   = typename ReducerType::pointer_type;
  using value_type     = typename ReducerType::value_type;
  using reference_type = typename ReducerType::reference_type;

  using iterate_type = typename flare::detail::HostIterateTile<
      MDRangePolicy, CombinedFunctorReducerType, WorkTag, reference_type>;

  const iterate_type m_iter;
  const pointer_type m_result_ptr;

  inline void exec_range(const Member &ibeg, const Member &iend,
                         reference_type update) const {
    for (Member i = ibeg; i < iend; ++i) {
      m_iter(i, update);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    exec_schedule<typename Policy::schedule_type::type>(exec, arg);
  }

  template <class Schedule>
  static std::enable_if_t<std::is_same<Schedule, flare::Static>::value>
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);

    const auto num_tiles = self.m_iter.m_rp.m_num_tiles;
    const WorkRange range(Policy(0, num_tiles).set_chunk_size(1),
                          exec.pool_rank(), exec.pool_size());

    const ReducerType &reducer = self.m_iter.m_func.get_reducer();
    self.exec_range(
        range.begin(), range.end(),
        reducer.init(static_cast<pointer_type>(exec.reduce_memory())));

    exec.fan_in_reduce(reducer);
  }

  template <class Schedule>
  static std::enable_if_t<std::is_same<Schedule, flare::Dynamic>::value>
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);

    const auto num_tiles = self.m_iter.m_rp.m_num_tiles;
    const WorkRange range(Policy(0, num_tiles).set_chunk_size(1),
                          exec.pool_rank(), exec.pool_size());

    exec.set_work_range(range.begin(), range.end(), 1);
    exec.reset_steal_target();
    exec.barrier();

    long work_index = exec.get_work_index();

    const ReducerType &reducer = self.m_iter.m_func.get_reducer();
    reference_type update =
        self.m_reducer.init(static_cast<pointer_type>(exec.reduce_memory()));
    while (work_index != -1) {
      const Member begin = static_cast<Member>(work_index);
      const Member end   = begin + 1 < num_tiles ? begin + 1 : num_tiles;
      self.exec_range(begin, end, update);
      work_index = exec.get_work_index();
    }

    exec.fan_in_reduce(self.m_reducer);
  }

 public:
  inline void execute() const {
    const ReducerType &reducer = m_iter.m_func.get_reducer();
    ThreadsExec::resize_scratch(reducer.value_size(), 0);

    ThreadsExec::start(&ParallelReduce::exec, this);

    ThreadsExec::fence();

    if (m_result_ptr) {
      const pointer_type data =
          (pointer_type)ThreadsExec::root_reduce_scratch();

      const unsigned n = reducer.value_count();
      for (unsigned i = 0; i < n; ++i) {
        m_result_ptr[i] = data[i];
      }
    }
  }

  template <class TensorType>
  ParallelReduce(const CombinedFunctorReducerType &arg_functor_reducer,
                 const MDRangePolicy &arg_policy,
                 const TensorType &arg_result_tensor)
      : m_iter(arg_policy, arg_functor_reducer),
        m_result_ptr(arg_result_tensor.data()) {
    static_assert(flare::is_tensor<TensorType>::value,
                  "flare::Threads reduce result must be a Tensor");

    static_assert(
        flare::detail::MemorySpaceAccess<typename TensorType::memory_space,
                                        flare::HostSpace>::accessible,
        "flare::Threads reduce result must be a Tensor accessible from "
        "HostSpace");
  }

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy &, const Functor &) {
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

#endif  // FLARE_BACKEND_THREADS_THREADS_PARALLEL_MDRANGE_H_
