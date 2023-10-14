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

#ifndef FLARE_BACKEND_THREADS_THREADS_PARALLEL_TEAM_H_
#define FLARE_BACKEND_THREADS_THREADS_PARALLEL_TEAM_H_

#include <flare/core/parallel/parallel.h>

namespace flare {
namespace detail {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, flare::TeamPolicy<Properties...>,
                  flare::Threads> {
 private:
  using Policy =
      flare::detail::TeamPolicyInternal<flare::Threads, Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const size_t m_shared;

  template <class TagType, class Schedule>
  inline static std::enable_if_t<std::is_void<TagType>::value &&
                                 std::is_same<Schedule, flare::Static>::value>
  exec_team(const FunctorType &functor, Member member) {
    for (; member.valid_static(); member.next_static()) {
      functor(member);
    }
  }

  template <class TagType, class Schedule>
  inline static std::enable_if_t<!std::is_void<TagType>::value &&
                                 std::is_same<Schedule, flare::Static>::value>
  exec_team(const FunctorType &functor, Member member) {
    const TagType t{};
    for (; member.valid_static(); member.next_static()) {
      functor(t, member);
    }
  }

  template <class TagType, class Schedule>
  inline static std::enable_if_t<std::is_void<TagType>::value &&
                                 std::is_same<Schedule, flare::Dynamic>::value>
  exec_team(const FunctorType &functor, Member member) {
    for (; member.valid_dynamic(); member.next_dynamic()) {
      functor(member);
    }
  }

  template <class TagType, class Schedule>
  inline static std::enable_if_t<!std::is_void<TagType>::value &&
                                 std::is_same<Schedule, flare::Dynamic>::value>
  exec_team(const FunctorType &functor, Member member) {
    const TagType t{};
    for (; member.valid_dynamic(); member.next_dynamic()) {
      functor(t, member);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    ParallelFor::exec_team<WorkTag, typename Policy::schedule_type::type>(
        self.m_functor, Member(&exec, self.m_policy, self.m_shared));

    exec.barrier();
    exec.fan_in();
  }
  template <typename Policy>
  Policy fix_policy(Policy policy) {
    if (policy.impl_vector_length() < 0) {
      policy.impl_set_vector_length(1);
    }
    if (policy.team_size() < 0) {
      policy.impl_set_team_size(
          policy.team_size_recommended(m_functor, ParallelForTag{}));
    }
    return policy;
  }

 public:
  inline void execute() const {
    ThreadsExec::resize_scratch(
        0, Policy::member_type::team_reduce_size() + m_shared);

    ThreadsExec::start(&ParallelFor::exec, this);

    ThreadsExec::fence();
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor),
        m_policy(fix_policy(arg_policy)),
        m_shared(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                     arg_functor, m_policy.team_size())) {}
};

template <class CombinedFunctorReducerType, class... Properties>
class ParallelReduce<CombinedFunctorReducerType,
                     flare::TeamPolicy<Properties...>, flare::Threads> {
 private:
  using Policy =
      flare::detail::TeamPolicyInternal<flare::Threads, Properties...>;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;
  using WorkTag     = typename Policy::work_tag;
  using Member      = typename Policy::member_type;

  using pointer_type   = typename ReducerType::pointer_type;
  using reference_type = typename ReducerType::reference_type;

  const CombinedFunctorReducerType m_functor_reducer;
  const Policy m_policy;
  const pointer_type m_result_ptr;
  const size_t m_shared;

  template <class TagType>
  inline static std::enable_if_t<std::is_void<TagType>::value> exec_team(
      const FunctorType &functor, Member member, reference_type update) {
    for (; member.valid_static(); member.next_static()) {
      functor(member, update);
    }
  }

  template <class TagType>
  inline static std::enable_if_t<!std::is_void<TagType>::value> exec_team(
      const FunctorType &functor, Member member, reference_type update) {
    const TagType t{};
    for (; member.valid_static(); member.next_static()) {
      functor(t, member, update);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);

    ParallelReduce::template exec_team<WorkTag>(
        self.m_functor_reducer.get_functor(),
        Member(&exec, self.m_policy, self.m_shared),
        self.m_functor_reducer.get_reducer().init(
            static_cast<pointer_type>(exec.reduce_memory())));

    exec.fan_in_reduce(self.m_functor_reducer.get_reducer());
  }

 public:
  inline void execute() const {
    const ReducerType &reducer = m_functor_reducer.get_reducer();

    if (m_policy.league_size() * m_policy.team_size() == 0) {
      if (m_result_ptr) {
        reducer.init(m_result_ptr);
        reducer.final(m_result_ptr);
      }
    } else {
      ThreadsExec::resize_scratch(
          reducer.value_size(),
          Policy::member_type::team_reduce_size() + m_shared);

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
  }

  template <typename Policy>
  Policy fix_policy(Policy policy) {
    if (policy.impl_vector_length() < 0) {
      policy.impl_set_vector_length(1);
    }
    if (policy.team_size() < 0) {
      policy.impl_set_team_size(policy.team_size_recommended(
          m_functor_reducer.get_functor(), m_functor_reducer.get_reducer(),
          ParallelReduceTag{}));
    }
    return policy;
  }

  template <class TensorType>
  inline ParallelReduce(const CombinedFunctorReducerType &arg_functor_reducer,
                        const Policy &arg_policy, const TensorType &arg_result)
      : m_functor_reducer(arg_functor_reducer),
        m_policy(fix_policy(arg_policy)),
        m_result_ptr(arg_result.data()),
        m_shared(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                     arg_functor_reducer.get_functor(), m_policy.team_size())) {
    static_assert(
        flare::detail::MemorySpaceAccess<typename TensorType::memory_space,
                                        flare::HostSpace>::accessible,
        "flare::Threads reduce result must be a Tensor accessible from "
        "HostSpace");
  }
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_BACKEND_THREADS_THREADS_PARALLEL_TEAM_H_
