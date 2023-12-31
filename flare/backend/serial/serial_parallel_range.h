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

#ifndef FLARE_BACKEND_SERIAL_SERIAL_PARALLEL_RANGE_H_
#define FLARE_BACKEND_SERIAL_SERIAL_PARALLEL_RANGE_H_

#include <flare/core/parallel/parallel.h>

namespace flare {
namespace detail {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, flare::RangePolicy<Traits...>, flare::Serial> {
 private:
  using Policy = flare::RangePolicy<Traits...>;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  std::enable_if_t<std::is_void<TagType>::value> exec() const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i);
    }
  }

  template <class TagType>
  std::enable_if_t<!std::is_void<TagType>::value> exec() const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i);
    }
  }

 public:
  inline void execute() const {
    this->template exec<typename Policy::work_tag>();
  }

  inline ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

/*--------------------------------------------------------------------------*/

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType, flare::RangePolicy<Traits...>,
                     flare::Serial> {
 private:
  using Policy      = flare::RangePolicy<Traits...>;
  using WorkTag     = typename Policy::work_tag;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using pointer_type   = typename ReducerType::pointer_type;
  using reference_type = typename ReducerType::reference_type;

  const CombinedFunctorReducerType m_functor_reducer;
  const Policy m_policy;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline std::enable_if_t<std::is_void<TagType>::value> exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor_reducer.get_functor()(i, update);
    }
  }

  template <class TagType>
  inline std::enable_if_t<!std::is_void<TagType>::value> exec(
      reference_type update) const {
    const TagType t{};

    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor_reducer.get_functor()(t, i, update);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size =
        m_functor_reducer.get_reducer().value_size();
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    auto* internal_instance = m_policy.space().impl_internal_space_instance();
    // Need to lock resize_thread_team_data
    std::lock_guard<std::mutex> lock(
        internal_instance->m_thread_team_data_mutex);
    internal_instance->resize_thread_team_data(
        pool_reduce_size, team_reduce_size, team_shared_size,
        thread_local_size);

    pointer_type ptr =
        m_result_ptr
            ? m_result_ptr
            : pointer_type(
                  internal_instance->m_thread_team_data.pool_reduce_local());

    reference_type update = m_functor_reducer.get_reducer().init(ptr);

    this->template exec<WorkTag>(update);

    m_functor_reducer.get_reducer().final(ptr);
  }

  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType& arg_functor_reducer,
                 const Policy& arg_policy, const ViewType& arg_result_view)
      : m_functor_reducer(arg_functor_reducer),
        m_policy(arg_policy),
        m_result_ptr(arg_result_view.data()) {
    static_assert(flare::is_view<ViewType>::value,
                  "flare::Serial reduce result must be a View");

    static_assert(
        flare::detail::MemorySpaceAccess<typename ViewType::memory_space,
                                        flare::HostSpace>::accessible,
        "flare::Serial reduce result must be a View accessible from "
        "HostSpace");
  }
};

/*--------------------------------------------------------------------------*/

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, flare::RangePolicy<Traits...>,
                   flare::Serial> {
 private:
  using Policy  = flare::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType, void>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const CombinedFunctorReducer<FunctorType, typename Analysis::Reducer>
      m_functor_reducer;
  const Policy m_policy;

  template <class TagType>
  inline std::enable_if_t<std::is_void<TagType>::value> exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor_reducer.get_functor()(i, update, true);
    }
  }

  template <class TagType>
  inline std::enable_if_t<!std::is_void<TagType>::value> exec(
      reference_type update) const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor_reducer.get_functor()(t, i, update, true);
    }
  }

 public:
  inline void execute() const {
    const typename Analysis::Reducer& final_reducer =
        m_functor_reducer.get_reducer();
    const size_t pool_reduce_size  = final_reducer.value_size();
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    // Need to lock resize_thread_team_data
    auto* internal_instance = m_policy.space().impl_internal_space_instance();
    std::lock_guard<std::mutex> lock(
        internal_instance->m_thread_team_data_mutex);
    internal_instance->resize_thread_team_data(
        pool_reduce_size, team_reduce_size, team_shared_size,
        thread_local_size);

    reference_type update = final_reducer.init(pointer_type(
        internal_instance->m_thread_team_data.pool_reduce_local()));

    this->template exec<WorkTag>(update);
  }

  inline ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor_reducer(arg_functor, typename Analysis::Reducer{arg_functor}),
        m_policy(arg_policy) {}
};

/*--------------------------------------------------------------------------*/
template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, flare::RangePolicy<Traits...>,
                            ReturnType, flare::Serial> {
 private:
  using Policy  = flare::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using Analysis = FunctorAnalysis<FunctorPatternInterface::SCAN, Policy,
                                   FunctorType, ReturnType>;

  using value_type     = typename Analysis::value_type;
  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const CombinedFunctorReducer<FunctorType, typename Analysis::Reducer>
      m_functor_reducer;
  const Policy m_policy;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline std::enable_if_t<std::is_void<TagType>::value> exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor_reducer.get_functor()(i, update, true);
    }
  }

  template <class TagType>
  inline std::enable_if_t<!std::is_void<TagType>::value> exec(
      reference_type update) const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor_reducer.get_functor()(t, i, update, true);
    }
  }

 public:
  inline void execute() {
    const size_t pool_reduce_size =
        m_functor_reducer.get_reducer().value_size();
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    // Need to lock resize_thread_team_data
    auto* internal_instance = m_policy.space().impl_internal_space_instance();
    std::lock_guard<std::mutex> lock(
        internal_instance->m_thread_team_data_mutex);
    internal_instance->resize_thread_team_data(
        pool_reduce_size, team_reduce_size, team_shared_size,
        thread_local_size);

    const typename Analysis::Reducer& final_reducer =
        m_functor_reducer.get_reducer();

    reference_type update = final_reducer.init(pointer_type(
        internal_instance->m_thread_team_data.pool_reduce_local()));

    this->template exec<WorkTag>(update);

    *m_result_ptr = update;
  }

  template <class ViewType,
            class Enable = std::enable_if_t<flare::is_view<ViewType>::value>>
  ParallelScanWithTotal(const FunctorType& arg_functor,
                        const Policy& arg_policy,
                        const ViewType& arg_result_view)
      : m_functor_reducer(arg_functor, typename Analysis::Reducer{arg_functor}),
        m_policy(arg_policy),
        m_result_ptr(arg_result_view.data()) {
    static_assert(
        flare::detail::MemorySpaceAccess<typename ViewType::memory_space,
                                        flare::HostSpace>::accessible,
        "flare::Serial parallel_scan result must be host-accessible!");
  }
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_BACKEND_SERIAL_SERIAL_PARALLEL_RANGE_H_
