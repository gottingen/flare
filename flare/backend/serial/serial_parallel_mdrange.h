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

#ifndef FLARE_BACKEND_SERIAL_SERIAL_PARALLEL_MDRANGE_H_
#define FLARE_BACKEND_SERIAL_SERIAL_PARALLEL_MDRANGE_H_

#include <flare/core/parallel/parallel.h>
#include <flare/core/policy/exp_mdrange_policy.h>

namespace flare {
namespace detail {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, flare::MDRangePolicy<Traits...>,
                  flare::Serial> {
 private:
  using MDRangePolicy = flare::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using iterate_type = typename flare::detail::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  const iterate_type m_iter;

  void exec() const {
    const typename Policy::member_type e = m_iter.m_rp.m_num_tiles;
    for (typename Policy::member_type i = 0; i < e; ++i) {
      m_iter(i);
    }
  }

 public:
  inline void execute() const { this->exec(); }
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
  inline ParallelFor(const FunctorType& arg_functor,
                     const MDRangePolicy& arg_policy)
      : m_iter(arg_policy, arg_functor) {}
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType,
                     flare::MDRangePolicy<Traits...>, flare::Serial> {
 private:
  using MDRangePolicy = flare::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using FunctorType   = typename CombinedFunctorReducerType::functor_type;
  using ReducerType   = typename CombinedFunctorReducerType::reducer_type;

  using WorkTag = typename MDRangePolicy::work_tag;

  using pointer_type   = typename ReducerType::pointer_type;
  using value_type     = typename ReducerType::value_type;
  using reference_type = typename ReducerType::reference_type;

  using iterate_type = typename flare::detail::HostIterateTile<
      MDRangePolicy, CombinedFunctorReducerType, WorkTag, reference_type>;
  const iterate_type m_iter;
  const pointer_type m_result_ptr;

  inline void exec(reference_type update) const {
    const typename Policy::member_type e = m_iter.m_rp.m_num_tiles;
    for (typename Policy::member_type i = 0; i < e; ++i) {
      m_iter(i, update);
    }
  }

 public:
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
  inline void execute() const {
    const ReducerType& reducer     = m_iter.m_func.get_reducer();
    const size_t pool_reduce_size  = reducer.value_size();
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    auto* internal_instance =
        m_iter.m_rp.space().impl_internal_space_instance();
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

    reference_type update = reducer.init(ptr);

    this->exec(update);

    reducer.final(ptr);
  }

  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType& arg_functor_reducer,
                 const MDRangePolicy& arg_policy,
                 const ViewType& arg_result_view)
      : m_iter(arg_policy, arg_functor_reducer),
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

}  // namespace detail
}  // namespace flare

#endif  // FLARE_BACKEND_SERIAL_SERIAL_PARALLEL_MDRANGE_H_
