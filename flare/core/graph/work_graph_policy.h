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

#ifndef FLARE_CORE_GRAPH_WORK_GRAPH_POLICY_H_
#define FLARE_CORE_GRAPH_WORK_GRAPH_POLICY_H_

#include <flare/core/policy/analyze_policy.h>
#include <flare/core/tensor/crs.h>

namespace flare::detail {

    template<class functor_type, class execution_space, class... policy_args>
    class WorkGraphExec;

}  // namespace flare::detail

namespace flare {

    template<class... Properties>
    class WorkGraphPolicy : public flare::detail::PolicyTraits<Properties...> {
    public:
        using execution_policy = WorkGraphPolicy<Properties...>;
        using self_type = WorkGraphPolicy<Properties...>;
        using traits = flare::detail::PolicyTraits<Properties...>;
        using index_type = typename traits::index_type;
        using member_type = index_type;
        using execution_space = typename traits::execution_space;
        using memory_space = typename execution_space::memory_space;
        using graph_type = flare::Crs<index_type, execution_space, void, index_type>;

        enum : std::int32_t {
            END_TOKEN = -1,
            BEGIN_TOKEN = -2,
            COMPLETED_TOKEN = -3
        };

    private:
        using ints_type = flare::View<std::int32_t *, memory_space>;

        // Let N = m_graph.numRows(), the total work
        // m_queue[  0 ..   N-1] = the ready queue
        // m_queue[  N .. 2*N-1] = the waiting queue counts
        // m_queue[2*N .. 2*N+2] = the ready queue hints

        graph_type const m_graph;
        ints_type m_queue;

        FLARE_INLINE_FUNCTION
        void push_work(const std::int32_t w) const noexcept {
            const std::int32_t N = m_graph.numRows();

            std::int32_t *const ready_queue = &m_queue[0];
            std::int32_t *const end_hint = &m_queue[2 * N + 1];

            // Push work to end of queue
            const std::int32_t j = atomic_fetch_add(end_hint, 1);

            if ((N <= j) || (END_TOKEN != atomic_exchange(ready_queue + j, w))) {
                // ERROR: past the end of queue or did not replace END_TOKEN
                flare::abort("WorkGraphPolicy push_work error");
            }

            memory_fence();
        }

    public:
        /**\brief  Attempt to pop the work item at the head of the queue.
         *
         *  Find entry 'i' such that
         *    ( m_queue[i] != BEGIN_TOKEN ) AND
         *    ( i == 0 OR m_queue[i-1] == BEGIN_TOKEN )
         *  if found then
         *    increment begin hint
         *    return atomic_exchange( m_queue[i] , BEGIN_TOKEN )
         *  else if i < total work
         *    return END_TOKEN
         *  else
         *    return COMPLETED_TOKEN
         *
         */
        FLARE_INLINE_FUNCTION
        std::int32_t pop_work() const noexcept {
            const std::int32_t N = m_graph.numRows();

            std::int32_t *const ready_queue = &m_queue[0];
            std::int32_t *const begin_hint = &m_queue[2 * N];

            // begin hint is guaranteed to be less than or equal to
            // actual begin location in the queue.

            for (std::int32_t i = flare::atomic_load(begin_hint); i < N; ++i) {
                const std::int32_t w = flare::atomic_load(&ready_queue[i]);

                if (w == END_TOKEN) {
                    return END_TOKEN;
                }

                if ((w != BEGIN_TOKEN) &&
                    (w == atomic_compare_exchange(ready_queue + i, w,
                                                  (std::int32_t) BEGIN_TOKEN))) {
                    // Attempt to claim ready work index succeeded,
                    // update the hint and return work index
                    atomic_increment(begin_hint);
                    return w;
                }
                // arrive here when ready_queue[i] == BEGIN_TOKEN
            }

            return COMPLETED_TOKEN;
        }

        FLARE_INLINE_FUNCTION
        void completed_work(std::int32_t w) const noexcept {
            flare::memory_fence();

            // Make sure the completed work function's memory accesses are flushed.

            const std::int32_t N = m_graph.numRows();

            std::int32_t *const count_queue = &m_queue[N];

            const std::int32_t B = m_graph.row_map(w);
            const std::int32_t E = m_graph.row_map(w + 1);

            for (std::int32_t i = B; i < E; ++i) {
                const std::int32_t j = m_graph.entries(i);
                if (1 == atomic_fetch_add(count_queue + j, -1)) {
                    push_work(j);
                }
            }
        }

        struct TagInit {
        };
        struct TagCount {
        };
        struct TagReady {
        };

        /**\brief  Initialize queue
         *
         *  m_queue[0..N-1] = END_TOKEN, the ready queue
         *  m_queue[N..2*N-1] = 0, the waiting count queue
         *  m_queue[2*N..2*N+1] = 0, begin/end hints for ready queue
         */
        FLARE_INLINE_FUNCTION
        void operator()(const TagInit, int i) const noexcept {
            m_queue[i] = i < m_graph.numRows() ? END_TOKEN : 0;
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TagCount, int i) const noexcept {
            std::int32_t *const count_queue = &m_queue[m_graph.numRows()];

            atomic_increment(count_queue + m_graph.entries[i]);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TagReady, int w) const noexcept {
            std::int32_t const *const count_queue = &m_queue[m_graph.numRows()];

            if (0 == count_queue[w]) push_work(w);
        }

        execution_space space() const { return execution_space(); }

        WorkGraphPolicy(const graph_type &arg_graph)
                : m_graph(arg_graph),
                  m_queue(view_alloc("queue", WithoutInitializing),
                          arg_graph.numRows() * 2 + 2) {
            {  // Initialize
                using policy_type = RangePolicy<std::int32_t, execution_space, TagInit>;
                using closure_type = flare::detail::ParallelFor<self_type, policy_type>;
                const closure_type closure(*this, policy_type(0, m_queue.size()));
                closure.execute();
                execution_space().fence(
                        "flare::WorkGraphPolicy::WorkGraphPolicy: fence after executing "
                        "graph init");
            }

            {  // execute-after counts
                using policy_type = RangePolicy<std::int32_t, execution_space, TagCount>;
                using closure_type = flare::detail::ParallelFor<self_type, policy_type>;
                const closure_type closure(*this, policy_type(0, m_graph.entries.size()));
                closure.execute();
                execution_space().fence(
                        "flare::WorkGraphPolicy::WorkGraphPolicy: fence after executing "
                        "graph count");
            }

            {  // Scheduling ready tasks
                using policy_type = RangePolicy<std::int32_t, execution_space, TagReady>;
                using closure_type = flare::detail::ParallelFor<self_type, policy_type>;
                const closure_type closure(*this, policy_type(0, m_graph.numRows()));
                closure.execute();
                execution_space().fence(
                        "flare::WorkGraphPolicy::WorkGraphPolicy: fence after executing "
                        "readied graph");
            }
        }
    };

}  // namespace flare

#ifdef FLARE_ENABLE_SERIAL

#include <flare/backend/serial/serial_work_graph_policy.h>

#endif

#ifdef FLARE_ENABLE_OPENMP
#include <flare/backend/openmp/openmp_work_graph_policy.h>
#endif

#ifdef FLARE_ON_CUDA_DEVICE
#include <flare/backend/cuda/cuda_work_graph_policy.h>
#endif

#ifdef FLARE_ENABLE_THREADS
#include <flare/backend/threads/threads_work_graph_policy.h>
#endif


#endif  // FLARE_CORE_GRAPH_WORK_GRAPH_POLICY_H_
