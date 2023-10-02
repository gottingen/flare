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

#ifndef FLARE_CORE_GRAPH_GRAPH_H_
#define FLARE_CORE_GRAPH_GRAPH_H_

#include <flare/core/defines.h>
#include <flare/core/common/error.h>  // FLARE_EXPECTS

#include <flare/core/graph/graph_fwd.h>
#include <flare/core/graph/graph_impl_fwd.h>

// GraphAccess needs to be defined, not just declared
#include <flare/core/graph/graph_impl.h>

#include <functional>
#include <memory>

namespace flare::experimental {

    template<class ExecutionSpace>
    struct [[nodiscard]] Graph {
    public:

        using execution_space = ExecutionSpace;
        using graph = Graph;


    private:

        friend struct flare::detail::GraphAccess;

        using impl_t = flare::detail::GraphImpl<ExecutionSpace>;
        std::shared_ptr<impl_t> m_impl_ptr = nullptr;

        // Note: only create_graph() uses this constructor, but we can't just make
        // that a friend instead of GraphAccess because of the way that friend
        // function template injection works.
        explicit Graph(std::shared_ptr<impl_t> arg_impl_ptr)
                : m_impl_ptr(std::move(arg_impl_ptr)) {}

    public:
        ExecutionSpace const &get_execution_space() const {
            return m_impl_ptr->get_execution_space();
        }

        void submit() const {
            FLARE_EXPECTS(bool(m_impl_ptr))
            (*m_impl_ptr).submit();
        }
    };

    template<class... PredecessorRefs>
    // constraints (not intended for subsumption, though...)
    //   ((remove_cvref_t<PredecessorRefs> is a specialization of
    //        GraphNodeRef with get_root().get_graph_impl() as its GraphImpl)
    //      && ...)
    auto when_all(PredecessorRefs &&... arg_pred_refs) {
        // TODO @graph @desul-integration check the constraints and preconditions
        //                                once we have folded conjunctions from
        //                                desul
        static_assert(sizeof...(PredecessorRefs) > 0,
                      "when_all() needs at least one predecessor.");
        auto graph_ptr_impl =
                flare::detail::GraphAccess::get_graph_weak_ptr(
                        std::get<0>(std::forward_as_tuple(arg_pred_refs...)))
                        .lock();
        auto node_ptr_impl = graph_ptr_impl->create_aggregate_ptr(arg_pred_refs...);
        graph_ptr_impl->add_node(node_ptr_impl);
        (graph_ptr_impl->add_predecessor(node_ptr_impl, arg_pred_refs), ...);
        return flare::detail::GraphAccess::make_graph_node_ref(
                std::move(graph_ptr_impl), std::move(node_ptr_impl));
    }

    template<class ExecutionSpace, class Closure>
    Graph<ExecutionSpace> create_graph(ExecutionSpace ex, Closure &&arg_closure) {
        // Create a shared pointer to the graph:
        // We need an attorney class here so we have an implementation friend to
        // create a Graph class without graph having public constructors. We can't
        // just make `create_graph` itself a friend because of the way that friend
        // function template injection works.
        auto rv = flare::detail::GraphAccess::construct_graph(std::move(ex));
        // Invoke the user's graph construction closure
        ((Closure &&) arg_closure)(flare::detail::GraphAccess::create_root_ref(rv));
        // and given them back the graph
        // FLARE_ENSURES(rv.m_impl_ptr.use_count() == 1)
        return rv;
    }

    template<
            class ExecutionSpace = DefaultExecutionSpace,
            class Closure = flare::detail::DoNotExplicitlySpecifyThisTemplateParameter>
    Graph<ExecutionSpace> create_graph(Closure &&arg_closure) {
        return create_graph(ExecutionSpace{}, (Closure &&) arg_closure);
    }

}  // namespace flare::experimental

// Even though these things are separable, include them here for now so that
// the user only needs to include graph.h to get the whole facility.
#include <flare/core/graph/graph_node.h>
#include <flare/core/graph/graph_node_impl.h>
#include <flare/core/graph/default_graph_impl.h>
#include <flare/backend/cuda/cuda_graph_impl.h>

#endif  // FLARE_CORE_GRAPH_GRAPH_H_
