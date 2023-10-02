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

#ifndef FLARE_CORE_GRAPH_DEFAULT_GRAPH_IMPL_H_
#define FLARE_CORE_GRAPH_DEFAULT_GRAPH_IMPL_H_

#include <flare/core/policy/exec_policy.h>
#include <flare/core/graph/graph.h>

#include <flare/core/graph/graph_impl_fwd.h>
#include <flare/core/graph/default_graph_fwd.h>

#include <flare/backend/serial/serial.h>
#include <flare/backend/openmp/openmp.h>
// FIXME @graph other backends?

#include <flare/core/common/optional_ref.h>
#include <flare/core/graph/ebo.h>

#include <set>

namespace flare::detail {

    template<class ExecutionSpace>
    struct GraphImpl : private ExecutionSpaceInstanceStorage<ExecutionSpace> {
    public:
        using root_node_impl_t =
                GraphNodeImpl<ExecutionSpace, flare::experimental::TypeErasedTag,
                        flare::experimental::TypeErasedTag>;

    private:
        using execution_space_instance_storage_base_t =
                ExecutionSpaceInstanceStorage<ExecutionSpace>;

        using node_details_t = GraphNodeBackendSpecificDetails<ExecutionSpace>;
        std::set<std::shared_ptr<node_details_t>> m_sinks;

    public:

        // Not moveable or copyable; it spends its whole live as a shared_ptr in the
        // Graph object
        GraphImpl() = default;

        GraphImpl(GraphImpl const &) = delete;

        GraphImpl(GraphImpl &&) = delete;

        GraphImpl &operator=(GraphImpl const &) = delete;

        GraphImpl &operator=(GraphImpl &&) = delete;

        ~GraphImpl() = default;

        explicit GraphImpl(ExecutionSpace arg_space)
                : execution_space_instance_storage_base_t(std::move(arg_space)) {}


        ExecutionSpace const &get_execution_space() const {
            return this
                    ->execution_space_instance_storage_base_t::execution_space_instance();
        }

        template<class NodeImpl>
        //  requires NodeImplPtr is a shared_ptr to specialization of GraphNodeImpl
        void add_node(std::shared_ptr<NodeImpl> const &arg_node_ptr) {
            static_assert(
                    NodeImpl::kernel_type::Policy::is_graph_kernel::value,
                    "Something has gone horribly wrong, but it's too complicated to "
                    "explain here.  Buy Daisy a coffee and she'll explain it to you.");
            // Since this is always called before any calls to add_predecessor involving
            // it, we can treat this node as a sink until we discover otherwise.
            arg_node_ptr->node_details_t::set_kernel(arg_node_ptr->get_kernel());
            auto spot = m_sinks.find(arg_node_ptr);
            FLARE_ASSERT(spot == m_sinks.end())
            m_sinks.insert(std::move(spot), std::move(arg_node_ptr));
        }

        template<class NodeImplPtr, class PredecessorRef>
        // requires PredecessorRef is a specialization of GraphNodeRef that has
        // already been added to this graph and NodeImpl is a specialization of
        // GraphNodeImpl that has already been added to this graph.
        void add_predecessor(NodeImplPtr arg_node_ptr, PredecessorRef arg_pred_ref) {
            auto node_ptr_spot = m_sinks.find(arg_node_ptr);
            auto pred_ptr = GraphAccess::get_node_ptr(arg_pred_ref);
            auto pred_ref_spot = m_sinks.find(pred_ptr);
            FLARE_ASSERT(node_ptr_spot != m_sinks.end())
            if (pred_ref_spot != m_sinks.end()) {
                // delegate responsibility for executing the predecessor to arg_node
                // and then remove the predecessor from the set of sinks
                (*node_ptr_spot)->set_predecessor(std::move(*pred_ref_spot));
                m_sinks.erase(pred_ref_spot);
            } else {
                // We still want to check that it's executed, even though someone else
                // should have executed it before us
                (*node_ptr_spot)->set_predecessor(std::move(pred_ptr));
            }
        }

        template<class... PredecessorRefs>
        // See requirements/expectations in GraphBuilder
        auto create_aggregate_ptr(PredecessorRefs &&...) {
            // The attachment to predecessors, which is all we really need, happens
            // in the generic layer, which calls through to add_predecessor for
            // each predecessor ref, so all we need to do here is create the (trivial)
            // aggregate node.
            using aggregate_kernel_impl_t =
                    GraphNodeAggregateKernelDefaultImpl<ExecutionSpace>;
            using aggregate_node_impl_t =
                    GraphNodeImpl<ExecutionSpace, aggregate_kernel_impl_t,
                            flare::experimental::TypeErasedTag>;
            return GraphAccess::make_node_shared_ptr<aggregate_node_impl_t>(
                    this->execution_space_instance(), _graph_node_kernel_ctor_tag{},
                    aggregate_kernel_impl_t{});
        }

        auto create_root_node_ptr() {
            auto rv = flare::detail::GraphAccess::make_node_shared_ptr<root_node_impl_t>(
                    get_execution_space(), _graph_node_is_root_ctor_tag{});
            m_sinks.insert(rv);
            return rv;
        }

        void submit() {
            // This reset is gross, but for the purposes of our simple host
            // implementation...
            for (auto &sink: m_sinks) {
                sink->reset_has_executed();
            }
            for (auto &sink: m_sinks) {
                sink->execute_node();
            }
        }

    };

}  // end namespace flare::detail

#include <flare/core/graph/default_graph_node_kernel.h>
#include <flare/core/graph/default_graph_node_impl.h>

#endif  // FLARE_CORE_GRAPH_DEFAULT_GRAPH_IMPL_H_
