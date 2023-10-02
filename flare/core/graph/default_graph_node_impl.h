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

#ifndef FLARE_CORE_GRAPH_DEFAULT_GRAPH_NODE_IMPL_H_
#define FLARE_CORE_GRAPH_DEFAULT_GRAPH_NODE_IMPL_H_

#include <flare/core/defines.h>

#include <flare/core/graph/default_graph_fwd.h>

#include <flare/core/graph/graph.h>

#include <vector>
#include <memory>

namespace flare::detail {

    template<class ExecutionSpace>
    struct GraphNodeBackendSpecificDetails {
    private:
        using execution_space_instance_storage_t =
                ExecutionSpaceInstanceStorage<ExecutionSpace>;
        using default_kernel_impl_t = GraphNodeKernelDefaultImpl<ExecutionSpace>;
        using default_aggregate_kernel_impl_t =
                GraphNodeAggregateKernelDefaultImpl<ExecutionSpace>;

        std::vector<std::shared_ptr<GraphNodeBackendSpecificDetails<ExecutionSpace>>>
                m_predecessors = {};

        flare::ObservingRawPtr<default_kernel_impl_t> m_kernel_ptr = nullptr;

        bool m_has_executed = false;
        bool m_is_aggregate = false;
        bool m_is_root = false;

        template<class>
        friend
        struct HostGraphImpl;

    protected:

        explicit GraphNodeBackendSpecificDetails() = default;

        explicit GraphNodeBackendSpecificDetails(
                _graph_node_is_root_ctor_tag) noexcept
                : m_has_executed(true), m_is_root(true) {}

        GraphNodeBackendSpecificDetails(GraphNodeBackendSpecificDetails const &) =
        delete;

        GraphNodeBackendSpecificDetails(GraphNodeBackendSpecificDetails &&) noexcept =
        delete;

        GraphNodeBackendSpecificDetails &operator=(
                GraphNodeBackendSpecificDetails const &) = delete;

        GraphNodeBackendSpecificDetails &operator=(
                GraphNodeBackendSpecificDetails &&) noexcept = delete;

        ~GraphNodeBackendSpecificDetails() = default;


    public:
        void set_kernel(default_kernel_impl_t &arg_kernel) {
            FLARE_EXPECTS(m_kernel_ptr == nullptr)
            m_kernel_ptr = &arg_kernel;
        }

        void set_kernel(default_aggregate_kernel_impl_t &arg_kernel) {
            FLARE_EXPECTS(m_kernel_ptr == nullptr)
            m_kernel_ptr = &arg_kernel;
            m_is_aggregate = true;
        }

        void set_predecessor(
                std::shared_ptr<GraphNodeBackendSpecificDetails<ExecutionSpace>>
                arg_pred_impl) {
            // This method delegates responsibility for executing the predecessor to
            // this node.  Each node can have at most one predecessor (which may be an
            // aggregate).
            FLARE_EXPECTS(m_predecessors.empty() || m_is_aggregate)
            FLARE_EXPECTS(bool(arg_pred_impl))
            FLARE_EXPECTS(!m_has_executed)
            m_predecessors.push_back(std::move(arg_pred_impl));
        }

        void execute_node() {
            // This node could have already been executed as the predecessor of some
            // other
            FLARE_EXPECTS(bool(m_kernel_ptr) || m_has_executed)
            // Just execute the predecessor here, since calling set_predecessor()
            // delegates the responsibility for running it to us.
            if (!m_has_executed) {
                // I'm pretty sure this doesn't need to be atomic under our current
                // supported semantics, but instinct I have feels like it should be...
                m_has_executed = true;
                for (auto const &predecessor: m_predecessors) {
                    predecessor->execute_node();
                }
                m_kernel_ptr->execute_kernel();
            }
            FLARE_ENSURES(m_has_executed)
        }

        // This is gross, but for the purposes of our simple default implementation...
        void reset_has_executed() {
            for (auto const &predecessor: m_predecessors) {
                predecessor->reset_has_executed();
            }
            // more readable, probably:
            //   if(!m_is_root) m_has_executed = false;
            m_has_executed = m_is_root;
        }
    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_GRAPH_DEFAULT_GRAPH_NODE_IMPL_H_
