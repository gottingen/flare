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

#ifndef FLARE_CORE_GRAPH_DEFAULT_GRAPH_NODE_KERNEL_H_
#define FLARE_CORE_GRAPH_DEFAULT_GRAPH_NODE_KERNEL_H_

#include <flare/core/defines.h>

#include <flare/core/graph/default_graph_fwd.h>

#include <flare/core/graph/graph.h>
#include <flare/core/parallel/parallel.h>
#include <flare/core/parallel/parallel_reduce.h>

namespace flare::detail {

    template<class ExecutionSpace>
    struct GraphNodeKernelDefaultImpl {
        // TODO @graphs decide if this should use vtable or intrusive erasure via
        //      function pointers like in the rest of the graph interface
        virtual void execute_kernel() = 0;
    };

    // TODO Indicate that this kernel specialization is only for the Host somehow?
    template<class ExecutionSpace, class PolicyType, class Functor,
            class PatternTag, class... Args>
    class GraphNodeKernelImpl
            : public PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                    Args..., ExecutionSpace>::type,
              public GraphNodeKernelDefaultImpl<ExecutionSpace> {
    public:
        using base_t =
                typename PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                        Args..., ExecutionSpace>::type;
        using execute_kernel_vtable_base_t =
                GraphNodeKernelDefaultImpl<ExecutionSpace>;
        // We have to use this name here because that's how it was done way back when
        // then implementations of detail::Parallel*<> were written
        using Policy = PolicyType;
        using graph_kernel = GraphNodeKernelImpl;

        // TODO @graph kernel name info propagation
        template<class PolicyDeduced, class... ArgsDeduced>
        GraphNodeKernelImpl(std::string const &, ExecutionSpace const &,
                            Functor arg_functor, PolicyDeduced &&arg_policy,
                            ArgsDeduced &&... args)
                : base_t(std::move(arg_functor), (PolicyDeduced &&) arg_policy,
                         (ArgsDeduced &&) args...),
                  execute_kernel_vtable_base_t() {}

        // FIXME @graph Forward through the instance once that works in the backends
        template<class PolicyDeduced, class... ArgsDeduced>
        GraphNodeKernelImpl(ExecutionSpace const &ex, Functor arg_functor,
                            PolicyDeduced &&arg_policy, ArgsDeduced &&... args)
                : GraphNodeKernelImpl("", ex, std::move(arg_functor),
                                      (PolicyDeduced &&) arg_policy,
                                      (ArgsDeduced &&) args...) {}

        void execute_kernel() final { this->base_t::execute(); }
    };

    template<class ExecutionSpace>
    struct GraphNodeAggregateKernelDefaultImpl
            : GraphNodeKernelDefaultImpl<ExecutionSpace> {
        // Aggregates don't need a policy, but for the purposes of checking the static
        // assertions about graph kernels,
        struct Policy {
            using is_graph_kernel = std::true_type;
        };
        using graph_kernel = GraphNodeAggregateKernelDefaultImpl;

        void execute_kernel() final {}
    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_GRAPH_DEFAULT_GRAPH_NODE_KERNEL_H_
