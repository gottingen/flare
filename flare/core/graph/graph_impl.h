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

#ifndef FLARE_CORE_GRAPH_GRAPH_IMPL_H_
#define FLARE_CORE_GRAPH_GRAPH_IMPL_H_

#include <flare/core/defines.h>
#include <flare/core_fwd.h>
#include <flare/core/graph/graph_fwd.h>
#include <flare/core/common/concepts.h>  // is_execution_policy
#include <flare/core/memory/pointer_ownership.h>
#include <flare/core/graph/graph_impl_fwd.h>

#include <memory>  // std::make_shared

namespace flare::detail {

    struct GraphAccess {
        template<class ExecutionSpace>
        static flare::experimental::Graph<ExecutionSpace> construct_graph(
                ExecutionSpace ex) {
            return flare::experimental::Graph<ExecutionSpace>{
                    std::make_shared<GraphImpl<ExecutionSpace>>(std::move(ex))};
        }

        template<class ExecutionSpace>
        static auto create_root_ref(
                flare::experimental::Graph<ExecutionSpace> &arg_graph) {
            auto const &graph_impl_ptr = arg_graph.m_impl_ptr;

            auto root_ptr = graph_impl_ptr->create_root_node_ptr();

            return flare::experimental::GraphNodeRef<ExecutionSpace>{
                    graph_impl_ptr, std::move(root_ptr)};
        }

        template<class NodeType, class... Args>
        static auto make_node_shared_ptr(Args &&... args) {
            static_assert(
                    flare::detail::is_specialization_of<NodeType, GraphNodeImpl>::value,
                    "flare Internal Error in graph interface");
            return std::make_shared<NodeType>((Args &&) args...);
        }

        template<class GraphImplWeakPtr, class ExecutionSpace, class Kernel,
                class Predecessor>
        static auto make_graph_node_ref(
                GraphImplWeakPtr graph_impl,
                std::shared_ptr<
                        flare::detail::GraphNodeImpl<ExecutionSpace, Kernel, Predecessor>>
                pred_impl) {
            return flare::experimental::GraphNodeRef<ExecutionSpace, Kernel,
                    Predecessor>{
                    std::move(graph_impl), std::move(pred_impl)};
        }

        template<class NodeRef>
        static auto get_node_ptr(NodeRef &&node_ref) {
            static_assert(
                    is_specialization_of<remove_cvref_t<NodeRef>,
                            flare::experimental::GraphNodeRef>::value,
                    "flare Internal Implementation error (bad argument to "
                    "`GraphAccess::get_node_ptr()`)");
            return ((NodeRef &&) node_ref).get_node_ptr();
        }

        template<class NodeRef>
        static auto get_graph_weak_ptr(NodeRef &&node_ref) {
            static_assert(
                    is_specialization_of<remove_cvref_t<NodeRef>,
                            flare::experimental::GraphNodeRef>::value,
                    "flare Internal Implementation error (bad argument to "
                    "`GraphAccess::get_graph_weak_ptr()`)");
            return ((NodeRef &&) node_ref).get_graph_weak_ptr();
        }

    };

    template<class Policy>
    struct _add_graph_kernel_tag;

    template<template<class...> class PolicyTemplate, class... PolicyTraits>
    struct _add_graph_kernel_tag<PolicyTemplate<PolicyTraits...>> {
        using type = PolicyTemplate<PolicyTraits..., IsGraphKernelTag>;
    };

}  // end namespace flare::detail

namespace flare::experimental {  // but not for users, so...

    // requires ExecutionPolicy<Policy>
    template<class Policy>
    constexpr auto require(Policy const &policy,
                           flare::detail::KernelInGraphProperty) {
        static_assert(flare::is_execution_policy<Policy>::value,
                      "Internal implementation error!");
        return typename flare::detail::_add_graph_kernel_tag<Policy>::type{policy};
    }

}  // end namespace flare::experimental


#endif  // FLARE_CORE_GRAPH_GRAPH_IMPL_H_
