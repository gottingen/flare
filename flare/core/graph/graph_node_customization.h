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

#ifndef FLARE_CORE_GRAPH_GRAPH_NODE_CUSTOMIZATION_H_
#define FLARE_CORE_GRAPH_GRAPH_NODE_CUSTOMIZATION_H_

#include <flare/core/defines.h>
#include <flare/core_fwd.h>
#include <flare/core/graph/graph_fwd.h>
#include <flare/core/graph/graph_impl_fwd.h>

namespace flare::detail {

    // Customizable for backends
    template<class ExecutionSpace, class Kernel, class PredecessorRef>
    struct GraphNodeBackendDetailsBeforeTypeErasure {
    protected:
        // Required constructors in customizations:
        GraphNodeBackendDetailsBeforeTypeErasure(
                ExecutionSpace const &, Kernel &, PredecessorRef const &,
                GraphNodeBackendSpecificDetails<ExecutionSpace> &
                /* this_as_details */) noexcept {}

        GraphNodeBackendDetailsBeforeTypeErasure(
                ExecutionSpace const &, _graph_node_is_root_ctor_tag,
                GraphNodeBackendSpecificDetails<ExecutionSpace> &
                /* this_as_details */) noexcept {}

        // Not copyable or movable at the concept level, so the default
        // implementation shouldn't be either.
        GraphNodeBackendDetailsBeforeTypeErasure() = delete;

        GraphNodeBackendDetailsBeforeTypeErasure(
                GraphNodeBackendDetailsBeforeTypeErasure const &) = delete;

        GraphNodeBackendDetailsBeforeTypeErasure(
                GraphNodeBackendDetailsBeforeTypeErasure &&) = delete;

        GraphNodeBackendDetailsBeforeTypeErasure &operator=(
                GraphNodeBackendDetailsBeforeTypeErasure const &) = delete;

        GraphNodeBackendDetailsBeforeTypeErasure &operator=(
                GraphNodeBackendDetailsBeforeTypeErasure &&) = delete;

        ~GraphNodeBackendDetailsBeforeTypeErasure() = default;

    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_GRAPH_GRAPH_NODE_CUSTOMIZATION_H_
