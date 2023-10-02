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

#ifndef FLARE_CORE_GRAPH_GRAPH_NODE_IMPL_H_
#define FLARE_CORE_GRAPH_GRAPH_NODE_IMPL_H_

#include <flare/core/defines.h>
#include <flare/core_fwd.h>
#include <flare/core/graph/graph_fwd.h>
#include <flare/core/graph/graph_impl.h>
#include <flare/core/graph/graph_node_customization.h>
#include <flare/core/graph/ebo.h>

#include <memory>

namespace flare::detail {

    // Base specialization for the case where both the kernel and the predecessor
    // type information is type-erased
    template<class ExecutionSpace>
    struct GraphNodeImpl<ExecutionSpace, flare::experimental::TypeErasedTag,
            flare::experimental::TypeErasedTag>
            : GraphNodeBackendSpecificDetails<ExecutionSpace>,
              ExecutionSpaceInstanceStorage<ExecutionSpace> {
    public:
        using node_ref_t =
                flare::experimental::GraphNodeRef<ExecutionSpace,
                        flare::experimental::TypeErasedTag,
                        flare::experimental::TypeErasedTag>;

    protected:
        using implementation_base_t = GraphNodeBackendSpecificDetails<ExecutionSpace>;
        using execution_space_storage_base_t =
                ExecutionSpaceInstanceStorage<ExecutionSpace>;

    public:
        virtual ~GraphNodeImpl() = default;

    protected:

        explicit GraphNodeImpl(ExecutionSpace const &ex) noexcept
                : implementation_base_t(), execution_space_storage_base_t(ex) {}


    public:

        template<class... Args>
        GraphNodeImpl(ExecutionSpace const &ex, _graph_node_is_root_ctor_tag,
                      Args &&... args) noexcept
                : implementation_base_t(_graph_node_is_root_ctor_tag{},
                                        (Args &&) args...),
                  execution_space_storage_base_t(ex) {}

        GraphNodeImpl() = delete;

        GraphNodeImpl(GraphNodeImpl const &) = delete;

        GraphNodeImpl(GraphNodeImpl &&) = delete;

        GraphNodeImpl &operator=(GraphNodeImpl const &) = delete;

        GraphNodeImpl &operator=(GraphNodeImpl &&) = delete;


        ExecutionSpace const &execution_space_instance() const {
            return this->execution_space_storage_base_t::execution_space_instance();
        }
    };

    // Specialization for the case with the concrete type of the kernel, but the
    // predecessor erased.
    template<class ExecutionSpace, class Kernel>
    struct GraphNodeImpl<ExecutionSpace, Kernel,
            flare::experimental::TypeErasedTag>
            : GraphNodeImpl<ExecutionSpace, flare::experimental::TypeErasedTag,
                    flare::experimental::TypeErasedTag> {
    private:
        using base_t =
                GraphNodeImpl<ExecutionSpace, flare::experimental::TypeErasedTag,
                        flare::experimental::TypeErasedTag>;

    public:

        using node_ref_t =
                flare::experimental::GraphNodeRef<ExecutionSpace, Kernel,
                        flare::experimental::TypeErasedTag>;
        using kernel_type = Kernel;

    private:

        Kernel m_kernel;


    public:

        template<class KernelDeduced>
        GraphNodeImpl(ExecutionSpace const &ex, _graph_node_kernel_ctor_tag,
                      KernelDeduced &&arg_kernel)
                : base_t(ex), m_kernel((KernelDeduced &&) arg_kernel) {}

        template<class... Args>
        GraphNodeImpl(ExecutionSpace const &ex, _graph_node_is_root_ctor_tag,
                      Args &&... args)
                : base_t(ex, _graph_node_is_root_ctor_tag{}, (Args &&) args...) {}

        // Not copyable or movable
        GraphNodeImpl() = delete;

        GraphNodeImpl(GraphNodeImpl const &) = delete;

        GraphNodeImpl(GraphNodeImpl &&) = delete;

        GraphNodeImpl &operator=(GraphNodeImpl const &) = delete;

        GraphNodeImpl &operator=(GraphNodeImpl &&) = delete;

        ~GraphNodeImpl() override = default;

        // Reference qualified to prevent dangling reference to data member
        Kernel &get_kernel() &{ return m_kernel; }

        Kernel const &get_kernel() const &{ return m_kernel; }

        Kernel &&get_kernel() && = delete;

    };

    // Specialization for the case where nothing is type-erased
    template<class ExecutionSpace, class Kernel, class PredecessorRef>
    struct GraphNodeImpl
            : GraphNodeImpl<ExecutionSpace, Kernel,
                    flare::experimental::TypeErasedTag>,
              GraphNodeBackendDetailsBeforeTypeErasure<ExecutionSpace, Kernel,
                      PredecessorRef> {
    private:
        using base_t = GraphNodeImpl<ExecutionSpace, Kernel,
                flare::experimental::TypeErasedTag>;
        using backend_details_base_t =
                GraphNodeBackendDetailsBeforeTypeErasure<ExecutionSpace, Kernel,
                        PredecessorRef>;
        // The fully type-erased base type, for the destroy function
        using type_erased_base_t =
                GraphNodeImpl<ExecutionSpace, flare::experimental::TypeErasedTag,
                        flare::experimental::TypeErasedTag>;

    public:

        using node_ref_t = flare::experimental::GraphNodeRef<ExecutionSpace, Kernel,
                PredecessorRef>;

    private:

        PredecessorRef m_predecessor_ref;

    public:

        // Not copyable or movable
        GraphNodeImpl() = delete;

        GraphNodeImpl(GraphNodeImpl const &) = delete;

        GraphNodeImpl(GraphNodeImpl &&) = delete;

        GraphNodeImpl &operator=(GraphNodeImpl const &) = delete;

        GraphNodeImpl &operator=(GraphNodeImpl &&) = delete;

        ~GraphNodeImpl() override = default;

        // Normal kernel-and-predecessor constructor
        template<class KernelDeduced, class PredecessorPtrDeduced>
        GraphNodeImpl(ExecutionSpace const &ex, _graph_node_kernel_ctor_tag,
                      KernelDeduced &&arg_kernel, _graph_node_predecessor_ctor_tag,
                      PredecessorPtrDeduced &&arg_predecessor)
                : base_t(ex, _graph_node_kernel_ctor_tag{},
                         (KernelDeduced &&) arg_kernel),
                // The backend gets the ability to store (weak, non-owning) references
                // to the kernel in it's final resting place here if it wants. The
                // predecessor is already a pointer, so it doesn't matter that it isn't
                // already at its final address
                  backend_details_base_t(ex, this->base_t::get_kernel(), arg_predecessor,
                                         *this),
                  m_predecessor_ref((PredecessorPtrDeduced &&) arg_predecessor) {}

        // Root-tagged constructor
        template<class... Args>
        GraphNodeImpl(ExecutionSpace const &ex, _graph_node_is_root_ctor_tag,
                      Args &&... args)
                : base_t(ex, _graph_node_is_root_ctor_tag{}, (Args &&) args...),
                  backend_details_base_t(ex, _graph_node_is_root_ctor_tag{}, *this),
                  m_predecessor_ref() {}

    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_GRAPH_GRAPH_NODE_IMPL_H_
