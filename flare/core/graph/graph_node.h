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

#ifndef FLARE_CORE_GRAPH_GRAPH_NODE_H_
#define FLARE_CORE_GRAPH_GRAPH_NODE_H_

#include <flare/core/defines.h>

#include <flare/core/common/error.h>  // contract macros

#include <flare/core_fwd.h>
#include <flare/core/graph/graph_fwd.h>
#include <flare/core/graph/graph_impl_fwd.h>
#include <flare/core/parallel/parallel_reduce.h>
#include <flare/core/graph/graph_impl_utilities.h>
#include <flare/core/graph/graph_impl.h>  // GraphAccess

#include <memory>  // std::shared_ptr

namespace flare::experimental {

    template<class ExecutionSpace, class Kernel /*= TypeErasedTag*/,
            class Predecessor /*= TypeErasedTag*/>
    class GraphNodeRef {
        // Note: because of these assertions, instantiating this class template is not
        //       intended to be SFINAE-safe, so do validation before you instantiate.

        static_assert(
                std::is_same<Predecessor, TypeErasedTag>::value ||
                flare::detail::is_specialization_of<Predecessor, GraphNodeRef>::value,
                "Invalid predecessor template parameter given to GraphNodeRef");

        static_assert(
                flare::is_execution_space<ExecutionSpace>::value,
                "Invalid execution space template parameter given to GraphNodeRef");

        static_assert(std::is_same<Predecessor, TypeErasedTag>::value ||
                      flare::detail::is_graph_kernel<Kernel>::value,
                      "Invalid kernel template parameter given to GraphNodeRef");

        static_assert(!flare::detail::is_more_type_erased<Kernel, Predecessor>::value,
                      "The kernel of a graph node can't be more type-erased than the "
                      "predecessor");


    public:

        using execution_space = ExecutionSpace;
        using graph_kernel = Kernel;
        using graph_predecessor = Predecessor;

    private:
        template<class, class, class>
        friend
        class GraphNodeRef;

        friend struct flare::detail::GraphAccess;

        using graph_impl_t = flare::detail::GraphImpl<ExecutionSpace>;
        std::weak_ptr<graph_impl_t> m_graph_impl;

        // TODO @graphs figure out if we can get away with a weak reference here?
        //              GraphNodeRef instances shouldn't be stored by users outside
        //              of the create_graph closure, and so if we restructure things
        //              slightly, we could make it so that the graph owns the
        //              node_impl_t instance and this only holds a std::weak_ptr to
        //              it.
        using node_impl_t =
                flare::detail::GraphNodeImpl<ExecutionSpace, Kernel, Predecessor>;
        std::shared_ptr<node_impl_t> m_node_impl;

        // Internally, use shallow constness
        node_impl_t &get_node_impl() const { return *m_node_impl.get(); }

        std::shared_ptr<node_impl_t> const &get_node_ptr() const &{
            return m_node_impl;
        }

        std::shared_ptr<node_impl_t> get_node_ptr() &&{
            return std::move(m_node_impl);
        }

        std::weak_ptr<graph_impl_t> get_graph_weak_ptr() const {
            return m_graph_impl;
        }

        // TODO kernel name propagation and exposure

        template<class NextKernelDeduced>
        auto _then_kernel(NextKernelDeduced &&arg_kernel) const {
            // readability note:
            //   std::remove_cvref_t<NextKernelDeduced> is a specialization of
            //   flare::detail::GraphNodeKernelImpl:
            static_assert(flare::detail::is_specialization_of<
                                  flare::detail::remove_cvref_t<NextKernelDeduced>,
                                  flare::detail::GraphNodeKernelImpl>::value,
                          "flare internal error");

            auto graph_ptr = m_graph_impl.lock();
            FLARE_EXPECTS(bool(graph_ptr))

            using next_kernel_t = flare::detail::remove_cvref_t<NextKernelDeduced>;

            using return_t = GraphNodeRef<ExecutionSpace, next_kernel_t, GraphNodeRef>;

            auto rv = flare::detail::GraphAccess::make_graph_node_ref(
                    m_graph_impl,
                    flare::detail::GraphAccess::make_node_shared_ptr<
                            typename return_t::node_impl_t>(
                            m_node_impl->execution_space_instance(),
                            flare::detail::_graph_node_kernel_ctor_tag{},
                            (NextKernelDeduced &&) arg_kernel,
                            // *this is the predecessor
                            flare::detail::_graph_node_predecessor_ctor_tag{}, *this));

            // Add the node itself to the backend's graph data structure, now that
            // everything is set up.
            graph_ptr->add_node(rv.m_node_impl);
            // Add the predecessaor we stored in the constructor above in the backend's
            // data structure, now that everything is set up.
            graph_ptr->add_predecessor(rv.m_node_impl, *this);
            FLARE_ENSURES(bool(rv.m_node_impl))
            return rv;
        }

        GraphNodeRef(std::weak_ptr<graph_impl_t> arg_graph_impl,
                     std::shared_ptr<node_impl_t> arg_node_impl)
                : m_graph_impl(std::move(arg_graph_impl)),
                  m_node_impl(std::move(arg_node_impl)) {}


    public:

        // Copyable and movable (basically just shared_ptr semantics
        GraphNodeRef() noexcept = default;

        GraphNodeRef(GraphNodeRef const &) = default;

        GraphNodeRef(GraphNodeRef &&) noexcept = default;

        GraphNodeRef &operator=(GraphNodeRef const &) = default;

        GraphNodeRef &operator=(GraphNodeRef &&) noexcept = default;

        ~GraphNodeRef() = default;

        template<
                class OtherKernel, class OtherPredecessor,
                std::enable_if_t<
                        // Not a copy/move constructor
                        !std::is_same<GraphNodeRef, GraphNodeRef<execution_space, OtherKernel,
                                OtherPredecessor>>::value &&
                        // must be an allowed type erasure of the kernel
                        flare::detail::is_compatible_type_erasure<OtherKernel,
                                graph_kernel>::value &&
                        // must be an allowed type erasure of the predecessor
                        flare::detail::is_compatible_type_erasure<
                                OtherPredecessor, graph_predecessor>::value,
                        int> = 0>
        /* implicit */
        GraphNodeRef(
                GraphNodeRef<execution_space, OtherKernel, OtherPredecessor> const &other)
                : m_graph_impl(other.m_graph_impl), m_node_impl(other.m_node_impl) {}

        // Note: because this is an implicit conversion (as is supposed to be the
        //       case with most type-erasing wrappers like this), we don't also need
        //       a converting assignment operator.

        template<
                class Policy, class Functor,
                std::enable_if_t<
                        // equivalent to:
                        //   requires flare::ExecutionPolicy<remove_cvref_t<Policy>>
                        is_execution_policy<flare::detail::remove_cvref_t<Policy>>::value,
                        // --------------------
                        int> = 0>
        auto then_parallel_for(std::string arg_name, Policy &&arg_policy,
                               Functor &&functor) const {
            //----------------------------------------
            FLARE_EXPECTS(!m_graph_impl.expired())
            FLARE_EXPECTS(bool(m_node_impl))
            // TODO @graph restore this expectation once we add comparability to space
            //      instances
            // FLARE_EXPECTS(
            //   arg_policy.space() == m_graph_impl->get_execution_space());

            // needs to static assert constraint: DataParallelFunctor<Functor>

            using policy_t = flare::detail::remove_cvref_t<Policy>;
            // constraint check: same execution space type (or defaulted, maybe?)
            static_assert(
                    std::is_same<typename policy_t::execution_space,
                            execution_space>::value,
                    // TODO @graph make defaulted execution space work
                    //|| policy_t::execution_space_is_defaulted,
                    "Execution Space mismatch between execution policy and graph");

            auto policy = experimental::require((Policy &&) arg_policy,
                                                flare::detail::KernelInGraphProperty{});

            using next_policy_t = decltype(policy);
            using next_kernel_t =
                    flare::detail::GraphNodeKernelImpl<ExecutionSpace, next_policy_t,
                            std::decay_t<Functor>,
                            flare::ParallelForTag>;
            return this->_then_kernel(next_kernel_t{std::move(arg_name), policy.space(),
                                                    (Functor &&) functor,
                                                    (Policy &&) policy});
        }

        template<
                class Policy, class Functor,
                std::enable_if_t<
                        // equivalent to:
                        //   requires flare::ExecutionPolicy<remove_cvref_t<Policy>>
                        is_execution_policy<flare::detail::remove_cvref_t<Policy>>::value,
                        // --------------------
                        int> = 0>
        auto then_parallel_for(Policy &&policy, Functor &&functor) const {
            // needs to static assert constraint: DataParallelFunctor<Functor>
            return this->then_parallel_for("", (Policy &&) policy,
                                           (Functor &&) functor);
        }

        template<class Functor>
        auto then_parallel_for(std::string name, std::size_t n,
                               Functor &&functor) const {
            // needs to static assert constraint: DataParallelFunctor<Functor>
            return this->then_parallel_for(std::move(name),
                                           flare::RangePolicy<execution_space>(0, n),
                                           (Functor &&) functor);
        }

        template<class Functor>
        auto then_parallel_for(std::size_t n, Functor &&functor) const {
            // needs to static assert constraint: DataParallelFunctor<Functor>
            return this->then_parallel_for("", n, (Functor &&) functor);
        }

        template<
                class Policy, class Functor, class ReturnType,
                std::enable_if_t<
                        // equivalent to:
                        //   requires flare::ExecutionPolicy<remove_cvref_t<Policy>>
                        is_execution_policy<flare::detail::remove_cvref_t<Policy>>::value,
                        // --------------------
                        int> = 0>
        auto then_parallel_reduce(std::string arg_name, Policy &&arg_policy,
                                  Functor &&functor,
                                  ReturnType &&return_value) const {
            auto graph_impl_ptr = m_graph_impl.lock();
            FLARE_EXPECTS(bool(graph_impl_ptr))
            FLARE_EXPECTS(bool(m_node_impl))
            // TODO @graph restore this expectation once we add comparability to space
            //      instances
            // FLARE_EXPECTS(
            //   arg_policy.space() == m_graph_impl->get_execution_space());

            // needs static assertion of constraint:
            //   DataParallelReductionFunctor<Functor, ReturnType>

            using policy_t = std::remove_cv_t<std::remove_reference_t<Policy>>;
            static_assert(
                    std::is_same<typename policy_t::execution_space,
                            execution_space>::value,
                    // TODO @graph make defaulted execution space work
                    // || policy_t::execution_space_is_defaulted,
                    "Execution Space mismatch between execution policy and graph");

            // This is also just an expectation, but it's one that we expect the user
            // to interact with (even in release mode), so we should throw an exception
            // with an explanation rather than just doing a contract assertion.
            // We can't static_assert this because of the way that Reducers store
            // whether or not they point to a View as a runtime boolean rather than part
            // of the type.
            if (flare::detail::parallel_reduce_needs_fence(
                    graph_impl_ptr->get_execution_space(), return_value)) {
                flare::detail::throw_runtime_exception(
                        "Parallel reductions in graphs can't operate on Reducers that "
                        "reference a scalar because they can't complete synchronously. Use a "
                        "flare::View instead and keep in mind the result will only be "
                        "available once the graph is submitted (or in tasks that depend on "
                        "this one).");
            }

            //----------------------------------------
            // This is a disaster, but I guess it's not a my disaster to fix right now
            using return_type_remove_cvref =
                    std::remove_cv_t<std::remove_reference_t<ReturnType>>;
            static_assert(flare::is_view<return_type_remove_cvref>::value ||
                          flare::is_reducer<return_type_remove_cvref>::value,
                          "Output argument to parallel reduce in a graph must be a "
                          "View or a Reducer");
            using return_type =
                // Yes, you do really have to do this...
                    std::conditional_t<flare::is_reducer<return_type_remove_cvref>::value,
                            return_type_remove_cvref,
                            const return_type_remove_cvref>;
            using functor_type = flare::detail::remove_cvref_t<Functor>;
            // see parallel_reduce.h for how these details are used there;
            // we're just doing the same thing here
            using return_value_adapter =
                    flare::detail::ParallelReduceReturnValue<void, return_type,
                            functor_type>;
            // End of flare reducer disaster
            //----------------------------------------

            auto policy = experimental::require((Policy &&) arg_policy,
                                                flare::detail::KernelInGraphProperty{});

            using passed_reducer_type = typename return_value_adapter::reducer_type;

            using reducer_selector = flare::detail::if_c<
                    std::is_same<InvalidType, passed_reducer_type>::value, functor_type,
                    passed_reducer_type>;
            using analysis = flare::detail::FunctorAnalysis<
                    flare::detail::FunctorPatternInterface::REDUCE, Policy,
                    typename reducer_selector::type,
                    typename return_value_adapter::value_type>;
            typename analysis::Reducer final_reducer(
                    reducer_selector::select(functor, return_value));
            flare::detail::CombinedFunctorReducer<functor_type,
                    typename analysis::Reducer>
                    functor_reducer(functor, final_reducer);

            using next_policy_t = decltype(policy);
            using next_kernel_t =
                    flare::detail::GraphNodeKernelImpl<ExecutionSpace, next_policy_t,
                            decltype(functor_reducer),
                            flare::ParallelReduceTag>;

            return this->_then_kernel(next_kernel_t{
                    std::move(arg_name), graph_impl_ptr->get_execution_space(),
                    functor_reducer, (Policy &&) policy,
                    return_value_adapter::return_value(return_value, functor)});
        }

        template<
                class Policy, class Functor, class ReturnType,
                std::enable_if_t<
                        // equivalent to:
                        //   requires flare::ExecutionPolicy<remove_cvref_t<Policy>>
                        is_execution_policy<flare::detail::remove_cvref_t<Policy>>::value,
                        // --------------------
                        int> = 0>
        auto then_parallel_reduce(Policy &&arg_policy, Functor &&functor,
                                  ReturnType &&return_value) const {
            return this->then_parallel_reduce("", (Policy &&) arg_policy,
                                              (Functor &&) functor,
                                              (ReturnType &&) return_value);
        }

        template<class Functor, class ReturnType>
        auto then_parallel_reduce(std::string label,
                                  typename execution_space::size_type idx_end,
                                  Functor &&functor,
                                  ReturnType &&return_value) const {
            return this->then_parallel_reduce(
                    std::move(label), flare::RangePolicy<execution_space>{0, idx_end},
                    (Functor &&) functor, (ReturnType &&) return_value);
        }

        template<class Functor, class ReturnType>
        auto then_parallel_reduce(typename execution_space::size_type idx_end,
                                  Functor &&functor,
                                  ReturnType &&return_value) const {
            return this->then_parallel_reduce("", idx_end, (Functor &&) functor,
                                              (ReturnType &&) return_value);
        }

        // TODO @graph parallel scan, deep copy, etc.
    };
}  // end namespace flare::experimental

#endif  // FLARE_CORE_GRAPH_GRAPH_NODE_H_
