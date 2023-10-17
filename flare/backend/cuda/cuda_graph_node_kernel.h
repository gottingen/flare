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

#ifndef FLARE_BACKEND_CUDA_CUDA_GRAPH_NODE_KERNEL_H_
#define FLARE_BACKEND_CUDA_CUDA_GRAPH_NODE_KERNEL_H_

#include <flare/core/defines.h>

#if defined(FLARE_ON_CUDA_DEVICE)

#include <flare/core/graph/graph_fwd.h>

#include <flare/core/graph/graph_impl.h>    // GraphAccess needs to be complete
#include <flare/core/memory/shared_alloc.h>  // SharedAllocationRecord

#include <flare/core/parallel/parallel.h>
#include <flare/core/parallel/parallel_reduce.h>
#include <flare/core/memory/pointer_ownership.h>

#include <flare/backend/cuda/cuda.h>

namespace flare::detail {

    template<class PolicyType, class Functor, class PatternTag, class... Args>
    class GraphNodeKernelImpl<flare::Cuda, PolicyType, Functor, PatternTag,
            Args...>
            : public PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                    Args..., flare::Cuda>::type {
    private:
        using base_t =
                typename PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                        Args..., flare::Cuda>::type;
        using size_type = flare::Cuda::size_type;
        // These are really functioning as optional references, though I'm not sure
        // that the cudaGraph_t one needs to be since it's a pointer under the
        // covers and we're not modifying it
        flare::ObservingRawPtr<const cudaGraph_t> m_graph_ptr = nullptr;
        flare::ObservingRawPtr<cudaGraphNode_t> m_graph_node_ptr = nullptr;
        // Note: owned pointer to CudaSpace memory (used for global memory launches),
        // which we're responsible for deallocating, but not responsible for calling
        // its destructor.
        using Record = flare::detail::SharedAllocationRecord<flare::CudaSpace, void>;
        // Basically, we have to make this mutable for the same reasons that the
        // global kernel buffers in the Cuda instance are mutable...
        mutable flare::OwningRawPtr<base_t> m_driver_storage = nullptr;

    public:
        using Policy = PolicyType;
        using graph_kernel = GraphNodeKernelImpl;

        // TODO Ensure the execution space of the graph is the same as the one
        //      attached to the policy?
        // TODO @graph kernel name info propagation
        template<class PolicyDeduced, class... ArgsDeduced>
        GraphNodeKernelImpl(std::string, flare::Cuda const &, Functor arg_functor,
                            PolicyDeduced &&arg_policy, ArgsDeduced &&... args)
        // This is super ugly, but it works for now and is the most minimal change
        // to the codebase for now...
                : base_t(std::move(arg_functor), (PolicyDeduced &&) arg_policy,
                         (ArgsDeduced &&) args...) {}

        // FIXME @graph Forward through the instance once that works in the backends
        template<class PolicyDeduced>
        GraphNodeKernelImpl(flare::Cuda const &ex, Functor arg_functor,
                            PolicyDeduced &&arg_policy)
                : GraphNodeKernelImpl("", ex, std::move(arg_functor),
                                      (PolicyDeduced &&) arg_policy) {}

        ~GraphNodeKernelImpl() {
            if (m_driver_storage) {
                // We should be the only owner, but this is still the easiest way to
                // allocate and deallocate aligned memory for these sorts of things
                Record::decrement(Record::get_record(m_driver_storage));
            }
        }

        void set_cuda_graph_ptr(cudaGraph_t *arg_graph_ptr) {
            m_graph_ptr = arg_graph_ptr;
        }

        void set_cuda_graph_node_ptr(cudaGraphNode_t *arg_node_ptr) {
            m_graph_node_ptr = arg_node_ptr;
        }

        cudaGraphNode_t *get_cuda_graph_node_ptr() const { return m_graph_node_ptr; }

        cudaGraph_t const *get_cuda_graph_ptr() const { return m_graph_ptr; }

        flare::ObservingRawPtr<base_t> allocate_driver_memory_buffer() const {
            FLARE_EXPECTS(m_driver_storage == nullptr)

            auto *record = Record::allocate(
                    flare::CudaSpace{}, "GraphNodeKernel global memory functor storage",
                    sizeof(base_t));

            Record::increment(record);
            m_driver_storage = reinterpret_cast<base_t *>(record->data());
            FLARE_ENSURES(m_driver_storage != nullptr)
            return m_driver_storage;
        }
    };

    struct CudaGraphNodeAggregateKernel {
        using graph_kernel = CudaGraphNodeAggregateKernel;

        // Aggregates don't need a policy, but for the purposes of checking the static
        // assertions about graph kerenls,
        struct Policy {
            using is_graph_kernel = std::true_type;
        };
    };

    template<class KernelType,
            class Tag =
            typename PatternTagFromImplSpecialization<KernelType>::type>
    struct get_graph_node_kernel_type
            : type_identity<
                    GraphNodeKernelImpl<flare::Cuda, typename KernelType::Policy,
                            typename KernelType::functor_type, Tag>> {
    };
    template<class KernelType>
    struct get_graph_node_kernel_type<KernelType, flare::ParallelReduceTag>
            : type_identity<GraphNodeKernelImpl<
                    flare::Cuda, typename KernelType::Policy,
                    CombinedFunctorReducer<typename KernelType::functor_type,
                            typename KernelType::reducer_type>,
                    flare::ParallelReduceTag>> {
    };

    template<class KernelType>
    auto *allocate_driver_storage_for_kernel(KernelType const &kernel) {
        using graph_node_kernel_t =
                typename get_graph_node_kernel_type<KernelType>::type;
        auto const &kernel_as_graph_kernel =
                static_cast<graph_node_kernel_t const &>(kernel);
        // TODO @graphs we need to somehow indicate the need for a fence in the
        //              destructor of the GraphImpl object (so that we don't have to
        //              just always do it)
        return kernel_as_graph_kernel.allocate_driver_memory_buffer();
    }

    template<class KernelType>
    auto const &get_cuda_graph_from_kernel(KernelType const &kernel) {
        using graph_node_kernel_t =
                typename get_graph_node_kernel_type<KernelType>::type;
        auto const &kernel_as_graph_kernel =
                static_cast<graph_node_kernel_t const &>(kernel);
        cudaGraph_t const *graph_ptr = kernel_as_graph_kernel.get_cuda_graph_ptr();
        FLARE_EXPECTS(graph_ptr != nullptr);
        return *graph_ptr;
    }

    template<class KernelType>
    auto &get_cuda_graph_node_from_kernel(KernelType const &kernel) {
        using graph_node_kernel_t =
                typename get_graph_node_kernel_type<KernelType>::type;
        auto const &kernel_as_graph_kernel =
                static_cast<graph_node_kernel_t const &>(kernel);
        auto *graph_node_ptr = kernel_as_graph_kernel.get_cuda_graph_node_ptr();
        FLARE_EXPECTS(graph_node_ptr != nullptr);
        return *graph_node_ptr;
    }


}  // end namespace flare::detail

#endif  // defined(FLARE_ON_CUDA_DEVICE)
#endif  // FLARE_BACKEND_CUDA_CUDA_GRAPH_NODE_KERNEL_H_
