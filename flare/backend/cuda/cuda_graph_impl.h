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

#ifndef FLARE_BACKEND_CUDA_CUDA_GRAPH_IMPL_H_
#define FLARE_BACKEND_CUDA_CUDA_GRAPH_IMPL_H_

#include <flare/core/defines.h>

#if defined(FLARE_ON_CUDA_DEVICE)

#include <flare/core/graph/graph_fwd.h>
#include <flare/core/graph/graph_impl.h>  // GraphAccess needs to be complete

// GraphNodeImpl needs to be complete because GraphImpl here is a full
// specialization and not just a partial one
#include <flare/core/graph/graph_node_impl.h>
#include <flare/backend/cuda/cuda_graph_node_impl.h>

#include <flare/backend/cuda/cuda.h>
#include <flare/backend/cuda/cuda_error.h>
#include <flare/backend/cuda/cuda_instance.h>

namespace flare {
namespace detail {

template <>
struct GraphImpl<flare::Cuda> {
 public:
  using execution_space = flare::Cuda;

 private:
  execution_space m_execution_space;
  cudaGraph_t m_graph          = nullptr;
  cudaGraphExec_t m_graph_exec = nullptr;

  using cuda_graph_flags_t = unsigned int;

  using node_details_t = GraphNodeBackendSpecificDetails<flare::Cuda>;

  void _instantiate_graph() {
    constexpr size_t error_log_size = 256;
    cudaGraphNode_t error_node      = nullptr;
    char error_log[error_log_size];
    FLARE_IMPL_CUDA_SAFE_CALL(
        (m_execution_space.impl_internal_space_instance()
             ->cuda_graph_instantiate_wrapper(&m_graph_exec, m_graph,
                                              &error_node, error_log,
                                              error_log_size)));
    // TODO @graphs print out errors
  }

 public:
  using root_node_impl_t =
      GraphNodeImpl<flare::Cuda, flare::experimental::TypeErasedTag,
                    flare::experimental::TypeErasedTag>;
  using aggregate_kernel_impl_t = CudaGraphNodeAggregateKernel;
  using aggregate_node_impl_t =
      GraphNodeImpl<flare::Cuda, aggregate_kernel_impl_t,
                    flare::experimental::TypeErasedTag>;

  // Not moveable or copyable; it spends its whole life as a shared_ptr in the
  // Graph object
  GraphImpl()                 = delete;
  GraphImpl(GraphImpl const&) = delete;
  GraphImpl(GraphImpl&&)      = delete;
  GraphImpl& operator=(GraphImpl const&) = delete;
  GraphImpl& operator=(GraphImpl&&) = delete;
  ~GraphImpl() {
    // TODO @graphs we need to somehow indicate the need for a fence in the
    //              destructor of the GraphImpl object (so that we don't have to
    //              just always do it)
    m_execution_space.fence("flare::GraphImpl::~GraphImpl: Graph Destruction");
    FLARE_EXPECTS(bool(m_graph))
    if (bool(m_graph_exec)) {
      FLARE_IMPL_CUDA_SAFE_CALL(
          (m_execution_space.impl_internal_space_instance()
               ->cuda_graph_exec_destroy_wrapper(m_graph_exec)));
    }
    FLARE_IMPL_CUDA_SAFE_CALL(
        (m_execution_space.impl_internal_space_instance()
             ->cuda_graph_destroy_wrapper(m_graph)));
  };

  explicit GraphImpl(flare::Cuda arg_instance)
      : m_execution_space(std::move(arg_instance)) {
    FLARE_IMPL_CUDA_SAFE_CALL(
        (m_execution_space.impl_internal_space_instance()
             ->cuda_graph_create_wrapper(&m_graph, cuda_graph_flags_t{0})));
  }

  void add_node(std::shared_ptr<aggregate_node_impl_t> const& arg_node_ptr) {
    // All of the predecessors are just added as normal, so all we need to
    // do here is add an empty node
    FLARE_IMPL_CUDA_SAFE_CALL(
        (m_execution_space.impl_internal_space_instance()
             ->cuda_graph_add_empty_node_wrapper(
                 &(arg_node_ptr->node_details_t::node), m_graph,
                 /* dependencies = */ nullptr,
                 /* numDependencies = */ 0)));
  }

  template <class NodeImpl>
  //  requires NodeImplPtr is a shared_ptr to specialization of GraphNodeImpl
  //  Also requires that the kernel has the graph node tag in it's policy
  void add_node(std::shared_ptr<NodeImpl> const& arg_node_ptr) {
    static_assert(
        NodeImpl::kernel_type::Policy::is_graph_kernel::value,
        "Something has gone horribly wrong, but it's too complicated to "
        "explain here.  Buy Daisy a coffee and she'll explain it to you.");
    FLARE_EXPECTS(bool(arg_node_ptr));
    // The Kernel launch from the execute() method has been shimmed to insert
    // the node into the graph
    auto& kernel = arg_node_ptr->get_kernel();
    // note: using arg_node_ptr->node_details_t::node caused an ICE in NVCC 10.1
    auto& cuda_node = static_cast<node_details_t*>(arg_node_ptr.get())->node;
    FLARE_EXPECTS(!bool(cuda_node));
    kernel.set_cuda_graph_ptr(&m_graph);
    kernel.set_cuda_graph_node_ptr(&cuda_node);
    kernel.execute();
    FLARE_ENSURES(bool(cuda_node));
  }

  template <class NodeImplPtr, class PredecessorRef>
  // requires PredecessorRef is a specialization of GraphNodeRef that has
  // already been added to this graph and NodeImpl is a specialization of
  // GraphNodeImpl that has already been added to this graph.
  void add_predecessor(NodeImplPtr arg_node_ptr, PredecessorRef arg_pred_ref) {
    FLARE_EXPECTS(bool(arg_node_ptr))
    auto pred_ptr = GraphAccess::get_node_ptr(arg_pred_ref);
    FLARE_EXPECTS(bool(pred_ptr))

    // clang-format off
    // NOTE const-qualifiers below are commented out because of an API break
    // from CUDA 10.0 to CUDA 10.1
    // cudaGraphAddDependencies(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*, size_t)
    // cudaGraphAddDependencies(cudaGraph_t, const cudaGraphNode_t*, const cudaGraphNode_t*, size_t)
    // clang-format on
    auto /*const*/& pred_cuda_node = pred_ptr->node_details_t::node;
    FLARE_EXPECTS(bool(pred_cuda_node))

    auto /*const*/& cuda_node = arg_node_ptr->node_details_t::node;
    FLARE_EXPECTS(bool(cuda_node))

    FLARE_IMPL_CUDA_SAFE_CALL(
        (m_execution_space.impl_internal_space_instance()
             ->cuda_graph_add_dependencies_wrapper(m_graph, &pred_cuda_node,
                                                   &cuda_node, 1)));
  }

  void submit() {
    if (!bool(m_graph_exec)) {
      _instantiate_graph();
    }
    FLARE_IMPL_CUDA_SAFE_CALL(
        (m_execution_space.impl_internal_space_instance()
             ->cuda_graph_launch_wrapper(m_graph_exec)));
  }

  execution_space const& get_execution_space() const noexcept {
    return m_execution_space;
  }

  auto create_root_node_ptr() {
    FLARE_EXPECTS(bool(m_graph))
    FLARE_EXPECTS(!bool(m_graph_exec))
    auto rv = std::make_shared<root_node_impl_t>(
        get_execution_space(), _graph_node_is_root_ctor_tag{});
    FLARE_IMPL_CUDA_SAFE_CALL(
        (m_execution_space.impl_internal_space_instance()
             ->cuda_graph_add_empty_node_wrapper(&(rv->node_details_t::node),
                                                 m_graph,
                                                 /* dependencies = */ nullptr,
                                                 /* numDependencies = */ 0)));
    FLARE_ENSURES(bool(rv->node_details_t::node))
    return rv;
  }

  template <class... PredecessorRefs>
  // See requirements/expectations in GraphBuilder
  auto create_aggregate_ptr(PredecessorRefs&&...) {
    // The attachment to predecessors, which is all we really need, happens
    // in the generic layer, which calls through to add_predecessor for
    // each predecessor ref, so all we need to do here is create the (trivial)
    // aggregate node.
    return std::make_shared<aggregate_node_impl_t>(
        m_execution_space, _graph_node_kernel_ctor_tag{},
        aggregate_kernel_impl_t{});
  }
};

}  // end namespace detail
}  // end namespace flare

#endif  // defined(FLARE_ON_CUDA_DEVICE)
#endif  // FLARE_BACKEND_CUDA_CUDA_GRAPH_IMPL_H_
