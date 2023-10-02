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

#ifndef FLARE_BACKEND_CUDA_CUDA_GRAPH_NODE_IMPL_H_
#define FLARE_BACKEND_CUDA_CUDA_GRAPH_NODE_IMPL_H_

#include <flare/core/defines.h>

#if defined(FLARE_ON_CUDA_DEVICE)

#include <flare/core/graph/graph_fwd.h>

#include <flare/core/graph/graph_impl.h>  // GraphAccess needs to be complete

#include <flare/backend/cuda/cuda.h>

namespace flare {
namespace detail {

template <>
struct GraphNodeBackendSpecificDetails<flare::Cuda> {
  cudaGraphNode_t node = nullptr;


  explicit GraphNodeBackendSpecificDetails() = default;

  explicit GraphNodeBackendSpecificDetails(
      _graph_node_is_root_ctor_tag) noexcept {}


};

template <class Kernel, class PredecessorRef>
struct GraphNodeBackendDetailsBeforeTypeErasure<flare::Cuda, Kernel,
                                                PredecessorRef> {
 protected:

  GraphNodeBackendDetailsBeforeTypeErasure(
      flare::Cuda const&, Kernel&, PredecessorRef const&,
      GraphNodeBackendSpecificDetails<flare::Cuda>&) noexcept {}

  GraphNodeBackendDetailsBeforeTypeErasure(
      flare::Cuda const&, _graph_node_is_root_ctor_tag,
      GraphNodeBackendSpecificDetails<flare::Cuda>&) noexcept {}

};

}  // end namespace detail
}  // end namespace flare

#include <flare/backend/cuda/cuda_graph_node_kernel.h>

#endif  // defined(FLARE_ON_CUDA_DEVICE)
#endif  // FLARE_BACKEND_CUDA_CUDA_GRAPH_NODE_IMPL_H_
