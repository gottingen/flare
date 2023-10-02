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


#ifndef FLARE_CORE_COMMON_FWD_H_
#define FLARE_CORE_COMMON_FWD_H_

#include <flare/core/defines.h>

#if defined(FLARE_ON_CUDA_DEVICE)
namespace flare {

class CudaSpace;            ///< Memory space on Cuda GPU
class CudaUVMSpace;         ///< Memory space on Cuda GPU with UVM
class CudaHostPinnedSpace;  ///< Memory space on Host accessible to Cuda GPU
class Cuda;                 ///< Execution space for Cuda GPU

namespace detail {

template <class ExecSpace>
void cuda_prefetch_pointer(const ExecSpace& /*space*/, const void* /*ptr*/,
                           size_t /*bytes*/, bool /*to_device*/) {}

void cuda_prefetch_pointer(const Cuda& space, const void* ptr, size_t bytes,
                           bool to_device);

}  // namespace detail
}  // namespace flare
#endif

#ifdef FLARE_ENABLE_HBWSPACE
namespace flare {

namespace experimental {
class HBWSpace;  /// Memory space for hbw_malloc from memkind (e.g. for KNL
                 /// processor)
}  // namespace experimental
}  // namespace flare
#endif

#if defined(FLARE_ENABLE_OPENMP)
namespace flare {
class OpenMP;  ///< OpenMP execution space.
}  // namespace flare
#endif

#if defined(FLARE_ENABLE_SERIAL)
namespace flare {
class Serial;  ///< Execution space main process on CPU.
}  // namespace flare
#endif

#if defined(FLARE_ENABLE_THREADS)
namespace flare {
class Threads;  ///< Execution space with C++11 threads back-end.
}  // namespace flare
#endif

#endif   // FLARE_CORE_COMMON_FWD_H_
