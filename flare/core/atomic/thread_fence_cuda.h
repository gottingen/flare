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


#ifndef FLARE_CORE_ATOMIC_THREAD_FENCE_CUDA_H_
#define FLARE_CORE_ATOMIC_THREAD_FENCE_CUDA_H_

#include <flare/core/atomic/common.h>

namespace flare {
namespace detail {

// clang-format off
inline __device__ void device_atomic_thread_fence(MemoryOrderRelease, MemoryScopeDevice) { __threadfence();        }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcquire, MemoryScopeDevice) { __threadfence();        }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcqRel , MemoryScopeDevice) { __threadfence();        }
inline __device__ void device_atomic_thread_fence(MemoryOrderSeqCst , MemoryScopeDevice) { __threadfence();        }
inline __device__ void device_atomic_thread_fence(MemoryOrderRelease, MemoryScopeCore  ) { __threadfence_block();  }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcquire, MemoryScopeCore  ) { __threadfence_block();  }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcqRel , MemoryScopeCore  ) { __threadfence_block();  }
inline __device__ void device_atomic_thread_fence(MemoryOrderSeqCst , MemoryScopeCore  ) { __threadfence_block();  }
#ifndef FLARE_CUDA_ARCH_IS_PRE_PASCAL
inline __device__ void device_atomic_thread_fence(MemoryOrderRelease, MemoryScopeNode  ) { __threadfence_system(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcquire, MemoryScopeNode  ) { __threadfence_system(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcqRel , MemoryScopeNode  ) { __threadfence_system(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderSeqCst , MemoryScopeNode  ) { __threadfence_system(); }
#endif
// clang-format on

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_THREAD_FENCE_CUDA_H_
