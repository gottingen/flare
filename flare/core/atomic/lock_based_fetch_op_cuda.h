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


#ifndef FLARE_CORE_ATOMIC_LOCK_BASED_FETCH_OP_CUDA_H_
#define FLARE_CORE_ATOMIC_LOCK_BASED_FETCH_OP_CUDA_H_

#include <flare/core/atomic/common.h>
#include <flare/core/atomic/lock_array_cuda.h>
#include <flare/core/atomic/thread_fence_cuda.h>
#include <type_traits>

namespace flare {
namespace detail {

template <class Oper,
          class T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires !atomic_always_lock_free(sizeof(T))
          std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0>
__device__ T device_atomic_fetch_oper(const Oper& op,
                                      T* const dest,
                                      dont_deduce_this_parameter_t<const T> val,
                                      MemoryOrder /*order*/,
                                      MemoryScope scope) {
  // This is a way to avoid deadlock in a warp or wave front
  T return_val;
  int done = 0;
  unsigned int mask = __activemask();
  unsigned int active = __ballot_sync(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (lock_address_cuda((void*)dest, scope)) {
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        *dest = op.apply(return_val, val);
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        unlock_address_cuda((void*)dest, scope);
        done = 1;
      }
    }
    done_active = __ballot_sync(mask, done);
  }
  return return_val;
}

template <class Oper,
          class T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires !atomic_always_lock_free(sizeof(T))
          std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0>
__device__ T device_atomic_oper_fetch(const Oper& op,
                                      T* const dest,
                                      dont_deduce_this_parameter_t<const T> val,
                                      MemoryOrder /*order*/,
                                      MemoryScope scope) {
  // This is a way to avoid deadlock in a warp or wave front
  T return_val;
  int done = 0;
  unsigned int mask = __activemask();
  unsigned int active = __ballot_sync(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (lock_address_cuda((void*)dest, scope)) {
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = op.apply(*dest, val);
        *dest = return_val;
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        unlock_address_cuda((void*)dest, scope);
        done = 1;
      }
    }
    done_active = __ballot_sync(mask, done);
  }
  return return_val;
}

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_LOCK_BASED_FETCH_OP_CUDA_H_
