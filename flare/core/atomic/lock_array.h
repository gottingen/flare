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


#ifndef FLARE_CORE_ATOMIC_LOCK_ARRAY_H_
#define FLARE_CORE_ATOMIC_LOCK_ARRAY_H_

#include <flare/core/atomic/compare_exchange.h>
#include <flare/core/defines.h>
#ifdef FLARE_HAVE_CUDA_ATOMICS
#include <flare/core/atomic/lock_array_cuda.h>
#endif

namespace flare {
namespace detail {

struct HostLocks {
  static constexpr uint32_t HOST_SPACE_ATOMIC_MASK = 0xFFFF;
  static constexpr uint32_t HOST_SPACE_ATOMIC_XOR_MASK = 0x5A39;
  template <class is_always_void = void>
  static int32_t* get_host_locks_() {
    static int32_t HOST_SPACE_ATOMIC_LOCKS_DEVICE[HOST_SPACE_ATOMIC_MASK + 1] = {};
    return HOST_SPACE_ATOMIC_LOCKS_DEVICE;
  }
  static inline int32_t* get_host_lock_(void* ptr) {
    return &get_host_locks_()[((uint64_t(ptr) >> 2) & HOST_SPACE_ATOMIC_MASK) ^
                              HOST_SPACE_ATOMIC_XOR_MASK];
  }
};

inline void init_lock_arrays() {
  static bool is_initialized = false;
  if (!is_initialized) {
    HostLocks::get_host_locks_();
    is_initialized = true;
  }

#ifdef FLARE_HAVE_CUDA_ATOMICS
  init_lock_arrays_cuda();
#endif

}

inline void finalize_lock_arrays() {
#ifdef FLARE_HAVE_CUDA_ATOMICS
  finalize_lock_arrays_cuda();
#endif

}

inline void ensure_lock_arrays_on_device() {
#ifdef FLARE_HAVE_CUDA_ATOMICS
  ensure_cuda_lock_arrays_on_device();
#endif

}

template <class MemoryScope>
bool lock_address(void* ptr, MemoryScope ms) {
  return 0 == atomic_exchange(
                  HostLocks::get_host_lock_(ptr), int32_t(1), MemoryOrderSeqCst(), ms);
}

template <class MemoryScope>
void unlock_address(void* ptr, MemoryScope ms) {
  (void)atomic_exchange(
      HostLocks::get_host_lock_(ptr), int32_t(0), MemoryOrderSeqCst(), ms);
}

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_LOCK_ARRAY_H_
