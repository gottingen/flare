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


#ifndef FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_CUDA_H_
#define FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_CUDA_H_

#include <flare/core/atomic/common.h>
#include <flare/core/atomic/lock_array_cuda.h>
#include <flare/core/atomic/thread_fence_cuda.h>
#include <type_traits>

// Including CUDA ptx based exchange atomics
// When building with clang we need to include the device functions always
// since clang must see a consistent overload set in both device and host compilation
// but that means we need to know on the host what to make visible, i.e. we need
// a host side compile knowledge of architecture.
// We simply can say flare atomic proper doesn't support clang CUDA build pre Volta,
// flare has that knowledge and so I use it here, allowing in flare to use
// clang with pre Volta as CUDA compiler
#ifndef FLARE_CUDA_ARCH_IS_PRE_VOLTA

#include <flare/core/atomic/cuda/cuda_asm_exchange.h>

#else

namespace flare {
namespace detail {

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4, T> device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      reinterpret_cast<unsigned int&>(compare),
                                      reinterpret_cast<unsigned int&>(value));
  return reinterpret_cast<T&>(return_val);
}
template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 8, T> device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long is 64-bit");
  unsigned long long int return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                reinterpret_cast<unsigned long long int&>(compare),
                reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8, T>
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelease, MemoryScope) {
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8, T>
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcquire, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  return return_val;
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8, T>
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcqRel, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4, T> device_atomic_exchange(
    T* const dest, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicExch(reinterpret_cast<unsigned int*>(dest),
                                       reinterpret_cast<unsigned int&>(value));
  return reinterpret_cast<T&>(return_val);
}
template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 8, T> device_atomic_exchange(
    T* const dest, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long is 64-bit");
  unsigned long long int return_val =
      atomicExch(reinterpret_cast<unsigned long long int*>(dest),
                 reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8, T> device_atomic_exchange(
    T* const dest, T value, MemoryOrderRelease, MemoryScope) {
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return reinterpret_cast<T&>(return_val);
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8, T> device_atomic_exchange(
    T* const dest, T value, MemoryOrderAcquire, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  return reinterpret_cast<T&>(return_val);
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8, T> device_atomic_exchange(
    T* const dest, T value, MemoryOrderAcqRel, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return reinterpret_cast<T&>(return_val);
}
}  // namespace detail
}  // namespace flare

#endif

// SeqCst is not directly supported by PTX, need the additional fences:

namespace flare {
namespace detail {
template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4, T> device_atomic_exchange(T* const dest,
                                                                      T value,
                                                                      MemoryOrderSeqCst,
                                                                      MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}
template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 8, T> device_atomic_exchange(T* const dest,
                                                                      T value,
                                                                      MemoryOrderSeqCst,
                                                                      MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}
template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 4, T> device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}
template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 8, T> device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<(sizeof(T) != 8) && (sizeof(T) != 4), T>
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  // This is a way to avoid deadlock in a warp or wave front
  T return_val;
  int done = 0;
  unsigned int mask = __activemask();
  unsigned int active = __ballot_sync(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (lock_address_cuda((void*)dest, scope)) {
        if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
          device_atomic_thread_fence(MemoryOrderRelease(), scope);
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        if (return_val == compare) {
          *dest = value;
          device_atomic_thread_fence(MemoryOrderRelease(), scope);
        }
        unlock_address_cuda((void*)dest, scope);
        done = 1;
      }
    }
    done_active = __ballot_sync(mask, done);
  }
  return return_val;
}

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<(sizeof(T) != 8) && (sizeof(T) != 4), T>
device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope scope) {
  // This is a way to avoid deadlock in a warp or wave front
  T return_val;
  int done = 0;
  unsigned int mask = __activemask();
  unsigned int active = __ballot_sync(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (lock_address_cuda((void*)dest, scope)) {
        if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
          device_atomic_thread_fence(MemoryOrderRelease(), scope);
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        *dest = value;
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

#endif  // FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_CUDA_H_
