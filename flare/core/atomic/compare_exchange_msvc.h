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


#ifndef FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_MSVC_H_
#define FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_MSVC_H_

#include <flare/core/atomic/common.h>
#include <flare/core/atomic/thread_fence_msvc.h>
#include <type_traits>

#ifndef FLARE_HAVE_16BYTE_COMPARE_AND_SWAP
#define FLARE_HAVE_16BYTE_COMPARE_AND_SWAP
#endif

namespace flare {
namespace detail {

// Forward declare these functions. They use compare_exchange themselves
// so the actual header file with them comes after this file is included.
template <class MemoryScope>
bool lock_address(void* ptr, MemoryScope ms);

template <class MemoryScope>
void unlock_address(void* ptr, MemoryScope ms);

}  // namespace detail
}  // namespace flare

namespace flare {
namespace detail {

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 1, T> host_atomic_exchange(T* const dest,
                                                         T val,
                                                         MemoryOrder,
                                                         MemoryScope) {
  char return_val = _InterlockedExchange8((char*)dest, *((char*)&val));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 2, T> host_atomic_exchange(T* const dest,
                                                         T val,
                                                         MemoryOrder,
                                                         MemoryScope) {
  short return_val = _InterlockedExchange16((short*)dest, *((short*)&val));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 4, T> host_atomic_exchange(T* const dest,
                                                         T val,
                                                         MemoryOrder,
                                                         MemoryScope) {
  long return_val = _InterlockedExchange((long*)dest, *((long*)&val));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 8, T> host_atomic_exchange(T* const dest,
                                                         T val,
                                                         MemoryOrder,
                                                         MemoryScope) {
  __int64 return_val = _InterlockedExchange64((__int64*)dest, *((__int64*)&val));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<(sizeof(T) != 1 && sizeof(T) != 2 && sizeof(T) != 4 && sizeof(T) != 8),
                 T>
host_atomic_exchange(T* const dest, T val, MemoryOrder, MemoryScope scope) {
  while (!lock_address((void*)dest, scope)) {
  }
  if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
    host_atomic_thread_fence(MemoryOrderRelease(), scope);
  host_atomic_thread_fence(MemoryOrderAcquire(), scope);
  T return_val = *dest;
  *dest = val;
  host_atomic_thread_fence(MemoryOrderRelease(), scope);

  unlock_address((void*)dest, scope);
  return return_val;
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 1, T> host_atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrder, MemoryScope) {
  char return_val =
      _InterlockedCompareExchange8((char*)dest, *((char*)&val), *((char*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 2, T> host_atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrder, MemoryScope) {
  short return_val =
      _InterlockedCompareExchange16((short*)dest, *((short*)&val), *((short*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 4, T> host_atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrder, MemoryScope) {
  long return_val =
      _InterlockedCompareExchange((long*)dest, *((long*)&val), *((long*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 8, T> host_atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrder, MemoryScope) {
  __int64 return_val = _InterlockedCompareExchange64(
      (__int64*)dest, *((__int64*)&val), *((__int64*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<sizeof(T) == 16, T> host_atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrder, MemoryScope) {
  Dummy16ByteValue* val16 = reinterpret_cast<Dummy16ByteValue*>(&val);
  (void)_InterlockedCompareExchange128(reinterpret_cast<__int64*>(dest),
                                       val16->value2,
                                       val16->value1,
                                       (reinterpret_cast<__int64*>(&compare)));
  return compare;
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<(sizeof(T) != 1 && sizeof(T) != 2 && sizeof(T) != 4 &&
                  sizeof(T) != 8 && sizeof(T) != 16),
                 T>
host_atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrder, MemoryScope scope) {
  while (!lock_address((void*)dest, scope)) {
  }
  if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
    host_atomic_thread_fence(MemoryOrderRelease(), scope);
  host_atomic_thread_fence(MemoryOrderAcquire(), scope);
  T return_val = *dest;
  if (return_val == compare) {
    *dest = val;
    host_atomic_thread_fence(MemoryOrderRelease(), scope);
  }

  unlock_address((void*)dest, scope);
  return return_val;
}

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_MSVC_H_
