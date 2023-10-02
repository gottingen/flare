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


#ifndef FLARE_CORE_ATOMIC_GENERIC_H_
#define FLARE_CORE_ATOMIC_GENERIC_H_
#include <flare/core/atomic/common.h>
#include <flare/core/atomic/compare_exchange.h>
#include <flare/core/atomic/fetch_op.h>
#include <flare/core/atomic/lock_array.h>
#include <flare/core/defines.h>
#include <flare/core/atomic/thread_fence.h>
#include <type_traits>

namespace flare {

template <class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_thread_fence(MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_thread_fence(order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_thread_fence(order, scope);)
}
template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_exchange(T* dest, T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_exchange(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_exchange(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_compare_exchange(T* dest, T cmp, T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(
      return detail::device_atomic_compare_exchange(dest, cmp, val, order, scope);)
  FLARE_IF_ON_HOST(
      return detail::host_atomic_compare_exchange(dest, cmp, val, order, scope);)
}

// Fetch_Oper atomics: return value before operation
template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_add(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_add(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_add(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_sub(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_sub(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_sub(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_max(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_max(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_max(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_min(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_min(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_min(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_mul(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_mul(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_mul(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_div(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_div(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_div(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_mod(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_mod(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_mod(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_and(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_and(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_and(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_or(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_or(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_or(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_xor(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_xor(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_xor(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_nand(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_nand(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_nand(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_fetch_lshift(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_lshift(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_lshift(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_fetch_rshift(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_rshift(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_rshift(dest, val, order, scope);)
}

// Oper Fetch atomics: return value after operation
template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_add_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_add_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_add_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_sub_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_sub_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_sub_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_max_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_max_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_max_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_min_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_min_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_min_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_mul_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_mul_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_mul_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_div_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_div_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_div_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_mod_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_mod_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_mod_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_and_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_and_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_and_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_or_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_or_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_or_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_xor_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_xor_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_xor_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_nand_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_nand_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_nand_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_lshift_fetch(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_lshift_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_lshift_fetch(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_rshift_fetch(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_rshift_fetch(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_rshift_fetch(dest, val, order, scope);)
}

// Other atomics

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_load(const T* const dest,
                                    MemoryOrder order,
                                    MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_load(dest, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_load(dest, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_store(T* const dest,
                                        const T val,
                                        MemoryOrder order,
                                        MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_store(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_store(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_add(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_add(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_add(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_sub(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_sub(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_sub(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_mul(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_mul(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_mul(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_div(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_div(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_div(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_min(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_min(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_min(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_max(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_max(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_max(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_inc_fetch(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_inc_fetch(dest, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_inc_fetch(dest, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_dec_fetch(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_dec_fetch(dest, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_dec_fetch(dest, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_fetch_inc(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_inc(dest, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_inc(dest, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_inc_mod(T* const dest, T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_inc_mod(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_inc_mod(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T atomic_fetch_dec(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_dec(dest, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_dec(dest, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION T
atomic_fetch_dec_mod(T* const dest, T val, MemoryOrder order, MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_fetch_dec_mod(dest, val, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_fetch_dec_mod(dest, val, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_inc(T* const dest,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_inc(dest, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_inc(dest, order, scope);)
}

template <class T, class MemoryOrder, class MemoryScope>
FLARE_INLINE_FUNCTION void atomic_dec(T* const dest,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  FLARE_IF_ON_DEVICE(return detail::device_atomic_dec(dest, order, scope);)
  FLARE_IF_ON_HOST(return detail::host_atomic_dec(dest, order, scope);)
}

// FIXME
template <class T,
          class SuccessMemoryOrder,
          class FailureMemoryOrder,
          class MemoryScope>
FLARE_INLINE_FUNCTION bool atomic_compare_exchange_strong(
    T* const dest,
    T& expected,
    T desired,
    SuccessMemoryOrder success,
    FailureMemoryOrder /*failure*/,
    MemoryScope scope) {
  T const old = atomic_compare_exchange(dest, expected, desired, success, scope);
  if (old != expected) {
    expected = old;
    return false;
  } else {
    return true;
  }
}

template <class T,
          class SuccessMemoryOrder,
          class FailureMemoryOrder,
          class MemoryScope>
FLARE_INLINE_FUNCTION bool atomic_compare_exchange_weak(T* const dest,
                                                        T& expected,
                                                        T desired,
                                                        SuccessMemoryOrder success,
                                                        FailureMemoryOrder failure,
                                                        MemoryScope scope) {
  return atomic_compare_exchange_strong(
      dest, expected, desired, success, failure, scope);
}

}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_GENERIC_H_
