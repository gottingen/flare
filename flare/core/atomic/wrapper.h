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

#ifndef FLARE_ATOMIC_WRAPPER_H_
#define FLARE_ATOMIC_WRAPPER_H_
#include <flare/core/defines.h>

#include <flare/core/atomic/compile_config.h>
#include <flare/core/atomic/atomic_ref.h>
#include <flare/core/atomic/compare_exchange.h>
#include <flare/core/atomic/generic.h>
#include <flare/core/atomic/lock_array.h>
#include <flare/core/defines.h>

#include <flare/core/common/volatile_load.h>

// clang-format off
namespace flare {

// FIXME: These functions don't have any use/test in unit tests ...
// ==========================================================
inline const char* atomic_query_version() { return "FLARE_ATOMICS"; }

#if defined(FLARE_COMPILER_GNU) && !defined(__PGIC__) && \
    !defined(__CUDA_ARCH__)

#define FLARE_NONTEMPORAL_PREFETCH_LOAD(addr) __builtin_prefetch(addr, 0, 0)
#define FLARE_NONTEMPORAL_PREFETCH_STORE(addr) __builtin_prefetch(addr, 1, 0)

#else

#define FLARE_NONTEMPORAL_PREFETCH_LOAD(addr) ((void)0)
#define FLARE_NONTEMPORAL_PREFETCH_STORE(addr) ((void)0)

#endif
// ============================================================

#ifdef FLARE_INTERNAL_NOT_PARALLEL
#define FLARE_ATOMIC_MEM_SCOPE flare::MemoryScopeCaller()
#else
#define FLARE_ATOMIC_MEM_SCOPE flare::MemoryScopeDevice()
#endif

template<class T> FLARE_INLINE_FUNCTION
T atomic_load(T* const dest) { return flare::atomic_load(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_store(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_store(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_assign(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { atomic_store(dest,val); }

FLARE_INLINE_FUNCTION
void memory_fence() {
  flare::atomic_thread_fence(flare::MemoryOrderSeqCst(), FLARE_ATOMIC_MEM_SCOPE);
}

FLARE_INLINE_FUNCTION
void load_fence() { return flare::atomic_thread_fence(flare::MemoryOrderAcquire(), FLARE_ATOMIC_MEM_SCOPE); }

FLARE_INLINE_FUNCTION
void store_fence() { return flare::atomic_thread_fence(flare::MemoryOrderRelease(), FLARE_ATOMIC_MEM_SCOPE); }

// atomic_fetch_op
template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_add (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_add (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_sub (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_sub (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_max (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_max (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_min (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_min (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_mul (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_mul (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_div (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_div (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_mod (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_mod (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_and (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_and (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_or  (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_or  (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_xor (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_xor (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_nand(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_nand(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_lshift(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_lshift(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_rshift(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_rshift(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_inc(T* const dest) { return flare::atomic_fetch_inc(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_dec(T* const dest) { return flare::atomic_fetch_dec(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }


// atomic_op_fetch
template<class T> FLARE_INLINE_FUNCTION
T atomic_add_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_add_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_sub_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_sub_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_max_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_max_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_min_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_min_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_mul_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_mul_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_div_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_div_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_mod_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_mod_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_and_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_and_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_or_fetch  (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_or_fetch  (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_xor_fetch (T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_xor_fetch (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_nand_fetch(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_nand_fetch(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_lshift_fetch(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_lshift_fetch(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_rshift_fetch(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_rshift_fetch(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_inc_fetch(T* const dest) { return flare::atomic_inc_fetch(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_dec_fetch(T* const dest) { return flare::atomic_dec_fetch(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }


// atomic_op
template<class T> FLARE_INLINE_FUNCTION
void atomic_add(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_add (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_sub(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_sub (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_mul(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_mul (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_div(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_div (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_min(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_min (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_max(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_max (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

// FIXME: Desul doesn't have atomic_and yet so call fetch_and
template<class T> FLARE_INLINE_FUNCTION
void atomic_and(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { (void) flare::atomic_fetch_and (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

// FIXME: Desul doesn't have atomic_or yet so call fetch_or
template<class T> FLARE_INLINE_FUNCTION
void atomic_or(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val)  { (void) flare::atomic_fetch_or (dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_inc(T* const dest) { return flare::atomic_inc(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_dec(T* const dest) { return flare::atomic_dec(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_increment(T* const dest) { return flare::atomic_inc(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_decrement(T* const dest) { return flare::atomic_dec(dest, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

// Exchange

template<class T> FLARE_INLINE_FUNCTION
T atomic_exchange(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_exchange(dest, val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
bool atomic_compare_exchange_strong(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> expected, flare::detail::dont_deduce_this_parameter_t<const T> desired) {
  T expected_ref = expected;
  return flare::atomic_compare_exchange_strong(dest, expected_ref, desired,
                  flare::MemoryOrderRelaxed(), flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE);
}

template<class T> FLARE_INLINE_FUNCTION
T atomic_compare_exchange(T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> compare, flare::detail::dont_deduce_this_parameter_t<const T> desired) {
  return flare::atomic_compare_exchange(dest, compare, desired,
                  flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE);
}

namespace detail {
  template<class T, class MemOrderSuccess, class MemOrderFailure> FLARE_INLINE_FUNCTION
  bool atomic_compare_exchange_strong(T* const dest, T& expected, const T desired, MemOrderSuccess succ, MemOrderFailure fail) {
    return flare::atomic_compare_exchange_strong(dest, expected, desired, succ, fail, FLARE_ATOMIC_MEM_SCOPE);
  }
  template<class T, class MemoryOrder>
  FLARE_INLINE_FUNCTION
  T atomic_load(const T* const src, MemoryOrder order) {
    return flare::atomic_load(src, order, FLARE_ATOMIC_MEM_SCOPE);
  }
  template<class T, class MemoryOrder>
  FLARE_INLINE_FUNCTION
  void atomic_store(T* const src, const T val, MemoryOrder order) {
    return flare::atomic_store(src, val, order, FLARE_ATOMIC_MEM_SCOPE);
  }
}  // namespace detail

}  // namespace flare

#undef FLARE_ATOMIC_MEM_SCOPE

// clang-format on
#endif  // FLARE_ATOMIC_WRAPPER_H_
