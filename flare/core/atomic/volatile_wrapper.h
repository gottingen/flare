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

#ifndef FLARE_ATOMIC_VOLATILE_WRAPPER_H_
#define FLARE_ATOMIC_VOLATILE_WRAPPER_H_
#include <flare/core/defines.h>
#include <flare/core/atomic/compile_config.h>
#include <flare/core/atomic/atomic_ref.h>
#include <flare/core/atomic/compare_exchange.h>
#include <flare/core/atomic/generic.h>
#include <flare/core/atomic/lock_array.h>
#include <flare/core/defines.h>

#ifdef FLARE_INTERNAL_NOT_PARALLEL
#define FLARE_ATOMIC_MEM_SCOPE flare::MemoryScopeCaller()
#else
#define FLARE_ATOMIC_MEM_SCOPE flare::MemoryScopeDevice()
#endif

// clang-format off
namespace flare {

template<class T> FLARE_INLINE_FUNCTION
T atomic_load(volatile T* const dest) { return flare::atomic_load(const_cast<T*>(dest), flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_store(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_store(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

// atomic_fetch_op
template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_add (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_add (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_sub (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_sub (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_max (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_max (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_min (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_min (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_mul (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_mul (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_div (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_div (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_mod (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_mod (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_and (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_and (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_or  (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_or  (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_xor (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_xor (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_nand(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_nand(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_lshift(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_lshift(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_rshift(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_fetch_rshift(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_inc(volatile T* const dest) { return flare::atomic_fetch_inc(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_fetch_dec(volatile T* const dest) { return flare::atomic_fetch_dec(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }


// atomic_op_fetch
template<class T> FLARE_INLINE_FUNCTION
T atomic_add_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_add_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_sub_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_sub_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_max_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_max_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_min_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_min_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_mul_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_mul_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_div_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_div_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_mod_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_mod_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_and_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_and_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_or_fetch  (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_or_fetch  (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_xor_fetch (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_xor_fetch (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_nand_fetch(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_nand_fetch(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_lshift_fetch(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_lshift_fetch(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_rshift_fetch(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_rshift_fetch(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_inc_fetch(volatile T* const dest) { return flare::atomic_inc_fetch(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
T atomic_dec_fetch(volatile T* const dest) { return flare::atomic_dec_fetch(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }


// atomic_op
template<class T> FLARE_INLINE_FUNCTION
void atomic_add(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_add (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_sub(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_sub (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_mul(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_mul (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_div(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_div (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_min(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_min (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_max(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_max (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

// FIXME: Desul doesn't have atomic_and yet so call fetch_and
template<class T> FLARE_INLINE_FUNCTION
void atomic_and(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { (void) flare::atomic_fetch_and (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

// FIXME: Desul doesn't have atomic_or yet so call fetch_or
template<class T> FLARE_INLINE_FUNCTION
void atomic_or (volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { (void) flare::atomic_fetch_or  (const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_inc(volatile T* const dest) { return flare::atomic_inc(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_dec(volatile T* const dest) { return flare::atomic_dec(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_increment(volatile T* const dest) { return flare::atomic_inc(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
void atomic_decrement(volatile T* const dest) { return flare::atomic_dec(const_cast<T*>(dest),flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

// Exchange

template<class T> FLARE_INLINE_FUNCTION
T atomic_exchange(volatile T* const dest, flare::detail::dont_deduce_this_parameter_t<const T> val) { return flare::atomic_exchange(const_cast<T*>(dest), val, flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE); }

template<class T> FLARE_INLINE_FUNCTION
bool atomic_compare_exchange_strong(volatile T* const dest, T& expected, const T desired) {
  return flare::atomic_compare_exchange_strong(const_cast<T*>(dest),expected, desired,
                  flare::MemoryOrderRelaxed(), flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE);
}

template<class T> FLARE_INLINE_FUNCTION
T atomic_compare_exchange(volatile T* const dest, const T compare, const T desired) {
  return flare::atomic_compare_exchange(const_cast<T*>(dest),compare, desired,
                  flare::MemoryOrderRelaxed(), FLARE_ATOMIC_MEM_SCOPE);
}

}
#undef FLARE_ATOMIC_MEM_SCOPE

// clang-format on
#endif  // FLARE_ATOMIC_VOLATILE_WRAPPER_H_
