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


#ifndef FLARE_CORE_ATOMIC_FETCH_OP_GCC_H_
#define FLARE_CORE_ATOMIC_FETCH_OP_GCC_H_

#include <flare/core/atomic/adapt_gcc.h>
#include <type_traits>

namespace flare {
namespace detail {

// clang-format off
#define FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(OP, MEMORY_ORDER, MEMORY_SCOPE)                                 \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic_fetch_##OP  (T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic_fetch_##OP  (dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }                                                                                                                              \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic_##OP##_fetch(T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic_##OP##_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }

#define FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(OP) \
   FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(OP, MemoryOrderRelaxed, MemoryScopeNode  ) \
   FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(OP, MemoryOrderRelaxed, MemoryScopeDevice) \
   FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(OP, MemoryOrderRelaxed, MemoryScopeCore  ) \
   FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(OP, MemoryOrderSeqCst , MemoryScopeNode  ) \
   FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(OP, MemoryOrderSeqCst , MemoryScopeDevice) \
   FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(OP, MemoryOrderSeqCst , MemoryScopeCore  )
// clang-format on

FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(add)
FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(sub)
FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(and)
FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(xor)
FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(or)
FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(nand)

#undef FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL
#undef FLARE_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_FETCH_OP_GCC_H_
