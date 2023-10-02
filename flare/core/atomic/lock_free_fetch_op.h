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


#ifndef FLARE_CORE_ATOMIC_LOCK_FREE_FETCH_OP_H_
#define FLARE_CORE_ATOMIC_LOCK_FREE_FETCH_OP_H_

#include <flare/core/atomic/common.h>
#include <flare/core/atomic/compare_exchange.h>
#include <type_traits>

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace flare {
namespace detail {

#define FLARE_IMPL_ATOMIC_FETCH_OPER(ANNOTATION, HOST_OR_DEVICE)                     \
  template <class Oper,                                                              \
            class T,                                                                 \
            class MemoryOrder,                                                       \
            class MemoryScope,                                                       \
            std::enable_if_t<atomic_always_lock_free(sizeof(T)), int> = 0>           \
  ANNOTATION T HOST_OR_DEVICE##_atomic_fetch_oper(                                   \
      const Oper& op,                                                                \
      T* const dest,                                                                 \
      dont_deduce_this_parameter_t<const T> val,                                     \
      MemoryOrder order,                                                             \
      MemoryScope scope) {                                                           \
    using cas_t = atomic_compare_exchange_t<T>;                                      \
    cas_t oldval = reinterpret_cast<cas_t&>(*dest);                                  \
    cas_t assume = oldval;                                                           \
                                                                                     \
    do {                                                                             \
      if (check_early_exit(op, reinterpret_cast<T&>(oldval), val))                   \
        return reinterpret_cast<T&>(oldval);                                         \
      assume = oldval;                                                               \
      T newval = op.apply(reinterpret_cast<T&>(assume), val);                        \
      oldval =                                                                       \
          HOST_OR_DEVICE##_atomic_compare_exchange(reinterpret_cast<cas_t*>(dest),   \
                                                   assume,                           \
                                                   reinterpret_cast<cas_t&>(newval), \
                                                   order,                            \
                                                   scope);                           \
    } while (assume != oldval);                                                      \
                                                                                     \
    return reinterpret_cast<T&>(oldval);                                             \
  }                                                                                  \
                                                                                     \
  template <class Oper,                                                              \
            class T,                                                                 \
            class MemoryOrder,                                                       \
            class MemoryScope,                                                       \
            std::enable_if_t<atomic_always_lock_free(sizeof(T)), int> = 0>           \
  ANNOTATION T HOST_OR_DEVICE##_atomic_oper_fetch(                                   \
      const Oper& op,                                                                \
      T* const dest,                                                                 \
      dont_deduce_this_parameter_t<const T> val,                                     \
      MemoryOrder order,                                                             \
      MemoryScope scope) {                                                           \
    using cas_t = atomic_compare_exchange_t<T>;                                      \
    cas_t oldval = reinterpret_cast<cas_t&>(*dest);                                  \
    T newval = val;                                                                  \
    cas_t assume = oldval;                                                           \
    do {                                                                             \
      if (check_early_exit(op, reinterpret_cast<T&>(oldval), val))                   \
        return reinterpret_cast<T&>(oldval);                                         \
      assume = oldval;                                                               \
      newval = op.apply(reinterpret_cast<T&>(assume), val);                          \
      oldval =                                                                       \
          HOST_OR_DEVICE##_atomic_compare_exchange(reinterpret_cast<cas_t*>(dest),   \
                                                   assume,                           \
                                                   reinterpret_cast<cas_t&>(newval), \
                                                   order,                            \
                                                   scope);                           \
    } while (assume != oldval);                                                      \
                                                                                     \
    return newval;                                                                   \
  }

FLARE_IMPL_ATOMIC_FETCH_OPER(FLARE_IMPL_HOST_FUNCTION, host)
FLARE_IMPL_ATOMIC_FETCH_OPER(FLARE_IMPL_DEVICE_FUNCTION, device)

#undef FLARE_IMPL_ATOMIC_FETCH_OPER

}  // namespace detail
}  // namespace flare

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif  // FLARE_CORE_ATOMIC_LOCK_FREE_FETCH_OP_H_
