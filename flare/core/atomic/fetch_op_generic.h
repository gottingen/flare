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


#ifndef FLARE_CORE_ATOMIC_FETCH_OP_GENERIC_H_
#define FLARE_CORE_ATOMIC_FETCH_OP_GENERIC_H_

#include <flare/core/atomic/common.h>
#include <flare/core/atomic/lock_based_fetch_op.h>
#include <flare/core/atomic/lock_free_fetch_op.h>
#include <flare/core/atomic/operator_function_objects.h>
#include <type_traits>

namespace flare {
namespace detail {

#define FLARE_IMPL_ATOMIC_FETCH_OP(ANNOTATION, HOST_OR_DEVICE, OP)        \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION T HOST_OR_DEVICE##_atomic_fetch_##OP(                        \
      T* const dest, const T val, MemoryOrder order, MemoryScope scope) { \
    return HOST_OR_DEVICE##_atomic_fetch_oper(                            \
        OP##_operator<T, const T>(), dest, val, order, scope);            \
  }                                                                       \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION T HOST_OR_DEVICE##_atomic_##OP##_fetch(                      \
      T* const dest, const T val, MemoryOrder order, MemoryScope scope) { \
    return HOST_OR_DEVICE##_atomic_oper_fetch(                            \
        OP##_operator<T, const T>(), dest, val, order, scope);            \
  }

#define FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(OP)           \
  FLARE_IMPL_ATOMIC_FETCH_OP(FLARE_IMPL_HOST_FUNCTION, host, OP) \
  FLARE_IMPL_ATOMIC_FETCH_OP(FLARE_IMPL_DEVICE_FUNCTION, device, OP)

FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(add)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(sub)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(max)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(min)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(mul)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(div)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(mod)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(and)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(or)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(xor)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(nand)

FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(inc_mod)
FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE(dec_mod)

#undef FLARE_IMPL_ATOMIC_FETCH_OP_HOST_AND_DEVICE
#undef FLARE_IMPL_ATOMIC_FETCH_OP

#define FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT(ANNOTATION, HOST_OR_DEVICE, OP)             \
  template <class T, class MemoryOrder, class MemoryScope>                           \
  ANNOTATION T HOST_OR_DEVICE##_atomic_fetch_##OP(                                   \
      T* const dest, const unsigned int val, MemoryOrder order, MemoryScope scope) { \
    return HOST_OR_DEVICE##_atomic_fetch_oper(                                       \
        OP##_operator<T, const unsigned int>(), dest, val, order, scope);            \
  }                                                                                  \
  template <class T, class MemoryOrder, class MemoryScope>                           \
  ANNOTATION T HOST_OR_DEVICE##_atomic_##OP##_fetch(                                 \
      T* const dest, const unsigned int val, MemoryOrder order, MemoryScope scope) { \
    return HOST_OR_DEVICE##_atomic_oper_fetch(                                       \
        OP##_operator<T, const unsigned int>(), dest, val, order, scope);            \
  }

#define FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT_HOST_AND_DEVICE(OP)           \
  FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT(FLARE_IMPL_HOST_FUNCTION, host, OP) \
  FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT(FLARE_IMPL_DEVICE_FUNCTION, device, OP)

FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT_HOST_AND_DEVICE(lshift)
FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT_HOST_AND_DEVICE(rshift)

#undef FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT_HOST_AND_DEVICE
#undef FLARE_IMPL_ATOMIC_FETCH_OP_SHIFT

#define FLARE_IMPL_ATOMIC_LOAD_AND_STORE(ANNOTATION, HOST_OR_DEVICE)           \
  template <class T, class MemoryOrder, class MemoryScope>                     \
  ANNOTATION T HOST_OR_DEVICE##_atomic_load(                                   \
      const T* const dest, MemoryOrder order, MemoryScope scope) {             \
    return HOST_OR_DEVICE##_atomic_fetch_oper(                                 \
        load_operator<T, const T>(), const_cast<T*>(dest), T(), order, scope); \
  }                                                                            \
                                                                               \
  template <class T, class MemoryOrder, class MemoryScope>                     \
  ANNOTATION void HOST_OR_DEVICE##_atomic_store(                               \
      T* const dest, const T val, MemoryOrder order, MemoryScope scope) {      \
    (void)HOST_OR_DEVICE##_atomic_fetch_oper(                                  \
        store_operator<T, const T>(), dest, val, order, scope);                \
  }

FLARE_IMPL_ATOMIC_LOAD_AND_STORE(FLARE_IMPL_HOST_FUNCTION, host)
FLARE_IMPL_ATOMIC_LOAD_AND_STORE(FLARE_IMPL_DEVICE_FUNCTION, device)

#undef FLARE_IMPL_ATOMIC_LOAD_AND_STORE

#define FLARE_IMPL_ATOMIC_OP(ANNOTATION, HOST_OR_DEVICE, OP)              \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION void HOST_OR_DEVICE##_atomic_##OP(                           \
      T* const dest, const T val, MemoryOrder order, MemoryScope scope) { \
    (void)HOST_OR_DEVICE##_atomic_fetch_##OP(dest, val, order, scope);    \
  }

#define FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE(OP)           \
  FLARE_IMPL_ATOMIC_OP(FLARE_IMPL_HOST_FUNCTION, host, OP) \
  FLARE_IMPL_ATOMIC_OP(FLARE_IMPL_DEVICE_FUNCTION, device, OP)

FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE(add)
FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE(sub)
FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE(mul)
FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE(div)
FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE(min)
FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE(max)

#undef FLARE_IMPL_ATOMIC_OP_HOST_AND_DEVICE
#undef FLARE_IMPL_ATOMIC_OP

#define FLARE_IMPL_ATOMIC_INCREMENT_DECREMENT(ANNOTATION, HOST_OR_DEVICE) \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION T HOST_OR_DEVICE##_atomic_inc_fetch(                         \
      T* const dest, MemoryOrder order, MemoryScope scope) {              \
    return HOST_OR_DEVICE##_atomic_add_fetch(dest, T(1), order, scope);   \
  }                                                                       \
                                                                          \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION T HOST_OR_DEVICE##_atomic_dec_fetch(                         \
      T* const dest, MemoryOrder order, MemoryScope scope) {              \
    return HOST_OR_DEVICE##_atomic_sub_fetch(dest, T(1), order, scope);   \
  }                                                                       \
                                                                          \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION T HOST_OR_DEVICE##_atomic_fetch_inc(                         \
      T* const dest, MemoryOrder order, MemoryScope scope) {              \
    return HOST_OR_DEVICE##_atomic_fetch_add(dest, T(1), order, scope);   \
  }                                                                       \
                                                                          \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION T HOST_OR_DEVICE##_atomic_fetch_dec(                         \
      T* const dest, MemoryOrder order, MemoryScope scope) {              \
    return HOST_OR_DEVICE##_atomic_fetch_sub(dest, T(1), order, scope);   \
  }                                                                       \
                                                                          \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION void HOST_OR_DEVICE##_atomic_inc(                            \
      T* const dest, MemoryOrder order, MemoryScope scope) {              \
    return HOST_OR_DEVICE##_atomic_add(dest, T(1), order, scope);         \
  }                                                                       \
                                                                          \
  template <class T, class MemoryOrder, class MemoryScope>                \
  ANNOTATION void HOST_OR_DEVICE##_atomic_dec(                            \
      T* const dest, MemoryOrder order, MemoryScope scope) {              \
    return HOST_OR_DEVICE##_atomic_sub(dest, T(1), order, scope);         \
  }

FLARE_IMPL_ATOMIC_INCREMENT_DECREMENT(FLARE_IMPL_HOST_FUNCTION, host)
FLARE_IMPL_ATOMIC_INCREMENT_DECREMENT(FLARE_IMPL_DEVICE_FUNCTION, device)

#undef FLARE_IMPL_ATOMIC_INCREMENT_DECREMENT

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_FETCH_OP_GENERIC_H_
