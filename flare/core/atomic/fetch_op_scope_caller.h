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


#ifndef FLARE_CORE_ATOMIC_FETCH_OP_SCOPE_CALLER_H_
#define FLARE_CORE_ATOMIC_FETCH_OP_SCOPE_CALLER_H_

#include <flare/core/atomic/common.h>
#include <flare/core/defines.h>

namespace flare {
namespace detail {

#define FLARE_IMPL_ATOMIC_FETCH_OPER(ANNOTATION, HOST_OR_DEVICE) \
  template <class Oper, class T, class MemoryOrder>              \
  ANNOTATION T HOST_OR_DEVICE##_atomic_fetch_oper(               \
      const Oper& op,                                            \
      T* const dest,                                             \
      dont_deduce_this_parameter_t<const T> val,                 \
      MemoryOrder /*order*/,                                     \
      MemoryScopeCaller /*scope*/) {                             \
    T oldval = *dest;                                            \
    *dest = op.apply(oldval, val);                               \
    return oldval;                                               \
  }                                                              \
                                                                 \
  template <class Oper, class T, class MemoryOrder>              \
  ANNOTATION T HOST_OR_DEVICE##_atomic_oper_fetch(               \
      const Oper& op,                                            \
      T* const dest,                                             \
      dont_deduce_this_parameter_t<const T> val,                 \
      MemoryOrder /*order*/,                                     \
      MemoryScopeCaller /*scope*/) {                             \
    T oldval = *dest;                                            \
    T newval = op.apply(oldval, val);                            \
    *dest = newval;                                              \
    return newval;                                               \
  }

FLARE_IMPL_ATOMIC_FETCH_OPER(FLARE_IMPL_HOST_FUNCTION, host)
FLARE_IMPL_ATOMIC_FETCH_OPER(FLARE_IMPL_DEVICE_FUNCTION, device)

#undef FLARE_IMPL_ATOMIC_FETCH_OPER

}  // namespace detail
}  // namespace flare

// FIXME consider implementing directly atomic_fetch_##OP and atomic_##OP##_fetch or
// dropping this placeholder

#endif  // FLARE_CORE_ATOMIC_FETCH_OP_SCOPE_CALLER_H_
