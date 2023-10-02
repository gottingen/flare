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


#ifndef FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_SCOPE_CALLER_H_
#define FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_SCOPE_CALLER_H_

#include <flare/core/atomic/common.h>

namespace flare {

#define FLARE_ATOMIC_EXCHANGE_SCOPECALLER(MEMORY_ORDER)               \
  template <class T>                                                  \
  FLARE_INLINE_FUNCTION T atomic_exchange(                            \
      T* dest, T value, MEMORY_ORDER, MemoryScopeCaller) {            \
    T return_val = *dest;                                             \
    *dest = value;                                                    \
    return return_val;                                                \
  }                                                                   \
                                                                      \
  template <class T>                                                  \
  FLARE_INLINE_FUNCTION T atomic_compare_exchange(                    \
      T* dest, T compare, T value, MEMORY_ORDER, MemoryScopeCaller) { \
    T current_val = *dest;                                            \
    if (current_val == compare) *dest = value;                        \
    return current_val;                                               \
  }

FLARE_ATOMIC_EXCHANGE_SCOPECALLER(MemoryOrderSeqCst)
FLARE_ATOMIC_EXCHANGE_SCOPECALLER(MemoryOrderAcqRel)
FLARE_ATOMIC_EXCHANGE_SCOPECALLER(MemoryOrderRelease)
FLARE_ATOMIC_EXCHANGE_SCOPECALLER(MemoryOrderAcquire)
FLARE_ATOMIC_EXCHANGE_SCOPECALLER(MemoryOrderRelaxed)

#undef FLARE_ATOMIC_EXCHANGE_SCOPECALLER

}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_SCOPE_CALLER_H_
