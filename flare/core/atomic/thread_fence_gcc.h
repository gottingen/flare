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


#ifndef FLARE_CORE_ATOMIC_THREAD_FENCE_GCC_H_
#define FLARE_CORE_ATOMIC_THREAD_FENCE_GCC_H_

#include <flare/core/atomic/adapt_gcc.h>

namespace flare {
namespace detail {

template <class MemoryOrder, class MemoryScope>
void host_atomic_thread_fence(MemoryOrder, MemoryScope) {
  __atomic_thread_fence(GCCMemoryOrder<MemoryOrder>::value);
}

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_THREAD_FENCE_GCC_H_
