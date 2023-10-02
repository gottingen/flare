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

#include <flare/core/defines.h>
#if defined(FLARE_CORE_ATOMIC_H_) && !defined(FLARE_CORE_MEMORY_MEMORY_FENCE_H_)
#define FLARE_CORE_MEMORY_MEMORY_FENCE_H_
namespace flare {

//////////////////////////////////////////////////////
// store_fence()
//
// If possible use a store fence on the architecture, if not run a full memory
// fence

FLARE_FORCEINLINE_FUNCTION
void store_fence() {
#if defined(FLARE_ENABLE_ASM) && defined(FLARE_ENABLE_ISA_X86_64)
  asm volatile("sfence" ::: "memory");
#else
  memory_fence();
#endif
}

//////////////////////////////////////////////////////
// load_fence()
//
// If possible use a load fence on the architecture, if not run a full memory
// fence

FLARE_FORCEINLINE_FUNCTION
void load_fence() {
#if defined(FLARE_ENABLE_ASM) && defined(FLARE_ENABLE_ISA_X86_64)
  asm volatile("lfence" ::: "memory");
#else
  memory_fence();
#endif
}

}  // namespace flare

#endif  // FLARE_CORE_MEMORY_MEMORY_FENCE_H_
