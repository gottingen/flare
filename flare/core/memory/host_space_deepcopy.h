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

#ifndef FLARE_CORE_MEMORY_HOST_SPACE_DEEPCOPY_H_
#define FLARE_CORE_MEMORY_HOST_SPACE_DEEPCOPY_H_

#include <cstdint>

namespace flare {

namespace detail {

void hostspace_fence(const DefaultHostExecutionSpace& exec);

void hostspace_parallel_deepcopy(void* dst, const void* src, ptrdiff_t n);
// DeepCopy called with an execution space that can't access HostSpace
void hostspace_parallel_deepcopy_async(void* dst, const void* src, ptrdiff_t n);
void hostspace_parallel_deepcopy_async(const DefaultHostExecutionSpace& exec,
                                       void* dst, const void* src, ptrdiff_t n);

}  // namespace detail

}  // namespace flare

#endif  // FLARE_CORE_MEMORY_HOST_SPACE_DEEPCOPY_H_
