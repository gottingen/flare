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

#ifndef FLARE_BACKEND_CUDA_CUDA_ABORT_H_
#define FLARE_BACKEND_CUDA_CUDA_ABORT_H_

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#include <flare/core/defines.h>
#if defined(FLARE_ON_CUDA_DEVICE)

#include <cuda.h>

extern "C" {
/*  Cuda runtime function, declared in <crt/device_runtime.h>
 *  Requires capability 2.x or better.
 */
extern __device__ void __assertfail(const void *message, const void *file,
                                    unsigned int line, const void *function,
                                    size_t charsize);
}

namespace flare {
namespace detail {

// required to workaround failures in random number generator unit tests with
// pre-volta architectures
#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
__device__ inline void cuda_abort(const char *const message) {
#else
[[noreturn]] __device__ inline void cuda_abort(const char *const message) {
#endif
  const char empty[] = "";

  __assertfail((const void *)message, (const void *)empty, (unsigned int)0,
               (const void *)empty, sizeof(char));

  // This loop is never executed. It's intended to suppress warnings that the
  // function returns, even though it does not. This is necessary because
  // __assertfail is not marked as [[noreturn]], even though it does not return.
  //  Disable with FLARE_ENABLE_DEBUG_BOUNDS_CHECK to workaround failures
  //  in random number generator unit tests with pre-volta architectures
#if !defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
  while (true)
    ;
#endif
}

}  // namespace detail
}  // namespace flare
#else
void FLARE_CORE_SRC_CUDA_ABORT_PREVENT_LINK_ERROR() {}
#endif  // FLARE_ON_CUDA_DEVICE
#endif  // FLARE_BACKEND_CUDA_CUDA_ABORT_H_
