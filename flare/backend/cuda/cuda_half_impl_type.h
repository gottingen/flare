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

#ifndef FLARE_BACKEND_CUDA_CUDA_HALF_IMPL_TYPE_H_
#define FLARE_BACKEND_CUDA_CUDA_HALF_IMPL_TYPE_H_

#include <flare/core/defines.h>

#ifdef FLARE_ON_CUDA_DEVICE
#if !(defined(FLARE_COMPILER_CLANG) && FLARE_COMPILER_CLANG < 900) && \
    !(defined(FLARE_ARCH_KEPLER) || defined(FLARE_ARCH_MAXWELL50) || \
      defined(FLARE_ARCH_MAXWELL52))

#include <cuda_fp16.h>

#if (CUDA_VERSION >= 11000)

#include <cuda_bf16.h>

#endif  // CUDA_VERSION >= 11000

#ifndef FLARE_IMPL_HALF_TYPE_DEFINED
// Make sure no one else tries to define half_t
#define FLARE_IMPL_HALF_TYPE_DEFINED
#define FLARE_IMPL_CUDA_HALF_TYPE_DEFINED

namespace flare::detail {
    struct half_impl_t {
        using type = __half;
    };
#if (CUDA_VERSION >= 11000)
#define FLARE_IMPL_BHALF_TYPE_DEFINED
    struct bhalf_impl_t {
        using type = __nv_bfloat16;
    };
#endif  // CUDA_VERSION >= 11000
}  // namespace flare::detail
#endif  // FLARE_IMPL_HALF_TYPE_DEFINED
#endif  // Disables for half_t on cuda:
// Clang/8||KEPLER30||KEPLER32||KEPLER37||MAXWELL50||MAXWELL52
#endif  // FLARE_ON_CUDA_DEVICE
#endif  // FLARE_BACKEND_CUDA_CUDA_HALF_IMPL_TYPE_H_
