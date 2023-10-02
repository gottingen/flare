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

#ifndef FLARE_BACKEND_CUDA_SETUP_CUDA_H_
#define FLARE_BACKEND_CUDA_SETUP_CUDA_H_

#if defined(FLARE_ON_CUDA_DEVICE)

// Compiling with a CUDA compiler.
//
//  Include <cuda.h> to pick up the CUDA_VERSION macro defined as:
//    CUDA_VERSION = ( MAJOR_VERSION * 1000 ) + ( MINOR_VERSION * 10 )
//
//  When generating device code the __CUDA_ARCH__ macro is defined as:
//    __CUDA_ARCH__ = ( MAJOR_CAPABILITY * 100 ) + ( MINOR_CAPABILITY * 10 )

#include <cuda_runtime.h>
#include <cuda.h>

#if defined(_WIN32)
#define FLARE_IMPL_WINDOWS_CUDA
#endif

#if !defined(CUDA_VERSION)
#error "#include <cuda.h> did not define CUDA_VERSION."
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 300)
// Compiling with CUDA compiler for device code.
#error "Cuda device capability >= 3.0 is required."
#endif

#define FLARE_LAMBDA [=] __host__ __device__
#define FLARE_CLASS_LAMBDA [ =, *this ] __host__ __device__

#define FLARE_IMPL_FORCEINLINE_FUNCTION __device__ __host__ __forceinline__
#define FLARE_IMPL_FORCEINLINE __forceinline__
#define FLARE_IMPL_INLINE_FUNCTION __device__ __host__ inline
#define FLARE_IMPL_FUNCTION __device__ __host__
#define FLARE_IMPL_HOST_FUNCTION __host__
#define FLARE_IMPL_DEVICE_FUNCTION __device__
#endif
#endif  // FLARE_BACKEND_CUDA_SETUP_CUDA_H_
