// Copyright 2023 The EA Authors.
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

#pragma once

#ifdef __DH__
#undef __DH__
#endif

#ifdef __CUDACC_RTC__
#define __DH__ __device__
#else
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define __DH__ __device__ __host__
#else
#define __DH__
#endif
#endif

namespace flare {
namespace cuda {}  // namespace cuda
}  // namespace flare

namespace detail = flare::cuda;
