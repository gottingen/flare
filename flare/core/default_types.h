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


#ifndef FLARE_KERNEL_COMMON_DEFAULT_TYPES_H_
#define FLARE_KERNEL_COMMON_DEFAULT_TYPES_H_

#include <flare/core.h>

#if defined(FLARE_INST_ORDINAL_INT)
using default_lno_t = int;
#elif defined(FLARE_INST_ORDINAL_INT64_T)
using default_lno_t     = int64_t;
#else
// build: default to int
using default_lno_t = int;
#endif
// Prefer int as the default offset type, because cuSPARSE doesn't support
// size_t for rowptrs.
#if defined(FLARE_INST_OFFSET_INT)
using default_size_type = int;
#elif defined(FLARE_INST_OFFSET_SIZE_T)
using default_size_type = size_t;
#else
// build: default to int
using default_size_type = int;
#endif

#if defined(FLARE_INST_LAYOUT_LEFT)
using default_layout = flare::LayoutLeft;
#elif defined(FLARE_INST_LAYOUT_RIGHT)
using default_layout    = flare::LayoutRight;
#else
using default_layout    = flare::LayoutLeft;
#endif

#if defined(FLARE_INST_DOUBLE)
using default_scalar = double;
#elif defined(FLARE_INST_FLOAT)
using default_scalar = float;
#elif defined(FLARE_INST_HALF)
using default_scalar    = flare::experimental::half_t;
#elif defined(FLARE_INST_BHALF)
using default_scalar = flare::experimental::bhalf_t;
#else
using default_scalar = double;
#endif

#if defined(FLARE_ON_CUDA_DEVICE)
using default_device = flare::Cuda;
#elif defined(FLARE_ENABLE_OPENMP)
using default_device = flare::OpenMP;
#elif defined(FLARE_ENABLE_THREADS)
using default_device = flare::Threads;
#else
using default_device = flare::Serial;
#endif

#endif  // FLARE_KERNEL_COMMON_DEFAULT_TYPES_H_
