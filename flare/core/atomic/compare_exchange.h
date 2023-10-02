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


#ifndef FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_H_
#define FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_H_

#include <flare/core/defines.h>

#ifdef FLARE_HAVE_GCC_ATOMICS
#include <flare/core/atomic/compare_exchange_gcc.h>
#endif
#ifdef FLARE_HAVE_MSVC_ATOMICS
#include <flare/core/atomic/compare_exchange_msvc.h>
#endif
#ifdef FLARE_HAVE_CUDA_ATOMICS
#include <flare/core/atomic/compare_exchange_cuda.h>
#endif


#include <flare/core/atomic/compare_exchange_scope_caller.h>

#endif  // FLARE_CORE_ATOMIC_COMPARE_EXCHANGE_H_
