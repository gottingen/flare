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

#ifndef FLARE_ATOMIC_COMPILE_CONFIG_H_
#define FLARE_ATOMIC_COMPILE_CONFIG_H_

#include <flare/core/defines.h>

#if defined(FLARE_ARCH_KEPLER) || defined(FLARE_ARCH_MAXWELL)
#define FLARE_CUDA_ARCH_IS_PRE_PASCAL
#endif

#if defined(FLARE_ARCH_KEPLER) || defined(FLARE_ARCH_MAXWELL) || \
    defined(FLARE_ARCH_PASCAL)
#define FLARE_CUDA_ARCH_IS_PRE_VOLTA
#endif

#endif   // FLARE_ATOMIC_COMPILE_CONFIG_H_
