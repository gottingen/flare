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
#ifndef FLARE_SIMD_H_
#define FLARE_SIMD_H_
#include <flare/core/defines.h>
#include <flare/core/simd_config.h>
#include <flare/simd/arch/scalar.h>
#include <flare/simd/aligned_allocator.h>

#if defined(FLARE_SIMD_NO_SUPPORTED_ARCHITECTURE)
// to type definition or anything appart from scalar definition and aligned allocator
#else
#include <flare/simd/types/batch.h>
#include <flare/simd/types/batch_constant.h>
#include <flare/simd/types/traits.h>

// This include must come last
#include <flare/simd/types/api.h>
#endif
#endif  // FLARE_SIMD_H_
