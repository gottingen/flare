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
#ifndef FLARE_SIMD_ARCH_ISA_H_
#define FLARE_SIMD_ARCH_ISA_H_

#include <flare/simd/arch.h>

#include <flare/simd/arch/generic_fwd.h>

#if FLARE_SIMD_ENABLE_SSE2
#include <flare/simd/arch/sse2.h>
#endif

#if FLARE_SIMD_ENABLE_SSE3
#include <flare/simd/arch/sse3.h>
#endif

#if FLARE_SIMD_ENABLE_SSSE3
#include <flare/simd/arch/ssse3.h>
#endif

#if FLARE_SIMD_ENABLE_SSE4_1
#include <flare/simd/arch/sse4_1.h>
#endif

#if FLARE_SIMD_ENABLE_SSE4_2
#include <flare/simd/arch/sse4_2.h>
#endif

#if FLARE_SIMD_ENABLE_FMA3_SSE
#include <flare/simd/arch/fma3_sse.h>
#endif

#if FLARE_SIMD_ENABLE_FMA4
#include <flare/simd/arch/fma4.h>
#endif

#if FLARE_SIMD_ENABLE_AVX
#include <flare/simd/arch/avx.h>
#endif

#if FLARE_SIMD_ENABLE_FMA3_AVX
#include <flare/simd/arch/fma3_avx.h>
#endif

#if FLARE_SIMD_ENABLE_AVX2
#include <flare/simd/arch/avx2.h>
#endif

#if FLARE_SIMD_ENABLE_FMA3_AVX2
#include <flare/simd/arch/fma3_avx2.h>
#endif

#if FLARE_SIMD_ENABLE_AVX512F
#include <flare/simd/arch/avx512f.h>
#endif

#if FLARE_SIMD_ENABLE_AVX512BW
#include <flare/simd/arch/avx512bw.h>
#endif

#if FLARE_SIMD_ENABLE_NEON
#include <flare/simd/arch/neon.h>
#endif

#if FLARE_SIMD_ENABLE_NEON64
#include <flare/simd/arch/neon64.h>
#endif

#if FLARE_SIMD_ENABLE_SVE
#include <flare/simd/arch/sve.h>
#endif

// Must come last to have access to all conversion specializations.
#include <flare/simd/arch/generic.h>

#endif  // FLARE_SIMD_ARCH_ISA_H_
