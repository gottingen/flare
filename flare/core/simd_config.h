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

#ifndef FLARE_CORE_SIMD_CONFIG_H_
#define FLARE_CORE_SIMD_CONFIG_H_


/**
 * high level free functions
 *
 * @defgroup flare_simd_config_macro Instruction Set Detection
 */

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if SSE2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE2__
#define FLARE_SIMD_ENABLE_SSE2 1
#else
#define FLARE_SIMD_ENABLE_SSE2 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if SSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE3__
#define FLARE_SIMD_ENABLE_SSE3 1
#else
#define FLARE_SIMD_ENABLE_SSE3 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if SSSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSSE3__
#define FLARE_SIMD_ENABLE_SSSE3 1
#else
#define FLARE_SIMD_ENABLE_SSSE3 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if SSE4.1 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_1__
#define FLARE_SIMD_ENABLE_SSE4_1 1
#else
#define FLARE_SIMD_ENABLE_SSE4_1 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if SSE4.2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_2__
#define FLARE_SIMD_ENABLE_SSE4_2 1
#else
#define FLARE_SIMD_ENABLE_SSE4_2 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX__
#define FLARE_SIMD_ENABLE_AVX 1
#else
#define FLARE_SIMD_ENABLE_AVX 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if AVX2 is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX2__
#define FLARE_SIMD_ENABLE_AVX2 1
#else
#define FLARE_SIMD_ENABLE_AVX2 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if FMA3 for SSE is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__SSE__)
#ifndef FLARE_SIMD_ENABLE_FMA3_SSE // Leave the opportunity to manually disable it, see #643
#define FLARE_SIMD_ENABLE_FMA3_SSE 1
#endif
#else

#if FLARE_SIMD_ENABLE_FMA3_SSE
#error "Manually set FLARE_SIMD_ENABLE_FMA3_SSE is incompatible with current compiler flags"
#endif

#define FLARE_SIMD_ENABLE_FMA3_SSE 0
#endif

#else

#if FLARE_SIMD_ENABLE_FMA3_SSE
#error "Manually set FLARE_SIMD_ENABLE_FMA3_SSE is incompatible with current compiler flags"
#endif

#define FLARE_SIMD_ENABLE_FMA3_SSE 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if FMA3 for AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__AVX__)
#ifndef FLARE_SIMD_ENABLE_FMA3_AVX // Leave the opportunity to manually disable it, see #643
#define FLARE_SIMD_ENABLE_FMA3_AVX 1
#endif
#else

#if FLARE_SIMD_ENABLE_FMA3_AVX
#error "Manually set FLARE_SIMD_ENABLE_FMA3_AVX is incompatible with current compiler flags"
#endif

#define FLARE_SIMD_ENABLE_FMA3_AVX 0
#endif

#if defined(__AVX2__)
#ifndef FLARE_SIMD_ENABLE_FMA3_AVX2 // Leave the opportunity to manually disable it, see #643
#define FLARE_SIMD_ENABLE_FMA3_AVX2 1
#endif
#else

#if FLARE_SIMD_ENABLE_FMA3_AVX2
#error "Manually set FLARE_SIMD_ENABLE_FMA3_AVX2 is incompatible with current compiler flags"
#endif

#define FLARE_SIMD_ENABLE_FMA3_AVX2 0
#endif

#else

#if FLARE_SIMD_ENABLE_FMA3_AVX
#error "Manually set FLARE_SIMD_ENABLE_FMA3_AVX is incompatible with current compiler flags"
#endif

#if FLARE_SIMD_ENABLE_FMA3_AVX2
#error "Manually set FLARE_SIMD_ENABLE_FMA3_AVX2 is incompatible with current compiler flags"
#endif

#define FLARE_SIMD_ENABLE_FMA3_AVX 0
#define FLARE_SIMD_ENABLE_FMA3_AVX2 0

#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if FMA4 is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA4__
#define FLARE_SIMD_ENABLE_FMA4 1
#else
#define FLARE_SIMD_ENABLE_FMA4 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if AVX512F is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512F__
// AVX512 instructions are supported starting with gcc 6
// see https://www.gnu.org/software/gcc/gcc-6/changes.html
// check clang first, newer clang always defines __GNUC__ = 4
#if defined(__clang__) && __clang_major__ >= 6
#define FLARE_SIMD_ENABLE_AVX512F 1
#elif defined(__GNUC__) && __GNUC__ < 6
#define FLARE_SIMD_ENABLE_AVX512F 0
#else
#define FLARE_SIMD_ENABLE_AVX512F 1
#if __GNUC__ == 6
#define FLARE_SIMD_AVX512_SHIFT_INTRINSICS_IMM_ONLY 1
#endif
#endif
#else
#define FLARE_SIMD_ENABLE_AVX512F 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if AVX512CD is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512CD__
// Avoids repeating the GCC workaround over and over
#define FLARE_SIMD_ENABLE_AVX512CD FLARE_SIMD_ENABLE_AVX512F
#else
#define FLARE_SIMD_ENABLE_AVX512CD 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if AVX512DQ is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512DQ__
#define FLARE_SIMD_ENABLE_AVX512DQ FLARE_SIMD_ENABLE_AVX512F
#else
#define FLARE_SIMD_ENABLE_AVX512DQ 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if AVX512BW is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512BW__
#define FLARE_SIMD_ENABLE_AVX512BW FLARE_SIMD_ENABLE_AVX512F
#else
#define FLARE_SIMD_ENABLE_AVX512BW 0
#endif

#ifdef __ARM_NEON

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if NEON is available at compile-time, to 0 otherwise.
 */
#if __ARM_ARCH >= 7
#define FLARE_SIMD_ENABLE_NEON 1
#else
#define FLARE_SIMD_ENABLE_NEON 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if NEON64 is available at compile-time, to 0 otherwise.
 */
#ifdef __aarch64__
#define FLARE_SIMD_ENABLE_NEON64 1
#else
#define FLARE_SIMD_ENABLE_NEON64 0
#endif
#else
#define FLARE_SIMD_ENABLE_NEON 0
#define FLARE_SIMD_ENABLE_NEON64 0
#endif

/**
 * @ingroup flare_simd_config_macro
 *
 * Set to 1 if SVE is available and bit width is pre-set at compile-time, to 0 otherwise.
 */
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
#define FLARE_SIMD_ENABLE_SVE 1
#define FLARE_SIMD_SVE_BITS __ARM_FEATURE_SVE_BITS
#else
#define FLARE_SIMD_ENABLE_SVE 0
#define FLARE_SIMD_SVE_BITS 0
#endif

// Workaround for MSVC compiler
#ifdef _MSC_VER

#if FLARE_SIMD_ENABLE_AVX512

#undef FLARE_SIMD_ENABLE_AVX2
#define FLARE_SIMD_ENABLE_AVX2 1

#endif

#if FLARE_SIMD_ENABLE_AVX2

#undef FLARE_SIMD_ENABLE_AVX
#define FLARE_SIMD_ENABLE_AVX 1

#undef FLARE_SIMD_ENABLE_FMA3_AVX
#define FLARE_SIMD_ENABLE_FMA3_AVX 1

#undef FLARE_SIMD_ENABLE_FMA3_AVX2
#define FLARE_SIMD_ENABLE_FMA3_AVX2 1

#endif

#if FLARE_SIMD_ENABLE_AVX

#undef FLARE_SIMD_ENABLE_SSE4_2
#define FLARE_SIMD_ENABLE_SSE4_2 1

#endif

#if FLARE_SIMD_ENABLE_SSE4_2

#undef FLARE_SIMD_ENABLE_SSE4_1
#define FLARE_SIMD_ENABLE_SSE4_1 1

#endif

#if FLARE_SIMD_ENABLE_SSE4_1

#undef FLARE_SIMD_ENABLE_SSSE3
#define FLARE_SIMD_ENABLE_SSSE3 1

#endif

#if FLARE_SIMD_ENABLE_SSSE3

#undef FLARE_SIMD_ENABLE_SSE3
#define FLARE_SIMD_ENABLE_SSE3 1

#endif

#if FLARE_SIMD_ENABLE_SSE3 || defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#undef FLARE_SIMD_ENABLE_SSE2
#define FLARE_SIMD_ENABLE_SSE2 1
#endif

#endif

#if !FLARE_SIMD_ENABLE_SSE2 && !FLARE_SIMD_ENABLE_SSE3 && !FLARE_SIMD_ENABLE_SSSE3 && !FLARE_SIMD_ENABLE_SSE4_1 && !FLARE_SIMD_ENABLE_SSE4_2 && !FLARE_SIMD_ENABLE_AVX && !FLARE_SIMD_ENABLE_AVX2 && !FLARE_SIMD_ENABLE_FMA3_SSE && !FLARE_SIMD_ENABLE_FMA4 && !FLARE_SIMD_ENABLE_FMA3_AVX && !FLARE_SIMD_ENABLE_FMA3_AVX2 && !FLARE_SIMD_ENABLE_AVX512F && !FLARE_SIMD_ENABLE_AVX512CD && !FLARE_SIMD_ENABLE_AVX512DQ && !FLARE_SIMD_ENABLE_AVX512BW && !FLARE_SIMD_ENABLE_NEON && !FLARE_SIMD_ENABLE_NEON64 && !FLARE_SIMD_ENABLE_SVE
#define FLARE_SIMD_NO_SUPPORTED_ARCHITECTURE
#endif


#endif  // FLARE_CORE_SIMD_CONFIG_H_
