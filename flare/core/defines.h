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

#ifndef FLARE_CORE_DEFINES_H_
#define FLARE_CORE_DEFINES_H_

//----------------------------------------------------------------------------
/** Pick up configure / build options via #define macros:
 *
 *  FLARE_ENABLE_CUDA                flare::Cuda execution and memory spaces
 *  FLARE_ENABLE_THREADS             flare::Threads execution space
 *  FLARE_ENABLE_OPENMP              flare::OpenMP execution space
 *                                    execution space
 *  FLARE_ENABLE_HWLOC               HWLOC library is available.
 *  FLARE_ENABLE_DEBUG_BOUNDS_CHECK  Insert array bounds checks, is expensive!
 */

#define FLARE_VERSION_LESS(MAJOR, MINOR, PATCH) \
  (FLARE_VERSION < ((MAJOR)*10000 + (MINOR)*100 + (PATCH)))

#define FLARE_VERSION_LESS_EQUAL(MAJOR, MINOR, PATCH) \
  (FLARE_VERSION <= ((MAJOR)*10000 + (MINOR)*100 + (PATCH)))

#define FLARE_VERSION_GREATER(MAJOR, MINOR, PATCH) \
  (FLARE_VERSION > ((MAJOR)*10000 + (MINOR)*100 + (PATCH)))

#define FLARE_VERSION_GREATER_EQUAL(MAJOR, MINOR, PATCH) \
  (FLARE_VERSION >= ((MAJOR)*10000 + (MINOR)*100 + (PATCH)))

#define FLARE_VERSION_EQUAL(MAJOR, MINOR, PATCH) \
  (FLARE_VERSION == ((MAJOR)*10000 + (MINOR)*100 + (PATCH)))

#if !FLARE_VERSION_EQUAL(FLARE_VERSION_MAJOR, FLARE_VERSION_MINOR, \
                          FLARE_VERSION_PATCH)
#error implementation bug
#endif

#include <flare/core/core_config.h>
#include <flare/backend/cuda/nvidia_gpu_architectures.h>

//----------------------------------------------------------------------------
/** Pick up compiler specific #define macros:
 *
 *  Macros for known compilers evaluate to an integral verrm -rf lue
 *
 *  FLARE_COMPILER_NVCC
 *  FLARE_COMPILER_GNU
 *  FLARE_COMPILER_INTEL
 *  FLARE_COMPILER_INTEL_LLVM
 *  FLARE_COMPILER_CRAYC
 *  FLARE_COMPILER_APPLECC
 *  FLARE_COMPILER_CLANG
 *  FLARE_COMPILER_NVHPC
 *  FLARE_COMPILER_MSVC
 *
 *  A suite of 'FLARE_ENABLE_PRAGMA_...' are defined for internal use.
 *
 *  Macros for marking functions to run in an execution space:
 *
 *  FLARE_FUNCTION
 *  FLARE_INLINE_FUNCTION        request compiler to inline
 *  FLARE_FORCEINLINE_FUNCTION   force compiler to inline, use with care!
 */

//----------------------------------------------------------------------------

#if !defined(FLARE_ENABLE_THREADS) && !defined(FLARE_ENABLE_CUDA) && !defined(FLARE_ENABLE_OPENMP)
#define FLARE_INTERNAL_NOT_PARALLEL
#endif

#define FLARE_ENABLE_CXX11_DISPATCH_LAMBDA

#if defined(FLARE_ENABLE_CUDA) && defined(__CUDACC__)
#define FLARE_ON_CUDA_DEVICE
#endif

#if defined(FLARE_ENABLE_CUDA)
#include <flare/backend/cuda/cuda_defines.h>
#endif  // FLARE_ENABLE_CUDA

//----------------------------------------------------------------------------
// Mapping compiler built-ins to FLARE_COMPILER_*** macros

#if defined(__NVCC__)
// NVIDIA compiler is being used.
// Code is parsed and separated into host and device code.
// Host code is compiled again with another compiler.
// Device code is compile to 'ptx'.
// NOTE: There is no __CUDACC_VER_PATCH__ officially, its __CUDACC_VER_BUILD__
// which does have more than one digit (potentially undefined number of them).
// This macro definition is in line with our other compiler defs
#define FLARE_COMPILER_NVCC \
  __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10
#endif  // #if defined( __NVCC__ )

/// atomics
#if defined(FLARE_ON_CUDA_DEVICE)
#define FLARE_HAVE_CUDA_ATOMICS
#endif

#if defined(__GNUC__)
#define FLARE_HAVE_GCC_ATOMICS
#endif

// Equivalent to above for MSVC atomics
#if defined(_MSC_VER)
#define FLARE_HAVE_MSVC_ATOMICS
#endif

#if !defined(FLARE_LAMBDA)
#define FLARE_LAMBDA [=]
#endif

#if !defined(FLARE_CLASS_LAMBDA)
#define FLARE_CLASS_LAMBDA [ =, *this ]
#endif

//#if !defined( __CUDA_ARCH__ ) // Not compiling Cuda code to 'ptx'.

// Intel compiler for host code.

#if defined(__INTEL_COMPILER)
#define FLARE_COMPILER_INTEL __INTEL_COMPILER

#elif defined(__INTEL_LLVM_COMPILER)
#define FLARE_COMPILER_INTEL_LLVM __INTEL_LLVM_COMPILER

// Cray compiler for device offload code
#elif defined(__cray__) && defined(__clang__)
#define FLARE_COMPILER_CRAY_LLVM \
  __cray_major__ * 100 + __cray_minor__ * 10 + __cray_patchlevel__

#elif defined(_CRAYC)
// CRAY compiler for host code
#define FLARE_COMPILER_CRAYC _CRAYC

#elif defined(__APPLE_CC__)
#define FLARE_COMPILER_APPLECC __APPLE_CC__

#elif defined(__NVCOMPILER)
#define FLARE_COMPILER_NVHPC                                 \
  __NVCOMPILER_MAJOR__ * 10000 + __NVCOMPILER_MINOR__ * 100 + \
      __NVCOMPILER_PATCHLEVEL__

#elif defined(__clang__)
// Check this after the Clang-based proprietary compilers which will also define
// __clang__
#define FLARE_COMPILER_CLANG \
  __clang_major__ * 100 + __clang_minor__ * 10 + __clang_patchlevel__

#elif defined(__GNUC__)
// Check this here because many compilers (at least Clang variants and Intel
// classic) define `__GNUC__` for compatibility
#define FLARE_COMPILER_GNU \
  __GNUC__ * 100 + __GNUC_MINOR__ * 10 + __GNUC_PATCHLEVEL__

#if (820 > FLARE_COMPILER_GNU)
#error "Compiling with GCC version earlier than 8.2.0 is not supported."
#endif

#elif defined(_MSC_VER)
// Check this after Intel and Clang because those define _MSC_VER for
// compatibility
#define FLARE_COMPILER_MSVC _MSC_VER
#endif

#if defined(_OPENMP)
//  Compiling with OpenMP.
//  The value of _OPENMP is an integer value YYYYMM
//  where YYYY and MM are the year and month designation
//  of the supported OpenMP API version.
#endif  // #if defined( _OPENMP )

//----------------------------------------------------------------------------
// Intel compiler macros

#if defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
#if defined(FLARE_COMPILER_INTEL_LLVM) && \
    FLARE_COMPILER_INTEL_LLVM >= 20230100
#define FLARE_ENABLE_PRAGMA_UNROLL 1
#define FLARE_ENABLE_PRAGMA_LOOPCOUNT 1
#define FLARE_ENABLE_PRAGMA_VECTOR 1

#define FLARE_ENABLE_PRAGMA_IVDEP 1
#endif

#if !defined(FLARE_MEMORY_ALIGNMENT)
#define FLARE_MEMORY_ALIGNMENT 64
#endif

#if defined(_WIN32)
#define FLARE_RESTRICT __restrict
#else
#define FLARE_RESTRICT __restrict__
#endif

#ifndef FLARE_IMPL_ALIGN_PTR
#if defined(_WIN32)
#define FLARE_IMPL_ALIGN_PTR(size) __declspec(align_value(size))
#else
#define FLARE_IMPL_ALIGN_PTR(size) __attribute__((align_value(size)))
#endif
#endif

#if defined(FLARE_COMPILER_INTEL) && (1900 > FLARE_COMPILER_INTEL)
#error "Compiling with Intel version earlier than 19.0.5 is not supported."
#endif

#if !defined(FLARE_ENABLE_ASM) && !defined(_WIN32)
#define FLARE_ENABLE_ASM 1
#endif

#if !defined(FLARE_IMPL_HOST_FORCEINLINE_FUNCTION)
#if !defined(_WIN32)
#define FLARE_IMPL_HOST_FORCEINLINE_FUNCTION \
  inline __attribute__((always_inline))
#define FLARE_IMPL_HOST_FORCEINLINE __attribute__((always_inline))
#else
#define FLARE_IMPL_HOST_FORCEINLINE_FUNCTION inline
#endif
#endif

#if defined(__MIC__)
// Compiling for Xeon Phi
#endif
#endif

//------------------------------------------------------`----------------------
// Cray compiler macros

#if defined(FLARE_COMPILER_CRAYC)
#endif

//----------------------------------------------------------------------------
// CLANG compiler macros

#if defined(FLARE_COMPILER_CLANG)
//#define FLARE_ENABLE_PRAGMA_UNROLL 1
//#define FLARE_ENABLE_PRAGMA_IVDEP 1
//#define FLARE_ENABLE_PRAGMA_LOOPCOUNT 1
//#define FLARE_ENABLE_PRAGMA_VECTOR 1

#if !defined(FLARE_IMPL_HOST_FORCEINLINE_FUNCTION)
#define FLARE_IMPL_HOST_FORCEINLINE_FUNCTION \
  inline __attribute__((always_inline))
#define FLARE_IMPL_HOST_FORCEINLINE __attribute__((always_inline))
#endif

#if !defined(FLARE_IMPL_ALIGN_PTR)
#define FLARE_IMPL_ALIGN_PTR(size) __attribute__((aligned(size)))
#endif

#endif

//----------------------------------------------------------------------------
// GNU Compiler macros

#if defined(FLARE_COMPILER_GNU)
//#define FLARE_ENABLE_PRAGMA_UNROLL 1
//#define FLARE_ENABLE_PRAGMA_IVDEP 1
//#define FLARE_ENABLE_PRAGMA_LOOPCOUNT 1
//#define FLARE_ENABLE_PRAGMA_VECTOR 1

#if !defined(FLARE_IMPL_HOST_FORCEINLINE_FUNCTION)
#define FLARE_IMPL_HOST_FORCEINLINE_FUNCTION \
  inline __attribute__((always_inline))
#define FLARE_IMPL_HOST_FORCEINLINE __attribute__((always_inline))
#endif

#define FLARE_RESTRICT __restrict__

#if !defined(FLARE_ENABLE_ASM) && !defined(__PGIC__) &&            \
    (defined(__amd64) || defined(__amd64__) || defined(__x86_64) || \
     defined(__x86_64__) || defined(__PPC64__))
#define FLARE_ENABLE_ASM 1
#endif
#endif

//----------------------------------------------------------------------------

#if defined(FLARE_COMPILER_NVHPC)
#define FLARE_ENABLE_PRAGMA_UNROLL 1
#define FLARE_ENABLE_PRAGMA_IVDEP 1
//#define FLARE_ENABLE_PRAGMA_LOOPCOUNT 1
#define FLARE_ENABLE_PRAGMA_VECTOR 1
#endif

//----------------------------------------------------------------------------

#if defined(FLARE_COMPILER_NVCC)
#if defined(__CUDA_ARCH__)
#define FLARE_ENABLE_PRAGMA_UNROLL 1
#endif
#endif

//----------------------------------------------------------------------------
// Define function marking macros if compiler specific macros are undefined:

#if !defined(FLARE_IMPL_HOST_FORCEINLINE_FUNCTION)
#define FLARE_IMPL_HOST_FORCEINLINE_FUNCTION inline
#endif

#if !defined(FLARE_IMPL_HOST_FORCEINLINE)
#define FLARE_IMPL_HOST_FORCEINLINE inline
#endif

#if !defined(FLARE_IMPL_FORCEINLINE_FUNCTION)
#define FLARE_IMPL_FORCEINLINE_FUNCTION FLARE_IMPL_HOST_FORCEINLINE_FUNCTION
#endif

#if !defined(FLARE_IMPL_FORCEINLINE)
#define FLARE_IMPL_FORCEINLINE FLARE_IMPL_HOST_FORCEINLINE
#endif

#if !defined(FLARE_IMPL_INLINE_FUNCTION)
#define FLARE_IMPL_INLINE_FUNCTION inline
#endif

#if !defined(FLARE_IMPL_FUNCTION)
#define FLARE_IMPL_FUNCTION /**/
#endif

#if !defined(FLARE_INLINE_FUNCTION_DELETED)
#define FLARE_INLINE_FUNCTION_DELETED
#endif

#if !defined(FLARE_DEFAULTED_FUNCTION)
#define FLARE_DEFAULTED_FUNCTION
#endif

#if !defined(FLARE_IMPL_HOST_FUNCTION)
#define FLARE_IMPL_HOST_FUNCTION
#endif

#if !defined(FLARE_IMPL_DEVICE_FUNCTION)
#define FLARE_IMPL_DEVICE_FUNCTION
#endif

// Temporary solution for ARCH not supporting printf in kernels.
// Might disappear at any point once we have found another solution.
#if !defined(FLARE_IMPL_DO_NOT_USE_PRINTF)
#define FLARE_IMPL_DO_NOT_USE_PRINTF(...) ::printf(__VA_ARGS__)
#endif

//----------------------------------------------------------------------------
// Define final version of functions. This is so that clang tidy can find these
// macros more easily
#if defined(__clang_analyzer__)
#define FLARE_FUNCTION \
  FLARE_IMPL_FUNCTION __attribute__((annotate("FLARE_FUNCTION")))
#define FLARE_INLINE_FUNCTION \
  FLARE_IMPL_INLINE_FUNCTION  \
  __attribute__((annotate("FLARE_INLINE_FUNCTION")))
#define FLARE_FORCEINLINE_FUNCTION \
  FLARE_IMPL_FORCEINLINE_FUNCTION  \
  __attribute__((annotate("FLARE_FORCEINLINE_FUNCTION")))
#else
#define FLARE_FUNCTION FLARE_IMPL_FUNCTION
#define FLARE_INLINE_FUNCTION FLARE_IMPL_INLINE_FUNCTION
#define FLARE_FORCEINLINE_FUNCTION FLARE_IMPL_FORCEINLINE_FUNCTION
#endif

//----------------------------------------------------------------------------
// Define empty macro for restrict if necessary:

#if !defined(FLARE_RESTRICT)
#define FLARE_RESTRICT
#endif

//----------------------------------------------------------------------------
// Define Macro for alignment:

#if !defined(FLARE_MEMORY_ALIGNMENT)
#define FLARE_MEMORY_ALIGNMENT 64
#endif

#if !defined(FLARE_MEMORY_ALIGNMENT_THRESHOLD)
#define FLARE_MEMORY_ALIGNMENT_THRESHOLD 1
#endif

#if !defined(FLARE_IMPL_ALIGN_PTR)
#define FLARE_IMPL_ALIGN_PTR(size) /* */
#endif

//----------------------------------------------------------------------------
// Determine the default execution space for parallel dispatch.
// There is zero or one default execution space specified.

#if 1 < ((defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_CUDA) ? 1 : 0) +         \
         (defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) ? 1 : 0) +       \
         (defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) ? 1 : 0) +      \
         (defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL) ? 1 : 0))
#error "More than one FLARE_ENABLE_DEFAULT_DEVICE_TYPE_* specified."
#endif

// If default is not specified then chose from enabled execution spaces.
// Priority: CUDA, OPENMP, THREADS, SERIAL
#if defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_CUDA)
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP)
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
#elif defined(FLARE_ON_CUDA_DEVICE)
#define FLARE_ENABLE_DEFAULT_DEVICE_TYPE_CUDA
#elif defined(FLARE_ENABLE_OPENMP)
#define FLARE_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP
#elif defined(FLARE_ENABLE_THREADS)
#define FLARE_ENABLE_DEFAULT_DEVICE_TYPE_THREADS
#else
#define FLARE_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
#endif


// Remove surrounding parentheses if present
#define FLARE_IMPL_STRIP_PARENS(X) FLARE_IMPL_ESC(FLARE_IMPL_ISH X)
#define FLARE_IMPL_ISH(...) FLARE_IMPL_ISH __VA_ARGS__
#define FLARE_IMPL_ESC(...) FLARE_IMPL_ESC_(__VA_ARGS__)
#define FLARE_IMPL_ESC_(...) FLARE_IMPL_VAN_##__VA_ARGS__
#define FLARE_IMPL_VAN_FLARE_IMPL_ISH

#if defined(FLARE_ENABLE_CUDA) && defined(FLARE_COMPILER_NVHPC)
#include <nv/target>
#define FLARE_IF_ON_DEVICE(CODE) NV_IF_TARGET(NV_IS_DEVICE, CODE)
#define FLARE_IF_ON_HOST(CODE) NV_IF_TARGET(NV_IS_HOST, CODE)
#endif

#if !defined(FLARE_IF_ON_HOST) && !defined(FLARE_IF_ON_DEVICE)
#if (defined(FLARE_ENABLE_CUDA) && defined(__CUDA_ARCH__))
#define FLARE_IF_ON_DEVICE(CODE) \
  { FLARE_IMPL_STRIP_PARENS(CODE) }
#define FLARE_IF_ON_HOST(CODE) \
  {}
#else
#define FLARE_IF_ON_DEVICE(CODE) \
  {}
#define FLARE_IF_ON_HOST(CODE) \
  { FLARE_IMPL_STRIP_PARENS(CODE) }
#endif
#endif


#define FLARE_INVALID_INDEX (~std::size_t(0))

#define FLARE_IMPL_CTOR_DEFAULT_ARG FLARE_INVALID_INDEX

// Guard intel compiler version 19 and older
// intel error #2651: attribute does not apply to any entity
// using <deprecated_type> FLARE_DEPRECATED = ...
#if defined(FLARE_ENABLE_DEPRECATION_WARNINGS) && !defined(__NVCC__) && \
    (!defined(FLARE_COMPILER_INTEL) || FLARE_COMPILER_INTEL >= 2021)
#define FLARE_DEPRECATED [[deprecated]]
#define FLARE_DEPRECATED_WITH_COMMENT(comment) [[deprecated(comment)]]
#else
#define FLARE_DEPRECATED
#define FLARE_DEPRECATED_WITH_COMMENT(comment)
#endif

#define FLARE_IMPL_STRINGIFY(x) #x
#define FLARE_IMPL_TOSTRING(x) FLARE_IMPL_STRINGIFY(x)

#ifdef _MSC_VER
#define FLARE_IMPL_DO_PRAGMA(x) __pragma(x)
#define FLARE_IMPL_WARNING(desc) \
  FLARE_IMPL_DO_PRAGMA(message(  \
      __FILE__ "(" FLARE_IMPL_TOSTRING(__LINE__) ") : warning: " #desc))
#else
#define FLARE_IMPL_DO_PRAGMA(x) _Pragma(#x)
#define FLARE_IMPL_WARNING(desc) FLARE_IMPL_DO_PRAGMA(message(#desc))
#endif

#define FLARE_ATTRIBUTE_NODISCARD [[nodiscard]]

#if (defined(FLARE_COMPILER_GNU) || defined(FLARE_COMPILER_CLANG) ||        \
     defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM) || \
     defined(FLARE_COMPILER_NVHPC)) &&                                       \
    !defined(_WIN32) && !defined(__ANDROID__)
#if __has_include(<execinfo.h>)
#define FLARE_IMPL_ENABLE_STACKTRACE
#endif
#define FLARE_IMPL_ENABLE_CXXABI
#endif

// WORKAROUND for AMD aomp which apparently defines CUDA_ARCH when building for
// AMD GPUs with OpenMP Target ???
#if defined(__CUDA_ARCH__) && !defined(__CUDACC__) && !defined(FLARE_ENABLE_CUDA)
#undef __CUDA_ARCH__
#endif

#if (defined(FLARE_IMPL_WINDOWS_CUDA) || defined(FLARE_COMPILER_MSVC)) && \
    !defined(FLARE_COMPILER_CLANG)
// MSVC (as of 16.5.5 at least) does not do empty base class optimization by
// default when there are multiple bases, even though the standard requires it
// for standard layout types.
#define FLARE_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION __declspec(empty_bases)
#else
#define FLARE_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION
#endif

#ifndef FLARE_DEBUG_LEVEL
#define FLARE_DEBUG_LEVEL 1
#endif

#if defined(__has_cpp_attribute)
// if this check passes, then the compiler supports feature test macros
#if __has_cpp_attribute(nodiscard) >= 201603L
// if this check passes, then the compiler supports [[nodiscard]] without a message
#define FLARE_NO_DISCARD [[nodiscard]]
#endif
#endif

#if !defined(FLARE_NO_DISCARD) && __cplusplus >= 201703L
// this means that the previous tests failed, but we are using C++17 or higher
#define FLARE_NO_DISCARD [[nodiscard]]
#endif

#if !defined(FLARE_NO_DISCARD) && (defined(__GNUC__) || defined(__clang__))
// this means that the previous checks failed, but we are using GCC or Clang
#define FLARE_NO_DISCARD __attribute__((warn_unused_result))
#endif

#if !defined(FLARE_NO_DISCARD)
// this means that all the previous checks failed, so we fallback to doing nothing
#define FLARE_NO_DISCARD
#endif

#ifdef __cpp_if_constexpr
// this means that the compiler supports the `if constexpr` construct
#define FLARE_IF_CONSTEXPR if constexpr
#endif

#if !defined(FLARE_IF_CONSTEXPR) && __cplusplus >= 201703L
// this means that the previous test failed, but we are using C++17 or higher
#define FLARE_IF_CONSTEXPR if constexpr
#endif

#if !defined(FLARE_IF_CONSTEXPR)
// this means that all the previous checks failed, so we fallback to a normal `if`
#define FLARE_IF_CONSTEXPR if
#endif

#endif  // #ifndef FLARE_CORE_DEFINES_H_
