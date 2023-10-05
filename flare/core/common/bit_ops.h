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

#ifndef FLARE_CORE_COMMON_BIT_OPS_H_
#define FLARE_CORE_COMMON_BIT_OPS_H_

#include <flare/core/defines.h>
#include <cstdint>
#include <climits>

#if defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
#include <immintrin.h>
#endif

namespace flare::detail {

    FLARE_FORCEINLINE_FUNCTION
    int int_log2_fallback(unsigned i) {
        constexpr int shift = sizeof(unsigned) * CHAR_BIT - 1;

        int offset = 0;
        if (i) {
            for (offset = shift; (i & (1 << offset)) == 0; --offset);
        }
        return offset;
    }

    FLARE_IMPL_DEVICE_FUNCTION
    inline int int_log2_device(unsigned i) {
#if defined(FLARE_ON_CUDA_DEVICE)
        constexpr int shift = sizeof(unsigned) * CHAR_BIT - 1;
        return shift - __clz(i);
#elif defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        return _bit_scan_reverse(i);
#else
        return int_log2_fallback(i);
#endif
    }

    FLARE_IMPL_HOST_FUNCTION
    inline int int_log2_host(unsigned i) {
// duplicating shift to avoid unused warning in else branch
#if defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        constexpr int shift = sizeof(unsigned) * CHAR_BIT - 1;
        (void)shift;
        return _bit_scan_reverse(i);
#elif defined(FLARE_COMPILER_CRAYC)
        constexpr int shift = sizeof(unsigned) * CHAR_BIT - 1;
        return i ? shift - _leadz32(i) : 0;
#elif defined(__GNUC__) || defined(__GNUG__)
        constexpr int shift = sizeof(unsigned) * CHAR_BIT - 1;
        return shift - __builtin_clz(i);
#else
        return int_log2_fallback(i);
#endif
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma push
#pragma diag_suppress implicit_return_from_non_void_function
#endif

    FLARE_FORCEINLINE_FUNCTION
    int int_log2(unsigned i) {
        FLARE_IF_ON_DEVICE((return int_log2_device(i);))
        FLARE_IF_ON_HOST((return int_log2_host(i);))
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma pop
#endif

/**\brief  Find first zero bit.
 *
 *  If none then return -1 ;
 */
    FLARE_FORCEINLINE_FUNCTION
    int bit_first_zero_fallback(unsigned i) noexcept {
        constexpr unsigned full = ~0u;

        int offset = -1;
        if (full != i) {
            for (offset = 0; i & (1 << offset); ++offset);
        }
        return offset;
    }

    FLARE_IMPL_DEVICE_FUNCTION
    inline int bit_first_zero_device(unsigned i) noexcept {
        constexpr unsigned full = ~0u;
#if defined(FLARE_ON_CUDA_DEVICE)
        return full != i ? __ffs(~i) - 1 : -1;
#elif defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        return full != i ? _bit_scan_forward(~i) : -1;
#else
        (void) full;
        return bit_first_zero_fallback(i);
#endif
    }

    FLARE_IMPL_HOST_FUNCTION
    inline int bit_first_zero_host(unsigned i) noexcept {
        constexpr unsigned full = ~0u;
#if defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        return full != i ? _bit_scan_forward(~i) : -1;
#elif defined(FLARE_COMPILER_CRAYC)
        return full != i ? _popcnt(i ^ (i + 1)) - 1 : -1;
#elif defined(FLARE_COMPILER_GNU) || defined(__GNUC__) || defined(__GNUG__)
        return full != i ? __builtin_ffs(~i) - 1 : -1;
#else
        (void)full;
        return bit_first_zero_fallback(i);
#endif
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma push
#pragma diag_suppress implicit_return_from_non_void_function
#endif

    FLARE_FORCEINLINE_FUNCTION
    int bit_first_zero(unsigned i) noexcept {
        FLARE_IF_ON_DEVICE((return bit_first_zero_device(i);))
        FLARE_IF_ON_HOST((return bit_first_zero_host(i);))
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma pop
#endif

    FLARE_FORCEINLINE_FUNCTION
    int bit_scan_forward_fallback(unsigned i) {
        int offset = -1;
        if (i) {
            for (offset = 0; (i & (1 << offset)) == 0; ++offset);
        }
        return offset;
    }

    FLARE_IMPL_DEVICE_FUNCTION inline int bit_scan_forward_device(unsigned i) {
#if defined(FLARE_ON_CUDA_DEVICE)
        return __ffs(i) - 1;
#elif defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        return _bit_scan_forward(i);
#else
        return bit_scan_forward_fallback(i);
#endif
    }

    FLARE_IMPL_HOST_FUNCTION inline int bit_scan_forward_host(unsigned i) {
#if defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        return _bit_scan_forward(i);
#elif defined(FLARE_COMPILER_CRAYC)
        return i ? _popcnt(~i & (i - 1)) : -1;
#elif defined(FLARE_COMPILER_GNU) || defined(__GNUC__) || defined(__GNUG__)
        return __builtin_ffs(i) - 1;
#else
        return bit_scan_forward_fallback(i);
#endif
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma push
#pragma diag_suppress implicit_return_from_non_void_function
#endif

    FLARE_FORCEINLINE_FUNCTION
    int bit_scan_forward(unsigned i) {
        FLARE_IF_ON_DEVICE((return bit_scan_forward_device(i);))
        FLARE_IF_ON_HOST((return bit_scan_forward_host(i);))
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma pop
#endif

/// Count the number of bits set.
    FLARE_FORCEINLINE_FUNCTION
    int bit_count_fallback(unsigned i) {
        // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive
        i = i - ((i >> 1) & ~0u / 3u);                           // temp
        i = (i & ~0u / 15u * 3u) + ((i >> 2) & ~0u / 15u * 3u);  // temp
        i = (i + (i >> 4)) & ~0u / 255u * 15u;                   // temp

        // count
        return (int) ((i * (~0u / 255u)) >> (sizeof(unsigned) - 1) * CHAR_BIT);
    }

    FLARE_IMPL_DEVICE_FUNCTION inline int bit_count_device(unsigned i) {
#if defined(FLARE_ON_CUDA_DEVICE)
        return __popc(i);
#elif defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        return _popcnt32(i);
#else
        return bit_count_fallback(i);
#endif
    }

    FLARE_IMPL_HOST_FUNCTION inline int bit_count_host(unsigned i) {
#if defined(FLARE_COMPILER_INTEL) || defined(FLARE_COMPILER_INTEL_LLVM)
        return _popcnt32(i);
#elif defined(FLARE_COMPILER_CRAYC)
        return _popcnt(i);
#elif defined(__GNUC__) || defined(__GNUG__)
        return __builtin_popcount(i);
#else
        return bit_count_fallback(i);
#endif
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma push
#pragma diag_suppress implicit_return_from_non_void_function
#endif

    FLARE_FORCEINLINE_FUNCTION
    int bit_count(unsigned i) {
        FLARE_IF_ON_DEVICE((return bit_count_device(i);))
        FLARE_IF_ON_HOST((return bit_count_host(i);))
    }

#if defined(__EDG__) && !defined(FLARE_COMPILER_INTEL)
#pragma pop
#endif

    FLARE_INLINE_FUNCTION
    unsigned integral_power_of_two_that_contains(const unsigned N) {
        const unsigned i = int_log2(N);
        return ((1u << i) < N) ? i + 1 : i;
    }

}  // namespace flare::detail

#endif  // FLARE_CORE_COMMON_BIT_OPS_H_
