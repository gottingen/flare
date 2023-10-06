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

#include <flare/core/defines.h>

#if defined(FLARE_CORE_ATOMIC_H_) && !defined(FLARE_COMMON_VOLATILE_LOAD_H_)
#define FLARE_COMMON_VOLATILE_LOAD_H_

#if defined(__GNUC__) /* GNU C   */ || defined(__GNUG__) /* GNU C++ */ || \
    defined(__clang__)

#define FLARE_IMPL_MAY_ALIAS __attribute__((__may_alias__))

#else

#define FLARE_IMPL_MAY_ALIAS

#endif

namespace flare {

    template<typename T>
    FLARE_FORCEINLINE_FUNCTION T volatile_load(T const volatile *const src_ptr) {
        typedef uint64_t FLARE_IMPL_MAY_ALIAS T64;  // NOLINT(modernize-use-using)
        typedef uint32_t FLARE_IMPL_MAY_ALIAS T32;  // NOLINT(modernize-use-using)
        typedef uint16_t FLARE_IMPL_MAY_ALIAS T16;  // NOLINT(modernize-use-using)
        typedef uint8_t FLARE_IMPL_MAY_ALIAS T8;    // NOLINT(modernize-use-using)

        enum {
            NUM_8 = sizeof(T),
            NUM_16 = NUM_8 / 2,
            NUM_32 = NUM_8 / 4,
            NUM_64 = NUM_8 / 8
        };

        union {
            T const volatile *const ptr;
            T64 const volatile *const ptr64;
            T32 const volatile *const ptr32;
            T16 const volatile *const ptr16;
            T8 const volatile *const ptr8;
        } src = {src_ptr};

        T result;

        union {
            T *const ptr;
            T64 *const ptr64;
            T32 *const ptr32;
            T16 *const ptr16;
            T8 *const ptr8;
        } dst = {&result};

        for (int i = 0; i < NUM_64; ++i) {
            dst.ptr64[i] = src.ptr64[i];
        }

        if (NUM_64 * 2 < NUM_32) {
            dst.ptr32[NUM_64 * 2] = src.ptr32[NUM_64 * 2];
        }

        if (NUM_32 * 2 < NUM_16) {
            dst.ptr16[NUM_32 * 2] = src.ptr16[NUM_32 * 2];
        }

        if (NUM_16 * 2 < NUM_8) {
            dst.ptr8[NUM_16 * 2] = src.ptr8[NUM_16 * 2];
        }

        return result;
    }

    template<typename T>
    FLARE_FORCEINLINE_FUNCTION void volatile_store(
            T volatile *const dst_ptr, T const volatile *const src_ptr) {
        typedef uint64_t FLARE_IMPL_MAY_ALIAS T64;  // NOLINT(modernize-use-using)
        typedef uint32_t FLARE_IMPL_MAY_ALIAS T32;  // NOLINT(modernize-use-using)
        typedef uint16_t FLARE_IMPL_MAY_ALIAS T16;  // NOLINT(modernize-use-using)
        typedef uint8_t FLARE_IMPL_MAY_ALIAS T8;    // NOLINT(modernize-use-using)

        enum {
            NUM_8 = sizeof(T),
            NUM_16 = NUM_8 / 2,
            NUM_32 = NUM_8 / 4,
            NUM_64 = NUM_8 / 8
        };

        union {
            T const volatile *const ptr;
            T64 const volatile *const ptr64;
            T32 const volatile *const ptr32;
            T16 const volatile *const ptr16;
            T8 const volatile *const ptr8;
        } src = {src_ptr};

        union {
            T volatile *const ptr;
            T64 volatile *const ptr64;
            T32 volatile *const ptr32;
            T16 volatile *const ptr16;
            T8 volatile *const ptr8;
        } dst = {dst_ptr};

        for (int i = 0; i < NUM_64; ++i) {
            dst.ptr64[i] = src.ptr64[i];
        }

        if (NUM_64 * 2 < NUM_32) {
            dst.ptr32[NUM_64 * 2] = src.ptr32[NUM_64 * 2];
        }

        if (NUM_32 * 2 < NUM_16) {
            dst.ptr16[NUM_32 * 2] = src.ptr16[NUM_32 * 2];
        }

        if (NUM_16 * 2 < NUM_8) {
            dst.ptr8[NUM_16 * 2] = src.ptr8[NUM_16 * 2];
        }
    }

    template<typename T>
    FLARE_FORCEINLINE_FUNCTION void volatile_store(T volatile *const dst_ptr,
                                                   T const *const src_ptr) {
        typedef uint64_t FLARE_IMPL_MAY_ALIAS T64;  // NOLINT(modernize-use-using)
        typedef uint32_t FLARE_IMPL_MAY_ALIAS T32;  // NOLINT(modernize-use-using)
        typedef uint16_t FLARE_IMPL_MAY_ALIAS T16;  // NOLINT(modernize-use-using)
        typedef uint8_t FLARE_IMPL_MAY_ALIAS T8;    // NOLINT(modernize-use-using)

        enum {
            NUM_8 = sizeof(T),
            NUM_16 = NUM_8 / 2,
            NUM_32 = NUM_8 / 4,
            NUM_64 = NUM_8 / 8
        };

        union {
            T const *const ptr;
            T64 const *const ptr64;
            T32 const *const ptr32;
            T16 const *const ptr16;
            T8 const *const ptr8;
        } src = {src_ptr};

        union {
            T volatile *const ptr;
            T64 volatile *const ptr64;
            T32 volatile *const ptr32;
            T16 volatile *const ptr16;
            T8 volatile *const ptr8;
        } dst = {dst_ptr};

        for (int i = 0; i < NUM_64; ++i) {
            dst.ptr64[i] = src.ptr64[i];
        }

        if (NUM_64 * 2 < NUM_32) {
            dst.ptr32[NUM_64 * 2] = src.ptr32[NUM_64 * 2];
        }

        if (NUM_32 * 2 < NUM_16) {
            dst.ptr16[NUM_32 * 2] = src.ptr16[NUM_32 * 2];
        }

        if (NUM_16 * 2 < NUM_8) {
            dst.ptr8[NUM_16 * 2] = src.ptr8[NUM_16 * 2];
        }
    }

    template<typename T>
    FLARE_FORCEINLINE_FUNCTION void volatile_store(T volatile *dst_ptr,
                                                   T const volatile &src) {
        volatile_store(dst_ptr, &src);
    }

    template<typename T>
    FLARE_FORCEINLINE_FUNCTION void volatile_store(T volatile *dst_ptr,
                                                   T const &src) {
        volatile_store(dst_ptr, &src);
    }

    template<typename T>
    FLARE_FORCEINLINE_FUNCTION T safe_load(T const *const ptr) {
#if !defined(__MIC__)
        return *ptr;
#else
        return volatile_load(ptr);
#endif
    }

}  // namespace flare

#undef FLARE_IMPL_MAY_ALIAS

#endif  // FLARE_COMMON_VOLATILE_LOAD_H_
