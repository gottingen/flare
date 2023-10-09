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
#ifndef FLARE_SIMD_TYPES_SSE4_2_REGISTER_H_
#define FLARE_SIMD_TYPES_SSE4_2_REGISTER_H_

#include <flare/simd/types/sse4_1_register.h>

#if FLARE_SIMD_ENABLE_SSE4_2
#include <nmmintrin.h>
#endif

namespace flare::simd {
    /**
     * @ingroup architectures
     *
     * SSE4.2 instructions
     */
    struct sse4_2 : sse4_1 {
        static constexpr bool supported() noexcept { return FLARE_SIMD_ENABLE_SSE4_2; }
        static constexpr bool available() noexcept { return true; }
        static constexpr unsigned version() noexcept { return generic::version(1, 4, 2); }
        static constexpr char const* name() noexcept { return "sse4.2"; }
    };

#if FLARE_SIMD_ENABLE_SSE4_2
    namespace types {
        FLARE_SIMD_DECLARE_SIMD_REGISTER_ALIAS(sse4_2, sse4_1);
    }  // namespace types
#endif
}

#endif  // FLARE_SIMD_TYPES_SSE4_2_REGISTER_H_
