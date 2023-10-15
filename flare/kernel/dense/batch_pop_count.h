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

#ifndef FLARE_KERNEL_DENSE_BATCH_POP_COUNT_H_
#define FLARE_KERNEL_DENSE_BATCH_POP_COUNT_H_

/// temp here, todo is move  to simd api group
#include <flare/core.h>

namespace flare::kernel::dense::detail {
    template<typename T, size_t N>
    struct batch_pop_count {
        FLARE_FORCEINLINE_FUNCTION static int pop_count(const flare::simd::batch<T, flare::simd::default_arch> &b);
    };

    template<typename T>
    struct batch_pop_count<T, 4> {
        static_assert(sizeof(T) == 8, "T must be 8 bytes");

        FLARE_FORCEINLINE_FUNCTION static int pop_count(const flare::simd::batch<T, flare::simd::default_arch> &b) {
            return flare::experimental::popcount_builtin(b.get(0)) +
                   flare::experimental::popcount_builtin(b.get(1)) +
                   flare::experimental::popcount_builtin(b.get(2)) +
                   flare::experimental::popcount_builtin(b.get(3));

        }
    };

    template<typename T>
    struct batch_pop_count<T, 8> {
        static_assert(sizeof(T) == 8, "T must be 8 bytes");

        FLARE_FORCEINLINE_FUNCTION static int pop_count(const flare::simd::batch<T, flare::simd::default_arch> &b) {
            return flare::experimental::popcount_builtin(b.get(0)) +
                   flare::experimental::popcount_builtin(b.get(1)) +
                   flare::experimental::popcount_builtin(b.get(2)) +
                   flare::experimental::popcount_builtin(b.get(3)) +
                   flare::experimental::popcount_builtin(b.get(4)) +
                   flare::experimental::popcount_builtin(b.get(5)) +
                   flare::experimental::popcount_builtin(b.get(6)) +
                   flare::experimental::popcount_builtin(b.get(7));
        }
    };

    template<typename T>
    struct batch_pop_count<T, 16> {
        static_assert(sizeof(T) == 8, "T must be 8 bytes");

        FLARE_FORCEINLINE_FUNCTION static int pop_count(const flare::simd::batch<T, flare::simd::default_arch> &b) {
            return flare::experimental::popcount_builtin(b.get(0)) +
                   flare::experimental::popcount_builtin(b.get(1)) +
                   flare::experimental::popcount_builtin(b.get(2)) +
                   flare::experimental::popcount_builtin(b.get(3)) +
                   flare::experimental::popcount_builtin(b.get(4)) +
                   flare::experimental::popcount_builtin(b.get(5)) +
                   flare::experimental::popcount_builtin(b.get(6)) +
                   flare::experimental::popcount_builtin(b.get(7)) +
                   flare::experimental::popcount_builtin(b.get(8)) +
                   flare::experimental::popcount_builtin(b.get(9)) +
                   flare::experimental::popcount_builtin(b.get(10)) +
                   flare::experimental::popcount_builtin(b.get(11)) +
                   flare::experimental::popcount_builtin(b.get(12)) +
                   flare::experimental::popcount_builtin(b.get(13)) +
                   flare::experimental::popcount_builtin(b.get(14)) +
                   flare::experimental::popcount_builtin(b.get(15));
        }
    };

}  // namespace flare::kernel::dense::detail
#endif  // FLARE_KERNEL_DENSE_BATCH_POP_COUNT_H_
