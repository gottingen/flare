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

#ifndef FLARE_CORE_COMMON_CHECKED_INTEGER_OPS_H_
#define FLARE_CORE_COMMON_CHECKED_INTEGER_OPS_H_

#include <type_traits>
#include <flare/core/common/error.h>

namespace flare::detail {

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)
#define FLARE_IMPL_USE_MUL_OVERFLOW_BUILTIN
#endif
#endif

    template<typename T>
    std::enable_if_t<std::is_integral_v<T>, bool> constexpr multiply_overflow(
            T a, T b, T &res) {
        static_assert(std::is_unsigned_v<T>,
                      "Operation not implemented for signed integers.");

#if defined(FLARE_IMPL_USE_MUL_OVERFLOW_BUILTIN)
        return __builtin_mul_overflow(a, b, &res);
#else
        auto product = a * b;
        if ((a == 0) || (b == 0) || (a == product / b)) {
            res = product;
            return false;
        } else {
            return true;
        }
#endif
    }

#undef FLARE_IMPL_USE_MUL_OVERFLOW_BUILTIN

    template<typename T>
    T multiply_overflow_abort(T a, T b) {
        T result;
        if (multiply_overflow(a, b, result))
            flare::abort("Arithmetic overflow detected.");

        return result;
    }

}  // namespace flare::detail

#endif  // FLARE_CORE_COMMON_CHECKED_INTEGER_OPS_H_
