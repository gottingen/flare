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
//
// Created by jeff on 23-10-17.
//

#ifndef FLARE_CORE_MATHS_POW_H_
#define FLARE_CORE_MATHS_POW_H_

#include <type_traits>

namespace flare::detail {


    // rounds the given 64-bit unsigned integer to the nearest power of 2
    template<typename T, std::enable_if_t<
            (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 8), void
    > * = nullptr>
    constexpr T next_pow2(T x) {
        if (x == 0) return 1;
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x |= x >> 32;
        x++;
        return x;
    }

// rounds the given 32-bit unsigned integer to the nearest power of 2
    template<typename T, std::enable_if_t<
            (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 4), void
    > * = nullptr>
    constexpr T next_pow2(T x) {
        if (x == 0) return 1;
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x++;
        return x;
    }

    // checks if the given number if a power of 2
    template<typename T, std::enable_if_t<
            std::is_integral_v<std::decay_t<T>>, void> * = nullptr
    >
    constexpr bool is_pow2(const T &x) {
        return x && (!(x & (x - 1)));
    }

    //// finds the ceil of x divided by b
    //template <typename T, std::enable_if_t<
    //  std::is_integral_v<std::decay_t<T>>, void>* = nullptr
    //>
    //constexpr T ceil(const T& x, const T& y) {
    //  //return (x + y - 1) / y;
    //  return (x-1) / y + 1;
    //}

    /**
    @brief returns floor(log2(n)), assumes n > 0
    */
    template<typename T>
    constexpr int log2(T n) {
        int log = 0;
        while (n >>= 1) {
            ++log;
        }
        return log;
    }


}  // namespace flare::detail

#endif  // FLARE_CORE_MATHS_POW_H_
