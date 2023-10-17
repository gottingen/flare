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
#ifndef FLARE_RUNTIME_UILITY_ITERATOR_H_
#define FLARE_RUNTIME_UILITY_ITERATOR_H_

#include <cstddef>
#include <type_traits>

namespace flare::rt {

    template<typename T>
    constexpr std::enable_if_t<std::is_integral<std::decay_t<T>>::value, bool>
    is_range_invalid(T beg, T end, T step) {
        return ((step == 0 && beg != end) ||
                (beg < end && step <= 0) ||
                (beg > end && step >= 0));
    }

    template<typename T>
    constexpr std::enable_if_t<std::is_integral<std::decay_t<T>>::value, size_t>
    distance(T beg, T end, T step) {
        return (end - beg + step + (step > 0 ? -1 : 1)) / step;
    }

}  // namespace flare::rt
#endif  // FLARE_RUNTIME_UILITY_ITERATOR_H_
