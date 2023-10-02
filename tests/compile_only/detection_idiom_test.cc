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

#include <flare/core/common/detection_idiom.h>

void test_nonesuch() {
    using flare::nonesuch;
    static_assert(!std::is_constructible<nonesuch>::value);
    static_assert(!std::is_destructible<nonesuch>::value);
    static_assert(!std::is_copy_constructible<nonesuch>::value);
    static_assert(!std::is_move_constructible<nonesuch>::value);
    static_assert(!std::is_aggregate<nonesuch>::value);
}

namespace Example {
// Example from https://en.cppreference.com/w/cpp/experimental/is_detected
    template<class T>
    using copy_assign_t = decltype(std::declval<T &>() = std::declval<const T &>());

    struct Meow {
    };

    struct Purr {
        void operator=(const Purr &) = delete;
    };

    static_assert(flare::is_detected<copy_assign_t, Meow>::value,
                  "Meow should be copy assignable!");
    static_assert(!flare::is_detected<copy_assign_t, Purr>::value,
                  "Purr should not be copy assignable!");
    static_assert(flare::is_detected_exact<Meow &, copy_assign_t, Meow>::value,
                  "Copy assignment of Meow should return Meow&!");

    template<class T>
    using diff_t = typename T::difference_type;

    template<class Ptr>
    using difference_type = flare::detected_or_t<std::ptrdiff_t, diff_t, Ptr>;

    struct Woof {
        using difference_type = int;
    };
    struct Bark {
    };

    static_assert(std::is_same<difference_type<Woof>, int>::value,
                  "Woof's difference_type should be int!");
    static_assert(std::is_same<difference_type<Bark>, std::ptrdiff_t>::value,
                  "Bark's difference_type should be ptrdiff_t!");
}  // namespace Example
