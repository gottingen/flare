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

#include <flare/core.h>

#include <type_traits>

namespace {

    constexpr bool test_view_rank() {
        // clang-format off
        static_assert(flare::View<int>::rank == 0);
        static_assert(flare::View<int[1]>::rank == 1);
        static_assert(flare::View<int *>::rank == 1);
        static_assert(flare::View<int[1][2]>::rank == 2);
        static_assert(flare::View<int *[1]>::rank == 2);
        static_assert(flare::View<int **>::rank == 2);
        static_assert(flare::View<int[1][2][3]>::rank == 3);
        static_assert(flare::View<int *[1][2]>::rank == 3);
        static_assert(flare::View<int **[1]>::rank == 3);
        static_assert(flare::View<int ***>::rank == 3);
        // clang-format on
        return true;
    }

    static_assert(test_view_rank());

    constexpr bool test_is_view_type_trait() {
        struct NotView {
        };
        static_assert(flare::is_view<flare::View<int>>::value);
        static_assert(flare::is_view_v<flare::View<int>>);
        static_assert(!flare::is_view_v<NotView>);
        static_assert(!flare::is_view<NotView>::value);
        return true;
    }

    static_assert(test_is_view_type_trait());

}  // namespace
