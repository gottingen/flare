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

#include <flare/core/array.h>

namespace {

    FLARE_FUNCTION constexpr bool test_array() {
        constexpr flare::Array<int, 3> a{{1, 2}};

        static_assert(!a.empty());
        static_assert(a.size() == 3);
        static_assert(a.max_size() == 3);

        static_assert(*a.data() == 1);
        static_assert(a[1] == 2);

        return true;
    }

    static_assert(test_array());

    FLARE_FUNCTION constexpr bool test_array_structured_binding_support() {
        constexpr flare::Array<float, 2> a{};
        auto &[xr, yr] = a;
        (void) xr;
        (void) yr;
        auto [x, y] = a;
        (void) x;
        (void) y;
        auto const &[xcr, ycr] = a;
        (void) xcr;
        (void) ycr;
        return true;
    }

    static_assert(test_array_structured_binding_support());

}  // namespace
