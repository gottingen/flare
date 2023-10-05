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

#include <doctest.h>
#include <flare/core.h>

namespace {

    template<typename T>
    void MDRangeReduceTester([[maybe_unused]] int bound, int k) {
        const auto policy_MD = flare::MDRangePolicy<flare::Rank<2>, TEST_EXECSPACE>(
                {0, 0}, {bound, 2});

        // No explicit fence() calls needed because result is in HostSpace
        {
            T lor_MD = 0;
            flare::parallel_reduce(
                    policy_MD,
                    FLARE_LAMBDA(const int i, const int, T &res) { res = res || i == k; },
                    flare::LOr<T>(lor_MD));
            REQUIRE_EQ(lor_MD, 1);
        }
        {
            // Stick just a few true values in the Logical-OR reduction space,
            // to try to make sure every value is being captured
            T land_MD = 0;
            flare::parallel_reduce(
                    policy_MD, FLARE_LAMBDA(const int, const int, T &res) { res = 1; },
                    flare::LAnd<T>(land_MD));
            REQUIRE_EQ(land_MD, 1);
        }
    }

    TEST_CASE("TEST_CATEGORY, mdrange_parallel_reduce_primitive_types") {
        for (int bound: {0, 1, 7, 32, 65, 7000}) {
            for (int k = 0; k < bound; ++k) {
                MDRangeReduceTester<bool>(bound, k);
                MDRangeReduceTester<signed char>(bound, k);
                MDRangeReduceTester<int8_t>(bound, k);
                MDRangeReduceTester<int16_t>(bound, k);
                MDRangeReduceTester<int32_t>(bound, k);
                MDRangeReduceTester<int64_t>(bound, k);
            }
        }
    }

}  // namespace
