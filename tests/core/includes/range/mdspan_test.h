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

#ifndef MDSPAN_TEST_H_
#define MDSPAN_TEST_H_

#include <flare/core.h>
#include <doctest.h>

namespace {
    void test_mdspan_minimal_functional() {
        int N = 100;
        flare::Tensor<int *, TEST_EXECSPACE> a("A", N);
        flare::parallel_for(
                "FillSequence", flare::RangePolicy<TEST_EXECSPACE>(0, N),
                FLARE_LAMBDA(int i) { a(i) = i; });

        flare::mdspan<int, flare::dextents<int, 1>> a_mds(a.data(), N);
        int errors;
        flare::parallel_reduce(
                "CheckMinimalMDSpan", flare::RangePolicy<TEST_EXECSPACE>(0, N),
                FLARE_LAMBDA(int i, int &err) {
                    flare::mdspan<int, flare::dextents<int, 1>> b_mds(a.data(), N);
#ifdef FLARE_ENABLE_CXX23
                    if (a_mds[i] != i) err++;
                    if (b_mds[i] != i) err++;
#else
                    if (a_mds(i) != i) err++;
                    if (b_mds(i) != i) err++;
#endif
                },
                errors);
        REQUIRE_EQ(errors, 0);
    }
}  // namespace

namespace {

    TEST_CASE("TEST_CATEGORY, mdspan_minimal_functional") {
        test_mdspan_minimal_functional();
    }

}  // namespace

#endif  // MDSPAN_TEST_H_
