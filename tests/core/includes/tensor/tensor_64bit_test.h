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
#include <doctest.h>

namespace Test {

    template<class Device>
    void test_64bit() {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
        int64_t N = 5000000000;
        int64_t sum = 0;
        {
            flare::parallel_reduce(
                    flare::RangePolicy<typename Device::execution_space,
                            flare::IndexType<int64_t>>(0, N),
                    FLARE_LAMBDA(const int64_t &, int64_t &lsum) { lsum += 1; }, sum);
            REQUIRE_EQ(N, sum);
        }
        {
            flare::Tensor<char *, Device> a("A", N);
            flare::deep_copy(a, char(1));
            flare::parallel_reduce(
                    flare::RangePolicy<typename Device::execution_space,
                            flare::IndexType<int64_t>>(0, N),
                    FLARE_LAMBDA(const int64_t &i, int64_t &lsum) {
                        lsum += int64_t(a(i));
                    },
                    sum);
            REQUIRE_EQ(N, sum);
            flare::parallel_for(
                    flare::RangePolicy<typename Device::execution_space,
                            flare::IndexType<int64_t>>(0, N),
                    FLARE_LAMBDA(const int64_t &i) { a(i) = 3; });
            flare::parallel_reduce(
                    flare::RangePolicy<typename Device::execution_space,
                            flare::IndexType<int64_t>>(0, N),
                    FLARE_LAMBDA(const int64_t &i, int64_t &lsum) {
                        lsum += int64_t(a(i));
                    },
                    sum);
            REQUIRE_EQ(N * 3, sum);
        }
        {
            int64_t N0 = 56925;
            int64_t N1 = 56927;

            flare::Tensor<char **, Device> m("Matrix", N0, N1);
            flare::deep_copy(m, char(1));
            flare::parallel_reduce(
                    flare::RangePolicy<typename Device::execution_space,
                            flare::IndexType<int64_t>>(0, N0 * N1),
                    FLARE_LAMBDA(const int64_t &i, int64_t &lsum) {
                        lsum += int64_t(m(i % N0, i / N0));
                    },
                    sum);
            REQUIRE_EQ(N0 * N1, sum);
            flare::parallel_reduce(
                    flare::MDRangePolicy<typename Device::execution_space, flare::Rank<2>,
                            flare::IndexType<int64_t>>({0, 0}, {N0, N1}),
                    FLARE_LAMBDA(const int64_t &i0, const int64_t &i1, int64_t &lsum) {
                        lsum += int64_t(m(i0, i1));
                    },
                    sum);
            REQUIRE_EQ(N0 * N1, sum);
        }
        {
            int N0 = 1024 * 1024 * 1500;
            int64_t P = 1713091;
            flare::Tensor<int *, Device> a("A", N0);
            flare::parallel_for(
                    "FillA",
                    flare::RangePolicy<typename Device::execution_space,
                            flare::IndexType<int>>(0, N0),
                    FLARE_LAMBDA(const int &i) { a(i) = i % P; });
            int64_t sum0 = 0;
            flare::parallel_reduce(
                    "FillA",
                    flare::RangePolicy<typename Device::execution_space,
                            flare::IndexType<int>>(0, N0),
                    FLARE_LAMBDA(const int &i, int64_t &lsum) { lsum += a(i); }, sum0);
            int64_t expected =
                    (P * (P - 1) / 2) * int64_t(N0 / P) + (N0 % P) * (N0 % P - 1) / 2;
            REQUIRE_EQ(expected, sum0);
        }
#endif
    }

#ifdef FLARE_ENABLE_LARGE_MEM_TESTS
    TEST_CASE("TEST_CATEGORY, view_64bit") { test_64bit<TEST_EXECSPACE>(); }
#endif

}  // namespace Test
