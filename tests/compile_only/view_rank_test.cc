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

namespace {

    template<class View, size_t Rank, size_t RankDynamic>
    constexpr bool test_view_rank_and_dynamic_rank() {
        static_assert(View::rank == Rank);
        static_assert(View::rank() == Rank);
        static_assert(View::rank_dynamic == RankDynamic);
        static_assert(View::rank_dynamic() == RankDynamic);
        static_assert(std::is_convertible_v<decltype(View::rank), size_t>);
        static_assert(std::is_same_v<decltype(View::rank()), size_t>);
        static_assert(std::is_convertible_v<decltype(View::rank_dynamic), size_t>);
        static_assert(std::is_same_v<decltype(View::rank_dynamic()), size_t>);
        auto rank = View::rank;  // not an integral type in contrast to flare version
        // less than 4.0.01
        static_assert(!std::is_integral_v<decltype(rank)>);
        auto rank_preferred = View::rank();  // since 4.0.01
        static_assert(std::is_same_v<decltype(rank_preferred), size_t>);
        (void) rank;
        (void) rank_preferred;
        return true;
    }

// clang-format off
    static_assert(test_view_rank_and_dynamic_rank<flare::View<long long>, 0, 0>());

    static_assert(test_view_rank_and_dynamic_rank<flare::View<unsigned[1]>, 1, 0>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<unsigned *>, 1, 1>());

    static_assert(test_view_rank_and_dynamic_rank<flare::View<double[1][2]>, 2, 0>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<double *[2]>, 2, 1>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<double **>, 2, 2>());

    static_assert(test_view_rank_and_dynamic_rank<flare::View<float[1][2][3]>, 3, 0>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<float *[2][3]>, 3, 1>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<float **[3]>, 3, 2>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<float ***>, 3, 3>());

    static_assert(test_view_rank_and_dynamic_rank<flare::View<int[1][2][3][4]>, 4, 0>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<int *[2][3][4]>, 4, 1>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<int **[3][4]>, 4, 2>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<int ***[4]>, 4, 3>());
    static_assert(test_view_rank_and_dynamic_rank<flare::View<int ****>, 4, 4>());
//clang-format on

}  // namespace
