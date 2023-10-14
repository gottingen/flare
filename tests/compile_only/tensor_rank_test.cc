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

    template<class Tensor, size_t Rank, size_t RankDynamic>
    constexpr bool test_tensor_rank_and_dynamic_rank() {
        static_assert(Tensor::rank == Rank);
        static_assert(Tensor::rank() == Rank);
        static_assert(Tensor::rank_dynamic == RankDynamic);
        static_assert(Tensor::rank_dynamic() == RankDynamic);
        static_assert(std::is_convertible_v<decltype(Tensor::rank), size_t>);
        static_assert(std::is_same_v<decltype(Tensor::rank()), size_t>);
        static_assert(std::is_convertible_v<decltype(Tensor::rank_dynamic), size_t>);
        static_assert(std::is_same_v<decltype(Tensor::rank_dynamic()), size_t>);
        auto rank = Tensor::rank;  // not an integral type in contrast to flare version
        // less than 4.0.01
        static_assert(!std::is_integral_v<decltype(rank)>);
        auto rank_preferred = Tensor::rank();  // since 4.0.01
        static_assert(std::is_same_v<decltype(rank_preferred), size_t>);
        (void) rank;
        (void) rank_preferred;
        return true;
    }

// clang-format off
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<long long>, 0, 0>());

    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<unsigned[1]>, 1, 0>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<unsigned *>, 1, 1>());

    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<double[1][2]>, 2, 0>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<double *[2]>, 2, 1>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<double **>, 2, 2>());

    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<float[1][2][3]>, 3, 0>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<float *[2][3]>, 3, 1>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<float **[3]>, 3, 2>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<float ***>, 3, 3>());

    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<int[1][2][3][4]>, 4, 0>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<int *[2][3][4]>, 4, 1>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<int **[3][4]>, 4, 2>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<int ***[4]>, 4, 3>());
    static_assert(test_tensor_rank_and_dynamic_rank<flare::Tensor<int ****>, 4, 4>());
//clang-format on

}  // namespace
