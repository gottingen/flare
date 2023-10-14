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

#include <cuda_category_test.h>
#include "inter_op_streams_test.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

namespace Test {
    // Test Interoperability with Cuda Streams
    TEST_CASE("cuda, raw_cuda_streams") {
        // Make sure that we use the same device for all allocations
        flare::initialize();

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        int *p;
        cudaMalloc(&p, sizeof(int) * 100);
        using MemorySpace = typename TEST_EXECSPACE::memory_space;

        {
            TEST_EXECSPACE space0(stream);
            flare::Tensor<int *, TEST_EXECSPACE> v(p, 100);
            flare::deep_copy(space0, v, 5);
            int sum;

            flare::parallel_for("Test::cuda::raw_cuda_stream::Range",
                                flare::RangePolicy<TEST_EXECSPACE>(space0, 0, 100),
                                FunctorRange<MemorySpace>(v));
            flare::parallel_reduce(
                    "Test::cuda::raw_cuda_stream::RangeReduce",
                    flare::RangePolicy<TEST_EXECSPACE, flare::LaunchBounds<128, 2>>(
                            space0, 0, 100),
                    FunctorRangeReduce<MemorySpace>(v), sum);
            space0.fence();
            REQUIRE_EQ(600, sum);

            flare::parallel_for("Test::cuda::raw_cuda_stream::MDRange",
                                flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>>(
                                        space0, {0, 0}, {10, 10}),
                                FunctorMDRange<MemorySpace>(v));
            flare::parallel_reduce(
                    "Test::cuda::raw_cuda_stream::MDRangeReduce",
                    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>,
                            flare::LaunchBounds<128, 2>>(space0, {0, 0},
                                                         {10, 10}),
                    FunctorMDRangeReduce<MemorySpace>(v), sum);
            space0.fence();
            REQUIRE_EQ(700, sum);

            flare::parallel_for("Test::cuda::raw_cuda_stream::Team",
                                flare::TeamPolicy<TEST_EXECSPACE>(space0, 10, 10),
                                FunctorTeam<MemorySpace, TEST_EXECSPACE>(v));
            flare::parallel_reduce(
                    "Test::cuda::raw_cuda_stream::Team",
                    flare::TeamPolicy<TEST_EXECSPACE, flare::LaunchBounds<128, 2>>(
                            space0, 10, 10),
                    FunctorTeamReduce<MemorySpace, TEST_EXECSPACE>(v), sum);
            space0.fence();
            REQUIRE_EQ(800, sum);
        }
        flare::finalize();
        offset_streams<<<100, 64, 0, stream>>>(p);
        FLARE_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());
        cudaStreamDestroy(stream);

        int h_p[100];
        cudaMemcpy(h_p, p, sizeof(int) * 100, cudaMemcpyDefault);
        FLARE_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());
        int64_t sum = 0;
        int64_t sum_expect = 0;
        for (int i = 0; i < 100; i++) {
            sum += h_p[i];
            sum_expect += 8 + i;
        }

        REQUIRE_EQ(sum, sum_expect);
    }
}  // namespace Test
