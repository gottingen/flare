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
#include <cuda_category_test.h>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <array>

namespace Test {

    __global__ void offset(int *p) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < 100) {
            p[idx] += idx;
        }
    }

    // Test whether allocations survive flare initialize/finalize if done via Raw
    // Cuda.
    TEST_CASE("cuda, raw_cuda_interop") {
        // Make sure that we use the same device for all allocations
        flare::initialize();

        int *p;
        FLARE_IMPL_CUDA_SAFE_CALL(cudaMalloc(&p, sizeof(int) * 100));

        flare::Tensor<int *, flare::MemoryTraits<flare::Unmanaged>> v(p, 100);
        flare::deep_copy(v, 5);

        flare::finalize();

        offset<<<100, 64>>>(p);
        FLARE_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());

        std::array<int, 100> h_p;
        cudaMemcpy(h_p.data(), p, sizeof(int) * 100, cudaMemcpyDefault);
        FLARE_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());
        int64_t sum = 0;
        int64_t sum_expect = 0;
        for (int i = 0; i < 100; i++) {
            sum += h_p[i];
            sum_expect += 5 + i;
        }

        REQUIRE_EQ(sum, sum_expect);
        FLARE_IMPL_CUDA_SAFE_CALL(cudaFree(p));
    }
}  // namespace Test
