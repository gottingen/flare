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

    template<typename Tensor>
    struct SumTensor {
        const Tensor m_tensor;
        FLARE_FUNCTION void operator()(const int i, const int j, int &update) const {
            update += m_tensor(i, j);
        }

        SumTensor(Tensor tensor) : m_tensor(tensor) {}

        int run() {
            int sum_tensor = 0;
            flare::parallel_reduce(
                    flare::MDRangePolicy<typename Tensor::execution_space, flare::Rank<2>>(
                            {0, 0}, {m_tensor.extent(0), m_tensor.extent(1)}),
                    *this, sum_tensor);
            return sum_tensor;
        }
    };

    template<typename ExecutionSpace>
    struct TestMDRangeLargeDeepCopy {
        static void run() {
            ExecutionSpace exec;
            using MemorySpace = typename ExecutionSpace::memory_space;
            const int s = 45;
            const int step_sizes[2] = {1, 10000};
            flare::Tensor<int **, MemorySpace> tensor("v", s * step_sizes[0],
                                                  (s + 1) * step_sizes[1]);
            flare::deep_copy(exec, tensor, 1);
            for (int step = 2; step < tensor.extent_int(0); ++step) {
                auto subtensor =
                        flare::subtensor(tensor, std::make_pair(0, (step + 1) * step_sizes[0]),
                                       std::make_pair(0, (step + 2) * step_sizes[1]));
                flare::Tensor<int **, MemorySpace> subtensor_copy(
                        "subtensor_copy", subtensor.extent(0), subtensor.extent(1));
                flare::deep_copy(TEST_EXECSPACE{}, subtensor_copy, subtensor);
                exec.fence();

                SumTensor<decltype(subtensor)> sum_subtensor(subtensor);
                int total_subtensor = sum_subtensor.run();
                SumTensor<decltype(subtensor_copy)> sum_subtensor_copy(subtensor_copy);
                int total_subtensor_copy = sum_subtensor_copy.run();

                REQUIRE_EQ(total_subtensor, total_subtensor_copy);
            }
        }
    };

    // Check that deep_copy with a large range for a dimension different from the
    // first one works successfully. There was a problem with this in the Cuda
    // backend.
    TEST_CASE("TEST_CATEGORY, mdrange_large_deep_copy") {
        TestMDRangeLargeDeepCopy<TEST_EXECSPACE>::run();
    }

}  // namespace Test
