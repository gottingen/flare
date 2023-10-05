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

//#include <gtest/gtest.h>

#include <flare/core.h>
#include <doctest.h>

namespace Test {

    template<typename View>
    struct SumView {
        const View m_view;
        FLARE_FUNCTION void operator()(const int i, const int j, int &update) const {
            update += m_view(i, j);
        }

        SumView(View view) : m_view(view) {}

        int run() {
            int sum_view = 0;
            flare::parallel_reduce(
                    flare::MDRangePolicy<typename View::execution_space, flare::Rank<2>>(
                            {0, 0}, {m_view.extent(0), m_view.extent(1)}),
                    *this, sum_view);
            return sum_view;
        }
    };

    template<typename ExecutionSpace>
    struct TestMDRangeLargeDeepCopy {
        static void run() {
            ExecutionSpace exec;
            using MemorySpace = typename ExecutionSpace::memory_space;
            const int s = 45;
            const int step_sizes[2] = {1, 10000};
            flare::View<int **, MemorySpace> view("v", s * step_sizes[0],
                                                  (s + 1) * step_sizes[1]);
            flare::deep_copy(exec, view, 1);
            for (int step = 2; step < view.extent_int(0); ++step) {
                auto subview =
                        flare::subview(view, std::make_pair(0, (step + 1) * step_sizes[0]),
                                       std::make_pair(0, (step + 2) * step_sizes[1]));
                flare::View<int **, MemorySpace> subview_copy(
                        "subview_copy", subview.extent(0), subview.extent(1));
                flare::deep_copy(TEST_EXECSPACE{}, subview_copy, subview);
                exec.fence();

                SumView<decltype(subview)> sum_subview(subview);
                int total_subview = sum_subview.run();
                SumView<decltype(subview_copy)> sum_subview_copy(subview_copy);
                int total_subview_copy = sum_subview_copy.run();

                REQUIRE_EQ(total_subview, total_subview_copy);
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
