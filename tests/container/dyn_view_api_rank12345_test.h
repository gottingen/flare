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

#include <container_dyn_view_api_test.h>

namespace Test {
    TEST_CASE("TEST_CATEGORY, dyn_rank_view_api_operator_rank12345") {
        TestDynViewAPI<double, TEST_EXECSPACE>::run_operator_test_rank12345();
    }

    template<typename SharedMemorySpace>
    void test_dyn_rank_view_resize() {
        int n = 1000;
        flare::DynRankView<double, SharedMemorySpace> device_view("device view", n);
        // Make sure we don't deallocate memory in flare::resize
        auto device_view_copy = device_view;

        flare::resize(device_view, 2 * n);

        // Loop in reverse to increase likelihood of missing fence detection assuming
        // that resize copies values in order.
        for (int i = 2 * n - 1; i >= 0; --i) device_view(i) = i + 1;

        flare::fence();

        // Check that flare::resize completed before setting the values on the host
        // manually (possibly because of missing fences).
        for (int i = 0; i < 2 * n; ++i) REQUIRE_EQ(device_view(i), i + 1);
    }

    template<typename SharedMemorySpace>
    void test_dyn_rank_view_realloc() {
        int n = 1000;
        flare::DynRankView<double, SharedMemorySpace> device_view("device view", n);
        // Make sure we don't deallocate memory in flare::realloc
        auto device_view_copy = device_view;

        flare::realloc(device_view, 2 * n);

        // Loop in reverse to increase likelihood of missing fence detection assuming
        // that realloc sets values in order.
        for (int i = 2 * n - 1; i >= 0; --i) device_view(i) = i + 1;

        flare::fence();

        // Check that flare::realloc completed before setting the values on the host
        // manually (possibly because of missing fences).
        for (int i = 0; i < 2 * n; ++i) REQUIRE_EQ(device_view(i), i + 1);
    }

#ifdef FLARE_HAS_SHARED_SPACE

    TEST_CASE("TEST_CATEGORY, dyn_rank_view_check_fence_resize_realloc") {
        if constexpr (std::is_same_v<TEST_EXECSPACE, flare::DefaultExecutionSpace>) {
            test_dyn_rank_view_resize<flare::SharedSpace>();
            test_dyn_rank_view_realloc<flare::SharedSpace>();
        } else {
            INFO("skipping since not default execution space");
        }
    }

#endif

}  // namespace Test
