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

template<class View, class ExecutionSpace>
struct TestViewMemoryAccessViolation {
    View v;
    static constexpr auto rank = View::rank;

    template<std::size_t... Is>
    FLARE_FUNCTION decltype(auto) bad_access(std::index_sequence<Is...>) const {
        return v((Is * 0)...);
    }

    FLARE_FUNCTION void operator()(int) const {
        ++bad_access(std::make_index_sequence<rank>{});
    }

    TestViewMemoryAccessViolation(View w, ExecutionSpace const &s,
                                  std::string const &matcher)
            : v(std::move(w)) {
        constexpr bool view_accessible_from_execution_space =
                flare::SpaceAccessibility<
                        /*AccessSpace=*/ExecutionSpace,
                        /*MemorySpace=*/typename View::memory_space>::accessible;
        REQUIRE_FALSE(view_accessible_from_execution_space);
        EXPECT_DEATH(
                {
                        flare::parallel_for(flare::RangePolicy<ExecutionSpace>(s, 0, 1),
                                            *this);
                flare::fence();
                },
                matcher);
    }
};

template<class View, class ExecutionSpace>
void test_view_memory_access_violation(View v, ExecutionSpace const &s,
                                       std::string const &m) {
    TestViewMemoryAccessViolation<View, ExecutionSpace>(std::move(v), s, m);
}

template<class View, class LblOrPtr, std::size_t... Is>
auto make_view_impl(LblOrPtr x, std::index_sequence<Is...>) {
    return View(x, (Is + 1)...);
}

template<class View, class LblOrPtr>
auto make_view(LblOrPtr x) {
    return make_view_impl<View>(std::move(x),
                                std::make_index_sequence<View::rank>());
}

template<class ExecutionSpace>
void test_view_memory_access_violations_from_host() {
    flare::DefaultHostExecutionSpace const host_exec_space{};
    // clang-format off
    using V0 = flare::View<int, ExecutionSpace>;
    using V1 = flare::View<int *, ExecutionSpace>;
    using V2 = flare::View<int **, ExecutionSpace>;
    using V3 = flare::View<int ***, ExecutionSpace>;
    using V4 = flare::View<int ****, ExecutionSpace>;
    using V5 = flare::View<int *****, ExecutionSpace>;
    using V6 = flare::View<int ******, ExecutionSpace>;
    using V7 = flare::View<int *******, ExecutionSpace>;
    using V8 = flare::View<int ********, ExecutionSpace>;
    std::string const prefix = "flare::View ERROR: attempt to access inaccessible memory space";
    std::string const lbl = "my_label";
    test_view_memory_access_violation(make_view<V0>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V1>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V2>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V3>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V4>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V5>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V6>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V7>(lbl), host_exec_space, prefix + ".*" + lbl);
    test_view_memory_access_violation(make_view<V8>(lbl), host_exec_space, prefix + ".*" + lbl);
    int *const ptr = nullptr;
    test_view_memory_access_violation(make_view<V0>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V1>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V2>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V3>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V4>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V5>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V6>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V7>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    test_view_memory_access_violation(make_view<V8>(ptr), host_exec_space, prefix + ".*UNMANAGED");
    // clang-format on
}

template<class ExecutionSpace>
void test_view_memory_access_violations_from_device() {
    ExecutionSpace const exec_space{};
    // clang-format off
    using V0 = flare::View<int, flare::HostSpace>;
    using V1 = flare::View<int *, flare::HostSpace>;
    using V2 = flare::View<int **, flare::HostSpace>;
    using V3 = flare::View<int ***, flare::HostSpace>;
    using V4 = flare::View<int ****, flare::HostSpace>;
    using V5 = flare::View<int *****, flare::HostSpace>;
    using V6 = flare::View<int ******, flare::HostSpace>;
    using V7 = flare::View<int *******, flare::HostSpace>;
    using V8 = flare::View<int ********, flare::HostSpace>;
    std::string const prefix = "flare::View ERROR: attempt to access inaccessible memory space";
    std::string const lbl = "my_label";
    test_view_memory_access_violation(make_view<V0>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V1>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V2>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V3>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V4>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V5>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V6>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V7>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V8>(lbl), exec_space, prefix + ".*UNAVAILABLE");
    int *const ptr = nullptr;
    test_view_memory_access_violation(make_view<V0>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V1>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V2>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V3>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V4>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V5>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V6>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V7>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    test_view_memory_access_violation(make_view<V8>(ptr), exec_space, prefix + ".*UNAVAILABLE");
    // clang-format on
}
/*
TEST_CASE("TEST_CATEGORY_DEATH, view_memory_access_violations_from_host") {
::testing::FLAGS_gtest_death_test_style = "threadsafe";

using ExecutionSpace = TEST_EXECSPACE;

if (flare::SpaceAccessibility<flare::HostSpace, typename ExecutionSpace::memory_space>::accessible) {
GTEST_SKIP()

<< "skipping since no memory access violation would occur";
}

test_view_memory_access_violations_from_host<ExecutionSpace>();

}

TEST_CASE("TEST_CATEGORY_DEATH, view_memory_access_violations_from_device") {
::testing::FLAGS_gtest_death_test_style = "threadsafe";

using ExecutionSpace = TEST_EXECSPACE;

if (flare::SpaceAccessibility<ExecutionSpace, flare::HostSpace>::accessible) {
GTEST_SKIP()

<< "skipping since no memory access violation would occur";
}

test_view_memory_access_violations_from_device<ExecutionSpace>();

}
*/
