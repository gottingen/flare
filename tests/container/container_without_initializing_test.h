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

#include <doctest.h>
#include <flare/core.h>
#include <flare/dual_view.h>
#include <flare/dynamic_view.h>
#include <flare/dyn_rank_view.h>
#include <flare/offset_view.h>
#include <flare/scatter_view.h>

#include <tool/tool_testing_utilities.h>

/// Some tests are skipped for @c CudaUVM memory space.
///@{
#ifdef FLARE_ON_CUDA_DEVICE
#define DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE                            \
  if constexpr (std::is_same_v<typename TEST_EXECSPACE::memory_space, \
                               flare::CudaUVMSpace>) {                \
        INFO( "skipping since CudaUVMSpace requires additional fences");    \
        return;                                                         \
    }
#else
#define DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE
#endif
///@}

TEST_CASE("TEST_CATEGORY, resize_realloc_no_init_dualview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DualView<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 5, 6, 7,
                                                              8);

    auto success = validate_absence(
            [&]() {
                flare::resize(flare::WithoutInitializing, bla, 5, 6, 7, 9);
                REQUIRE_EQ(bla.template view<TEST_EXECSPACE>().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 8, 8, 8);
                REQUIRE_EQ(bla.template view<TEST_EXECSPACE>().label(), "bla");
                flare::realloc(flare::view_alloc(flare::WithoutInitializing), bla, 5,
                               6, 7, 8);
                REQUIRE_EQ(bla.template view<TEST_EXECSPACE>().label(), "bla");
            },
            [&](BeginParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_realloc_no_alloc_dualview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableAllocs());
    flare::DualView<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6,
                                                              5);

    auto success = validate_absence(
            [&]() {
                flare::resize(bla, 8, 7, 6, 5);
                REQUIRE_EQ(bla.template view<TEST_EXECSPACE>().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 7, 6, 5);
                REQUIRE_EQ(bla.template view<TEST_EXECSPACE>().label(), "bla");
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            },
            [&](AllocateDataEvent) {
                return MatchDiagnostic{true, {"Found alloc event"}};
            },
            [&](DeallocateDataEvent) {
                return MatchDiagnostic{true, {"Found dealloc event"}};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_exec_space_dualview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                       Config::EnableKernels());
    flare::DualView<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6,
                                                              5);

    auto success = validate_absence(
            [&]() {
                flare::resize(
                        flare::view_alloc(TEST_EXECSPACE{}, flare::WithoutInitializing),
                        bla, 5, 6, 7, 8);
                REQUIRE_EQ(bla.template view<TEST_EXECSPACE>().label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("flare::resize(View)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndFenceEvent event) {
                if (event.descriptor().find("flare::resize(View)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            },
            [&](BeginParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, realloc_exec_space_dualview") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using view_type = flare::DualView<int *, TEST_EXECSPACE>;
    view_type v(flare::view_alloc(TEST_EXECSPACE{}, "bla"), 8);

    auto success = validate_absence(
            [&]() {
                flare::realloc(flare::view_alloc(TEST_EXECSPACE{}), v, 8);
                REQUIRE_EQ(v.template view<TEST_EXECSPACE>().label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if ((event.descriptor().find("Debug Only Check for Execution Error") !=
                     std::string::npos) ||
                    (event.descriptor().find("HostSpace fence") != std::string::npos))
                    return MatchDiagnostic{false};
                return MatchDiagnostic{true, {"Found fence event!"}};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_realloc_no_init_dynrankview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DynRankView<int, TEST_EXECSPACE> bla("bla", 5, 6, 7, 8);

    auto success = validate_absence(
            [&]() {
                flare::resize(flare::WithoutInitializing, bla, 5, 6, 7, 9);
                REQUIRE_EQ(bla.label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 8, 8, 8);
                REQUIRE_EQ(bla.label(), "bla");
                flare::realloc(flare::view_alloc(flare::WithoutInitializing), bla, 5,
                               6, 7, 8);
                REQUIRE_EQ(bla.label(), "bla");
            },
            [&](BeginParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_exec_space_dynrankview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                       Config::EnableKernels());
    flare::DynRankView<int, TEST_EXECSPACE> bla("bla", 8, 7, 6, 5);

    auto success = validate_absence(
            [&]() {
                flare::resize(
                        flare::view_alloc(TEST_EXECSPACE{}, flare::WithoutInitializing),
                        bla, 5, 6, 7, 8);
                REQUIRE_EQ(bla.label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("flare::resize(View)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndFenceEvent event) {
                if (event.descriptor().find("flare::resize(View)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            },
            [&](BeginParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, realloc_exec_space_dynrankview") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

// FIXME_THREADS The Threads backend fences every parallel_for
#ifdef FLARE_ENABLE_THREADS
    if (std::is_same<TEST_EXECSPACE, flare::Threads>::value)
      GTEST_SKIP() << "skipping since the Threads backend isn't asynchronous";
#endif

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using view_type = flare::DynRankView<int, TEST_EXECSPACE>;
    view_type outer_view, outer_view2;

    auto success = validate_absence(
            [&]() {
                view_type inner_view(flare::view_alloc(TEST_EXECSPACE{}, "bla"), 8);
                // Avoid testing the destructor
                outer_view = inner_view;
                flare::realloc(
                        flare::view_alloc(flare::WithoutInitializing, TEST_EXECSPACE{}),
                        inner_view, 10);
                REQUIRE_EQ(inner_view.label(), "bla");
                outer_view2 = inner_view;
            },
            [&](BeginFenceEvent event) {
                if ((event.descriptor().find("Debug Only Check for Execution Error") !=
                     std::string::npos) ||
                    (event.descriptor().find("HostSpace fence") != std::string::npos))
                    return MatchDiagnostic{false};
                return MatchDiagnostic{true, {"Found fence event!"}};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_realloc_no_init_scatterview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::ScatterView<
            int ****[1][2][3], typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>
            bla("bla", 4, 5, 6, 7);

    auto success = validate_absence(
            [&]() {
                flare::resize(flare::WithoutInitializing, bla, 4, 5, 6, 8);
                REQUIRE_EQ(bla.subview().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 8, 8, 8);
                REQUIRE_EQ(bla.subview().label(), "bla");
                flare::realloc(flare::view_alloc(flare::WithoutInitializing), bla, 5,
                               6, 7, 8);
                REQUIRE_EQ(bla.subview().label(), "bla");
            },
            [&](BeginParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_realloc_no_alloc_scatterview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableAllocs());
    flare::experimental::ScatterView<
            int ****[1][2][3], typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>
            bla("bla", 7, 6, 5, 4);

    auto success = validate_absence(
            [&]() {
                flare::resize(bla, 7, 6, 5, 4);
                REQUIRE_EQ(bla.subview().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 7, 6, 5, 4);
                REQUIRE_EQ(bla.subview().label(), "bla");
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            },
            [&](AllocateDataEvent) {
                return MatchDiagnostic{true, {"Found alloc event"}};
            },
            [&](DeallocateDataEvent) {
                return MatchDiagnostic{true, {"Found dealloc event"}};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_exec_space_scatterview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                       Config::EnableKernels());
    flare::experimental::ScatterView<
            int ****[1][2][3], typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>
            bla("bla", 7, 6, 5, 4);

    auto success = validate_absence(
            [&]() {
                flare::resize(
                        flare::view_alloc(TEST_EXECSPACE{}, flare::WithoutInitializing),
                        bla, 5, 6, 7, 8);
                REQUIRE_EQ(bla.subview().label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("flare::resize(View)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndFenceEvent event) {
                if (event.descriptor().find("flare::resize(View)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            },
            [&](BeginParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndParallelForEvent event) {
                if (event.descriptor().find("initialization") != std::string::npos)
                    return MatchDiagnostic{true, {"Found end event"}};
                return MatchDiagnostic{false};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, realloc_exec_space_scatterview") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

// FIXME_THREADS The Threads backend fences every parallel_for
#ifdef FLARE_ENABLE_THREADS
    if (std::is_same<typename TEST_EXECSPACE, flare::Threads>::value)
      GTEST_SKIP() << "skipping since the Threads backend isn't asynchronous";
#endif

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using view_type = flare::experimental::ScatterView<
            int *, typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>;
    view_type outer_view, outer_view2;

    auto success = validate_absence(
            [&]() {
                view_type inner_view(flare::view_alloc(TEST_EXECSPACE{}, "bla"), 8);
                // Avoid testing the destructor
                outer_view = inner_view;
                flare::realloc(
                        flare::view_alloc(flare::WithoutInitializing, TEST_EXECSPACE{}),
                        inner_view, 10);
                REQUIRE_EQ(inner_view.subview().label(), "bla");
                outer_view2 = inner_view;
                flare::realloc(flare::view_alloc(TEST_EXECSPACE{}), inner_view, 10);
                REQUIRE_EQ(inner_view.subview().label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if ((event.descriptor().find("Debug Only Check for Execution Error") !=
                     std::string::npos) ||
                    (event.descriptor().find("HostSpace fence") != std::string::npos))
                    return MatchDiagnostic{false};
                return MatchDiagnostic{true, {"Found fence event!"}};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynrankview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DynRankView<int, TEST_EXECSPACE> device_view("device view", 10);
    flare::DynRankView<int, flare::HostSpace> host_view("host view", 10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device =
                        flare::create_mirror(flare::WithoutInitializing, device_view);
                REQUIRE_EQ(device_view.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(flare::WithoutInitializing,
                                                        TEST_EXECSPACE{}, host_view);
                REQUIRE_EQ(host_view.size(), mirror_host.size());
                auto mirror_device_view = flare::create_mirror_view(
                        flare::WithoutInitializing, device_view);
                REQUIRE_EQ(device_view.size(), mirror_device_view.size());
                auto mirror_host_view = flare::create_mirror_view(
                        flare::WithoutInitializing, TEST_EXECSPACE{}, host_view);
                REQUIRE_EQ(host_view.size(), mirror_host_view.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynrankview_viewctor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DynRankView<int, flare::DefaultExecutionSpace> device_view(
            "device view", 10);
    flare::DynRankView<int, flare::HostSpace> host_view("host view", 10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror(
                        flare::view_alloc(flare::WithoutInitializing), device_view);
                REQUIRE_EQ(device_view.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::view_alloc(flare::WithoutInitializing,
                                          flare::DefaultHostExecutionSpace{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host.size());
                auto mirror_device_view = flare::create_mirror_view(
                        flare::view_alloc(flare::WithoutInitializing), device_view);
                REQUIRE_EQ(device_view.size(), mirror_device_view.size());
                auto mirror_host_view = flare::create_mirror_view(
                        flare::view_alloc(flare::WithoutInitializing,
                                          flare::DefaultExecutionSpace{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host_view.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_view_and_copy_dynrankview") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableFences());

    flare::DynRankView<int, flare::HostSpace> host_view("host view", 10);
    decltype(flare::create_mirror_view_and_copy(TEST_EXECSPACE{},
                                                host_view)) device_view;

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror_view_and_copy(
                        flare::view_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_device.size());
                // Avoid fences for deallocation when mirror_device goes out of scope.
                device_view = mirror_device;
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found parallel_for event"}};
            },
            [&](BeginFenceEvent) {
                return MatchDiagnostic{true, {"Found fence event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_offsetview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::OffsetView<int *, TEST_EXECSPACE> device_view(
            "device view", {0, 10});
    flare::experimental::OffsetView<int *, flare::HostSpace> host_view(
            "host view", {0, 10});

    auto success = validate_absence(
            [&]() {
                device_view = flare::experimental::OffsetView<int *, TEST_EXECSPACE>(
                        flare::view_alloc(flare::WithoutInitializing, "device view"),
                        {0, 10});

                auto mirror_device =
                        flare::create_mirror(flare::WithoutInitializing, device_view);
                REQUIRE_EQ(device_view.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::WithoutInitializing, flare::DefaultHostExecutionSpace{},
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host.size());
                auto mirror_device_view = flare::create_mirror_view(
                        flare::WithoutInitializing, device_view);
                REQUIRE_EQ(device_view.size(), mirror_device_view.size());
                auto mirror_host_view = flare::create_mirror_view(
                        flare::WithoutInitializing, flare::DefaultHostExecutionSpace{},
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host_view.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_offsetview_view_ctor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::OffsetView<int *, flare::DefaultExecutionSpace>
            device_view("device view", {0, 10});
    flare::experimental::OffsetView<int *, flare::HostSpace> host_view(
            "host view", {0, 10});

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror(
                        flare::view_alloc(flare::WithoutInitializing), device_view);
                REQUIRE_EQ(device_view.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::view_alloc(flare::WithoutInitializing,
                                          flare::DefaultHostExecutionSpace{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host.size());
                auto mirror_device_view = flare::create_mirror_view(
                        flare::view_alloc(flare::WithoutInitializing), device_view);
                REQUIRE_EQ(device_view.size(), mirror_device_view.size());
                auto mirror_host_view = flare::create_mirror_view(
                        flare::view_alloc(flare::WithoutInitializing,
                                          flare::DefaultHostExecutionSpace{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host_view.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_view_and_copy_offsetview") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableFences());

    flare::experimental::OffsetView<int *, flare::HostSpace> host_view(
            "host view", {0, 10});
    decltype(flare::create_mirror_view_and_copy(TEST_EXECSPACE{},
                                                host_view)) device_view;

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror_view_and_copy(
                        flare::view_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_device.size());
                // Avoid fences for deallocation when mirror_device goes out of scope.
                device_view = mirror_device;
                auto mirror_device_mirror = flare::create_mirror_view_and_copy(
                        flare::view_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        mirror_device);
                REQUIRE_EQ(mirror_device_mirror.size(), mirror_device.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found parallel_for event"}};
            },
            [&](BeginFenceEvent) {
                return MatchDiagnostic{true, {"Found fence event"}};
            });
    REQUIRE(success);
}


TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynamicview") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::DynamicView<int *, TEST_EXECSPACE> device_view(
            "device view", 2, 10);
    device_view.resize_serial(10);
    flare::experimental::DynamicView<int *, flare::HostSpace> host_view(
            "host view", 2, 10);
    host_view.resize_serial(10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device =
                        flare::create_mirror(flare::WithoutInitializing, device_view);
                REQUIRE_EQ(device_view.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(flare::WithoutInitializing,
                                                        TEST_EXECSPACE{}, host_view);
                REQUIRE_EQ(host_view.size(), mirror_host.size());
                auto mirror_device_view = flare::create_mirror_view(
                        flare::WithoutInitializing, device_view);
                REQUIRE_EQ(device_view.size(), mirror_device_view.size());
                auto mirror_host_view = flare::create_mirror_view(
                        flare::WithoutInitializing, TEST_EXECSPACE{}, host_view);
                REQUIRE_EQ(host_view.size(), mirror_host_view.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_view_and_copy_dynamicview") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableFences());

    flare::experimental::DynamicView<int *, flare::HostSpace> host_view(
            "host view", 2, 10);
    host_view.resize_serial(10);
    decltype(flare::create_mirror_view_and_copy(TEST_EXECSPACE{},
                                                host_view)) device_view;

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror_view_and_copy(
                        flare::view_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_device.size());
                // Avoid fences for deallocation when mirror_device goes out of scope.
                device_view = mirror_device;
                auto mirror_device_mirror = flare::create_mirror_view_and_copy(
                        flare::view_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        mirror_device);
                REQUIRE_EQ(mirror_device_mirror.size(), mirror_device.size());
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("DynamicView::resize_serial: Fence after "
                                            "copying chunks to the device") !=
                    std::string::npos)
                    return MatchDiagnostic{false};
                return MatchDiagnostic{true, {"Found fence event"}};
            },
            [&](EndFenceEvent) { return MatchDiagnostic{false}; },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found parallel_for event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynamicview_view_ctor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::DynamicView<int *, flare::DefaultExecutionSpace>
            device_view("device view", 2, 10);
    device_view.resize_serial(10);
    flare::experimental::DynamicView<int *, flare::HostSpace> host_view(
            "host view", 2, 10);
    host_view.resize_serial(10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror(
                        flare::view_alloc(flare::WithoutInitializing), device_view);
                REQUIRE_EQ(device_view.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::view_alloc(flare::WithoutInitializing,
                                          flare::DefaultExecutionSpace{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host.size());
                auto mirror_device_view = flare::create_mirror_view(
                        flare::view_alloc(flare::WithoutInitializing), device_view);
                REQUIRE_EQ(device_view.size(), mirror_device_view.size());
                auto mirror_host_view = flare::create_mirror_view(
                        flare::view_alloc(flare::WithoutInitializing,
                                          flare::DefaultExecutionSpace{}),
                        host_view);
                REQUIRE_EQ(host_view.size(), mirror_host_view.size());
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("DynamicView::resize_serial: Fence after "
                                            "copying chunks to the device") !=
                    std::string::npos)
                    return MatchDiagnostic{false};
                return MatchDiagnostic{true, {"Found fence event"}};
            },
            [&](EndFenceEvent) { return MatchDiagnostic{false}; },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}
