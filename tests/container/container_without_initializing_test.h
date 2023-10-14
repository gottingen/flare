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
#include <flare/dual_tensor.h>
#include <flare/dynamic_tensor.h>
#include <flare/dyn_rank_tensor.h>
#include <flare/offset_tensor.h>
#include <flare/scatter_tensor.h>

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

TEST_CASE("TEST_CATEGORY, resize_realloc_no_init_dualtensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DualTensor<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 5, 6, 7,
                                                              8);

    auto success = validate_absence(
            [&]() {
                flare::resize(flare::WithoutInitializing, bla, 5, 6, 7, 9);
                REQUIRE_EQ(bla.template tensor<TEST_EXECSPACE>().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 8, 8, 8);
                REQUIRE_EQ(bla.template tensor<TEST_EXECSPACE>().label(), "bla");
                flare::realloc(flare::tensor_alloc(flare::WithoutInitializing), bla, 5,
                               6, 7, 8);
                REQUIRE_EQ(bla.template tensor<TEST_EXECSPACE>().label(), "bla");
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

TEST_CASE("TEST_CATEGORY, resize_realloc_no_alloc_dualtensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableAllocs());
    flare::DualTensor<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6,
                                                              5);

    auto success = validate_absence(
            [&]() {
                flare::resize(bla, 8, 7, 6, 5);
                REQUIRE_EQ(bla.template tensor<TEST_EXECSPACE>().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 7, 6, 5);
                REQUIRE_EQ(bla.template tensor<TEST_EXECSPACE>().label(), "bla");
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

TEST_CASE("TEST_CATEGORY, resize_exec_space_dualtensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                       Config::EnableKernels());
    flare::DualTensor<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6,
                                                              5);

    auto success = validate_absence(
            [&]() {
                flare::resize(
                        flare::tensor_alloc(TEST_EXECSPACE{}, flare::WithoutInitializing),
                        bla, 5, 6, 7, 8);
                REQUIRE_EQ(bla.template tensor<TEST_EXECSPACE>().label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("flare::resize(Tensor)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndFenceEvent event) {
                if (event.descriptor().find("flare::resize(Tensor)") !=
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

TEST_CASE("TEST_CATEGORY, realloc_exec_space_dualtensor") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::DualTensor<int *, TEST_EXECSPACE>;
    tensor_type v(flare::tensor_alloc(TEST_EXECSPACE{}, "bla"), 8);

    auto success = validate_absence(
            [&]() {
                flare::realloc(flare::tensor_alloc(TEST_EXECSPACE{}), v, 8);
                REQUIRE_EQ(v.template tensor<TEST_EXECSPACE>().label(), "bla");
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

TEST_CASE("TEST_CATEGORY, resize_realloc_no_init_dynranktensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DynRankTensor<int, TEST_EXECSPACE> bla("bla", 5, 6, 7, 8);

    auto success = validate_absence(
            [&]() {
                flare::resize(flare::WithoutInitializing, bla, 5, 6, 7, 9);
                REQUIRE_EQ(bla.label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 8, 8, 8);
                REQUIRE_EQ(bla.label(), "bla");
                flare::realloc(flare::tensor_alloc(flare::WithoutInitializing), bla, 5,
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

TEST_CASE("TEST_CATEGORY, resize_exec_space_dynranktensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                       Config::EnableKernels());
    flare::DynRankTensor<int, TEST_EXECSPACE> bla("bla", 8, 7, 6, 5);

    auto success = validate_absence(
            [&]() {
                flare::resize(
                        flare::tensor_alloc(TEST_EXECSPACE{}, flare::WithoutInitializing),
                        bla, 5, 6, 7, 8);
                REQUIRE_EQ(bla.label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("flare::resize(Tensor)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndFenceEvent event) {
                if (event.descriptor().find("flare::resize(Tensor)") !=
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

TEST_CASE("TEST_CATEGORY, realloc_exec_space_dynranktensor") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

// FIXME_THREADS The Threads backend fences every parallel_for
#ifdef FLARE_ENABLE_THREADS
    if (std::is_same<TEST_EXECSPACE, flare::Threads>::value)
      GTEST_SKIP() << "skipping since the Threads backend isn't asynchronous";
#endif

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::DynRankTensor<int, TEST_EXECSPACE>;
    tensor_type outer_tensor, outer_tensor2;

    auto success = validate_absence(
            [&]() {
                tensor_type inner_tensor(flare::tensor_alloc(TEST_EXECSPACE{}, "bla"), 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
                flare::realloc(
                        flare::tensor_alloc(flare::WithoutInitializing, TEST_EXECSPACE{}),
                        inner_tensor, 10);
                REQUIRE_EQ(inner_tensor.label(), "bla");
                outer_tensor2 = inner_tensor;
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

TEST_CASE("TEST_CATEGORY, resize_realloc_no_init_scattertensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::ScatterTensor<
            int ****[1][2][3], typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>
            bla("bla", 4, 5, 6, 7);

    auto success = validate_absence(
            [&]() {
                flare::resize(flare::WithoutInitializing, bla, 4, 5, 6, 8);
                REQUIRE_EQ(bla.subtensor().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 8, 8, 8, 8);
                REQUIRE_EQ(bla.subtensor().label(), "bla");
                flare::realloc(flare::tensor_alloc(flare::WithoutInitializing), bla, 5,
                               6, 7, 8);
                REQUIRE_EQ(bla.subtensor().label(), "bla");
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

TEST_CASE("TEST_CATEGORY, resize_realloc_no_alloc_scattertensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableAllocs());
    flare::experimental::ScatterTensor<
            int ****[1][2][3], typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>
            bla("bla", 7, 6, 5, 4);

    auto success = validate_absence(
            [&]() {
                flare::resize(bla, 7, 6, 5, 4);
                REQUIRE_EQ(bla.subtensor().label(), "bla");
                flare::realloc(flare::WithoutInitializing, bla, 7, 6, 5, 4);
                REQUIRE_EQ(bla.subtensor().label(), "bla");
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

TEST_CASE("TEST_CATEGORY, resize_exec_space_scattertensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                       Config::EnableKernels());
    flare::experimental::ScatterTensor<
            int ****[1][2][3], typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>
            bla("bla", 7, 6, 5, 4);

    auto success = validate_absence(
            [&]() {
                flare::resize(
                        flare::tensor_alloc(TEST_EXECSPACE{}, flare::WithoutInitializing),
                        bla, 5, 6, 7, 8);
                REQUIRE_EQ(bla.subtensor().label(), "bla");
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("flare::resize(Tensor)") !=
                    std::string::npos)
                    return MatchDiagnostic{true, {"Found begin event"}};
                return MatchDiagnostic{false};
            },
            [&](EndFenceEvent event) {
                if (event.descriptor().find("flare::resize(Tensor)") !=
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

TEST_CASE("TEST_CATEGORY, realloc_exec_space_scattertensor") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

// FIXME_THREADS The Threads backend fences every parallel_for
#ifdef FLARE_ENABLE_THREADS
    if (std::is_same<typename TEST_EXECSPACE, flare::Threads>::value)
      GTEST_SKIP() << "skipping since the Threads backend isn't asynchronous";
#endif

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::experimental::ScatterTensor<
            int *, typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>;
    tensor_type outer_tensor, outer_tensor2;

    auto success = validate_absence(
            [&]() {
                tensor_type inner_tensor(flare::tensor_alloc(TEST_EXECSPACE{}, "bla"), 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
                flare::realloc(
                        flare::tensor_alloc(flare::WithoutInitializing, TEST_EXECSPACE{}),
                        inner_tensor, 10);
                REQUIRE_EQ(inner_tensor.subtensor().label(), "bla");
                outer_tensor2 = inner_tensor;
                flare::realloc(flare::tensor_alloc(TEST_EXECSPACE{}), inner_tensor, 10);
                REQUIRE_EQ(inner_tensor.subtensor().label(), "bla");
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

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynranktensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DynRankTensor<int, TEST_EXECSPACE> device_tensor("device tensor", 10);
    flare::DynRankTensor<int, flare::HostSpace> host_tensor("host tensor", 10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device =
                        flare::create_mirror(flare::WithoutInitializing, device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(flare::WithoutInitializing,
                                                        TEST_EXECSPACE{}, host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host.size());
                auto mirror_device_tensor = flare::create_mirror_tensor(
                        flare::WithoutInitializing, device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device_tensor.size());
                auto mirror_host_tensor = flare::create_mirror_tensor(
                        flare::WithoutInitializing, TEST_EXECSPACE{}, host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host_tensor.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynranktensor_tensorctor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::DynRankTensor<int, flare::DefaultExecutionSpace> device_tensor(
            "device tensor", 10);
    flare::DynRankTensor<int, flare::HostSpace> host_tensor("host tensor", 10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror(
                        flare::tensor_alloc(flare::WithoutInitializing), device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::tensor_alloc(flare::WithoutInitializing,
                                          flare::DefaultHostExecutionSpace{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host.size());
                auto mirror_device_tensor = flare::create_mirror_tensor(
                        flare::tensor_alloc(flare::WithoutInitializing), device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device_tensor.size());
                auto mirror_host_tensor = flare::create_mirror_tensor(
                        flare::tensor_alloc(flare::WithoutInitializing,
                                          flare::DefaultExecutionSpace{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host_tensor.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_tensor_and_copy_dynranktensor") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableFences());

    flare::DynRankTensor<int, flare::HostSpace> host_tensor("host tensor", 10);
    decltype(flare::create_mirror_tensor_and_copy(TEST_EXECSPACE{},
                                                host_tensor)) device_tensor;

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror_tensor_and_copy(
                        flare::tensor_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_device.size());
                // Avoid fences for deallocation when mirror_device goes out of scope.
                device_tensor = mirror_device;
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found parallel_for event"}};
            },
            [&](BeginFenceEvent) {
                return MatchDiagnostic{true, {"Found fence event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_offsettensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::OffsetTensor<int *, TEST_EXECSPACE> device_tensor(
            "device tensor", {0, 10});
    flare::experimental::OffsetTensor<int *, flare::HostSpace> host_tensor(
            "host tensor", {0, 10});

    auto success = validate_absence(
            [&]() {
                device_tensor = flare::experimental::OffsetTensor<int *, TEST_EXECSPACE>(
                        flare::tensor_alloc(flare::WithoutInitializing, "device tensor"),
                        {0, 10});

                auto mirror_device =
                        flare::create_mirror(flare::WithoutInitializing, device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::WithoutInitializing, flare::DefaultHostExecutionSpace{},
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host.size());
                auto mirror_device_tensor = flare::create_mirror_tensor(
                        flare::WithoutInitializing, device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device_tensor.size());
                auto mirror_host_tensor = flare::create_mirror_tensor(
                        flare::WithoutInitializing, flare::DefaultHostExecutionSpace{},
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host_tensor.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_offsettensor_tensor_ctor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::OffsetTensor<int *, flare::DefaultExecutionSpace>
            device_tensor("device tensor", {0, 10});
    flare::experimental::OffsetTensor<int *, flare::HostSpace> host_tensor(
            "host tensor", {0, 10});

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror(
                        flare::tensor_alloc(flare::WithoutInitializing), device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::tensor_alloc(flare::WithoutInitializing,
                                          flare::DefaultHostExecutionSpace{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host.size());
                auto mirror_device_tensor = flare::create_mirror_tensor(
                        flare::tensor_alloc(flare::WithoutInitializing), device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device_tensor.size());
                auto mirror_host_tensor = flare::create_mirror_tensor(
                        flare::tensor_alloc(flare::WithoutInitializing,
                                          flare::DefaultHostExecutionSpace{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host_tensor.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_tensor_and_copy_offsettensor") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableFences());

    flare::experimental::OffsetTensor<int *, flare::HostSpace> host_tensor(
            "host tensor", {0, 10});
    decltype(flare::create_mirror_tensor_and_copy(TEST_EXECSPACE{},
                                                host_tensor)) device_tensor;

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror_tensor_and_copy(
                        flare::tensor_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_device.size());
                // Avoid fences for deallocation when mirror_device goes out of scope.
                device_tensor = mirror_device;
                auto mirror_device_mirror = flare::create_mirror_tensor_and_copy(
                        flare::tensor_alloc(TEST_EXECSPACE{},
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


TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynamictensor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::DynamicTensor<int *, TEST_EXECSPACE> device_tensor(
            "device tensor", 2, 10);
    device_tensor.resize_serial(10);
    flare::experimental::DynamicTensor<int *, flare::HostSpace> host_tensor(
            "host tensor", 2, 10);
    host_tensor.resize_serial(10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device =
                        flare::create_mirror(flare::WithoutInitializing, device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(flare::WithoutInitializing,
                                                        TEST_EXECSPACE{}, host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host.size());
                auto mirror_device_tensor = flare::create_mirror_tensor(
                        flare::WithoutInitializing, device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device_tensor.size());
                auto mirror_host_tensor = flare::create_mirror_tensor(
                        flare::WithoutInitializing, TEST_EXECSPACE{}, host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host_tensor.size());
            },
            [&](BeginParallelForEvent) {
                return MatchDiagnostic{true, {"Found begin event"}};
            },
            [&](EndParallelForEvent) {
                return MatchDiagnostic{true, {"Found end event"}};
            });
    REQUIRE(success);
}

TEST_CASE("TEST_CATEGORY, create_mirror_tensor_and_copy_dynamictensor") {
    DOCTEST_SKIP_IF_CUDAUVM_MEMORY_SPACE

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableFences());

    flare::experimental::DynamicTensor<int *, flare::HostSpace> host_tensor(
            "host tensor", 2, 10);
    host_tensor.resize_serial(10);
    decltype(flare::create_mirror_tensor_and_copy(TEST_EXECSPACE{},
                                                host_tensor)) device_tensor;

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror_tensor_and_copy(
                        flare::tensor_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_device.size());
                // Avoid fences for deallocation when mirror_device goes out of scope.
                device_tensor = mirror_device;
                auto mirror_device_mirror = flare::create_mirror_tensor_and_copy(
                        flare::tensor_alloc(TEST_EXECSPACE{},
                                          typename TEST_EXECSPACE::memory_space{}),
                        mirror_device);
                REQUIRE_EQ(mirror_device_mirror.size(), mirror_device.size());
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("DynamicTensor::resize_serial: Fence after "
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

TEST_CASE("TEST_CATEGORY, create_mirror_no_init_dynamictensor_tensor_ctor") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::experimental::DynamicTensor<int *, flare::DefaultExecutionSpace>
            device_tensor("device tensor", 2, 10);
    device_tensor.resize_serial(10);
    flare::experimental::DynamicTensor<int *, flare::HostSpace> host_tensor(
            "host tensor", 2, 10);
    host_tensor.resize_serial(10);

    auto success = validate_absence(
            [&]() {
                auto mirror_device = flare::create_mirror(
                        flare::tensor_alloc(flare::WithoutInitializing), device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device.size());
                auto mirror_host = flare::create_mirror(
                        flare::tensor_alloc(flare::WithoutInitializing,
                                          flare::DefaultExecutionSpace{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host.size());
                auto mirror_device_tensor = flare::create_mirror_tensor(
                        flare::tensor_alloc(flare::WithoutInitializing), device_tensor);
                REQUIRE_EQ(device_tensor.size(), mirror_device_tensor.size());
                auto mirror_host_tensor = flare::create_mirror_tensor(
                        flare::tensor_alloc(flare::WithoutInitializing,
                                          flare::DefaultExecutionSpace{}),
                        host_tensor);
                REQUIRE_EQ(host_tensor.size(), mirror_host_tensor.size());
            },
            [&](BeginFenceEvent event) {
                if (event.descriptor().find("DynamicTensor::resize_serial: Fence after "
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
