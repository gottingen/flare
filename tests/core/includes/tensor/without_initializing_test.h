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
#include "tool/tool_testing_utilities.h"

TEST_CASE("TEST_CATEGORY, resize_realloc_no_init") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::Tensor<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 5, 6, 7, 8);

    auto success = validate_absence(
            [&]() {
                flare::resize(flare::WithoutInitializing, bla, 5, 6, 7, 9);
                flare::realloc(flare::WithoutInitializing, bla, 8, 8, 8, 8);
                flare::realloc(flare::tensor_alloc(flare::WithoutInitializing), bla, 5,
                               6, 7, 8);
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

TEST_CASE("TEST_CATEGORY, resize_realloc_no_alloc") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels(),
                       Config::EnableAllocs());
    flare::Tensor<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6, 5);

    auto success = validate_absence(
            [&]() {
                flare::resize(bla, 8, 7, 6, 5);
                flare::realloc(flare::WithoutInitializing, bla, 8, 7, 6, 5);
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

TEST_CASE("TEST_CATEGORY, realloc_exec_space") {
#ifdef FLARE_ON_CUDA_DEVICE
    if (std::is_same<typename TEST_EXECSPACE::memory_space,
                     flare::CudaUVMSpace>::value)
      INFO("skipping since CudaUVMSpace requires additional fences");
      return;
#endif

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::Tensor<int *, TEST_EXECSPACE>;
    tensor_type outer_tensor, outer_tensor2;

    auto success = validate_absence(
            [&]() {
                tensor_type inner_tensor(flare::tensor_alloc(TEST_EXECSPACE{}, "bla"), 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
                flare::realloc(
                        flare::tensor_alloc(flare::WithoutInitializing, TEST_EXECSPACE{}),
                        inner_tensor, 10);
                outer_tensor2 = inner_tensor;
                flare::realloc(flare::tensor_alloc(TEST_EXECSPACE{}), inner_tensor, 10);
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

namespace {
    struct NonTriviallyCopyable {
        FLARE_FUNCTION NonTriviallyCopyable() {}

        FLARE_FUNCTION NonTriviallyCopyable(const NonTriviallyCopyable &) {}
    };
}  // namespace

TEST_CASE("TEST_CATEGORY, tensor_alloc") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::Tensor<NonTriviallyCopyable *, TEST_EXECSPACE>;
    tensor_type outer_tensor;

    auto success = validate_existence(
            [&]() {
                tensor_type inner_tensor(flare::tensor_alloc("bla"), 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
            },
            [&](BeginFenceEvent event) {
                return MatchDiagnostic{
                        event.descriptor().find(
                                "flare::detail::TensorValueFunctor: Tensor init/destroy fence") !=
                        std::string::npos};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, view_alloc_exec_space") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::Tensor<NonTriviallyCopyable *, TEST_EXECSPACE>;
    tensor_type outer_tensor;

    auto success = validate_absence(
            [&]() {
                tensor_type inner_tensor(flare::tensor_alloc(TEST_EXECSPACE{}, "bla"), 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
            },
            [&](BeginFenceEvent event) {
                return MatchDiagnostic{
                        event.descriptor().find(
                                "flare::detail::TensorValueFunctor: Tensor init/destroy fence") !=
                        std::string::npos};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, view_alloc_int") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::Tensor<int *, TEST_EXECSPACE>;
    tensor_type outer_tensor;

    auto success = validate_existence(
            [&]() {
                tensor_type inner_tensor("bla", 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
            },
            [&](BeginFenceEvent event) {
                return MatchDiagnostic{
                        event.descriptor().find(
                                "flare::detail::TensorValueFunctor: Tensor init/destroy fence") !=
                        std::string::npos};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, view_alloc_exec_space_int") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences());
    using tensor_type = flare::Tensor<int *, TEST_EXECSPACE>;
    tensor_type outer_tensor;

    auto success = validate_absence(
            [&]() {
                tensor_type inner_tensor(flare::tensor_alloc(TEST_EXECSPACE{}, "bla"), 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
            },
            [&](BeginFenceEvent event) {
                return MatchDiagnostic{
                        event.descriptor().find(
                                "flare::detail::TensorValueFunctor: Tensor init/destroy fence") !=
                        std::string::npos};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, deep_copy_zero_memset") {

    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableKernels());
    flare::Tensor<int *, TEST_EXECSPACE> bla("bla", 8);

    auto success =
            validate_absence([&]() { flare::deep_copy(bla, 0); },
                             [&](BeginParallelForEvent) {
                                 return MatchDiagnostic{true, {"Found begin event"}};
                             },
                             [&](EndParallelForEvent) {
                                 return MatchDiagnostic{true, {"Found end event"}};
                             });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, resize_exec_space") {
    using namespace flare::Test::Tools;
    listen_tool_events(Config::DisableAll(), Config::EnableFences(),
                       Config::EnableKernels());
    flare::Tensor<int ****[1][2][3][4], TEST_EXECSPACE> bla("bla", 8, 7, 6, 5);

    auto success = validate_absence(
            [&]() {
                flare::resize(
                        flare::tensor_alloc(TEST_EXECSPACE{}, flare::WithoutInitializing),
                        bla, 5, 6, 7, 8);
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

TEST_CASE("TEST_CATEGORY, view_allocation_int") {
    using ExecutionSpace = TEST_EXECSPACE;
    if (flare::SpaceAccessibility<
            /*AccessSpace=*/flare::HostSpace,
            /*MemorySpace=*/ExecutionSpace::memory_space>::accessible) {
        INFO("skipping since the fence checked for isn't necessary");
        return;
    }
    using namespace flare::Test::Tools;
    listen_tool_events(Config::EnableAll());
    using tensor_type = flare::Tensor<int *, TEST_EXECSPACE>;
    tensor_type outer_tensor;

    auto success = validate_existence(
            [&]() {
                tensor_type inner_tensor(
                        flare::tensor_alloc(flare::WithoutInitializing, "bla"), 8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
            },
            [&](BeginFenceEvent event) {
                return MatchDiagnostic{
                        event.descriptor().find(
                                "fence after copying header from HostSpace") !=
                        std::string::npos};
            });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

TEST_CASE("TEST_CATEGORY, view_allocation_exec_space_int") {
#ifdef FLARE_ON_CUDA_DEVICE
    if (std::is_same<TEST_EXECSPACE::memory_space, flare::CudaUVMSpace>::value)
      INFO("skipping since the CudaUVMSpace requires additiional fences");
#endif

    using namespace flare::Test::Tools;
    listen_tool_events(Config::EnableAll());
    using tensor_type = flare::Tensor<int *, TEST_EXECSPACE>;
    tensor_type outer_tensor;

    auto success = validate_absence(
            [&]() {
                tensor_type inner_tensor(flare::tensor_alloc(flare::WithoutInitializing,
                                                       TEST_EXECSPACE{}, "bla"),
                                     8);
                // Avoid testing the destructor
                outer_tensor = inner_tensor;
            },
            [&](BeginFenceEvent) { return MatchDiagnostic{true}; });
    REQUIRE(success);
    listen_tool_events(Config::DisableAll());
}

struct NotDefaultConstructible {
    NotDefaultConstructible() = delete;
};

TEST_CASE("TEST_CATEGORY, view_not_default_constructible") {
    using Space = TEST_EXECSPACE;
    flare::Tensor<NotDefaultConstructible, Space> my_view(flare::tensor_alloc(
            "not_default_constructible", flare::WithoutInitializing));
}
