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

template<typename TestView, typename MemorySpace>
void check_memory_space(TestView, MemorySpace) {
    static_assert(std::is_same_v<typename TestView::memory_space, MemorySpace>);
}

template<class View>
auto host_mirror_test_space(View) {
    return std::conditional_t<
            flare::SpaceAccessibility<flare::HostSpace,
                    typename View::memory_space>::accessible,
            typename View::memory_space, flare::HostSpace>{};
}

template<typename View>
void test_create_mirror_properties(const View &view) {
    using namespace flare;
    using DeviceMemorySpace = typename DefaultExecutionSpace::memory_space;

    // clang-format off

    // create_mirror
    check_memory_space(create_mirror(WithoutInitializing, view), host_mirror_test_space(view));
    check_memory_space(create_mirror(view), host_mirror_test_space(view));
    check_memory_space(create_mirror(WithoutInitializing, DefaultExecutionSpace{}, view), DeviceMemorySpace{});
    check_memory_space(create_mirror(DefaultExecutionSpace{}, view), DeviceMemorySpace{});

    // create_mirror_view
    check_memory_space(create_mirror_view(WithoutInitializing, view), host_mirror_test_space(view));
    check_memory_space(create_mirror_view(view), host_mirror_test_space(view));
    check_memory_space(create_mirror_view(WithoutInitializing, DefaultExecutionSpace{}, view), DeviceMemorySpace{});
    check_memory_space(create_mirror_view(DefaultExecutionSpace{}, view), DeviceMemorySpace{});

    // create_mirror view_alloc
    check_memory_space(create_mirror(view_alloc(WithoutInitializing), view), host_mirror_test_space(view));
    check_memory_space(create_mirror(view_alloc(), view), host_mirror_test_space(view));
    check_memory_space(create_mirror(view_alloc(WithoutInitializing, DeviceMemorySpace{}), view), DeviceMemorySpace{});
    check_memory_space(create_mirror(view_alloc(DeviceMemorySpace{}), view), DeviceMemorySpace{});

    // create_mirror_view view_alloc
    check_memory_space(create_mirror_view(view_alloc(WithoutInitializing), view), host_mirror_test_space(view));
    check_memory_space(create_mirror_view(view_alloc(), view), host_mirror_test_space(view));
    check_memory_space(create_mirror_view(view_alloc(WithoutInitializing, DeviceMemorySpace{}), view),
                       DeviceMemorySpace{});
    check_memory_space(create_mirror_view(view_alloc(DeviceMemorySpace{}), view), DeviceMemorySpace{});

    // create_mirror view_alloc + execution space
    check_memory_space(create_mirror(view_alloc(DefaultHostExecutionSpace{}, WithoutInitializing), view),
                       host_mirror_test_space(view));
    check_memory_space(create_mirror(view_alloc(DefaultHostExecutionSpace{}), view), host_mirror_test_space(view));
    check_memory_space(
            create_mirror(view_alloc(DefaultExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), view),
            DeviceMemorySpace{});
    check_memory_space(create_mirror(view_alloc(DefaultExecutionSpace{}, DeviceMemorySpace{}), view),
                       DeviceMemorySpace{});

    // create_mirror_view view_alloc + execution space
    check_memory_space(create_mirror_view(view_alloc(DefaultHostExecutionSpace{}, WithoutInitializing), view),
                       host_mirror_test_space(view));
    check_memory_space(create_mirror_view(view_alloc(DefaultHostExecutionSpace{}), view), host_mirror_test_space(view));
    check_memory_space(
            create_mirror_view(view_alloc(DefaultExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), view),
            DeviceMemorySpace{});
    check_memory_space(create_mirror_view(view_alloc(DefaultExecutionSpace{}, DeviceMemorySpace{}), view),
                       DeviceMemorySpace{});

    // create_mirror_view_and_copy
    check_memory_space(create_mirror_view_and_copy(HostSpace{}, view), HostSpace{});
    check_memory_space(create_mirror_view_and_copy(DeviceMemorySpace{}, view), DeviceMemorySpace{});

    // create_mirror_view_and_copy view_alloc
    check_memory_space(create_mirror_view_and_copy(view_alloc(HostSpace{}), view), HostSpace{});
    check_memory_space(create_mirror_view_and_copy(view_alloc(DeviceMemorySpace{}), view), DeviceMemorySpace{});

    // create_mirror_view_and_copy view_alloc + execution space
    check_memory_space(create_mirror_view_and_copy(view_alloc(HostSpace{}, DefaultHostExecutionSpace{}), view),
                       HostSpace{});
    check_memory_space(create_mirror_view_and_copy(view_alloc(DeviceMemorySpace{}, DefaultExecutionSpace{}), view),
                       DeviceMemorySpace{});

    // clang-format on
}

void test() {
    flare::View<int *, flare::DefaultExecutionSpace> device_view("device view",
                                                                 10);
    flare::View<int *, flare::HostSpace> host_view("host view", 10);

    test_create_mirror_properties(device_view);
    test_create_mirror_properties(host_view);
}
