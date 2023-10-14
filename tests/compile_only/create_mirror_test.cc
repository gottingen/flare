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

template<typename TestTensor, typename MemorySpace>
void check_memory_space(TestTensor, MemorySpace) {
    static_assert(std::is_same_v<typename TestTensor::memory_space, MemorySpace>);
}

template<class Tensor>
auto host_mirror_test_space(Tensor) {
    return std::conditional_t<
            flare::SpaceAccessibility<flare::HostSpace,
                    typename Tensor::memory_space>::accessible,
            typename Tensor::memory_space, flare::HostSpace>{};
}

template<typename Tensor>
void test_create_mirror_properties(const Tensor &tensor) {
    using namespace flare;
    using DeviceMemorySpace = typename DefaultExecutionSpace::memory_space;

    // clang-format off

    // create_mirror
    check_memory_space(create_mirror(WithoutInitializing, tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror(tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror(WithoutInitializing, DefaultExecutionSpace{}, tensor), DeviceMemorySpace{});
    check_memory_space(create_mirror(DefaultExecutionSpace{}, tensor), DeviceMemorySpace{});

    // create_mirror_tensor
    check_memory_space(create_mirror_tensor(WithoutInitializing, tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror_tensor(tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror_tensor(WithoutInitializing, DefaultExecutionSpace{}, tensor), DeviceMemorySpace{});
    check_memory_space(create_mirror_tensor(DefaultExecutionSpace{}, tensor), DeviceMemorySpace{});

    // create_mirror tensor_alloc
    check_memory_space(create_mirror(tensor_alloc(WithoutInitializing), tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror(tensor_alloc(), tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror(tensor_alloc(WithoutInitializing, DeviceMemorySpace{}), tensor), DeviceMemorySpace{});
    check_memory_space(create_mirror(tensor_alloc(DeviceMemorySpace{}), tensor), DeviceMemorySpace{});

    // create_mirror_tensor tensor_alloc
    check_memory_space(create_mirror_tensor(tensor_alloc(WithoutInitializing), tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror_tensor(tensor_alloc(), tensor), host_mirror_test_space(tensor));
    check_memory_space(create_mirror_tensor(tensor_alloc(WithoutInitializing, DeviceMemorySpace{}), tensor),
                       DeviceMemorySpace{});
    check_memory_space(create_mirror_tensor(tensor_alloc(DeviceMemorySpace{}), tensor), DeviceMemorySpace{});

    // create_mirror tensor_alloc + execution space
    check_memory_space(create_mirror(tensor_alloc(DefaultHostExecutionSpace{}, WithoutInitializing), tensor),
                       host_mirror_test_space(tensor));
    check_memory_space(create_mirror(tensor_alloc(DefaultHostExecutionSpace{}), tensor), host_mirror_test_space(tensor));
    check_memory_space(
            create_mirror(tensor_alloc(DefaultExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), tensor),
            DeviceMemorySpace{});
    check_memory_space(create_mirror(tensor_alloc(DefaultExecutionSpace{}, DeviceMemorySpace{}), tensor),
                       DeviceMemorySpace{});

    // create_mirror_tensor tensor_alloc + execution space
    check_memory_space(create_mirror_tensor(tensor_alloc(DefaultHostExecutionSpace{}, WithoutInitializing), tensor),
                       host_mirror_test_space(tensor));
    check_memory_space(create_mirror_tensor(tensor_alloc(DefaultHostExecutionSpace{}), tensor), host_mirror_test_space(tensor));
    check_memory_space(
            create_mirror_tensor(tensor_alloc(DefaultExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), tensor),
            DeviceMemorySpace{});
    check_memory_space(create_mirror_tensor(tensor_alloc(DefaultExecutionSpace{}, DeviceMemorySpace{}), tensor),
                       DeviceMemorySpace{});

    // create_mirror_tensor_and_copy
    check_memory_space(create_mirror_tensor_and_copy(HostSpace{}, tensor), HostSpace{});
    check_memory_space(create_mirror_tensor_and_copy(DeviceMemorySpace{}, tensor), DeviceMemorySpace{});

    // create_mirror_tensor_and_copy tensor_alloc
    check_memory_space(create_mirror_tensor_and_copy(tensor_alloc(HostSpace{}), tensor), HostSpace{});
    check_memory_space(create_mirror_tensor_and_copy(tensor_alloc(DeviceMemorySpace{}), tensor), DeviceMemorySpace{});

    // create_mirror_tensor_and_copy tensor_alloc + execution space
    check_memory_space(create_mirror_tensor_and_copy(tensor_alloc(HostSpace{}, DefaultHostExecutionSpace{}), tensor),
                       HostSpace{});
    check_memory_space(create_mirror_tensor_and_copy(tensor_alloc(DeviceMemorySpace{}, DefaultExecutionSpace{}), tensor),
                       DeviceMemorySpace{});

    // clang-format on
}

void test() {
    flare::Tensor<int *, flare::DefaultExecutionSpace> device_tensor("device tensor",
                                                                 10);
    flare::Tensor<int *, flare::HostSpace> host_tensor("host tensor", 10);

    test_create_mirror_properties(device_tensor);
    test_create_mirror_properties(host_tensor);
}
