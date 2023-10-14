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
#include <flare/dyn_rank_tensor.h>
#include <flare/core/exp_inter_op.h>

// Tensor
static_assert(
        std::is_same<
                flare::experimental::python_tensor_type_t<flare::Tensor<double *>>,
                flare::Tensor<double *,
                        typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space,
                        flare::experimental::DefaultTensorHooks>>::value,
        "Error! Unexpected python_tensor_type for: Tensor");

// DynRankTensor
static_assert(
        std::is_same<
                flare::experimental::python_tensor_type_t<flare::DynRankTensor<double>>,
                flare::DynRankTensor<
                        double, typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space>>::value,
        "Error! Unexpected python_tensor_type for: DynRankTensor");

// Tensor + Execution Space
static_assert(
        std::is_same<
                flare::experimental::python_tensor_type_t<
                        flare::Tensor<double *, flare::DefaultExecutionSpace>>,
                flare::Tensor<double *,
                        typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space,
                        flare::experimental::DefaultTensorHooks>>::value,
        "Error! Unexpected python_tensor_type for: Tensor + Execution Space");

// DynRankTensor + Execution Space
static_assert(
        std::is_same<
                flare::experimental::python_tensor_type_t<
                        flare::DynRankTensor<double, flare::DefaultExecutionSpace>>,
                flare::DynRankTensor<
                        double, typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space>>::value,
        "Error! Unexpected python_tensor_type for: DynRankTensor + Execution Space");

// Tensor + Memory space
static_assert(
        std::is_same<flare::experimental::python_tensor_type_t<
                flare::Tensor<int64_t *, flare::HostSpace>>,
                flare::Tensor<int64_t *, flare::LayoutRight, flare::HostSpace,
                        flare::experimental::DefaultTensorHooks>>::value,
        "Error! Unexpected python_tensor_type for: Tensor + Memory space");

// DynRankTensor + Memory space
static_assert(
        std::is_same<flare::experimental::python_tensor_type_t<
                flare::DynRankTensor<int16_t, flare::HostSpace>>,
                flare::DynRankTensor<int16_t, flare::LayoutRight,
                        flare::HostSpace>>::value,
        "Error! Unexpected python_tensor_type for: DynRankTensor + Memory space");

// Tensor + Layout + Execution space
static_assert(
        std::is_same<
                flare::experimental::python_tensor_type_t<flare::Tensor<
                        int **, flare::LayoutLeft, flare::DefaultExecutionSpace>>,
                flare::Tensor<int **, flare::LayoutLeft,
                        typename flare::DefaultExecutionSpace::memory_space,
                        flare::experimental::DefaultTensorHooks>>::value,
        "Error! Unexpected python_tensor_type for: Tensor + Layout + Execution space");

// DynRankTensor + Layout + Execution space
static_assert(
        std::is_same<flare::experimental::python_tensor_type_t<flare::DynRankTensor<
                int, flare::LayoutLeft, flare::DefaultExecutionSpace>>,
                flare::DynRankTensor<int, flare::LayoutLeft,
                        typename flare::DefaultExecutionSpace::
                        memory_space>>::value,
        "Error! Unexpected python_tensor_type for: DynRankTensor + Layout + Execution "
        "space");

// Tensor + Layout + Memory Space
static_assert(
        std::is_same<flare::experimental::python_tensor_type_t<flare::Tensor<
                uint32_t **, flare::LayoutLeft, flare::HostSpace>>,
                flare::Tensor<uint32_t **, flare::LayoutLeft, flare::HostSpace,
                        flare::experimental::DefaultTensorHooks>>::value,
        "Error! Unexpected python_tensor_type for: Tensor + Layout + Memory Space");

// DynRankTensor + Layout + Memory Space
static_assert(
        std::is_same<flare::experimental::python_tensor_type_t<flare::DynRankTensor<
                uint64_t, flare::LayoutLeft, flare::HostSpace>>,
                flare::DynRankTensor<uint64_t, flare::LayoutLeft,
                        flare::HostSpace>>::value,
        "Error! Unexpected python_tensor_type for: DynRankTensor + Layout + Memory "
        "Space");

// Tensor + Layout + Execution space + Memory Trait
static_assert(
        std::is_same<
                flare::experimental::python_tensor_type_t<flare::Tensor<
                        float ***, flare::LayoutLeft, flare::DefaultHostExecutionSpace,
                        flare::MemoryTraits<flare::RandomAccess>>>,
                flare::Tensor<float ***, flare::LayoutLeft,
                        typename flare::DefaultHostExecutionSpace::memory_space,
                        flare::experimental::DefaultTensorHooks,
                        flare::MemoryTraits<flare::RandomAccess>>>::value,
        "Error! Unexpected python_tensor_type for: Tensor + Layout + Execution space + "
        "Memory Trait");

// DynRankTensor + Layout + Execution space  + Memory trait
static_assert(
        std::is_same<
                flare::experimental::python_tensor_type_t<flare::DynRankTensor<
                        float, flare::LayoutLeft, flare::DefaultHostExecutionSpace,
                        flare::MemoryTraits<flare::Atomic>>>,
                flare::DynRankTensor<
                        float, flare::LayoutLeft,
                        typename flare::DefaultHostExecutionSpace::memory_space,
                        flare::MemoryTraits<flare::Atomic>>>::value,
        "Error! Unexpected python_tensor_type for: DynRankTensor + Layout + Execution "
        "space  + Memory trait");
