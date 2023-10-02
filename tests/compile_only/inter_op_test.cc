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
#include <flare/dyn_rank_view.h>
#include <flare/core/exp_inter_op.h>

// View
static_assert(
        std::is_same<
                flare::experimental::python_view_type_t<flare::View<double *>>,
                flare::View<double *,
                        typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space,
                        flare::experimental::DefaultViewHooks>>::value,
        "Error! Unexpected python_view_type for: View");

// DynRankView
static_assert(
        std::is_same<
                flare::experimental::python_view_type_t<flare::DynRankView<double>>,
                flare::DynRankView<
                        double, typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space>>::value,
        "Error! Unexpected python_view_type for: DynRankView");

// View + Execution Space
static_assert(
        std::is_same<
                flare::experimental::python_view_type_t<
                        flare::View<double *, flare::DefaultExecutionSpace>>,
                flare::View<double *,
                        typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space,
                        flare::experimental::DefaultViewHooks>>::value,
        "Error! Unexpected python_view_type for: View + Execution Space");

// DynRankView + Execution Space
static_assert(
        std::is_same<
                flare::experimental::python_view_type_t<
                        flare::DynRankView<double, flare::DefaultExecutionSpace>>,
                flare::DynRankView<
                        double, typename flare::DefaultExecutionSpace::array_layout,
                        typename flare::DefaultExecutionSpace::memory_space>>::value,
        "Error! Unexpected python_view_type for: DynRankView + Execution Space");

// View + Memory space
static_assert(
        std::is_same<flare::experimental::python_view_type_t<
                flare::View<int64_t *, flare::HostSpace>>,
                flare::View<int64_t *, flare::LayoutRight, flare::HostSpace,
                        flare::experimental::DefaultViewHooks>>::value,
        "Error! Unexpected python_view_type for: View + Memory space");

// DynRankView + Memory space
static_assert(
        std::is_same<flare::experimental::python_view_type_t<
                flare::DynRankView<int16_t, flare::HostSpace>>,
                flare::DynRankView<int16_t, flare::LayoutRight,
                        flare::HostSpace>>::value,
        "Error! Unexpected python_view_type for: DynRankView + Memory space");

// View + Layout + Execution space
static_assert(
        std::is_same<
                flare::experimental::python_view_type_t<flare::View<
                        int **, flare::LayoutLeft, flare::DefaultExecutionSpace>>,
                flare::View<int **, flare::LayoutLeft,
                        typename flare::DefaultExecutionSpace::memory_space,
                        flare::experimental::DefaultViewHooks>>::value,
        "Error! Unexpected python_view_type for: View + Layout + Execution space");

// DynRankView + Layout + Execution space
static_assert(
        std::is_same<flare::experimental::python_view_type_t<flare::DynRankView<
                int, flare::LayoutLeft, flare::DefaultExecutionSpace>>,
                flare::DynRankView<int, flare::LayoutLeft,
                        typename flare::DefaultExecutionSpace::
                        memory_space>>::value,
        "Error! Unexpected python_view_type for: DynRankView + Layout + Execution "
        "space");

// View + Layout + Memory Space
static_assert(
        std::is_same<flare::experimental::python_view_type_t<flare::View<
                uint32_t **, flare::LayoutLeft, flare::HostSpace>>,
                flare::View<uint32_t **, flare::LayoutLeft, flare::HostSpace,
                        flare::experimental::DefaultViewHooks>>::value,
        "Error! Unexpected python_view_type for: View + Layout + Memory Space");

// DynRankView + Layout + Memory Space
static_assert(
        std::is_same<flare::experimental::python_view_type_t<flare::DynRankView<
                uint64_t, flare::LayoutLeft, flare::HostSpace>>,
                flare::DynRankView<uint64_t, flare::LayoutLeft,
                        flare::HostSpace>>::value,
        "Error! Unexpected python_view_type for: DynRankView + Layout + Memory "
        "Space");

// View + Layout + Execution space + Memory Trait
static_assert(
        std::is_same<
                flare::experimental::python_view_type_t<flare::View<
                        float ***, flare::LayoutLeft, flare::DefaultHostExecutionSpace,
                        flare::MemoryTraits<flare::RandomAccess>>>,
                flare::View<float ***, flare::LayoutLeft,
                        typename flare::DefaultHostExecutionSpace::memory_space,
                        flare::experimental::DefaultViewHooks,
                        flare::MemoryTraits<flare::RandomAccess>>>::value,
        "Error! Unexpected python_view_type for: View + Layout + Execution space + "
        "Memory Trait");

// DynRankView + Layout + Execution space  + Memory trait
static_assert(
        std::is_same<
                flare::experimental::python_view_type_t<flare::DynRankView<
                        float, flare::LayoutLeft, flare::DefaultHostExecutionSpace,
                        flare::MemoryTraits<flare::Atomic>>>,
                flare::DynRankView<
                        float, flare::LayoutLeft,
                        typename flare::DefaultHostExecutionSpace::memory_space,
                        flare::MemoryTraits<flare::Atomic>>>::value,
        "Error! Unexpected python_view_type for: DynRankView + Layout + Execution "
        "space  + Memory trait");
