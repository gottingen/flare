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

#ifndef FLARE_CORE_TENSOR_VIEW_UNIFORM_TYPE_H_
#define FLARE_CORE_TENSOR_VIEW_UNIFORM_TYPE_H_

#include <flare/core/defines.h>

namespace flare::detail {
    template<class ScalarType, int Rank>
    struct ViewScalarToDataType {
        using type = typename ViewScalarToDataType<ScalarType, Rank - 1>::type *;
    };

    template<class ScalarType>
    struct ViewScalarToDataType<ScalarType, 0> {
        using type = ScalarType;
    };

    template<class LayoutType, int Rank>
    struct ViewUniformLayout {
        using array_layout = LayoutType;
    };

    template<class LayoutType>
    struct ViewUniformLayout<LayoutType, 0> {
        using array_layout = flare::LayoutLeft;
    };

    template<>
    struct ViewUniformLayout<flare::LayoutRight, 1> {
        using array_layout = flare::LayoutLeft;
    };

    template<class ViewType, int Traits>
    struct ViewUniformType {
        using data_type = typename ViewType::data_type;
        using const_data_type = std::add_const_t<typename ViewType::data_type>;
        using runtime_data_type =
                typename ViewScalarToDataType<typename ViewType::value_type,
                        ViewType::rank>::type;
        using runtime_const_data_type = typename ViewScalarToDataType<
                std::add_const_t<typename ViewType::value_type>, ViewType::rank>::type;

        using array_layout =
                typename ViewUniformLayout<typename ViewType::array_layout,
                        ViewType::rank>::array_layout;

        using device_type = typename ViewType::device_type;
        using anonymous_device_type =
                typename flare::Device<typename device_type::execution_space,
                        flare::AnonymousSpace>;

        using memory_traits = typename flare::MemoryTraits<Traits>;
        using type =
                flare::View<data_type, array_layout, device_type, memory_traits>;
        using const_type =
                flare::View<const_data_type, array_layout, device_type, memory_traits>;
        using runtime_type =
                flare::View<runtime_data_type, array_layout, device_type, memory_traits>;
        using runtime_const_type = flare::View<runtime_const_data_type, array_layout,
                device_type, memory_traits>;

        using nomemspace_type = flare::View<data_type, array_layout,
                anonymous_device_type, memory_traits>;
        using const_nomemspace_type =
                flare::View<const_data_type, array_layout, anonymous_device_type,
                        memory_traits>;
        using runtime_nomemspace_type =
                flare::View<runtime_data_type, array_layout, anonymous_device_type,
                        memory_traits>;
        using runtime_const_nomemspace_type =
                flare::View<runtime_const_data_type, array_layout, anonymous_device_type,
                        memory_traits>;
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_TENSOR_VIEW_UNIFORM_TYPE_H_
