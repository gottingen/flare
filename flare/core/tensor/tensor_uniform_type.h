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

#ifndef FLARE_CORE_TENSOR_TENSOR_UNIFORM_TYPE_H_
#define FLARE_CORE_TENSOR_TENSOR_UNIFORM_TYPE_H_

#include <flare/core/defines.h>

namespace flare::detail {
    template<class ScalarType, int Rank>
    struct TensorScalarToDataType {
        using type = typename TensorScalarToDataType<ScalarType, Rank - 1>::type *;
    };

    template<class ScalarType>
    struct TensorScalarToDataType<ScalarType, 0> {
        using type = ScalarType;
    };

    template<class LayoutType, int Rank>
    struct TensorUniformLayout {
        using array_layout = LayoutType;
    };

    template<class LayoutType>
    struct TensorUniformLayout<LayoutType, 0> {
        using array_layout = flare::LayoutLeft;
    };

    template<>
    struct TensorUniformLayout<flare::LayoutRight, 1> {
        using array_layout = flare::LayoutLeft;
    };

    template<class TensorType, int Traits>
    struct TensorUniformType {
        using data_type = typename TensorType::data_type;
        using const_data_type = std::add_const_t<typename TensorType::data_type>;
        using runtime_data_type =
                typename TensorScalarToDataType<typename TensorType::value_type,
                        TensorType::rank>::type;
        using runtime_const_data_type = typename TensorScalarToDataType<
                std::add_const_t<typename TensorType::value_type>, TensorType::rank>::type;

        using array_layout =
                typename TensorUniformLayout<typename TensorType::array_layout,
                        TensorType::rank>::array_layout;

        using device_type = typename TensorType::device_type;
        using anonymous_device_type =
                typename flare::Device<typename device_type::execution_space,
                        flare::AnonymousSpace>;

        using memory_traits = typename flare::MemoryTraits<Traits>;
        using type =
                flare::Tensor<data_type, array_layout, device_type, memory_traits>;
        using const_type =
                flare::Tensor<const_data_type, array_layout, device_type, memory_traits>;
        using runtime_type =
                flare::Tensor<runtime_data_type, array_layout, device_type, memory_traits>;
        using runtime_const_type = flare::Tensor<runtime_const_data_type, array_layout,
                device_type, memory_traits>;

        using nomemspace_type = flare::Tensor<data_type, array_layout,
                anonymous_device_type, memory_traits>;
        using const_nomemspace_type =
                flare::Tensor<const_data_type, array_layout, anonymous_device_type,
                        memory_traits>;
        using runtime_nomemspace_type =
                flare::Tensor<runtime_data_type, array_layout, anonymous_device_type,
                        memory_traits>;
        using runtime_const_nomemspace_type =
                flare::Tensor<runtime_const_data_type, array_layout, anonymous_device_type,
                        memory_traits>;
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_TENSOR_TENSOR_UNIFORM_TYPE_H_
