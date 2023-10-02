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


#ifndef FLARE_CORE_TENSOR_DEFAULT_ACCESSOR_H_
#define FLARE_CORE_TENSOR_DEFAULT_ACCESSOR_H_

#include <flare/core/tensor/macros.h>
#include <cstddef> // size_t

namespace flare {

    template<class ElementType>
    struct default_accessor {

        using offset_policy = default_accessor;
        using element_type = ElementType;
        using reference = ElementType &;
        using data_handle_type = ElementType *;

        MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr default_accessor() noexcept = default;

        MDSPAN_TEMPLATE_REQUIRES(
                class OtherElementType,
        /* requires */ (
                        _MDSPAN_TRAIT(std::is_convertible, OtherElementType(*)[], element_type(*)[])
                )
        )
        MDSPAN_INLINE_FUNCTION
        constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}

        MDSPAN_INLINE_FUNCTION
        constexpr data_handle_type
        offset(data_handle_type p, size_t i) const noexcept {
            return p + i;
        }

        MDSPAN_FORCE_INLINE_FUNCTION
        constexpr reference access(data_handle_type p, size_t i) const noexcept {
            return p[i];
        }

    };

} // end namespace flare

#endif  // FLARE_CORE_TENSOR_DEFAULT_ACCESSOR_H_
