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

#ifndef FLARE_KERNEL_COMMON_HELPER_H_
#define FLARE_KERNEL_COMMON_HELPER_H_

#include <flare/kernel/common/default_types.h>
#include <type_traits>

namespace flare::detail {

        // Unify Layout of a View to PreferredLayoutType if possible
        // (either matches already, or is rank-0/rank-1 and contiguous)
        // Used to reduce number of code instantiations.
        template <class ViewType, class PreferredLayoutType>
        struct GetUnifiedLayoutPreferring {
            typedef typename std::conditional<
                    ((ViewType::rank == 1) && (!std::is_same<typename ViewType::array_layout,
                            flare::LayoutStride>::value)) ||
                    ((ViewType::rank == 0)),
                    PreferredLayoutType, typename ViewType::array_layout>::type array_layout;
        };

        template <class ViewType>
        struct GetUnifiedLayout {
            using array_layout =
                    typename GetUnifiedLayoutPreferring<ViewType,
                            default_layout>::array_layout;
        };

        template <class T, class TX, bool do_const,
                bool isView = flare::is_view<T>::value>
        struct GetUnifiedScalarViewType {
            typedef typename TX::non_const_value_type type;
        };

        template <class T, class TX>
        struct GetUnifiedScalarViewType<T, TX, false, true> {
            typedef flare::View<typename T::non_const_value_type*,
                    typename flare::detail::GetUnifiedLayoutPreferring<
                            T, typename TX::array_layout>::array_layout,
                    typename T::device_type,
                    flare::MemoryTraits<flare::Unmanaged> >
                    type;
        };

        template <class T, class TX>
        struct GetUnifiedScalarViewType<T, TX, true, true> {
            typedef flare::View<typename T::const_value_type*,
                    typename flare::detail::GetUnifiedLayoutPreferring<
                            T, typename TX::array_layout>::array_layout,
                    typename T::device_type,
                    flare::MemoryTraits<flare::Unmanaged> >
                    type;
        };
        /*
        template <class... Ts>
        struct are_integral : std::bool_constant<((std::is_integral_v<Ts> ||
                                                   std::is_enum_v<Ts>)&&...)> {};
                                                   */

        template <class... Ts>
        inline constexpr bool are_integral_v = are_integral<Ts...>::value;

}  // namespace flare::detail

#endif  // FLARE_KERNEL_COMMON_HELPER_H_
