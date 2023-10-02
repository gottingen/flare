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

#ifndef FLARE_CORE_TRAITS_WORK_ITEM_PROPERTY_TRAIT_H_
#define FLARE_CORE_TRAITS_WORK_ITEM_PROPERTY_TRAIT_H_

#include <flare/core/defines.h>
#include <flare/core/common/concepts.h>  // WorkItemProperty
#include <flare/core/traits/policy_trait_adaptor.h>
#include <flare/core/traits/traits_fwd.h>

namespace flare::detail {


    struct WorkItemPropertyTrait : TraitSpecificationBase<WorkItemPropertyTrait> {
        struct base_traits {
            using work_item_property = flare::experimental::WorkItemProperty::None_t;
            FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
        };
        template<class WorkItemProp, class AnalyzeNextTrait>
        struct mixin_matching_trait : AnalyzeNextTrait {
            using base_t = AnalyzeNextTrait;
            using base_t::base_t;
            using work_item_property = WorkItemProp;
        };
        template<class T>
        using trait_matches_specification =
                flare::experimental::is_work_item_property<T>;
    };

}  // namespace flare::detail

namespace flare::experimental {

    template<class Policy, unsigned long Property>
    constexpr auto require(const Policy p,
                           WorkItemProperty::ImplWorkItemProperty<Property>) {
        static_assert(flare::is_execution_policy<Policy>::value, "");
        using new_policy_t = flare::detail::WorkItemPropertyTrait::policy_with_trait<
                Policy, WorkItemProperty::ImplWorkItemProperty<Property>>;
        return new_policy_t{p};
    }

}  // namespace flare::experimental

#endif  // FLARE_CORE_TRAITS_WORK_ITEM_PROPERTY_TRAIT_H_
