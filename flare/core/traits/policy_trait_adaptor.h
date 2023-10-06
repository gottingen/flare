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

#include <flare/core/common/utilities.h>  // type_list

#include <flare/core/traits/traits_fwd.h>

#ifndef FLARE_CORE_TRAITS_POLICY_TRAIT_ADAPTOR_H_
#define FLARE_CORE_TRAITS_POLICY_TRAIT_ADAPTOR_H_

namespace flare::detail {

// General strategy: given a TraitSpecification, go through the entries in the
// parameter pack of the policy template and find the first one that returns
// `true` for the nested `trait_matches_specification` variable template. If
// that nested variable template is not found these overloads should be safely
// ignored, and the trait can specialize PolicyTraitAdapterImpl to get the
// desired behavior.


// To handle the WorkTag case, we need more than just a predicate; we need
// something that we can default to in the unspecialized case, just like we
// do for AnalyzeExecPolicy
    template<class TraitSpec, class Trait, class Enable = void>
    struct PolicyTraitMatcher : std::false_type {
    };

    template<class TraitSpec, class Trait>
    struct PolicyTraitMatcher<
            TraitSpec, Trait,
            std::enable_if_t<
                    TraitSpec::template trait_matches_specification<Trait>::value>>
            : std::true_type {
    };


// Matching version, replace the trait
    template<class TraitSpec, template<class...> class PolicyTemplate,
            class... ProcessedTraits, class MatchingTrait,
            class... ToProcessTraits, class NewTrait>
    struct PolicyTraitAdaptorImpl<
            TraitSpec, PolicyTemplate, type_list<ProcessedTraits...>,
            type_list<MatchingTrait, ToProcessTraits...>, NewTrait,
            std::enable_if_t<PolicyTraitMatcher<TraitSpec, MatchingTrait>::value>> {
        static_assert(PolicyTraitMatcher<TraitSpec, NewTrait>::value, "");
        using type = PolicyTemplate<ProcessedTraits..., NewTrait, ToProcessTraits...>;
    };

// Non-matching version, check the next option
    template<class TraitSpec, template<class...> class PolicyTemplate,
            class... ProcessedTraits, class NonMatchingTrait,
            class... ToProcessTraits, class NewTrait>
    struct PolicyTraitAdaptorImpl<
            TraitSpec, PolicyTemplate, type_list<ProcessedTraits...>,
            type_list<NonMatchingTrait, ToProcessTraits...>, NewTrait,
            std::enable_if_t<!PolicyTraitMatcher<TraitSpec, NonMatchingTrait>::value>> {
        using type = typename PolicyTraitAdaptorImpl<
                TraitSpec, PolicyTemplate,
                type_list<ProcessedTraits..., NonMatchingTrait>,
                type_list<ToProcessTraits...>, NewTrait>::type;
    };

    // Base case: no matches found; just add the trait to the end of the list
    template<class TraitSpec, template<class...> class PolicyTemplate,
            class... ProcessedTraits, class NewTrait>
    struct PolicyTraitAdaptorImpl<TraitSpec, PolicyTemplate,
            type_list<ProcessedTraits...>, type_list<>,
            NewTrait> {
        static_assert(PolicyTraitMatcher<TraitSpec, NewTrait>::value, "");
        using type = PolicyTemplate<ProcessedTraits..., NewTrait>;
    };


    template<class TraitSpec, template<class...> class PolicyTemplate,
            class... Traits, class NewTrait>
    struct PolicyTraitAdaptor<TraitSpec, PolicyTemplate<Traits...>, NewTrait>
            : PolicyTraitAdaptorImpl<TraitSpec, PolicyTemplate, type_list<>,
                    type_list<Traits...>, NewTrait> {
    };


    template<class TraitSpec>
    struct TraitSpecificationBase {
        using trait_specification = TraitSpec;
        template<class Policy, class Trait>
        using policy_with_trait =
                typename PolicyTraitAdaptor<TraitSpec, Policy, Trait>::type;
    };


}  // end namespace flare::detail

#endif  // FLARE_CORE_TRAITS_POLICY_TRAIT_ADAPTOR_H_
