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

#ifndef FLARE_CORE_POLICY_ANALYZE_POLICY_H_
#define FLARE_CORE_POLICY_ANALYZE_POLICY_H_

#include <flare/core_fwd.h>
#include <flare/core/common/concepts.h>  // IndexType
#include <flare/core/traits/traits_fwd.h>
#include <flare/core/traits/policy_trait_adaptor.h>

#include <flare/core/traits/execution_space_trait.h>
#include <flare/core/traits/graph_kernel_trait.h>
#include <flare/core/traits/index_type_trait.h>
#include <flare/core/policy/iteration_pattern_trait.h>
#include <flare/core/traits/launch_bounds_trait.h>
#include <flare/core/traits/occupancy_control_trait.h>
#include <flare/core/traits/schedule_trait.h>
#include <flare/core/traits/work_item_property_trait.h>
#include <flare/core/traits/work_tag_trait.h>

namespace flare::detail {

    // Mix in the defaults (base_traits) for the traits that aren't yet handled

    template<class TraitSpecList>
    struct FLARE_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION AnalyzeExecPolicyBaseTraits;
    template<class... TraitSpecifications>
    struct FLARE_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION
    AnalyzeExecPolicyBaseTraits<type_list<TraitSpecifications...>>
            : TraitSpecifications::base_traits ... {
    };

    //------------------------------------------------------------------------------
    // Note: unspecialized, so that the default pathway is to fall back to using
    // the PolicyTraitMatcher. See AnalyzeExecPolicyUseMatcher below
    template<class Enable, class... Traits>
    struct AnalyzeExecPolicy
            : AnalyzeExecPolicyUseMatcher<void, execution_policy_trait_specifications,
                    Traits...> {
        using base_t =
                AnalyzeExecPolicyUseMatcher<void, execution_policy_trait_specifications,
                        Traits...>;
        using base_t::base_t;
    };

    //------------------------------------------------------------------------------
    // Ignore void for backwards compatibility purposes, though hopefully no one is
    // using this in application code
    template<class... Traits>
    struct AnalyzeExecPolicy<void, void, Traits...>
            : AnalyzeExecPolicy<void, Traits...> {
        using base_t = AnalyzeExecPolicy<void, Traits...>;
        using base_t::base_t;
    };

    template<>
    struct AnalyzeExecPolicy<void>
            : AnalyzeExecPolicyBaseTraits<execution_policy_trait_specifications> {
        // Ensure default constructibility since a converting constructor causes it to
        // be deleted.
        AnalyzeExecPolicy() = default;

        // Base converting constructor and assignment operator: unless an individual
        // policy analysis deletes a constructor, assume it's convertible
        template<class Other>
        AnalyzeExecPolicy(ExecPolicyTraitsWithDefaults<Other> const &) {}

        template<class Other>
        AnalyzeExecPolicy &operator=(ExecPolicyTraitsWithDefaults<Other> const &) {
            return *this;
        }
    };

    // We can avoid having to have policies specialize AnalyzeExecPolicy themselves
    // by piggy-backing off of the PolicyTraitMatcher that we need to have for
    // things like require() anyway. We mixin the effects of the trait using
    // the `mixin_matching_trait` nested alias template in the trait specification

    // General PolicyTraitMatcher version

    // Matching case
    template<class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
    struct AnalyzeExecPolicyUseMatcher<
            std::enable_if_t<PolicyTraitMatcher<TraitSpec, Trait>::value>,
            type_list<TraitSpec, TraitSpecs...>, Trait, Traits...>
            : TraitSpec::template mixin_matching_trait<
                    Trait, AnalyzeExecPolicy<void, Traits...>> {
        using base_t = typename TraitSpec::template mixin_matching_trait<
                Trait, AnalyzeExecPolicy<void, Traits...>>;
        using base_t::base_t;
    };

// Non-matching case
    template<class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
    struct AnalyzeExecPolicyUseMatcher<
            std::enable_if_t<!PolicyTraitMatcher<TraitSpec, Trait>::value>,
            type_list<TraitSpec, TraitSpecs...>, Trait, Traits...>
            : AnalyzeExecPolicyUseMatcher<void, type_list<TraitSpecs...>, Trait,
                    Traits...> {
        using base_t = AnalyzeExecPolicyUseMatcher<void, type_list<TraitSpecs...>,
                Trait, Traits...>;
        using base_t::base_t;
    };

// No match found case:
    template<class>
    struct show_name_of_invalid_execution_policy_trait;
    template<class Trait, class... Traits>
    struct AnalyzeExecPolicyUseMatcher<void, type_list<>, Trait, Traits...> {
        static constexpr auto trigger_error_message =
                show_name_of_invalid_execution_policy_trait<Trait>{};
        static_assert(
                /* always false: */ std::is_void<Trait>::value,
                                    "Unknown execution policy trait. Search compiler output for "
                                    "'show_name_of_invalid_execution_policy_trait' to see the type of the "
                                    "invalid trait.");
    };

    // All traits matched case:
    template<>
    struct AnalyzeExecPolicyUseMatcher<void, type_list<>>
            : AnalyzeExecPolicy<void> {
        using base_t = AnalyzeExecPolicy<void>;
        using base_t::base_t;
    };

    //------------------------------------------------------------------------------
    // Used for defaults that depend on other analysis results
    template<class AnalysisResults>
    struct ExecPolicyTraitsWithDefaults : AnalysisResults {
        using base_t = AnalysisResults;
        using base_t::base_t;
        // The old code turned this into an integral type for backwards compatibility,
        // so that's what we're doing here. The original comment was:
        //   nasty hack to make index_type into an integral_type
        //   instead of the wrapped IndexType<T> for backwards compatibility
        using index_type = typename std::conditional_t<
                base_t::index_type_is_defaulted,
                flare::IndexType<typename base_t::execution_space::size_type>,
                typename base_t::index_type>::type;
    };

    template<typename... Traits>
    struct PolicyTraits
            : ExecPolicyTraitsWithDefaults<AnalyzeExecPolicy<void, Traits...>> {
        using base_t =
                ExecPolicyTraitsWithDefaults<AnalyzeExecPolicy<void, Traits...>>;
        using base_t::base_t;
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_POLICY_ANALYZE_POLICY_H_
