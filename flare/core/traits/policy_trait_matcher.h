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

#ifndef FLARE_CORE_TRAITS_POLICY_TRAIT_MATCHER_H_
#define FLARE_CORE_TRAITS_POLICY_TRAIT_MATCHER_H_

namespace flare::detail {
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

}  // end namespace flare::detail

#endif  // FLARE_CORE_TRAITS_POLICY_TRAIT_MATCHER_H_
