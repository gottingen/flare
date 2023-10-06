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

#ifndef FLARE_CORE_POLICY_ITERATION_PATTERN_TRAIT_H_
#define FLARE_CORE_POLICY_ITERATION_PATTERN_TRAIT_H_

#include <flare/core/common/concepts.h>                   // is_iteration_pattern
#include <flare/core/traits/policy_trait_adaptor.h>  // TraitSpecificationBase
#include <flare/core/policy/rank.h>                       // Rank
#include <flare/core/memory/layout.h>                     // Iterate
#include <type_traits>                           // is_void

namespace flare::detail {


    template<class T>
    struct show_extra_iteration_pattern_erroneously_given_to_execution_policy;
    template<>
    struct show_extra_iteration_pattern_erroneously_given_to_execution_policy<
            void> {
    };
    struct IterationPatternTrait : TraitSpecificationBase<IterationPatternTrait> {
        struct base_traits {
            using iteration_pattern = void;  // TODO set default iteration pattern
            FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
        };
        template<class IterPattern, class AnalyzeNextTrait>
        struct mixin_matching_trait : AnalyzeNextTrait {
            using base_t = AnalyzeNextTrait;
            using base_t::base_t;
            static constexpr auto show_iteration_pattern_error_in_compilation_message =
                    show_extra_iteration_pattern_erroneously_given_to_execution_policy<
                            typename base_t::iteration_pattern>{};
            static_assert(
                    std::is_void<typename base_t::iteration_pattern>::value,
                    "flare Error: More than one index type given. Search "
                    "compiler output for 'show_extra_iteration_pattern' to see the "
                    "type of the errant tag.");
            using iteration_pattern = IterPattern;
        };
    };


    template<unsigned N, Iterate OuterDir, Iterate InnerDir>
    struct PolicyTraitMatcher<IterationPatternTrait, Rank < N, OuterDir, InnerDir>>
    : std::true_type {
};


}  // end namespace flare::detail

#endif  // FLARE_CORE_POLICY_ITERATION_PATTERN_TRAIT_H_
