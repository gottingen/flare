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

#ifndef FLARE_CORE_TRAITS_EXECUTION_SPACE_TRAIT_H_
#define FLARE_CORE_TRAITS_EXECUTION_SPACE_TRAIT_H_

#include <flare/core/defines.h>
#include <flare/core/common/concepts.h>  // is_execution_space
#include <flare/core/traits/policy_trait_adaptor.h>
#include <flare/core/traits/traits_fwd.h>

namespace flare {
namespace detail {

template <class T>
struct show_extra_execution_space_erroneously_given_to_execution_policy;
template <>
struct show_extra_execution_space_erroneously_given_to_execution_policy<void> {
};
struct ExecutionSpaceTrait : TraitSpecificationBase<ExecutionSpaceTrait> {
  struct base_traits {
    static constexpr auto execution_space_is_defaulted = true;

    using execution_space = flare::DefaultExecutionSpace;
    FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class T>
  using trait_matches_specification = flare::is_execution_space<T>;
  template <class ExecSpace, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;

    static constexpr auto show_execution_space_error_in_compilation_message =
        show_extra_execution_space_erroneously_given_to_execution_policy<
            std::conditional_t<base_t::execution_space_is_defaulted, void,
                               typename base_t::execution_space>>{};
    static_assert(base_t::execution_space_is_defaulted,
                  "flare Error: More than one execution space given. Search "
                  "compiler output for 'show_extra_execution_space' to see the "
                  "type of the errant tag.");

    static constexpr auto execution_space_is_defaulted = false;

    using execution_space = ExecSpace;
  };
};

}  // end namespace detail
}  // end namespace flare

#endif  // FLARE_CORE_TRAITS_EXECUTION_SPACE_TRAIT_H_
