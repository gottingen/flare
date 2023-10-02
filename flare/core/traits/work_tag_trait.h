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

#ifndef FLARE_CORE_TRAITS_WORK_TAG_TRAIT_H_
#define FLARE_CORE_TRAITS_WORK_TAG_TRAIT_H_

#include <flare/core/defines.h>
#include <flare/core/common/concepts.h>  // is_execution_space
#include <flare/core/traits/policy_trait_adaptor.h>
#include <flare/core/traits/traits_fwd.h>
#include <flare/core/common/utilities.h>  // type_list_any, type_list_remove_first

namespace flare {
namespace detail {

template <class T>
struct show_extra_work_tag_erroneously_given_to_execution_policy;
template <>
struct show_extra_work_tag_erroneously_given_to_execution_policy<void> {};

using _exec_policy_traits_without_work_tag = typename type_list_remove_first<
    WorkTagTrait, execution_policy_trait_specifications>::type;

template <class Trait>
struct _trait_matches_spec_predicate {
  template <class TraitSpec>
  struct apply {
    using type = typename PolicyTraitMatcher<TraitSpec, Trait>::type;
    static constexpr bool value = type::value;
  };
};

struct WorkTagTrait : TraitSpecificationBase<WorkTagTrait> {
  struct base_traits {
    using work_tag = void;
    FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class WorkTag, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;
    using work_tag = WorkTag;
    static constexpr auto show_work_tag_error_in_compilation_message =
        show_extra_work_tag_erroneously_given_to_execution_policy<
            typename base_t::work_tag>{};
    static_assert(
        std::is_void<typename base_t::work_tag>::value,
        "flare Error: More than one work tag given. Search compiler output "
        "for 'show_extra_work_tag' to see the type of the errant tag.");
  };
  // Since we don't have subsumption in pre-C++20, we need to have the work tag
  // "trait" handling code ensure that none of the other conditions are met.
  // * Compile time cost complexity note: at first glance it looks like this
  //   "rechecks" all of the other trait specs when used in the context of the
  //   full list of execution policy traits, but actually since we've already
  //   checked all of them to get to the end of the list, the compiler will
  //   have already generated those definitions, so there should be little extra
  //   cost to this. However, in the scenario where we use work tag in isolation
  //   (like if we were to add a `require()`-like thing that changes the work
  //   tag of an existing execution policy instance), we need to check all of
  //   the other traits to make sure that we're not replacing something else,
  //   given that the concept of a work tag is basically unconstrained and could
  //   be anything.  This should still be as efficient at compile time as the
  //   old code that just did a big long series of nested std::conditionals, but
  //   we should benchmark this assumption if it becomes a problem.
  template <class T>
  using trait_matches_specification = std::bool_constant<
      std::is_empty<T>::value &&
      !type_list_any<_trait_matches_spec_predicate<T>::template apply,
                     _exec_policy_traits_without_work_tag>::value>;
};


}  // end namespace detail
}  // end namespace flare

#endif  // FLARE_CORE_TRAITS_WORK_TAG_TRAIT_H_
