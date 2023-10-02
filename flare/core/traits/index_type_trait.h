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

#ifndef FLARE_CORE_TRAITS_INDEX_TYPE_TRAIT_H_
#define FLARE_CORE_TRAITS_INDEX_TYPE_TRAIT_H_

#include <flare/core/defines.h>
#include <flare/core/common/concepts.h>  // IndexType
#include <flare/core/traits/policy_trait_adaptor.h>
#include <flare/core/traits/traits_fwd.h>

namespace flare {
namespace detail {

template <class Trait, class AnalyzeNextTrait>
struct IndexTypePolicyMixin;


template <class T>
struct show_extra_index_type_erroneously_given_to_execution_policy;
template <>
struct show_extra_index_type_erroneously_given_to_execution_policy<void> {};
struct IndexTypeTrait : TraitSpecificationBase<IndexTypeTrait> {
  struct base_traits {
    static constexpr bool index_type_is_defaulted = true;
    using index_type = dependent_policy_trait_default;
    FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class IdxType, class AnalyzeNextTrait>
  using mixin_matching_trait = IndexTypePolicyMixin<IdxType, AnalyzeNextTrait>;
};


// Index type given as IndexType template
template <class IntegralIndexType, class AnalyzeNextTrait>
struct IndexTypePolicyMixin<flare::IndexType<IntegralIndexType>,
                            AnalyzeNextTrait> : AnalyzeNextTrait {
  using base_t = AnalyzeNextTrait;
  using base_t::base_t;
  static constexpr auto show_index_type_error_in_compilation_message =
      show_extra_index_type_erroneously_given_to_execution_policy<
          std::conditional_t<base_t::index_type_is_defaulted, void,
                             typename base_t::schedule_type>>{};
  static_assert(base_t::index_type_is_defaulted,
                "flare Error: More than one index type given. Search "
                "compiler output for 'show_extra_index_type' to see the "
                "type of the errant tag.");
  static constexpr bool index_type_is_defaulted = false;
  using index_type = flare::IndexType<IntegralIndexType>;
};

// IndexType given as an integral type directly (the matcher already checks
// this, so we don't have specialize to re-check it here)
template <class IntegralIndexType, class AnalyzeNextTrait>
struct IndexTypePolicyMixin : AnalyzeNextTrait {
  using base_t = AnalyzeNextTrait;
  using base_t::base_t;
  static constexpr auto show_index_type_error_in_compilation_message =
      show_extra_index_type_erroneously_given_to_execution_policy<
          std::conditional_t<base_t::index_type_is_defaulted, void,
                             typename base_t::schedule_type>>{};
  static_assert(base_t::index_type_is_defaulted,
                "flare Error: More than one index type given. Search "
                "compiler output for 'show_extra_index_type' to see the "
                "type of the errant tag.");
  static_assert(std::is_integral<IntegralIndexType>::value, "");
  static constexpr bool index_type_is_defaulted = false;
  using index_type = flare::IndexType<IntegralIndexType>;
};


template <class IntegralIndexType>
struct PolicyTraitMatcher<IndexTypeTrait, IndexType<IntegralIndexType>>
    : std::true_type {};

template <class IntegralIndexType>
struct PolicyTraitMatcher<
    IndexTypeTrait, IntegralIndexType,
    std::enable_if_t<std::is_integral<IntegralIndexType>::value>>
    : std::true_type {};


}  // end namespace detail
}  // end namespace flare

#endif  // FLARE_CORE_TRAITS_INDEX_TYPE_TRAIT_H_
