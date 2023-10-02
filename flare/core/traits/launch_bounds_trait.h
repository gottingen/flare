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

#ifndef FLARE_CORE_TRAITS_LAUNCH_BOUNDS_TRAIT_H_
#define FLARE_CORE_TRAITS_LAUNCH_BOUNDS_TRAIT_H_

#include <flare/core/defines.h>
#include <flare/core/common/concepts.h>  // LaunchBounds
#include <flare/core/traits/policy_trait_adaptor.h>
#include <flare/core/traits/traits_fwd.h>

namespace flare {
namespace detail {

struct LaunchBoundsTrait : TraitSpecificationBase<LaunchBoundsTrait> {
  struct base_traits {
    static constexpr bool launch_bounds_is_defaulted = true;

    using launch_bounds = LaunchBounds<>;
    FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class LaunchBoundParam, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;

    static constexpr bool launch_bounds_is_defaulted = false;

    static_assert(base_t::launch_bounds_is_defaulted,
                  "flare Error: More than one launch_bounds given");

    using launch_bounds = LaunchBoundParam;
  };
};


template <unsigned int maxT, unsigned int minB>
struct PolicyTraitMatcher<LaunchBoundsTrait, LaunchBounds<maxT, minB>>
    : std::true_type {};

}  // end namespace detail
}  // end namespace flare

#endif  // FLARE_CORE_TRAITS_LAUNCH_BOUNDS_TRAIT_H_
