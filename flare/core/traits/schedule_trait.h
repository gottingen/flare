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

#ifndef FLARE_CORE_TRAITS_SCHEDULE_TRAIT_H_
#define FLARE_CORE_TRAITS_SCHEDULE_TRAIT_H_

#include <flare/core/defines.h>
#include <flare/core/common/concepts.h>  // is_schedule_type, Schedule
#include <flare/core/traits/policy_trait_adaptor.h>
#include <flare/core/traits/traits_fwd.h>

namespace flare::detail {

template <class T>
struct show_extra_schedule_type_erroneously_given_to_execution_policy;
template <>
struct show_extra_schedule_type_erroneously_given_to_execution_policy<void> {};
struct ScheduleTrait : TraitSpecificationBase<ScheduleTrait> {
  struct base_traits {
    static constexpr auto schedule_type_is_defaulted = true;

    using schedule_type = Schedule<Static>;
    FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class Sched, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;
    using base_t::base_t;
    using schedule_type = Sched;
    static constexpr auto show_schedule_type_error_in_compilation_message =
        show_extra_schedule_type_erroneously_given_to_execution_policy<
            std::conditional_t<base_t::schedule_type_is_defaulted, void,
                               typename base_t::schedule_type>>{};
    static_assert(base_t::schedule_type_is_defaulted,
                  "flare Error: More than one schedule type given. Search "
                  "compiler output for 'show_extra_schedule_type' to see the "
                  "type of the errant tag.");
    static constexpr bool schedule_type_is_defaulted = false;
  };
};

template <class Sched>
struct PolicyTraitMatcher<ScheduleTrait, Schedule<Sched>> : std::true_type {};


}  // end namespace detail

namespace flare::experimental {


template <class Policy, class ScheduleType>
constexpr auto require(Policy const& p, flare::Schedule<ScheduleType>) {
  static_assert(flare::is_execution_policy<Policy>::value, "");
  using new_policy_t = flare::detail::ScheduleTrait::policy_with_trait<
      Policy, flare::Schedule<ScheduleType>>;
  return new_policy_t{p};
}

}  // end namespace flare

#endif  // FLARE_CORE_TRAITS_SCHEDULE_TRAIT_H_
