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

#ifndef FLARE_CORE_TRAITS_OCCUPANCY_CONTROL_TRAIT_H_
#define FLARE_CORE_TRAITS_OCCUPANCY_CONTROL_TRAIT_H_

#include <flare/core/common/error.h>  // FLARE_EXPECTS macro

#include <flare/core/traits/policy_trait_adaptor.h>

#include <flare/core/traits/traits_fwd.h>

namespace flare {

namespace experimental {

struct MaximizeOccupancy;

struct DesiredOccupancy {
  int m_occ = 100;
  explicit constexpr DesiredOccupancy(int occ) : m_occ(occ) {
    FLARE_EXPECTS(0 <= occ && occ <= 100);
  }
  explicit constexpr operator int() const { return m_occ; }
  constexpr int value() const { return m_occ; }
  DesiredOccupancy() = default;
  explicit DesiredOccupancy(MaximizeOccupancy const&) : DesiredOccupancy() {}
};

struct MaximizeOccupancy {
  explicit MaximizeOccupancy() = default;
};

}  // end namespace experimental

namespace detail {

template <class Policy, class AnalyzeNextTrait>
struct OccupancyControlPolicyMixin;


struct OccupancyControlTrait : TraitSpecificationBase<OccupancyControlTrait> {
  struct base_traits {
    using occupancy_control = flare::experimental::MaximizeOccupancy;
    static constexpr bool experimental_contains_desired_occupancy = false;
    // Default access occupancy_control, for when it is the (stateless) default
    static constexpr occupancy_control impl_get_occupancy_control() {
      return occupancy_control{};
    }
    FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
  };
  template <class OccControl, class AnalyzeNextTrait>
  using mixin_matching_trait =
      OccupancyControlPolicyMixin<OccControl, AnalyzeNextTrait>;
  template <class T>
  using trait_matches_specification = std::bool_constant<
      std::is_same<T, flare::experimental::DesiredOccupancy>::value ||
      std::is_same<T, flare::experimental::MaximizeOccupancy>::value>;
};


template <class AnalyzeNextTrait>
struct OccupancyControlPolicyMixin<flare::experimental::DesiredOccupancy,
                                   AnalyzeNextTrait> : AnalyzeNextTrait {
  using base_t            = AnalyzeNextTrait;
  using occupancy_control = flare::experimental::DesiredOccupancy;
  static constexpr bool experimental_contains_desired_occupancy = true;

  // Treat this as private, but make it public so that MSVC will still treat
  // this as a standard layout class and make it the right size: storage for a
  // stateful desired occupancy
  //   private:
  occupancy_control m_desired_occupancy = occupancy_control{};

  OccupancyControlPolicyMixin() = default;
  // Converting constructor
  // Just rely on the convertibility of occupancy_control to transfer the data
  template <class Other>
  OccupancyControlPolicyMixin(ExecPolicyTraitsWithDefaults<Other> const& other)
      : base_t(other),
        m_desired_occupancy(other.impl_get_occupancy_control()) {}

  // Converting assignment operator
  // Just rely on the convertibility of occupancy_control to transfer the data
  template <class Other>
  OccupancyControlPolicyMixin& operator=(
      ExecPolicyTraitsWithDefaults<Other> const& other) {
    *static_cast<base_t*>(this) = other;
    this->impl_set_desired_occupancy(
        occupancy_control{other.impl_get_occupancy_control()});
    return *this;
  }

  // Access to occupancy control instance, usable in generic context
  constexpr occupancy_control impl_get_occupancy_control() const {
    return m_desired_occupancy;
  }

  // Access to desired occupancy (getter and setter)
  flare::experimental::DesiredOccupancy impl_get_desired_occupancy() const {
    return m_desired_occupancy;
  }

  void impl_set_desired_occupancy(occupancy_control desired_occupancy) {
    m_desired_occupancy = desired_occupancy;
  }
};

template <class AnalyzeNextTrait>
struct OccupancyControlPolicyMixin<flare::experimental::MaximizeOccupancy,
                                   AnalyzeNextTrait> : AnalyzeNextTrait {
  using base_t = AnalyzeNextTrait;
  using base_t::base_t;
  using occupancy_control = flare::experimental::MaximizeOccupancy;
  static constexpr bool experimental_contains_desired_occupancy = false;
};

}  // end namespace detail

namespace experimental {

template <typename Policy>
auto prefer(Policy const& p, DesiredOccupancy occ) {
  using new_policy_t =
      flare::detail::OccupancyControlTrait::policy_with_trait<Policy,
                                                             DesiredOccupancy>;
  new_policy_t pwo{p};
  pwo.impl_set_desired_occupancy(occ);
  return pwo;
}

template <typename Policy>
constexpr auto prefer(Policy const& p, MaximizeOccupancy) {
  static_assert(flare::is_execution_policy<Policy>::value, "");
  using new_policy_t =
      flare::detail::OccupancyControlTrait::policy_with_trait<Policy,
                                                             MaximizeOccupancy>;
  return new_policy_t{p};
}

}  // end namespace experimental

}  // end namespace flare

#endif  // FLARE_CORE_TRAITS_OCCUPANCY_CONTROL_TRAIT_H_
